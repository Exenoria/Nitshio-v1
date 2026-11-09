"""Detección de actividad de voz (VAD) y segmentación de utterances.

El segmentador recibe frames de audio de duración fija y devuelve fragmentos
completos cuando detecta el final de una frase. Usa ``webrtcvad`` si está
instalado y cae a un detector por energía si no lo está, para que el proyecto
siga funcionando en entornos mínimos.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Protocol

import numpy as np

from stt.base import preparar_audio


class VADError(RuntimeError):
    """No se pudo inicializar o usar el detector de voz."""


class DetectorVoz(Protocol):
    """Contrato mínimo de un detector de voz por frame."""

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:  # pragma: no cover
        ...


class _VadEnergia:
    """Fallback simple: considera voz si la energía RMS supera un umbral."""

    def __init__(self, umbral: float = 0.012) -> None:
        self.umbral = float(umbral)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        del sample_rate
        if not frame:
            return False
        muestras = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if not muestras.size:
            return False
        rms = float(np.sqrt(np.mean((muestras / 32768.0) ** 2)))
        return rms >= self.umbral


def crear_detector_voz(aggressiveness: int = 2) -> DetectorVoz:
    """Crea el detector WebRTC VAD; si no está, usa el fallback por energía."""
    try:
        import webrtcvad  # type: ignore[import-not-found]

        return webrtcvad.Vad(int(aggressiveness))
    except (ImportError, OSError, ValueError):
        return _VadEnergia()


class VADSegmenter:
    """Convierte un flujo de frames en utterances completas.

    Parámetros en milisegundos, igual que ``config.stt.vad``. El audio de
    entrada puede ser float32 [-1, 1] o bytes PCM 16-bit. La salida siempre es
    un array float32 mono.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        min_speech_ms: int = 250,
        end_silence_ms: int = 700,
        pre_roll_ms: int = 300,
        detector: DetectorVoz | None = None,
    ) -> None:
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise VADError(f"Sample rate no soportado por WebRTC VAD: {sample_rate}")
        if frame_ms not in {10, 20, 30}:
            raise VADError("frame_ms debe ser 10, 20 o 30")
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.min_speech_frames = max(1, math.ceil(min_speech_ms / frame_ms))
        self.end_silence_frames = max(1, round(end_silence_ms / frame_ms))
        self.pre_roll_frames = max(0, math.ceil(pre_roll_ms / frame_ms))
        self.detector = detector or crear_detector_voz(aggressiveness)

        self._pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._speech: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._speaking = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    @property
    def speaking(self) -> bool:
        """Indica si hay una utterance en curso."""
        return self._speaking

    def process(self, frame: np.ndarray | bytes) -> list[np.ndarray]:
        """Procesa un frame y devuelve las utterances que se completaron.

        Si el frame no mide exactamente ``frame_samples``, se ajusta: se
        rellena con silencio o se parte en varios frames.
        """
        audio = preparar_audio(frame, self.sample_rate)
        completadas: list[np.ndarray] = []
        for trozo in self._dividir(audio):
            completada = self._procesar_frame(trozo)
            if completada is not None:
                completadas.append(completada)
        return completadas

    def flush(self) -> list[np.ndarray]:
        """Cierra la utterance en curso (por ejemplo, al detener el micrófono)."""
        completadas: list[np.ndarray] = []
        if self._speaking:
            completada = self._finalizar()
            if completada is not None:
                completadas.append(completada)
        self.reset()
        return completadas

    def reset(self) -> None:
        """Descarta el estado y los frames acumulados."""
        self._pre_roll.clear()
        self._speech.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._speaking = False

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------
    def _dividir(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []
        frames: list[np.ndarray] = []
        for inicio in range(0, audio.size, self.frame_samples):
            trozo = audio[inicio : inicio + self.frame_samples]
            if trozo.size < self.frame_samples:
                trozo = np.pad(trozo, (0, self.frame_samples - trozo.size))
            frames.append(trozo.astype(np.float32, copy=False))
        return frames

    def _es_voz(self, frame: np.ndarray) -> bool:
        pcm = np.clip(frame, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
        try:
            return bool(self.detector.is_speech(pcm16, self.sample_rate))
        except Exception as exc:  # pragma: no cover - defensa ante binarios raros
            raise VADError(f"Fallo del detector de voz: {exc}") from exc

    def _procesar_frame(self, frame: np.ndarray) -> np.ndarray | None:
        voz = self._es_voz(frame)

        if voz:
            if not self._speaking:
                self._speaking = True
                self._speech = list(self._pre_roll)
                self._pre_roll.clear()
                self._speech_frames = 0
            self._speech.append(frame)
            self._speech_frames += 1
            self._silence_frames = 0
            return None

        if self._speaking:
            # Se conserva el silencio final para que la frase no suene cortada.
            self._speech.append(frame)
            self._silence_frames += 1
            if self._silence_frames >= self.end_silence_frames:
                return self._finalizar()
            return None

        if self.pre_roll_frames:
            self._pre_roll.append(frame)
        return None

    def _finalizar(self) -> np.ndarray | None:
        valida = self._speech_frames >= self.min_speech_frames
        if not valida or not self._speech:
            self.reset()
            return None
        # Quita el silencio final acumulado, pero deja un frame de margen.
        if self._silence_frames:
            conservar = max(0, len(self._speech) - self._silence_frames + 1)
            audio = np.concatenate(self._speech[:conservar])
        else:
            audio = np.concatenate(self._speech)
        self.reset()
        return audio.astype(np.float32, copy=False)


def crear_segmentador(config: dict[str, Any], sample_rate: int | None = None) -> VADSegmenter:
    """Crea un :class:`VADSegmenter` desde ``config.stt``."""
    stt_cfg = config.get("stt") or {}
    vad_cfg = stt_cfg.get("vad") or {}
    return VADSegmenter(
        sample_rate=int(sample_rate or stt_cfg.get("sample_rate", 16000)),
        frame_ms=int(vad_cfg.get("frame_ms", 30)),
        aggressiveness=int(vad_cfg.get("aggressiveness", 2)),
        min_speech_ms=int(vad_cfg.get("min_speech_ms", 250)),
        end_silence_ms=int(vad_cfg.get("end_silence_ms", 700)),
        pre_roll_ms=int(vad_cfg.get("pre_roll_ms", 300)),
    )
