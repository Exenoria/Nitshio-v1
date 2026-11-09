"""Reproducción de audio con selección de dispositivo y barge-in.

El reproductor usa ``sounddevice`` para WAV y ``ffplay`` para formatos
comprimidos (por ejemplo, MP3 de Fish Audio). El dispositivo se elige por
nombre; ``CABLE Input`` es el valor recomendado para que VSeeFace lea de
``CABLE Output``.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from tts.base import AudioChunk


class PlayerError(RuntimeError):
    """No se pudo reproducir el audio."""


class _BackendAudio(Protocol):
    def query_devices(self) -> list[Any]: ...

    def play(
        self,
        data: np.ndarray,
        sample_rate: int,
        device: int | None,
        stop_event: threading.Event,
    ) -> None: ...

    def stop(self) -> None: ...


class _BackendSoundDevice:
    """Backend real basado en sounddevice."""

    def query_devices(self) -> list[Any]:
        import sounddevice as sd

        return list(sd.query_devices())

    def play(
        self,
        data: np.ndarray,
        sample_rate: int,
        device: int | None,
        stop_event: threading.Event,
    ) -> None:
        import sounddevice as sd

        if data.size == 0:
            return
        try:
            stream = sd.OutputStream(
                samplerate=int(sample_rate),
                channels=1,
                dtype="float32",
                device=device,
            )
            with stream:
                bloque = 1024
                for inicio in range(0, data.size, bloque):
                    if stop_event.is_set():
                        break
                    stream.write(data[inicio : inicio + bloque])
        except Exception as exc:
            raise PlayerError(f"sounddevice no pudo reproducir: {exc}") from exc

    def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass


def _nombre_dispositivo(dispositivo: Any) -> str:
    if isinstance(dispositivo, dict):
        return str(dispositivo.get("name", ""))
    return str(getattr(dispositivo, "name", ""))


def _canales_salida(dispositivo: Any) -> int:
    if isinstance(dispositivo, dict):
        return int(dispositivo.get("max_output_channels", 0) or 0)
    return int(getattr(dispositivo, "max_output_channels", 0) or 0)


class AudioPlayer:
    """Reproduce chunks de audio de forma secuencial y cancelable."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        backend: _BackendAudio | None = None,
    ) -> None:
        audio_cfg = (config.get("tts") or {}).get("audio") or {}
        self.output_device = str(audio_cfg.get("output_device", "") or "")
        self.volume = float(audio_cfg.get("volume", 1.0) or 1.0)
        self.queue_max_chunks = int(audio_cfg.get("queue_max_chunks", 8))
        self.barge_in = bool(audio_cfg.get("barge_in", True))
        self._backend = backend or _BackendSoundDevice()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._device_index: int | None = None
        self._device_resuelto = False

    # ------------------------------------------------------------------
    # Dispositivos
    # ------------------------------------------------------------------
    def resolver_dispositivo(self, nombre: str | None = None) -> int | None:
        """Busca un dispositivo de salida por coincidencia parcial de nombre."""
        buscado = (nombre if nombre is not None else self.output_device).strip()
        if not buscado:
            return None
        try:
            dispositivos = self._backend.query_devices()
        except Exception:
            return None
        for indice, dispositivo in enumerate(dispositivos):
            if _canales_salida(dispositivo) <= 0:
                continue
            if buscado.lower() in _nombre_dispositivo(dispositivo).lower():
                return indice
        return None

    def dispositivo(self) -> int | None:
        """Devuelve el índice de salida, resolviéndolo una sola vez."""
        if not self._device_resuelto:
            self._device_index = self.resolver_dispositivo()
            self._device_resuelto = True
        return self._device_index

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def begin(self) -> None:
        """Prepara una nueva utterance y limpia la señal de cancelación."""
        self._stop.clear()

    def stop(self) -> None:
        """Cancela la reproducción en curso (barge-in)."""
        self._stop.set()
        with self._lock:
            self._backend.stop()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------
    # Reproducción
    # ------------------------------------------------------------------
    def play(self, chunk: AudioChunk) -> None:
        """Reproduce un chunk y bloquea hasta terminar o ser cancelado."""
        if self._stop.is_set():
            return
        if chunk.format == "wav":
            self._play_wav(chunk)
        else:
            self._play_archivo(chunk)

    def _play_wav(self, chunk: AudioChunk) -> None:
        try:
            import soundfile as sf

            datos, sample_rate = sf.read(
                io.BytesIO(chunk.data), dtype="float32", always_2d=False
            )
        except Exception as exc:
            raise PlayerError(f"No se pudo leer el WAV: {exc}") from exc
        arreglo = np.asarray(datos, dtype=np.float32)
        if arreglo.ndim > 1:
            arreglo = arreglo.mean(axis=1)
        if self.volume != 1.0:
            arreglo = np.clip(arreglo * self.volume, -1.0, 1.0)
        self._backend.play(
            arreglo, int(sample_rate), self.dispositivo(), self._stop
        )

    def _play_archivo(self, chunk: AudioChunk) -> None:
        ffplay = shutil.which("ffplay")
        if not ffplay:
            raise PlayerError(
                "ffplay no está en PATH; no se pueden reproducir formatos "
                "comprimidos. Instala FFmpeg o usa Piper (WAV)."
            )
        sufijo = f".{chunk.format}" if chunk.format else ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as tmp:
            tmp.write(chunk.data)
            ruta = Path(tmp.name)
        try:
            proceso = subprocess.Popen(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    str(ruta),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while proceso.poll() is None:
                if self._stop.is_set():
                    proceso.terminate()
                    break
                try:
                    proceso.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
            if proceso.poll() is None:
                proceso.kill()
        finally:
            try:
                ruta.unlink(missing_ok=True)
            except OSError:
                pass

    def play_chunks(self, chunks: Iterable[AudioChunk]) -> None:
        """Reproduce una secuencia de chunks respetando la cancelación."""
        for chunk in chunks:
            if self._stop.is_set():
                break
            self.play(chunk)


def reproducir_tono(
    frecuencia: float = 440.0, duracion_s: float = 0.25, sample_rate: int = 22050
) -> AudioChunk:
    """Genera un WAV de prueba (útil para diagnosticar la salida de audio)."""
    import wave

    muestras = int(sample_rate * duracion_s)
    t = np.arange(muestras, dtype=np.float32) / sample_rate
    onda = 0.2 * np.sin(2 * np.pi * frecuencia * t)
    pcm = (onda * 32767).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return AudioChunk(
        data=buffer.getvalue(), sample_rate=sample_rate, format="wav"
    )
