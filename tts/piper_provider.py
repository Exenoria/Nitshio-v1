"""Proveedor TTS local basado en Piper.

Piper sintetiza en CPU y devuelve PCM 16-bit. Este adaptador lo envuelve en
WAV para que el reproductor y VSeeFace reciban un formato estándar.
"""

from __future__ import annotations

import io
import threading
import wave
from pathlib import Path
from typing import Any, Callable

from tts.base import AudioChunk, TTSError, TTSProvider

ROOT = Path(__file__).resolve().parents[1]


class PiperProvider(TTSProvider):
    """Sintetiza voz con un modelo Piper ``.onnx`` local."""

    name = "piper"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        voice_factory: Callable[[], Any] | None = None,
    ) -> None:
        cfg = config or {}
        if "tts" in cfg and "piper" not in cfg:
            cfg = cfg.get("tts") or {}
        piper_cfg = cfg.get("piper") or {}
        self.model_path = self._ruta(piper_cfg.get("model_path", ""))
        config_path = piper_cfg.get("config_path")
        self.config_path = self._ruta(config_path) if config_path else None
        self.speed = float(piper_cfg.get("speed", 1.0) or 1.0)
        self.noise_scale = float(piper_cfg.get("noise_scale", 0.667))
        self.noise_w = float(piper_cfg.get("noise_w", 0.8))
        self._voice_factory = voice_factory
        self._voice: Any | None = None
        self._lock = threading.Lock()
        self.sample_rate = 22050

    @staticmethod
    def _ruta(valor: str | Path) -> Path:
        path = Path(valor)
        return path if path.is_absolute() else ROOT / path

    def _cargar(self) -> Any:
        if self._voice is not None:
            return self._voice
        with self._lock:
            if self._voice is not None:
                return self._voice
            if self._voice_factory is not None:
                self._voice = self._voice_factory()
                return self._voice
            if (
                not self.model_path
                or self.model_path == ROOT
                or self.model_path.suffix.lower() != ".onnx"
            ):
                raise TTSError(
                    "tts.piper.model_path no apunta a un archivo .onnx válido: "
                    f"{self.model_path}"
                )
            if not self.model_path.exists():
                raise TTSError(
                    f"No existe la voz Piper: {self.model_path}. "
                    "Ejecuta `python scripts/download_voice.py`."
                )
            try:
                from piper import PiperVoice
            except ImportError as exc:  # pragma: no cover
                raise TTSError(
                    "piper-tts no está instalado. Instala requirements.txt."
                ) from exc
            try:
                self._voice = PiperVoice.load(
                    str(self.model_path),
                    config_path=(
                        str(self.config_path)
                        if self.config_path and self.config_path.exists()
                        else None
                    ),
                )
            except Exception as exc:
                raise TTSError(f"No se pudo cargar la voz Piper: {exc}") from exc
            return self._voice

    def synthesize(self, text: str) -> AudioChunk:
        texto = " ".join(str(text or "").split())
        if not texto:
            raise TTSError("No hay texto para sintetizar.")
        voice = self._cargar()
        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                length_scale=1.0 / max(self.speed, 0.1),
                noise_scale=self.noise_scale,
                noise_w_scale=self.noise_w,
            )
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                voice.synthesize_wav(texto, wav, syn_config=syn_config)
            datos = buffer.getvalue()
        except Exception as exc:
            raise TTSError(f"Piper falló al sintetizar: {exc}") from exc
        if len(datos) <= 44:
            raise TTSError("Piper devolvió audio vacío.")
        try:
            with wave.open(io.BytesIO(datos), "rb") as wav_lectura:
                self.sample_rate = int(wav_lectura.getframerate())
        except (wave.Error, EOFError):
            pass
        return AudioChunk(
            data=datos,
            sample_rate=self.sample_rate,
            format="wav",
            text=texto,
        )

    def close(self) -> None:
        self._voice = None
