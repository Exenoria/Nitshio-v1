"""Proveedor de STT basado en Moonshine Voice en español.

``moonshine-voice`` descarga y ejecuta modelos ONNX en CPU. La importación es
perezosa: el proyecto funciona en modo texto aunque el paquete no esté
instalado. El modelo se selecciona por configuración (``stt.model``), nunca
por un nombre hardcodeado en el pipeline.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from stt.base import STTError, STTProvider, Transcript, preparar_audio

_MAPA_ARCH = {
    "tiny": "TINY",
    "base": "BASE",
    "small": "SMALL",
    "medium": "MEDIUM",
    "tiny_streaming": "TINY_STREAMING",
    "base_streaming": "BASE_STREAMING",
    "small_streaming": "SMALL_STREAMING",
    "medium_streaming": "MEDIUM_STREAMING",
}


class MoonshineProvider(STTProvider):
    """Transcribe audio con Moonshine Voice (ONNX, CPU)."""

    name = "moonshine"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        if "stt" in cfg and "language" not in cfg:
            cfg = cfg.get("stt") or {}
        self.language = str(cfg.get("language", "es"))
        self.model_name = str(cfg.get("model", "small_streaming"))
        cache_dir = (cfg.get("moonshine") or {}).get("cache_dir")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._transcriber: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Carga perezosa
    # ------------------------------------------------------------------
    def _cargar(self) -> Any:
        if self._transcriber is not None:
            return self._transcriber
        with self._lock:
            if self._transcriber is not None:
                return self._transcriber
            try:
                from moonshine_voice import ModelArch, get_model_for_language
                from moonshine_voice.transcriber import Transcriber
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise STTError(
                    "moonshine-voice no está instalado. Instala requirements.txt "
                    "o cambia stt.provider a whisper."
                ) from exc

            arch_name = _MAPA_ARCH.get(self.model_name.lower())
            if arch_name is None:
                raise STTError(
                    f"Modelo Moonshine desconocido: {self.model_name!r}. "
                    f"Opciones: {', '.join(sorted(_MAPA_ARCH))}"
                )
            arch = getattr(ModelArch, arch_name, None)
            if arch is None:
                raise STTError(
                    f"La versión instalada de moonshine-voice no expone "
                    f"ModelArch.{arch_name}."
                )
            try:
                model_path, arch = get_model_for_language(
                    self.language,
                    arch,
                    cache_root=self.cache_dir,
                )
                self._transcriber = Transcriber(model_path, arch)
            except Exception as exc:
                raise STTError(
                    f"No se pudo cargar Moonshine ({self.model_name}, "
                    f"{self.language}): {exc}"
                ) from exc
            return self._transcriber

    # ------------------------------------------------------------------
    # STTProvider
    # ------------------------------------------------------------------
    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Transcript:
        muestras = preparar_audio(audio, sample_rate)
        if muestras.size == 0:
            return Transcript(
                text="", language=self.language, provider=self.name
            )
        try:
            transcriber = self._cargar()
            resultado = transcriber.transcribe_without_streaming(
                muestras.tolist(), sample_rate
            )
        except STTError:
            raise
        except Exception as exc:
            raise STTError(f"Moonshine falló al transcribir: {exc}") from exc

        lineas = [
            str(getattr(linea, "text", "")).strip()
            for linea in getattr(resultado, "lines", []) or []
        ]
        texto = " ".join(parte for parte in lineas if parte).strip()
        return Transcript(
            text=texto,
            language=self.language,
            duration_s=round(muestras.size / max(sample_rate, 1), 2),
            provider=self.name,
        )

    def close(self) -> None:
        transcriber = self._transcriber
        self._transcriber = None
        if transcriber is not None:
            try:
                transcriber.close()
            except Exception:
                pass
