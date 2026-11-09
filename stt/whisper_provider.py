"""Proveedor de STT alternativo basado en faster-whisper.

Se usa como fallback si Moonshine no está disponible. La dependencia es
opcional: solo se importa cuando ``stt.provider`` es ``whisper``.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from stt.base import STTError, STTProvider, Transcript, preparar_audio


class WhisperProvider(STTProvider):
    """Transcribe audio con faster-whisper en CPU."""

    name = "whisper"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        if "stt" in cfg and "language" not in cfg:
            cfg = cfg.get("stt") or {}
        self.language = str(cfg.get("language", "es"))
        whisper_cfg = cfg.get("whisper") or {}
        self.model_name = str(whisper_cfg.get("model", "small"))
        self.device = str(whisper_cfg.get("device", "cpu"))
        self.compute_type = str(whisper_cfg.get("compute_type", "int8"))
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _cargar(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise STTError(
                    "faster-whisper no está instalado. Instala "
                    "requirements-optional.txt o usa Moonshine."
                ) from exc
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                raise STTError(f"No se pudo cargar Whisper: {exc}") from exc
            return self._model

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Transcript:
        muestras = preparar_audio(audio, sample_rate)
        if muestras.size == 0:
            return Transcript(
                text="", language=self.language, provider=self.name
            )
        try:
            segmentos, info = self._cargar().transcribe(
                muestras,
                language=self.language,
                vad_filter=False,
            )
            texto = " ".join(segmento.text.strip() for segmento in segmentos).strip()
        except STTError:
            raise
        except Exception as exc:
            raise STTError(f"Whisper falló al transcribir: {exc}") from exc
        return Transcript(
            text=texto,
            language=getattr(info, "language", self.language) or self.language,
            confidence=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration_s=round(muestras.size / max(sample_rate, 1), 2),
            provider=self.name,
        )
