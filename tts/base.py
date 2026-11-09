"""Interfaz común de los proveedores de texto a voz (TTS)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class TTSError(RuntimeError):
    """El proveedor de TTS no pudo sintetizar el texto."""


@dataclass(frozen=True)
class AudioChunk:
    """Fragmento de audio sintetizado listo para reproducir."""

    data: bytes
    sample_rate: int
    format: str = "wav"
    text: str = ""

    @property
    def duration_s(self) -> float:
        """Duración aproximada del chunk si es WAV PCM."""
        if self.format != "wav" or len(self.data) <= 44:
            return 0.0
        # Cabecera WAV estándar de 44 bytes, PCM 16-bit mono.
        return (len(self.data) - 44) / (2 * max(self.sample_rate, 1))


class TTSProvider(ABC):
    """Contrato mínimo que debe cumplir todo motor de TTS."""

    name: str = "base"
    sample_rate: int = 22050

    @abstractmethod
    def synthesize(self, text: str) -> AudioChunk:
        """Sintetiza un texto corto y devuelve audio WAV (o el formato nativo)."""

    def close(self) -> None:
        """Libera recursos. Los proveedores sin estado no hacen nada."""


def crear_tts(config: dict[str, Any]) -> TTSProvider:
    """Crea el proveedor de TTS indicado por ``config.tts.provider``."""
    cfg = config.get("tts") or {}
    proveedor = str(cfg.get("provider", "piper")).strip().lower()
    if proveedor == "piper":
        from tts.piper_provider import PiperProvider

        return PiperProvider(cfg)
    if proveedor == "fish":
        from tts.fish_provider import FishProvider

        return FishProvider(cfg)
    raise TTSError(f"Proveedor de TTS desconocido: {proveedor!r}")
