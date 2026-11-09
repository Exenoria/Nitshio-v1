"""Interfaz común de los proveedores de reconocimiento de voz (STT).

Todos los proveedores reciben audio mono en ``float32`` con valores entre
``-1.0`` y ``1.0`` y devuelven un :class:`Transcript`. Así el pipeline no
depende de la representación interna de Moonshine, Whisper o cualquier otro
motor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class STTError(RuntimeError):
    """El proveedor de STT no pudo transcribir el audio."""


@dataclass(frozen=True)
class Transcript:
    """Resultado de una transcripción."""

    text: str
    language: str = "es"
    confidence: float | None = None
    duration_s: float = 0.0
    provider: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", " ".join(str(self.text).split()))


class STTProvider(ABC):
    """Contrato mínimo que debe cumplir todo motor de STT."""

    name: str = "base"

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Transcript:
        """Transcribe audio mono float32 y devuelve el texto reconocido."""

    def close(self) -> None:
        """Libera recursos. Los proveedores sin estado no hacen nada."""


def preparar_audio(
    audio: np.ndarray | bytes, sample_rate: int
) -> np.ndarray:
    """Normaliza audio de entrada a mono float32 en el rango [-1, 1].

    Acepta ``bytes`` PCM 16-bit little-endian, arrays int16 o arrays float.
    """
    if isinstance(audio, (bytes, bytearray)):
        datos = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        return datos / 32768.0
    arreglo = np.asarray(audio)
    if arreglo.ndim > 1:
        # Mezcla a mono promediando canales.
        arreglo = arreglo.mean(axis=1)
    if arreglo.dtype == np.int16:
        return arreglo.astype(np.float32) / 32768.0
    if arreglo.dtype == np.int32:
        return arreglo.astype(np.float32) / 2147483648.0
    arreglo = arreglo.astype(np.float32, copy=False)
    if arreglo.size and float(np.max(np.abs(arreglo))) > 1.0:
        arreglo = arreglo / 32768.0
    return np.clip(arreglo, -1.0, 1.0)


def crear_stt(config: dict[str, Any]) -> STTProvider:
    """Crea el proveedor de STT indicado por ``config.stt.provider``.

    La importación es perezosa para que el proyecto pueda usarse en modo texto
    sin tener instalados Moonshine ni Whisper.
    """
    cfg = config.get("stt") or {}
    proveedor = str(cfg.get("provider", "moonshine")).strip().lower()
    if proveedor == "moonshine":
        from stt.moonshine_provider import MoonshineProvider

        return MoonshineProvider(cfg)
    if proveedor in {"whisper", "faster-whisper", "faster_whisper"}:
        from stt.whisper_provider import WhisperProvider

        return WhisperProvider(cfg)
    raise STTError(f"Proveedor de STT desconocido: {proveedor!r}")
