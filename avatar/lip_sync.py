"""Lip-sync opcional por amplitud para VMC.

Por defecto VSeeFace hace lip-sync leyendo ``CABLE Output``; este módulo es un
complemento para quien prefiera enviar visemas por VMC. No se activa solo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stt.base import preparar_audio


@dataclass
class LipSync:
    """Calcula un valor de apertura de boca a partir de audio."""

    threshold: float = 0.02
    gain: float = 2.5
    viseme: str = "A"

    def amplitud(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        muestras = preparar_audio(audio, sample_rate)
        if muestras.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(muestras**2)))
        return max(0.0, min(1.0, rms * self.gain))

    def visema(self, audio: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
        """Devuelve ``{viseme: apertura}`` listo para VMC."""
        valor = self.amplitud(audio, sample_rate)
        if valor < self.threshold:
            return {self.viseme: 0.0}
        return {self.viseme: valor}
