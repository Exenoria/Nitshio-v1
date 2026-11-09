"""Tests del VAD y de la segmentación de utterances."""

from __future__ import annotations

import numpy as np
import pytest

from stt.vad import VADError, VADSegmenter


class DetectorSecuencia:
    """Detector falso que devuelve una secuencia de decisiones."""

    def __init__(self, valores: list[bool]) -> None:
        self.valores = iter(valores)
        self.ultimo = False

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        del frame, sample_rate
        try:
            self.ultimo = next(self.valores)
        except StopIteration:
            pass
        return self.ultimo


def _frame(valor: float = 0.0) -> np.ndarray:
    return np.full(480, valor, dtype=np.float32)


def _segmentador(valores: list[bool]) -> VADSegmenter:
    return VADSegmenter(
        sample_rate=16000,
        frame_ms=30,
        min_speech_ms=90,
        end_silence_ms=90,
        pre_roll_ms=60,
        detector=DetectorSecuencia(valores),
    )


def test_emite_utterance_con_preroll_y_silencio_final() -> None:
    # 2 silencios (pre-roll), 4 voz, 3 silencios -> cierre.
    vad = _segmentador([False, False] + [True] * 4 + [False] * 3)
    salida: list[np.ndarray] = []
    for valor in [0.0, 0.0] + [0.2] * 4 + [0.0] * 3:
        salida.extend(vad.process(_frame(valor)))
    assert len(salida) == 1
    # 2 de pre-roll + 4 de voz + 1 de silencio conservado.
    assert salida[0].size == 7 * 480


def test_descarta_utterance_demasiado_corta() -> None:
    vad = _segmentador([True, False, False, False])
    salida: list[np.ndarray] = []
    salida.extend(vad.process(_frame(0.2)))
    for _ in range(3):
        salida.extend(vad.process(_frame(0.0)))
    assert salida == []


def test_flush_cierra_utterance_en_curso() -> None:
    vad = _segmentador([True, True, True])
    for _ in range(3):
        assert vad.process(_frame(0.2)) == []
    salida = vad.flush()
    assert len(salida) == 1
    assert salida[0].size == 3 * 480
    assert not vad.speaking


def test_reset_limpia_estado() -> None:
    vad = _segmentador([True, True])
    vad.process(_frame(0.2))
    vad.process(_frame(0.2))
    vad.reset()
    assert not vad.speaking
    assert vad.flush() == []


def test_frame_corto_se_rellena() -> None:
    vad = _segmentador([True])
    salida = vad.process(np.ones(100, dtype=np.float32) * 0.2)
    assert salida == []
    assert vad.speaking


def test_frame_largo_se_parte() -> None:
    vad = _segmentador([True, True, True])
    salida = vad.process(np.ones(480 * 3, dtype=np.float32) * 0.2)
    assert salida == []
    assert vad.speaking


def test_configuracion_invalida() -> None:
    with pytest.raises(VADError):
        VADSegmenter(sample_rate=44100)
    with pytest.raises(VADError):
        VADSegmenter(frame_ms=25)


def test_detector_que_falla_lanza_vad_error() -> None:
    class DetectorRoto:
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            raise RuntimeError("boom")

    vad = VADSegmenter(detector=DetectorRoto())
    with pytest.raises(VADError):
        vad.process(_frame(0.1))


def test_detector_energia_fallback() -> None:
    from stt.vad import _VadEnergia

    detector = _VadEnergia(umbral=0.01)
    silencio = np.zeros(480, dtype=np.int16).tobytes()
    voz = (np.ones(480, dtype=np.int16) * 2000).tobytes()
    assert detector.is_speech(silencio, 16000) is False
    assert detector.is_speech(voz, 16000) is True
