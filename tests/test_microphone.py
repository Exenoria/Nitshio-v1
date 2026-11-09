"""Tests de captura de micrófono, timeout y segmentación."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from stt.microphone import Microphone


class StreamFalso:
    """Stream de audio falso que entrega silencio o un patrón fijo."""

    def __init__(self, valor: float = 0.0, canales: int = 1) -> None:
        self.valor = valor
        self.canales = canales
        self.lecturas = 0
        self.iniciado = 0
        self.detenido = 0
        self.cerrado = 0

    def start(self) -> None:
        self.iniciado += 1

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        self.lecturas += 1
        datos = np.full((frames, self.canales), self.valor, dtype=np.float32)
        return datos, False

    def stop(self) -> None:
        self.detenido += 1

    def close(self) -> None:
        self.cerrado += 1


class SegmentadorFalso:
    """Segmentador que devuelve utterances en llamadas concretas."""

    def __init__(self, utterances: list[tuple[int, np.ndarray]] | None = None) -> None:
        self.utterances = list(utterances or [])
        self.llamadas = 0

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        self.llamadas += 1
        if self.utterances and self.llamadas >= self.utterances[0][0]:
            _, audio = self.utterances.pop(0)
            return [audio]
        return []


def _config() -> dict[str, Any]:
    return {
        "stt": {
            "sample_rate": 16000,
            "channels": 1,
            "vad": {"frame_ms": 30},
        }
    }


def _microfono(
    stream: StreamFalso, segmentador: SegmentadorFalso
) -> Microphone:
    return Microphone(
        _config(),
        segmenter=segmentador,
        stream_factory=lambda **kwargs: stream,
    )


def test_listen_once_respeta_timeout_sin_utterance() -> None:
    stream = StreamFalso()
    segmentador = SegmentadorFalso()
    micro = _microfono(stream, segmentador)

    resultado = micro.listen_once(timeout_s=0.05)

    assert resultado is None
    assert stream.lecturas >= 1
    assert stream.iniciado == 1
    assert stream.cerrado == 1


def test_listen_once_devuelve_utterance() -> None:
    audio = np.ones(160, dtype=np.float32)
    stream = StreamFalso()
    segmentador = SegmentadorFalso([(1, audio)])
    micro = _microfono(stream, segmentador)

    resultado = micro.listen_once(timeout_s=1.0)

    assert resultado is not None
    assert np.array_equal(resultado, audio)
    assert stream.cerrado == 1


def test_listen_once_respeta_stop_event() -> None:
    import threading

    stream = StreamFalso()
    segmentador = SegmentadorFalso()
    micro = _microfono(stream, segmentador)
    stop = threading.Event()
    stop.set()

    resultado = micro.listen_once(stop_event=stop, timeout_s=1.0)

    assert resultado is None
    assert stream.cerrado == 1


def test_frames_convierte_estereo_a_mono() -> None:
    stream = StreamFalso(valor=0.5, canales=2)
    segmentador = SegmentadorFalso()
    micro = _microfono(stream, segmentador)

    generador = micro.frames()
    frame = next(generador)
    generador.close()

    assert frame.ndim == 1
    assert frame.shape[0] == micro.frame_samples
    assert np.allclose(frame, 0.5)
    assert stream.cerrado == 1


def test_utterances_usa_el_segmentador() -> None:
    audio = np.ones(160, dtype=np.float32)
    stream = StreamFalso()
    segmentador = SegmentadorFalso([(1, audio)])
    micro = _microfono(stream, segmentador)

    generador = micro.utterances()
    utterance = next(generador)
    generador.close()

    assert np.array_equal(utterance, audio)
    assert segmentador.llamadas >= 1
    assert stream.cerrado == 1


def test_crear_stream_falla_sin_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def import_falso(nombre: str, *args: Any, **kwargs: Any):
        if nombre == "sounddevice":
            raise ImportError("no instalado")
        return real_import(nombre, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_falso)
    micro = Microphone(_config(), segmenter=SegmentadorFalso())

    with pytest.raises(Exception, match="sounddevice"):
        micro._crear_stream()
