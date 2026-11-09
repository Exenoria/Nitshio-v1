"""Tests del pipeline de voz, el listener de micrófono y el modo llamada."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from brain.llm.orchestrator import TurnResult
from brain.llm.parser import NitshioResponse, ParsedResponse
from pipeline.voice import (
    CallMode,
    MicrophoneListener,
    VoiceEvent,
    VoicePipeline,
)
from stt.base import Transcript


def _turn(texto: str = "respuesta") -> TurnResult:
    parsed = ParsedResponse(
        data=NitshioResponse(response=texto, emotion="neutral"), raw="{}"
    )
    return TurnResult(
        user_text="hola",
        response=texto,
        emotion="neutral",
        parsed=parsed,
    )


class STTFalso:
    name = "falso"

    def __init__(self, texto: str = "hola") -> None:
        self.texto = texto
        self.llamadas: list[tuple[np.ndarray, int]] = []

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Transcript:
        self.llamadas.append((audio, sample_rate))
        return Transcript(
            text=self.texto, language="es", provider="falso", duration_s=0.2
        )

    def close(self) -> None:
        pass


class MicrofonoFalso:
    sample_rate = 16000

    def __init__(self, audios: list[np.ndarray] | None = None) -> None:
        self.audios = list(audios or [])
        self.cerrado = 0
        self.escuchas = 0

    def listen_once(self, timeout_s: float | None = None) -> np.ndarray | None:
        self.escuchas += 1
        return self.audios.pop(0) if self.audios else None

    def utterances(self, stop_event: threading.Event | None = None):
        for audio in self.audios:
            if stop_event is not None and stop_event.is_set():
                return
            yield audio

    def close(self) -> None:
        self.cerrado += 1


class OrquestadorFalso:
    def __init__(self, texto: str = "respuesta") -> None:
        self.texto = texto
        self.llamadas: list[tuple[str, bool]] = []

    def process(self, texto: str, *, speak: bool = False) -> TurnResult:
        self.llamadas.append((texto, speak))
        return _turn(self.texto)


def _audio(valor: float = 0.1) -> np.ndarray:
    return np.full(1600, valor, dtype=np.float32)


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------
def test_listen_transcript() -> None:
    stt = STTFalso("hola mundo")
    micro = MicrofonoFalso([_audio()])
    pipeline = VoicePipeline(
        {},
        OrquestadorFalso(),
        stt=stt,
        microphone=micro,
    )
    transcript = pipeline.listen_transcript()
    assert transcript is not None
    assert transcript.text == "hola mundo"
    assert stt.llamadas[0][1] == 16000


def test_listen_transcript_sin_audio() -> None:
    pipeline = VoicePipeline(
        {}, OrquestadorFalso(), stt=STTFalso(), microphone=MicrofonoFalso()
    )
    assert pipeline.listen_transcript() is None


def test_run_once() -> None:
    orquestador = OrquestadorFalso("Te escuché.")
    pipeline = VoicePipeline(
        {},
        orquestador,
        stt=STTFalso("hola"),
        microphone=MicrofonoFalso([_audio()]),
    )
    evento = pipeline.run_once(speak=True)
    assert isinstance(evento, VoiceEvent)
    assert evento.transcript.text == "hola"
    assert evento.turn.response == "Te escuché."
    assert orquestador.llamadas == [("hola", True)]


def test_run_once_texto_vacio() -> None:
    orquestador = OrquestadorFalso()
    pipeline = VoicePipeline(
        {},
        orquestador,
        stt=STTFalso("   "),
        microphone=MicrofonoFalso([_audio()]),
    )
    assert pipeline.run_once() is None
    assert orquestador.llamadas == []


def test_run_forever_respeta_stop() -> None:
    stop = threading.Event()
    stop.set()
    pipeline = VoicePipeline(
        {}, OrquestadorFalso(), stt=STTFalso(), microphone=MicrofonoFalso()
    )
    pipeline.run_forever(stop_event=stop)


def test_close_cierra_microfono() -> None:
    micro = MicrofonoFalso()
    pipeline = VoicePipeline(
        {}, OrquestadorFalso(), stt=STTFalso(), microphone=micro
    )
    pipeline.close()
    assert micro.cerrado == 1


# ---------------------------------------------------------------------------
# MicrophoneListener
# ---------------------------------------------------------------------------
def test_listener_encola_utterances() -> None:
    micro = MicrofonoFalso([_audio(0.1), _audio(0.2)])
    listener = MicrophoneListener(micro, STTFalso())
    listener.start()
    primera = listener.get(timeout=1.0)
    segunda = listener.get(timeout=1.0)
    listener.stop()
    assert primera is not None
    assert segunda is not None
    assert primera[0] == pytest.approx(0.1)
    assert segunda[0] == pytest.approx(0.2)
    assert micro.cerrado == 1


def test_listener_transcribe() -> None:
    stt = STTFalso("texto")
    micro = MicrofonoFalso()
    listener = MicrophoneListener(micro, stt)
    transcript = listener.transcribe(_audio())
    assert transcript.text == "texto"
    assert stt.llamadas[0][1] == 16000


# ---------------------------------------------------------------------------
# CallMode
# ---------------------------------------------------------------------------
class TTSLento:
    def __init__(self) -> None:
        self.textos: list[str] = []
        self.detenido = 0
        self._evento = threading.Event()

    def speak(self, texto: str) -> None:
        self.textos.append(texto)
        self._evento.wait(timeout=2)

    def stop(self) -> None:
        self.detenido += 1
        self._evento.set()


class TTSRapido:
    def __init__(self) -> None:
        self.textos: list[str] = []
        self.detenido = 0

    def speak(self, texto: str) -> None:
        self.textos.append(texto)

    def stop(self) -> None:
        self.detenido += 1


class ListenerInterrumpible:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio
        self.consultas = 0

    def get(self, timeout: float = 0.2) -> np.ndarray | None:
        self.consultas += 1
        return self.audio if self.consultas == 1 else None


def test_hablar_con_barge_in_interrumpe() -> None:
    tts = TTSLento()
    pipeline = VoicePipeline(
        {}, OrquestadorFalso(), stt=STTFalso(), microphone=MicrofonoFalso()
    )
    pipeline.tts = tts
    call = CallMode(pipeline, barge_in=True)
    interrupcion = call._hablar_con_barge_in("hola", ListenerInterrumpible(_audio()))
    assert interrupcion is not None
    assert tts.detenido == 1
    assert tts.textos == ["hola"]


def test_call_mode_run_un_turno(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _audio()

    class ListenerFalso:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.queue = [audio]
            self.error = None
            self.cerrado = 0

        def start(self) -> None:
            pass

        def get(self, timeout: float = 0.2) -> np.ndarray | None:
            return self.queue.pop(0) if self.queue else None

        def transcribe(self, dato: np.ndarray) -> Transcript:
            return Transcript(text="hola", provider="falso")

        def stop(self) -> None:
            self.cerrado += 1

    monkeypatch.setattr("pipeline.voice.MicrophoneListener", ListenerFalso)
    tts = TTSRapido()
    orquestador = OrquestadorFalso("respuesta hablada")
    pipeline = VoicePipeline(
        {},
        orquestador,
        stt=STTFalso(),
        tts=tts,
        microphone=MicrofonoFalso(),
    )
    eventos: list[VoiceEvent] = []
    call = CallMode(pipeline, on_event=eventos.append, barge_in=True)
    call.run(max_turns=1)
    assert len(eventos) == 1
    assert eventos[0].turn.response == "respuesta hablada"
    assert orquestador.llamadas == [("hola", False)]
    assert tts.textos == ["respuesta hablada"]


def test_call_mode_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = TTSRapido()
    pipeline = VoicePipeline(
        {},
        OrquestadorFalso(),
        stt=STTFalso(),
        tts=tts,
        microphone=MicrofonoFalso(),
    )
    call = CallMode(pipeline)
    call.stop()
    assert tts.detenido == 1
