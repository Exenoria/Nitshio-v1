"""Pipeline de voz: micrófono -> VAD -> STT -> orquestador -> TTS.

La captura corre en un hilo productor para que el modo llamada pueda detectar
barge-in mientras Nitshio habla. El TTS se reproduce en otro hilo y se cancela
cuando llega una utterance nueva.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from brain.llm.orchestrator import Orchestrator, TurnResult
from stt.base import STTError, STTProvider, Transcript, crear_stt
from stt.microphone import Microphone, MicrophoneError
from tts.manager import TTSManager


@dataclass
class VoiceEvent:
    """Evento de un turno de voz para logs o interfaz."""

    transcript: Transcript
    turn: TurnResult


class MicrophoneListener:
    """Hilo que captura utterances y las deja en una cola."""

    def __init__(
        self,
        microphone: Microphone,
        stt: STTProvider,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.microphone = microphone
        self.stt = stt
        self.stop_event = stop_event or threading.Event()
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="nitshio-mic", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            for utterance in self.microphone.utterances(
                stop_event=self.stop_event
            ):
                try:
                    self.queue.put(utterance, timeout=0.5)
                except queue.Full:
                    # Si el consumidor está muy lento, se descarta lo más viejo.
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.queue.put_nowait(utterance)
        except Exception as exc:
            self.error = exc

    def get(self, timeout: float = 0.2) -> np.ndarray | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def transcribe(self, audio: np.ndarray) -> Transcript:
        return self.stt.transcribe(audio, self.microphone.sample_rate)

    def stop(self) -> None:
        self.stop_event.set()
        self.microphone.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None


class VoicePipeline:
    """Conversación por voz de un turno o continua."""

    def __init__(
        self,
        config: dict[str, Any],
        orchestrator: Orchestrator,
        *,
        stt: STTProvider | None = None,
        tts: TTSManager | None = None,
        microphone: Microphone | None = None,
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.stt = stt or crear_stt(config)
        self.tts = tts
        self.microphone = microphone or Microphone(config)
        self._listener: MicrophoneListener | None = None

    # ------------------------------------------------------------------
    # Un turno
    # ------------------------------------------------------------------
    def listen_transcript(
        self, *, timeout_s: float | None = None
    ) -> Transcript | None:
        audio = self.microphone.listen_once(timeout_s=timeout_s)
        if audio is None or audio.size == 0:
            return None
        return self.stt.transcribe(audio, self.microphone.sample_rate)

    def run_once(self, *, speak: bool = True) -> VoiceEvent | None:
        transcript = self.listen_transcript()
        if transcript is None or not transcript.text.strip():
            return None
        turn = self.orchestrator.process(transcript.text, speak=speak)
        return VoiceEvent(transcript=transcript, turn=turn)

    def run_forever(
        self,
        *,
        on_event: Callable[[VoiceEvent], None] | None = None,
        stop_event: threading.Event | None = None,
        speak: bool = True,
    ) -> None:
        stop = stop_event or threading.Event()
        while not stop.is_set():
            try:
                evento = self.run_once(speak=speak)
            except (STTError, MicrophoneError) as exc:
                if on_event is None:
                    raise
                # Se informa y se sigue; un fallo de audio no debe tumbar el CLI.
                print(f"[voz] {exc}")
                continue
            if evento is not None and on_event is not None:
                on_event(evento)

    def close(self) -> None:
        self.microphone.close()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class CallMode:
    """Modo llamada con barge-in por chunks y captura continua.

    El micrófono sigue escuchando mientras el TTS habla. Si llega una
    utterance nueva, se cancela la locución y se procesa el nuevo mensaje.
    """

    def __init__(
        self,
        pipeline: VoicePipeline,
        *,
        on_event: Callable[[VoiceEvent], None] | None = None,
        barge_in: bool | None = None,
        logger: Any = None,
    ) -> None:
        self.pipeline = pipeline
        self.on_event = on_event
        self.barge_in = (
            bool(barge_in)
            if barge_in is not None
            else bool(
                ((pipeline.config.get("pipeline") or {}).get("call_mode") or {}).get(
                    "barge_in", True
                )
            )
        )
        self.logger = logger
        self._listener: MicrophoneListener | None = None
        self._stop = threading.Event()

    def run(self, *, max_turns: int | None = None) -> None:
        listener = MicrophoneListener(
            self.pipeline.microphone, self.pipeline.stt, stop_event=self._stop
        )
        self._listener = listener
        listener.start()
        pendiente: np.ndarray | None = None
        turnos = 0
        try:
            while not self._stop.is_set():
                audio = pendiente if pendiente is not None else listener.get(0.2)
                pendiente = None
                if audio is None:
                    if listener.error is not None:
                        raise listener.error
                    continue
                try:
                    transcript = listener.transcribe(audio)
                except STTError as exc:
                    if self.logger:
                        self.logger.warning("STT falló: %s", exc)
                    continue
                if not transcript.text.strip():
                    continue
                turno = self.pipeline.orchestrator.process(
                    transcript.text, speak=False
                )
                evento = VoiceEvent(transcript=transcript, turn=turno)
                if self.on_event is not None:
                    self.on_event(evento)
                turnos += 1
                if self.pipeline.tts is not None:
                    pendiente = self._hablar_con_barge_in(turno.response, listener)
                if max_turns is not None and turnos >= max_turns:
                    break
        finally:
            listener.stop()
            self._listener = None

    def _hablar_con_barge_in(
        self, texto: str, listener: MicrophoneListener
    ) -> np.ndarray | None:
        """Habla y devuelve la utterance que interrumpió, si la hubo."""
        hilo = threading.Thread(
            target=self.pipeline.tts.speak,
            args=(texto,),
            name="nitshio-tts",
            daemon=True,
        )
        hilo.start()
        interrupcion: np.ndarray | None = None
        while hilo.is_alive():
            nueva = listener.get(0.05)
            if nueva is not None:
                interrupcion = nueva
                if self.barge_in:
                    self.pipeline.tts.stop()
                break
            hilo.join(0.05)
        if hilo.is_alive():
            hilo.join(timeout=2)
        return interrupcion

    def stop(self) -> None:
        self._stop.set()
        if self.pipeline.tts is not None:
            self.pipeline.tts.stop()
        if self._listener is not None:
            self._listener.stop()
