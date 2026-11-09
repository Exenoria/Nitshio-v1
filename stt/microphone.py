"""Captura de micrófono y segmentación con VAD.

El módulo aísla ``sounddevice`` para que el pipeline pueda probarse con un
stream falso. La salida son arrays float32 mono, listos para el proveedor STT.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any, Callable

import numpy as np

from stt.base import STTProvider, Transcript
from stt.vad import VADSegmenter, crear_segmentador


class MicrophoneError(RuntimeError):
    """No se pudo abrir o leer el micrófono."""


class Microphone:
    """Lee frames del micrófono y emite utterances cerradas por VAD."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        segmenter: VADSegmenter | None = None,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        stt_cfg = config.get("stt") or {}
        self.sample_rate = int(stt_cfg.get("sample_rate", 16000))
        self.channels = int(stt_cfg.get("channels", 1))
        self.frame_ms = int((stt_cfg.get("vad") or {}).get("frame_ms", 30))
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.segmenter = segmenter or crear_segmentador(
            config, sample_rate=self.sample_rate
        )
        self._stream_factory = stream_factory
        self._stream: Any | None = None
        self._lock = threading.Lock()

    def _crear_stream(self) -> Any:
        if self._stream_factory is not None:
            return self._stream_factory(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.frame_samples,
            )
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover
            raise MicrophoneError(
                "sounddevice no está instalado. Instala requirements.txt."
            ) from exc
        try:
            return sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.frame_samples,
            )
        except Exception as exc:
            raise MicrophoneError(f"No se pudo abrir el micrófono: {exc}") from exc

    def frames(self, stop_event: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Genera frames mono mientras el stream esté activo."""
        with self._lock:
            if self._stream is None:
                self._stream = self._crear_stream()
                try:
                    self._stream.start()
                except Exception as exc:
                    raise MicrophoneError(f"No se pudo iniciar el micrófono: {exc}") from exc
            stream = self._stream
        try:
            while not (stop_event and stop_event.is_set()):
                try:
                    datos, overflow = stream.read(self.frame_samples)
                except Exception as exc:
                    raise MicrophoneError(f"Error leyendo el micrófono: {exc}") from exc
                if overflow:
                    # Un overflow no es fatal: se descarta el frame marcado.
                    continue
                arreglo = np.asarray(datos, dtype=np.float32)
                if arreglo.ndim > 1:
                    arreglo = arreglo.mean(axis=1)
                yield np.clip(arreglo, -1.0, 1.0)
        finally:
            self.close()

    def utterances(
        self, stop_event: threading.Event | None = None
    ) -> Iterator[np.ndarray]:
        """Genera utterances completas usando el VAD configurado."""
        for frame in self.frames(stop_event=stop_event):
            for utterance in self.segmenter.process(frame):
                yield utterance

    def listen_once(
        self,
        stop_event: threading.Event | None = None,
        *,
        timeout_s: float | None = None,
    ) -> np.ndarray | None:
        """Escucha hasta obtener una utterance (o agotar ``timeout_s``)."""
        import time

        inicio = time.monotonic()
        for frame in self.frames(stop_event=stop_event):
            if timeout_s is not None and time.monotonic() - inicio > timeout_s:
                return None
            for utterance in self.segmenter.process(frame):
                return utterance
        return None

    def close(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass


def transcribir(
    provider: STTProvider, audio: np.ndarray, sample_rate: int = 16000
) -> Transcript:
    """Atajo para transcribir un array ya capturado."""
    return provider.transcribe(audio, sample_rate=sample_rate)
