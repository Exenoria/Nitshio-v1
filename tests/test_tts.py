"""Tests de Piper, Fish, reproductor y manager de TTS."""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np
import pytest

from tts.base import AudioChunk, TTSError
from tts.fish_provider import FishProvider
from tts.manager import TTSManager
from tts.piper_provider import PiperProvider
from tts.player import AudioPlayer, PlayerError, reproducir_tono


class VozFalsa:
    def synthesize_wav(self, text: str, wav: wave.Wave_write, syn_config=None) -> None:
        del text, syn_config
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x00" * 2205)


def test_piper_sintetiza_wav() -> None:
    provider = PiperProvider(
        {"piper": {"model_path": "x.onnx"}},
        voice_factory=lambda: VozFalsa(),
    )
    chunk = provider.synthesize("Hola, Nitshio.")
    assert chunk.format == "wav"
    assert chunk.sample_rate == 22050
    assert len(chunk.data) > 44
    assert chunk.duration_s == pytest.approx(0.1, abs=0.01)
    with wave.open(io.BytesIO(chunk.data), "rb") as wav:
        assert wav.getframerate() == 22050


def test_piper_texto_vacio() -> None:
    provider = PiperProvider({"piper": {"model_path": "x.onnx"}})
    with pytest.raises(TTSError):
        provider.synthesize("   ")


def test_piper_ruta_invalida() -> None:
    provider = PiperProvider({"piper": {"model_path": ""}})
    with pytest.raises(TTSError, match="model_path"):
        provider.synthesize("hola")


def test_fish_requiere_clave(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    provider = FishProvider({"fish": {}})
    with pytest.raises(TTSError, match="FISH_API_KEY"):
        provider.synthesize("hola")


class RespuestaFalsa:
    def __init__(self, content: bytes = b"ID3fake") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class SessionFalsa:
    def __init__(self) -> None:
        self.llamadas: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> RespuestaFalsa:
        self.llamadas.append({"url": url, **kwargs})
        return RespuestaFalsa()


def test_fish_sintetiza(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_API_KEY", "secreto")
    session = SessionFalsa()
    provider = FishProvider(
        {
            "fish": {
                "voice_id": "voz-1",
                "format": "mp3",
                "model": "s1",
            }
        },
        session=session,
    )
    chunk = provider.synthesize("Hola")
    assert chunk.format == "mp3"
    assert chunk.data == b"ID3fake"
    assert session.llamadas[0]["headers"]["Authorization"] == "Bearer secreto"
    assert session.llamadas[0]["json"]["reference_id"] == "voz-1"


class BackendFalso:
    def __init__(self) -> None:
        self.dispositivos = [
            {"name": "Altavoces", "max_output_channels": 2},
            {"name": "CABLE Input (VB-Audio)", "max_output_channels": 2},
        ]
        self.reproducidos: list[tuple[int, int, int | None]] = []
        self.detuvieron = 0

    def query_devices(self) -> list[dict[str, Any]]:
        return self.dispositivos

    def play(self, data, sample_rate, device, stop_event) -> None:
        self.reproducidos.append((len(data), sample_rate, device))

    def stop(self) -> None:
        self.detuvieron += 1


def _config_audio() -> dict[str, Any]:
    return {
        "tts": {
            "audio": {
                "output_device": "CABLE Input",
                "volume": 1.0,
                "queue_max_chunks": 8,
                "barge_in": True,
            }
        }
    }


def test_player_resuelve_dispositivo_por_nombre() -> None:
    backend = BackendFalso()
    player = AudioPlayer(_config_audio(), backend=backend)
    assert player.resolver_dispositivo() == 1
    assert player.resolver_dispositivo("altavoces") == 0
    assert player.resolver_dispositivo("inexistente") is None


def test_player_reproduce_wav() -> None:
    backend = BackendFalso()
    player = AudioPlayer(_config_audio(), backend=backend)
    chunk = reproducir_tono(duracion_s=0.05)
    player.play(chunk)
    assert len(backend.reproducidos) == 1
    assert backend.reproducidos[0][1] == 22050
    assert backend.reproducidos[0][2] == 1


def test_player_stop_cancela() -> None:
    backend = BackendFalso()
    player = AudioPlayer(_config_audio(), backend=backend)
    player.stop()
    assert player.stopped
    player.play(reproducir_tono(duracion_s=0.05))
    assert backend.reproducidos == []
    assert backend.detuvieron == 1


def test_player_wav_invalido() -> None:
    backend = BackendFalso()
    player = AudioPlayer(_config_audio(), backend=backend)
    with pytest.raises(PlayerError):
        player.play(AudioChunk(data=b"no es wav", sample_rate=22050))


class ProviderFalso:
    def __init__(self) -> None:
        self.textos: list[str] = []

    def synthesize(self, text: str) -> AudioChunk:
        self.textos.append(text)
        return AudioChunk(data=b"wav", sample_rate=22050, text=text)

    def close(self) -> None:
        pass


class PlayerFalso:
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []
        self.iniciado = 0
        self.detenido = 0
        self._stopped = False

    def begin(self) -> None:
        self.iniciado += 1
        self._stopped = False

    def play(self, chunk: AudioChunk) -> None:
        if not self._stopped:
            self.chunks.append(chunk)

    def stop(self) -> None:
        self.detenido += 1
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped


def test_manager_dividde_y_reproduce() -> None:
    provider = ProviderFalso()
    player = PlayerFalso()
    config = {
        "tts": {
            "chunking": {
                "max_chars": 30,
                "first_chunk_max_chars": 20,
                "min_chars": 5,
            }
        }
    }
    manager = TTSManager(config, provider=provider, player=player)
    chunks = manager.speak("Hola. ¿Cómo estás? Bien, gracias.")
    assert chunks
    assert len(provider.textos) == len(chunks)
    assert len(player.chunks) == len(chunks)
    assert player.iniciado == 1
    assert manager.last_chunks == provider.textos


def test_manager_barge_in_detiene_chunks() -> None:
    class ProviderQueDetiene(ProviderFalso):
        def __init__(self, player: PlayerFalso) -> None:
            super().__init__()
            self.player = player

        def synthesize(self, text: str) -> AudioChunk:
            chunk = super().synthesize(text)
            self.player.stop()
            return chunk

    player = PlayerFalso()
    provider = ProviderQueDetiene(player)
    manager = TTSManager(
        {"tts": {"chunking": {}}}, provider=provider, player=player
    )
    manager.speak("Uno. Dos. Tres. Cuatro. Cinco.")
    assert len(provider.textos) == 1
