"""Orquestación de TTS por chunks con soporte de barge-in."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from tts.base import AudioChunk, TTSProvider, crear_tts
from tts.chunker import dividir_en_chunks
from tts.player import AudioPlayer


class TTSManager:
    """Divide, sintetiza y reproduce la respuesta de Nitshio."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        provider: TTSProvider | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        self.config = config
        self.chunking = (config.get("tts") or {}).get("chunking") or {}
        self.provider = provider or crear_tts(config)
        self.player = player or AudioPlayer(config)
        self.last_chunks: list[str] = []

    def dividir(self, texto: str) -> list[str]:
        """Aplica la configuración de chunking al texto."""
        return dividir_en_chunks(
            texto,
            max_chars=int(self.chunking.get("max_chars", 180)),
            first_chunk_max_chars=int(
                self.chunking.get("first_chunk_max_chars", 90)
            ),
            min_chars=int(self.chunking.get("min_chars", 12)),
        )

    def speak(
        self,
        texto: str,
        *,
        on_chunk: Callable[[str, AudioChunk], None] | None = None,
    ) -> list[AudioChunk]:
        """Sintetiza y reproduce todos los chunks de ``texto``.

        ``on_chunk`` se llama tras sintetizar cada fragmento; sirve para
        enviar visemas o registrar latencias. Si el reproductor se detiene
        (barge-in), se dejan de sintetizar los chunks restantes.
        """
        chunks_texto = self.dividir(texto)
        self.last_chunks = chunks_texto
        self.player.begin()
        reproducidos: list[AudioChunk] = []
        for fragmento in chunks_texto:
            if self.player.stopped:
                break
            audio = self.provider.synthesize(fragmento)
            reproducidos.append(audio)
            if on_chunk is not None:
                on_chunk(fragmento, audio)
            self.player.play(audio)
        return reproducidos

    def stop(self) -> None:
        """Cancela la locución actual (lo usa el modo llamada)."""
        self.player.stop()

    def close(self) -> None:
        self.provider.close()


def sintetizar_chunks(
    manager: TTSManager, texto: str
) -> Iterable[AudioChunk]:
    """Genera audio sin reproducirlo (útil para pruebas y para VSeeFace)."""
    for fragmento in manager.dividir(texto):
        yield manager.provider.synthesize(fragmento)
