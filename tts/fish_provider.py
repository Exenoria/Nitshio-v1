"""Proveedor TTS opcional de Fish Audio.

Requiere ``FISH_API_KEY`` en ``.env``. Si no hay clave, el pipeline debe usar
Piper; este adaptador no se activa por accidente.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from tts.base import AudioChunk, TTSError, TTSProvider


class FishProvider(TTSProvider):
    """Sintetiza voz con la API HTTP de Fish Audio."""

    name = "fish"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        cfg = config or {}
        if "tts" in cfg and "fish" not in cfg:
            cfg = cfg.get("tts") or {}
        fish_cfg = cfg.get("fish") or {}
        self.base_url = str(
            fish_cfg.get("base_url", "https://api.fish.audio/v1/tts")
        )
        self.api_key_env = str(fish_cfg.get("api_key_env", "FISH_API_KEY"))
        self.model = str(fish_cfg.get("model", "s1"))
        self.voice_id = str(fish_cfg.get("voice_id", ""))
        self.format = str(fish_cfg.get("format", "mp3"))
        self.latency = str(fish_cfg.get("latency", "balanced"))
        self.timeout = float(fish_cfg.get("timeout_seconds", 60))
        self.sample_rate = 22050
        self.session = session or requests.Session()

    def _clave(self) -> str:
        clave = os.getenv(self.api_key_env, "").strip()
        if not clave:
            raise TTSError(
                f"Falta {self.api_key_env} en .env; Fish Audio es opcional. "
                "Usa tts.provider: piper para voz local."
            )
        return clave

    def synthesize(self, text: str) -> AudioChunk:
        texto = " ".join(str(text or "").split())
        if not texto:
            raise TTSError("No hay texto para sintetizar.")
        payload: dict[str, Any] = {
            "text": texto,
            "format": self.format,
            "latency": self.latency,
            "model": self.model,
        }
        if self.voice_id:
            payload["reference_id"] = self.voice_id
        try:
            respuesta = self.session.post(
                self.base_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._clave()}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            respuesta.raise_for_status()
        except requests.RequestException as exc:
            detalle = ""
            respuesta = getattr(exc, "response", None)
            if respuesta is not None:
                detalle = f" — {respuesta.text[:300]}"
            raise TTSError(f"Fish Audio falló: {exc}{detalle}") from exc
        if not respuesta.content:
            raise TTSError("Fish Audio devolvió audio vacío.")
        return AudioChunk(
            data=respuesta.content,
            sample_rate=self.sample_rate,
            format=self.format,
            text=texto,
        )
