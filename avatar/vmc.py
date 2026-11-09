"""Envío de blendshapes y emociones a VSeeFace por VMC/OSC.

VSeeFace escucha VMC en ``127.0.0.1:39539`` por defecto. Se envían valores
``/VMC/Ext/Blend/Val`` y un ``/VMC/Ext/Blend/Apply`` al final de cada tick.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Protocol

from avatar.emotions import (
    EmotionConfig,
    EmotionController,
    EmotionError,
    cargar_emociones,
)


class VMCError(RuntimeError):
    """No se pudo enviar el mensaje VMC."""


class _Sender(Protocol):
    def send_message(self, address: str, value: Any) -> None: ...


class VMCClient:
    """Cliente OSC mínimo para VMC."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 39539,
        *,
        address_prefix: str = "/VMC/Ext",
        sender: _Sender | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.address_prefix = address_prefix.rstrip("/")
        self._sender = sender

    @property
    def sender(self) -> _Sender:
        if self._sender is None:
            try:
                from pythonosc.udp_client import SimpleUDPClient
            except ImportError as exc:  # pragma: no cover
                raise VMCError(
                    "python-osc no está instalado. Instala requirements.txt."
                ) from exc
            self._sender = SimpleUDPClient(self.host, self.port)
        return self._sender

    def send(self, address: str, *args: Any) -> None:
        try:
            self.sender.send_message(address, list(args))
        except Exception as exc:
            raise VMCError(f"No se pudo enviar {address}: {exc}") from exc

    def send_blendshape(self, nombre: str, valor: float) -> None:
        self.send(
            f"{self.address_prefix}/Blend/Val",
            str(nombre),
            float(max(0.0, min(1.0, valor))),
        )

    def apply(self) -> None:
        self.send(f"{self.address_prefix}/Blend/Apply")

    def send_emotion(self, valores: dict[str, float]) -> None:
        for nombre, valor in valores.items():
            self.send_blendshape(nombre, valor)
        self.apply()

    def send_time(self, segundos: float | None = None) -> None:
        self.send(f"{self.address_prefix}/T", float(segundos or time.time()))

    def close(self) -> None:
        self._sender = None


class AvatarController:
    """Controla la emoción actual y la envía a VSeeFace a ``tick_hz``."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        vmc: VMCClient | None = None,
        emotion_config: EmotionConfig | None = None,
        root: Path | None = None,
    ) -> None:
        avatar_cfg = config.get("avatar") or {}
        self.enabled = bool(avatar_cfg.get("enabled", True))
        self.tick_hz = max(1, int(avatar_cfg.get("tick_hz", 60)))
        emociones_rel = (
            (config.get("persona") or {}).get(
                "emotions_file", "config/emotions.yaml"
            )
        )
        if emotion_config is None:
            path = Path(emociones_rel)
            if not path.is_absolute():
                path = (root or Path(__file__).resolve().parents[1]) / path
            emotion_config = cargar_emociones(path)
        self.emotion_config = emotion_config
        self.emotion = EmotionController(emotion_config)
        self.vmc = vmc or VMCClient(
            str(avatar_cfg.get("host", "127.0.0.1")),
            int(avatar_cfg.get("port", 39539)),
            address_prefix=str(avatar_cfg.get("address_prefix", "/VMC/Ext")),
        )
        self._ultimos: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def set_emotion(self, nombre: str) -> None:
        self.emotion.set_emotion(nombre)

    def send_current(self, *, force: bool = False) -> None:
        """Envía los valores actuales; por defecto solo si cambiaron."""
        with self._lock:
            valores = self.emotion.current_values()
            if not force and valores == self._ultimos:
                return
            if not self.enabled:
                self._ultimos = valores
                return
            try:
                self.vmc.send_emotion(valores)
            except VMCError:
                if force:
                    raise
            self._ultimos = valores

    def tick(self, dt_ms: float | None = None) -> dict[str, float]:
        """Avanza la emoción y envía el estado a VSeeFace."""
        if dt_ms is None:
            dt_ms = 1000.0 / self.tick_hz
        valores = self.emotion.tick(dt_ms)
        self.send_current()
        return valores

    def run(self, stop_event: threading.Event | None = None) -> None:
        """Bucle de animación hasta que se pida parar."""
        stop = stop_event or self._stop
        intervalo = 1.0 / self.tick_hz
        siguiente = time.monotonic()
        while not stop.is_set():
            self.tick(1000.0 / self.tick_hz)
            siguiente += intervalo
            espera = siguiente - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            else:
                siguiente = time.monotonic()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, name="nitshio-avatar", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        try:
            # Completa la transición a neutral antes de enviar el último estado.
            self.emotion.set_neutral()
            self.emotion.tick(
                self.emotion.config.return_to_neutral_ms
                + self.emotion.config.crossfade_ms
                + 1000
            )
            self.send_current(force=True)
        except (VMCError, EmotionError):
            pass

    def close(self) -> None:
        self.stop()
        self.vmc.close()


def probar_conexion(
    host: str = "127.0.0.1", port: int = 39539, *, sender: _Sender | None = None
) -> VMCClient:
    """Crea un cliente y envía un tick de diagnóstico."""
    cliente = VMCClient(host, port, sender=sender)
    cliente.send_time()
    cliente.send_emotion({"Neutral": 1.0})
    return cliente
