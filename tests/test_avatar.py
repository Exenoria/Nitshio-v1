"""Tests de emociones, transiciones y envío VMC a VSeeFace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from avatar.emotions import (
    EmotionController,
    EmotionError,
    cargar_emociones,
    emociones_disponibles,
)
from avatar.vmc import (
    AvatarController,
    VMCClient,
    probar_conexion,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def emociones():
    return cargar_emociones(ROOT / "config" / "emotions.yaml")


class SenderFalso:
    def __init__(self) -> None:
        self.mensajes: list[tuple[str, Any]] = []

    def send_message(self, address: str, value: Any) -> None:
        self.mensajes.append((address, value))


class VMCFalso:
    def __init__(self) -> None:
        self.enviados: list[dict[str, float]] = []
        self.cerrado = False

    def send_emotion(self, valores: dict[str, float]) -> None:
        self.enviados.append(dict(valores))

    def close(self) -> None:
        self.cerrado = True


# ---------------------------------------------------------------------------
# Configuración y controlador de emociones
# ---------------------------------------------------------------------------
def test_carga_emociones_reales(emociones) -> None:
    assert emociones.default == "neutral"
    assert set(emociones_disponibles(emociones)) == {
        "neutral",
        "fun",
        "angry",
        "joy",
        "sorrow",
        "surprised",
    }
    assert "Neutral" in emociones.blendshapes
    assert emociones.emotions["angry"].intensity == 1.0


def test_emocion_desconocida_falla(emociones) -> None:
    controller = EmotionController(emociones)
    with pytest.raises(EmotionError, match="desconocida"):
        controller.set_emotion("celosa")


def test_transicion_completa_vuelve_a_neutral(emociones) -> None:
    controller = EmotionController(emociones)
    controller.set_emotion("angry")
    assert controller.active_emotion == "angry"
    assert controller.phase == "in"

    valores = controller.tick(150)
    assert valores["Angry"] == pytest.approx(1.0, abs=0.01)
    assert controller.phase == "hold"

    controller.tick(1200)
    assert controller.phase == "out"

    valores = controller.tick(700)
    assert controller.phase == "idle"
    assert controller.active_emotion == "neutral"
    assert valores["Neutral"] == pytest.approx(1.0, abs=0.01)
    assert valores["Angry"] == pytest.approx(0.0, abs=0.01)


def test_misma_emocion_no_reinicia_fade(emociones) -> None:
    controller = EmotionController(emociones)
    controller.set_emotion("joy")
    controller.tick(50)
    antes = controller.current_values()
    controller.set_emotion("joy")
    assert controller.current_values() == antes
    assert controller.phase == "in"


def test_tick_seconds(emociones) -> None:
    controller = EmotionController(emociones)
    controller.set_emotion("fun")
    valores = controller.tick_seconds(0.22)
    assert valores["Fun"] == pytest.approx(0.9, abs=0.01)


def test_set_neutral(emociones) -> None:
    controller = EmotionController(emociones)
    controller.set_emotion("sorrow")
    controller.tick(350)
    controller.set_neutral()
    assert controller.active_emotion == "neutral"


# ---------------------------------------------------------------------------
# Cliente VMC
# ---------------------------------------------------------------------------
def test_vmc_envia_blendshape_y_lo_limita() -> None:
    sender = SenderFalso()
    cliente = VMCClient(sender=sender)
    cliente.send_blendshape("Angry", 1.5)
    cliente.send_blendshape("Angry", -0.2)
    assert sender.mensajes == [
        ("/VMC/Ext/Blend/Val", ["Angry", 1.0]),
        ("/VMC/Ext/Blend/Val", ["Angry", 0.0]),
    ]


def test_vmc_send_emotion_aplica_al_final() -> None:
    sender = SenderFalso()
    cliente = VMCClient(sender=sender)
    cliente.send_emotion({"Angry": 0.5, "Neutral": 0.0})
    assert len(sender.mensajes) == 3
    assert sender.mensajes[-1] == ("/VMC/Ext/Blend/Apply", [])


def test_vmc_send_time() -> None:
    sender = SenderFalso()
    cliente = VMCClient(sender=sender)
    cliente.send_time(12.5)
    assert sender.mensajes == [("/VMC/Ext/T", [12.5])]


def test_probar_conexion_envia_diagnostico() -> None:
    sender = SenderFalso()
    probar_conexion(sender=sender)
    assert sender.mensajes[0][0] == "/VMC/Ext/T"
    assert isinstance(sender.mensajes[0][1][0], float)
    assert sender.mensajes[1] == ("/VMC/Ext/Blend/Val", ["Neutral", 1.0])
    assert sender.mensajes[-1][0] == "/VMC/Ext/Blend/Apply"


# ---------------------------------------------------------------------------
# Controlador de avatar
# ---------------------------------------------------------------------------
def _config_avatar(enabled: bool = True) -> dict[str, Any]:
    return {
        "avatar": {
            "enabled": enabled,
            "host": "127.0.0.1",
            "port": 39539,
            "tick_hz": 60,
        },
        "persona": {"emotions_file": "config/emotions.yaml"},
    }


def test_avatar_envia_cambio_de_emocion(emociones) -> None:
    vmc = VMCFalso()
    avatar = AvatarController(
        _config_avatar(), vmc=vmc, emotion_config=emociones, root=ROOT
    )
    avatar.set_emotion("joy")
    avatar.tick(200)
    assert len(vmc.enviados) == 1
    assert vmc.enviados[0]["Joy"] == pytest.approx(0.95)

    avatar.send_current()
    assert len(vmc.enviados) == 1

    avatar.tick(1400)
    avatar.tick(200)
    assert len(vmc.enviados) == 2


def test_avatar_desactivado_no_envia(emociones) -> None:
    vmc = VMCFalso()
    avatar = AvatarController(
        _config_avatar(enabled=False),
        vmc=vmc,
        emotion_config=emociones,
        root=ROOT,
    )
    avatar.set_emotion("angry")
    avatar.send_current(force=True)
    assert vmc.enviados == []


def test_avatar_close_envia_neutral(emociones) -> None:
    vmc = VMCFalso()
    avatar = AvatarController(
        _config_avatar(), vmc=vmc, emotion_config=emociones, root=ROOT
    )
    avatar.set_emotion("angry")
    avatar.tick(150)
    avatar.close()
    assert vmc.cerrado is True
    assert vmc.enviados
    assert vmc.enviados[-1]["Neutral"] == pytest.approx(1.0)
