#!/usr/bin/env python
"""Recorre las emociones de ``config/emotions.yaml`` y las envía por VMC.

Uso:
    python scripts/test_emotions.py --all
    python scripts/test_emotions.py --emotion angry
    python scripts/test_emotions.py --all --dry-run
    python scripts/test_emotions.py --all --host 127.0.0.1 --port 39539

``--dry-run`` no envía UDP: imprime los mensajes que se enviarían. Úsalo para
comprobar el mapeo sin abrir VSeeFace.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatar.emotions import cargar_emociones
from avatar.vmc import AvatarController, VMCClient
from brain.config import cargar_config


class SenderDryRun:
    """Sender falso que imprime los mensajes OSC."""

    def __init__(self) -> None:
        self.mensajes: list[tuple[str, Any]] = []

    def send_message(self, address: str, value: Any) -> None:
        self.mensajes.append((address, value))
        print(f"  OSC {address} {value}")


def _duracion_ms(emocion) -> int:
    return emocion.fade_in_ms + emocion.hold_ms + emocion.fade_out_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prueba de emociones VMC.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--emotion", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 1

    emociones_rel = (config.get("persona") or {}).get(
        "emotions_file", "config/emotions.yaml"
    )
    ruta_emociones = Path(emociones_rel)
    if not ruta_emociones.is_absolute():
        ruta_emociones = ROOT / ruta_emociones
    try:
        emotion_config = cargar_emociones(ruta_emociones)
    except Exception as exc:
        print(f"ERROR de emociones: {exc}", file=sys.stderr)
        return 1

    avatar_cfg = config.get("avatar") or {}
    sender = SenderDryRun() if args.dry_run else None
    vmc = VMCClient(
        args.host or str(avatar_cfg.get("host", "127.0.0.1")),
        args.port or int(avatar_cfg.get("port", 39539)),
        address_prefix=str(avatar_cfg.get("address_prefix", "/VMC/Ext")),
        sender=sender,
    )
    controlador = AvatarController(
        config, vmc=vmc, emotion_config=emotion_config, root=ROOT
    )
    tick_ms = 1000.0 / max(1, controlador.tick_hz)

    if args.emotion:
        nombres = [args.emotion]
    elif args.all:
        nombres = list(emotion_config.emotions)
    else:
        nombres = [emotion_config.default]

    print(f"Probando {len(nombres)} emoción(es) en {vmc.host}:{vmc.port}")
    for nombre in nombres:
        if nombre not in emotion_config.emotions:
            print(f"ERROR: emoción desconocida {nombre!r}", file=sys.stderr)
            return 1
        emocion = emotion_config.emotions[nombre]
        print(f"\n[{nombre}] {emocion.display_name} -> {emocion.blendshape}")
        controlador.set_emotion(nombre)
        total = _duracion_ms(emocion) + 100
        transcurrido = 0
        picos: dict[str, float] = {}
        while transcurrido < total:
            valores = controlador.tick(tick_ms)
            for blendshape, valor in valores.items():
                picos[blendshape] = max(picos.get(blendshape, 0.0), valor)
            transcurrido += tick_ms
            if not args.dry_run:
                time.sleep(tick_ms / 1000.0)
        print(f"  valores máximos: {picos}")
        print(f"  valores finales: {controlador.emotion.current_values()}")

    controlador.emotion.set_neutral()
    controlador.send_current(force=True)
    controlador.close()
    print("\nEmociones probadas; vuelta a neutral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
