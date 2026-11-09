#!/usr/bin/env python
"""Descarga la voz Piper configurada en ``config/config.yaml``.

Uso:
    python scripts/download_voice.py
    python scripts/download_voice.py --force

Los archivos se guardan en las rutas configuradas (``tts.piper.model_path`` y
``config_path``). No se hardcodea el nombre de la voz en el código.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config


def _ruta(valor: str) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def descargar(url: str, destino: Path, *, force: bool = False) -> None:
    if destino.exists() and destino.stat().st_size > 0 and not force:
        print(f"Ya existe: {destino}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".part")
    print(f"Descargando {url}")
    with requests.get(url, stream=True, timeout=120) as respuesta:
        respuesta.raise_for_status()
        total = int(respuesta.headers.get("content-length", 0) or 0)
        descargado = 0
        with temporal.open("wb") as fh:
            for bloque in respuesta.iter_content(chunk_size=1024 * 256):
                if not bloque:
                    continue
                fh.write(bloque)
                descargado += len(bloque)
                if total:
                    porcentaje = 100 * descargado / total
                    print(
                        f"\r  {porcentaje:5.1f}% "
                        f"({descargado / 1024 / 1024:.1f} MB)",
                        end="",
                        flush=True,
                    )
    print()
    temporal.replace(destino)
    print(f"Guardado: {destino}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga la voz Piper.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 1

    piper = (config.get("tts") or {}).get("piper") or {}
    modelo = piper.get("model_path")
    configuracion = piper.get("config_path")
    if not modelo or not configuracion:
        print(
            "ERROR: configura tts.piper.model_path y config_path.",
            file=sys.stderr,
        )
        return 1
    base = str(
        piper.get(
            "download_base_url",
            "https://huggingface.co/rhasspy/piper-voices/resolve/main",
        )
    ).rstrip("/")
    ruta_voz = str(
        piper.get(
            "voice_repo_path",
            "es/es_MX/claude/high/es_MX-claude-high",
        )
    ).strip("/")
    url_modelo = f"{base}/{ruta_voz}.onnx"
    url_config = f"{base}/{ruta_voz}.onnx.json"
    try:
        descargar(url_modelo, _ruta(modelo), force=args.force)
        descargar(url_config, _ruta(configuracion), force=args.force)
    except requests.RequestException as exc:
        print(f"ERROR al descargar: {exc}", file=sys.stderr)
        return 1
    print("Voz Piper lista.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
