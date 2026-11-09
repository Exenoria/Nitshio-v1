#!/usr/bin/env python
"""Punto de entrada de Nitshio.

Modos:
    python run.py --mode text
    python run.py --mode text --once "Hola, Nitshio"
    python run.py --mode voice
    python run.py --mode call
    python run.py --mode diagnose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config, host_ollama
from brain.llm.ollama_client import OllamaClient, OllamaError
from brain.llm.orchestrator import crear_orquestador
from brain.persona.loader import PersonaError, cargar_persona
from pipeline.text import TextPipeline


def _configurar_salida() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _configurar_logging(config: dict) -> logging.Logger:
    logging_cfg = config.get("logging") or {}
    nivel = getattr(logging, str(logging_cfg.get("level", "INFO")).upper(), logging.INFO)
    manejadores: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    archivo = logging_cfg.get("file")
    if archivo:
        ruta = Path(archivo)
        if not ruta.is_absolute():
            ruta = ROOT / ruta
        ruta.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        manejadores.append(
            RotatingFileHandler(
                ruta,
                maxBytes=int(logging_cfg.get("max_bytes", 5_242_880)),
                backupCount=int(logging_cfg.get("backup_count", 3)),
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=manejadores,
    )
    return logging.getLogger("nitshio")


def _confirmar(mensaje: str) -> bool:
    try:
        respuesta = input(f"{mensaje} [s/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return respuesta in {"s", "si", "sí", "y", "yes"}


def _crear_cliente(config: dict) -> OllamaClient:
    ollama_cfg = config.get("ollama") or {}
    return OllamaClient(
        host_ollama(config),
        timeout=float(ollama_cfg.get("request_timeout_seconds", 180)),
        keep_alive=str(ollama_cfg.get("keep_alive", "30m")),
    )


def _diagnostico(config: dict) -> int:
    print("=" * 64)
    print("  NITSHIO — diagnóstico del pipeline")
    print("=" * 64)
    print(f"Raíz: {ROOT}")
    print(f"Modelo activo: {(config['ollama']['models']).get('active')}")
    print(f"Router: {(config['ollama']['models']).get('router')}")
    print(f"Embeddings: {(config['ollama']['models']).get('embeddings')}")

    cliente = _crear_cliente(config)
    try:
        version = cliente.health().get("version", "?")
        tags = [m.get("name", "") for m in cliente.list_models()]
        print(f"Ollama: OK (v{version})")
        print("Modelos: " + ", ".join(tags))
    except OllamaError as exc:
        print(f"Ollama: ERROR — {exc}")

    print("\nAudio de salida:")
    try:
        import sounddevice as sd

        for indice, dispositivo in enumerate(sd.query_devices()):
            salidas = int(dispositivo.get("max_output_channels", 0))
            if salidas:
                print(f"  [{indice}] {dispositivo['name']} ({salidas} canales)")
        print(
            "  Configurado: "
            + str(((config.get("tts") or {}).get("audio") or {}).get("output_device"))
        )
    except Exception as exc:
        print(f"  sounddevice no disponible: {exc}")

    print("\nTTS Piper:")
    piper_cfg = (config.get("tts") or {}).get("piper") or {}
    modelo = Path(str(piper_cfg.get("model_path", "")))
    if not modelo.is_absolute():
        modelo = ROOT / modelo
    print(f"  Voz: {modelo} ({'OK' if modelo.exists() else 'FALTA'})")
    if not modelo.exists():
        print("  Descárgala con: python scripts/download_voice.py")

    print("\nSTT Moonshine:")
    stt_cfg = config.get("stt") or {}
    print(
        f"  Proveedor: {stt_cfg.get('provider')} | "
        f"Idioma: {stt_cfg.get('language')} | Modelo: {stt_cfg.get('model')}"
    )
    try:
        import moonshine_voice  # noqa: F401

        print("  Paquete: OK")
    except ImportError:
        print("  Paquete: FALTA (pip install -r requirements.txt)")

    print("\nAvatar VMC:")
    avatar_cfg = config.get("avatar") or {}
    print(
        f"  {avatar_cfg.get('host')}:{avatar_cfg.get('port')} "
        f"(prefix {avatar_cfg.get('address_prefix')})"
    )
    print("\nHerramientas habilitadas:")
    print("  " + ", ".join((config.get("tools") or {}).get("enabled") or []))
    print("=" * 64)
    return 0


def main(argv: list[str] | None = None) -> int:
    _configurar_salida()
    parser = argparse.ArgumentParser(description="Nitshio — waifu local.")
    parser.add_argument(
        "--mode",
        choices=["text", "voice", "call", "diagnose"],
        default="text",
    )
    parser.add_argument("--once", default=None, help="Un mensaje y salir.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Override de ollama.models.active.")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--no-avatar", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--speak", action="store_true", help="Leer respuestas en texto.")
    parser.add_argument("--json", action="store_true", help="Salida JSON en --once.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 1
    if args.model:
        config["ollama"]["models"]["active"] = args.model
    logger = _configurar_logging(config)

    if args.mode == "diagnose":
        return _diagnostico(config)

    try:
        persona = cargar_persona(config)
    except (OSError, ValueError, PersonaError) as exc:
        print(f"ERROR de personalidad: {exc}", file=sys.stderr)
        return 1

    cliente = _crear_cliente(config)
    try:
        cliente.health()
    except OllamaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Abre Ollama o ejecuta `ollama serve`.", file=sys.stderr)
        return 1

    con_tts = not args.no_tts and (args.mode in {"voice", "call"} or args.speak)
    orquestador = crear_orquestador(
        config,
        persona,
        cliente,
        con_memoria=not args.no_memory,
        con_herramientas=not args.no_tools,
        con_avatar=not args.no_avatar,
        con_tts=con_tts,
        confirm=_confirmar,
        logger=logger,
    )
    if orquestador.avatar is not None:
        try:
            orquestador.avatar.start()
        except Exception as exc:
            logger.warning("No se pudo iniciar el avatar: %s", exc)

    try:
        if args.mode == "text":
            pipeline = TextPipeline(orquestador, debug=args.debug)
            if args.once is not None:
                resultado = pipeline.responder(args.once, speak=args.speak)
                if args.json:
                    print(
                        json.dumps(
                            resultado.parsed.data.model_dump(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                else:
                    pipeline.imprimir(resultado)
                return 0
            return pipeline.run(speak=args.speak)

        from pipeline.voice import CallMode, VoicePipeline
        from stt.base import crear_stt
        from tts.manager import TTSManager

        stt = crear_stt(config)
        tts = orquestador.tts or (None if args.no_tts else TTSManager(config))
        orquestador.tts = tts
        pipeline_voz = VoicePipeline(
            config, orquestador, stt=stt, tts=tts
        )

        def on_event(evento) -> None:
            print(
                f"Tú (voz): {evento.transcript.text}\n"
                f"Nitshio [{evento.turn.emotion}]: {evento.turn.response}"
            )

        if args.mode == "voice":
            if args.once is not None:
                # En modo voz, --once acepta texto para probar el pipeline sin micrófono.
                resultado = orquestador.process(args.once, speak=True)
                if args.json:
                    print(
                        json.dumps(
                            resultado.parsed.data.model_dump(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                else:
                    print(
                        f"Nitshio [{resultado.emotion}]: {resultado.response}"
                    )
                return 0
            try:
                pipeline_voz.run_forever(on_event=on_event)
            except KeyboardInterrupt:
                print("\nDetenido.")
            return 0

        modo_llamada = CallMode(pipeline_voz, on_event=on_event, logger=logger)
        try:
            modo_llamada.run()
        except KeyboardInterrupt:
            print("\nDetenido.")
            modo_llamada.stop()
        return 0
    finally:
        if orquestador.avatar is not None:
            orquestador.avatar.close()
        if orquestador.tts is not None:
            orquestador.tts.close()
        if orquestador.memory is not None:
            orquestador.memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
