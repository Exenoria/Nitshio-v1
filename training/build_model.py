#!/usr/bin/env python
"""Construye un Modelfile de Nitshio y, opcionalmente, un modelo en Ollama.

Ejemplos:
    python training/build_model.py --version nitshio-v1 --create --activate
    python training/build_model.py --version nitshio-v2 --print
    python training/build_model.py --version nitshio-v3 --create

El script valida los few-shots antes de compilar y no hardcodea modelos: el
modelo base y los parámetros se leen de ``config/config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config
from brain.persona.loader import PersonaError, cargar_persona, validar_few_shots
from training.versions import (
    DEFAULT_VERSIONS,
    VersionError,
    activate_version,
    actualizar_config,
    register_version,
)

DEFAULT_TEMPLATE = ROOT / "models" / "Modelfile.template"
DEFAULT_OUTPUT = ROOT / "models" / "Modelfile"
VERSIONS_FILE = DEFAULT_VERSIONS


def _quoted(texto: str) -> str:
    """Devuelve un string entre comillas compatible con Modelfile."""
    return json.dumps(texto, ensure_ascii=False)


def render_modelfile(
    config: dict[str, Any],
    persona: Any,
    template_path: Path | None = None,
) -> str:
    """Renderiza el Modelfile completo a partir de config y persona."""
    ruta_template = Path(template_path) if template_path else DEFAULT_TEMPLATE
    if not ruta_template.exists():
        raise FileNotFoundError(f"No existe el template: {ruta_template}")
    template = ruta_template.read_text(encoding="utf-8")

    modelos = config["ollama"]["models"]
    base = str(modelos["base"])
    system = persona.system_prompt
    if '"""' in system:
        raise ValueError(
            "El prompt de personalidad contiene triple comilla; cambia el "
            "delimitador del SYSTEM en el template."
        )

    opciones = dict(config["ollama"].get("options", {}).get("primary", {}))
    lineas_parametros: list[str] = []
    for clave in ("temperature", "top_p", "repeat_penalty", "num_ctx", "num_predict"):
        if clave in opciones:
            lineas_parametros.append(f"PARAMETER {clave} {opciones[clave]}")
    # Stops del formato Phi-3.5. Se declaran siempre para que el modelo no se
    # salga de su turno.
    for stop in ("<|system|>", "<|user|>", "<|end|>", "<|assistant|>"):
        lineas_parametros.append(f'PARAMETER stop "{stop}"')

    mensajes: list[str] = []
    for ejemplo in persona.few_shots:
        mensajes.append(f"MESSAGE user {_quoted(ejemplo.user)}")
        mensajes.append(f"MESSAGE assistant {_quoted(ejemplo.assistant_json())}")

    render = template
    render = render.replace("__BASE_MODEL__", base)
    render = render.replace("__SYSTEM__", system)
    render = render.replace("__PARAMETERS__", "\n".join(lineas_parametros))
    render = render.replace("__MESSAGES__", "\n".join(mensajes))
    return render


def registrar_version(
    version: str,
    base: str,
    persona: Any,
    modelfile: str,
    *,
    activate: bool = False,
    dataset_path: Path | str | None = None,
    notes: str = "",
    versions_file: Path | str = VERSIONS_FILE,
) -> dict[str, Any]:
    """Registra la versión en ``models/versions.json``."""
    return register_version(
        version,
        base,
        type="modelfile",
        persona=persona,
        modelfile=modelfile,
        dataset_path=dataset_path,
        notes=notes or "Modelo construido desde Modelfile.",
        activate=activate,
        registry_path=versions_file,
    )


def activar_en_config(version: str, config_path: Path | None = None) -> None:
    """Cambia ``ollama.models.active`` sin reescribir todo el YAML."""
    actualizar_config(version, config_path or (ROOT / "config" / "config.yaml"))


def crear_en_ollama(version: str, modelfile_path: Path) -> None:
    """Ejecuta ``ollama create`` y muestra su salida."""
    if not modelfile_path.exists():
        raise FileNotFoundError(modelfile_path)
    comando = ["ollama", "create", version, "-f", str(modelfile_path)]
    print("Ejecutando:", " ".join(comando))
    try:
        subprocess.run(comando, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se encontró el comando 'ollama'. Instala Ollama y asegúrate "
            "de que esté en PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ollama create falló con código {exc.returncode}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construye nitshio-vN.")
    parser.add_argument(
        "--version",
        default="nitshio-v1",
        help="Nombre del modelo/versión (ej. nitshio-v1).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Ejecuta `ollama create` después de escribir el Modelfile.",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Actualiza ollama.models.active en config/config.yaml.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Imprime el Modelfile en vez de escribirlo.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta alternativa a config.yaml.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset usado; se guarda su hash en versions.json.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Nota libre para el registro de versión.",
    )
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
        persona = cargar_persona(config)
    except (OSError, ValueError, PersonaError) as exc:
        print(f"ERROR de configuración/personalidad: {exc}", file=sys.stderr)
        return 1

    problemas = validar_few_shots(persona)
    if problemas:
        print("ERROR: los few-shots no son válidos:", file=sys.stderr)
        for problema in problemas:
            print(f"  - {problema}", file=sys.stderr)
        return 1

    try:
        modelfile = render_modelfile(config, persona)
    except (OSError, ValueError) as exc:
        print(f"ERROR al renderizar: {exc}", file=sys.stderr)
        return 1

    if args.print:
        print(modelfile)
        return 0

    DEFAULT_OUTPUT.write_text(modelfile, encoding="utf-8")
    print(f"Modelfile escrito en {DEFAULT_OUTPUT}")

    try:
        registrar_version(
            args.version,
            str(config["ollama"]["models"]["base"]),
            persona,
            modelfile,
            activate=False,
            dataset_path=args.dataset,
            notes=args.notes,
        )
    except (OSError, VersionError) as exc:
        print(f"ERROR al registrar la versión: {exc}", file=sys.stderr)
        return 1
    print(f"Versión registrada: {args.version}")

    if args.create:
        try:
            crear_en_ollama(args.version, DEFAULT_OUTPUT)
        except (OSError, RuntimeError) as exc:
            print(f"ERROR al crear el modelo: {exc}", file=sys.stderr)
            return 1
        print(f"Modelo creado en Ollama: {args.version}")

    if args.activate:
        try:
            activate_version(
                args.version,
                config_path=args.config or (ROOT / "config" / "config.yaml"),
            )
        except (OSError, VersionError) as exc:
            print(f"ERROR al activar: {exc}", file=sys.stderr)
            return 1
        print(f"Modelo activo en config: {args.version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
