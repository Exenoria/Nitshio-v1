"""Registro de versiones de Nitshio: activar, listar y hacer rollback.

``models/versions.json`` es la fuente de verdad. El YAML solo guarda el tag
activo que consume el pipeline; este módulo mantiene ambos sincronizados sin
reescribir el YAML completo (conserva comentarios y formato).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSIONS = ROOT / "models" / "versions.json"
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"

_ACTIVE_RE = re.compile(
    r'(?m)^(\s*active:\s*)"([^"]*)"(\s*(?:#.*)?)$'
)


class VersionError(RuntimeError):
    """El registro de versiones o el YAML son inválidos."""


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _sha256_archivo(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for bloque in iter(lambda: fh.read(65536), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _ruta(valor: Path | str) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def _normalizar_registro(data: dict[str, Any]) -> dict[str, Any]:
    versiones = data.get("versions")
    if not isinstance(versiones, list):
        versiones = []
    limpias: list[dict[str, Any]] = []
    for version in versiones:
        if isinstance(version, dict) and version.get("name"):
            limpias.append(dict(version))
    return {
        "active": data.get("active") or None,
        "previous_active": data.get("previous_active") or None,
        "versions": limpias,
    }


def load_registry(path: Path | str = DEFAULT_VERSIONS) -> dict[str, Any]:
    """Carga el registro; si no existe, devuelve uno vacío."""
    ruta = _ruta(path)
    if not ruta.exists():
        return {"active": None, "previous_active": None, "versions": []}
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VersionError(f"No se pudo leer {ruta}: {exc}") from exc
    if not isinstance(data, dict):
        raise VersionError(f"El registro {ruta} no es un objeto JSON.")
    return _normalizar_registro(data)


def save_registry(data: dict[str, Any], path: Path | str = DEFAULT_VERSIONS) -> Path:
    """Guarda el registro de forma atómica."""
    ruta = _ruta(path)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    normalizado = _normalizar_registro(data)
    temporal.write_text(
        json.dumps(normalizado, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporal.replace(ruta)
    return ruta


def find_version(
    name: str, data: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Devuelve la entrada de una versión o ``None``."""
    registro = data if data is not None else load_registry()
    for version in registro.get("versions", []):
        if version.get("name") == name:
            return version
    return None


def list_versions(
    data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Lista versiones en el orden del registro."""
    registro = data if data is not None else load_registry()
    return list(registro.get("versions", []))


def register_version(
    name: str,
    base: str,
    *,
    type: str = "modelfile",  # noqa: A002 - nombre público del campo JSON
    persona: Any | None = None,
    modelfile: str | None = None,
    modelfile_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    dataset_sha256: str | None = None,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    gguf_path: str | None = None,
    notes: str = "",
    activate: bool = False,
    registry_path: Path | str = DEFAULT_VERSIONS,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Registra (o actualiza) una versión y opcionalmente la activa."""
    nombre = str(name or "").strip()
    base_limpio = str(base or "").strip()
    if not nombre:
        raise VersionError("El nombre de la versión no puede estar vacío.")
    if not base_limpio:
        raise VersionError("El modelo base no puede estar vacío.")

    data = load_registry(registry_path)
    anterior = find_version(nombre, data)
    entrada: dict[str, Any] = dict(anterior or {})
    entrada.update(
        {
            "name": nombre,
            "base": base_limpio,
            "type": type,
            "created_at": entrada.get("created_at") or _ahora(),
            "updated_at": _ahora(),
        }
    )
    if persona is not None:
        entrada["persona_sha256"] = _sha256_texto(
            str(getattr(persona, "system_prompt", ""))
        )
        entrada["few_shots"] = len(getattr(persona, "few_shots", ()) or ())
    if modelfile is not None:
        entrada["modelfile_sha256"] = _sha256_texto(modelfile)
    if modelfile_path is not None:
        ruta_modelfile = _ruta(modelfile_path)
        entrada["modelfile_path"] = str(ruta_modelfile)
        if "modelfile_sha256" not in entrada:
            entrada["modelfile_sha256"] = _sha256_archivo(ruta_modelfile)
    if dataset_sha256:
        entrada["dataset_sha256"] = dataset_sha256
    elif dataset_path is not None:
        ruta_dataset = _ruta(dataset_path)
        entrada["dataset_path"] = str(ruta_dataset)
        entrada["dataset_sha256"] = _sha256_archivo(ruta_dataset)
    if parameters is not None:
        entrada["parameters"] = dict(parameters)
    if metrics is not None:
        entrada["metrics"] = metrics
    if gguf_path:
        entrada["gguf_path"] = str(gguf_path)
    if notes:
        entrada["notes"] = str(notes)
    entrada.setdefault("metrics", None)
    entrada.setdefault("notes", "Modelo registrado.")

    data["versions"] = [
        version
        for version in data.get("versions", [])
        if version.get("name") != nombre
    ] + [entrada]
    data["versions"].sort(key=lambda item: str(item.get("created_at", "")))
    if activate:
        anterior_activo = data.get("active")
        _activar_en_registro(data, nombre)
        data["previous_active"] = anterior_activo
        actualizar_config(nombre, config_path or DEFAULT_CONFIG)
    save_registry(data, registry_path)
    return entrada


def _activar_en_registro(data: dict[str, Any], name: str) -> None:
    if find_version(name, data) is None:
        raise VersionError(f"La versión {name!r} no está registrada.")
    for version in data.get("versions", []):
        if version.get("name") == name:
            version["status"] = "active"
            version["activated_at"] = _ahora()
        elif version.get("status") == "active":
            version["status"] = "archived"
    data["active"] = name


def actualizar_config(version: str, config_path: Path | str = DEFAULT_CONFIG) -> Path:
    """Cambia solo ``ollama.models.active`` conservando el resto del YAML."""
    ruta = _ruta(config_path)
    if not ruta.exists():
        raise VersionError(f"No existe la configuración: {ruta}")
    texto = ruta.read_text(encoding="utf-8")
    nuevo, reemplazos = _ACTIVE_RE.subn(
        lambda match: f'{match.group(1)}"{version}"{match.group(3)}',
        texto,
        count=1,
    )
    if reemplazos != 1:
        raise VersionError(
            "No se encontró 'active: \"...\"' en config/config.yaml; "
            "cámbiala manualmente o revisa el formato."
        )
    ruta.write_text(nuevo, encoding="utf-8")
    return ruta


def activate_version(
    name: str,
    *,
    registry_path: Path | str = DEFAULT_VERSIONS,
    config_path: Path | str = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Activa una versión registrada y actualiza el YAML."""
    data = load_registry(registry_path)
    entrada = find_version(name, data)
    if entrada is None:
        raise VersionError(f"La versión {name!r} no está registrada.")
    actual = data.get("active")
    if actual != name:
        data["previous_active"] = actual
    _activar_en_registro(data, name)
    # Se actualiza el YAML antes de guardar para no dejar el registro
    # apuntando a una versión que el pipeline no puede cargar.
    actualizar_config(name, config_path)
    save_registry(data, registry_path)
    return entrada


def rollback_version(
    *,
    registry_path: Path | str = DEFAULT_VERSIONS,
    config_path: Path | str = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Vuelve a la versión anterior o, si no hay, a la última distinta."""
    data = load_registry(registry_path)
    actual = data.get("active")
    objetivo = data.get("previous_active")
    if not objetivo or objetivo == actual or find_version(objetivo, data) is None:
        candidatas = [
            version
            for version in data.get("versions", [])
            if version.get("name") != actual
        ]
        if not candidatas:
            raise VersionError("No hay otra versión a la que volver.")
        candidatas.sort(key=lambda item: str(item.get("created_at", "")))
        objetivo = str(candidatas[-1]["name"])
    if find_version(objetivo, data) is None:
        raise VersionError(f"La versión de rollback {objetivo!r} no existe.")
    data["previous_active"] = actual
    _activar_en_registro(data, objetivo)
    actualizar_config(objetivo, config_path)
    save_registry(data, registry_path)
    return find_version(objetivo, data) or {}


def update_metrics(
    name: str,
    metrics: dict[str, Any],
    *,
    registry_path: Path | str = DEFAULT_VERSIONS,
) -> dict[str, Any]:
    """Adjunta métricas de evaluación a una versión registrada."""
    data = load_registry(registry_path)
    entrada = find_version(name, data)
    if entrada is None:
        raise VersionError(f"La versión {name!r} no está registrada.")
    entrada["metrics"] = dict(metrics)
    entrada["metrics_updated_at"] = _ahora()
    save_registry(data, registry_path)
    return entrada


def verificar_en_ollama(name: str) -> None:
    """Comprueba con ``ollama show`` que el tag existe."""
    try:
        subprocess.run(
            ["ollama", "show", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise VersionError(
            "No se encontró el comando 'ollama'. Instala Ollama y agrégalo al PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detalle = (exc.stderr or exc.stdout or "").strip()
        raise VersionError(
            f"Ollama no reconoce la versión {name!r}: {detalle or exc.returncode}"
        ) from exc


def _tabla(versiones: list[dict[str, Any]], activa: str | None) -> str:
    lineas = [
        "  Nombre        Base                         Tipo       Estado",
        "  ------------  ---------------------------  ---------  --------",
    ]
    for version in versiones:
        nombre = str(version.get("name", ""))
        base = str(version.get("base", ""))
        tipo = str(version.get("type", ""))
        estado = "ACTIVA" if nombre == activa else str(version.get("status", ""))
        lineas.append(f"  {nombre:<12}  {base:<27}  {tipo:<9}  {estado}")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gestiona versiones de Nitshio.")
    parser.add_argument(
        "--versions",
        type=Path,
        default=DEFAULT_VERSIONS,
        help="Ruta alternativa a models/versions.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Ruta alternativa a config/config.yaml.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("list", help="Lista las versiones registradas.")
    show = sub.add_parser("show", help="Muestra una versión en JSON.")
    show.add_argument("name")

    registrar = sub.add_parser("register", help="Registra una versión.")
    registrar.add_argument("name")
    registrar.add_argument("--base", required=True)
    registrar.add_argument("--type", default="modelfile")
    registrar.add_argument("--notes", default="")
    registrar.add_argument("--modelfile", type=Path)
    registrar.add_argument("--dataset", type=Path)
    registrar.add_argument("--gguf")
    registrar.add_argument("--activate", action="store_true")

    activar = sub.add_parser("activate", help="Activa una versión.")
    activar.add_argument("name")
    activar.add_argument(
        "--verify",
        action="store_true",
        help="Comprueba con 'ollama show' antes de activar.",
    )

    sub.add_parser("rollback", help="Vuelve a la versión anterior.")

    metricas = sub.add_parser("metrics", help="Adjunta métricas a una versión.")
    metricas.add_argument("name")
    metricas.add_argument("file", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.comando == "list":
            data = load_registry(args.versions)
            print(_tabla(data.get("versions", []), data.get("active")))
            return 0
        if args.comando == "show":
            data = load_registry(args.versions)
            entrada = find_version(args.name, data)
            if entrada is None:
                raise VersionError(f"No existe la versión {args.name!r}.")
            print(json.dumps(entrada, ensure_ascii=False, indent=2))
            return 0
        if args.comando == "register":
            entrada = register_version(
                args.name,
                args.base,
                type=args.type,
                modelfile_path=args.modelfile,
                dataset_path=args.dataset,
                gguf_path=args.gguf,
                notes=args.notes,
                activate=args.activate,
                registry_path=args.versions,
                config_path=args.config,
            )
            print(json.dumps(entrada, ensure_ascii=False, indent=2))
            return 0
        if args.comando == "activate":
            if args.verify:
                verificar_en_ollama(args.name)
            entrada = activate_version(
                args.name,
                registry_path=args.versions,
                config_path=args.config,
            )
            print(f"Versión activa: {entrada.get('name')}")
            return 0
        if args.comando == "rollback":
            entrada = rollback_version(
                registry_path=args.versions,
                config_path=args.config,
            )
            print(f"Rollback a: {entrada.get('name')}")
            return 0
        if args.comando == "metrics":
            datos = json.loads(args.file.read_text(encoding="utf-8"))
            if not isinstance(datos, dict):
                raise VersionError("El archivo de métricas debe ser un objeto JSON.")
            update_metrics(args.name, datos, registry_path=args.versions)
            print(f"Métricas actualizadas: {args.name}")
            return 0
    except (OSError, VersionError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
