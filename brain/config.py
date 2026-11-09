"""Carga centralizada de configuración y rutas del proyecto."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """La configuración del proyecto es inválida."""


def cargar_config(path: Path | str | None = None) -> dict[str, Any]:
    """Carga ``config/config.yaml`` y las variables de ``.env``.

    ``.env`` no sobreescribe variables ya definidas en el proceso, para que
    puedas hacer overrides desde la terminal.
    """
    load_dotenv(ROOT / ".env", override=False)
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise ConfigError(f"No existe la configuración: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"La configuración no es un objeto YAML: {config_path}")
    for clave in ("project", "ollama", "persona", "paths"):
        if clave not in data:
            raise ConfigError(f"Falta la sección '{clave}' en {config_path}")
    if "models" not in data["ollama"]:
        raise ConfigError(f"Falta 'ollama.models' en {config_path}")
    return data


def ruta_proyecto(relativa: str | Path) -> Path:
    """Resuelve una ruta relativa a la raíz del proyecto."""
    path = Path(relativa)
    return path if path.is_absolute() else ROOT / path


def host_ollama(config: dict[str, Any]) -> str:
    """Devuelve el host de Ollama, permitiendo override por variable de entorno."""
    return os.getenv("OLLAMA_HOST") or str(
        (config.get("ollama") or {}).get("host", "http://localhost:11434")
    )


def env(nombre: str, default: str | None = None) -> str | None:
    """Lee una variable de entorno ya cargada desde ``.env``."""
    valor = os.getenv(nombre)
    return valor if valor not in (None, "") else default
