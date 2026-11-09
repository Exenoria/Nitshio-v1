"""Contrato base de las herramientas (plugins) de Nitshio.

Cada herramienta vive en su propio módulo dentro de ``tools/`` y expone
``create_tool(context)`` o una instancia ``TOOL``. El registro solo carga las
herramientas habilitadas en ``config.tools.enabled``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ToolError(RuntimeError):
    """La herramienta no pudo ejecutar la acción solicitada."""


@dataclass
class ToolContext:
    """Dependencias compartidas que recibe cada herramienta."""

    root: Path
    config: dict[str, Any]
    confirm: Callable[[str], bool] | None = None
    logger: logging.Logger | None = None
    memory: Any = None
    ollama: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def ruta(self, relativa: str | Path) -> Path:
        path = Path(relativa)
        return path if path.is_absolute() else self.root / path


@dataclass
class ToolResult:
    """Resultado uniforme de una herramienta."""

    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def exito(cls, content: str, **data: Any) -> "ToolResult":
        return cls(ok=True, content=str(content), data=dict(data))

    @classmethod
    def fallo(cls, error: str, **data: Any) -> "ToolResult":
        return cls(ok=False, content="", error=str(error), data=dict(data))


class Tool(ABC):
    """Interfaz de un plugin de herramienta."""

    name: str = "base"
    description: str = ""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    requires_confirmation: bool = False

    def schema(self) -> dict[str, Any]:
        """Esquema que se muestra al router y al modelo."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @abstractmethod
    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Ejecuta la herramienta con argumentos ya validados."""


def validar_argumentos(
    schema: dict[str, Any], argumentos: dict[str, Any]
) -> list[str]:
    """Validación básica de JSON Schema sin dependencias externas.

    Comprueba tipo objeto, claves requeridas, tipos simples y
    ``additionalProperties: false``. No pretende ser un validador completo.
    """
    problemas: list[str] = []
    if not isinstance(argumentos, dict):
        return ["Los argumentos deben ser un objeto JSON."]
    propiedades = schema.get("properties") or {}
    requeridas = schema.get("required") or []
    for clave in requeridas:
        if clave not in argumentos:
            problemas.append(f"Falta el argumento obligatorio '{clave}'.")
    if schema.get("additionalProperties") is False:
        for clave in argumentos:
            if clave not in propiedades:
                problemas.append(f"Argumento no permitido: '{clave}'.")
    tipos = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for clave, valor in argumentos.items():
        especificacion = propiedades.get(clave)
        if not isinstance(especificacion, dict):
            continue
        tipo = especificacion.get("type")
        esperado = tipos.get(str(tipo))
        if esperado is None:
            continue
        if isinstance(valor, bool) and tipo in {"integer", "number"}:
            problemas.append(f"'{clave}' debe ser {tipo}, no booleano.")
        elif not isinstance(valor, esperado):
            problemas.append(f"'{clave}' debe ser de tipo {tipo}.")
        if "enum" in especificacion and valor not in especificacion["enum"]:
            problemas.append(
                f"'{clave}' debe ser uno de: "
                + ", ".join(str(v) for v in especificacion["enum"])
            )
    return problemas
