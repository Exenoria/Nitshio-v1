"""Herramienta de autoconocimiento: leer el código y la documentación propios.

Solo permite rutas dentro de la raíz del proyecto, con extensiones de texto y
límites de tamaño. No sustituye a ``files``: es de solo lectura.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolContext, ToolResult

_EXTENSIONES = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
    ".bat",
    ".ps1",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".ipynb",
}
_IGNORAR = {".venv", ".git", "__pycache__", "data", "logs", "node_modules", ".pytest_cache"}


class ReadOwnCodeTool(Tool):
    name = "read_own_code"
    description = (
        "Lee el código o la documentación de Nitshio para responder preguntas "
        "sobre cómo funciona, dónde está un archivo o qué configuración usa."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list"],
                "description": "Leer un archivo o listar archivos del proyecto.",
            },
            "path": {
                "type": "string",
                "description": "Ruta relativa a la raíz del proyecto.",
            },
            "start_line": {
                "type": "integer",
                "description": "Primera línea a leer (1-based).",
            },
            "end_line": {
                "type": "integer",
                "description": "Última línea a leer (inclusive).",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, context: ToolContext) -> None:
        self.root = Path(os.path.realpath(str(context.root)))
        self.max_bytes = int(
            ((context.config.get("tools") or {}).get("security") or {}).get(
                "max_read_bytes", 200_000
            )
        )

    def _resolver(self, ruta: str) -> Path:
        candidato = Path(ruta)
        if not candidato.is_absolute():
            candidato = self.root / candidato
        real = Path(os.path.realpath(str(candidato)))
        if not (real == self.root or self.root in real.parents):
            raise ValueError(f"Ruta fuera del proyecto: {ruta!r}")
        if real.is_symlink():
            raise ValueError("No se permiten symlinks en read_own_code.")
        if not real.exists():
            raise ValueError(f"No existe: {ruta}")
        if real.is_file() and real.suffix.lower() not in _EXTENSIONES:
            raise ValueError(
                f"Extensión no permitida: {real.suffix or '(sin extensión)'}"
            )
        return real

    def _listar(self, ruta: str) -> ToolResult:
        base = self._resolver(ruta or ".")
        if base.is_file():
            return ToolResult.exito(str(base.relative_to(self.root)))
        archivos: list[str] = []
        for actual in sorted(base.rglob("*")):
            if any(parte in _IGNORAR for parte in actual.parts):
                continue
            if actual.is_file() and actual.suffix.lower() in _EXTENSIONES:
                archivos.append(str(actual.relative_to(self.root)).replace("\\", "/"))
            if len(archivos) >= 500:
                break
        return ToolResult.exito(
            "\n".join(archivos), files=archivos, truncated=len(archivos) >= 500
        )

    def _leer(self, ruta: str, start_line: int | None, end_line: int | None) -> ToolResult:
        path = self._resolver(ruta)
        if not path.is_file():
            raise ValueError(f"No es un archivo: {ruta}")
        if path.stat().st_size > self.max_bytes:
            raise ValueError(
                f"El archivo supera el límite de {self.max_bytes} bytes."
            )
        texto = path.read_text(encoding="utf-8", errors="replace")
        lineas = texto.splitlines()
        inicio = max(1, int(start_line or 1))
        fin = min(len(lineas), int(end_line or len(lineas)))
        if fin < inicio:
            raise ValueError("end_line no puede ser menor que start_line.")
        seleccion = lineas[inicio - 1 : fin]
        numerado = "\n".join(
            f"{numero:>4}: {linea}"
            for numero, linea in enumerate(seleccion, start=inicio)
        )
        return ToolResult.exito(
            numerado,
            path=str(path.relative_to(self.root)).replace("\\", "/"),
            start_line=inicio,
            end_line=fin,
            total_lines=len(lineas),
        )

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        accion = str(arguments.get("action", "read")).strip().lower()
        ruta = str(arguments.get("path", "")).strip()
        try:
            if accion == "list":
                return self._listar(ruta or ".")
            if accion == "read":
                if not ruta:
                    return ToolResult.fallo("Falta 'path' para leer.")
                return self._leer(
                    ruta,
                    arguments.get("start_line"),
                    arguments.get("end_line"),
                )
            return ToolResult.fallo(f"Acción desconocida: {accion!r}")
        except (OSError, ValueError) as exc:
            return ToolResult.fallo(str(exc))


def create_tool(context: ToolContext) -> ReadOwnCodeTool:
    return ReadOwnCodeTool(context)
