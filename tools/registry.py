"""Registro y carga dinámica de herramientas habilitadas."""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Iterable

from tools.base import Tool, ToolContext, ToolResult, validar_argumentos


class ToolRegistry:
    """Carga plugins de ``tools/`` y ejecuta el que pida el router."""

    def __init__(
        self,
        context: ToolContext,
        *,
        enabled: Iterable[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.context = context
        self.logger = logger or context.logger or logging.getLogger("nitshio.tools")
        self.enabled = list(
            enabled
            if enabled is not None
            else ((context.config.get("tools") or {}).get("enabled") or [])
        )
        self._tools: dict[str, Tool] = {}
        self.errors: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    def register(self, tool: Tool) -> None:
        """Registra una instancia ya creada (útil en tests y extensiones)."""
        if not getattr(tool, "name", ""):
            raise ValueError("La herramienta no tiene nombre.")
        self._tools[tool.name] = tool
        if tool.name not in self.enabled:
            self.enabled.append(tool.name)

    def discover(self) -> "ToolRegistry":
        """Importa los módulos listados en ``tools.enabled``."""
        for nombre in list(self.enabled):
            if nombre in self._tools:
                continue
            try:
                modulo = importlib.import_module(f"tools.{nombre}")
            except Exception as exc:
                self.errors[nombre] = f"No se pudo importar tools.{nombre}: {exc}"
                self.logger.warning(self.errors[nombre])
                continue
            try:
                if hasattr(modulo, "create_tool"):
                    tool = modulo.create_tool(self.context)
                elif hasattr(modulo, "TOOL"):
                    tool = modulo.TOOL
                else:
                    raise ValueError(
                        "el módulo no expone create_tool(context) ni TOOL"
                    )
            except Exception as exc:
                self.errors[nombre] = f"Error creando {nombre}: {exc}"
                self.logger.warning(self.errors[nombre])
                continue
            self._tools[getattr(tool, "name", nombre)] = tool
        return self

    # ------------------------------------------------------------------
    # Catálogo y ejecución
    # ------------------------------------------------------------------
    def catalog(self) -> list[dict[str, Any]]:
        """Lista de esquemas para el prompt del router."""
        return [
            tool.schema()
            for nombre, tool in sorted(self._tools.items())
            if nombre in self.enabled
        ]

    def catalog_json(self) -> str:
        return json.dumps(self.catalog(), ensure_ascii=False, indent=2)

    def names(self) -> list[str]:
        return [n for n in sorted(self._tools) if n in self.enabled]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name) if name in self.enabled else None

    def execute(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Valida y ejecuta una herramienta, sin propagar excepciones."""
        tool = self.get(name)
        if tool is None:
            return ToolResult.fallo(
                f"Herramienta desconocida o deshabilitada: {name!r}",
                disponibles=self.names(),
            )
        argumentos = arguments if isinstance(arguments, dict) else {}
        problemas = validar_argumentos(tool.parameters, argumentos)
        if problemas:
            return ToolResult.fallo(
                "Argumentos inválidos: " + "; ".join(problemas)
            )
        try:
            resultado = tool.execute(argumentos, self.context)
        except Exception as exc:
            self.logger.exception("Herramienta %s falló", name)
            return ToolResult.fallo(f"La herramienta {name} falló: {exc}")
        if not isinstance(resultado, ToolResult):
            return ToolResult.fallo(
                f"La herramienta {name} devolvió un tipo inválido."
            )
        return resultado
