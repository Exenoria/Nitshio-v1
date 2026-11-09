"""Herramienta para abrir una búsqueda de Google en el navegador."""

from __future__ import annotations

import webbrowser
from typing import Any
from urllib.parse import quote_plus

from tools.base import Tool, ToolContext, ToolResult


class OpenGoogleTool(Tool):
    name = "open_google"
    description = "Abre una búsqueda de Google en el navegador predeterminado."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Texto a buscar en Google.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        consulta = str(arguments.get("query", "")).strip()
        if not consulta:
            return ToolResult.fallo("La búsqueda está vacía.")
        url = "https://www.google.com/search?q=" + quote_plus(consulta)
        abierto = webbrowser.open(url)
        return ToolResult.exito(
            f"Abrí Google con la búsqueda: {consulta}",
            url=url,
            opened=bool(abierto),
        )


def create_tool(context: ToolContext) -> OpenGoogleTool:
    del context
    return OpenGoogleTool()
