"""Herramienta para abrir YouTube (búsqueda o video) en el navegador."""

from __future__ import annotations

import webbrowser
from typing import Any
from urllib.parse import quote_plus, urlparse

from tools.base import Tool, ToolContext, ToolResult

_DOMINIOS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


class OpenYoutubeTool(Tool):
    name = "open_youtube"
    description = (
        "Abre YouTube en el navegador. Acepta una búsqueda o una URL de "
        "YouTube válida."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Búsqueda para YouTube.",
            },
            "url": {
                "type": "string",
                "description": "URL de YouTube a abrir.",
            },
        },
        "additionalProperties": False,
    }

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        url = str(arguments.get("url", "")).strip()
        consulta = str(arguments.get("query", "")).strip()
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or parsed.netloc not in _DOMINIOS:
                return ToolResult.fallo(
                    "Solo se permiten URLs http(s) de YouTube."
                )
        elif consulta:
            url = "https://www.youtube.com/results?search_query=" + quote_plus(
                consulta
            )
        else:
            return ToolResult.fallo("Indica 'query' o 'url'.")
        abierto = webbrowser.open(url)
        return ToolResult.exito(
            f"Abrí YouTube: {consulta or url}", url=url, opened=bool(abierto)
        )


def create_tool(context: ToolContext) -> OpenYoutubeTool:
    del context
    return OpenYoutubeTool()
