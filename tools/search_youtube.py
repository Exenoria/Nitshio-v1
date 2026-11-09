"""Búsqueda de videos en YouTube con yt-dlp."""

from __future__ import annotations

import webbrowser
from typing import Any
from urllib.parse import quote_plus

from tools.base import Tool, ToolContext, ToolResult


class SearchYoutubeTool(Tool):
    name = "search_youtube"
    description = (
        "Busca videos en YouTube y devuelve título, URL, canal y duración."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Consulta de búsqueda.",
            },
            "max_results": {
                "type": "integer",
                "description": "Número máximo de resultados (1-10).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, context: ToolContext) -> None:
        tools_cfg = context.config.get("tools") or {}
        self.max_results = max(
            1, min(10, int((tools_cfg.get("search") or {}).get("max_results", 5)))
        )

    def _buscar_ytdlp(self, consulta: str, limite: int) -> list[dict[str, Any]]:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("yt-dlp no está instalado.") from exc
        opciones = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }
        with YoutubeDL(opciones) as ydl:
            info = ydl.extract_info(
                f"ytsearch{limite}:{consulta}", download=False
            )
        entradas = info.get("entries") or []
        resultados: list[dict[str, Any]] = []
        for entrada in entradas:
            if not entrada:
                continue
            resultados.append(
                {
                    "title": str(entrada.get("title", "")),
                    "url": str(
                        entrada.get("webpage_url")
                        or entrada.get("url")
                        or ""
                    ),
                    "channel": str(
                        entrada.get("channel")
                        or entrada.get("uploader")
                        or ""
                    ),
                    "duration_s": entrada.get("duration"),
                }
            )
        return resultados

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        consulta = str(arguments.get("query", "")).strip()
        if not consulta:
            return ToolResult.fallo("La búsqueda está vacía.")
        limite = max(
            1, min(10, int(arguments.get("max_results", self.max_results)))
        )
        try:
            resultados = self._buscar_ytdlp(consulta, limite)
        except Exception as exc:
            url = "https://www.youtube.com/results?search_query=" + quote_plus(
                consulta
            )
            webbrowser.open(url)
            return ToolResult.fallo(
                f"No pude buscar en YouTube ({exc}); abrí la búsqueda en el "
                f"navegador: {url}",
                url=url,
                opened=True,
            )
        if not resultados:
            url = "https://www.youtube.com/results?search_query=" + quote_plus(
                consulta
            )
            webbrowser.open(url)
            return ToolResult.exito(
                f"No encontré resultados; abrí YouTube: {url}",
                results=[],
                url=url,
                opened=True,
            )
        lineas = []
        for indice, resultado in enumerate(resultados, start=1):
            duracion = resultado.get("duration_s")
            extra = f" ({int(duracion)} s)" if isinstance(duracion, (int, float)) else ""
            lineas.append(
                f"{indice}. {resultado['title']}{extra}\n"
                f"   {resultado['url']}\n"
                f"   Canal: {resultado['channel']}"
            )
        return ToolResult.exito(
            "\n".join(lineas), results=resultados, query=consulta
        )


def create_tool(context: ToolContext) -> SearchYoutubeTool:
    return SearchYoutubeTool(context)
