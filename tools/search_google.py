"""Búsqueda web con Google Custom Search o DuckDuckGo.

Sin claves de Google se usa DuckDuckGo (``ddgs``). Si todo falla, se abre
Google en el navegador y se devuelve un aviso; nunca se inventan resultados.
"""

from __future__ import annotations

import os
import webbrowser
from typing import Any
from urllib.parse import quote_plus

import requests

from tools.base import Tool, ToolContext, ToolResult


class SearchGoogleTool(Tool):
    name = "search_google"
    description = (
        "Busca información actual en la web y devuelve títulos, URLs y "
        "fragmentos. Usa Google CSE si hay claves; si no, DuckDuckGo."
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
        search_cfg = tools_cfg.get("search") or {}
        self.max_results = max(1, min(10, int(search_cfg.get("max_results", 5))))
        self.google_key_env = str(
            search_cfg.get("google_cse_key_env", "GOOGLE_CSE_KEY")
        )
        self.google_cx_env = str(
            search_cfg.get("google_cse_cx_env", "GOOGLE_CSE_CX")
        )
        self.timeout = float(search_cfg.get("timeout_seconds", 20))

    def _google_cse(self, consulta: str, limite: int) -> list[dict[str, str]] | None:
        clave = os.getenv(self.google_key_env, "").strip()
        cx = os.getenv(self.google_cx_env, "").strip()
        if not clave or not cx:
            return None
        try:
            respuesta = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": clave,
                    "cx": cx,
                    "q": consulta,
                    "num": limite,
                    "hl": "es",
                },
                timeout=self.timeout,
            )
            respuesta.raise_for_status()
            items = respuesta.json().get("items") or []
        except requests.RequestException:
            return None
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("link", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in items
        ]

    def _ddg(self, consulta: str, limite: int) -> list[dict[str, str]]:
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ddgs no está instalado.") from exc
        with DDGS() as ddgs:
            crudos = list(ddgs.text(consulta, max_results=limite))
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("href", item.get("url", ""))),
                "snippet": str(item.get("body", item.get("snippet", ""))),
            }
            for item in crudos
        ]

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        consulta = str(arguments.get("query", "")).strip()
        if not consulta:
            return ToolResult.fallo("La búsqueda está vacía.")
        limite = max(
            1, min(10, int(arguments.get("max_results", self.max_results)))
        )
        resultados = self._google_cse(consulta, limite)
        if resultados is None:
            try:
                resultados = self._ddg(consulta, limite)
            except Exception:
                resultados = []
        if not resultados:
            url = "https://www.google.com/search?q=" + quote_plus(consulta)
            webbrowser.open(url)
            return ToolResult.exito(
                "No pude obtener resultados; abrí Google en el navegador "
                f"para que los revises: {url}",
                results=[],
                url=url,
                opened=True,
            )
        lineas = [
            f"{indice}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for indice, r in enumerate(resultados, start=1)
        ]
        return ToolResult.exito(
            "\n".join(lineas),
            results=resultados,
            query=consulta,
        )


def create_tool(context: ToolContext) -> SearchGoogleTool:
    return SearchGoogleTool(context)
