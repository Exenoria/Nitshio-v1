"""Reproduce una canción de YouTube descargando solo el audio.

La descarga se hace con yt-dlp en ``data/cache/audio`` y la reproducción con
``ffplay``. Se respeta ``max_duration_seconds`` para no descargar horas de
audio por accidente.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolContext, ToolResult


class PlaySongTool(Tool):
    name = "play_song"
    description = (
        "Busca una canción en YouTube, descarga el audio y la reproduce. "
        "Indica el nombre de la canción o artista."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Canción, artista o búsqueda.",
            },
            "max_duration_seconds": {
                "type": "integer",
                "description": "Duración máxima permitida en segundos.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, context: ToolContext) -> None:
        tools_cfg = context.config.get("tools") or {}
        media = tools_cfg.get("media") or {}
        self.cache_dir = context.ruta(
            media.get("cache_dir", "data/cache/audio")
        )
        self.player = str(media.get("player", "ffplay"))
        self.audio_format = str(media.get("audio_format", "bestaudio"))
        self.max_duration = int(media.get("max_duration_seconds", 600))

    def _buscar(self, consulta: str) -> dict[str, Any] | None:
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
                f"ytsearch1:{consulta}", download=False
            )
        entradas = info.get("entries") or []
        if not entradas:
            return None
        entrada = entradas[0]
        return {
            "title": str(entrada.get("title", consulta)),
            "url": str(entrada.get("webpage_url") or entrada.get("url") or ""),
            "duration_s": entrada.get("duration"),
            "channel": str(entrada.get("channel") or entrada.get("uploader") or ""),
        }

    def _descargar(self, entrada: dict[str, Any]) -> Path:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("yt-dlp no está instalado.") from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        opciones = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": self.audio_format,
            "outtmpl": str(self.cache_dir / "%(id)s.%(ext)s"),
        }
        with YoutubeDL(opciones) as ydl:
            info = ydl.extract_info(entrada["url"], download=True)
            ruta = Path(ydl.prepare_filename(info))
        if not ruta.exists():
            # yt-dlp puede cambiar la extensión tras postprocesar.
            candidatos = sorted(self.cache_dir.glob(f"{info.get('id', '*')}.*"))
            if not candidatos:
                raise RuntimeError("yt-dlp no generó ningún archivo de audio.")
            ruta = candidatos[0]
        return ruta

    def _reproducir(self, ruta: Path) -> None:
        ejecutable = shutil.which(self.player)
        if not ejecutable:
            raise RuntimeError(
                f"No se encontró el reproductor {self.player!r} en PATH."
            )
        subprocess.run(
            [
                ejecutable,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                str(ruta),
            ],
            check=False,
        )

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        consulta = str(arguments.get("query", "")).strip()
        if not consulta:
            return ToolResult.fallo("La búsqueda está vacía.")
        limite = int(
            arguments.get("max_duration_seconds", self.max_duration)
        )
        try:
            entrada = self._buscar(consulta)
        except Exception as exc:
            return ToolResult.fallo(f"No pude buscar la canción: {exc}")
        if not entrada or not entrada.get("url"):
            return ToolResult.fallo(f"No encontré resultados para {consulta!r}.")
        duracion = entrada.get("duration_s")
        if isinstance(duracion, (int, float)) and limite > 0 and duracion > limite:
            return ToolResult.fallo(
                f"La canción dura {int(duracion)} s y el límite es {limite} s.",
                title=entrada.get("title"),
                duration_s=duracion,
            )
        try:
            ruta = self._descargar(entrada)
            self._reproducir(ruta)
        except Exception as exc:
            return ToolResult.fallo(f"No pude reproducir la canción: {exc}")
        return ToolResult.exito(
            f"Reproduciendo: {entrada['title']}",
            title=entrada["title"],
            url=entrada["url"],
            path=str(ruta),
            duration_s=duracion,
        )


def create_tool(context: ToolContext) -> PlaySongTool:
    return PlaySongTool(context)
