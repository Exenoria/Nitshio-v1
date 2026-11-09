"""Captura de pantalla y análisis opcional con el modelo de visión."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import requests

from tools.base import Tool, ToolContext, ToolResult


class ScreenTool(Tool):
    name = "screen"
    description = (
        "Captura la pantalla y, si se pide, la describe con el modelo de "
        "visión local. Útil para revisar errores, código o ventanas."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["capture", "analyze"],
                "description": "capture solo guarda; analyze describe la imagen.",
            },
            "monitor": {
                "type": "integer",
                "description": "Índice del monitor (1 = principal).",
            },
            "prompt": {
                "type": "string",
                "description": "Pregunta opcional para el modelo de visión.",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        tools_cfg = context.config.get("tools") or {}
        screen_cfg = tools_cfg.get("screen") or {}
        self.monitor = int(screen_cfg.get("monitor", 1))
        self.max_width = int(screen_cfg.get("max_width", 1280))
        self.save_dir = context.ruta(
            screen_cfg.get("save_dir", "data/cache/screens")
        )
        self.vision_prompt = str(
            screen_cfg.get(
                "vision_prompt",
                "Describe lo que se ve en esta captura de pantalla en español.",
            )
        )
        ollama_cfg = context.config.get("ollama") or {}
        self.host = str(ollama_cfg.get("host", "http://localhost:11434")).rstrip("/")
        self.vision_model = str(
            (ollama_cfg.get("models") or {}).get("vision", "")
        )
        self.timeout = float(ollama_cfg.get("request_timeout_seconds", 180))

    def _capturar(self, monitor: int) -> Path:
        try:
            import mss
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mss/Pillow no están instalados. Instala requirements.txt."
            ) from exc
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with mss.mss() as sct:
            monitores = sct.monitors
            indice = monitor if 0 <= monitor < len(monitores) else 1
            captura = sct.grab(monitores[indice])
            imagen = Image.frombytes("RGB", captura.size, captura.rgb)
        if imagen.width > self.max_width:
            alto = int(imagen.height * self.max_width / imagen.width)
            imagen = imagen.resize((self.max_width, alto))
        ruta = self.save_dir / "screen.png"
        imagen.save(ruta, format="PNG")
        return ruta

    def _analizar(self, ruta: Path, prompt: str) -> str:
        if not self.vision_model:
            raise RuntimeError("No hay modelo de visión configurado.")
        datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
        try:
            respuesta = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": prompt,
                    "images": [datos],
                    "stream": False,
                },
                timeout=self.timeout,
            )
            respuesta.raise_for_status()
            return str(respuesta.json().get("response", "")).strip()
        except requests.RequestException as exc:
            raise RuntimeError(f"El modelo de visión falló: {exc}") from exc

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        accion = str(arguments.get("action", "analyze")).strip().lower()
        monitor = int(arguments.get("monitor", self.monitor))
        try:
            ruta = self._capturar(monitor)
        except Exception as exc:
            return ToolResult.fallo(f"No pude capturar la pantalla: {exc}")
        if accion == "capture":
            return ToolResult.exito(
                f"Captura guardada en {ruta}", path=str(ruta)
            )
        prompt = str(arguments.get("prompt", "")).strip() or self.vision_prompt
        try:
            descripcion = self._analizar(ruta, prompt)
        except Exception as exc:
            return ToolResult.fallo(
                f"Capturé la pantalla en {ruta}, pero no pude analizarla: {exc}",
                path=str(ruta),
            )
        return ToolResult.exito(
            descripcion or "(el modelo no devolvió descripción)",
            path=str(ruta),
        )


def create_tool(context: ToolContext) -> ScreenTool:
    return ScreenTool(context)
