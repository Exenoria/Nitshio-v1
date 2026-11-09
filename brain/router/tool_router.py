"""Router de herramientas con qwen3.5:0.8b (configurable).

El router no ejecuta nada: solo devuelve una decisión validada. El orquestador
decide si la ejecuta y cómo inyecta el resultado en el prompt de personalidad.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from brain.llm.ollama_client import OllamaClient, OllamaError

ROOT = Path(__file__).resolve().parents[2]


class RouterError(RuntimeError):
    """El router no pudo obtener una decisión válida."""


@dataclass(frozen=True)
class ToolCall:
    """Decisión del router."""

    tool: str = "none"
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""
    valid: bool = True

    @property
    def usa_herramienta(self) -> bool:
        return self.tool not in {"", "none", "null"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "confidence": self.confidence,
            "reason": self.reason,
            "valid": self.valid,
        }


def router_schema() -> dict[str, Any]:
    """JSON Schema que se envía a Ollama para la decisión del router."""
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["tool", "arguments", "confidence", "reason"],
        "additionalProperties": False,
    }


def cargar_prompt_router(path: Path, catalogo: Iterable[dict[str, Any]]) -> str:
    """Carga el prompt del router e inyecta el catálogo de herramientas."""
    if not path.exists():
        raise RouterError(f"No existe el prompt del router: {path}")
    plantilla = path.read_text(encoding="utf-8")
    herramientas = json.dumps(list(catalogo), ensure_ascii=False, indent=2)
    return plantilla.replace("__TOOLS__", herramientas)


def _parsear_decision(texto: str) -> dict[str, Any]:
    """Parsea el JSON del router, con reparación si hace falta."""
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            reparado = repair_json(texto, return_objects=True)
            datos = reparado if isinstance(reparado, dict) else json.loads(str(reparado))
        except Exception as exc:
            raise RouterError(f"JSON del router inválido: {exc}") from exc
    if not isinstance(datos, dict):
        raise RouterError("La decisión del router no es un objeto JSON.")
    return datos


class ToolRouter:
    """Decide qué herramienta usar para un mensaje."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        prompt_path: Path,
        *,
        confidence_threshold: float = 0.55,
        options: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_path = prompt_path
        self.confidence_threshold = float(confidence_threshold)
        self.options = dict(options or {})
        self.timeout_seconds = float(timeout_seconds)

    def route(
        self,
        user_text: str,
        catalogo: Iterable[dict[str, Any]],
        *,
        history: Iterable[dict[str, str]] | None = None,
    ) -> ToolCall:
        """Devuelve una decisión; ante cualquier fallo usa ``tool=none``."""
        herramientas = list(catalogo)
        nombres = {str(h.get("name", "")) for h in herramientas}
        try:
            system = cargar_prompt_router(self.prompt_path, herramientas)
        except RouterError:
            raise
        mensajes: list[dict[str, str]] = [{"role": "system", "content": system}]
        for mensaje in history or []:
            if mensaje.get("role") in {"user", "assistant"} and mensaje.get("content"):
                mensajes.append(
                    {
                        "role": str(mensaje["role"]),
                        "content": str(mensaje["content"]),
                    }
                )
        mensajes.append({"role": "user", "content": str(user_text)})
        try:
            data = self.client.chat(
                self.model,
                mensajes,
                schema=router_schema(),
                options=self.options,
            )
        except OllamaError as exc:
            raise RouterError(f"El router no pudo llamar a Ollama: {exc}") from exc
        crudo = self.client._contenido(data)
        try:
            decision = _parsear_decision(crudo)
        except RouterError as exc:
            return ToolCall(
                tool="none",
                confidence=0.0,
                reason=f"Decisión inválida: {exc}",
                raw=crudo,
                valid=False,
            )

        tool = str(decision.get("tool", "none")).strip()
        argumentos = decision.get("arguments")
        if not isinstance(argumentos, dict):
            argumentos = {}
        try:
            confianza = float(decision.get("confidence", 0.0))
        except (TypeError, ValueError):
            confianza = 0.0
        confianza = max(0.0, min(1.0, confianza))
        razon = str(decision.get("reason", "")).strip()

        if tool not in nombres and tool not in {"none", "null", ""}:
            return ToolCall(
                tool="none",
                arguments={},
                confidence=confianza,
                reason=f"Herramienta desconocida: {tool!r}. {razon}".strip(),
                raw=crudo,
                valid=False,
            )
        if tool in {"null", ""}:
            tool = "none"
        if tool == "none":
            return ToolCall(
                tool="none",
                arguments={},
                confidence=confianza,
                reason=razon,
                raw=crudo,
            )
        if confianza < self.confidence_threshold:
            return ToolCall(
                tool="none",
                arguments={},
                confidence=confianza,
                reason=(
                    f"Confianza {confianza:.2f} por debajo del umbral "
                    f"{self.confidence_threshold:.2f}. {razon}"
                ).strip(),
                raw=crudo,
            )
        return ToolCall(
            tool=tool,
            arguments=argumentos,
            confidence=confianza,
            reason=razon,
            raw=crudo,
        )


def crear_router(
    config: dict[str, Any],
    client: OllamaClient,
    *,
    root: Path | None = None,
) -> ToolRouter:
    """Crea el router desde ``config.router`` y ``config.ollama``."""
    root = root or ROOT
    router_cfg = config.get("router") or {}
    ollama_cfg = config.get("ollama") or {}
    modelo = str((ollama_cfg.get("models") or {}).get("router", ""))
    if not modelo:
        raise RouterError("Falta ollama.models.router en la configuración.")
    prompt_rel = str(router_cfg.get("prompt_file", "config/prompts/router.md"))
    prompt_path = Path(prompt_rel)
    if not prompt_path.is_absolute():
        prompt_path = root / prompt_path
    opciones = dict(
        (ollama_cfg.get("options") or {}).get("router") or {}
    )
    return ToolRouter(
        client,
        modelo,
        prompt_path,
        confidence_threshold=float(
            router_cfg.get("confidence_threshold", 0.55)
        ),
        options=opciones,
        timeout_seconds=float(router_cfg.get("timeout_seconds", 60)),
    )
