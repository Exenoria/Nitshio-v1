"""Tests del router de herramientas con un cliente falso."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brain.router.tool_router import (
    RouterError,
    ToolRouter,
    cargar_prompt_router,
    crear_router,
    router_schema,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "config" / "prompts" / "router.md"
CATALOGO = [
    {
        "name": "search_google",
        "description": "Busca en la web",
        "parameters": {"type": "object", "properties": {}},
    }
]


class ClienteFalso:
    def __init__(self, contenido: str) -> None:
        self.contenido = contenido
        self.llamadas: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any):
        self.llamadas.append(
            {"model": model, "messages": messages, **kwargs}
        )
        return {"message": {"content": self.contenido}}

    def _contenido(self, data: dict[str, Any]) -> str:
        return str(data["message"]["content"])


def _router(contenido: str, **kwargs: Any) -> tuple[ToolRouter, ClienteFalso]:
    cliente = ClienteFalso(contenido)
    router = ToolRouter(
        cliente,
        "router-model",
        PROMPT,
        confidence_threshold=0.55,
        **kwargs,
    )
    return router, cliente


def _decision(**cambios: Any) -> str:
    data = {
        "tool": "search_google",
        "arguments": {"query": "clima"},
        "confidence": 0.9,
        "reason": "El usuario pide información actual.",
    }
    data.update(cambios)
    return json.dumps(data, ensure_ascii=False)


def test_schema_router() -> None:
    schema = router_schema()
    assert schema["required"] == ["tool", "arguments", "confidence", "reason"]


def test_prompt_inyecta_catalogo() -> None:
    prompt = cargar_prompt_router(PROMPT, CATALOGO)
    assert "search_google" in prompt
    assert "__TOOLS__" not in prompt


def test_decision_valida() -> None:
    router, cliente = _router(_decision())
    llamada = router.route("¿cómo está el clima?", CATALOGO)
    assert llamada.tool == "search_google"
    assert llamada.arguments == {"query": "clima"}
    assert llamada.confidence == 0.9
    assert llamada.valid
    assert cliente.llamadas[0]["schema"] == router_schema()


def test_confianza_baja_usa_none() -> None:
    router, _ = _router(_decision(confidence=0.2))
    llamada = router.route("hola", CATALOGO)
    assert llamada.tool == "none"
    assert "umbral" in llamada.reason


def test_herramienta_desconocida() -> None:
    router, _ = _router(_decision(tool="inventada"))
    llamada = router.route("hola", CATALOGO)
    assert llamada.tool == "none"
    assert not llamada.valid
    assert "desconocida" in llamada.reason


def test_none_explicito() -> None:
    router, _ = _router(_decision(tool="none", arguments={}, confidence=0.99))
    llamada = router.route("hola", CATALOGO)
    assert llamada.tool == "none"
    assert llamada.valid


def test_json_invalido_usa_none() -> None:
    router, _ = _router("esto no es json")
    llamada = router.route("hola", CATALOGO)
    assert llamada.tool == "none"
    assert not llamada.valid


def test_json_reparable() -> None:
    router, _ = _router(
        '{"tool": "search_google", "arguments": {"query": "x"}, '
        '"confidence": 0.8, "reason": "ok",}'
    )
    llamada = router.route("busca x", CATALOGO)
    assert llamada.tool == "search_google"
    assert llamada.valid


def test_argumentos_no_objeto_se_normalizan() -> None:
    router, _ = _router(_decision(arguments="malo"))
    llamada = router.route("hola", CATALOGO)
    assert llamada.arguments == {}


def test_historial_se_envia() -> None:
    router, cliente = _router(_decision(tool="none"))
    router.route(
        "hola",
        CATALOGO,
        history=[{"role": "user", "content": "antes"}],
    )
    mensajes = cliente.llamadas[0]["messages"]
    assert any(m["content"] == "antes" for m in mensajes)


def test_prompt_inexistente() -> None:
    cliente = ClienteFalso(_decision())
    router = ToolRouter(cliente, "m", Path("no-existe.md"))
    with pytest.raises(RouterError):
        router.route("hola", CATALOGO)


def test_crear_router_desde_config() -> None:
    config = {
        "ollama": {
            "models": {"router": "qwen3.5:0.8b"},
            "options": {"router": {"temperature": 0}},
        },
        "router": {
            "prompt_file": "config/prompts/router.md",
            "confidence_threshold": 0.6,
        },
    }
    cliente = ClienteFalso(_decision())
    router = crear_router(config, cliente, root=ROOT)
    assert router.model == "qwen3.5:0.8b"
    assert router.confidence_threshold == 0.6
