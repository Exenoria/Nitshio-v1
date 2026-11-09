"""Tests del registro de herramientas y validación de argumentos."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from tools.base import Tool, ToolContext, ToolResult, validar_argumentos
from tools.registry import ToolRegistry


class ToolFalsa(Tool):
    name = "falsa"
    description = "Herramienta de prueba"
    parameters = {
        "type": "object",
        "properties": {
            "texto": {"type": "string"},
            "numero": {"type": "integer"},
            "modo": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["texto"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.llamadas: list[dict[str, Any]] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.llamadas.append(arguments)
        return ToolResult.exito(f"ok: {arguments.get('texto')}")


def _contexto(tmp_path: Path) -> ToolContext:
    return ToolContext(root=tmp_path, config={"tools": {"enabled": ["falsa"]}})


def test_validar_argumentos_ok() -> None:
    schema = ToolFalsa.parameters
    assert validar_argumentos(schema, {"texto": "hola", "numero": 2}) == []


def test_validar_argumentos_faltante() -> None:
    problemas = validar_argumentos(ToolFalsa.parameters, {})
    assert any("obligatorio" in p for p in problemas)


def test_validar_argumentos_tipo() -> None:
    problemas = validar_argumentos(
        ToolFalsa.parameters, {"texto": 2, "numero": "x"}
    )
    assert len(problemas) >= 2


def test_validar_argumentos_extra() -> None:
    problemas = validar_argumentos(
        ToolFalsa.parameters, {"texto": "hola", "extra": 1}
    )
    assert any("no permitido" in p for p in problemas)


def test_validar_argumentos_enum() -> None:
    problemas = validar_argumentos(
        ToolFalsa.parameters, {"texto": "hola", "modo": "c"}
    )
    assert any("uno de" in p for p in problemas)


def test_registro_ejecuta(tmp_path: Path) -> None:
    tool = ToolFalsa()
    registro = ToolRegistry(_contexto(tmp_path))
    registro.register(tool)
    assert registro.names() == ["falsa"]
    assert registro.catalog()[0]["name"] == "falsa"
    resultado = registro.execute("falsa", {"texto": "hola"})
    assert resultado.ok
    assert tool.llamadas == [{"texto": "hola"}]


def test_registro_desconocida(tmp_path: Path) -> None:
    registro = ToolRegistry(_contexto(tmp_path))
    resultado = registro.execute("nope", {})
    assert not resultado.ok
    assert "desconocida" in (resultado.error or "")


def test_registro_argumentos_invalidos(tmp_path: Path) -> None:
    tool = ToolFalsa()
    registro = ToolRegistry(_contexto(tmp_path))
    registro.register(tool)
    resultado = registro.execute("falsa", {"texto": 3})
    assert not resultado.ok
    assert "Argumentos inválidos" in (resultado.error or "")


def test_registro_captura_excepcion(tmp_path: Path) -> None:
    class ToolRota(ToolFalsa):
        name = "rota"

        def execute(self, arguments, context):
            raise RuntimeError("boom")

    registro = ToolRegistry(_contexto(tmp_path), enabled=["rota"])
    registro.register(ToolRota())
    resultado = registro.execute("rota", {"texto": "hola"})
    assert not resultado.ok
    assert "boom" in (resultado.error or "")


def test_registro_ignora_deshabilitadas(tmp_path: Path) -> None:
    tool = ToolFalsa()
    registro = ToolRegistry(_contexto(tmp_path), enabled=[])
    registro.register(tool)
    # register añade a enabled; para probar el filtro se quita después.
    registro.enabled.remove("falsa")
    assert registro.get("falsa") is None
    assert registro.catalog() == []


def test_discover_con_modulo_falso(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modulo = types.SimpleNamespace(create_tool=lambda context: ToolFalsa())
    monkeypatch.setattr(
        "tools.registry.importlib.import_module", lambda nombre: modulo
    )
    registro = ToolRegistry(_contexto(tmp_path)).discover()
    assert registro.names() == ["falsa"]
    assert registro.errors == {}


def test_discover_reporta_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def falla(nombre: str):
        raise ImportError("no existe")

    monkeypatch.setattr("tools.registry.importlib.import_module", falla)
    registro = ToolRegistry(_contexto(tmp_path)).discover()
    assert "falsa" in registro.errors
    assert registro.names() == []
