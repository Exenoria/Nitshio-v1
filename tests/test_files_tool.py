"""Tests de seguridad y operaciones de la herramienta de archivos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.base import ToolContext
from tools.files import FilesTool


def _config(tmp_path: Path, **security: Any) -> dict[str, Any]:
    base = {
        "confirm_delete": True,
        "use_recycle_bin": True,
        "delete_log": "logs/deletions.jsonl",
        "max_read_bytes": 1000,
        "max_write_bytes": 1000,
        "allow_symlinks": False,
    }
    base.update(security)
    return {
        "tools": {
            "allowed_paths": ["workspace"],
            "security": base,
        }
    }


def _contexto(tmp_path: Path, **security: Any) -> ToolContext:
    return ToolContext(root=tmp_path, config=_config(tmp_path, **security))


def _tool(tmp_path: Path, **security: Any) -> FilesTool:
    return FilesTool(_contexto(tmp_path, **security))


def _ejecutar(tool: FilesTool, **args: Any):
    return tool.execute(args, tool.context)


def test_escribir_leer_listar(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    resultado = _ejecutar(tool, action="write", path="workspace/nota.txt", content="hola")
    assert resultado.ok
    assert (tmp_path / "workspace" / "nota.txt").read_text(encoding="utf-8") == "hola"

    lectura = _ejecutar(tool, action="read", path="workspace/nota.txt")
    assert lectura.ok
    assert lectura.content == "hola"

    listado = _ejecutar(tool, action="list", path="workspace")
    assert listado.ok
    assert any(entrada["name"] == "nota.txt" for entrada in listado.data["entries"])


def test_ruta_fuera_de_allowed_paths(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    resultado = _ejecutar(tool, action="write", path="fuera.txt", content="x")
    assert not resultado.ok
    assert "allowed_paths" in (resultado.error or "")


def test_escape_con_puntos(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    resultado = _ejecutar(
        tool, action="read", path="workspace/../fuera.txt"
    )
    assert not resultado.ok


def test_symlink_rechazado(tmp_path: Path) -> None:
    fuera = tmp_path / "fuera.txt"
    fuera.write_text("secreto", encoding="utf-8")
    enlace = tmp_path / "workspace" / "enlace.txt"
    enlace.parent.mkdir(parents=True, exist_ok=True)
    try:
        enlace.symlink_to(fuera)
    except (OSError, NotImplementedError):
        pytest.skip("Windows no permite crear symlinks sin privilegios.")
    tool = _tool(tmp_path)
    resultado = _ejecutar(tool, action="read", path="workspace/enlace.txt")
    assert not resultado.ok
    assert "symlink" in (resultado.error or "").lower()


def test_lectura_demasiado_grande(tmp_path: Path) -> None:
    tool = _tool(tmp_path, max_read_bytes=10)
    archivo = tmp_path / "workspace" / "grande.txt"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("x" * 20, encoding="utf-8")
    resultado = _ejecutar(tool, action="read", path="workspace/grande.txt")
    assert not resultado.ok
    assert "límite" in (resultado.error or "")


def test_escritura_demasiado_grande(tmp_path: Path) -> None:
    tool = _tool(tmp_path, max_write_bytes=5)
    resultado = _ejecutar(
        tool, action="write", path="workspace/x.txt", content="123456"
    )
    assert not resultado.ok


def test_archivo_binario_no_se_lee(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    archivo = tmp_path / "workspace" / "binario.dat"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_bytes(b"\x00\x01\x02")
    resultado = _ejecutar(tool, action="read", path="workspace/binario.dat")
    assert not resultado.ok
    assert "binario" in (resultado.error or "")


def test_delete_sin_confirmacion_falla(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    archivo = tmp_path / "workspace" / "borrar.txt"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("x", encoding="utf-8")
    resultado = _ejecutar(tool, action="delete", path="workspace/borrar.txt")
    assert not resultado.ok
    assert "confirmación" in (resultado.error or "")
    assert archivo.exists()


def test_delete_confirmacion_rechazada(tmp_path: Path) -> None:
    contexto = _contexto(tmp_path)
    contexto.confirm = lambda _mensaje: False
    tool = FilesTool(contexto)
    archivo = tmp_path / "workspace" / "borrar.txt"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("x", encoding="utf-8")
    resultado = tool.execute(
        {"action": "delete", "path": "workspace/borrar.txt"}, contexto
    )
    assert not resultado.ok
    assert "cancelado" in (resultado.error or "").lower()
    assert archivo.exists()
    log = (tmp_path / "logs" / "deletions.jsonl").read_text(encoding="utf-8")
    assert "cancelado" in log


def test_delete_confirmado_usa_papelera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contexto = _contexto(tmp_path)
    contexto.confirm = lambda _mensaje: True
    tool = FilesTool(contexto)
    archivo = tmp_path / "workspace" / "borrar.txt"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("x", encoding="utf-8")
    enviados: list[Path] = []
    monkeypatch.setattr(tool, "_enviar_a_papelera", lambda path: enviados.append(path))
    resultado = tool.execute(
        {"action": "delete", "path": "workspace/borrar.txt"}, contexto
    )
    assert resultado.ok
    assert enviados == [archivo]
    assert resultado.data["mode"] == "recycle_bin"
    log = (tmp_path / "logs" / "deletions.jsonl").read_text(encoding="utf-8")
    assert json.loads(log.strip())["status"] == "ok"


def test_delete_permanente_explicito(tmp_path: Path) -> None:
    contexto = _contexto(tmp_path, use_recycle_bin=False, confirm_delete=False)
    tool = FilesTool(contexto)
    archivo = tmp_path / "workspace" / "borrar.txt"
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("x", encoding="utf-8")
    resultado = tool.execute(
        {"action": "delete", "path": "workspace/borrar.txt"}, contexto
    )
    assert resultado.ok
    assert not archivo.exists()


def test_delete_carpeta_sin_recursive_falla(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    carpeta = tmp_path / "workspace" / "carpeta"
    carpeta.mkdir(parents=True)
    resultado = _ejecutar(tool, action="delete", path="workspace/carpeta")
    assert not resultado.ok
    assert "recursive" in (resultado.error or "")


def test_mover(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    _ejecutar(tool, action="write", path="workspace/a.txt", content="x")
    resultado = _ejecutar(
        tool,
        action="move",
        path="workspace/a.txt",
        destination="workspace/b.txt",
    )
    assert resultado.ok
    assert not (tmp_path / "workspace" / "a.txt").exists()
    assert (tmp_path / "workspace" / "b.txt").exists()


def test_exists_y_mkdir(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    assert _ejecutar(tool, action="mkdir", path="workspace/sub").ok
    assert _ejecutar(tool, action="exists", path="workspace/sub").data["exists"] is True
    assert _ejecutar(tool, action="exists", path="workspace/no").data["exists"] is False
