"""Tests del registro de versiones, activación y rollback."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.versions import (
    VersionError,
    activate_version,
    actualizar_config,
    find_version,
    list_versions,
    load_registry,
    register_version,
    rollback_version,
    update_metrics,
)


def _config(tmp_path: Path) -> Path:
    ruta = tmp_path / "config.yaml"
    ruta.write_text(
        "# comentario que debe sobrevivir\n"
        "ollama:\n"
        "  models:\n"
        '    active: "viejo"\n'
        "  options:\n"
        "    primary:\n"
        "      num_ctx: 2048\n",
        encoding="utf-8",
    )
    return ruta


def test_register_activate_y_rollback(tmp_path: Path) -> None:
    versions = tmp_path / "versions.json"
    config = _config(tmp_path)
    persona = SimpleNamespace(system_prompt="system", few_shots=[1, 2])
    a = register_version(
        "nitshio-a",
        "base-a",
        persona=persona,
        modelfile="FROM base-a",
        registry_path=versions,
        config_path=config,
    )
    assert a["persona_sha256"]
    b = register_version(
        "nitshio-b",
        "base-b",
        modelfile="FROM base-b",
        registry_path=versions,
        config_path=config,
    )
    assert b["name"] == "nitshio-b"

    activate_version("nitshio-a", registry_path=versions, config_path=config)
    data = load_registry(versions)
    assert data["active"] == "nitshio-a"
    assert data["previous_active"] is None
    assert 'active: "nitshio-a"' in config.read_text(encoding="utf-8")
    assert "# comentario que debe sobrevivir" in config.read_text(encoding="utf-8")

    activate_version("nitshio-b", registry_path=versions, config_path=config)
    data = load_registry(versions)
    assert data["active"] == "nitshio-b"
    assert data["previous_active"] == "nitshio-a"

    entrada = rollback_version(registry_path=versions, config_path=config)
    assert entrada["name"] == "nitshio-a"
    data = load_registry(versions)
    assert data["active"] == "nitshio-a"
    assert data["previous_active"] == "nitshio-b"
    assert 'active: "nitshio-a"' in config.read_text(encoding="utf-8")


def test_activar_version_inexistente(tmp_path: Path) -> None:
    versions = tmp_path / "versions.json"
    config = _config(tmp_path)
    with pytest.raises(VersionError, match="no está registrada"):
        activate_version("nope", registry_path=versions, config_path=config)


def test_register_activate_directo(tmp_path: Path) -> None:
    versions = tmp_path / "versions.json"
    config = _config(tmp_path)
    register_version(
        "nitshio-a",
        "base-a",
        registry_path=versions,
        config_path=config,
        activate=True,
    )
    register_version(
        "nitshio-b",
        "base-b",
        registry_path=versions,
        config_path=config,
        activate=True,
    )
    data = load_registry(versions)
    assert data["active"] == "nitshio-b"
    assert data["previous_active"] == "nitshio-a"
    assert 'active: "nitshio-b"' in config.read_text(encoding="utf-8")


def test_update_metrics_y_list(tmp_path: Path) -> None:
    versions = tmp_path / "versions.json"
    register_version("nitshio-a", "base", registry_path=versions)
    update_metrics("nitshio-a", {"score": 0.9}, registry_path=versions)
    entrada = find_version("nitshio-a", load_registry(versions))
    assert entrada is not None
    assert entrada["metrics"]["score"] == 0.9
    assert [v["name"] for v in list_versions(load_registry(versions))] == [
        "nitshio-a"
    ]


def test_registro_corrupto_no_se_sobrescribe(tmp_path: Path) -> None:
    versions = tmp_path / "versions.json"
    versions.write_text("{roto", encoding="utf-8")
    with pytest.raises(VersionError):
        load_registry(versions)


def test_actualizar_config_sin_linea_active(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("ollama:\n  models:\n    active: viejo\n", encoding="utf-8")
    with pytest.raises(VersionError, match="No se encontró"):
        actualizar_config("nitshio-x", config)


def test_register_sin_nombre(tmp_path: Path) -> None:
    with pytest.raises(VersionError):
        register_version("", "base", registry_path=tmp_path / "v.json")


def test_json_ordenado_por_fecha(tmp_path: Path) -> None:
    versions = tmp_path / "versions.json"
    register_version("nitshio-a", "base", registry_path=versions)
    register_version("nitshio-b", "base", registry_path=versions)
    data = json.loads(versions.read_text(encoding="utf-8"))
    assert [v["name"] for v in data["versions"]] == ["nitshio-a", "nitshio-b"]
