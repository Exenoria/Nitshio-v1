"""Tests de scripts/check_docs.py (sin depender de git)."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.check_docs import (
    REQUIRED_DOCS,
    archivos_pendientes,
    entrada_mas_reciente,
    esta_documentado,
    main,
    verificar,
)

CHANGELOG = """# Changelog

## [0.2.0] — 2026-11-07

### Añadido

- Algo nuevo.

### Archivos afectados

`brain/modulo.py`, `tests/`.

## [0.1.0] — 2026-11-01

### Archivos afectados

`brain/viejo.py`.
"""


def _crear_docs(root: Path) -> None:
    for relativa in REQUIRED_DOCS:
        path = root / relativa
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# doc\n", encoding="utf-8")
    (root / "docs" / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")


def _fijar_mtime(path: Path, valor: float) -> None:
    os.utime(path, (valor, valor))


def test_entrada_mas_reciente() -> None:
    entrada = entrada_mas_reciente(CHANGELOG)
    assert "0.2.0" in entrada
    assert "0.1.0" not in entrada


def test_esta_documentado_por_ruta_carpeta_y_nombre() -> None:
    entrada = entrada_mas_reciente(CHANGELOG)
    assert esta_documentado("brain/modulo.py", entrada)
    assert esta_documentado("tests/test_algo.py", entrada)
    assert not esta_documentado("brain/otro.py", entrada)


def test_archivos_pendientes(tmp_path: Path) -> None:
    _crear_docs(tmp_path)
    changelog = tmp_path / "docs" / "CHANGELOG.md"
    cutoff = changelog.stat().st_mtime
    modulo = tmp_path / "brain" / "modulo.py"
    modulo.parent.mkdir(parents=True)
    modulo.write_text("x = 1\n", encoding="utf-8")
    _fijar_mtime(modulo, cutoff - 10)

    otro = tmp_path / "brain" / "otro.py"
    otro.write_text("y = 2\n", encoding="utf-8")
    _fijar_mtime(otro, cutoff + 10)

    entrada = entrada_mas_reciente(CHANGELOG)
    pendientes = archivos_pendientes(tmp_path, entrada, cutoff, tolerance=0)
    assert pendientes == ["brain/otro.py"]


def test_verificar_ok(tmp_path: Path) -> None:
    _crear_docs(tmp_path)
    changelog = tmp_path / "docs" / "CHANGELOG.md"
    cutoff = changelog.stat().st_mtime
    modulo = tmp_path / "brain" / "modulo.py"
    modulo.parent.mkdir(parents=True)
    modulo.write_text("x = 1\n", encoding="utf-8")
    _fijar_mtime(modulo, cutoff - 10)

    problemas, revisados = verificar(tmp_path, tolerance=0)
    assert problemas == []
    assert revisados >= 1


def test_verificar_detecta_falta_de_docs(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    problemas, _ = verificar(tmp_path)
    assert any("README.md" in p for p in problemas)


def test_verificar_detecta_sin_archivos_afectados(tmp_path: Path) -> None:
    _crear_docs(tmp_path)
    (tmp_path / "docs" / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [0.2.0] — 2026-11-07\n\n### Añadido\n\n- x\n",
        encoding="utf-8",
    )
    problemas, _ = verificar(tmp_path)
    assert any("Archivos afectados" in p for p in problemas)


def test_verificar_detecta_codigo_sin_documentar(tmp_path: Path) -> None:
    _crear_docs(tmp_path)
    changelog = tmp_path / "docs" / "CHANGELOG.md"
    cutoff = changelog.stat().st_mtime
    nuevo = tmp_path / "brain" / "nuevo.py"
    nuevo.parent.mkdir(parents=True)
    nuevo.write_text("z = 3\n", encoding="utf-8")
    _fijar_mtime(nuevo, cutoff + 10)

    problemas, _ = verificar(tmp_path, tolerance=0)
    assert any("brain/nuevo.py" in p for p in problemas)


def test_main_devuelve_uno_con_problemas(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    assert main(["--root", str(tmp_path)]) == 1


def test_main_devuelve_cero(tmp_path: Path) -> None:
    _crear_docs(tmp_path)
    assert main(["--root", str(tmp_path), "--tolerance", "0"]) == 0
