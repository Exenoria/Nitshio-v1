#!/usr/bin/env python
"""Comprueba que el código modificado esté documentado en el changelog.

Regla del proyecto: ningún cambio de código se da por terminado si no tiene una
entrada en ``docs/CHANGELOG.md`` y la documentación correspondiente.

El proyecto no usa git de forma obligatoria, así que la comprobación se basa en
fechas de modificación:

1. Se toma la entrada más reciente de ``docs/CHANGELOG.md``.
2. Se buscan archivos de código (``.py``, ``.yaml``, ``.json``, etc.) con fecha
   de modificación posterior a la del changelog.
3. Cada archivo debe aparecer mencionado en la entrada más reciente, ya sea por
   su ruta completa, por su carpeta (``brain/``) o por su nombre de archivo.

También verifica que existan los documentos obligatorios y que la última
entrada tenga la sección ``### Archivos afectados``.

Uso:
    python scripts/check_docs.py
    python scripts/check_docs.py --root C:\\Users\\x\\Nitshio
    python scripts/check_docs.py --tolerance 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Carpetas que contienen código o configuración del producto.
CODE_DIRS = (
    "brain",
    "stt",
    "tts",
    "memory",
    "tools",
    "avatar",
    "pipeline",
    "scripts",
    "tests",
    "training",
    "config",
)

# Archivos sueltos que también son código.
CODE_FILES = ("run.py", "run.bat", "install.ps1")

CODE_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".toml",
    ".ini",
    ".cfg",
    ".bat",
    ".ps1",
    ".md",
}

EXCLUDED_PARTS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data",
    "logs",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}

REQUIRED_DOCS = (
    "README.md",
    "docs/PLAN.md",
    "docs/ARCHITECTURE.md",
    "docs/CHANGELOG.md",
    "docs/DECISIONS.md",
    "docs/TRAINING.md",
    "docs/EXTENDING.md",
    "docs/TROUBLESHOOTING.md",
    "docs/VSEEFACE.md",
)

_ENTRY_RE = re.compile(r"(?ms)^## \[[^\]]+\].*?(?=^## \[|\Z)")


def entrada_mas_reciente(texto: str) -> str:
    """Devuelve la primera sección ``## [versión]`` del changelog."""
    match = _ENTRY_RE.search(texto or "")
    return match.group(0).strip() if match else ""


def _iter_code_files(root: Path) -> list[Path]:
    archivos: list[Path] = []
    for nombre in CODE_DIRS:
        base = root / nombre
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in CODE_SUFFIXES:
                continue
            if any(parte in EXCLUDED_PARTS for parte in path.parts):
                continue
            archivos.append(path)
    for nombre in CODE_FILES:
        path = root / nombre
        if path.is_file():
            archivos.append(path)
    return archivos


def esta_documentado(relativa: str, entrada: str) -> bool:
    """True si la ruta aparece en la entrada, por ruta, carpeta o nombre."""
    rel = relativa.replace("\\", "/")
    if rel in entrada:
        return True
    padre = Path(rel).parent.as_posix()
    if padre not in {".", ""}:
        # ``brain/`` o ``brain/*`` como carpeta, no como prefijo de
        # ``brain/modulo.py`` (que no documenta a ``brain/otro.py``).
        patron_carpeta = rf"(?<![\w/.-]){re.escape(padre)}/(?![\w.-])"
        if re.search(patron_carpeta, entrada):
            return True
    nombre = Path(rel).name
    patron_nombre = rf"(?<![\w/.-]){re.escape(nombre)}(?![\w.-])"
    return re.search(patron_nombre, entrada) is not None


def archivos_pendientes(
    root: Path, entrada: str, cutoff: float, *, tolerance: float = 1.0
) -> list[str]:
    """Lista archivos modificados después del changelog y no documentados."""
    pendientes: list[str] = []
    for path in _iter_code_files(root):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime <= cutoff + tolerance:
            continue
        relativa = path.relative_to(root).as_posix()
        if not esta_documentado(relativa, entrada):
            pendientes.append(relativa)
    return sorted(pendientes)


def verificar(
    root: Path | None = None, *, tolerance: float = 1.0
) -> tuple[list[str], int]:
    """Devuelve ``(problemas, archivos_revisados)``."""
    root = (root or ROOT).resolve()
    problemas: list[str] = []

    for relativa in REQUIRED_DOCS:
        if not (root / relativa).is_file():
            problemas.append(f"Falta el documento obligatorio: {relativa}")

    changelog = root / "docs" / "CHANGELOG.md"
    if not changelog.is_file():
        return problemas, 0

    texto = changelog.read_text(encoding="utf-8", errors="replace")
    entrada = entrada_mas_reciente(texto)
    if not entrada:
        problemas.append(
            "docs/CHANGELOG.md no tiene ninguna entrada '## [versión]'."
        )
        return problemas, 0
    if "### Archivos afectados" not in entrada:
        problemas.append(
            "La última entrada del changelog no tiene '### Archivos afectados'."
        )

    cutoff = changelog.stat().st_mtime
    pendientes = archivos_pendientes(
        root, entrada, cutoff, tolerance=tolerance
    )
    for relativa in pendientes:
        problemas.append(
            f"Sin documentar en la última entrada del changelog: {relativa}"
        )
    return problemas, len(_iter_code_files(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifica changelog y documentación obligatoria."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Segundos de margen para fechas de modificación (default: 1).",
    )
    args = parser.parse_args(argv)

    problemas, revisados = verificar(args.root, tolerance=args.tolerance)
    print("=" * 64)
    print("  NITSHIO — comprobación de documentación")
    print("=" * 64)
    print(f"Raíz: {args.root.resolve()}")
    print(f"Archivos de código revisados: {revisados}")
    if problemas:
        print(f"\nPROBLEMAS ({len(problemas)}):")
        for problema in problemas:
            print(f"  - {problema}")
        print("\nActualiza docs/CHANGELOG.md y la documentación correspondiente.")
        return 1
    print("\nOK: changelog y documentos obligatorios al día.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
