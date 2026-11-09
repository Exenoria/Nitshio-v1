#!/usr/bin/env python
"""Compatibilidad: construcción de Modelfile desde ``scripts/``.

La implementación canónica vive en ``training/build_model.py``. Este archivo
se conserva para que los comandos y tests existentes sigan funcionando.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.build_model import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_TEMPLATE,
    VERSIONS_FILE,
    activar_en_config,
    crear_en_ollama,
    main,
    registrar_version,
    render_modelfile,
)

__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_TEMPLATE",
    "VERSIONS_FILE",
    "activar_en_config",
    "crear_en_ollama",
    "main",
    "registrar_version",
    "render_modelfile",
]

if __name__ == "__main__":
    raise SystemExit(main())
