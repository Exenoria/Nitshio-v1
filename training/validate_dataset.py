#!/usr/bin/env python
"""Valida un dataset JSONL de Nitshio y, opcionalmente, lo divide.

Ejemplos:
    python training/validate_dataset.py training/datasets/seed.jsonl
    python training/validate_dataset.py training/datasets/seed.jsonl --strict
    python training/validate_dataset.py training/datasets/seed.jsonl \\
        --output training/datasets/clean.jsonl \\
        --split-dir training/datasets/splits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config
from brain.persona.loader import cargar_emociones
from training.schema import (
    DatasetError,
    hash_examples,
    load_dataset,
    save_jsonl,
    split_dataset,
    write_split,
)


def _ruta(valor: str | Path) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def _resumen(resultado: Any) -> dict[str, Any]:
    return {
        "path": str(resultado.path),
        "ok": resultado.ok,
        "examples": len(resultado.examples),
        "errors": len(resultado.errors),
        "duplicates": resultado.duplicates,
        "total_lines": resultado.total_lines,
        "blank_lines": resultado.blank_lines,
        "by_emotion": resultado.counts_by_emotion(),
        "hash": hash_examples(resultado.examples),
        "issues": [
            {
                "line": issue.line,
                "kind": issue.kind,
                "message": issue.message,
            }
            for issue in resultado.issues
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida y divide el dataset JSONL de entrenamiento."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="training/datasets/seed.jsonl",
        help="Ruta del JSONL (default: training/datasets/seed.jsonl).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla también si hay duplicados (por defecto solo avisa).",
    )
    parser.add_argument(
        "--no-near-duplicates",
        action="store_true",
        help="Desactiva la detección de casi duplicados.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Escribe el dataset limpio (sin duplicados).",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Escribe train.jsonl, val.jsonl, test.jsonl y manifest.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="Salida JSON.")
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
        emociones, _ = cargar_emociones(
            _ruta(config["persona"]["emotions_file"])
        )
        resultado = load_dataset(
            _ruta(args.dataset),
            emociones,
            detect_near_duplicates=not args.no_near_duplicates,
        )
    except (OSError, ValueError, DatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    resumen = _resumen(resultado)
    if args.output:
        ruta_salida = save_jsonl(_ruta(args.output), resultado.examples)
        resumen["output"] = str(ruta_salida)
    if args.split_dir:
        split = split_dataset(
            resultado.examples,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
        )
        rutas = write_split(
            split,
            _ruta(args.split_dir),
            source_hash=resumen["hash"],
        )
        resumen["split"] = {
            "seed": split.seed,
            "ratios": split.ratios,
            "counts": {
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
            },
            "files": {clave: str(ruta) for clave, ruta in rutas.items()},
        }

    if args.json:
        print(json.dumps(resumen, ensure_ascii=False, indent=2))
    else:
        print("=" * 64)
        print("  NITSHIO — validación de dataset")
        print("=" * 64)
        print(f"Archivo: {resultado.path}")
        print(f"Líneas: {resultado.total_lines} (vacías: {resultado.blank_lines})")
        print(f"Ejemplos válidos: {len(resultado.examples)}")
        print(f"Errores: {len(resultado.errors)}")
        print(f"Duplicados descartados: {resultado.duplicates}")
        print(f"Hash: {resumen['hash']}")
        print("Por emoción:", json.dumps(resumen["by_emotion"], ensure_ascii=False))
        for issue in resultado.issues:
            print(f"  - {issue}")
        if args.output:
            print(f"Dataset limpio: {resumen['output']}")
        if "split" in resumen:
            print("Partición:", json.dumps(resumen["split"]["counts"], ensure_ascii=False))
        print("\nOK" if resultado.ok else "\nHAY ERRORES DE FORMATO")

    if not resultado.ok:
        return 1
    if args.strict and resultado.duplicates:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
