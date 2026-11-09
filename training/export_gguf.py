#!/usr/bin/env python
"""Fusiona adaptadores LoRA y exporta un GGUF cuantizado.

Se ejecuta en la nube después del fine-tune. Requiere el repositorio de
llama.cpp para convertir y cuantizar (el notebook lo clona).

Ejemplo:
    python training/export_gguf.py \\
        --adapter-dir /kaggle/working/nitshio-lora \\
        --llama-cpp-dir /kaggle/working/llama.cpp \\
        --output-dir /kaggle/working/nitshio-gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config


class ExportError(RuntimeError):
    """No se pudo fusionar o convertir el modelo."""


def _ruta(valor: str | Path) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _buscar_conversor(llama_cpp_dir: Path) -> Path:
    candidatos = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert-hf-to-gguf.py",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    raise ExportError(
        f"No se encontró convert_hf_to_gguf.py en {llama_cpp_dir}. "
        "Clona llama.cpp o pasa --llama-cpp-dir."
    )


def _buscar_cuantizador(llama_cpp_dir: Path) -> Path:
    nombres = ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]
    candidatos = [llama_cpp_dir / nombre for nombre in nombres]
    candidatos.extend(
        llama_cpp_dir / "build" / "bin" / nombre for nombre in nombres
    )
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    en_path = shutil.which("llama-quantize") or shutil.which("quantize")
    if en_path:
        return Path(en_path)
    raise ExportError(
        "No se encontró llama-quantize. Compila llama.cpp o pásalo en el PATH."
    )


def _fusionar(
    base_model: str,
    adapter_dir: Path,
    merged_dir: Path,
) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depende de la nube
        raise ExportError(
            "Faltan dependencias de exportación (torch, peft, transformers). "
            "Instala requirements-train.txt."
        ) from exc

    merged_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    modelo = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    modelo = PeftModel.from_pretrained(modelo, str(adapter_dir))
    modelo = modelo.merge_and_unload()
    modelo.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))


def _convertir_y_cuantizar(
    merged_dir: Path,
    llama_cpp_dir: Path,
    output_dir: Path,
    quantization: str,
) -> tuple[Path, Path]:
    conversor = _buscar_conversor(llama_cpp_dir)
    cuantizador = _buscar_cuantizador(llama_cpp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    f16 = output_dir / "nitshio-f16.gguf"
    final = output_dir / f"nitshio-{quantization}.gguf"

    comando_conversion = [
        sys.executable,
        str(conversor),
        str(merged_dir),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    ]
    print("Ejecutando:", " ".join(comando_conversion))
    subprocess.run(comando_conversion, check=True)
    if not f16.exists():
        raise ExportError(f"El conversor no generó {f16}.")

    comando_cuantizar = [
        str(cuantizador),
        str(f16),
        str(final),
        quantization,
    ]
    print("Ejecutando:", " ".join(comando_cuantizar))
    subprocess.run(comando_cuantizar, check=True)
    if not final.exists():
        raise ExportError(f"El cuantizador no generó {final}.")
    return f16, final


def exportar(args: argparse.Namespace) -> dict[str, Any]:
    config = cargar_config(args.config)
    training_cfg = config.get("training") or {}
    adapter_dir = _ruta(args.adapter_dir)
    if not adapter_dir.exists():
        raise ExportError(f"No existe el directorio de adaptadores: {adapter_dir}")
    base_model = args.base_model or training_cfg.get("hf_model")
    if not base_model:
        raise ExportError("Falta training.hf_model o --base-model.")
    quantization = args.quantization or (training_cfg.get("gguf") or {}).get(
        "quantization", "q4_K_M"
    )
    output_dir = _ruta(args.output_dir)
    merged_dir = output_dir / "merged-hf"
    plan = {
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "merged_dir": str(merged_dir),
        "quantization": quantization,
    }
    if args.dry_run:
        return {"dry_run": True, **plan}

    _fusionar(base_model, adapter_dir, merged_dir)
    f16 = final = None
    if not args.skip_convert:
        if not args.llama_cpp_dir:
            raise ExportError(
                "Falta --llama-cpp-dir para convertir a GGUF. También puedes "
                "usar --skip-convert para dejar solo el modelo fusionado."
            )
        f16, final = _convertir_y_cuantizar(
            merged_dir,
            _ruta(args.llama_cpp_dir),
            output_dir,
            quantization,
        )

    manifest = {
        **plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "merged": str(merged_dir),
        "f16_gguf": str(f16) if f16 else None,
        "quantized_gguf": str(final) if final else None,
    }
    if final and final.exists():
        manifest["quantized_sha256"] = _sha256(final)
        manifest["quantized_bytes"] = final.stat().st_size
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"dry_run": False, **manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fusiona LoRA y exporta GGUF para Ollama."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/nitshio-gguf")
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--llama-cpp-dir", type=Path, default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        resultado = exportar(args)
    except (OSError, ValueError, ExportError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
