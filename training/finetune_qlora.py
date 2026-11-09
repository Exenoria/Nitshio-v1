#!/usr/bin/env python
"""Fine-tune QLoRA de Nitshio en Kaggle/Colab (GPU).

Este script NO se ejecuta en la T1200 local: requiere CUDA y las dependencias
de ``requirements-train.txt``. El notebook ``training/cloud/Nitshio_QLoRA.ipynb``
lo llama después de instalar el entorno.

Ejemplo:
    python training/finetune_qlora.py \\
        --dataset training/datasets/synthetic.jsonl \\
        --output-dir /kaggle/working/nitshio-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.config import cargar_config
from brain.persona.loader import cargar_persona
from training.formatting import ejemplo_a_texto, resumen_dataset
from training.schema import load_dataset, split_dataset


class TrainingError(RuntimeError):
    """El entorno o los datos no permiten entrenar."""


def _ruta(valor: str | Path) -> Path:
    path = Path(valor)
    return path if path.is_absolute() else ROOT / path


def _plan(config: dict[str, Any], dataset: Path, args: argparse.Namespace) -> dict[str, Any]:
    training_cfg = config.get("training") or {}
    qlora = training_cfg.get("qlora") or {}
    persona = cargar_persona(config)
    resultado = load_dataset(dataset, persona.emotions)
    if not resultado.ok:
        raise TrainingError(
            "El dataset tiene errores; ejecuta training/validate_dataset.py."
        )
    split = split_dataset(
        resultado.examples,
        train=0.9,
        val=0.1,
        test=0.0,
        seed=args.seed,
    )
    return {
        "base_model": args.base_model or training_cfg.get("hf_model"),
        "dataset": str(dataset),
        "dataset_examples": len(resultado.examples),
        "train": len(split.train),
        "val": len(split.val),
        "by_emotion": resultado.counts_by_emotion(),
        "max_seq_length": args.max_seq_length or qlora.get("max_seq_length", 2048),
        "load_in_4bit": (
            qlora.get("load_in_4bit", True) if not args.no_4bit else False
        ),
        "lora_r": args.lora_r or qlora.get("lora_r", 16),
        "lora_alpha": args.lora_alpha or qlora.get("lora_alpha", 16),
        "lora_dropout": qlora.get("lora_dropout", 0.0),
        "batch_size": args.batch_size or qlora.get("batch_size", 2),
        "grad_accum": args.grad_accum or qlora.get("grad_accum", 4),
        "epochs": args.epochs or qlora.get("epochs", 3),
        "learning_rate": args.learning_rate or qlora.get("learning_rate", 0.0002),
        "target_modules": list(qlora.get("target_modules") or []),
        "seed": args.seed,
        "output_dir": str(_ruta(args.output_dir)),
    }


def _importar_entrenamiento() -> dict[str, Any]:
    """Importa el stack pesado solo cuando se va a entrenar de verdad."""
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as exc:  # pragma: no cover - depende de la nube
        raise TrainingError(
            "Faltan dependencias de entrenamiento. En Kaggle/Colab instala "
            "requirements-train.txt y el instalador de Unsloth. "
            f"Detalle: {exc}"
        ) from exc
    return {
        "Dataset": Dataset,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "SFTTrainer": SFTTrainer,
        "FastLanguageModel": FastLanguageModel,
        "is_bfloat16_supported": is_bfloat16_supported,
    }


def _crear_trainer(
    stack: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    plan: dict[str, Any],
) -> Any:
    """Crea SFTTrainer compatible con TRL antiguo y nuevo."""
    TrainingArguments = stack["TrainingArguments"]
    SFTTrainer = stack["SFTTrainer"]
    bf16 = bool(stack["is_bfloat16_supported"]())
    argumentos = TrainingArguments(
        output_dir=plan["output_dir"],
        per_device_train_batch_size=int(plan["batch_size"]),
        gradient_accumulation_steps=int(plan["grad_accum"]),
        num_train_epochs=float(plan["epochs"]),
        learning_rate=float(plan["learning_rate"]),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=not bf16,
        bf16=bf16,
        report_to="none",
        seed=int(plan["seed"]),
        optim="adamw_8bit",
    )
    base = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "dataset_text_field": "text",
        "args": argumentos,
    }
    try:
        return SFTTrainer(max_seq_length=int(plan["max_seq_length"]), **base)
    except TypeError:
        # TRL reciente movió max_seq_length a SFTConfig. Se intenta con
        # SFTConfig antes de rendirse.
        try:
            from trl import SFTConfig

            sft_args = SFTConfig(
                output_dir=plan["output_dir"],
                per_device_train_batch_size=int(plan["batch_size"]),
                gradient_accumulation_steps=int(plan["grad_accum"]),
                num_train_epochs=float(plan["epochs"]),
                learning_rate=float(plan["learning_rate"]),
                logging_steps=10,
                save_strategy="epoch",
                eval_strategy="epoch",
                save_total_limit=2,
                fp16=not bf16,
                bf16=bf16,
                report_to="none",
                seed=int(plan["seed"]),
                optim="adamw_8bit",
                max_length=int(plan["max_seq_length"]),
            )
            base["args"] = sft_args
            return SFTTrainer(**base)
        except (ImportError, TypeError) as exc:
            raise TrainingError(
                "No se pudo construir SFTTrainer con esta versión de TRL. "
                "Revisa requirements-train.txt."
            ) from exc


def entrenar(args: argparse.Namespace) -> dict[str, Any]:
    config = cargar_config(args.config)
    dataset = _ruta(args.dataset)
    if not dataset.exists():
        raise TrainingError(f"No existe el dataset: {dataset}")
    plan = _plan(config, dataset, args)
    if not plan["base_model"]:
        raise TrainingError(
            "Falta training.hf_model en config.yaml o --base-model."
        )

    if args.dry_run:
        return {"dry_run": True, **plan}

    stack = _importar_entrenamiento()
    persona = cargar_persona(config)
    FastLanguageModel = stack["FastLanguageModel"]
    AutoTokenizer = stack["AutoTokenizer"]
    Dataset = stack["Dataset"]

    tokenizer = AutoTokenizer.from_pretrained(plan["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resultado = load_dataset(dataset, persona.emotions)
    split = split_dataset(
        resultado.examples,
        train=0.9,
        val=0.1,
        test=0.0,
        seed=int(plan["seed"]),
    )
    train_records = [
        ejemplo_a_texto(ejemplo, tokenizer, persona.system_prompt)
        for ejemplo in split.train
    ]
    val_records = [
        ejemplo_a_texto(ejemplo, tokenizer, persona.system_prompt)
        for ejemplo in split.val
    ]
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records) if val_records else None

    model, _ = FastLanguageModel.from_pretrained(
        model_name=plan["base_model"],
        max_seq_length=int(plan["max_seq_length"]),
        dtype=None,
        load_in_4bit=bool(plan["load_in_4bit"]),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(plan["lora_r"]),
        target_modules=plan["target_modules"] or None,
        lora_alpha=int(plan["lora_alpha"]),
        lora_dropout=float(plan["lora_dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(plan["seed"]),
    )

    trainer = _crear_trainer(
        stack,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        plan=plan,
    )
    trainer.train()
    salida = _ruta(plan["output_dir"])
    salida.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(salida))
    tokenizer.save_pretrained(str(salida))
    trainer.save_state()

    manifest = {
        **plan,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": resumen_dataset(dataset, persona.emotions),
        "trainer_metrics": {
            clave: valor
            for clave, valor in (trainer.state.log_history[-1] or {}).items()
            if isinstance(valor, (int, float, str))
        },
    }
    try:
        import torch

        manifest["gpu"] = torch.cuda.get_device_name(0)
        manifest["torch"] = torch.__version__
    except Exception:
        pass
    (salida / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"dry_run": False, **manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune QLoRA de Nitshio (solo nube con GPU)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("training/datasets/synthetic.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nitshio-lora"))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        resultado = entrenar(args)
    except (OSError, ValueError, TrainingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
