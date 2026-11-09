"""Tests de formato de entrenamiento y de los scripts de nube en dry-run."""

from __future__ import annotations

import json
from pathlib import Path

from training.formatting import (
    cargar_textos,
    ejemplo_a_mensajes,
    ejemplo_a_texto,
    escribir_jsonl,
    renderizar_chat,
    resumen_dataset,
)
from training.schema import DatasetExample


class TokenizerFalso:
    def apply_chat_template(self, mensajes, *, tokenize=False, add_generation_prompt=False):
        partes = [f"{m['role']}: {m['content']}\n" for m in mensajes]
        if add_generation_prompt:
            partes.append("assistant:")
        return "".join(partes)


class TokenizerSinPlantilla:
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("sin plantilla")


def test_ejemplo_a_mensajes() -> None:
    ejemplo = DatasetExample(user="Hola", response="Hmph.", emotion="angry")
    mensajes = ejemplo_a_mensajes(ejemplo, "SYSTEM")
    assert mensajes[0] == {"role": "system", "content": "SYSTEM"}
    assert mensajes[1]["role"] == "user"
    assert json.loads(mensajes[2]["content"])["emotion"] == "angry"


def test_renderizar_chat_usa_tokenizer() -> None:
    ejemplo = DatasetExample(user="Hola", response="Hmph.", emotion="angry")
    texto = ejemplo_a_texto(ejemplo, TokenizerFalso(), "SYSTEM")["text"]
    assert "system: SYSTEM" in texto
    assert "user: Hola" in texto
    assert "assistant:" in texto


def test_renderizar_chat_fallback() -> None:
    mensajes = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    texto = renderizar_chat(TokenizerSinPlantilla(), mensajes)
    assert "<|system|>" in texto
    assert "<|user|>" in texto


def test_cargar_textos_y_resumen(tmp_path: Path) -> None:
    ruta = tmp_path / "datos.jsonl"
    ruta.write_text(
        json.dumps(
            {"user": "Hola", "response": "Hmph.", "emotion": "angry"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    textos = cargar_textos(ruta, ["angry"], TokenizerFalso(), "S")
    assert len(textos) == 1
    assert "Hola" in textos[0]["text"]
    resumen = resumen_dataset(ruta, ["angry"])
    assert resumen["examples"] == 1
    assert resumen["by_emotion"] == {"angry": 1}
    assert len(resumen["dataset_sha256"]) == 64


def test_escribir_jsonl(tmp_path: Path) -> None:
    ruta = tmp_path / "x.jsonl"
    escribir_jsonl(ruta, [{"a": 1}, {"b": 2}])
    assert len(ruta.read_text(encoding="utf-8").splitlines()) == 2


def test_notebook_es_json_valido() -> None:
    ruta = Path("training/cloud/Nitshio_QLoRA.ipynb")
    data = json.loads(ruta.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 8
    assert any("finetune_qlora.py" in "".join(c.get("source", [])) for c in data["cells"])


def test_validate_dataset_cli(tmp_path: Path) -> None:
    from training.validate_dataset import main

    salida = tmp_path / "splits"
    codigo = main(
        [
            "training/datasets/seed.jsonl",
            "--strict",
            "--split-dir",
            str(salida),
            "--json",
        ]
    )
    assert codigo == 0
    assert (salida / "train.jsonl").exists()
    assert (salida / "manifest.json").exists()


def test_finetune_dry_run(tmp_path: Path, capsys) -> None:
    from training.finetune_qlora import main

    codigo = main(
        [
            "--dataset",
            "training/datasets/seed.jsonl",
            "--output-dir",
            str(tmp_path / "lora"),
            "--dry-run",
        ]
    )
    assert codigo == 0
    salida = capsys.readouterr().out
    assert '"dry_run": true' in salida
    assert '"dataset_examples": 48' in salida


def test_export_dry_run(tmp_path: Path, capsys) -> None:
    from training.export_gguf import main

    adaptadores = tmp_path / "adapter"
    adaptadores.mkdir()
    codigo = main(
        [
            "--adapter-dir",
            str(adaptadores),
            "--output-dir",
            str(tmp_path / "gguf"),
            "--dry-run",
        ]
    )
    assert codigo == 0
    salida = capsys.readouterr().out
    assert '"dry_run": true' in salida
    assert '"quantization": "q4_K_M"' in salida
