"""Tests del servicio que alimenta la UI de entrenamiento."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from brain.config import cargar_config
from brain.llm.ollama_client import StructuredChatResult
from brain.llm.parser import parse_response
from training.service import TrainingService, TrainingServiceError
from training.versions import register_version


class ClienteFalso:
    def chat_structured(self, model, messages, **kwargs):
        crudo = json.dumps(
            {"response": "Hmph. ¿Qué quieres ahora?", "emotion": "neutral"},
            ensure_ascii=False,
        )
        parsed = parse_response(crudo, kwargs["allowed_emotions"])
        return StructuredChatResult(
            parsed=parsed, attempts=1, raw_responses=[crudo], chat_data={}
        )


class MemoriaFalsa:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, collection, text, *, metadata=None, id=None):
        self.items.append(
            {
                "collection": collection,
                "text": text,
                "metadata": dict(metadata or {}),
            }
        )
        return f"mem-{len(self.items)}"


@pytest.fixture()
def servicio(tmp_path: Path) -> TrainingService:
    config = copy.deepcopy(cargar_config())
    config["training"]["dataset_dir"] = str(tmp_path / "datasets")
    config["training"]["versions_file"] = str(tmp_path / "versions.json")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'ollama:\n  models:\n    active: "nitshio-v1"\n', encoding="utf-8"
    )
    return TrainingService(
        config,
        client=ClienteFalso(),
        memory=MemoriaFalsa(),
        config_path=config_path,
    )


def test_chat_y_memoria(servicio: TrainingService) -> None:
    turno = servicio.chat("Hola")
    assert turno.response
    assert turno.emotion == "neutral"
    assert servicio.memory is not None
    assert len(servicio.memory.items) == 1


def test_feedback_sin_correccion(servicio: TrainingService) -> None:
    turno = servicio.chat("Hola")
    resultado = servicio.save_feedback(
        turno.user, turno.response, turno.emotion, "up"
    )
    assert resultado.saved_memory
    assert resultado.correction_path is None


def test_feedback_con_correccion(servicio: TrainingService) -> None:
    turno = servicio.chat("Hola")
    resultado = servicio.save_feedback(
        turno.user,
        turno.response,
        turno.emotion,
        "down",
        corrected_response="Hmph. ¿Otra vez tú?",
        corrected_emotion="angry",
    )
    assert resultado.correction_path
    assert Path(resultado.correction_path).exists()
    correcciones = servicio.load_corrections()
    assert len(correcciones) == 1
    assert correcciones[0].emotion == "angry"


def test_append_example(servicio: TrainingService, tmp_path: Path) -> None:
    destino = servicio.append_example(
        "Nueva pregunta", "Nueva respuesta, tonto.", "angry"
    )
    assert destino.exists()
    assert "Nueva pregunta" in destino.read_text(encoding="utf-8")


def test_editor_dataset(servicio: TrainingService, tmp_path: Path) -> None:
    ruta = tmp_path / "editado.jsonl"
    texto = json.dumps(
        {"user": "Hola", "response": "Hmph.", "emotion": "neutral"},
        ensure_ascii=False,
    )
    assert servicio.validate_dataset_text(texto) == []
    guardado = servicio.save_dataset_text(texto, path=ruta)
    assert guardado.exists()
    guardado_texto = servicio.dataset_text(ruta)
    assert json.loads(guardado_texto)["user"] == "Hola"
    assert guardado_texto.endswith("\n")


def test_editor_dataset_invalido(servicio: TrainingService) -> None:
    errores = servicio.validate_dataset_text('{"user":"Hola"}')
    assert errores
    with pytest.raises(TrainingServiceError):
        servicio.save_dataset_text('{"user":"Hola"}')


def test_versiones_activar_y_rollback(
    servicio: TrainingService, tmp_path: Path
) -> None:
    register_version(
        "nitshio-a",
        "base-a",
        registry_path=servicio.versions_file,
        config_path=servicio.config_path,
    )
    register_version(
        "nitshio-b",
        "base-b",
        registry_path=servicio.versions_file,
        config_path=servicio.config_path,
    )
    filas = servicio.version_rows()
    assert {fila[0] for fila in filas} == {"nitshio-a", "nitshio-b"}

    servicio.activate("nitshio-a")
    assert servicio.versions()["active"] == "nitshio-a"
    servicio.activate("nitshio-b")
    assert servicio.versions()["previous_active"] == "nitshio-a"
    entrada = servicio.rollback()
    assert entrada["name"] == "nitshio-a"
    assert servicio.versions()["active"] == "nitshio-a"


def test_cloud_plan(servicio: TrainingService) -> None:
    plan = servicio.cloud_plan()
    assert plan["notebook"].endswith("Nitshio_QLoRA.ipynb")
    assert "finetune_qlora.py" in plan["command"]
    assert "Kaggle" in plan["message"]
