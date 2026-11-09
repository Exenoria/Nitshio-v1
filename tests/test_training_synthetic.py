"""Tests del generador sintético con un cliente Ollama falso."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.config import cargar_config
from training.generate_synthetic import (
    SyntheticError,
    extraer_ejemplos,
    generar,
)
from training.schema import DatasetExample


class ClienteSintetico:
    def __init__(self, ejemplos: list[dict]) -> None:
        self.ejemplos = ejemplos
        self.llamadas = 0

    def chat(self, model, messages, **kwargs):
        self.llamadas += 1
        return {
            "message": {
                "content": json.dumps(
                    {"examples": self.ejemplos}, ensure_ascii=False
                )
            }
        }


def _ejemplo(user: str, response: str, emotion: str = "angry") -> dict:
    return {"user": user, "response": response, "emotion": emotion}


def test_extraer_ejemplos_con_texto_extra() -> None:
    crudo = 'Claro:\n{"examples":[{"user":"hola","response":"hmph","emotion":"angry"}]}\n'
    ejemplos = extraer_ejemplos(crudo)
    assert ejemplos[0]["user"] == "hola"


def test_extraer_ejemplos_lista_directa() -> None:
    crudo = '[{"user":"hola","response":"hmph","emotion":"angry"}]'
    assert extraer_ejemplos(crudo)[0]["emotion"] == "angry"


def test_extraer_ejemplos_invalido() -> None:
    with pytest.raises(SyntheticError):
        extraer_ejemplos("no hay json")


def test_generar_y_escribir(tmp_path: Path) -> None:
    config = cargar_config()
    cliente = ClienteSintetico(
        [
            _ejemplo("Oye, ¿me ayudas?", "Depende de qué, menso.", "angry"),
            _ejemplo("Te ves cansada.", "No es tu problema, idiota.", "angry"),
        ]
    )
    salida = tmp_path / "synthetic.jsonl"
    resumen = generar(
        config,
        emociones=["angry"],
        per_emotion=2,
        batch_size=2,
        max_rounds=1,
        client=cliente,
        seed_examples=[],
        output=salida,
        quiet=True,
    )
    assert resumen["generated"] == 2
    assert resumen["by_emotion"] == {"angry": 2}
    assert salida.exists()
    assert len(salida.read_text(encoding="utf-8").splitlines()) == 2


def test_generar_descarta_invalidos_y_duplicados(tmp_path: Path) -> None:
    config = cargar_config()
    valido = _ejemplo("Oye, ¿me ayudas?", "Depende de qué, menso.", "angry")
    cliente = ClienteSintetico(
        [
            valido,
            dict(valido),
            _ejemplo("Otra cosa", "Respuesta", "joy"),
            {"user": "", "response": "x", "emotion": "angry"},
        ]
    )
    resumen = generar(
        config,
        emociones=["angry"],
        per_emotion=1,
        batch_size=4,
        max_rounds=1,
        client=cliente,
        seed_examples=[],
        output=tmp_path / "out.jsonl",
        quiet=True,
    )
    assert resumen["generated"] == 1
    assert resumen["discarded"] == 3


def test_generar_no_escribe_en_dry_run(tmp_path: Path) -> None:
    config = cargar_config()
    cliente = ClienteSintetico([_ejemplo("Hola", "Hmph.", "angry")])
    salida = tmp_path / "no_debe_existir.jsonl"
    resumen = generar(
        config,
        emociones=["angry"],
        per_emotion=1,
        batch_size=1,
        max_rounds=1,
        client=cliente,
        seed_examples=[],
        output=salida,
        dry_run=True,
        quiet=True,
    )
    assert resumen["dry_run"] is True
    assert resumen["generated"] == 1
    assert not salida.exists()


def test_generar_rechaza_emocion_desconocida() -> None:
    config = cargar_config()
    with pytest.raises(SyntheticError, match="no declarada"):
        generar(
            config,
            emociones=["feliz"],
            per_emotion=1,
            batch_size=1,
            max_rounds=1,
            client=ClienteSintetico([]),
            seed_examples=[],
            quiet=True,
        )


def test_generar_respeta_semilla_existente(tmp_path: Path) -> None:
    config = cargar_config()
    semilla = DatasetExample(
        user="Hola", response="Hmph. ¿Qué quieres?", emotion="angry"
    )
    cliente = ClienteSintetico(
        [
            _ejemplo("Hola", "Hmph. ¿Qué quieres?", "angry"),
            _ejemplo("Nueva pregunta", "Nueva respuesta, tonto.", "angry"),
        ]
    )
    resumen = generar(
        config,
        emociones=["angry"],
        per_emotion=1,
        batch_size=2,
        max_rounds=1,
        client=cliente,
        seed_examples=[semilla],
        output=tmp_path / "out.jsonl",
        quiet=True,
    )
    assert resumen["generated"] == 1
    assert resumen["total_dataset"] == 2
