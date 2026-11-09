"""Tests del esquema, validación, deduplicación y partición del dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.schema import (
    DatasetError,
    DatasetExample,
    contar_emociones,
    hash_examples,
    load_dataset,
    merge_examples,
    parse_example,
    save_jsonl,
    split_dataset,
    write_split,
)

EMOCIONES = ["neutral", "fun", "angry", "joy", "sorrow", "surprised"]


def _linea(user: str, response: str, emotion: str = "neutral") -> str:
    return json.dumps(
        {"user": user, "response": response, "emotion": emotion},
        ensure_ascii=False,
    )


def test_parse_example_valido() -> None:
    ejemplo = parse_example(
        {"user": "Hola", "response": "Hmph.", "emotion": "angry"}, EMOCIONES
    )
    assert ejemplo.user == "Hola"
    assert json.loads(ejemplo.assistant_json())["emotion"] == "angry"


@pytest.mark.parametrize(
    "data",
    [
        {"user": "", "response": "Hola", "emotion": "neutral"},
        {"user": "Hola", "response": "", "emotion": "neutral"},
        {"user": "Hola", "response": "Hmph", "emotion": "happy"},
        {
            "user": "Hola",
            "response": "Hmph",
            "emotion": "neutral",
            "extra": "x",
        },
    ],
)
def test_parse_example_rechaza_invalidos(data: dict) -> None:
    with pytest.raises(DatasetError):
        parse_example(data, EMOCIONES)


def test_load_dataset_valido(tmp_path: Path) -> None:
    ruta = tmp_path / "datos.jsonl"
    ruta.write_text(
        "\n".join(
            [
                _linea("Hola", "¿Qué quieres?", "angry"),
                _linea("Estoy triste", "Cuéntame.", "sorrow"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resultado = load_dataset(ruta, EMOCIONES)
    assert resultado.ok
    assert len(resultado.examples) == 2
    assert resultado.issues == []
    assert resultado.counts_by_emotion() == {"angry": 1, "sorrow": 1}


def test_load_dataset_reporta_errores(tmp_path: Path) -> None:
    ruta = tmp_path / "datos.jsonl"
    ruta.write_text(
        "\n".join(
            [
                _linea("Hola", "Bien", "neutral"),
                "{no es json",
                _linea("Otra", "Mal", "happy"),
                json.dumps({"user": "Sin emoción", "response": "x"}),
            ]
        ),
        encoding="utf-8",
    )
    resultado = load_dataset(ruta, EMOCIONES)
    assert not resultado.ok
    assert len(resultado.examples) == 1
    assert len(resultado.errors) == 3


def test_deduplicacion_exacta_y_casi_duplicada(tmp_path: Path) -> None:
    larga = "Esta es una respuesta larga con varias palabras para comparar el contenido"
    ruta = tmp_path / "datos.jsonl"
    ruta.write_text(
        "\n".join(
            [
                _linea("Hola amigo", larga, "neutral"),
                _linea("Hola amigo", larga, "neutral"),
                _linea(
                    "Hola amigo",
                    "Esta es una respuesta larga con varias palabras para comparar el contenid",
                    "neutral",
                ),
            ]
        ),
        encoding="utf-8",
    )
    resultado = load_dataset(ruta, EMOCIONES, near_duplicate_threshold=0.8)
    assert resultado.ok
    assert len(resultado.examples) == 1
    assert resultado.duplicates == 2
    assert all(issue.kind == "duplicate" for issue in resultado.warnings)


def test_hash_es_independiente_del_orden() -> None:
    a = DatasetExample(user="a", response="b", emotion="neutral")
    b = DatasetExample(user="c", response="d", emotion="fun")
    assert hash_examples([a, b]) == hash_examples([b, a])


def test_merge_examples_conserva_el_primero() -> None:
    a = DatasetExample(user="Hola", response="Uno", emotion="neutral")
    b = DatasetExample(user="Hola", response="Uno", emotion="neutral")
    c = DatasetExample(user="Adiós", response="Dos", emotion="fun")
    assert merge_examples([a], [b, c]) == [a, c]


def _ejemplos() -> list[DatasetExample]:
    ejemplos: list[DatasetExample] = []
    for emocion in EMOCIONES:
        for indice in range(4):
            ejemplos.append(
                DatasetExample(
                    user=f"Usuario {emocion} {indice}",
                    response=f"Respuesta {emocion} {indice}",
                    emotion=emocion,
                )
            )
    return ejemplos


def test_split_estratificado_sin_solapamientos() -> None:
    ejemplos = _ejemplos()
    split = split_dataset(ejemplos, seed=7)
    todos = split.train + split.val + split.test
    assert len(todos) == len(ejemplos)
    assert len({e.canonical() for e in todos}) == len(ejemplos)
    assert split.ratios == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert len(split.train) == 12
    assert len(split.val) == 6
    assert len(split.test) == 6


def test_split_es_determinista() -> None:
    ejemplos = _ejemplos()
    a = split_dataset(ejemplos, seed=123)
    b = split_dataset(ejemplos, seed=123)
    c = split_dataset(ejemplos, seed=124)
    assert [e.canonical() for e in a.train] == [e.canonical() for e in b.train]
    assert [e.canonical() for e in a.train] != [e.canonical() for e in c.train]


def test_write_split_y_manifest(tmp_path: Path) -> None:
    split = split_dataset(_ejemplos(), seed=1)
    rutas = write_split(split, tmp_path, source_hash="abc")
    assert rutas["train"].exists()
    assert rutas["val"].exists()
    assert rutas["test"].exists()
    manifest = json.loads(rutas["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_hash"] == "abc"
    assert manifest["counts"]["total"] == 24
    assert manifest["by_emotion"]["train"]["neutral"] >= 1


def test_save_jsonl_y_contar(tmp_path: Path) -> None:
    ruta = tmp_path / "salida.jsonl"
    ejemplos = _ejemplos()[:3]
    save_jsonl(ruta, ejemplos)
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 3
    assert contar_emociones(ejemplos)["neutral"] == 3
