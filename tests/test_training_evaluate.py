"""Tests del evaluador de modelos (sin red real)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.config import cargar_config
from brain.llm.ollama_client import StructuredChatResult
from brain.llm.parser import fallback_response, parse_response
from training.evaluate import (
    EvaluationError,
    PromptResult,
    calcular_metricas,
    cargar_prompts,
    evaluate_model,
    generar_reporte,
)

EMOCIONES = ["neutral", "fun", "angry", "joy", "sorrow", "surprised"]


class EmbedderFalso:
    def embed(self, textos: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, float(len(texto) % 7)] for texto in textos
        ]

    def embed_query(self, texto: str) -> list[float]:
        return self.embed([texto])[0]


class ClienteFalso:
    def __init__(self, *, fallar: bool = False, juez: int = 4) -> None:
        self.fallar = fallar
        self.juez = juez
        self.llamadas: list[str] = []

    def chat_structured(self, model, messages, **kwargs):
        self.llamadas.append(model)
        ultimo = messages[-1]["content"]
        if self.fallar:
            crudo = "esto no es JSON"
            parsed = fallback_response(crudo, kwargs["allowed_emotions"])
        else:
            emocion = "sorrow" if "triste" in ultimo.lower() else "neutral"
            respuesta = (
                "Cuéntame qué pasó, no estás solo."
                if emocion == "sorrow"
                else "Hmph. ¿Qué quieres ahora?"
            )
            crudo = json.dumps(
                {"response": respuesta, "emotion": emocion}, ensure_ascii=False
            )
            parsed = parse_response(crudo, kwargs["allowed_emotions"])
        return StructuredChatResult(
            parsed=parsed,
            attempts=1,
            raw_responses=[crudo],
            chat_data={"eval_count": 10, "eval_duration": 1_000_000_000},
        )

    def chat(self, model, messages, **kwargs):
        return {
            "message": {
                "content": json.dumps(
                    {"score": self.juez, "reason": "mantiene el personaje"}
                )
            }
        }


def _prompts(tmp_path: Path) -> Path:
    ruta = tmp_path / "prompts.yaml"
    ruta.write_text(
        """
prompts:
  - id: saludo
    user: "Hola"
    expected_emotion: neutral
    reference: "Hmph. ¿Qué quieres?"
    tags: [saludo]
  - id: triste
    user: "Estoy triste"
    expected_emotion: sorrow
    reference: "Cuéntame, no estás solo."
    tags: [apoyo]
""",
        encoding="utf-8",
    )
    return ruta


def test_cargar_prompts(tmp_path: Path) -> None:
    prompts = cargar_prompts(_prompts(tmp_path), EMOCIONES)
    assert len(prompts) == 2
    assert prompts[0].expected_emotion == "neutral"


def test_cargar_prompts_invalido(tmp_path: Path) -> None:
    ruta = tmp_path / "mal.yaml"
    ruta.write_text(
        "prompts:\n  - id: x\n    user: hola\n    expected_emotion: feliz\n"
        "    reference: hola\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="emoción inválida"):
        cargar_prompts(ruta, EMOCIONES)


def test_evaluate_model_metricas(tmp_path: Path) -> None:
    config = cargar_config()
    prompts = cargar_prompts(_prompts(tmp_path), EMOCIONES)
    resultado = evaluate_model(
        config,
        "modelo-falso",
        prompts,
        client=ClienteFalso(),
        embedder=EmbedderFalso(),
        judge=False,
    )
    assert resultado.metrics["count"] == 2
    assert resultado.metrics["json_valid_rate"] == 1.0
    assert resultado.metrics["emotion_accuracy"] == 1.0
    assert resultado.metrics["similarity_backend"] == "embeddings"
    assert resultado.metrics["tokens_per_second_mean"] == 10.0
    assert all(item.similarity >= 0 for item in resultado.prompts)
    assert resultado.prompts[1].emotion == "sorrow"


def test_evaluate_model_con_juez(tmp_path: Path) -> None:
    config = cargar_config()
    prompts = cargar_prompts(_prompts(tmp_path), EMOCIONES)
    resultado = evaluate_model(
        config,
        "modelo-falso",
        prompts,
        client=ClienteFalso(juez=5),
        embedder=EmbedderFalso(),
        judge=True,
        judge_model="juez-falso",
    )
    assert resultado.metrics["personality_mean"] == 1.0
    assert resultado.judge_model == "juez-falso"


def test_evaluate_model_fallback(tmp_path: Path) -> None:
    config = cargar_config()
    prompts = cargar_prompts(_prompts(tmp_path), EMOCIONES)
    resultado = evaluate_model(
        config,
        "modelo-falso",
        prompts,
        client=ClienteFalso(fallar=True),
        embedder=EmbedderFalso(),
        judge=False,
    )
    assert resultado.metrics["json_valid_rate"] == 0.0
    assert all(item.fallback for item in resultado.prompts)


def test_similitud_lexica_sin_embedder(tmp_path: Path) -> None:
    config = cargar_config()
    prompts = cargar_prompts(_prompts(tmp_path), EMOCIONES)
    resultado = evaluate_model(
        config,
        "modelo-falso",
        prompts,
        client=ClienteFalso(),
        embedder=None,
        forzar_lexica=True,
        judge=False,
    )
    assert resultado.metrics["similarity_backend"] == "lexica"
    assert resultado.prompts[0].similarity_backend == "lexica"


def test_reporte_markdown(tmp_path: Path) -> None:
    config = cargar_config()
    prompts = cargar_prompts(_prompts(tmp_path), EMOCIONES)
    resultado = evaluate_model(
        config,
        "modelo-falso",
        prompts,
        client=ClienteFalso(),
        embedder=EmbedderFalso(),
        judge=False,
    )
    reporte = generar_reporte(
        [resultado], prompts_path=str(_prompts(tmp_path)), judge=False
    )
    assert "# Evaluación de Nitshio" in reporte
    assert "modelo-falso" in reporte
    assert "Resumen" in reporte
    assert "Detalle" in reporte


def test_calcular_metricas_vacio() -> None:
    metricas = calcular_metricas([], cargar_config())
    assert metricas["count"] == 0
    assert metricas["composite_score"] == 0.0


def test_prompt_result_serializable() -> None:
    item = PromptResult(
        prompt_id="x",
        user="hola",
        expected_emotion="neutral",
        reference="hola",
        response="hola",
        emotion="neutral",
        json_valid=True,
        emotion_correct=True,
        similarity=0.5,
        length_chars=4,
        latency_seconds=0.1,
    )
    data = item.as_dict()
    assert data["prompt_id"] == "x"
    assert data["similarity"] == 0.5
