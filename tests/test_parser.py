"""Tests del parser JSON de Nitshio."""

from __future__ import annotations

import json

import pytest

from brain.llm.parser import (
    ParseError,
    fallback_response,
    parse_response,
    repair_json_text,
    response_schema,
)

EMOCIONES = ["neutral", "fun", "angry", "joy", "sorrow", "surprised"]


def test_respuesta_valida() -> None:
    parsed = parse_response(
        '{"response": "Hola, pendejo.", "emotion": "angry"}', EMOCIONES
    )
    assert parsed.data.response == "Hola, pendejo."
    assert parsed.data.emotion == "angry"
    assert parsed.repaired is False
    assert parsed.fallback is False


def test_bloque_de_codigo() -> None:
    parsed = parse_response(
        '```json\n{"response": "Hmph.", "emotion": "neutral"}\n```',
        EMOCIONES,
    )
    assert parsed.data.emotion == "neutral"


def test_texto_extra() -> None:
    parsed = parse_response(
        'Claro, aquí está:\n{"response": "Tsk.", "emotion": "fun"}\nEspero que sirva.',
        EMOCIONES,
    )
    assert parsed.extra_text is True
    assert parsed.data.response == "Tsk."


def test_emocion_invalida() -> None:
    with pytest.raises(ParseError, match="Emoción inválida"):
        parse_response(
            '{"response": "Hola", "emotion": "happy"}', EMOCIONES
        )


def test_falta_emotion() -> None:
    with pytest.raises(ParseError, match="esquema"):
        parse_response('{"response": "Hola"}', EMOCIONES)


def test_clave_extra_rechazada() -> None:
    with pytest.raises(ParseError, match="esquema"):
        parse_response(
            '{"response": "Hola", "emotion": "neutral", "action": "x"}',
            EMOCIONES,
        )


def test_repara_coma_final() -> None:
    texto = '{"response": "Hola", "emotion": "neutral",}'
    reparado = repair_json_text(texto)
    assert json.loads(reparado)["response"] == "Hola"
    parsed = parse_response(texto, EMOCIONES)
    assert parsed.repaired is True


def test_repara_salto_de_linea_literal() -> None:
    texto = '{"response": "Hola\npendejo", "emotion": "angry"}'
    parsed = parse_response(texto, EMOCIONES)
    assert "Hola" in parsed.data.response
    assert "pendejo" in parsed.data.response


def test_repara_comillas_tipograficas() -> None:
    texto = '{"response": “Hola”, "emotion": "neutral"}'
    parsed = parse_response(texto, EMOCIONES)
    assert parsed.data.response == "Hola"


def test_fallback_rescata_response() -> None:
    parsed = fallback_response(
        '{"response": "No me hables así", "emotion": "enojaaa", "basura": ',
        EMOCIONES,
    )
    assert parsed.fallback is True
    assert parsed.data.response == "No me hables así"
    assert parsed.data.emotion == "neutral"


def test_fallback_texto_plano() -> None:
    parsed = fallback_response("Hmph, no pienso decir nada.", EMOCIONES)
    assert parsed.fallback is True
    assert parsed.data.response == "Hmph, no pienso decir nada."
    assert parsed.data.emotion == "neutral"


def test_fallback_vacio() -> None:
    parsed = fallback_response("", EMOCIONES)
    assert parsed.data.response
    assert parsed.data.emotion == "neutral"


def test_schema_tiene_enum() -> None:
    schema = response_schema(EMOCIONES)
    assert schema["properties"]["emotion"]["enum"] == EMOCIONES
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"response", "emotion"}
