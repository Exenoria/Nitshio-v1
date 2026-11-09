"""Tests del chunker de TTS."""

from __future__ import annotations

from tts.chunker import dividir_en_chunks, normalizar_texto


def test_texto_vacio() -> None:
    assert dividir_en_chunks("") == []
    assert dividir_en_chunks("   ") == []


def test_normaliza_espacios() -> None:
    assert normalizar_texto("  hola \n\n mundo  ") == "hola mundo"


def test_respeta_limites_y_primero_mas_corto() -> None:
    texto = (
        "Hola, soy Nitshio. ¿Cómo estás hoy? Espero que bien. "
        "No me importa, pero dime."
    )
    chunks = dividir_en_chunks(
        texto, max_chars=40, first_chunk_max_chars=22, min_chars=5
    )
    assert chunks
    assert len(chunks[0]) <= 22
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert all(chunk.strip() == chunk for chunk in chunks)


def test_une_texto_sin_perder_palabras() -> None:
    texto = "Uno dos tres. Cuatro cinco seis. Siete ocho nueve."
    chunks = dividir_en_chunks(texto, max_chars=30, first_chunk_max_chars=30)
    reconstruido = " ".join(chunks)
    assert reconstruido == texto


def test_parte_palabra_larga() -> None:
    palabra = "x" * 95
    chunks = dividir_en_chunks(palabra, max_chars=20, first_chunk_max_chars=20)
    assert len(chunks) == 5
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_parte_por_clausulas() -> None:
    texto = "Primero, segundo, tercero, cuarto, quinto, sexto."
    chunks = dividir_en_chunks(texto, max_chars=25, first_chunk_max_chars=25)
    assert len(chunks) > 1
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_fusiona_chunk_final_corto() -> None:
    chunks = dividir_en_chunks(
        "Una frase completa. Sí.",
        max_chars=40,
        first_chunk_max_chars=40,
        min_chars=10,
    )
    assert chunks == ["Una frase completa. Sí."]


def test_limites_minimos_no_rompen() -> None:
    chunks = dividir_en_chunks("hola mundo", max_chars=1, first_chunk_max_chars=1)
    assert chunks
    assert all(len(chunk) <= 8 for chunk in chunks)
