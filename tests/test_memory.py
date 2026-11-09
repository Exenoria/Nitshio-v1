"""Tests de la memoria ChromaDB con un embedder determinista."""

from __future__ import annotations

import hashlib
from typing import Any

import chromadb
import pytest

from memory.store import MemoryStore, MemoryStoreError


class EmbedderFalso:
    """Embedder de 16 dimensiones basado en hash (sin red)."""

    def embed(self, textos: list[str]) -> list[list[float]]:
        vectores = []
        for texto in textos:
            digest = hashlib.sha256(texto.encode("utf-8")).digest()
            vectores.append([b / 255.0 for b in digest[:16]])
        return vectores

    def embed_query(self, texto: str) -> list[float]:
        return self.embed([texto])[0]


def _config(tmp_path) -> dict[str, Any]:
    return {
        "memory": {
            "persist_dir": str(tmp_path / "chroma"),
            "collections": {
                "memories": "memories",
                "conversations": "conversations",
                "self_knowledge": "self_knowledge",
                "documentation": "documentation",
            },
            "top_k": 5,
            "min_score": 0.0,
            "max_injected_chars": 500,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "models": {"embeddings": "nomic-embed-text:latest"},
        },
    }


@pytest.fixture()
def store(tmp_path):
    memoria = MemoryStore(
        _config(tmp_path),
        embedder=EmbedderFalso(),
        client=chromadb.EphemeralClient(),
    )
    yield memoria
    memoria.close()


def test_guardar_y_contar(store: MemoryStore) -> None:
    identificador = store.remember("A Nitshio le gusta el café")
    assert identificador
    assert store.count("memories") == 1


def test_texto_vacio_falla(store: MemoryStore) -> None:
    with pytest.raises(MemoryStoreError):
        store.remember("   ")


def test_coleccion_desconocida(store: MemoryStore) -> None:
    with pytest.raises(MemoryStoreError, match="desconocida"):
        store.add("otra", "hola")


def test_query_devuelve_hits(store: MemoryStore) -> None:
    store.remember("El usuario prefiere café por la mañana.")
    store.remember("El usuario odia las arañas.")
    hits = store.query("memories", "café")
    assert hits
    assert hits[0].collection == "memories"
    assert 0.0 <= hits[0].score <= 1.0


def test_query_vacia(store: MemoryStore) -> None:
    assert store.query("memories", "   ") == []


def test_add_many(store: MemoryStore) -> None:
    ids = store.add_many("memories", ["uno", "dos", "tres"])
    assert len(ids) == 3
    assert store.count("memories") == 3


def test_contexto_rag(store: MemoryStore) -> None:
    store.remember("Nitshio vive en C:\\Users\\x\\Nitshio.")
    store.remember("El modelo activo es nitshio-v1.")
    contexto = store.build_context("¿dónde vive Nitshio?")
    assert "Recuerdos relevantes" in contexto
    assert "memories" in contexto


def test_contexto_limita_caracteres(store: MemoryStore) -> None:
    store.remember("x" * 400)
    contexto = store.build_context("x", max_chars=60)
    assert len(contexto) <= 60


def test_guardar_conversacion(store: MemoryStore) -> None:
    store.save_conversation("hola", "hmph", emotion="angry")
    assert store.count("conversations") == 1
    hits = store.query("conversations", "hola")
    assert hits[0].metadata["emotion"] == "angry"


def test_eliminar(store: MemoryStore) -> None:
    identificador = store.remember("borrar")
    store.delete("memories", identificador)
    assert store.count("memories") == 0


def test_metadata_personalizada(store: MemoryStore) -> None:
    store.remember("dato", metadata={"tipo": "preferencia"})
    hits = store.query("memories", "dato")
    assert hits[0].metadata["tipo"] == "preferencia"
