"""Embeddings locales a través de Ollama.

ChromaDB solo almacena vectores; este módulo los calcula con el modelo
configurado (``nomic-embed-text`` por defecto). Se usa la API nueva
``/api/embed`` y se cae a ``/api/embeddings`` en versiones antiguas.
"""

from __future__ import annotations

from typing import Any, Protocol

import requests


class EmbeddingError(RuntimeError):
    """No se pudieron calcular embeddings."""


class Embedder(Protocol):
    """Contrato mínimo para cualquier proveedor de embeddings."""

    def embed(self, textos: list[str]) -> list[list[float]]: ...

    def embed_query(self, texto: str) -> list[float]: ...


class OllamaEmbeddings:
    """Calcula embeddings con un modelo de Ollama."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text:latest",
        *,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = session or requests.Session()
        self.dimension: int | None = None

    def embed(self, textos: list[str]) -> list[list[float]]:
        if not textos:
            return []
        limpios = [str(t) for t in textos]
        try:
            respuesta = self.session.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": limpios},
                timeout=self.timeout,
            )
            if respuesta.status_code == 404:
                return [self._embed_antiguo(texto) for texto in limpios]
            respuesta.raise_for_status()
            datos = respuesta.json()
            vectores = datos.get("embeddings")
            if isinstance(vectores, list) and len(vectores) == len(limpios):
                self.dimension = len(vectores[0]) if vectores else None
                return [[float(v) for v in vector] for vector in vectores]
            raise EmbeddingError("Ollama /api/embed no devolvió embeddings.")
        except requests.RequestException as exc:
            raise EmbeddingError(f"Fallo de embeddings en Ollama: {exc}") from exc

    def _embed_antiguo(self, texto: str) -> list[float]:
        try:
            respuesta = self.session.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": texto},
                timeout=self.timeout,
            )
            respuesta.raise_for_status()
            vector = respuesta.json().get("embedding") or []
            if not vector:
                raise EmbeddingError("Ollama /api/embeddings devolvió un vector vacío.")
            self.dimension = len(vector)
            return [float(v) for v in vector]
        except requests.RequestException as exc:
            raise EmbeddingError(f"Fallo de embeddings en Ollama: {exc}") from exc

    def embed_query(self, texto: str) -> list[float]:
        vectores = self.embed([texto])
        if not vectores:
            raise EmbeddingError("No se generó embedding para la consulta.")
        return vectores[0]


def crear_embeddings(config: dict[str, Any]) -> OllamaEmbeddings:
    """Crea el embedder configurado en ``config.memory``/``config.ollama``."""
    ollama_cfg = config.get("ollama") or {}
    modelos = ollama_cfg.get("models") or {}
    host = str(ollama_cfg.get("host", "http://localhost:11434"))
    modelo = str(modelos.get("embeddings", "nomic-embed-text:latest"))
    return OllamaEmbeddings(
        host=host,
        model=modelo,
        timeout=float(ollama_cfg.get("embedding_timeout_seconds", 60)),
    )
