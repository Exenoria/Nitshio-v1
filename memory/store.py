"""Memoria persistente con ChromaDB y recuperación para RAG.

Las colecciones separan ciclos de vida distintos:

- ``memories``: hechos y preferencias del usuario.
- ``conversations``: turnos de conversación.
- ``self_knowledge``: código y estructura del proyecto.
- ``documentation``: documentación indexada.

Los embeddings se calculan fuera de Chroma (con Ollama) para no depender de
su función de embeddings por defecto ni descargar modelos ocultos.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from memory.embeddings import Embedder, EmbeddingError, crear_embeddings

ROOT = Path(__file__).resolve().parents[1]


class MemoryStoreError(RuntimeError):
    """No se pudo abrir o consultar la memoria."""


@dataclass(frozen=True)
class MemoryHit:
    """Resultado de una búsqueda semántica."""

    id: str
    text: str
    score: float
    collection: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Almacén de recuerdos con búsqueda por similitud."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        embedder: Embedder | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        memory_cfg = config.get("memory") or {}
        self.persist_dir = self._ruta(
            memory_cfg.get("persist_dir", "data/chroma")
        )
        self.collections: dict[str, str] = dict(
            memory_cfg.get("collections") or {}
        )
        if not self.collections:
            self.collections = {
                "memories": "memories",
                "conversations": "conversations",
                "self_knowledge": "self_knowledge",
                "documentation": "documentation",
            }
        self.top_k = int(memory_cfg.get("top_k", 5))
        self.min_score = float(memory_cfg.get("min_score", 0.2))
        self.max_injected_chars = int(memory_cfg.get("max_injected_chars", 2500))
        self.embedder = embedder or crear_embeddings(config)
        self._client = client
        self._colecciones: dict[str, Any] = {}

    @staticmethod
    def _ruta(valor: str | Path) -> Path:
        path = Path(valor)
        return path if path.is_absolute() else ROOT / path

    # ------------------------------------------------------------------
    # Cliente y colecciones
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover
                raise MemoryStoreError(
                    "chromadb no está instalado. Instala requirements.txt."
                ) from exc
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            except Exception as exc:
                raise MemoryStoreError(f"No se pudo abrir ChromaDB: {exc}") from exc
        return self._client

    def _collection(self, clave: str) -> Any:
        nombre = self.collections.get(clave)
        if not nombre:
            raise MemoryStoreError(
                f"Colección desconocida: {clave!r}. "
                f"Configuradas: {', '.join(self.collections)}"
            )
        if nombre in self._colecciones:
            return self._colecciones[nombre]
        try:
            coleccion = self.client.get_or_create_collection(name=nombre)
        except Exception as exc:
            raise MemoryStoreError(
                f"No se pudo abrir la colección {nombre!r}: {exc}"
            ) from exc
        self._colecciones[nombre] = coleccion
        return coleccion

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------
    def add(
        self,
        collection: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> str:
        """Añade un texto y devuelve su id."""
        texto = " ".join(str(text or "").split()).strip()
        if not texto:
            raise MemoryStoreError("No se puede guardar un recuerdo vacío.")
        try:
            vector = self.embedder.embed([texto])[0]
        except (EmbeddingError, IndexError) as exc:
            raise MemoryStoreError(f"No se pudo calcular el embedding: {exc}") from exc
        identificador = id or uuid.uuid4().hex
        meta = dict(metadata or {})
        meta.setdefault("collection", collection)
        try:
            self._collection(collection).add(
                ids=[identificador],
                documents=[texto],
                embeddings=[vector],
                metadatas=[meta],
            )
        except Exception as exc:
            raise MemoryStoreError(f"No se pudo guardar en memoria: {exc}") from exc
        return identificador

    def add_many(
        self,
        collection: str,
        texts: Iterable[str],
        *,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Añade varios textos en lote."""
        lista = [str(t) for t in texts if str(t).strip()]
        if not lista:
            return []
        try:
            vectores = self.embedder.embed(lista)
        except EmbeddingError as exc:
            raise MemoryStoreError(f"No se pudieron calcular embeddings: {exc}") from exc
        ids = [uuid.uuid4().hex for _ in lista]
        metas = []
        for indice, _ in enumerate(lista):
            meta = dict((metadatas or [])[indice] if metadatas else {})
            meta.setdefault("collection", collection)
            metas.append(meta)
        try:
            self._collection(collection).add(
                ids=ids,
                documents=lista,
                embeddings=vectores,
                metadatas=metas,
            )
        except Exception as exc:
            raise MemoryStoreError(f"No se pudo guardar el lote: {exc}") from exc
        return ids

    def remember(
        self, text: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        """Guarda un hecho o preferencia en ``memories``."""
        return self.add("memories", text, metadata=metadata)

    def save_conversation(
        self,
        user_text: str,
        assistant_text: str,
        *,
        emotion: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Guarda un turno completo en ``conversations``."""
        texto = f"Usuario: {user_text}\nNitshio: {assistant_text}"
        meta = dict(metadata or {})
        meta.update({"emotion": emotion, "type": "turn"})
        return self.add("conversations", texto, metadata=meta)

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------
    def query(
        self,
        collection: str,
        text: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[MemoryHit]:
        """Busca los recuerdos más similares a ``text``."""
        consulta = " ".join(str(text or "").split()).strip()
        if not consulta:
            return []
        try:
            vector = self.embedder.embed_query(consulta)
        except EmbeddingError as exc:
            raise MemoryStoreError(f"No se pudo calcular la consulta: {exc}") from exc
        limite = max(1, int(top_k or self.top_k))
        try:
            resultado = self._collection(collection).query(
                query_embeddings=[vector],
                n_results=limite,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise MemoryStoreError(f"No se pudo consultar memoria: {exc}") from exc

        ids = (resultado.get("ids") or [[]])[0]
        documentos = (resultado.get("documents") or [[]])[0]
        metadatos = (resultado.get("metadatas") or [[]])[0]
        distancias = (resultado.get("distances") or [[]])[0]
        hits: list[MemoryHit] = []
        for indice, identificador in enumerate(ids):
            documento = documentos[indice] if indice < len(documentos) else ""
            distancia = float(distancias[indice]) if indice < len(distancias) else 1.0
            # Distancia coseno/L2 -> score aproximado en (0, 1].
            score = 1.0 / (1.0 + max(0.0, distancia))
            if score < self.min_score:
                continue
            meta = metadatos[indice] if indice < len(metadatos) else {}
            hits.append(
                MemoryHit(
                    id=str(identificador),
                    text=str(documento),
                    score=round(score, 4),
                    collection=collection,
                    metadata=dict(meta or {}),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def build_context(
        self,
        text: str,
        *,
        collections: Iterable[str] | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Construye el bloque de contexto RAG para inyectar en el prompt."""
        claves = list(collections or ("memories", "conversations", "self_knowledge"))
        vistos: set[str] = set()
        todos: list[MemoryHit] = []
        for clave in claves:
            try:
                for hit in self.query(clave, text):
                    if hit.id not in vistos:
                        vistos.add(hit.id)
                        todos.append(hit)
            except MemoryStoreError:
                continue
        if not todos:
            return ""
        todos.sort(key=lambda hit: hit.score, reverse=True)
        limite = int(max_chars or self.max_injected_chars)
        lineas: list[str] = ["Recuerdos relevantes:"]
        usados = len(lineas[0])
        for hit in todos:
            linea = f"- ({hit.collection}, {hit.score:.2f}) {hit.text}"
            if usados + len(linea) + 1 > limite:
                break
            lineas.append(linea)
            usados += len(linea) + 1
        return "\n".join(lineas) if len(lineas) > 1 else ""

    def count(self, collection: str) -> int:
        """Cuenta documentos de una colección."""
        try:
            return int(self._collection(collection).count())
        except Exception as exc:
            raise MemoryStoreError(f"No se pudo contar {collection}: {exc}") from exc

    def delete(self, collection: str, id: str) -> None:
        """Elimina un documento por id."""
        try:
            self._collection(collection).delete(ids=[id])
        except Exception as exc:
            raise MemoryStoreError(f"No se pudo eliminar el recuerdo: {exc}") from exc

    def close(self) -> None:
        """Cierra el cliente de Chroma si lo permite."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._colecciones.clear()
