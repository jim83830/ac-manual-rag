"""Chroma 向量資料庫的讀寫。embedding 由我們自己算好傳進來，不用 Chroma 內建模型。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from rag.chunk import Chunk

COLLECTION = "manual_chunks"
_NO_PAGE = -1  # Chroma metadata 不能存 None


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    distance: float


def _to_metadata(chunk: Chunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "doc_name": chunk.doc_name,
        "pdf_page": chunk.pdf_page,
        "printed_page": chunk.printed_page if chunk.printed_page is not None else _NO_PAGE,
        "section": chunk.section,
    }


def _from_record(chunk_id: str, text: str, meta: dict) -> Chunk:
    printed = int(meta["printed_page"])
    return Chunk(
        id=chunk_id,
        doc_id=meta["doc_id"],
        doc_name=meta["doc_name"],
        pdf_page=int(meta["pdf_page"]),
        printed_page=None if printed == _NO_PAGE else printed,
        section=meta["section"],
        text=text,
    )


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def replace_doc(self, doc_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 與 embeddings 數量不一致")
        self._collection.delete(where={"doc_id": doc_id})
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )

    def query(self, embedding: list[float], k: int, doc_id: str | None = None) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, total),
            where={"doc_id": doc_id} if doc_id else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(chunk=_from_record(chunk_id, text, meta), distance=float(distance))
            for chunk_id, text, meta, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def count(self) -> int:
        return self._collection.count()
