import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


class ChromaVectorStore:
    def __init__(self, database_path: str):
        Path(database_path).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(
            path=database_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="enterprise_knowledge_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def find_by_hash(self, digest: str) -> dict | None:
        with self._lock:
            rows = self._collection.get(
                where={"file_hash": digest},
                include=["metadatas"],
                limit=1,
            )
        if not rows["ids"]:
            return None
        metadata = rows["metadatas"][0]
        document = self.get_document(metadata["document_id"])
        return document

    def add_document(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        page_count: int,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> dict:
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("Each document chunk must have exactly one embedding")
        uploaded_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._collection.add(
                ids=[f"{document_id}-{chunk['chunk_index']}" for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                embeddings=embeddings,
                metadatas=[
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "file_hash": file_hash,
                        "page_count": page_count,
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "uploaded_at": uploaded_at,
                    }
                    for chunk in chunks
                ],
            )
        return {
            "id": document_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "uploaded_at": uploaded_at,
        }

    def get_document(self, document_id: str) -> dict | None:
        with self._lock:
            rows = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
        if not rows["ids"]:
            return None
        metadata = rows["metadatas"][0]
        return {
            "id": document_id,
            "filename": metadata["filename"],
            "page_count": metadata["page_count"],
            "chunk_count": len(rows["ids"]),
            "uploaded_at": metadata["uploaded_at"],
        }

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._collection.get(include=["metadatas"])
        documents = {}
        for metadata in rows["metadatas"]:
            document_id = metadata["document_id"]
            document = documents.setdefault(
                document_id,
                {
                    "id": document_id,
                    "filename": metadata["filename"],
                    "page_count": metadata["page_count"],
                    "chunk_count": 0,
                    "uploaded_at": metadata["uploaded_at"],
                },
            )
            document["chunk_count"] += 1
        return sorted(
            documents.values(), key=lambda document: document["uploaded_at"], reverse=True
        )

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            rows = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            if not rows["ids"]:
                return False
            self._collection.delete(ids=rows["ids"])
            return True

    def search(self, embedding: list[float], limit: int = 5) -> list[dict]:
        with self._lock:
            count = self._collection.count()
            if not count:
                return []
            rows = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(limit, count),
                include=["documents", "metadatas", "distances"],
            )
        results = []
        for document, metadata, distance in zip(
            rows["documents"][0], rows["metadatas"][0], rows["distances"][0]
        ):
            results.append(
                {
                    "text": document,
                    "filename": metadata["filename"],
                    "document_id": metadata["document_id"],
                    "page_number": metadata["page_number"],
                    "distance": float(distance),
                }
            )
        return results

    def is_ready(self) -> bool:
        self._collection.count()
        return True

    def has_documents(self) -> bool:
        return self._collection.count() > 0
