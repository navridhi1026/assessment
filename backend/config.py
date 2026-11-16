import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    chroma_path: str
    embedding_model: str
    ollama_url: str
    ollama_model: str
    relevance_distance_threshold: float
    max_upload_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            chroma_path=os.getenv("CHROMA_PATH", "./data/chroma"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            relevance_distance_threshold=float(
                os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "1.15")
            ),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        )
