from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend import main
from backend.config import Settings
from backend.embeddings import SentenceTransformerEmbedder
from backend.pdf_processing import PDFProcessingError, extract_pdf_pages
from backend.question_answering import NO_ANSWER
from backend.vector_store import ChromaVectorStore


def create_pdf(text: str | None = "Retention period is seven years.") -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class FakeVectorStore:
    def __init__(self):
        self.documents = {}
        self.chunks = []

    @staticmethod
    def content_hash(content):
        import hashlib

        return hashlib.sha256(content).hexdigest()

    def find_by_hash(self, digest):
        for document in self.documents.values():
            if document["file_hash"] == digest:
                return {
                    key: value
                    for key, value in document.items()
                    if key != "file_hash"
                }
        return None

    def add_document(self, document_id, filename, file_hash, page_count, chunks, embeddings):
        document = {
            "id": document_id,
            "filename": filename,
            "file_hash": file_hash,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "uploaded_at": "2026-01-01T00:00:00+00:00",
        }
        self.documents[document_id] = document
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(
                {
                    **chunk,
                    "document_id": document_id,
                    "filename": filename,
                    "distance": 0.1,
                    "embedding": embedding,
                }
            )
        return {key: value for key, value in document.items() if key != "file_hash"}

    def get_document(self, document_id):
        return self.documents.get(document_id)

    def list_documents(self):
        return [
            {key: value for key, value in doc.items() if key != "file_hash"}
            for doc in self.documents.values()
        ]

    def delete_document(self, document_id):
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.chunks = [item for item in self.chunks if item["document_id"] != document_id]
        return True

    def search(self, embedding, limit=5):
        return [{key: value for key, value in item.items() if key != "embedding"}
                for item in self.chunks[:limit]]

    def is_ready(self):
        return True

    def has_documents(self):
        return bool(self.chunks)


class FakeEmbedder:
    def __init__(self, _model_name):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, _text):
        self.query_calls += 1
        return [1.0, 0.0]


class FakeOllama:
    def __init__(self, *_args):
        self.calls = []

    def answer(self, question, passages):
        self.calls.append((question, passages))
        return "Records must be kept for seven years [policy.pdf, p. 1]."


def make_client(tmp_path, monkeypatch, store=None, llm_class=FakeOllama):
    vector_store = store or FakeVectorStore()
    embedder = FakeEmbedder("test-embedder")
    monkeypatch.setattr(main, "ChromaVectorStore", lambda _path: vector_store)
    monkeypatch.setattr(main, "SentenceTransformerEmbedder", lambda _name: embedder)
    monkeypatch.setattr(main, "OllamaClient", llm_class)
    settings = Settings(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="test-embedder",
        ollama_url="http://ollama:11434",
        ollama_model="qwen2.5:1.5b",
        relevance_distance_threshold=1.15,
        max_upload_bytes=1024 * 1024,
    )
    return TestClient(main.create_app(settings)), vector_store, embedder


def test_extract_pdf_pages_returns_page_numbers_and_text():
    pages = extract_pdf_pages(create_pdf())

    assert pages == [{"page_number": 1, "text": "Retention period is seven years."}]


def test_extract_pdf_rejects_bad_or_text_empty_documents():
    with pytest.raises(PDFProcessingError, match="could not be read"):
        extract_pdf_pages(b"not a PDF")
    with pytest.raises(PDFProcessingError, match="No searchable text"):
        extract_pdf_pages(create_pdf(None))


def test_sentence_transformer_embedding_adapter(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            import numpy as np

            assert texts == ["policy text", "question text"]
            assert kwargs["normalize_embeddings"] is True
            return np.array([[0.6, 0.8], [1.0, 0.0]])

    class FakeSentenceTransformer:
        def __init__(self, model_name, device):
            assert model_name == "test-model"
            assert device == "cpu"

        def encode(self, texts, **kwargs):
            return FakeModel().encode(texts, **kwargs)

    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    embedder = SentenceTransformerEmbedder("test-model")
    assert embedder.embed_documents(["policy text", "question text"]) == [
        [0.6, 0.8],
        [1.0, 0.0],
    ]


def test_chroma_persistence_similarity_search_and_deletion(tmp_path):
    store = ChromaVectorStore(str(tmp_path / "chroma"))
    chunks = [
        {"text": "Retention policy records are kept for seven years.", "page_number": 2,
         "chunk_index": 0},
        {"text": "Office locations include New York and London.", "page_number": 3,
         "chunk_index": 1},
    ]
    document = store.add_document(
        "doc-1", "policy.pdf", "hash-1", 4, chunks, [[1.0, 0.0], [0.0, 1.0]]
    )

    reopened = ChromaVectorStore(str(tmp_path / "chroma"))
    results = reopened.search([0.99, 0.01], limit=2)

    assert document["page_count"] == 4
    assert reopened.list_documents()[0]["filename"] == "policy.pdf"
    assert results[0]["text"].startswith("Retention policy")
    assert reopened.delete_document("doc-1") is True
    assert reopened.list_documents() == []


def test_upload_ask_deduplicates_and_persists_document_metadata(tmp_path, monkeypatch):
    client, store, embedder = make_client(tmp_path, monkeypatch)
    pdf = create_pdf()

    with client:
        upload = client.post(
            "/upload", files={"file": ("policy.pdf", pdf, "application/pdf")}
        )
        assert upload.status_code == 201
        assert upload.json()["already_present"] is False
        assert upload.json()["page_count"] == 1
        assert embedder.document_calls == 1

        duplicate = client.post(
            "/upload", files={"file": ("renamed.pdf", pdf, "application/pdf")}
        )
        assert duplicate.status_code == 201
        assert duplicate.json()["already_present"] is True
        assert len(client.get("/documents").json()) == 1
        assert embedder.document_calls == 1

        answer = client.post("/ask", json={"question": "How long are records kept?"})
        assert answer.status_code == 200
        assert "seven years" in answer.json()["answer"]
        assert answer.json()["sources"][0]["filename"] == "policy.pdf"
        assert answer.json()["sources"][0]["page_number"] == 1

        document_id = upload.json()["id"]
        assert client.delete(f"/documents/{document_id}").status_code == 204
        assert client.get("/documents").json() == []
        assert client.post("/ask", json={"question": "What is the policy?"}).json()[
            "answer"
        ] == NO_ANSWER
        assert store.has_documents() is False


def test_invalid_extension_signature_and_empty_pdf_are_reported(tmp_path, monkeypatch):
    client, _, _ = make_client(tmp_path, monkeypatch)
    with client:
        wrong_extension = client.post(
            "/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert wrong_extension.status_code == 400
        bad_pdf = client.post(
            "/upload", files={"file": ("notes.pdf", b"not a PDF", "application/pdf")}
        )
        assert bad_pdf.status_code == 400
        blank_pdf = client.post(
            "/upload",
            files={"file": ("scan.pdf", create_pdf(None), "application/pdf")},
        )
        assert blank_pdf.status_code == 422
        assert "OCR" in blank_pdf.json()["detail"]


def test_question_without_relevant_chunks_does_not_call_llm(tmp_path, monkeypatch):
    client, store, embedder = make_client(tmp_path, monkeypatch)
    with client:
        client.post("/upload", files={"file": ("policy.pdf", create_pdf(), "application/pdf")})
        store.chunks[0]["distance"] = 1.8
        response = client.post("/ask", json={"question": "What is the policy?"})
        assert response.status_code == 200
        assert response.json() == {"answer": NO_ANSWER, "sources": []}


def test_ollama_unavailable_is_reported_and_health_metrics_work(tmp_path, monkeypatch):
    class UnavailableOllama(FakeOllama):
        def answer(self, _question, _passages):
            raise httpx.ConnectError(
                "connection refused", request=httpx.Request("POST", "http://ollama")
            )

    client, _, _ = make_client(tmp_path, monkeypatch, llm_class=UnavailableOllama)
    with client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        client.post("/upload", files={"file": ("policy.pdf", create_pdf(), "application/pdf")})
        unavailable = client.post("/ask", json={"question": "What is the policy?"})
        assert unavailable.status_code == 503
        assert "Ollama" in unavailable.json()["detail"]
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "http_requests_total" in metrics.text
        assert "process_memory_bytes" in metrics.text
        assert "application_uptime_seconds" in metrics.text
