import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import httpx
import psutil
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from backend.chunking import chunk_pages
from backend.config import Settings
from backend.embeddings import SentenceTransformerEmbedder
from backend.llm import OllamaClient
from backend.pdf_processing import PDFProcessingError, extract_pdf_pages
from backend.question_answering import QuestionAnsweringService
from backend.vector_store import ChromaVectorStore


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        standard = logging.makeLogRecord({}).__dict__
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in standard and key not in payload
            }
        )
        return json.dumps(payload, ensure_ascii=True, default=str)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
for handler in logging.getLogger().handlers:
    handler.setFormatter(JsonFormatter())
logger = logging.getLogger("knowledge-assistant")

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests received", ["method", "route", "status"]
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "route"]
)
APP_START_TIME = time.time()
APP_UPTIME = Gauge("application_uptime_seconds", "Application uptime in seconds")
PROCESS_CPU = Gauge("process_cpu_percent", "Application process CPU utilization")
PROCESS_MEMORY = Gauge("process_memory_bytes", "Application process resident memory")
PROCESS = psutil.Process()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        vector_store = ChromaVectorStore(app_settings.chroma_path)
        embedder = SentenceTransformerEmbedder(app_settings.embedding_model)
        llm = OllamaClient(app_settings.ollama_url, app_settings.ollama_model)
        application.state.vector_store = vector_store
        application.state.embedder = embedder
        application.state.qa = QuestionAnsweringService(
            vector_store,
            embedder,
            llm,
            app_settings.relevance_distance_threshold,
        )
        logger.info("application_started")
        yield
        logger.info("application_stopped")

    application = FastAPI(
        title="Enterprise AI Knowledge Assistant",
        description="Semantic, source-grounded PDF question answering with local Ollama.",
        version="2.0.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def observe_requests(request: Request, call_next):
        started = time.perf_counter()
        route = request.scope.get("route")
        route_name = route.path if route else request.url.path
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - started
            HTTP_REQUESTS.labels(request.method, route_name, "500").inc()
            HTTP_DURATION.labels(request.method, route_name).observe(elapsed)
            logger.exception(
                "request_failed", extra={"method": request.method, "path": route_name}
            )
            raise

        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, route_name, str(response.status_code)).inc()
        HTTP_DURATION.labels(request.method, route_name).observe(elapsed)
        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": route_name,
                "status_code": response.status_code,
                "duration_seconds": round(elapsed, 4),
            },
        )
        return response

    @application.get("/health")
    @application.get("/healthz", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @application.get("/ready")
    @application.get("/readyz", include_in_schema=False)
    def readiness(request: Request):
        request.app.state.vector_store.is_ready()
        return {"status": "ready"}

    @application.get("/metrics", include_in_schema=False)
    def metrics():
        APP_UPTIME.set(time.time() - APP_START_TIME)
        PROCESS_CPU.set(PROCESS.cpu_percent())
        PROCESS_MEMORY.set(PROCESS.memory_info().rss)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.post("/upload", status_code=201)
    @application.post("/api/documents", status_code=201, include_in_schema=False)
    def upload_document(request: Request, file: UploadFile = File(...)):
        raw_filename = (file.filename or "").replace("\\", "/")
        filename = PurePosixPath(raw_filename).name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Upload a file with a .pdf extension.")

        content = file.file.read(app_settings.max_upload_bytes + 1)
        if len(content) > app_settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds the {app_settings.max_upload_bytes // (1024 * 1024)} MiB upload limit.",
            )
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        try:
            pages = extract_pdf_pages(content)
        except PDFProcessingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        chunks = chunk_pages(pages)
        store = request.app.state.vector_store
        file_hash = store.content_hash(content)
        existing = store.find_by_hash(file_hash)
        if existing:
            return {**existing, "already_present": True}

        try:
            vectors = request.app.state.embedder.embed_documents(
                [chunk["text"] for chunk in chunks]
            )
            document = store.add_document(
                str(uuid.uuid4()),
                filename,
                file_hash,
                len(pages),
                chunks,
                vectors,
            )
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("document_indexing_failed")
            raise HTTPException(
                status_code=503, detail="Could not create document embeddings."
            ) from error
        logger.info(
            "document_indexed",
            extra={
                "document_id": document["id"],
                "filename": filename,
                "page_count": document["page_count"],
                "chunk_count": document["chunk_count"],
            },
        )
        return {**document, "already_present": False}

    @application.get("/documents")
    @application.get("/api/documents", include_in_schema=False)
    def list_documents(request: Request):
        return request.app.state.vector_store.list_documents()

    @application.delete("/documents/{document_id}", status_code=204)
    @application.delete(
        "/api/documents/{document_id}", status_code=204, include_in_schema=False
    )
    def delete_document(document_id: str, request: Request):
        if not request.app.state.vector_store.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return Response(status_code=204)

    @application.post("/ask")
    @application.post("/api/ask", include_in_schema=False)
    def ask_question(payload: QuestionRequest, request: Request):
        if not request.app.state.vector_store.is_ready():
            raise HTTPException(status_code=503, detail="Vector database is unavailable.")
        try:
            result = request.app.state.qa.ask(payload.question)
        except httpx.HTTPError as error:
            logger.exception("ollama_request_failed")
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not reach Ollama at {app_settings.ollama_url}. "
                    "Start Ollama and make sure the configured model is available."
                ),
            ) from error
        except ValueError as error:
            logger.exception("question_answering_failed")
            raise HTTPException(status_code=502, detail=str(error)) from error
        logger.info(
            "question_answered",
            extra={"source_count": len(result["sources"])},
        )
        return result

    return application


app = create_app()
