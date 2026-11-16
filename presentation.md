---
marp: true
theme: default
paginate: true
title: Enterprise AI Knowledge Assistant
---

# Enterprise AI Knowledge Assistant

**Local-first PDF question answering with semantic retrieval**

Technical assessment · RAG · Docker · Kubernetes · Observability

---

## Problem and outcome

- Enterprise policies and references are difficult to search across long PDF documents.
- This application indexes searchable PDF content and responds to natural-language questions.
- Answers include the source PDF, page number, and retrieved excerpt.
- The application runs with local embeddings and a local Ollama language model; no paid API is required.

---

## Architecture

```text
Streamlit UI ── REST ── FastAPI ── PDF extraction ── chunking
                                  │                       │
                                  └─ Sentence Transformers embeddings
                                                           │
                                                    Persistent ChromaDB
                                                           │
Question ── embedding ── semantic similarity search ── source passages
                                                           │
                                                   Ollama qwen2.5:1.5b

FastAPI /metrics ── Prometheus ── Grafana
JSON container logs ── Promtail ── Loki ── Grafana
```

---

## RAG request lifecycle

1. Validate the PDF signature/size and extract text page by page.
2. Split text into 900-character passages with 150-character overlap.
3. Embed passages using local `all-MiniLM-L6-v2` and store text, vectors, and source metadata in ChromaDB.
4. Embed each question and retrieve the closest passages by cosine similarity.
5. Reject passages beyond the configured relevance threshold.
6. Ask Ollama to answer from retrieved context only; return a clear no-evidence result when no passage is relevant.

---

## Source review

- `backend/pdf_processing.py`: parser and scanned/empty PDF errors.
- `backend/chunking.py`: chunk boundaries and page metadata.
- `backend/embeddings.py`: local model adapter, loaded on first use.
- `backend/vector_store.py`: persistent Chroma collection and similarity search.
- `backend/llm.py`: context-only Ollama prompt and citation instructions.
- `backend/question_answering.py`: retrieval, relevance cutoff, and answer routing.
- `backend/main.py`: validated REST API, JSON request logs, health and metrics.
- `frontend/streamlit_app.py`: upload, document management, conversation, and cited source UI.

---

## Docker and Kubernetes

- Docker Compose runs separate backend/frontend services, Ollama plus a model-pull job, Prometheus, Loki, Promtail, and Grafana.
- Named volumes persist Chroma vectors/documents, model cache, Ollama model, logs, metrics, and dashboard state.
- Kubernetes manifests include application and Ollama deployments/services, PVCs, probes, resource limits, and observability.
- The backend uses the in-cluster Ollama address `http://ollama:11434`; it never assumes pod `localhost` is the laptop.
- CI validates and builds the image; cluster deployment is gated by explicit GitHub configuration.

---

## Dashboard and logs

- Prometheus scrapes request counts, status codes, latency, process uptime, CPU, and memory.
- Grafana provisions Prometheus/Loki sources and the assessment dashboard.
- Promtail discovers Docker logs locally and Kubernetes pod logs in the cluster.
- Review flow: issue an API question → inspect request/latency panels → query backend logs in Grafana Explore.

---

## Live demo walkthrough

1. Run `docker compose up --build -d` and open `http://localhost:8501`.
2. Upload a searchable PDF and wait for local embedding generation.
3. Ask a document-specific question; expand citations and confirm file/page/excerpt.
4. Ask something absent from the source; confirm the explicit no-evidence response.
5. Remove the PDF and verify it is no longer listed or retrieved.
6. Open Grafana at `http://localhost:3000`; check metrics and Loki Explore.
7. Verify Kubernetes rollouts only when a local cluster is enabled.

---

## Limitations and production next steps

- Scanned PDFs need OCR; current extraction handles text embedded in PDFs.
- A single Chroma instance is intended for a local assessment, not horizontal API scaling.
- Authentication, tenant isolation, governed document retention, and production network controls are not included.
- Loki and Promtail configuration requires security, access, and retention review before production.
- Production follow-up: access control, audit trail, retrieval evaluation, backup/restore, durable HA observability, and model-governance controls.
