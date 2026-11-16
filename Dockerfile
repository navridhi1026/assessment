FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PATH=/data/chroma \
    HF_HOME=/models/huggingface

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --disable-pip-version-check \
      --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu" \
    && pip install --disable-pip-version-check -r requirements.txt \
    && addgroup --system --gid 10001 assistant \
    && adduser --system --uid 10001 --ingroup assistant assistant \
    && mkdir -p /data/chroma /models/huggingface \
    && chown -R assistant:assistant /data /models /workspace

COPY --chown=assistant:assistant backend ./backend
COPY --chown=assistant:assistant frontend ./frontend
USER 10001:10001
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
