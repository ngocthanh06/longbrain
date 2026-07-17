FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docsearch.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional doc-search stack (BGE-M3 via sentence-transformers, ~6GB with
# torch). Off by default so memory-only users keep the lean image; enable
# with scripts/enable_doc_search.sh. A lean image with DOC_EMBED_* set does
# NOT crash — document endpoints return 503 doc_embedder_unavailable while
# memory keeps working.
ARG INSTALL_DOC_SEARCH=false
# CPU-only torch index: on amd64 the default index pulls ~5GB of CUDA libs
# that a Docker deployment can never use; the cpu wheel drops them. On
# arm64 (Apple Silicon) wheels are CPU-only anyway — no behavior change.
RUN if [ "$INSTALL_DOC_SEARCH" = "true" ]; then \
      pip install --no-cache-dir -r requirements-docsearch.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Bake the fastembed embedding model into the image so first start needs no
# network and no host-side model management.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
# The optional doc embedder (DOC_EMBED_PROVIDER=huggingface, e.g. BAAI/bge-m3,
# ~2.3GB) is NOT baked in: it downloads on first use into the /data volume so
# container recreation never re-downloads (SEARCH_SPEC constraint 3).
ENV HF_HOME=/data/hf_cache
ARG EMBED_PROVIDER=fastembed
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN if [ "$EMBED_PROVIDER" = "fastembed" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}')"; \
    fi

RUN mkdir -p /data/documents

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
