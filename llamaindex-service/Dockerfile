FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the fastembed embedding model into the image so first start needs no
# network and no host-side model management.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
ARG EMBED_PROVIDER=fastembed
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN if [ "$EMBED_PROVIDER" = "fastembed" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}')"; \
    fi

RUN mkdir -p /data/documents

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
