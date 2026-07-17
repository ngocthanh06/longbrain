"""Public stateless model-runtime API contracts."""

import pytest
from fastapi import HTTPException

from app import config
from app import main as service


class BatchEmbed:
    def get_text_embedding_batch(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class CompletionResult:
    text = "answer"


class StatelessLLM:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return CompletionResult()


@pytest.fixture(autouse=True)
def restore_runtime_state():
    previous = dict(service.state)
    yield
    service.state.clear()
    service.state.update(previous)


def test_embeddings_document_profile_returns_fingerprint(monkeypatch):
    monkeypatch.setattr(config, "DOC_EMBED_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "DOC_EMBED_MODEL", "example/doc-model")
    service.state.update(doc_embed_model=BatchEmbed(), doc_embed_dim=2)

    result = service.embeddings(service.EmbeddingsRequest(
        profile="document", texts=["one", "two"],
    ))

    assert result["vectors"] == [[3.0, 1.0], [3.0, 1.0]]
    assert result["fingerprint"] == "huggingface:example/doc-model:2"


def test_embeddings_fail_closed_for_cloud_provider(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "openai")
    monkeypatch.setattr(config, "EMBED_MODEL", "example-cloud-model")
    service.state.update(embed_model=BatchEmbed(), embed_dim=2)

    with pytest.raises(HTTPException) as exc:
        service.embeddings(service.EmbeddingsRequest(texts=["private text"]))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "local_provider_required"


def test_embeddings_reject_oversized_batch(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "fastembed")
    service.state.update(embed_model=BatchEmbed(), embed_dim=2)

    with pytest.raises(HTTPException) as exc:
        service.embeddings(service.EmbeddingsRequest(
            texts=["x" * (service._MAX_EMBED_TEXT_CHARS + 1)],
        ))
    assert exc.value.status_code == 413


def test_completion_is_stateless_and_forwards_generation_options(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LLM_MODEL", "example-local-model")
    llm = StatelessLLM()
    service.state.update(llm=llm)

    result = service.completion(service.CompletionRequest(
        prompt="Use only supplied passages", temperature=0.2, max_tokens=256,
    ))

    assert result == {
        "text": "answer",
        "provider": "ollama",
        "model": "example-local-model",
    }
    assert llm.calls == [("Use only supplied passages", {
        "temperature": 0.2, "max_tokens": 256,
    })]


def test_completion_fail_closed_for_cloud_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    service.state.update(llm=StatelessLLM())

    with pytest.raises(HTTPException) as exc:
        service.completion(service.CompletionRequest(prompt="private passages"))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "local_provider_required"


def test_completion_unavailable_without_llm(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    service.state.update(llm=None)

    with pytest.raises(HTTPException) as exc:
        service.completion(service.CompletionRequest(prompt="question"))
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "completion_unavailable"
