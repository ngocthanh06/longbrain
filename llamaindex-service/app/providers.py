"""Provider factories for the embedding model and the (optional) LLM.

Adapters are imported lazily so only the configured provider's SDK needs to
be importable at runtime.
"""

from app import config


def build_embed_model():
    if config.EMBED_PROVIDER == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        return FastEmbedEmbedding(model_name=config.EMBED_MODEL)
    if config.EMBED_PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL
        )
    if config.EMBED_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=config.EMBED_MODEL)
    if config.EMBED_PROVIDER == "nvidia":
        from llama_index.embeddings.nvidia import NVIDIAEmbedding

        return NVIDIAEmbedding(model=config.EMBED_MODEL)
    raise RuntimeError(
        f"Unknown EMBED_PROVIDER={config.EMBED_PROVIDER!r} "
        "(expected fastembed | ollama | openai | nvidia)"
    )


def build_llm():
    """Return the configured LLM, or None when LLM_PROVIDER=none."""
    if config.LLM_PROVIDER in ("none", ""):
        return None
    if config.LLM_PROVIDER == "anthropic":
        from llama_index.llms.anthropic import Anthropic

        return Anthropic(model=config.LLM_MODEL)
    if config.LLM_PROVIDER == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=config.LLM_MODEL)
    if config.LLM_PROVIDER == "nvidia":
        # NIM is OpenAI-compatible; OpenAILike works with ANY model name,
        # unlike the NVIDIA adapter whose static catalog lags new models.
        import os

        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=config.LLM_MODEL,
            api_base=config.NVIDIA_BASE_URL,
            api_key=os.getenv("NVIDIA_API_KEY", ""),
            is_chat_model=True,
            context_window=128000,
            timeout=config.LLM_REQUEST_TIMEOUT,
        )
    if config.LLM_PROVIDER == "gemini":
        from llama_index.llms.gemini import Gemini

        return Gemini(model=config.LLM_MODEL)  # reads GOOGLE_API_KEY
    if config.LLM_PROVIDER == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={config.LLM_PROVIDER!r} "
        "(expected none | anthropic | openai | nvidia | gemini | ollama)"
    )
