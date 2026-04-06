"""Embedder wrapper — local HuggingFace sentence-transformers, no API key required."""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import Settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance configured from Settings.

    Downloads the model on first call and caches it locally via HuggingFace's
    model cache (~/.cache/huggingface). Subsequent calls reuse the in-memory
    instance — no repeated downloads or re-initialisation.

    Returns:
        Configured HuggingFaceEmbeddings instance (all-MiniLM-L6-v2 by default).
    """
    settings = Settings()
    log.info("loading embedding model '%s' (first call only)", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
