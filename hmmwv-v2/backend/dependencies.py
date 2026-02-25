"""
Shared singleton instances for FastAPI dependency injection.
VectorStore and PDFProcessor are initialized once at startup (lifespan).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singletons — initialized in main.py lifespan
_vector_store = None
_pdf_processor = None


def get_vector_store():
    """Return the initialized VectorStore singleton."""
    if _vector_store is None:
        raise RuntimeError("VectorStore not initialized. Call init_singletons() first.")
    return _vector_store


def get_pdf_processor():
    """Return the initialized PDFProcessor singleton."""
    if _pdf_processor is None:
        raise RuntimeError("PDFProcessor not initialized. Call init_singletons() first.")
    return _pdf_processor


def init_singletons():
    """
    Initialize VectorStore and PDFProcessor once at startup.
    Called from FastAPI lifespan context manager in main.py.
    Loading the 24MB BM25 index takes ~1-2s — done once, not per request.
    """
    global _vector_store, _pdf_processor

    from core.vector_store import VectorStore
    from core.pdf_processor import PDFProcessor

    logger.info("Initializing VectorStore (loading BM25 index)…")
    _vector_store = VectorStore()
    n = _vector_store.initialize()
    logger.info(f"VectorStore ready: {n} chunks loaded")

    logger.info("Initializing PDFProcessor…")
    _pdf_processor = PDFProcessor()
    logger.info("PDFProcessor ready")
