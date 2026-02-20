"""
HMMWV Technical Assistant — Vector Store
==========================================
ChromaDB-backed semantic search for RAG retrieval.
"""

import logging
import sys
from pathlib import Path

import importlib.util

if "config" not in sys.modules:
    _cfg = Path(__file__).resolve().parent.parent / "config.py"
    _sp = importlib.util.spec_from_file_location("config", str(_cfg))
    _m = importlib.util.module_from_spec(_sp)
    sys.modules["config"] = _m
    _sp.loader.exec_module(_m)

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages the ChromaDB vector store for HMMWV technical content."""

    def __init__(self):
        self._client = None
        self._collection = None

    # ── Initialization ─────────────────────────────────────────────────────

    def initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(CHROMA_PERSIST_DIR),
                    anonymized_telemetry=False,
                )
            )
        except TypeError:
            # Newer versions of chromadb use different settings
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_PERSIST_DIR),
            )

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(f"Vector store initialized. Collection '{COLLECTION_NAME}' has {count} documents.")
        return count

    @property
    def collection(self):
        if self._collection is None:
            self.initialize()
        return self._collection

    # ── Indexing ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Add text chunks to the vector store.
        Each chunk must have 'text' and 'metadata' keys.
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Create a deterministic ID based on content
            source = chunk["metadata"].get("source_file", "unknown")
            page = chunk["metadata"].get("page_number", 0)
            chunk_idx = chunk["metadata"].get("chunk_index", i)
            doc_id = f"{source}__p{page}__c{chunk_idx}"

            ids.append(doc_id)
            documents.append(chunk["text"])

            # ChromaDB metadata must be flat (str, int, float, bool)
            flat_meta = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
                else:
                    flat_meta[k] = str(v)
            metadatas.append(flat_meta)

        # Upsert in batches to avoid memory issues
        batch_size = 100
        added = 0
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            added += min(batch_size, len(ids) - start)

        logger.info(f"Added/updated {added} chunks in vector store")
        return added

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = TOP_K_RESULTS, where: dict = None) -> list[dict]:
        """
        Semantic search over the indexed content.
        Returns list of results with text, metadata, and relevance score.
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

        # Flatten results into a cleaner format
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "id": results["ids"][0][i] if results["ids"] else "",
                })
        return output

    def search_by_source(self, query: str, source_file: str, n_results: int = 5) -> list[dict]:
        """Search within a specific source PDF."""
        return self.search(query, n_results=n_results, where={"source_file": source_file})

    # ── Management ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        # Get unique source files
        try:
            sample = self.collection.get(limit=min(count, 1000), include=["metadatas"])
            sources = set()
            for meta in (sample["metadatas"] or []):
                if "source_file" in meta:
                    sources.add(meta["source_file"])
        except Exception:
            sources = set()

        return {
            "total_chunks": count,
            "source_files": sorted(sources),
            "num_sources": len(sources),
        }

    def clear(self):
        """Clear the entire collection (use with caution)."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")

    def delete_source(self, source_file: str):
        """Remove all chunks from a specific source PDF."""
        try:
            self.collection.delete(where={"source_file": source_file})
            logger.info(f"Deleted all chunks from {source_file}")
        except Exception as e:
            logger.error(f"Error deleting source {source_file}: {e}")
