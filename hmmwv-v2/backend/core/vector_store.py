"""
VectorStore — BM25 search engine extracted from app.py, Streamlit-free.
Loads from the existing chroma_db/vector_store.json index.
"""

import math
import re
import json
import logging
from collections import Counter
from pathlib import Path

from config import (
    CHROMA_PERSIST_DIR, TOP_K_RESULTS,
    BM25_K1, BM25_B, MIN_SIMILARITY,
    _STOP_WORDS, _expand_query,
)

logger = logging.getLogger(__name__)


class VectorStore:
    _STORE_FILE = "vector_store.json"

    def __init__(self):
        self._documents:   list  = []
        self._metadatas:   list  = []
        self._ids:         list  = []
        self._idf:         dict  = {}
        self._doc_tf:      list  = []
        self._doc_lengths: list  = []
        self._avgdl:       float = 1.0
        self._initialized = False

    def _store_path(self) -> Path:
        return CHROMA_PERSIST_DIR / self._STORE_FILE

    def _save(self):
        self._store_path().write_text(json.dumps(
            {"documents": self._documents, "metadatas": self._metadatas, "ids": self._ids}
        ))

    def _load(self):
        p = self._store_path()
        if p.exists():
            data = json.loads(p.read_text())
            self._documents = data.get("documents", [])
            self._metadatas = data.get("metadatas", [])
            self._ids       = data.get("ids", [])

    @staticmethod
    def _tokenize(text: str) -> list:
        """Tokenise to lowercase alphanumeric tokens ≥2 chars, removing stop words."""
        tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS]

    def _build_index(self):
        """
        Build a BM25 index from self._documents.
        BM25 score(D, Q) = Σ IDF(q) * tf(q,D)*(k1+1) / (tf(q,D) + k1*(1-b+b*|D|/avgdl))
        IDF(q) = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)   [Robertson IDF]
        """
        if not self._documents:
            self._idf, self._doc_tf, self._doc_lengths, self._avgdl = {}, [], [], 1.0
            return

        doc_tokens        = [self._tokenize(d) for d in self._documents]
        n_docs            = len(doc_tokens)
        self._doc_tf      = [Counter(t) for t in doc_tokens]
        self._doc_lengths = [len(t) for t in doc_tokens]
        self._avgdl       = sum(self._doc_lengths) / n_docs

        df: Counter = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))

        max_df = max(1, int(n_docs * 0.85))

        self._idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
            if freq <= max_df
        }

    def initialize(self):
        self._load()
        self._build_index()
        self._initialized = True
        logger.info(f"VectorStore initialized: {len(self._documents)} chunks")
        return len(self._documents)

    def add_chunks(self, chunks: list) -> int:
        existing = set(self._ids)
        added = 0
        for i, chunk in enumerate(chunks):
            src   = chunk["metadata"].get("source_file", "unknown")
            page  = chunk["metadata"].get("page_number", 0)
            cidx  = chunk["metadata"].get("chunk_index", i)
            doc_id = f"{src}__p{page}__c{cidx}"
            if doc_id in existing:
                continue
            self._ids.append(doc_id)
            self._documents.append(chunk["text"])
            self._metadatas.append({
                k: v if isinstance(v, (str, int, float, bool)) else str(v)
                for k, v in chunk["metadata"].items()
            })
            added += 1
        if added > 0:
            self._save()
            self._build_index()
        return added

    def search(self, query: str, n_results: int = TOP_K_RESULTS, where: dict = None) -> list:
        """
        BM25 retrieval with query expansion and minimum-similarity filtering.
        Returns results ranked by BM25 score, normalised to [0, 1].
        """
        if not self._documents:
            return []

        expanded_query = _expand_query(query)
        query_terms    = set(self._tokenize(expanded_query))

        scores = []
        k1, b  = BM25_K1, BM25_B
        avgdl  = self._avgdl

        for idx, doc_tf in enumerate(self._doc_tf):
            if where and not all(self._metadatas[idx].get(k) == v for k, v in where.items()):
                continue

            dl    = self._doc_lengths[idx]
            score = 0.0
            for term in query_terms:
                idf = self._idf.get(term, 0.0)
                if idf <= 0.0:
                    continue
                tf_val = doc_tf.get(term, 0)
                if tf_val == 0:
                    continue
                numerator   = tf_val * (k1 + 1.0)
                denominator = tf_val + k1 * (1.0 - b + b * dl / avgdl)
                score      += idf * numerator / denominator

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        max_score = scores[0][1] if scores and scores[0][1] > 0 else 1.0
        for idx, raw_score in scores[:n_results * 2]:
            normalised = raw_score / max_score
            if normalised < MIN_SIMILARITY:
                break
            results.append({
                "text":     self._documents[idx],
                "metadata": self._metadatas[idx],
                "distance": 1.0 - normalised,
                "id":       self._ids[idx],
            })
            if len(results) == n_results:
                break

        return results

    def get_stats(self) -> dict:
        sources = {m["source_file"] for m in self._metadatas if "source_file" in m}
        return {
            "total_chunks": len(self._documents),
            "source_files": sorted(sources),
            "num_sources":  len(sources),
        }

    def clear(self):
        self._documents, self._metadatas, self._ids = [], [], []
        self._idf, self._doc_tf, self._doc_lengths, self._avgdl = {}, [], [], 1.0
        p = self._store_path()
        if p.exists():
            p.unlink()
