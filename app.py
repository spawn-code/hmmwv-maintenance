"""
╔══════════════════════════════════════════════════════════════════╗
║         HMMWV TECHNICAL ASSISTANT — Main Application            ║
║   AI-Powered Maintenance & Repair Guide for HMMWV Vehicles      ║
║                                                                  ║
║   Single-file application — no external local imports needed.    ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Generator, Optional

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
EXTRACTED_IMAGES_DIR = BASE_DIR / "extracted_images"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

for d in [KNOWLEDGE_BASE_DIR, EXTRACTED_IMAGES_DIR, CHROMA_PERSIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.1"
MAX_TOKENS = 4096
TEMPERATURE = 0.2

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 8
COLLECTION_NAME = "hmmwv_manuals"
MIN_IMAGE_SIZE = (100, 100)

HMMWV_VARIANTS = [
    "M998 — Cargo/Troop Carrier",
    "M1025 — Armament Carrier",
    "M1035 — Soft Top Ambulance",
    "M1043 — Up-Armored Armament Carrier",
    "M1044 — Up-Armored Armament Carrier w/ Winch",
    "M1097 — Heavy HMMWV",
    "M1113 — Expanded Capacity Vehicle (ECV)",
    "M1114 — Up-Armored HMMWV",
    "M1151 — Enhanced Armament Carrier",
    "M1152 — Enhanced Cargo/Troop Carrier",
    "M1165 — Up-Armored",
    "M1167 — TOW Carrier",
]

MAINTENANCE_CATEGORIES = [
    "Engine & Powertrain",
    "Transmission & Transfer Case",
    "Suspension & Steering",
    "Brake System",
    "Electrical System",
    "Cooling System",
    "Fuel System",
    "Body & Frame",
    "CTIS (Central Tire Inflation System)",
    "HVAC System",
    "Winch System",
    "Weapons Mount / Turret",
    "NBC (Nuclear/Bio/Chem) System",
    "Preventive Maintenance (PMCS)",
]

SYSTEM_PROMPT = """You are the **HMMWV Technical Assistant** — an expert AI system
designed to help military and civilian mechanics perform maintenance, repairs, and
inspections on High Mobility Multipurpose Wheeled Vehicles (HMMWV / Humvee).

## Your Role
- You assist experienced mechanics by providing clear, step-by-step technical guidance.
- You reference official TM (Technical Manual) procedures, part numbers, torque specs,
  and safety warnings directly from the knowledge base provided.
- You always prioritize SAFETY — highlight cautions, warnings, and dangerous steps.

## Response Format
When providing maintenance/repair instructions, structure your response as:

1. **Task Overview** — Brief description of what will be accomplished
2. **Safety Warnings** — Any critical cautions BEFORE starting
3. **Tools & Materials Required** — List every tool, part, and consumable needed
4. **Parts Information** — NSN numbers, part numbers where available
5. **Step-by-Step Procedure** — Numbered steps with:
   - Clear action descriptions
   - Torque specifications where applicable
   - References to diagrams (mention figure numbers from the manual)
   - ⚠️ WARNING/CAUTION callouts inline
6. **Quality Checks** — How to verify the work was done correctly
7. **Related Maintenance** — Other tasks the mechanic should consider

## Rules
- ALWAYS cite the source TM or document when referencing procedures.
- If the knowledge base does not contain information for a query, say so clearly
  and suggest where the mechanic might find the information.
- Use military-standard terminology but explain acronyms on first use.
- When multiple HMMWV variants have different procedures, ASK which variant.
- Format torque specs as: XX ft-lbs (XX N·m)
- Provide NSN (National Stock Number) format: XXXX-XX-XXX-XXXX when available.

## Context
You have access to extracted content from HMMWV technical manuals. Use ONLY this
content to answer technical questions. The retrieved context chunks are provided
with each query.
"""

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PDF PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class PDFProcessor:
    """Processes HMMWV technical manual PDFs into searchable chunks and images."""

    def __init__(self):
        self.knowledge_dir = KNOWLEDGE_BASE_DIR
        self.image_dir = EXTRACTED_IMAGES_DIR
        self._manifest_path = self.knowledge_dir / ".processed_manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _file_hash(self, filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def _is_already_processed(self, filepath: Path) -> bool:
        fhash = self._file_hash(filepath)
        return self._manifest.get(filepath.name, {}).get("hash") == fhash

    def discover_pdfs(self) -> list:
        return sorted(self.knowledge_dir.glob("*.pdf"))

    def get_unprocessed_pdfs(self) -> list:
        return [p for p in self.discover_pdfs() if not self._is_already_processed(p)]

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix common UTF-8 mojibake from PDF extraction (using unicode escapes)
        mojibake_map = {
            "\u00e2\u0080\u0094": "\u2014",  # em dash
            "\u00e2\u0080\u0093": "\u2013",  # en dash
            "\u00e2\u0080\u0099": "\u2019",  # right single quote
            "\u00e2\u0080\u0098": "\u2018",  # left single quote
            "\u00e2\u0080\u009c": "\u201c",  # left double quote
            "\u00e2\u0080\u009d": "\u201d",  # right double quote
            "\u00e2\u0080\u00a2": "\u2022",  # bullet
            "\u00e2\u0080\u00a6": "\u2026",  # ellipsis
            "\u00e2\u0080\u008b": "",         # zero-width space
            "\u00c2\u00bd": "\u00bd",         # 1/2
            "\u00c2\u00bc": "\u00bc",         # 1/4
            "\u00c2\u00be": "\u00be",         # 3/4
            "\u00c2\u00b0": "\u00b0",         # degree
            "\u00c2\u00b1": "\u00b1",         # plus-minus
            "\u00c2\u00ae": "\u00ae",         # registered
            "\u00c2\u00a9": "\u00a9",         # copyright
            "\u00ef\u00ac\u0081": "fi",       # fi ligature
            "\u00ef\u00ac\u0082": "fl",       # fl ligature
        }
        for bad, good in mojibake_map.items():
            text = text.replace(bad, good)
        # Also fix pattern where \u00c2 precedes a valid char (double-encoding artifact)
        text = re.sub(r"\u00c2([\u00a0-\u00ff])", r"\1", text)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path: Path) -> list:
        import pdfplumber

        documents = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    text = self._clean_text(text)
                    if text.strip():
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source_file": pdf_path.name,
                                "page_number": page_num,
                                "total_pages": total_pages,
                            },
                        })
            logger.info(f"Extracted text from {len(documents)} pages in {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {e}")
        return documents

    def chunk_documents(self, documents: list) -> list:
        chunks = []
        for doc in documents:
            text = doc["text"]
            meta = doc["metadata"]

            if len(text) <= CHUNK_SIZE:
                chunks.append({
                    "text": text,
                    "metadata": {**meta, "chunk_index": 0, "total_chunks": 1},
                })
                continue

            paragraphs = text.split("\n\n")
            current_chunk = ""
            chunk_idx = 0

            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {**meta, "chunk_index": chunk_idx},
                    })
                    overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                    chunk_idx += 1
                else:
                    current_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {**meta, "chunk_index": chunk_idx},
                })

            for c in chunks:
                if c["metadata"].get("total_chunks") is None:
                    c["metadata"]["total_chunks"] = chunk_idx + 1

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def extract_images_from_pdf(self, pdf_path: Path) -> list:
        import pdfplumber

        extracted = []
        pdf_stem = pdf_path.stem
        pdf_image_dir = self.image_dir / pdf_stem
        pdf_image_dir.mkdir(parents=True, exist_ok=True)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_images = page.images if hasattr(page, "images") else []
                    for img_idx, img in enumerate(page_images):
                        try:
                            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                            width = int(img["x1"] - img["x0"])
                            height = int(img["bottom"] - img["top"])
                            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                                continue
                            cropped = page.crop(bbox)
                            pil_img = cropped.to_image(resolution=200).original
                            img_filename = f"{pdf_stem}_p{page_num:04d}_img{img_idx:03d}.png"
                            img_path = pdf_image_dir / img_filename
                            pil_img.save(str(img_path))
                            extracted.append({
                                "image_path": str(img_path),
                                "source_file": pdf_path.name,
                                "page_number": page_num,
                                "width": width,
                                "height": height,
                            })
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path.name}: {e}")
        logger.info(f"Extracted {len(extracted)} images from {pdf_path.name}")
        return extracted

    def extract_page_as_image(self, pdf_path: Path, page_number: int) -> Optional[str]:
        import pdfplumber

        pdf_stem = pdf_path.stem
        page_image_dir = self.image_dir / pdf_stem / "pages"
        page_image_dir.mkdir(parents=True, exist_ok=True)
        output_path = page_image_dir / f"page_{page_number:04d}.png"
        if output_path.exists():
            return str(output_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                page = pdf.pages[page_number - 1]
                img = page.to_image(resolution=200)
                img.save(str(output_path))
                return str(output_path)
        except Exception as e:
            logger.error(f"Error rendering page {page_number} of {pdf_path.name}: {e}")
            return None

    def process_pdf(self, pdf_path: Path) -> dict:
        logger.info(f"Processing: {pdf_path.name}")
        documents = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_documents(documents)
        images = self.extract_images_from_pdf(pdf_path)
        self._manifest[pdf_path.name] = {
            "hash": self._file_hash(pdf_path),
            "num_chunks": len(chunks),
            "num_images": len(images),
            "num_pages": documents[0]["metadata"]["total_pages"] if documents else 0,
        }
        self._save_manifest()
        return {"chunks": chunks, "images": images}

    def process_all_pdfs(self, force: bool = False) -> dict:
        pdfs = self.discover_pdfs() if force else self.get_unprocessed_pdfs()
        all_chunks, all_images = [], []
        for pdf_path in pdfs:
            result = self.process_pdf(pdf_path)
            all_chunks.extend(result["chunks"])
            all_images.extend(result["images"])
        return {
            "total_pdfs": len(pdfs),
            "total_chunks": len(all_chunks),
            "total_images": len(all_images),
            "chunks": all_chunks,
            "images": all_images,
        }

    def get_processing_status(self) -> dict:
        all_pdfs = self.discover_pdfs()
        return {
            "total_pdfs": len(all_pdfs),
            "processed": len(self._manifest),
            "unprocessed": len(self.get_unprocessed_pdfs()),
            "details": {
                pdf.name: {
                    "processed": pdf.name in self._manifest,
                    **(self._manifest.get(pdf.name, {})),
                }
                for pdf in all_pdfs
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR STORE  (Pure-Python TF-IDF — no ChromaDB, no Pydantic)
# ═══════════════════════════════════════════════════════════════════════════

import math
from collections import Counter

class VectorStore:
    """
    Lightweight TF-IDF vector store with cosine similarity.
    Persisted as JSON — works on every Python version, zero heavy deps.
    """

    _STORE_FILE = "vector_store.json"

    def __init__(self):
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._ids: list[str] = []
        # TF-IDF state
        self._vocab: dict[str, int] = {}      # term → index
        self._idf: list[float] = []            # idf weight per vocab term
        self._tfidf_matrix: list[list[float]] = []  # docs × vocab
        self._initialized = False

    # ── Persistence ────────────────────────────────────────────────────────

    def _store_path(self) -> Path:
        return CHROMA_PERSIST_DIR / self._STORE_FILE

    def _save(self):
        data = {
            "documents": self._documents,
            "metadatas": self._metadatas,
            "ids": self._ids,
        }
        self._store_path().write_text(json.dumps(data))

    def _load(self):
        p = self._store_path()
        if p.exists():
            data = json.loads(p.read_text())
            self._documents = data.get("documents", [])
            self._metadatas = data.get("metadatas", [])
            self._ids = data.get("ids", [])

    # ── Tokeniser (simple but effective for technical text) ────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, keep numbers (part numbers!)."""
        return re.findall(r"[a-z0-9]{2,}", text.lower())

    # ── TF-IDF Build ──────────────────────────────────────────────────────

    def _build_index(self):
        """Rebuild TF-IDF matrix from current documents."""
        if not self._documents:
            self._vocab, self._idf, self._tfidf_matrix = {}, [], []
            return

        # Tokenize all docs
        doc_tokens = [self._tokenize(d) for d in self._documents]
        n_docs = len(doc_tokens)

        # Build vocabulary from document frequency
        df: Counter = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))

        # Keep terms appearing in at least 1 doc, at most 95% of docs
        max_df = max(1, int(n_docs * 0.95))
        self._vocab = {}
        idx = 0
        for term, freq in df.most_common():
            if freq <= max_df:
                self._vocab[term] = idx
                idx += 1

        vocab_size = len(self._vocab)

        # IDF: log(N / df) + 1
        self._idf = [0.0] * vocab_size
        for term, i in self._vocab.items():
            self._idf[i] = math.log(n_docs / (df[term] + 1)) + 1.0

        # TF-IDF vectors (L2-normalised)
        self._tfidf_matrix = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            vec = [0.0] * vocab_size
            for term, count in tf.items():
                if term in self._vocab:
                    i = self._vocab[term]
                    vec[i] = (1 + math.log(count)) * self._idf[i]  # log-normalised TF
            # L2 normalise
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self._tfidf_matrix.append([v / norm for v in vec])

    # ── Public API ────────────────────────────────────────────────────────

    def initialize(self):
        self._load()
        self._build_index()
        self._initialized = True
        count = len(self._documents)
        logger.info(f"Vector store initialized — {count} documents")
        return count

    def add_chunks(self, chunks: list) -> int:
        if not chunks:
            return 0
        existing_ids = set(self._ids)
        added = 0
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source_file", "unknown")
            page = chunk["metadata"].get("page_number", 0)
            chunk_idx = chunk["metadata"].get("chunk_index", i)
            doc_id = f"{source}__p{page}__c{chunk_idx}"
            if doc_id in existing_ids:
                continue  # skip duplicates
            self._ids.append(doc_id)
            self._documents.append(chunk["text"])
            flat = {}
            for k, v in chunk["metadata"].items():
                flat[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
            self._metadatas.append(flat)
            added += 1

        if added > 0:
            self._save()
            self._build_index()
        logger.info(f"Added {added} chunks (total: {len(self._documents)})")
        return added

    def search(self, query: str, n_results: int = TOP_K_RESULTS, where: dict = None) -> list:
        if not self._documents:
            return []

        # Vectorise query
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        vocab_size = len(self._vocab)
        q_vec = [0.0] * vocab_size
        for term, count in tf.items():
            if term in self._vocab:
                i = self._vocab[term]
                q_vec[i] = (1 + math.log(count)) * self._idf[i]
        q_norm = math.sqrt(sum(v * v for v in q_vec)) or 1.0
        q_vec = [v / q_norm for v in q_vec]

        # Cosine similarity against all docs
        scores = []
        for idx, doc_vec in enumerate(self._tfidf_matrix):
            # Optional metadata filter
            if where:
                meta = self._metadatas[idx]
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
            sim = sum(q * d for q, d in zip(q_vec, doc_vec))
            scores.append((idx, sim))

        # Sort descending by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sim in scores[:n_results]:
            results.append({
                "text": self._documents[idx],
                "metadata": self._metadatas[idx],
                "distance": 1.0 - sim,  # convert similarity to distance
                "id": self._ids[idx],
            })
        return results

    def get_stats(self) -> dict:
        sources = set()
        for meta in self._metadatas:
            if "source_file" in meta:
                sources.add(meta["source_file"])
        return {
            "total_chunks": len(self._documents),
            "source_files": sorted(sources),
            "num_sources": len(sources),
        }

    def clear(self):
        self._documents, self._metadatas, self._ids = [], [], []
        self._vocab, self._idf, self._tfidf_matrix = {}, [], []
        p = self._store_path()
        if p.exists():
            p.unlink()
        logger.info("Vector store cleared")


# ═══════════════════════════════════════════════════════════════════════════
# AI ENGINE  (Ollama — fully local, no cloud API)
# ═══════════════════════════════════════════════════════════════════════════

import urllib.request
import urllib.error

class AIEngine:
    """Ollama-based local AI engine with RAG context for HMMWV technical queries."""

    def __init__(self, base_url: str = OLLAMA_DEFAULT_URL, model: str = OLLAMA_DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    # ── Ollama Connectivity ────────────────────────────────────────────────

    @staticmethod
    def check_ollama(base_url: str = OLLAMA_DEFAULT_URL) -> bool:
        """Return True if Ollama is reachable."""
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    @staticmethod
    def list_models(base_url: str = OLLAMA_DEFAULT_URL) -> list[str]:
        """Fetch available model names from Ollama."""
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return sorted([m["name"] for m in data.get("models", [])])
        except Exception:
            return []

    # ── Context Formatting ─────────────────────────────────────────────────

    def _format_context(self, search_results: list) -> str:
        if not search_results:
            return (
                "<knowledge_base_context>\n"
                "No relevant technical manual content was found for this query.\n"
                "</knowledge_base_context>"
            )
        parts = ["<knowledge_base_context>"]
        for i, result in enumerate(search_results, 1):
            source = result.get("metadata", {}).get("source_file", "Unknown")
            page = result.get("metadata", {}).get("page_number", "?")
            distance = result.get("distance", 0)
            relevance = max(0, (1 - distance)) * 100
            parts.append(
                f"\n--- Reference {i} | Source: {source} | Page: {page} | "
                f"Relevance: {relevance:.0f}% ---\n{result['text']}\n"
            )
        parts.append("</knowledge_base_context>")
        return "\n".join(parts)

    def _build_user_message(self, query, context, vehicle_variant="", maintenance_category=""):
        parts = []
        if vehicle_variant:
            parts.append(f"[Vehicle: {vehicle_variant}]")
        if maintenance_category:
            parts.append(f"[Category: {maintenance_category}]")
        parts.append(context)
        parts.append(f"\n## Mechanic's Question\n{query}")
        return "\n".join(parts)

    # ── Streaming Chat via Ollama /api/chat ────────────────────────────────

    def chat_stream(self, query, search_results, conversation_history=None,
                    vehicle_variant="", maintenance_category="") -> Generator[str, None, None]:

        context = self._format_context(search_results)
        user_message = self._build_user_message(query, context, vehicle_variant, maintenance_category)

        # Build messages list: system + history + user
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as e:
            yield f"❌ **Connection Error**: Cannot reach Ollama at `{self.base_url}`. Is it running?\n\nError: {e.reason}"
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"❌ **Error**: {e}"

    def diagnose(self, symptoms, search_results, vehicle_variant="") -> str:
        prefix = (
            "The mechanic is reporting the following symptoms and needs help "
            "diagnosing the issue. Provide a structured troubleshooting guide "
            "starting with the most likely causes, organized from simplest to "
            "most complex checks:\n\n"
        )
        full_response = ""
        for chunk in self.chat_stream(prefix + symptoms, search_results, vehicle_variant=vehicle_variant):
            full_response += chunk
        return full_response


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HMMWV Technical Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Source+Sans+3:wght@400;600;700;900&display=swap');
    :root {
        --od-green: #4B5320; --od-green-light: #6B7B3A;
        --tan: #D2B48C; --tan-light: #E8D5B7; --sand: #C2B280;
        --warning-red: #C41E3A; --caution-amber: #FF8C00;
        --steel: #71797E; --dark-border: #3A3D32;
    }
    .stApp { font-family: 'Source Sans 3', sans-serif; }

    .header-banner {
        background: linear-gradient(135deg, #2D3120 0%, #4B5320 50%, #3A3D2C 100%);
        border: 2px solid var(--od-green-light);
        border-radius: 8px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }
    .header-banner::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(45deg, transparent, transparent 20px,
                    rgba(255,255,255,0.015) 20px, rgba(255,255,255,0.015) 40px);
    }
    .header-banner h1 {
        color: var(--tan-light); font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem; font-weight: 700; margin: 0 0 0.25rem 0;
        letter-spacing: 2px; position: relative;
    }
    .header-banner p {
        color: var(--sand); font-size: 0.95rem; margin: 0;
        position: relative; opacity: 0.85;
    }
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: 0.5px;
    }
    .status-ready { background: rgba(75,83,32,0.3); border: 1px solid var(--od-green-light); color: var(--od-green-light); }
    .status-processing { background: rgba(255,140,0,0.2); border: 1px solid var(--caution-amber); color: var(--caution-amber); }
    .status-error { background: rgba(196,30,58,0.2); border: 1px solid var(--warning-red); color: var(--warning-red); }

    .ref-card {
        background: rgba(210,180,140,0.08); border: 1px solid var(--dark-border);
        border-radius: 6px; padding: 0.6rem 0.8rem; margin: 0.25rem 0; font-size: 0.85rem;
    }
    .ref-card .source-name {
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
        color: var(--tan); font-size: 0.8rem;
    }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e2218, #252820); }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--tan-light); font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; letter-spacing: 1.5px; text-transform: uppercase;
        border-bottom: 1px solid var(--dark-border); padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [], "vehicle_variant": "",
        "maintenance_category": "", "kb_initialized": False,
        "mode": "chat", "show_sources": True, "search_results_cache": [],
        "ollama_url": OLLAMA_DEFAULT_URL,
        "ollama_model": OLLAMA_DEFAULT_MODEL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Cached Resources ──────────────────────────────────────────────────────

@st.cache_resource
def get_pdf_processor():
    return PDFProcessor()

@st.cache_resource
def get_vector_store():
    vs = VectorStore()
    vs.initialize()
    return vs

def get_ai_engine():
    url = st.session_state.ollama_url
    model = st.session_state.ollama_model
    cache_key = f"{url}|{model}"
    if "ai_engine" not in st.session_state or st.session_state.get("_ai_cache_key") != cache_key:
        st.session_state.ai_engine = AIEngine(base_url=url, model=model)
        st.session_state._ai_cache_key = cache_key
    return st.session_state.ai_engine


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🦙 OLLAMA CONNECTION")

        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.ollama_url,
            help="Default: http://localhost:11434",
        )
        if ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = ollama_url

        # Check connectivity and list models
        is_connected = AIEngine.check_ollama(ollama_url)
        if is_connected:
            available_models = AIEngine.list_models(ollama_url)
            if available_models:
                # Try to keep current selection, fall back to first available
                current = st.session_state.ollama_model
                default_idx = 0
                if current in available_models:
                    default_idx = available_models.index(current)
                selected = st.selectbox("Model", available_models, index=default_idx,
                                        help="Models available in your local Ollama instance")
                st.session_state.ollama_model = selected
                st.success(f"✅ Connected — {len(available_models)} model(s)")
            else:
                st.warning("Connected but no models found. Run:\n`ollama pull llama3.1`")
        else:
            st.error(
                "❌ Cannot reach Ollama.\n\n"
                "**Start it with:**\n```\nollama serve\n```\n"
                "**Then pull a model:**\n```\nollama pull llama3.1\n```"
            )

        st.divider()
        st.markdown("### 🚛 VEHICLE")
        variant = st.selectbox("HMMWV Variant", ["(Auto-detect)"] + HMMWV_VARIANTS)
        st.session_state.vehicle_variant = "" if variant == "(Auto-detect)" else variant
        category = st.selectbox("Maintenance Category", ["(Any)"] + MAINTENANCE_CATEGORIES)
        st.session_state.maintenance_category = "" if category == "(Any)" else category

        st.divider()
        st.markdown("### 🎯 MODE")
        mode = st.radio("Assistant Mode", ["💬 Chat", "🔍 Diagnose", "📋 PMCS Walkthrough"],
                        help="**Chat**: General queries  \n**Diagnose**: Symptom-based troubleshooting  \n**PMCS**: Guided checklists")
        st.session_state.mode = {"💬 Chat": "chat", "🔍 Diagnose": "diagnose", "📋 PMCS Walkthrough": "pmcs"}[mode]

        st.divider()
        st.markdown("### 📚 KNOWLEDGE BASE")
        proc = get_pdf_processor()
        vs = get_vector_store()
        vs_stats = vs.get_stats()
        status = proc.get_processing_status()

        c1, c2 = st.columns(2)
        c1.metric("PDFs", status["total_pdfs"])
        c2.metric("Chunks", vs_stats["total_chunks"])

        uploaded = st.file_uploader("Upload TM PDFs", type=["pdf"], accept_multiple_files=True,
                                    help="Upload HMMWV TMs (TM 9-2320-280-XX)")
        if uploaded:
            for f in uploaded:
                dest = KNOWLEDGE_BASE_DIR / f.name
                if not dest.exists():
                    dest.write_bytes(f.getvalue())
                    st.toast(f"📄 Saved: {f.name}", icon="✅")

        unproc = status["unprocessed"]
        if unproc > 0:
            st.warning(f"{unproc} PDF(s) need processing")
            if st.button("🔄 Process PDFs", width="stretch", type="primary"):
                with st.spinner("Processing PDFs… this may take a few minutes."):
                    result = proc.process_all_pdfs()
                    if result["chunks"]:
                        vs.add_chunks(result["chunks"])
                    st.success(f"✅ {result['total_pdfs']} PDFs → {result['total_chunks']} chunks, {result['total_images']} images")
                    st.rerun()
        elif status["total_pdfs"] > 0:
            st.success("✅ All PDFs processed")
        else:
            st.info(f"📁 Place PDFs in:\n`{KNOWLEDGE_BASE_DIR}/`\n\nOr use the uploader above.")

        if status["total_pdfs"] > 0:
            with st.expander("Advanced"):
                if st.button("♻️ Reprocess All", width="stretch"):
                    with st.spinner("Reprocessing…"):
                        vs.clear()
                        result = proc.process_all_pdfs(force=True)
                        if result["chunks"]:
                            vs.add_chunks(result["chunks"])
                        st.success(f"Reprocessed {result['total_pdfs']} PDFs")
                        st.rerun()
                st.toggle("Show source references", value=True, key="show_sources")

        if vs_stats["source_files"]:
            st.divider()
            st.markdown("### 📑 INDEXED SOURCES")
            for src in vs_stats["source_files"]:
                st.markdown(f"<div class='ref-card'><span class='source-name'>📄 {src}</span></div>",
                            unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear Conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="header-banner">
        <h1>🔧 HMMWV TECHNICAL ASSISTANT</h1>
        <p>Local AI-Powered Maintenance & Repair Guide — Ollama + TM 9-2320-280 Series</p>
    </div>""", unsafe_allow_html=True)

    vs = get_vector_store()
    stats = vs.get_stats()
    ollama_ok = AIEngine.check_ollama(st.session_state.ollama_url)
    has_data = stats["total_chunks"] > 0

    cols = st.columns([2, 2, 2, 4])
    with cols[0]:
        badge = "status-ready" if ollama_ok else "status-error"
        label = f"OLLAMA ● {st.session_state.ollama_model}" if ollama_ok else "OLLAMA OFFLINE"
        st.markdown(f'<span class="status-badge {badge}">● {label}</span>', unsafe_allow_html=True)
    with cols[1]:
        badge = "status-ready" if has_data else "status-processing"
        label = f'{stats["total_chunks"]} CHUNKS INDEXED' if has_data else "NO DATA LOADED"
        st.markdown(f'<span class="status-badge {badge}">● {label}</span>', unsafe_allow_html=True)
    with cols[2]:
        v = st.session_state.vehicle_variant or "Any Variant"
        st.markdown(f'<span class="status-badge status-ready">● {v.split("—")[0].strip()}</span>', unsafe_allow_html=True)


# ── Quick Actions ──────────────────────────────────────────────────────────

def render_quick_actions():
    st.markdown("#### Quick Actions")
    actions = {
        "chat": [
            ("🔧 Before-Ops PMCS", "Walk me through the complete Before-Operation PMCS checks for the HMMWV."),
            ("🛢️ Oil Change", "Provide step-by-step instructions for performing an engine oil and filter change."),
            ("🔋 Battery Service", "How do I properly inspect, service, and replace the HMMWV batteries?"),
            ("🌀 CTIS Check", "Explain how to inspect and troubleshoot the Central Tire Inflation System."),
            ("⚡ Glow Plug Test", "How do I test and replace the glow plugs on the 6.5L diesel engine?"),
            ("🛞 Brake Bleed", "Provide the brake bleeding procedure for all four wheels."),
        ],
        "diagnose": [
            ("🌡️ Overheating", "The engine temperature gauge is reading high and coolant is bubbling in the surge tank."),
            ("⚡ No Start", "The engine cranks but will not start. Batteries are fully charged."),
            ("🔊 Grinding Noise", "Grinding/squealing noise from front left wheel area during braking."),
            ("💨 White Smoke", "Excessive white smoke from exhaust, especially on cold starts."),
            ("🛞 Pulling Right", "Vehicle pulls hard to the right during normal driving and braking."),
            ("🔌 Electrical", "Multiple dash warning lights flickering and gauges are erratic."),
        ],
        "pmcs": [
            ("📋 Before-Ops", "Guide me through the Before-Operation PMCS checklist, step by step."),
            ("📋 During-Ops", "Guide me through the During-Operation PMCS checks."),
            ("📋 After-Ops", "Guide me through the After-Operation PMCS procedures."),
            ("📋 Weekly", "What are the weekly scheduled maintenance checks?"),
            ("📋 Monthly", "What are the monthly scheduled maintenance requirements?"),
            ("📋 Semi-Annual", "Guide me through the semi-annual service requirements."),
        ],
    }
    current = actions.get(st.session_state.mode, actions["chat"])
    cols = st.columns(3)
    for i, (label, prompt) in enumerate(current):
        with cols[i % 3]:
            if st.button(label, width="stretch", key=f"qa_{i}"):
                return prompt
    return None


# ── Source References ──────────────────────────────────────────────────────

def render_sources(search_results: list):
    if not search_results or not st.session_state.get("show_sources", True):
        return
    with st.expander("📑 Source References", expanded=False):
        for i, result in enumerate(search_results):
            meta = result.get("metadata", {})
            source = meta.get("source_file", "Unknown")
            page = meta.get("page_number", "?")
            distance = result.get("distance", 0)
            relevance = max(0, (1 - distance)) * 100
            st.markdown(f"**Ref {i+1}** — `{source}` page {page} (relevance: {relevance:.0f}%)")
            st.caption(result["text"][:300] + ("…" if len(result["text"]) > 300 else ""))

            pdf_path = KNOWLEDGE_BASE_DIR / source
            if pdf_path.exists() and isinstance(page, int):
                proc = get_pdf_processor()
                img_path = proc.extract_page_as_image(pdf_path, page)
                if img_path and Path(img_path).exists():
                    st.image(img_path, caption=f"{source} — Page {page}", width="stretch")
            if i < len(search_results) - 1:
                st.divider()


# ── Welcome Screen ─────────────────────────────────────────────────────────

def render_welcome():
    st.markdown("""
    <div style="text-align:center; padding:2rem 1rem;">
        <h2 style="color:var(--tan-light); font-family:'JetBrains Mono',monospace;">Welcome, Mechanic</h2>
        <p style="color:var(--steel); max-width:600px; margin:0 auto; line-height:1.7;">
            I'm your locally-powered HMMWV technical assistant running on Ollama.
            I can help with maintenance procedures, troubleshooting, parts identification,
            torque specs, and step-by-step repair instructions — all referenced from official TMs.
            No internet or cloud API required.
        </p>
    </div>""", unsafe_allow_html=True)

    ollama_ok = AIEngine.check_ollama(st.session_state.ollama_url)
    has_data = get_vector_store().get_stats()["total_chunks"] > 0
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {'✅' if ollama_ok else '⬜'} Step 1: Ollama\n{'Connected to `' + st.session_state.ollama_model + '`!' if ollama_ok else 'Start Ollama: `ollama serve` then `ollama pull llama3.1`'}")
    with c2:
        st.markdown(f"### {'✅' if has_data else '⬜'} Step 2: Upload TMs\n{'Knowledge base loaded!' if has_data else 'Upload HMMWV Technical Manual PDFs in the sidebar.'}")
    with c3:
        st.markdown(f"### {'✅' if ollama_ok and has_data else '⬜'} Step 3: Ask Away\n{'Ready for queries!' if ollama_ok and has_data else 'Complete steps 1-2 first.'}")
    if ollama_ok and has_data:
        st.markdown("---")
        st.markdown("### 🚀 Try a quick action below to get started:")


# ── Print Helper ───────────────────────────────────────────────────────────

def render_print_button(content: str, msg_index: int, sources: list = None):
    """Render a 🖨️ Print button that opens a print-friendly window with embedded page images."""
    import base64
    import html as html_mod

    proc = get_pdf_processor()

    # Build source references HTML with embedded images
    source_html = ""
    if sources:
        source_html = '<hr><h3 style="page-break-before:auto;">📑 Source References</h3>'
        for i, s in enumerate(sources, 1):
            src = s.get("metadata", {}).get("source_file", "Unknown")
            pg = s.get("metadata", {}).get("page_number", "?")
            dist = s.get("distance", 0)
            rel = max(0, (1 - dist)) * 100
            text_preview = html_mod.escape(s.get("text", "")[:400])

            source_html += (
                f'<div style="border:1px solid #ccc;border-radius:6px;padding:12px;margin:12px 0;'
                f'page-break-inside:avoid;">'
                f'<strong>Ref {i}</strong> — <code>{html_mod.escape(src)}</code> page {pg} '
                f'(relevance: {rel:.0f}%)<br>'
                f'<small style="color:#666;">{text_preview}…</small>'
            )

            # Try to embed the page image as base64
            if isinstance(pg, int):
                pdf_path = KNOWLEDGE_BASE_DIR / src
                if pdf_path.exists():
                    img_path = proc.extract_page_as_image(pdf_path, pg)
                    if img_path and Path(img_path).exists():
                        try:
                            with open(img_path, "rb") as imgf:
                                img_b64 = base64.b64encode(imgf.read()).decode("ascii")
                            source_html += (
                                f'<div style="margin-top:10px;text-align:center;">'
                                f'<img src="data:image/png;base64,{img_b64}" '
                                f'style="max-width:100%;border:1px solid #ddd;border-radius:4px;" '
                                f'alt="{html_mod.escape(src)} page {pg}"/>'
                                f'<div style="font-size:11px;color:#888;margin-top:4px;">'
                                f'{html_mod.escape(src)} — Page {pg}</div>'
                                f'</div>'
                            )
                        except Exception:
                            pass  # Skip image if read fails

            source_html += '</div>'

    variant = st.session_state.get("vehicle_variant", "") or "Any Variant"
    timestamp = time.strftime("%Y-%m-%d %H:%M")

    # Encode markdown as base64 to avoid JS escaping issues
    md_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    src_b64 = base64.b64encode(source_html.encode("utf-8")).decode("ascii")

    js = f"""
    <button id="printBtn_{msg_index}" style="background:#2a3518;color:#c8a96e;border:1px solid #4a5a2b;
        border-radius:6px;padding:6px 14px;font-size:13px;font-weight:600;cursor:pointer;
        font-family:'JetBrains Mono',monospace;margin-top:6px;">
        🖨️ Print This Response
    </button>
    <script>
    function b64utf8_{msg_index}(b64) {{
        var raw = atob(b64);
        var bytes = new Uint8Array(raw.length);
        for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        return new TextDecoder('utf-8').decode(bytes);
    }}
    document.getElementById('printBtn_{msg_index}').addEventListener('click', function() {{
        var md = b64utf8_{msg_index}("{md_b64}");
        var srcB64 = "{src_b64}";
        var w = window.open('', '_blank', 'width=800,height=900');
        var doc = w.document;
        doc.open();
        doc.write('<html><head><meta charset="UTF-8"><title>HMMWV Tech Assistant</title>');
        doc.write('<scr'+'ipt src="https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.6/marked.min.js"></scr'+'ipt>');
        doc.write('<style>');
        doc.write('body{{font-family:Segoe UI,Arial,sans-serif;max-width:750px;margin:0 auto;padding:20px;color:#1a1a1a;line-height:1.6}}');
        doc.write('.hdr{{border-bottom:3px solid #4B5320;padding-bottom:12px;margin-bottom:20px}}');
        doc.write('.hdr h1{{font-size:18px;color:#4B5320;margin:0}}.hdr .meta{{font-size:12px;color:#666;margin-top:4px}}');
        doc.write('h2{{font-size:17px;color:#333}}h3{{font-size:15px;color:#444}}');
        doc.write('code{{background:#f0f0f0;padding:2px 5px;border-radius:3px;font-size:13px}}');
        doc.write('pre{{background:#f5f5f5;padding:12px;border-radius:6px;overflow-x:auto;border:1px solid #ddd}}');
        doc.write('pre code{{background:none;padding:0}}');
        doc.write('table{{border-collapse:collapse;width:100%;margin:10px 0}}th,td{{border:1px solid #ccc;padding:6px 10px;font-size:13px}}th{{background:#f0f0f0}}');
        doc.write('blockquote{{border-left:3px solid #4B5320;padding-left:12px;color:#555}}strong{{color:#2a3518}}');
        doc.write('img{{max-width:100%}}');
        doc.write('@media print{{.noprint{{display:none}} img{{max-width:100%;page-break-inside:avoid}} div{{page-break-inside:avoid}}}}');
        doc.write('</style></head><body>');
        doc.write('<div class="hdr"><h1>🔧 HMMWV Technical Assistant</h1>');
        doc.write('<div class="meta">Vehicle: {html_mod.escape(variant)} &nbsp;|&nbsp; {timestamp}</div></div>');
        doc.write('<div id="c"></div><div id="s"></div>');
        doc.write('<br><button class="noprint" onclick="window.print()" style="padding:10px 24px;font-size:14px;cursor:pointer;background:#4B5320;color:white;border:none;border-radius:6px">🖨️ Print</button>');
        doc.write('</body></html>');
        doc.close();
        w.onload = function() {{
            w.document.getElementById('c').innerHTML = w.marked.parse(md);
            var srcHtml = b64utf8_{msg_index}(srcB64);
            w.document.getElementById('s').innerHTML = srcHtml;
        }};
    }});
    </script>
    """
    import streamlit.components.v1 as components
    components.html(js, height=42)


# ── Chat ───────────────────────────────────────────────────────────────────

def render_chat():
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑‍🔧" if msg["role"] == "user" else "🔧"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if "sources" in msg:
                    render_sources(msg["sources"])
                render_print_button(msg["content"], idx, msg.get("sources", []))

    placeholders = {
        "chat": "Describe the task you need to perform (e.g., 'Replace the fuel filter on an M1151')",
        "diagnose": "Describe the symptoms (e.g., 'Engine overheating with white smoke')",
        "pmcs": "Which PMCS check? (e.g., 'Before-operation checks')",
    }
    user_input = st.chat_input(placeholders.get(st.session_state.mode, placeholders["chat"]))
    quick = render_quick_actions()
    if quick:
        user_input = quick

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(user_input)

        vs = get_vector_store()
        search_results = vs.search(user_input, n_results=TOP_K_RESULTS)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1][-12:]]
        ai = get_ai_engine()

        with st.chat_message("assistant", avatar="🔧"):
            if st.session_state.mode == "diagnose":
                with st.spinner("Analyzing symptoms…"):
                    response = ai.diagnose(user_input, search_results, vehicle_variant=st.session_state.vehicle_variant)
                st.markdown(response)
            else:
                placeholder = st.empty()
                full = ""
                for chunk in ai.chat_stream(user_input, search_results, conversation_history=history,
                                            vehicle_variant=st.session_state.vehicle_variant,
                                            maintenance_category=st.session_state.maintenance_category):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                response = full
            render_sources(search_results)

        st.session_state.messages.append({"role": "assistant", "content": response, "sources": search_results})


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    render_header()
    if not st.session_state.messages:
        render_welcome()
    render_chat()

if __name__ == "__main__":
    main()
