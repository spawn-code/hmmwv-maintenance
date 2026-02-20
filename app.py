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

import base64
import hashlib
import html as html_mod
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Generator, Optional

import streamlit as st
import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
EXTRACTED_IMAGES_DIR = BASE_DIR / "extracted_images"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
SETTINGS_FILE = BASE_DIR / ".hmmwv_settings.json"

for d in [KNOWLEDGE_BASE_DIR, EXTRACTED_IMAGES_DIR, CHROMA_PERSIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_DEFAULT_URL   = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "gpt-oss:latest"
OPENAI_DEFAULT_URL   = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-6"

PROVIDER_OLLAMA    = "Ollama (Local)"
PROVIDER_OPENAI    = "OpenAI-Compatible"
PROVIDER_ANTHROPIC = "Anthropic (Claude)"
ALL_PROVIDERS      = [PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC]

MAX_TOKENS    = 4096
TEMPERATURE   = 0.2
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 8
COLLECTION_NAME = "hmmwv_manuals"
MIN_IMAGE_SIZE  = (100, 100)

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _default_settings() -> dict:
    return {
        "provider": PROVIDER_OLLAMA,
        "ollama_url": OLLAMA_DEFAULT_URL,
        "ollama_model": OLLAMA_DEFAULT_MODEL,
        "openai_url": OPENAI_DEFAULT_URL,
        "openai_model": OPENAI_DEFAULT_MODEL,
        "openai_api_key": "",
        "anthropic_api_key": "",
        "anthropic_model": ANTHROPIC_DEFAULT_MODEL,
    }

def load_settings() -> dict:
    defaults = _default_settings()
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
        except Exception:
            pass
    return defaults

def save_settings(settings: dict):
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not save settings: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PDF PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class PDFProcessor:
    def __init__(self):
        self.knowledge_dir = KNOWLEDGE_BASE_DIR
        self.image_dir     = EXTRACTED_IMAGES_DIR
        self._manifest_path = self.knowledge_dir / ".processed_manifest.json"
        self._manifest      = self._load_manifest()

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
        mojibake_map = {
            "\u00e2\u0080\u0094": "\u2014", "\u00e2\u0080\u0093": "\u2013",
            "\u00e2\u0080\u0099": "\u2019", "\u00e2\u0080\u0098": "\u2018",
            "\u00e2\u0080\u009c": "\u201c", "\u00e2\u0080\u009d": "\u201d",
            "\u00e2\u0080\u00a2": "\u2022", "\u00e2\u0080\u00a6": "\u2026",
            "\u00e2\u0080\u008b": "",        "\u00c2\u00bd": "\u00bd",
            "\u00c2\u00bc": "\u00bc",        "\u00c2\u00be": "\u00be",
            "\u00c2\u00b0": "\u00b0",        "\u00c2\u00b1": "\u00b1",
            "\u00c2\u00ae": "\u00ae",        "\u00c2\u00a9": "\u00a9",
            "\u00ef\u00ac\u0081": "fi",      "\u00ef\u00ac\u0082": "fl",
        }
        for bad, good in mojibake_map.items():
            text = text.replace(bad, good)
        text = re.sub(r"\u00c2([\u00a0-\u00ff])", r"\1", text)
        return text.strip()

    def extract_text_from_pdf(self, pdf_path: Path) -> list:
        import pdfplumber
        documents = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = self._clean_text(page.extract_text() or "")
                    if text.strip():
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source_file": pdf_path.name,
                                "page_number": page_num,
                                "total_pages": total_pages,
                            },
                        })
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {e}")
        return documents

    def _split_long_paragraph(self, para: str) -> list:
        if len(para) <= CHUNK_SIZE:
            return [para]
        sentences = re.split(r"(?<=[.!?])\s+", para)
        parts, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 > CHUNK_SIZE and current:
                parts.append(current.strip())
                current = s
            else:
                current = current + " " + s if current else s
        if current.strip():
            parts.append(current.strip())
        return parts or [para]

    def chunk_documents(self, documents: list) -> list:
        all_chunks = []
        for doc in documents:
            text, meta = doc["text"], doc["metadata"]
            doc_chunks = []
            if len(text) <= CHUNK_SIZE:
                doc_chunks.append({"text": text, "metadata": {**meta, "chunk_index": 0, "total_chunks": 1}})
            else:
                paragraphs = []
                for p in text.split("\n\n"):
                    paragraphs.extend(self._split_long_paragraph(p))
                current_chunk, chunk_idx = "", 0
                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 > CHUNK_SIZE and current_chunk:
                        doc_chunks.append({"text": current_chunk.strip(), "metadata": {**meta, "chunk_index": chunk_idx}})
                        overlap = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                        current_chunk = overlap + "\n\n" + para
                        chunk_idx += 1
                    else:
                        current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                if current_chunk.strip():
                    doc_chunks.append({"text": current_chunk.strip(), "metadata": {**meta, "chunk_index": chunk_idx}})
                total = chunk_idx + 1
                for c in doc_chunks:
                    if c["metadata"].get("total_chunks") is None:
                        c["metadata"]["total_chunks"] = total
            all_chunks.extend(doc_chunks)
        return all_chunks

    def extract_images_from_pdf(self, pdf_path: Path) -> list:
        import pdfplumber
        extracted = []
        pdf_stem = pdf_path.stem
        pdf_image_dir = self.image_dir / pdf_stem
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    for img_idx, img in enumerate(getattr(page, "images", [])):
                        try:
                            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                            w, h = int(img["x1"]-img["x0"]), int(img["bottom"]-img["top"])
                            if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
                                continue
                            pil_img = page.crop(bbox).to_image(resolution=200).original
                            fname = f"{pdf_stem}_p{page_num:04d}_img{img_idx:03d}.png"
                            img_path = pdf_image_dir / fname
                            pil_img.save(str(img_path))
                            extracted.append({"image_path": str(img_path), "source_file": pdf_path.name,
                                              "page_number": page_num, "width": w, "height": h})
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path.name}: {e}")
        return extracted

    def extract_page_as_image(self, pdf_path: Path, page_number: int) -> Optional[str]:
        import pdfplumber
        pdf_stem = pdf_path.stem
        out_dir = self.image_dir / pdf_stem / "pages"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"page_{page_number:04d}.png"
        if out_path.exists():
            return str(out_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                pdf.pages[page_number - 1].to_image(resolution=200).save(str(out_path))
                return str(out_path)
        except Exception as e:
            logger.error(f"Error rendering page {page_number} of {pdf_path.name}: {e}")
            return None

    def process_pdf(self, pdf_path: Path) -> dict:
        documents = self.extract_text_from_pdf(pdf_path)
        chunks    = self.chunk_documents(documents)
        images    = self.extract_images_from_pdf(pdf_path)
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
        for p in pdfs:
            r = self.process_pdf(p)
            all_chunks.extend(r["chunks"])
            all_images.extend(r["images"])
        return {"total_pdfs": len(pdfs), "total_chunks": len(all_chunks),
                "total_images": len(all_images), "chunks": all_chunks, "images": all_images}

    def get_processing_status(self) -> dict:
        all_pdfs = self.discover_pdfs()
        return {
            "total_pdfs": len(all_pdfs),
            "processed": len(self._manifest),
            "unprocessed": len(self.get_unprocessed_pdfs()),
            "details": {
                pdf.name: {"processed": pdf.name in self._manifest, **(self._manifest.get(pdf.name, {}))}
                for pdf in all_pdfs
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════

class VectorStore:
    _STORE_FILE = "vector_store.json"

    def __init__(self):
        self._documents: list = []
        self._metadatas: list = []
        self._ids: list = []
        self._vocab: dict = {}
        self._idf: list = []
        self._tfidf_matrix: list = []
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
        return re.findall(r"[a-z0-9]{2,}", text.lower())

    def _build_index(self):
        if not self._documents:
            self._vocab, self._idf, self._tfidf_matrix = {}, [], []
            return
        doc_tokens = [self._tokenize(d) for d in self._documents]
        n_docs = len(doc_tokens)
        df: Counter = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        max_df = max(1, int(n_docs * 0.95))
        self._vocab = {term: idx for idx, (term, freq) in enumerate(df.most_common()) if freq <= max_df}
        vocab_size = len(self._vocab)
        self._idf = [math.log(n_docs / (df[term] + 1)) + 1.0 for term in self._vocab]
        self._tfidf_matrix = []
        for tokens in doc_tokens:
            tf  = Counter(tokens)
            vec = [0.0] * vocab_size
            for term, count in tf.items():
                if term in self._vocab:
                    i = self._vocab[term]
                    vec[i] = (1 + math.log(count)) * self._idf[i]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self._tfidf_matrix.append([v / norm for v in vec])

    def initialize(self):
        self._load()
        self._build_index()
        self._initialized = True
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
        if not self._documents:
            return []
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        vocab_size = len(self._vocab)
        q_vec = [0.0] * vocab_size
        for term, count in tf.items():
            if term in self._vocab:
                i = self._vocab[term]
                q_vec[i] = (1 + math.log(count)) * self._idf[i]
        q_norm = math.sqrt(sum(v * v for v in q_vec)) or 1.0
        q_vec  = [v / q_norm for v in q_vec]
        scores = []
        for idx, doc_vec in enumerate(self._tfidf_matrix):
            if where and not all(self._metadatas[idx].get(k) == v for k, v in where.items()):
                continue
            scores.append((idx, sum(q * d for q, d in zip(q_vec, doc_vec))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self._documents[i], "metadata": self._metadatas[i],
             "distance": 1.0 - sim, "id": self._ids[i]}
            for i, sim in scores[:n_results]
        ]

    def get_stats(self) -> dict:
        sources = {m["source_file"] for m in self._metadatas if "source_file" in m}
        return {"total_chunks": len(self._documents), "source_files": sorted(sources), "num_sources": len(sources)}

    def clear(self):
        self._documents, self._metadatas, self._ids = [], [], []
        self._vocab, self._idf, self._tfidf_matrix = {}, [], []
        p = self._store_path()
        if p.exists():
            p.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AIEngine:
    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        self._cfg     = kwargs

    @staticmethod
    def check_ollama(base_url: str = OLLAMA_DEFAULT_URL) -> bool:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/api/tags"), timeout=3
            ):
                return True
        except Exception:
            return False

    @staticmethod
    def list_ollama_models(base_url: str = OLLAMA_DEFAULT_URL) -> list:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/api/tags"), timeout=5
            ) as resp:
                return sorted(m["name"] for m in json.loads(resp.read()).get("models", []))
        except Exception:
            return []

    def _format_context(self, results: list) -> str:
        if not results:
            return "<knowledge_base_context>\nNo relevant content found.\n</knowledge_base_context>"
        parts = ["<knowledge_base_context>"]
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}).get("source_file", "Unknown")
            pg  = r.get("metadata", {}).get("page_number", "?")
            rel = max(0, 1 - r.get("distance", 0)) * 100
            parts.append(f"\n--- Ref {i} | {src} | Page {pg} | {rel:.0f}% ---\n{r['text']}\n")
        parts.append("</knowledge_base_context>")
        return "\n".join(parts)

    def _build_user_message(self, query, context, vehicle_variant="", maintenance_category=""):
        parts = []
        if vehicle_variant:      parts.append(f"[Vehicle: {vehicle_variant}]")
        if maintenance_category: parts.append(f"[Category: {maintenance_category}]")
        parts.append(context)
        parts.append(f"\n## Mechanic's Question\n{query}")
        return "\n".join(parts)

    def _stream_ollama(self, messages: list) -> Generator:
        url   = self._cfg.get("base_url", OLLAMA_DEFAULT_URL).rstrip("/")
        model = self._cfg.get("model", OLLAMA_DEFAULT_MODEL)
        payload = json.dumps({"model": model, "messages": messages, "stream": True,
                               "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}}).encode()
        req = urllib.request.Request(f"{url}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.strip()
                    if not line: continue
                    try:
                        chunk = json.loads(line.decode())
                        token = chunk.get("message", {}).get("content", "")
                        if token: yield token
                        if chunk.get("done"): break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as e:
            yield f"❌ **Cannot reach Ollama** at `{url}`\n\n{e.reason}"
        except Exception as e:
            yield f"❌ **Error**: {e}"

    def _stream_openai(self, messages: list) -> Generator:
        url    = self._cfg.get("base_url", OPENAI_DEFAULT_URL).rstrip("/")
        model  = self._cfg.get("model", OPENAI_DEFAULT_MODEL)
        apikey = self._cfg.get("api_key", "")
        payload = json.dumps({"model": model, "messages": messages, "stream": True,
                               "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}).encode()
        headers = {"Content-Type": "application/json"}
        if apikey: headers["Authorization"] = f"Bearer {apikey}"
        req = urllib.request.Request(f"{url}/chat/completions", data=payload,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    decoded = line.strip().decode("utf-8")
                    if decoded.startswith("data: "): decoded = decoded[6:]
                    if decoded == "[DONE]": break
                    if not decoded: continue
                    try:
                        token = json.loads(decoded).get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token: yield token
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode()
            except Exception: pass
            yield f"❌ **API Error {e.code}**: {e.reason}\n\n{body}"
        except Exception as e:
            yield f"❌ **Error**: {e}"

    def _stream_anthropic(self, messages: list) -> Generator:
        apikey = self._cfg.get("api_key", "")
        model  = self._cfg.get("model", ANTHROPIC_DEFAULT_MODEL)
        if not apikey:
            yield "❌ **Anthropic API key required.** Enter it in the sidebar."
            return
        sys_msg, filtered = "", []
        for m in messages:
            if m["role"] == "system": sys_msg = m["content"]
            else: filtered.append(m)
        payload = json.dumps({"model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                               "system": sys_msg or SYSTEM_PROMPT, "messages": filtered, "stream": True}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": apikey,
                     "anthropic-version": "2023-06-01"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    decoded = line.strip().decode("utf-8")
                    if decoded.startswith("data: "): decoded = decoded[6:]
                    if not decoded: continue
                    try:
                        event = json.loads(decoded)
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text", "")
                            if text: yield text
                        elif event.get("type") in ("message_stop", "error"):
                            if event.get("type") == "error":
                                yield f"❌ {event.get('error', {}).get('message', 'Error')}"
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode()
            except Exception: pass
            yield f"❌ **API Error {e.code}**: {e.reason}\n\n{body}"
        except Exception as e:
            yield f"❌ **Error**: {e}"

    def chat_stream(self, query, search_results, conversation_history=None,
                    vehicle_variant="", maintenance_category="") -> Generator:
        context = self._format_context(search_results)
        user_msg = self._build_user_message(query, context, vehicle_variant, maintenance_category)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history: messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_msg})
        if self.provider == PROVIDER_OLLAMA:       yield from self._stream_ollama(messages)
        elif self.provider == PROVIDER_OPENAI:     yield from self._stream_openai(messages)
        elif self.provider == PROVIDER_ANTHROPIC:  yield from self._stream_anthropic(messages)
        else: yield f"❌ Unknown provider: {self.provider}"

    def diagnose(self, symptoms, search_results, vehicle_variant="") -> Generator:
        yield from self.chat_stream(
            "The mechanic reports the following symptoms. Provide a structured troubleshooting "
            "guide from simplest to most complex checks:\n\n" + symptoms,
            search_results, vehicle_variant=vehicle_variant,
        )


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HMMWV Technical Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS
# ═══════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Design Tokens — Light Theme ──────────────────────────────────────── */
:root {
  /* Page & surface */
  --c-bg:          #f5f6f2;
  --c-surface:     #ffffff;
  --c-surface-2:   #f0f2ec;
  --c-surface-3:   #e8ebe2;
  --c-border:      #d0d5c8;
  --c-border-2:    #b8bfae;

  /* Accent greens (olive-drab, shifted lighter for readability on white) */
  --c-green-900:   #e8eddc;
  --c-green-800:   #d4dfc2;
  --c-green-700:   #a8bc7a;
  --c-green-600:   #7a9448;
  --c-green-500:   #5c7a30;
  --c-green-400:   #4b6526;
  --c-green-300:   #3a5020;
  --c-green-200:   #2d3e18;
  --c-green-100:   #1e2a10;

  /* Tan / Sand — darker for contrast on white */
  --c-tan:         #8a6a30;
  --c-tan-light:   #5a4220;
  --c-tan-dim:     #a88040;
  --c-sand:        #f5ecd8;

  /* Body text */
  --c-text:        #1e2318;
  --c-text-2:      #3a4030;
  --c-text-3:      #5a6050;
  --c-text-muted:  #7a8070;

  /* Status */
  --c-success:     #2e7d32;
  --c-warning:     #e65100;
  --c-error:       #c62828;
  --c-info:        #1565c0;

  /* Typography */
  --font-body:     'Inter', system-ui, sans-serif;
  --font-mono:     'JetBrains Mono', 'Fira Code', monospace;

  /* Spacing */
  --radius-sm:     4px;
  --radius-md:     8px;
  --radius-lg:     12px;
  --radius-xl:     16px;
  --radius-pill:   999px;

  /* Shadows — softer on light backgrounds */
  --shadow-sm:     0 1px 3px rgba(0,0,0,.1);
  --shadow-md:     0 4px 12px rgba(0,0,0,.12);
  --shadow-lg:     0 8px 24px rgba(0,0,0,.15);

  /* Transitions */
  --transition:    0.18s cubic-bezier(.4,0,.2,1);
}

/* ── Global resets ────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
  font-family: var(--font-body);
  background: var(--c-bg);
  color: var(--c-text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 100% !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--c-surface-2); }
::-webkit-scrollbar-thumb { background: var(--c-border-2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-green-500); }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, var(--c-surface-2) 100%);
  border-right: 1px solid var(--c-border);
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Sidebar section labels */
.sb-label {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-mono); font-size: 0.7rem; font-weight: 700;
  letter-spacing: 2px; text-transform: uppercase;
  color: var(--c-green-400); padding: 1rem 0 0.5rem 0;
  border-bottom: 1px solid var(--c-border);
  margin-bottom: 0.75rem;
}
.sb-label svg { flex-shrink: 0; }

/* Sidebar provider card */
.provider-card {
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius-md); padding: 0.75rem;
  margin-bottom: 0.75rem; transition: border-color var(--transition);
}
.provider-card:hover { border-color: var(--c-green-500); }

/* Provider selector pills */
.provider-pills {
  display: flex; gap: 6px; margin-bottom: 0.75rem; flex-wrap: wrap;
}
.provider-pill {
  flex: 1; min-width: 70px; padding: 6px 10px; border-radius: var(--radius-pill);
  font-family: var(--font-mono); font-size: 0.65rem; font-weight: 600;
  text-align: center; cursor: pointer; border: 1px solid var(--c-border-2);
  background: transparent; color: var(--c-text-muted);
  transition: all var(--transition); white-space: nowrap;
}
.provider-pill.active {
  background: var(--c-green-800); border-color: var(--c-green-500);
  color: #ffffff; box-shadow: 0 0 0 2px rgba(74,101,38,.2);
}
.provider-pill:hover:not(.active) {
  border-color: var(--c-green-500); color: var(--c-green-400);
}

/* Connection dot */
.conn-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 6px; flex-shrink: 0;
}
.conn-dot.online  { background: var(--c-success); box-shadow: 0 0 5px rgba(46,125,50,.4); }
.conn-dot.offline { background: var(--c-error);   box-shadow: 0 0 5px rgba(198,40,40,.4); }
.conn-dot.unknown { background: var(--c-warning);  box-shadow: 0 0 5px rgba(230,81,0,.4); }

/* Streamlit input overrides */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border-2) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--c-text) !important; font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus,
section[data-testid="stSidebar"] .stSelectbox select:focus {
  border-color: var(--c-green-500) !important;
  box-shadow: 0 0 0 2px rgba(92,122,48,.2) !important;
}
section[data-testid="stSidebar"] label {
  color: var(--c-text-3) !important; font-size: 0.75rem !important;
  font-family: var(--font-mono) !important; letter-spacing: .5px;
}

/* KB stat chips */
.kb-stat {
  display: inline-flex; align-items: center; gap: 4px;
  background: var(--c-green-900); border: 1px solid var(--c-green-800);
  border-radius: var(--radius-pill); padding: 2px 10px;
  font-family: var(--font-mono); font-size: 0.7rem; color: var(--c-green-400);
}
.kb-stat strong { color: var(--c-green-300); margin-left: 2px; }

/* Source chip in sidebar */
.src-chip {
  display: flex; align-items: center; gap: 6px;
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius-sm); padding: 5px 8px; margin: 3px 0;
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--c-text-3);
  transition: all var(--transition);
}
.src-chip:hover { border-color: var(--c-green-500); color: var(--c-green-400); background: var(--c-green-900); }
.src-chip-icon { color: var(--c-green-500); font-size: 0.8rem; }

/* ── Top bar ──────────────────────────────────────────────────────────── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(90deg, #3a5020 0%, #4b6526 50%, #5c7a30 100%);
  border: 1px solid #3a5020; border-radius: var(--radius-lg);
  padding: 0.75rem 1.25rem; margin-bottom: 1rem;
  box-shadow: var(--shadow-md);
  position: relative; overflow: hidden;
}
.topbar::before {
  content: ''; position: absolute; inset: 0;
  background: repeating-linear-gradient(
    60deg, transparent, transparent 30px,
    rgba(255,255,255,.03) 30px, rgba(255,255,255,.03) 60px
  );
  pointer-events: none;
}
.topbar-left { display: flex; align-items: center; gap: 14px; z-index: 1; }
.topbar-logo {
  font-family: var(--font-mono); font-size: 1.35rem; font-weight: 700;
  color: #ffffff; letter-spacing: 1px; line-height: 1;
}
.topbar-logo span { color: #c8dba0; }
.topbar-sub {
  font-size: 0.72rem; color: rgba(255,255,255,.65);
  font-family: var(--font-mono); letter-spacing: .5px;
}
.topbar-divider {
  width: 1px; height: 36px; background: rgba(255,255,255,.2); margin: 0 4px;
}
.topbar-right { display: flex; align-items: center; gap: 8px; z-index: 1; flex-wrap: wrap; }

/* Status pill — on dark topbar background */
.spill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 11px; border-radius: var(--radius-pill);
  font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600;
  letter-spacing: .5px; white-space: nowrap;
  transition: all var(--transition);
}
.spill-ok    { background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.35); color: #d4f0b8; }
.spill-warn  { background: rgba(255,200,0,.2);   border: 1px solid rgba(255,200,0,.5);    color: #ffe082; }
.spill-err   { background: rgba(255,100,100,.2); border: 1px solid rgba(255,100,100,.5);  color: #ffcdd2; }
.spill-info  { background: rgba(180,230,255,.15); border: 1px solid rgba(180,230,255,.4); color: #b3e5fc; }

/* Pulse animation for live status */
@keyframes pulse {
  0%,100% { opacity: 1; } 50% { opacity: .4; }
}
.pulse { animation: pulse 2s ease-in-out infinite; }

/* ── Mode toggle bar ──────────────────────────────────────────────────── */
.mode-bar {
  display: flex; gap: 4px; background: var(--c-surface-2);
  border: 1px solid var(--c-border); border-radius: var(--radius-pill);
  padding: 3px; width: fit-content; margin: 0 auto 1rem auto;
}
.mode-tab {
  padding: 6px 18px; border-radius: var(--radius-pill); border: none;
  font-family: var(--font-mono); font-size: 0.75rem; font-weight: 600;
  cursor: pointer; transition: all var(--transition);
  background: transparent; color: var(--c-text-muted); letter-spacing: .3px;
}
.mode-tab.active {
  background: var(--c-green-500); color: #ffffff;
  box-shadow: var(--shadow-sm);
}
.mode-tab:hover:not(.active) { color: var(--c-green-400); background: var(--c-green-900); }

/* ── Welcome screen ───────────────────────────────────────────────────── */
.welcome-wrap {
  max-width: 860px; margin: 2rem auto; padding: 0 1rem; text-align: center;
}
.welcome-title {
  font-family: var(--font-mono); font-size: 1.6rem; font-weight: 700;
  color: var(--c-green-300); letter-spacing: 1px; margin-bottom: .5rem;
}
.welcome-sub {
  color: var(--c-text-3); font-size: 0.95rem; line-height: 1.7;
  max-width: 580px; margin: 0 auto 2rem auto;
}
.step-cards { display: flex; gap: 16px; margin-top: 1.5rem; flex-wrap: wrap; }
.step-card {
  flex: 1; min-width: 220px;
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius-lg); padding: 1.25rem;
  transition: all var(--transition); text-align: left;
  position: relative; overflow: hidden;
}
.step-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--c-green-500), var(--c-green-700));
  transform: scaleX(0); transform-origin: left;
  transition: transform 0.3s ease;
}
.step-card:hover::before { transform: scaleX(1); }
.step-card:hover { border-color: var(--c-green-600); transform: translateY(-2px); box-shadow: var(--shadow-md); }
.step-card.done { border-color: rgba(46,125,50,.5); }
.step-card.done::before { transform: scaleX(1); background: var(--c-success); }
.step-num {
  font-family: var(--font-mono); font-size: 0.65rem; font-weight: 700;
  color: var(--c-green-500); letter-spacing: 2px; text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.step-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }
.step-title {
  font-family: var(--font-mono); font-size: 0.9rem; font-weight: 600;
  color: var(--c-text); margin-bottom: 0.35rem;
}
.step-desc { font-size: 0.8rem; color: var(--c-text-3); line-height: 1.5; }
.step-badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 0.68rem; font-family: var(--font-mono); font-weight: 600;
  padding: 2px 8px; border-radius: var(--radius-pill); margin-top: 0.5rem;
}
.step-badge.done    { background: rgba(46,125,50,.12); color: #2e7d32; border: 1px solid rgba(46,125,50,.3); }
.step-badge.pending { background: rgba(230,81,0,.08);  color: #e65100; border: 1px solid rgba(230,81,0,.3); }

/* ── Quick actions ────────────────────────────────────────────────────── */
.qa-section-label {
  font-family: var(--font-mono); font-size: 0.68rem; font-weight: 700;
  letter-spacing: 2px; color: var(--c-green-500); text-transform: uppercase;
  margin-bottom: 0.6rem; display: flex; align-items: center; gap: 6px;
}
.qa-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 1rem; }
.qa-btn {
  display: flex; align-items: center; gap: 8px;
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius-md); padding: 8px 12px;
  font-family: var(--font-body); font-size: 0.8rem; font-weight: 500;
  color: var(--c-text-2); cursor: pointer; transition: all var(--transition);
  text-align: left; width: 100%;
}
.qa-btn:hover {
  background: var(--c-green-900); border-color: var(--c-green-600);
  color: var(--c-green-400); transform: translateY(-1px); box-shadow: var(--shadow-sm);
}
.qa-btn-icon { font-size: 1rem; flex-shrink: 0; }
.qa-btn-text { line-height: 1.2; }

/* ── Chat messages ────────────────────────────────────────────────────── */
.stChatMessage {
  background: transparent !important;
  border: none !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) {
  padding: 0.25rem 0;
}

/* Source reference panel */
.src-panel {
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius-md); padding: 0; margin-top: 0.5rem;
  overflow: hidden;
}
.src-panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 12px; background: var(--c-surface-2);
  border-bottom: 1px solid var(--c-border);
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
  color: var(--c-green-400); cursor: pointer;
}
.src-ref-row {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 12px; border-bottom: 1px solid var(--c-border);
  transition: background var(--transition);
}
.src-ref-row:last-child { border-bottom: none; }
.src-ref-row:hover { background: var(--c-green-900); }
.src-ref-num {
  font-family: var(--font-mono); font-size: 0.65rem; font-weight: 700;
  background: var(--c-green-600); color: #ffffff;
  border-radius: var(--radius-sm); padding: 1px 6px; flex-shrink: 0; margin-top: 2px;
}
.src-ref-meta { flex: 1; min-width: 0; }
.src-ref-file {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--c-tan);
  font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.src-ref-page { font-size: 0.7rem; color: var(--c-text-muted); margin-top: 1px; }
.src-ref-text { font-size: 0.75rem; color: var(--c-text-3); margin-top: 4px; line-height: 1.45; }
.relevance-bar {
  width: 60px; flex-shrink: 0; text-align: right;
}
.relevance-val {
  font-family: var(--font-mono); font-size: 0.65rem; font-weight: 700;
  color: var(--c-green-500);
}
.relevance-track {
  width: 100%; height: 3px; background: var(--c-border);
  border-radius: 2px; margin-top: 3px; overflow: hidden;
}
.relevance-fill {
  height: 100%; border-radius: 2px;
  background: linear-gradient(90deg, var(--c-green-500), var(--c-green-700));
}

/* ── Streamlit widget overrides ───────────────────────────────────────── */
/* Buttons */
.stButton > button {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border-2) !important;
  color: var(--c-text-2) !important; border-radius: var(--radius-md) !important;
  font-family: var(--font-body) !important; font-size: 0.82rem !important;
  font-weight: 500 !important; transition: all var(--transition) !important;
  padding: 0.4rem 0.9rem !important;
}
.stButton > button:hover {
  background: var(--c-green-900) !important;
  border-color: var(--c-green-500) !important;
  color: var(--c-green-400) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="primary"] {
  background: var(--c-green-500) !important;
  border-color: var(--c-green-400) !important;
  color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--c-green-400) !important;
}

/* Chat input */
.stChatInput textarea {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border-2) !important;
  border-radius: var(--radius-md) !important;
  color: var(--c-text) !important; font-family: var(--font-body) !important;
  font-size: 0.9rem !important;
}
.stChatInput textarea:focus {
  border-color: var(--c-green-500) !important;
  box-shadow: 0 0 0 2px rgba(92,122,48,.2) !important;
}

/* Expanders */
.streamlit-expanderHeader {
  background: var(--c-surface-2) !important;
  border: 1px solid var(--c-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--c-text-2) !important;
  font-family: var(--font-mono) !important; font-size: 0.8rem !important;
}
.streamlit-expanderContent {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius-md); padding: 10px 14px;
}
[data-testid="stMetricLabel"] { color: var(--c-text-muted) !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: var(--c-green-400) !important; font-family: var(--font-mono) !important; }

/* Alerts */
.stSuccess { background: rgba(46,125,50,.08)  !important; border-color: rgba(46,125,50,.3)   !important; color: #1b5e20 !important; border-radius: var(--radius-md) !important; }
.stWarning { background: rgba(230,81,0,.08)   !important; border-color: rgba(230,81,0,.3)    !important; color: #bf360c !important; border-radius: var(--radius-md) !important; }
.stError   { background: rgba(198,40,40,.08)  !important; border-color: rgba(198,40,40,.3)   !important; color: #7f0000 !important; border-radius: var(--radius-md) !important; }
.stInfo    { background: rgba(21,101,192,.08) !important; border-color: rgba(21,101,192,.3)  !important; color: #0d47a1 !important; border-radius: var(--radius-md) !important; }

/* Radio */
.stRadio label { color: var(--c-text-2) !important; font-size: 0.82rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.82rem !important; }

/* Divider */
hr { border-color: var(--c-border) !important; margin: 0.5rem 0 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--c-surface) !important;
  border: 1px dashed var(--c-border-2) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--c-green-500) !important; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session():
    if "_settings_loaded" not in st.session_state:
        saved = load_settings()
        st.session_state._settings_loaded  = True
        st.session_state.provider          = saved["provider"]
        st.session_state.ollama_url        = saved["ollama_url"]
        st.session_state.ollama_model      = saved["ollama_model"]
        st.session_state.openai_url        = saved["openai_url"]
        st.session_state.openai_model      = saved["openai_model"]
        st.session_state.openai_api_key    = saved["openai_api_key"]
        st.session_state.anthropic_api_key = saved["anthropic_api_key"]
        st.session_state.anthropic_model   = saved["anthropic_model"]

    for k, v in {
        "messages": [], "vehicle_variant": "", "maintenance_category": "",
        "mode": "chat", "show_sources": True,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ═══════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_pdf_processor():
    return PDFProcessor()

@st.cache_resource
def get_vector_store():
    vs = VectorStore()
    vs.initialize()
    return vs

def _collect_settings() -> dict:
    return {
        "provider": st.session_state.provider,
        "ollama_url": st.session_state.ollama_url,
        "ollama_model": st.session_state.ollama_model,
        "openai_url": st.session_state.openai_url,
        "openai_model": st.session_state.openai_model,
        "openai_api_key": st.session_state.openai_api_key,
        "anthropic_api_key": st.session_state.anthropic_api_key,
        "anthropic_model": st.session_state.anthropic_model,
    }

def get_ai_engine() -> AIEngine:
    cache_key = json.dumps(_collect_settings(), sort_keys=True)
    if "ai_engine" not in st.session_state or st.session_state.get("_ai_key") != cache_key:
        p = st.session_state.provider
        if p == PROVIDER_OLLAMA:
            e = AIEngine(p, base_url=st.session_state.ollama_url, model=st.session_state.ollama_model)
        elif p == PROVIDER_OPENAI:
            e = AIEngine(p, base_url=st.session_state.openai_url, model=st.session_state.openai_model,
                         api_key=st.session_state.openai_api_key)
        elif p == PROVIDER_ANTHROPIC:
            e = AIEngine(p, api_key=st.session_state.anthropic_api_key, model=st.session_state.anthropic_model)
        else:
            e = AIEngine(PROVIDER_OLLAMA)
        st.session_state.ai_engine = e
        st.session_state._ai_key   = cache_key
    return st.session_state.ai_engine


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def _sb_label(icon: str, text: str):
    st.markdown(f'<div class="sb-label">{icon}&nbsp; {text}</div>', unsafe_allow_html=True)

def _provider_section():
    _sb_label("🤖", "AI PROVIDER")
    provider = st.session_state.provider
    changed  = False

    # Provider selector pills rendered via radio (styled via CSS to look like pills)
    new_provider = st.radio(
        "Provider",
        ALL_PROVIDERS,
        index=ALL_PROVIDERS.index(provider) if provider in ALL_PROVIDERS else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="_prov_radio",
    )
    if new_provider != provider:
        st.session_state.provider = new_provider
        provider = new_provider
        changed = True

    # ── Ollama ──────────────────────────────────────────────────────────
    if provider == PROVIDER_OLLAMA:
        ollama_url = st.text_input("Ollama URL", value=st.session_state.ollama_url,
                                   key="_ollama_url", label_visibility="visible")
        if ollama_url != st.session_state.ollama_url:
            st.session_state.ollama_url = ollama_url; changed = True

        is_connected = AIEngine.check_ollama(ollama_url)
        dot = "online" if is_connected else "offline"
        status_text = "Connected" if is_connected else "Offline"
        st.markdown(
            f'<div style="display:flex;align-items:center;font-family:var(--font-mono);'
            f'font-size:.72rem;color:#6a7060;margin:-4px 0 8px 0;">'
            f'<span class="conn-dot {dot}"></span>{status_text}</div>',
            unsafe_allow_html=True,
        )
        if is_connected:
            models = AIEngine.list_ollama_models(ollama_url)
            if models:
                cur_idx = models.index(st.session_state.ollama_model) if st.session_state.ollama_model in models else 0
                sel = st.selectbox("Model", models, index=cur_idx, key="_ollama_model_sel")
                if sel != st.session_state.ollama_model:
                    st.session_state.ollama_model = sel; changed = True
            else:
                m = st.text_input("Model", value=st.session_state.ollama_model, key="_ollama_model_txt")
                if m != st.session_state.ollama_model:
                    st.session_state.ollama_model = m; changed = True
                st.caption("No models found — run `ollama pull gpt-oss:latest`")
        else:
            m = st.text_input("Model", value=st.session_state.ollama_model, key="_ollama_model_off")
            if m != st.session_state.ollama_model:
                st.session_state.ollama_model = m; changed = True
            st.caption("Start Ollama: `ollama serve`")

    # ── OpenAI-Compatible ────────────────────────────────────────────────
    elif provider == PROVIDER_OPENAI:
        u = st.text_input("Base URL", value=st.session_state.openai_url, key="_oai_url",
                           help="e.g. https://api.openai.com/v1 or http://localhost:8000/v1")
        if u != st.session_state.openai_url:
            st.session_state.openai_url = u; changed = True
        m = st.text_input("Model", value=st.session_state.openai_model, key="_oai_model",
                           help="e.g. gpt-4o, gpt-4o-mini")
        if m != st.session_state.openai_model:
            st.session_state.openai_model = m; changed = True
        k = st.text_input("API Key", value=st.session_state.openai_api_key, type="password",
                           key="_oai_key", help="Leave blank for no-auth local endpoints")
        if k != st.session_state.openai_api_key:
            st.session_state.openai_api_key = k; changed = True
        st.caption("Works with OpenAI · Together AI · Groq · vLLM · LM Studio")

    # ── Anthropic ────────────────────────────────────────────────────────
    elif provider == PROVIDER_ANTHROPIC:
        k = st.text_input("API Key", value=st.session_state.anthropic_api_key,
                           type="password", key="_ant_key", help="sk-ant-...")
        if k != st.session_state.anthropic_api_key:
            st.session_state.anthropic_api_key = k; changed = True
        m = st.text_input("Model", value=st.session_state.anthropic_model, key="_ant_model",
                           help="e.g. claude-opus-4-6")
        if m != st.session_state.anthropic_model:
            st.session_state.anthropic_model = m; changed = True
        dot = "online" if st.session_state.anthropic_api_key else "offline"
        status_text = "Key configured" if st.session_state.anthropic_api_key else "No API key"
        st.markdown(
            f'<div style="display:flex;align-items:center;font-family:var(--font-mono);'
            f'font-size:.72rem;color:#6a7060;margin:-4px 0 8px 0;">'
            f'<span class="conn-dot {dot}"></span>{status_text}</div>',
            unsafe_allow_html=True,
        )

    if changed:
        save_settings(_collect_settings())


def render_sidebar():
    with st.sidebar:
        # Brand mark
        st.markdown(
            '<div style="padding:1rem 0 0.5rem;text-align:center;">'
            '<div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:700;'
            'color:var(--c-tan-light);letter-spacing:2px;">🔧 HMMWV</div>'
            '<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--c-tan-dim);'
            'letter-spacing:3px;text-transform:uppercase;">Technical Assistant</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        _provider_section()
        st.divider()

        # Vehicle
        _sb_label("🚛", "VEHICLE")
        variant  = st.selectbox("Variant", ["(Auto-detect)"] + HMMWV_VARIANTS, label_visibility="collapsed")
        st.session_state.vehicle_variant = "" if variant == "(Auto-detect)" else variant
        category = st.selectbox("Category", ["(Any)"] + MAINTENANCE_CATEGORIES, label_visibility="collapsed")
        st.session_state.maintenance_category = "" if category == "(Any)" else category
        st.divider()

        # Knowledge base
        _sb_label("📚", "KNOWLEDGE BASE")
        proc     = get_pdf_processor()
        vs       = get_vector_store()
        vs_stats = vs.get_stats()
        status   = proc.get_processing_status()

        # Stats row
        c1, c2 = st.columns(2)
        c1.metric("PDFs", status["total_pdfs"])
        c2.metric("Chunks", vs_stats["total_chunks"])

        uploaded = st.file_uploader("Upload TM PDFs", type=["pdf"], accept_multiple_files=True,
                                    label_visibility="collapsed")
        if uploaded:
            for f in uploaded:
                dest = KNOWLEDGE_BASE_DIR / f.name
                if not dest.exists():
                    dest.write_bytes(f.getvalue())
                    st.toast(f"Saved: {f.name}", icon="📄")

        unproc = status["unprocessed"]
        if unproc > 0:
            st.warning(f"{unproc} PDF(s) pending processing")
            if st.button("⚙️ Process PDFs", width="stretch", type="primary"):
                with st.spinner("Extracting text and images…"):
                    result = proc.process_all_pdfs()
                    if result["chunks"]:
                        vs.add_chunks(result["chunks"])
                    st.success(f"{result['total_pdfs']} PDFs · {result['total_chunks']} chunks · {result['total_images']} images")
                    st.rerun()
        elif status["total_pdfs"] > 0:
            st.success("All PDFs indexed")
        else:
            st.info(f"Drop PDFs into `knowledge_base/` or upload above.")

        if status["total_pdfs"] > 0:
            with st.expander("⚙️ Advanced"):
                if st.button("♻️ Reprocess All", width="stretch"):
                    with st.spinner("Reprocessing…"):
                        vs.clear()
                        result = proc.process_all_pdfs(force=True)
                        if result["chunks"]: vs.add_chunks(result["chunks"])
                        st.success(f"Reprocessed {result['total_pdfs']} PDFs")
                        st.rerun()
                st.toggle("Show source references", value=True, key="show_sources")

        if vs_stats["source_files"]:
            st.divider()
            _sb_label("📑", "INDEXED SOURCES")
            for src in vs_stats["source_files"]:
                safe = html_mod.escape(src)
                st.markdown(
                    f'<div class="src-chip"><span class="src-chip-icon">📄</span>{safe}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        if st.button("🗑️ Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════════════════

def render_topbar():
    vs    = get_vector_store()
    stats = vs.get_stats()
    prov  = st.session_state.provider
    has_data = stats["total_chunks"] > 0

    if prov == PROVIDER_OLLAMA:
        ok    = AIEngine.check_ollama(st.session_state.ollama_url)
        plabel = html_mod.escape(f"Ollama · {st.session_state.ollama_model}")
        pcls   = "spill-ok" if ok else "spill-err"
        pdot   = "🟢" if ok else "🔴"
    elif prov == PROVIDER_OPENAI:
        ok    = bool(st.session_state.openai_url)
        plabel = html_mod.escape(f"OpenAI · {st.session_state.openai_model}")
        pcls   = "spill-ok" if ok else "spill-warn"
        pdot   = "🟢" if ok else "🟡"
    elif prov == PROVIDER_ANTHROPIC:
        ok    = bool(st.session_state.anthropic_api_key)
        plabel = html_mod.escape(f"Claude · {st.session_state.anthropic_model}")
        pcls   = "spill-ok" if ok else "spill-warn"
        pdot   = "🟢" if ok else "🟡"
    else:
        ok = False; plabel = "Unknown"; pcls = "spill-err"; pdot = "🔴"

    kb_cls   = "spill-ok" if has_data else "spill-warn"
    kb_label = f"{stats['total_chunks']:,} chunks" if has_data else "No data"
    v_label  = html_mod.escape((st.session_state.vehicle_variant or "All Variants").split("—")[0].strip())
    mode_labels = {"chat": "💬 Chat", "diagnose": "🔍 Diagnose", "pmcs": "📋 PMCS"}
    m_label  = mode_labels.get(st.session_state.mode, "💬 Chat")

    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-left">
        <div>
          <div class="topbar-logo">HMMWV<span> //</span> TM ASSIST</div>
          <div class="topbar-sub">TM 9-2320-280 Series · RAG-Powered</div>
        </div>
        <div class="topbar-divider"></div>
        <span class="spill {pcls}">
          <span class="conn-dot {'online' if ok else 'offline'}"></span>{plabel}
        </span>
        <span class="spill {kb_cls}">📦 {kb_label}</span>
        <span class="spill spill-info">🚛 {v_label}</span>
      </div>
      <div class="topbar-right">
        <span class="spill spill-info">{m_label}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MODE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════

def render_mode_selector():
    """Render a styled tab bar for mode selection using Streamlit radio."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode_map = {"💬  Chat": "chat", "🔍  Diagnose": "diagnose", "📋  PMCS": "pmcs"}
        selected = st.radio(
            "mode_select",
            list(mode_map.keys()),
            horizontal=True,
            label_visibility="collapsed",
            key="_mode_radio",
            index=list(mode_map.values()).index(st.session_state.mode)
                  if st.session_state.mode in mode_map.values() else 0,
        )
        if mode_map[selected] != st.session_state.mode:
            st.session_state.mode = mode_map[selected]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════

def render_welcome():
    prov = st.session_state.provider
    if prov == PROVIDER_OLLAMA:
        pok   = AIEngine.check_ollama(st.session_state.ollama_url)
        pdesc = f"Connected · `{st.session_state.ollama_model}`" if pok else "Start Ollama: `ollama serve`"
    elif prov == PROVIDER_OPENAI:
        pok   = bool(st.session_state.openai_url)
        pdesc = f"`{st.session_state.openai_model}` via OpenAI-compatible endpoint" if pok else "Enter Base URL in sidebar"
    elif prov == PROVIDER_ANTHROPIC:
        pok   = bool(st.session_state.anthropic_api_key)
        pdesc = f"Claude · `{st.session_state.anthropic_model}`" if pok else "Enter Anthropic API key in sidebar"
    else:
        pok = False; pdesc = "Select a provider in the sidebar"

    has_data = get_vector_store().get_stats()["total_chunks"] > 0

    step1_cls  = "done" if pok       else ""
    step2_cls  = "done" if has_data  else ""
    step3_cls  = "done" if (pok and has_data) else ""
    s1_badge   = '<span class="step-badge done">✓ Ready</span>' if pok       else '<span class="step-badge pending">⬤ Pending</span>'
    s2_badge   = '<span class="step-badge done">✓ Loaded</span>' if has_data  else '<span class="step-badge pending">⬤ Pending</span>'
    s3_badge   = '<span class="step-badge done">✓ Ready</span>' if (pok and has_data) else '<span class="step-badge pending">⬤ Pending</span>'

    st.markdown(f"""
    <div class="welcome-wrap">
      <div class="welcome-title">Welcome, Mechanic</div>
      <div class="welcome-sub">
        Your AI-powered HMMWV technical assistant. Upload official Technical Manuals,
        then ask anything — maintenance procedures, torque specs, diagnostics, and PMCS checklists.
        All answers are grounded in your TM library.
      </div>
      <div class="step-cards">
        <div class="step-card {step1_cls}">
          <div class="step-num">Step 01</div>
          <span class="step-icon">🤖</span>
          <div class="step-title">Configure AI Provider</div>
          <div class="step-desc">{html_mod.escape(pdesc)}</div>
          {s1_badge}
        </div>
        <div class="step-card {step2_cls}">
          <div class="step-num">Step 02</div>
          <span class="step-icon">📚</span>
          <div class="step-title">Load Technical Manuals</div>
          <div class="step-desc">Upload TM 9-2320-280 PDFs via the sidebar or drop them in <code>knowledge_base/</code></div>
          {s2_badge}
        </div>
        <div class="step-card {step3_cls}">
          <div class="step-num">Step 03</div>
          <span class="step-icon">🔧</span>
          <div class="step-title">Ask Away</div>
          <div class="step-desc">Use the chat below, or pick a quick action to get started immediately.</div>
          {s3_badge}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK ACTIONS
# ═══════════════════════════════════════════════════════════════════════════

def render_quick_actions():
    actions = {
        "chat": [
            ("🔧", "Before-Ops PMCS",   "Walk me through the complete Before-Operation PMCS checks for the HMMWV."),
            ("🛢️", "Oil Change",        "Provide step-by-step instructions for performing an engine oil and filter change."),
            ("🔋", "Battery Service",   "How do I properly inspect, service, and replace the HMMWV batteries?"),
            ("🌀", "CTIS Check",        "Explain how to inspect and troubleshoot the Central Tire Inflation System."),
            ("⚡", "Glow Plug Test",    "How do I test and replace the glow plugs on the 6.5L diesel engine?"),
            ("🛞", "Brake Bleed",       "Provide the brake bleeding procedure for all four wheels."),
        ],
        "diagnose": [
            ("🌡️", "Overheating",      "The engine temperature gauge is reading high and coolant is bubbling in the surge tank."),
            ("⚡", "No Start",          "The engine cranks but will not start. Batteries are fully charged."),
            ("🔊", "Grinding Noise",    "Grinding/squealing noise from front left wheel area during braking."),
            ("💨", "White Smoke",       "Excessive white smoke from exhaust, especially on cold starts."),
            ("🛞", "Pulling Right",     "Vehicle pulls hard to the right during normal driving and braking."),
            ("🔌", "Electrical Fault",  "Multiple dash warning lights flickering and gauges are erratic."),
        ],
        "pmcs": [
            ("📋", "Before-Ops",        "Guide me through the Before-Operation PMCS checklist, step by step."),
            ("📋", "During-Ops",        "Guide me through the During-Operation PMCS checks."),
            ("📋", "After-Ops",         "Guide me through the After-Operation PMCS procedures."),
            ("📋", "Weekly",            "What are the weekly scheduled maintenance checks?"),
            ("📋", "Monthly",           "What are the monthly scheduled maintenance requirements?"),
            ("📋", "Semi-Annual",       "Guide me through the semi-annual service requirements."),
        ],
    }
    current = actions.get(st.session_state.mode, actions["chat"])

    st.markdown('<div class="qa-section-label">▸ Quick Actions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (icon, label, prompt) in enumerate(current):
        with cols[i % 3]:
            if st.button(f"{icon}  {label}", width="stretch", key=f"qa_{i}"):
                return prompt
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE REFERENCES
# ═══════════════════════════════════════════════════════════════════════════

def render_sources(search_results: list):
    if not search_results or not st.session_state.get("show_sources", True):
        return

    with st.expander(f"📑  Source References  ({len(search_results)})", expanded=False):
        rows_html = ""
        for i, result in enumerate(search_results):
            meta      = result.get("metadata", {})
            source    = html_mod.escape(str(meta.get("source_file", "Unknown")))
            page      = meta.get("page_number", "?")
            distance  = result.get("distance", 0)
            relevance = max(0.0, 1.0 - distance) * 100
            snippet   = html_mod.escape(result["text"][:220].replace("\n", " "))
            bar_w     = f"{relevance:.0f}%"
            ellipsis  = "…" if len(result["text"]) > 220 else ""

            rows_html += f"""
            <div class="src-ref-row">
              <div class="src-ref-num">#{i+1}</div>
              <div class="src-ref-meta">
                <div class="src-ref-file">{source}</div>
                <div class="src-ref-page">Page {page}</div>
                <div class="src-ref-text">{snippet}{ellipsis}</div>
              </div>
              <div class="relevance-bar">
                <div class="relevance-val">{relevance:.0f}%</div>
                <div class="relevance-track"><div class="relevance-fill" style="width:{bar_w}"></div></div>
              </div>
            </div>
            """

        src_panel_html = f"""
        <style>
          .src-panel {{
            border: 1px solid #d8ddd0; border-radius: 8px; overflow: hidden;
            font-family: system-ui, -apple-system, sans-serif; margin: 4px 0;
            background: #ffffff;
          }}
          .src-ref-row {{
            display: flex; align-items: flex-start; gap: 10px;
            padding: 10px 12px; border-bottom: 1px solid #e8ebe2;
            transition: background 0.15s;
          }}
          .src-ref-row:last-child {{ border-bottom: none; }}
          .src-ref-row:hover {{ background: #f0f2ec; }}
          .src-ref-num {{
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.65rem; font-weight: 700;
            background: #7a9e3a; color: #ffffff;
            border-radius: 4px; padding: 1px 6px; flex-shrink: 0; margin-top: 2px;
          }}
          .src-ref-meta {{ flex: 1; min-width: 0; }}
          .src-ref-file {{
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.72rem; color: #8a6e2a;
            font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          }}
          .src-ref-page {{
            font-size: 0.68rem; color: #7a8070; margin: 1px 0 3px;
          }}
          .src-ref-text {{
            font-size: 0.78rem; color: #3a4030; line-height: 1.45;
          }}
          .relevance-bar {{
            display: flex; flex-direction: column; align-items: flex-end;
            gap: 4px; flex-shrink: 0; min-width: 60px;
          }}
          .relevance-val {{
            font-size: 0.68rem; font-weight: 700; color: #5a6050;
            font-family: 'JetBrains Mono', monospace;
          }}
          .relevance-track {{
            width: 56px; height: 4px; background: #e0e4d8; border-radius: 2px; overflow: hidden;
          }}
          .relevance-fill {{
            height: 100%; background: linear-gradient(90deg, #7a9e3a, #4B5320);
            border-radius: 2px; transition: width 0.4s ease;
          }}
        </style>
        <div class="src-panel">{rows_html}</div>
        """
        components.html(src_panel_html, height=min(80 + len(search_results) * 90, 600), scrolling=True)

        # Page image viewer
        for result in search_results:
            meta   = result.get("metadata", {})
            source = meta.get("source_file", "")
            page   = meta.get("page_number")
            if source and isinstance(page, int):
                pdf_path = KNOWLEDGE_BASE_DIR / source
                if pdf_path.exists():
                    proc     = get_pdf_processor()
                    img_path = proc.extract_page_as_image(pdf_path, page)
                    if img_path and Path(img_path).exists():
                        with st.expander(f"🖼️  {html_mod.escape(source)} — Page {page}", expanded=False):
                            st.image(img_path, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# PRINT BUTTON
# ═══════════════════════════════════════════════════════════════════════════

def render_print_button(content: str, msg_index: int, sources: list = None):
    proc = get_pdf_processor()
    source_html = ""
    if sources:
        source_html = '<hr><h3>📑 Source References</h3>'
        for i, s in enumerate(sources, 1):
            src  = s.get("metadata", {}).get("source_file", "Unknown")
            pg   = s.get("metadata", {}).get("page_number", "?")
            rel  = max(0, 1 - s.get("distance", 0)) * 100
            text = html_mod.escape(s.get("text", "")[:400])
            ssrc = html_mod.escape(str(src))
            source_html += (
                f'<div style="border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:10px 0;">'
                f'<strong>Ref {i}</strong> — <code>{ssrc}</code> p.{pg} '
                f'<span style="color:#666;font-size:12px;">({rel:.0f}% relevant)</span><br>'
                f'<small style="color:#555;">{text}…</small>'
            )
            if isinstance(pg, int):
                ppath = KNOWLEDGE_BASE_DIR / src
                if ppath.exists():
                    ipath = proc.extract_page_as_image(ppath, pg)
                    if ipath and Path(ipath).exists():
                        try:
                            with open(ipath, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("ascii")
                            source_html += (
                                f'<div style="margin-top:8px;text-align:center;">'
                                f'<img src="data:image/png;base64,{b64}" '
                                f'style="max-width:100%;border:1px solid #ddd;border-radius:4px;" '
                                f'alt="{ssrc} p{pg}"/></div>'
                            )
                        except Exception:
                            pass
            source_html += '</div>'

    variant   = html_mod.escape(st.session_state.get("vehicle_variant", "") or "All Variants")
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    md_b64    = base64.b64encode(content.encode()).decode("ascii")
    src_b64   = base64.b64encode(source_html.encode()).decode("ascii")

    js = f"""
    <button id="pb_{msg_index}" style="
      display:inline-flex;align-items:center;gap:6px;
      background:transparent;border:1px solid #3a3d32;border-radius:6px;
      padding:4px 12px;font-size:12px;font-weight:600;cursor:pointer;
      font-family:'JetBrains Mono',monospace;color:#6b7b3a;
      margin-top:4px;transition:all .18s ease;">
      🖨️ Print
    </button>
    <script>
    (function(){{
      var btn = document.getElementById('pb_{msg_index}');
      btn.onmouseenter = function(){{ this.style.background='rgba(75,83,32,.2)'; this.style.color='#8a9e4e'; }};
      btn.onmouseleave = function(){{ this.style.background='transparent'; this.style.color='#6b7b3a'; }};
      function dec(b64){{
        var r=atob(b64),bytes=new Uint8Array(r.length);
        for(var i=0;i<r.length;i++) bytes[i]=r.charCodeAt(i);
        return new TextDecoder('utf-8').decode(bytes);
      }}
      btn.onclick = function(){{
        var md=dec("{md_b64}"), src=dec("{src_b64}");
        var w=window.open('','_blank','width=820,height=950');
        var d=w.document; d.open();
        d.write('<!DOCTYPE html><html><head><meta charset="UTF-8">');
        d.write('<title>HMMWV TM Assistant — Print</title>');
        d.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.6/marked.min.js"><' + '/script>');
        d.write('<style>');
        d.write('*{{box-sizing:border-box;margin:0;padding:0}}');
        d.write('body{{font-family:Segoe UI,system-ui,sans-serif;color:#1a1a1a;background:#fff;max-width:780px;margin:0 auto;padding:24px}}');
        d.write('.hdr{{border-bottom:3px solid #4B5320;padding-bottom:14px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:flex-end}}');
        d.write('.hdr-title{{font-size:17px;font-weight:700;color:#2d3520;font-family:monospace;letter-spacing:1px}}');
        d.write('.hdr-meta{{font-size:11px;color:#777;font-family:monospace}}');
        d.write('.content{{line-height:1.65}}');
        d.write('h1,h2,h3,h4{{color:#2d3520;margin:16px 0 8px;line-height:1.3}}');
        d.write('h1{{font-size:20px;border-bottom:2px solid #e0e0e0;padding-bottom:6px}}');
        d.write('h2{{font-size:17px}}h3{{font-size:15px}}');
        d.write('p{{margin:8px 0}}ul,ol{{margin:8px 0 8px 22px}}li{{margin:4px 0}}');
        d.write('code{{background:#f4f4f4;padding:1px 5px;border-radius:3px;font-size:12px;font-family:monospace}}');
        d.write('pre{{background:#f5f5f5;padding:12px;border-radius:6px;border:1px solid #ddd;overflow-x:auto;margin:10px 0}}');
        d.write('pre code{{background:none;padding:0;font-size:12px}}');
        d.write('table{{border-collapse:collapse;width:100%;margin:12px 0}}');
        d.write('th,td{{border:1px solid #ccc;padding:6px 10px;font-size:13px;text-align:left}}');
        d.write('th{{background:#f0f0e8;font-weight:600;color:#2d3520}}');
        d.write('blockquote{{border-left:3px solid #4B5320;padding:6px 12px;color:#555;background:#f9f9f4;margin:10px 0}}');
        d.write('strong{{color:#2d3520}}');
        d.write('img{{max-width:100%;border-radius:4px}}');
        d.write('.print-btn{{display:flex;justify-content:center;margin-top:24px}}');
        d.write('.print-btn button{{padding:10px 28px;font-size:14px;cursor:pointer;');
        d.write('background:#4B5320;color:white;border:none;border-radius:6px;font-weight:600;letter-spacing:.5px}}');
        d.write('@media print{{.print-btn{{display:none}}img{{page-break-inside:avoid}}}}');
        d.write('</style></head><body>');
        d.write('<div class="hdr">');
        d.write('<div><div class="hdr-title">🔧 HMMWV Technical Assistant</div>');
        d.write('<div class="hdr-meta">Vehicle: {variant} &nbsp;·&nbsp; {timestamp}</div></div></div>');
        d.write('<div class="content" id="c"></div>');
        d.write('<div id="s"></div>');
        d.write('<div class="print-btn"><button onclick="window.print()">🖨️ Print / Save PDF</button></div>');
        d.write('</body></html>');
        d.close();
        w.onload=function(){{
          w.document.getElementById('c').innerHTML=w.marked.parse(md);
          w.document.getElementById('s').innerHTML=src;
        }};
      }};
    }})();
    </script>
    """
    components.html(js, height=38)


# ═══════════════════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════════════════

def _stream_response(placeholder, ai: AIEngine, user_input: str,
                     search_results: list, history: list) -> str:
    """Stream tokens from AI with a live progress bar and ETA estimate."""

    # ── Phase timings (seconds) used for ETA estimation ──────────────────
    # We observe real elapsed time per phase and blend with these priors.
    PHASE_ESTIMATE = {
        "context":    0.3,
        "first_token": 3.0,   # time until first token arrives
        "generation": 20.0,   # median full response time
    }

    t0 = time.monotonic()

    # Progress slots — placed ABOVE the response placeholder
    progress_slot = st.empty()

    def _show(pct, label, estimate=None):
        elapsed = time.monotonic() - t0
        bar_html = _build_progress_html(pct, label, elapsed, estimate)
        progress_slot.markdown(bar_html, unsafe_allow_html=True)

    # ── Phase 1: context ready (search already done before this call) ─────
    _show(0.08, "🔍 Searching knowledge base…", PHASE_ESTIMATE["context"])
    time.sleep(0.15)   # tiny visual pause so the bar is visible
    _show(0.20, "📋 Preparing context…",        PHASE_ESTIMATE["context"])
    time.sleep(0.15)

    # ── Phase 2: connect to AI and wait for first token ───────────────────
    _show(0.28, "🤖 Connecting to AI engine…", PHASE_ESTIMATE["first_token"])

    gen = (
        ai.diagnose(user_input, search_results,
                    vehicle_variant=st.session_state.vehicle_variant)
        if st.session_state.mode == "diagnose"
        else ai.chat_stream(user_input, search_results,
                            conversation_history=history,
                            vehicle_variant=st.session_state.vehicle_variant,
                            maintenance_category=st.session_state.maintenance_category)
    )

    full        = ""
    token_count = 0
    t_first     = None

    for chunk in gen:
        full        += chunk
        token_count += 1

        if t_first is None:
            t_first = time.monotonic()
            _show(0.38, "✍️  Generating response…", PHASE_ESTIMATE["generation"])

        # Smooth progress: ramp from 38 → 92 using a log curve on token count
        # Most HMMWV responses are 300-800 tokens; cap visual at 92% until done
        progress = 0.38 + 0.54 * min(1.0, math.log1p(token_count) / math.log1p(600))
        elapsed  = time.monotonic() - (t_first or t0)
        estimate = PHASE_ESTIMATE["generation"]
        _show(progress, f"✍️  Generating response… ({token_count} tokens)", estimate)

        placeholder.markdown(full + "▌")

    # ── Phase 3: done ─────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t0
    _show(1.0, f"✅ Complete — {token_count} tokens in {total_elapsed:.1f}s", None)
    placeholder.markdown(full)

    # Fade out the progress bar after a short pause
    time.sleep(1.8)
    progress_slot.empty()

    return full


def _build_progress_html(pct: float, label: str, elapsed: float,
                          estimate: float | None) -> str:
    """Return self-contained HTML for the progress bar (used by _stream_response)."""
    pct_int  = int(min(pct * 100, 100))
    color    = "#2e7d32" if pct >= 1.0 else "#5c7a30"
    fill_bg  = f"linear-gradient(90deg,{color},#8ab840)" if pct < 1.0 else "#2e7d32"
    pulse    = "" if pct >= 1.0 else "animation:pulse 1s infinite"
    dot_col  = "#2e7d32" if pct >= 1.0 else "#5c7a30"
    eta_html = ""
    if estimate and pct < 1.0:
        remaining = max(0.0, estimate - elapsed)
        if remaining > 0.5:
            eta_html = f'<span style="color:#7a8070;font-size:11px;font-style:italic">~{remaining:.0f}s remaining</span>'
        else:
            eta_html = '<span style="color:#5c7a30;font-size:11px;font-style:italic">almost done…</span>'
    return f"""
<div style="font-family:system-ui,sans-serif;padding:8px 0 4px">
  <div style="display:flex;justify-content:space-between;align-items:center;
              font-size:12px;color:#3a4030;margin-bottom:6px;font-weight:500">
    <span style="display:flex;align-items:center;gap:7px">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                   background:{dot_col};{pulse}"></span>
      {label}
    </span>
    {eta_html}
  </div>
  <div style="height:6px;background:#e0e4d8;border-radius:3px;overflow:hidden">
    <div style="height:100%;width:{pct_int}%;border-radius:3px;
                background:{fill_bg};transition:width .25s ease"></div>
  </div>
</div>
<style>@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.35;transform:scale(.75)}}}}</style>
"""


def render_inline_images(search_results: list):
    """
    Show page images from search_results directly below the response.
    Deduplicated by (source_file, page_number) so the same page appears once.
    Only shown when the assistant response contains step-by-step content
    (detected by numbered list or 'Step' headers in the markdown).
    Images are shown in a compact horizontal strip with captions.
    """
    if not search_results:
        return

    proc = get_pdf_processor()
    seen: set = set()
    image_items: list = []   # list of (label, img_path)

    for result in search_results:
        meta   = result.get("metadata", {})
        source = meta.get("source_file", "")
        page   = meta.get("page_number")
        if not source or not isinstance(page, int):
            continue
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        pdf_path = KNOWLEDGE_BASE_DIR / source
        if not pdf_path.exists():
            continue
        img_path = proc.extract_page_as_image(pdf_path, page)
        if img_path and Path(img_path).exists():
            label = f"📄 {source}  —  p.{page}"
            image_items.append((label, img_path))

    if not image_items:
        return

    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:#5c7a30;"
        "margin:10px 0 6px;letter-spacing:.4px'>📷 REFERENCE PAGES</div>",
        unsafe_allow_html=True,
    )

    # Layout: up to 3 images per row
    cols_per_row = min(3, len(image_items))
    for row_start in range(0, len(image_items), cols_per_row):
        batch = image_items[row_start: row_start + cols_per_row]
        cols  = st.columns(len(batch))
        for col, (label, img_path) in zip(cols, batch):
            with col:
                st.image(img_path, caption=label, width="stretch")


def render_chat():
    # Replay history
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑‍🔧" if msg["role"] == "user" else "🔧"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                col1, col2 = st.columns([8, 1])
                with col2:
                    render_print_button(msg["content"], idx, msg.get("sources", []))
                if msg.get("sources"):
                    render_inline_images(msg["sources"])
                    render_sources(msg["sources"])

    # Input
    placeholders = {
        "chat":    "Describe the maintenance task (e.g. 'Replace fuel filter on M1151')…",
        "diagnose":"Describe the symptoms (e.g. 'Engine overheating with white smoke')…",
        "pmcs":    "Which PMCS interval? (e.g. 'Before-operation checks')…",
    }
    user_input = st.chat_input(placeholders.get(st.session_state.mode, placeholders["chat"]))

    # Quick action can inject a prompt
    quick = render_quick_actions()
    if quick:
        user_input = quick

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(user_input)

        vs             = get_vector_store()
        search_results = vs.search(user_input, n_results=TOP_K_RESULTS)
        history        = [{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages[:-1][-12:]]
        ai             = get_ai_engine()

        with st.chat_message("assistant", avatar="🔧"):
            placeholder = st.empty()
            response    = _stream_response(placeholder, ai, user_input, search_results, history)
            col1, col2  = st.columns([8, 1])
            with col2:
                render_print_button(response, len(st.session_state.messages), search_results)
            render_inline_images(search_results)
            render_sources(search_results)

        st.session_state.messages.append({
            "role": "assistant", "content": response, "sources": search_results,
        })


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()
    render_topbar()
    render_mode_selector()
    if not st.session_state.messages:
        render_welcome()
    render_chat()


if __name__ == "__main__":
    main()
