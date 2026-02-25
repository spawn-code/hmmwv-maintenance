"""
PDFProcessor — extracted from app.py, Streamlit-free.
Handles PDF text extraction, chunking, and image extraction.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from config import (
    KNOWLEDGE_BASE_DIR, EXTRACTED_IMAGES_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.knowledge_dir   = KNOWLEDGE_BASE_DIR
        self.image_dir       = EXTRACTED_IMAGES_DIR
        self._manifest_path  = self.knowledge_dir / ".processed_manifest.json"
        self._manifest       = self._load_manifest()

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

    _RE_TM_SECTION = re.compile(
        r'^(SECTION|CHAPTER|APPENDIX)\s+[IVXLCDM\d]+\.?\s+.+', re.IGNORECASE)
    _RE_TM_PARA    = re.compile(r'^\d+[-–]\d+(\.\d+)*\.\s+[A-Z].{2,}')
    _RE_TM_FIGURE  = re.compile(r'^(Figure|Fig|TABLE|Table)\s+\d+', re.IGNORECASE)
    _RE_TASK_TITLE = re.compile(r'^[A-Z][A-Z0-9 /,()-]{4,60}$')

    def _detect_section_heading(self, text: str) -> str:
        lines = text.split("\n")
        figure_heading = ""
        for line in lines[:25]:
            line = line.strip()
            if not line or len(line) < 4:
                continue
            if self._RE_TM_SECTION.match(line):
                return line[:120]
            if self._RE_TM_PARA.match(line):
                return line[:120]
            if self._RE_TASK_TITLE.match(line) and len(line) > 6:
                return line[:120]
            if not figure_heading and self._RE_TM_FIGURE.match(line):
                figure_heading = line[:120]
        return figure_heading

    def extract_text_from_pdf(self, pdf_path: Path) -> list:
        import pdfplumber
        documents = []
        current_section = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = self._clean_text(page.extract_text() or "")
                    if not text.strip():
                        continue
                    detected = self._detect_section_heading(text)
                    if detected:
                        current_section = detected
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source_file":   pdf_path.name,
                            "page_number":   page_num,
                            "total_pages":   total_pages,
                            "section_title": current_section,
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
            section    = meta.get("section_title", "")
            heading_prefix = f"[{section}]\n" if section else ""
            doc_chunks = []

            if len(text) <= CHUNK_SIZE:
                enriched = heading_prefix + text
                doc_chunks.append({
                    "text": enriched,
                    "metadata": {**meta, "chunk_index": 0, "total_chunks": 1},
                })
            else:
                paragraphs = []
                for p in text.split("\n\n"):
                    paragraphs.extend(self._split_long_paragraph(p))
                current_chunk, chunk_idx = "", 0
                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 > CHUNK_SIZE and current_chunk:
                        enriched = heading_prefix + current_chunk.strip()
                        doc_chunks.append({
                            "text": enriched,
                            "metadata": {**meta, "chunk_index": chunk_idx},
                        })
                        overlap = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                        current_chunk = overlap + "\n\n" + para
                        chunk_idx += 1
                    else:
                        current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                if current_chunk.strip():
                    enriched = heading_prefix + current_chunk.strip()
                    doc_chunks.append({
                        "text": enriched,
                        "metadata": {**meta, "chunk_index": chunk_idx},
                    })
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
                            extracted.append({
                                "image_path": str(img_path),
                                "source_file": pdf_path.name,
                                "page_number": page_num,
                                "width": w, "height": h,
                            })
                        except Exception:
                            continue
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path.name}: {e}")
        return extracted

    def extract_page_as_image(self, pdf_path: Path, page_number: int, dpi: int = 300) -> Optional[str]:
        import pdfplumber
        pdf_stem = pdf_path.stem
        out_dir = self.image_dir / pdf_stem / "pages"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"page_{page_number:04d}_{dpi}dpi.png"
        if out_path.exists():
            return str(out_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                pdf.pages[page_number - 1].to_image(resolution=dpi).save(str(out_path))
                return str(out_path)
        except Exception as e:
            logger.error(f"Error rendering page {page_number} of {pdf_path.name}: {e}")
            return None

    def process_pdf(self, pdf_path: Path) -> dict:
        documents = self.extract_text_from_pdf(pdf_path)
        chunks    = self.chunk_documents(documents)
        images    = self.extract_images_from_pdf(pdf_path)
        self._manifest[pdf_path.name] = {
            "hash":       self._file_hash(pdf_path),
            "num_chunks": len(chunks),
            "num_images": len(images),
            "num_pages":  documents[0]["metadata"]["total_pages"] if documents else 0,
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
        return {
            "total_pdfs": len(pdfs), "total_chunks": len(all_chunks),
            "total_images": len(all_images),
            "chunks": all_chunks, "images": all_images,
        }

    def get_processing_status(self) -> dict:
        all_pdfs = self.discover_pdfs()
        return {
            "total_pdfs":  len(all_pdfs),
            "processed":   len(self._manifest),
            "unprocessed": len(self.get_unprocessed_pdfs()),
            "details": {
                pdf.name: {
                    "processed": pdf.name in self._manifest,
                    **(self._manifest.get(pdf.name, {}))
                }
                for pdf in all_pdfs
            },
        }
