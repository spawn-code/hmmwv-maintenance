"""
HMMWV Technical Assistant — PDF Processor
==========================================
Handles PDF ingestion: extracts text chunks and images/diagrams.
"""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import importlib.util

# Load config by absolute path (bulletproof regardless of cwd/sys.path)
if "config" not in sys.modules:
    _cfg = Path(__file__).resolve().parent.parent / "config.py"
    _sp = importlib.util.spec_from_file_location("config", str(_cfg))
    _m = importlib.util.module_from_spec(_sp)
    sys.modules["config"] = _m
    _sp.loader.exec_module(_m)

import pdfplumber
from PIL import Image

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EXTRACTED_IMAGES_DIR,
    KNOWLEDGE_BASE_DIR,
    MIN_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Processes HMMWV technical manual PDFs into searchable chunks and images."""

    def __init__(self):
        self.knowledge_dir = KNOWLEDGE_BASE_DIR
        self.image_dir = EXTRACTED_IMAGES_DIR
        self._processed_manifest_path = self.knowledge_dir / ".processed_manifest.json"
        self._manifest = self._load_manifest()

    # ── Manifest Management ────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        if self._processed_manifest_path.exists():
            return json.loads(self._processed_manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self._processed_manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _file_hash(self, filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def _is_already_processed(self, filepath: Path) -> bool:
        fhash = self._file_hash(filepath)
        return self._manifest.get(filepath.name, {}).get("hash") == fhash

    # ── PDF Discovery ──────────────────────────────────────────────────────

    def discover_pdfs(self) -> list[Path]:
        """Find all PDF files in the knowledge_base directory."""
        pdfs = sorted(self.knowledge_dir.glob("*.pdf"))
        logger.info(f"Discovered {len(pdfs)} PDF(s) in {self.knowledge_dir}")
        return pdfs

    def get_unprocessed_pdfs(self) -> list[Path]:
        """Return only PDFs that haven't been processed yet (or changed)."""
        return [p for p in self.discover_pdfs() if not self._is_already_processed(p)]

    # ── Text Extraction ────────────────────────────────────────────────────

    def extract_text_from_pdf(self, pdf_path: Path) -> list[dict]:
        """
        Extract text from a PDF, returning a list of page-level documents.
        Each document includes metadata about source file and page number.
        """
        documents = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    # Clean up extraction artifacts
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

    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Remove excessive newlines but keep paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix common OCR issues in military docs
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        return text.strip()

    # ── Chunking ───────────────────────────────────────────────────────────

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Split page-level documents into overlapping chunks for embedding.
        Preserves metadata and adds chunk-level info.
        """
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

            # Split on paragraph boundaries where possible
            paragraphs = text.split("\n\n")
            current_chunk = ""
            chunk_idx = 0

            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {**meta, "chunk_index": chunk_idx},
                    })
                    # Keep overlap from end of current chunk
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

            # Backfill total_chunks
            for c in chunks:
                if c["metadata"].get("total_chunks") is None:
                    c["metadata"]["total_chunks"] = chunk_idx + 1

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    # ── Image Extraction ───────────────────────────────────────────────────

    def extract_images_from_pdf(self, pdf_path: Path) -> list[dict]:
        """
        Extract embedded images from PDF pages.
        Returns metadata about each extracted image.
        """
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
                            # Extract image bounding box for reference
                            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                            width = int(img["x1"] - img["x0"])
                            height = int(img["bottom"] - img["top"])

                            if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                                continue

                            # Crop the page to get the image region
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
                                "description": f"Diagram from {pdf_path.name}, page {page_num}",
                            })
                        except Exception as img_err:
                            logger.debug(f"Could not extract image {img_idx} from page {page_num}: {img_err}")
                            continue

        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path.name}: {e}")

        logger.info(f"Extracted {len(extracted)} images from {pdf_path.name}")
        return extracted

    def extract_page_as_image(self, pdf_path: Path, page_number: int) -> Optional[str]:
        """
        Render an entire PDF page as an image (for showing diagrams in context).
        Returns the path to the saved image, or None on failure.
        """
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

    # ── Full Processing Pipeline ───────────────────────────────────────────

    def process_pdf(self, pdf_path: Path) -> dict:
        """
        Full processing pipeline for a single PDF.
        Returns dict with chunks and image metadata.
        """
        logger.info(f"Processing: {pdf_path.name}")

        # Extract text and chunk
        documents = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_documents(documents)

        # Extract images
        images = self.extract_images_from_pdf(pdf_path)

        # Update manifest
        self._manifest[pdf_path.name] = {
            "hash": self._file_hash(pdf_path),
            "num_chunks": len(chunks),
            "num_images": len(images),
            "num_pages": documents[0]["metadata"]["total_pages"] if documents else 0,
        }
        self._save_manifest()

        return {"chunks": chunks, "images": images}

    def process_all_pdfs(self, force: bool = False) -> dict:
        """
        Process all PDFs in the knowledge base directory.
        Set force=True to reprocess even already-processed files.
        """
        pdfs = self.discover_pdfs() if force else self.get_unprocessed_pdfs()
        all_chunks = []
        all_images = []

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
        """Get current processing status for all PDFs."""
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
