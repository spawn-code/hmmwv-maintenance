"""
Knowledge base router — stats, index management, and page pre-rendering.
"""

import logging

from fastapi import APIRouter, BackgroundTasks

from dependencies import get_vector_store, get_pdf_processor
from models import KnowledgeStats, IndexResponse, RenderPagesResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/knowledge/stats")
def knowledge_stats() -> KnowledgeStats:
    vs = get_vector_store()
    proc = get_pdf_processor()
    vs_stats   = vs.get_stats()
    proc_stats = proc.get_processing_status()
    return KnowledgeStats(
        total_chunks=vs_stats["total_chunks"],
        num_sources=vs_stats["num_sources"],
        source_files=vs_stats["source_files"],
        total_pdfs=proc_stats["total_pdfs"],
        unprocessed_pdfs=proc_stats["unprocessed"],
    )


@router.post("/knowledge/index")
def index_pdfs() -> IndexResponse:
    """Process all unprocessed PDFs and add chunks to the vector store."""
    proc = get_pdf_processor()
    vs   = get_vector_store()
    result = proc.process_all_pdfs(force=False)
    chunks_added = 0
    if result["chunks"]:
        chunks_added = vs.add_chunks(result["chunks"])
    return IndexResponse(
        indexed=result["total_pdfs"],
        chunks_added=chunks_added,
    )


@router.post("/knowledge/render-pages")
def render_all_pages(
    background_tasks: BackgroundTasks,
    dpi: int = 150,
) -> RenderPagesResponse:
    """
    Pre-render every page of every indexed PDF as a PNG image.

    Why this matters:
    Military Technical Manual PDFs draw their diagrams (exploded views,
    assembly schematics, wiring diagrams) entirely as vector graphics.
    pdfplumber's page.images only captures embedded raster objects (photos),
    so vector diagrams are invisible to the standard image extractor.
    Rendering each page to a raster image at 150 dpi captures everything —
    text, photos, and vector diagrams alike.

    Images are cached on disk (extracted_images/{stem}/pages/page_NNNN_150dpi.png).
    The chat endpoint automatically falls back to these rendered pages when no
    embedded images are found for a search result page.

    This endpoint starts the rendering in the background and returns immediately.
    Expect ~1-3 seconds per page; a 500-page PDF takes 8-25 minutes.
    """
    proc     = get_pdf_processor()
    all_pdfs = proc.discover_pdfs()

    def _render_all():
        import pdfplumber
        total_pages   = 0
        total_skipped = 0
        for pdf_path in all_pdfs:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                for page_num in range(1, num_pages + 1):
                    result = proc.extract_page_as_image(pdf_path, page_num, dpi=dpi)
                    if result:
                        total_pages += 1
                    else:
                        total_skipped += 1
                logger.info(
                    f"[render-pages] Finished {pdf_path.name} "
                    f"({num_pages} pages)"
                )
            except Exception as e:
                logger.error(f"[render-pages] Failed {pdf_path.name}: {e}")
        logger.info(
            f"[render-pages] Complete — {total_pages} pages rendered, "
            f"{total_skipped} skipped, {dpi} dpi"
        )

    background_tasks.add_task(_render_all)

    return RenderPagesResponse(
        message=(
            f"Pre-rendering started for {len(all_pdfs)} PDFs at {dpi} dpi. "
            "This runs in the background — diagrams will be available as each "
            "page is rendered. Check server logs for progress."
        ),
        total_pdfs=len(all_pdfs),
    )


@router.delete("/knowledge/index")
def clear_index() -> dict:
    """Clear the vector store (dangerous — will remove all indexed content)."""
    vs = get_vector_store()
    vs.clear()
    return {"ok": True, "message": "Vector store cleared. Run POST /knowledge/index to rebuild."}
