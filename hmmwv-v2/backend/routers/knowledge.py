"""
Knowledge base router — stats and index management.
"""

from fastapi import APIRouter
from dependencies import get_vector_store, get_pdf_processor
from models import KnowledgeStats, IndexResponse

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


@router.delete("/knowledge/index")
def clear_index() -> dict:
    """Clear the vector store (dangerous — will remove all indexed content)."""
    vs = get_vector_store()
    vs.clear()
    return {"ok": True, "message": "Vector store cleared. Run POST /knowledge/index to rebuild."}
