"""
Chat router — POST /chat/stream (Server-Sent Events).

Single-agent mode: streams tokens directly from AIEngine.chat_stream().
Deep analysis mode: runs MultiAgentPipeline in a thread pool, emits agent_status
events via a queue while agents run in parallel, then streams final text.

SSE event types emitted:
  {"type": "token",        "content": "text chunk"}
  {"type": "agent_status", "step": "procedure", "label": "Procedure Writer", "done": false}
  {"type": "agent_status", "step": "procedure", "done": true, "elapsed": 12.4}
  {"type": "sources",      "data": [{...}, ...]}
  {"type": "images",       "data": [{...}, ...]}
  {"type": "done"}
  {"type": "error",        "message": "..."}
"""

import asyncio
import json
import queue
import threading
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from config import (
    load_settings, EXTRACTED_IMAGES_DIR, KNOWLEDGE_BASE_DIR, TOP_K_RESULTS,
    PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC,
    OLLAMA_DEFAULT_URL, OLLAMA_DEFAULT_MODEL,
    OPENAI_DEFAULT_URL, OPENAI_DEFAULT_MODEL,
    ANTHROPIC_DEFAULT_MODEL,
)
from core.ai_engine import AIEngine
from core.multi_agent import MultiAgentPipeline
from dependencies import get_vector_store
from models import ChatRequest
from routers.sessions import append_messages_to_session, get_conversation_history

router = APIRouter()


# ── helper: build AIEngine from settings dict ─────────────────────────────

def _make_main_engine(settings: dict) -> AIEngine:
    provider = settings.get("provider", PROVIDER_OLLAMA)
    if provider == PROVIDER_OLLAMA:
        return AIEngine(provider,
                        base_url=settings.get("ollama_url", OLLAMA_DEFAULT_URL),
                        model=settings.get("ollama_model", OLLAMA_DEFAULT_MODEL))
    elif provider == PROVIDER_OPENAI:
        return AIEngine(provider,
                        base_url=settings.get("openai_url", OPENAI_DEFAULT_URL),
                        model=settings.get("openai_model", OPENAI_DEFAULT_MODEL),
                        api_key=settings.get("openai_api_key", ""))
    elif provider == PROVIDER_ANTHROPIC:
        return AIEngine(provider,
                        api_key=settings.get("anthropic_api_key", ""),
                        model=settings.get("anthropic_model", ANTHROPIC_DEFAULT_MODEL))
    return AIEngine(PROVIDER_OLLAMA,
                    base_url=OLLAMA_DEFAULT_URL, model=OLLAMA_DEFAULT_MODEL)


# ── helper: find page images for search results ───────────────────────────

def _find_images(search_results: list, max_images: int = 12) -> list:
    """
    Find images for pages that appear in the top search results.

    Strategy (two-pass per page):
      1. Return pre-extracted embedded raster images if present.
         Military TMs sometimes embed photos this way.
      2. Fall back to a full-page render (PNG at 150 dpi).
         This is the correct approach for TM technical diagrams, which are
         drawn as vector graphics — invisible to pdfplumber page.images but
         perfectly captured by rendering the page to a raster image.
         The rendered file is cached on disk; subsequent requests are instant.

    Returns list of ImageRef dicts: {url, source, page}.
    """
    from dependencies import get_pdf_processor   # local import avoids circular

    images = []
    seen   = set()
    proc   = get_pdf_processor()

    for result in search_results:
        meta        = result.get("metadata", {})
        source_file = meta.get("source_file", "")
        page_num    = meta.get("page_number", 0)
        if not source_file or not page_num:
            continue

        page_int   = int(page_num)
        source_dir = source_file.replace(".pdf", "").replace(".PDF", "")
        img_dir    = EXTRACTED_IMAGES_DIR / source_dir
        page_str   = f"p{page_int:04d}"

        # ── 1. Pre-extracted embedded raster images ────────────────────────
        found_embedded = False
        if img_dir.exists():
            for img_path in sorted(img_dir.glob(f"*_{page_str}_img*.png")):
                url = f"/images/{source_dir}/{img_path.name}"
                if url not in seen:
                    seen.add(url)
                    images.append({"url": url, "source": source_file, "page": page_int})
                    found_embedded = True

        # ── 2. Full-page render fallback (catches vector diagrams) ─────────
        if not found_embedded:
            render_name = f"page_{page_int:04d}_150dpi.png"
            render_path = EXTRACTED_IMAGES_DIR / source_dir / "pages" / render_name
            render_url  = f"/images/{source_dir}/pages/{render_name}"

            if not render_path.exists():
                # Render on demand — result is cached so next request is fast
                pdf_path = KNOWLEDGE_BASE_DIR / source_file
                if pdf_path.exists():
                    proc.extract_page_as_image(pdf_path, page_int, dpi=150)

            if render_path.exists() and render_url not in seen:
                seen.add(render_url)
                images.append({"url": render_url, "source": source_file, "page": page_int})

        if len(images) >= max_images:
            break

    return images[:max_images]


# ── SSE formatter ─────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Main streaming endpoint ───────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    Frontend MUST use fetch + ReadableStream (not EventSource) because this is POST.
    """

    async def event_generator():
        settings     = load_settings()
        vs           = get_vector_store()
        search_results = vs.search(request.query, n_results=TOP_K_RESULTS)
        history      = get_conversation_history(request.session_id, max_messages=12)
        full_response = ""
        used_results = search_results

        try:
            if request.deep_analysis:
                # ── Multi-agent pipeline ──────────────────────────────────
                status_queue = queue.Queue(maxsize=100)

                pipeline = MultiAgentPipeline(
                    query=request.query,
                    search_results=search_results,
                    settings=settings,
                    vehicle_variant=request.vehicle_variant,
                    maintenance_category=request.maintenance_category,
                    status_queue=status_queue,
                )

                loop   = asyncio.get_event_loop()
                status = {}
                lock   = threading.Lock()

                # Run pipeline in thread pool
                future = loop.run_in_executor(None, pipeline.run, status, lock)

                # Drain status queue while pipeline runs
                while not future.done():
                    try:
                        event = status_queue.get_nowait()
                        yield _sse(event)
                    except queue.Empty:
                        await asyncio.sleep(0.05)

                # Drain any remaining events
                while not status_queue.empty():
                    try:
                        event = status_queue.get_nowait()
                        yield _sse(event)
                    except queue.Empty:
                        break

                # Get pipeline result
                result   = await future
                used_results = result.get("used_results", search_results)
                final_text   = result.get("final", "")
                full_response = final_text

                # Stream final text as tokens (chunk by ~20 chars for smooth display)
                chunk_size = 20
                for i in range(0, len(final_text), chunk_size):
                    chunk = final_text[i:i + chunk_size]
                    yield _sse({"type": "token", "content": chunk})
                    await asyncio.sleep(0)  # yield control back to event loop

            else:
                # ── Single-agent streaming ────────────────────────────────
                ai = _make_main_engine(settings)

                # Stream tokens from generator (blocking) via run_in_executor
                token_queue: asyncio.Queue = asyncio.Queue()

                def _stream_to_queue():
                    try:
                        for token in ai.chat_stream(
                            request.query, search_results, history,
                            request.vehicle_variant, request.maintenance_category,
                        ):
                            loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token))
                    except Exception as e:
                        loop.call_soon_threadsafe(token_queue.put_nowait, ("error", str(e)))
                    finally:
                        loop.call_soon_threadsafe(token_queue.put_nowait, ("done", None))

                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, _stream_to_queue)

                while True:
                    kind, value = await token_queue.get()
                    if kind == "done":
                        break
                    elif kind == "error":
                        yield _sse({"type": "error", "message": value})
                        return
                    else:
                        full_response += value
                        yield _sse({"type": "token", "content": value})

            # ── Send sources and images ───────────────────────────────────
            sources = [
                {"text": r["text"][:400], "metadata": r["metadata"],
                 "distance": r["distance"], "id": r["id"]}
                for r in used_results
            ]
            # Run in executor: _find_images may render PDF pages (blocking I/O)
            images = await asyncio.get_event_loop().run_in_executor(
                None, _find_images, used_results
            )

            yield _sse({"type": "sources", "data": sources})
            yield _sse({"type": "images",  "data": images})
            yield _sse({"type": "done"})

            # ── Persist messages to session file ──────────────────────────
            append_messages_to_session(
                session_id=request.session_id,
                user_query=request.query,
                assistant_content=full_response,
                sources=sources,
                images=images,
            )

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
