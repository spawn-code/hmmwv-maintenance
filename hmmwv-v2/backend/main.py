"""
HMMWV Technical Assistant v2 — FastAPI backend entry point.

Start with:
    cd hmmwv-v2/backend
    uvicorn main:app --reload --port 8000

API docs:  http://localhost:8000/docs
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add backend directory to path so imports work without a package install
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EXTRACTED_IMAGES_DIR, SESSIONS_DIR
from dependencies import init_singletons
from routers import chat, sessions, settings, knowledge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: warm up singletons once at startup ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting HMMWV Technical Assistant v2 backend…")
    init_singletons()     # loads 24MB BM25 index, builds index structures
    logger.info("Backend ready. Listening for requests.")
    yield
    logger.info("Backend shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HMMWV Technical Assistant v2",
    description="AI-powered maintenance and repair guide for HMMWV vehicles.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://localhost:3000",   # alternative dev server
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for extracted PDF page images
# e.g. GET /images/Basic-Humvee-Parts-Book-1/Basic-Humvee-Parts-Book-1_p0040_img000.png
app.mount(
    "/images",
    StaticFiles(directory=str(EXTRACTED_IMAGES_DIR)),
    name="images",
)

# ─── Routers ──────────────────────────────────────────────────────────────

app.include_router(chat.router,      tags=["chat"])
app.include_router(sessions.router,  tags=["sessions"])
app.include_router(settings.router,  tags=["settings"])
app.include_router(knowledge.router, tags=["knowledge"])


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "version": "2.0.0"}
