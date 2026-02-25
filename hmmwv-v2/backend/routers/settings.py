"""
Settings router — GET/PUT /settings, GET /settings/ollama/models
"""

from fastapi import APIRouter, Query
from config import (
    load_settings, save_settings, _default_settings,
    ALL_PROVIDERS, HMMWV_VARIANTS, MAINTENANCE_CATEGORIES,
    OLLAMA_DEFAULT_URL,
)
from core.ai_engine import AIEngine
from models import SettingsModel, OllamaModelsResponse

router = APIRouter()


@router.get("/settings")
def get_settings() -> dict:
    return load_settings()


@router.put("/settings")
def update_settings(body: dict) -> dict:
    current = load_settings()
    current.update(body)
    save_settings(current)
    return current


@router.get("/settings/providers")
def get_providers() -> dict:
    return {"providers": ALL_PROVIDERS}


@router.get("/settings/variants")
def get_variants() -> dict:
    return {"variants": HMMWV_VARIANTS}


@router.get("/settings/categories")
def get_categories() -> dict:
    return {"categories": MAINTENANCE_CATEGORIES}


@router.get("/settings/ollama/models")
def get_ollama_models(url: str = Query(default=OLLAMA_DEFAULT_URL)) -> OllamaModelsResponse:
    connected = AIEngine.check_ollama(url)
    models = AIEngine.list_ollama_models(url) if connected else []
    return OllamaModelsResponse(models=models, connected=connected)
