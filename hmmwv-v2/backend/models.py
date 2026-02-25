"""
Pydantic models (request/response schemas) for the HMMWV Technical Assistant v2 API.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str
    query: str
    vehicle_variant: str = ""
    maintenance_category: str = ""
    deep_analysis: bool = False     # True = multi-agent pipeline


# ═══════════════════════════════════════════════════════════════════════════
# SESSIONS
# ═══════════════════════════════════════════════════════════════════════════

class SourceRef(BaseModel):
    text: str
    metadata: dict
    distance: float
    id: str


class ImageRef(BaseModel):
    url: str        # e.g. "/images/Basic-Humvee-Parts-Book-1/..."
    source: str     # PDF filename
    page: int


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    sources: List[SourceRef] = []
    images: List[ImageRef] = []


class Session(BaseModel):
    id: str
    title: str
    vehicle_variant: str = ""
    maintenance_category: str = ""
    created_at: str
    updated_at: str
    messages: List[SessionMessage] = []


class SessionSummary(BaseModel):
    """Lightweight session object (no messages) used for the sidebar list."""
    id: str
    title: str
    vehicle_variant: str = ""
    maintenance_category: str = ""
    created_at: str
    updated_at: str
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    vehicle_variant: str = ""
    maintenance_category: str = ""


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    vehicle_variant: Optional[str] = None
    maintenance_category: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

class SettingsModel(BaseModel):
    provider: str = "Ollama (Local)"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:latest"
    openai_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-6"
    youtube_api_key: str = ""
    youtube_enabled: bool = True
    youtube_max_results: int = 3
    agent_mode: bool = False
    agent1_enabled: bool = False
    agent1_provider: str = "Ollama (Local)"
    agent1_model: str = "gpt-oss:latest"
    agent2_provider: str = "Ollama (Local)"
    agent2_model: str = "gpt-oss:latest"
    agent3_provider: str = "Ollama (Local)"
    agent3_model: str = "gpt-oss:latest"
    agent4_provider: str = "Ollama (Local)"
    agent4_model: str = "gpt-oss:latest"
    agent5_enabled: bool = True
    agent5_provider: str = "Ollama (Local)"
    agent5_model: str = "gpt-oss:latest"
    agent6_provider: str = "Ollama (Local)"
    agent6_model: str = "gpt-oss:latest"


class OllamaModelsResponse(BaseModel):
    models: List[str]
    connected: bool


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeStats(BaseModel):
    total_chunks: int
    num_sources: int
    source_files: List[str]
    total_pdfs: int
    unprocessed_pdfs: int


class IndexResponse(BaseModel):
    indexed: int
    chunks_added: int


class RenderPagesResponse(BaseModel):
    message: str
    total_pdfs: int
