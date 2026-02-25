// ═══════════════════════════════════════════════════════════════════════════
// HMMWV Technical Assistant v2 — TypeScript types
// ═══════════════════════════════════════════════════════════════════════════

export interface SourceRef {
  text: string
  metadata: {
    source_file: string
    page_number: number
    total_pages?: number
    chunk_index?: number
    section_title?: string
  }
  distance: number
  id: string
}

export interface ImageRef {
  url: string       // e.g. "/images/Basic-Humvee-Parts-Book-1/...png"
  source: string    // PDF filename
  page: number
}

export interface SessionMessage {
  id?: string                   // client-side key helper
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: SourceRef[]
  images?: ImageRef[]
}

export interface Session {
  id: string
  title: string
  vehicle_variant: string
  maintenance_category: string
  created_at: string
  updated_at: string
  messages: SessionMessage[]
}

export interface SessionSummary {
  id: string
  title: string
  vehicle_variant: string
  maintenance_category: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface SessionGroup {
  label: 'Today' | 'Yesterday' | 'Older'
  sessions: SessionSummary[]
}

export interface AgentStep {
  step: 'retriever' | 'procedure' | 'safety' | 'parts' | 'simplifier' | 'editor'
  label: string
  done: boolean
  elapsed?: number
}

export type SSEEventType = 'token' | 'agent_status' | 'sources' | 'images' | 'done' | 'error'

export interface SSEEvent {
  type: SSEEventType
  content?: string        // for "token"
  step?: string           // for "agent_status"
  label?: string          // for "agent_status"
  done?: boolean          // for "agent_status"
  elapsed?: number        // for "agent_status"
  data?: SourceRef[] | ImageRef[]  // for "sources" | "images"
  message?: string        // for "error"
}

export interface ChatRequest {
  session_id: string
  query: string
  vehicle_variant?: string
  maintenance_category?: string
  deep_analysis?: boolean
}

export interface SettingsModel {
  provider: string
  ollama_url: string
  ollama_model: string
  openai_url: string
  openai_model: string
  openai_api_key: string
  anthropic_api_key: string
  anthropic_model: string
  youtube_api_key: string
  youtube_enabled: boolean
  youtube_max_results: number
  agent_mode: boolean
  agent1_enabled: boolean
  agent1_provider: string
  agent1_model: string
  agent2_provider: string
  agent2_model: string
  agent3_provider: string
  agent3_model: string
  agent4_provider: string
  agent4_model: string
  agent5_enabled: boolean
  agent5_provider: string
  agent5_model: string
  agent6_provider: string
  agent6_model: string
}

export interface KnowledgeStats {
  total_chunks: number
  num_sources: number
  source_files: string[]
  total_pdfs: number
  unprocessed_pdfs: number
}

export interface OllamaModelsResponse {
  models: string[]
  connected: boolean
}
