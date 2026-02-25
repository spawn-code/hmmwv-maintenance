# MechAssist v2 — HMMWV Diagnostics

ChatGPT-style interface for the HMMWV Technical Assistant.
FastAPI backend + React SPA, dark military theme.

---

## Quick Start

```bash
cd hmmwv-v2
chmod +x start.sh
./start.sh
```

Then open **http://localhost:5173** in your browser.

The script starts both servers:
- **Backend** — FastAPI on `localhost:8000` (loads BM25 index at startup, ~1–2s)
- **Frontend** — Vite dev server on `localhost:5173` (proxies API calls to backend)

---

## Requirements

### Backend
- Python 3.10+
- The worktree's existing `venv/` is used automatically if present
- Dependencies: `pip install -r backend/requirements.txt`

### Frontend
- Node.js 18+ / npm 9+
- Run `npm install` in `frontend/` if `node_modules/` is missing

---

## Architecture

```
hmmwv-v2/
├── backend/                FastAPI (port 8000)
│   ├── main.py             App entry, lifespan, CORS, static /images mount
│   ├── config.py           All constants, prompts, settings helpers
│   ├── models.py           Pydantic schemas
│   ├── dependencies.py     VectorStore + PDFProcessor singletons
│   ├── core/
│   │   ├── vector_store.py BM25 retrieval (18,752 chunks)
│   │   ├── ai_engine.py    Multi-provider LLM (Ollama/OpenAI/Anthropic)
│   │   ├── multi_agent.py  6-agent parallel pipeline
│   │   └── pdf_processor.py PDF text + image extraction
│   ├── routers/
│   │   ├── chat.py         POST /chat/stream  (SSE)
│   │   ├── sessions.py     CRUD /sessions/*
│   │   ├── settings.py     GET/PUT /settings
│   │   └── knowledge.py    GET /knowledge/stats, POST /knowledge/index
│   └── sessions/           JSON session files (auto-created)
└── frontend/               React 19 + Vite 7 + Tailwind CSS 4 (port 5173)
    └── src/
        ├── App.tsx          Router: / → ChatPage, /settings → SettingsPage
        ├── components/
        │   ├── layout/      AppShell, Sidebar
        │   ├── sidebar/     SidebarTabs, SessionList, SessionItem
        │   ├── chat/        ChatPage, MessageList, MessageBubble, WelcomeCard, …
        │   ├── input/       ChatInputBar, VehicleSelector, CategorySelector
        │   └── settings/    SettingsPage (AI Provider, Knowledge Base, Advanced)
        ├── store/           Zustand stores (session, chat, settings)
        ├── hooks/           useStreamingChat, useAutoScroll
        └── api/             Fetch wrappers (sessions, settings, knowledge)
```

---

## SSE Streaming

The chat endpoint uses **Server-Sent Events via HTTP POST** (not EventSource):

```
POST /chat/stream
Content-Type: application/json

{ "session_id": "...", "query": "...", "deep_analysis": false }
```

Event types:
| Type | Payload |
|------|---------|
| `token` | `{ content: "text chunk" }` |
| `agent_status` | `{ step, label, done, elapsed? }` |
| `sources` | `{ data: SourceRef[] }` |
| `images` | `{ data: ImageRef[] }` |
| `done` | — |
| `error` | `{ message: "..." }` |

---

## Settings

Settings are persisted to `backend/settings.json` and loaded on startup.

| Provider | Required fields |
|----------|----------------|
| Ollama (Local) | `ollama_url`, `ollama_model` |
| OpenAI Compatible | `openai_url`, `openai_api_key`, `openai_model` |
| Anthropic | `anthropic_api_key`, `anthropic_model` |

---

## Adding Documents

1. Drop PDF files into `knowledge_base/` (worktree root)
2. Open **Settings → Knowledge Base → Index New PDFs**
3. New documents are immediately searchable

---

## Chat Sessions

Sessions are stored as JSON files in `backend/sessions/`.
They persist across restarts and are grouped in the sidebar as **Today / Yesterday / Older**.

---

## Deep Analysis Mode

Toggle **🔍 Deep Analysis** in the chat input bar to activate the 6-agent pipeline:

1. **Retriever** — BM25 search + query expansion
2. **Procedure Writer** — step-by-step repair procedure *(parallel)*
3. **Safety Officer** — safety warnings & precautions *(parallel)*
4. **Parts Specialist** — parts/tools list *(parallel)*
5. **Simplifier** — plain-language rewrite
6. **Editor** — synthesized final answer

Agents 2–4 run in parallel (ThreadPoolExecutor). Total latency ~15–45s depending on the model.
