"""
MultiAgentPipeline — 6-specialist agent system extracted from app.py, Streamlit-free.

Key change from original: _make_engine() reads from a settings dict (self.settings)
instead of st.session_state. A status_queue parameter allows the SSE router to
receive live progress events while the pipeline runs in a thread pool.

Pipeline:
  1. RETRIEVER   — LLM re-ranks BM25 chunks (optional)
  2. PROCEDURE  ─┐
  3. SAFETY     ─┤ parallel (ThreadPoolExecutor)
  4. PARTS      ─┘
  5. SIMPLIFIER  — rewrites procedure for novice mechanics (optional)
  6. EDITOR      — synthesizes all outputs into final structured answer
"""

import concurrent.futures
import json
import queue
import re
import threading
import time
import logging
from typing import Optional

from config import (
    PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC,
    OLLAMA_DEFAULT_URL, OLLAMA_DEFAULT_MODEL,
    OPENAI_DEFAULT_URL, OPENAI_DEFAULT_MODEL,
    ANTHROPIC_DEFAULT_MODEL,
    TOP_K_RESULTS,
    AGENT1_RETRIEVER_PROMPT, AGENT2_PROCEDURE_PROMPT,
    AGENT3_SAFETY_PROMPT, AGENT4_PARTS_PROMPT,
    AGENT5_SIMPLIFIER_PROMPT, AGENT6_EDITOR_PROMPT,
)
from core.ai_engine import AIEngine

logger = logging.getLogger(__name__)


class MultiAgentPipeline:
    """
    6-agent pattern with parallel specialists.
    settings dict replaces st.session_state from the original app.
    status_queue (optional) receives live JSON-serializable status events.
    """

    AGENT_META = {
        1: ("Retriever",  "🔍", "#3b82f6"),
        2: ("Procedure",  "⚙️",  "#10b981"),
        3: ("Safety",     "🦺", "#ef4444"),
        4: ("Parts",      "🔩", "#0d9488"),
        5: ("Simplifier", "📝", "#8b5cf6"),
        6: ("Editor",     "✍️",  "#2563eb"),
    }

    def __init__(
        self,
        query: str,
        search_results: list,
        settings: dict,                          # replaces st.session_state reads
        vehicle_variant: str = "",
        maintenance_category: str = "",
        status_queue: Optional[queue.Queue] = None,  # for SSE bridging
    ):
        self.query    = query
        self.results  = search_results
        self.settings = settings
        self.variant  = vehicle_variant
        self.category = maintenance_category
        self.status_queue = status_queue

    def _emit(self, step: str, label: str, done: bool, elapsed: Optional[float] = None):
        """Push an agent_status event to the SSE queue (non-blocking)."""
        if self.status_queue is not None:
            event = {"type": "agent_status", "step": step, "label": label, "done": done}
            if elapsed is not None:
                event["elapsed"] = round(elapsed, 1)
            try:
                self.status_queue.put_nowait(event)
            except queue.Full:
                pass

    def _make_engine(self, agent_num: int) -> AIEngine:
        """Build an AIEngine for a given agent using per-agent settings dict."""
        s = self.settings
        provider = s.get(f"agent{agent_num}_provider", s.get("provider", PROVIDER_OLLAMA))
        model    = s.get(f"agent{agent_num}_model",    s.get("ollama_model", OLLAMA_DEFAULT_MODEL))
        if provider == PROVIDER_OLLAMA:
            return AIEngine(provider,
                            base_url=s.get("ollama_url", OLLAMA_DEFAULT_URL),
                            model=model)
        elif provider == PROVIDER_OPENAI:
            return AIEngine(provider,
                            base_url=s.get("openai_url", OPENAI_DEFAULT_URL),
                            model=model,
                            api_key=s.get("openai_api_key", ""))
        elif provider == PROVIDER_ANTHROPIC:
            return AIEngine(provider,
                            api_key=s.get("anthropic_api_key", ""),
                            model=model)
        return AIEngine(PROVIDER_OLLAMA, model=model)

    def _build_context(self, results: Optional[list] = None) -> str:
        r = results if results is not None else self.results
        if not r:
            return "No technical manual content retrieved."
        parts = []
        for i, chunk in enumerate(r):
            meta = chunk.get("metadata", {})
            src  = meta.get("source_file", "Unknown")
            pg   = meta.get("page_number", "?")
            sec  = meta.get("section_title", "")
            rel  = max(0, 1 - chunk.get("distance", 0)) * 100
            hdr  = f"Ref {i+1} | {src} | Page {pg}"
            if sec:
                hdr += f" | {sec}"
            hdr += f" | {rel:.0f}% relevant"
            parts.append(f"--- {hdr} ---\n{chunk['text']}\n")
        return "\n".join(parts)

    def _run_agent(self, agent_num: int, system_prompt: str,
                   user_content: str,
                   status: dict, lock: threading.Lock) -> str:
        """Run one agent (blocking). Updates shared status dict + SSE queue. Thread-safe."""
        name = self.AGENT_META[agent_num][0]
        t0   = time.monotonic()
        with lock:
            status[agent_num] = ("running", t0)
        self._emit(name.lower(), name, done=False)
        try:
            engine   = self._make_engine(agent_num)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ]
            result  = engine.chat_complete(messages)
            elapsed = time.monotonic() - t0
            with lock:
                status[agent_num] = (f"done ({elapsed:.1f}s)", t0)
            self._emit(name.lower(), name, done=True, elapsed=elapsed)
            return result
        except Exception as e:
            elapsed = time.monotonic() - t0
            with lock:
                status[agent_num] = (f"error ({elapsed:.1f}s)", t0)
            self._emit(name.lower(), name, done=True, elapsed=elapsed)
            return f"[Agent {agent_num} failed: {e}]"

    def run(self, status: Optional[dict] = None, lock: Optional[threading.Lock] = None) -> dict:
        """
        Execute the full pipeline. Returns dict with:
          procedure, simplified, safety, parts, final, used_results, timings
        """
        if status is None:
            status = {}
        if lock is None:
            lock = threading.Lock()

        s = self.settings
        variant_tag  = f"Vehicle: {self.variant}\n"  if self.variant  else ""
        category_tag = f"Category: {self.category}\n" if self.category else ""
        query_block  = f"{variant_tag}{category_tag}\nMechanic's Question:\n{self.query}"

        # ── Agent 1: Context Re-ranker (optional) ─────────────────────────
        used_results = self.results
        context      = self._build_context()

        if s.get("agent1_enabled", False) and len(self.results) > 4:
            chunk_list = "\n".join(
                f"[{i}] {r['text'][:200]}…" for i, r in enumerate(self.results)
            )
            a1_user = (
                f"Mechanic's Question:\n{self.query}\n\n"
                f"Retrieved chunks (select most relevant, return JSON array of indices):\n"
                f"{chunk_list}"
            )
            a1_out = self._run_agent(1, AGENT1_RETRIEVER_PROMPT, a1_user, status, lock)
            try:
                m = re.search(r'\[[\d,\s]+\]', a1_out)
                if m:
                    indices = json.loads(m.group())
                    indices = [i for i in indices if 0 <= i < len(self.results)]
                    if indices:
                        used_results = [self.results[i] for i in indices[:TOP_K_RESULTS]]
                        context = self._build_context(used_results)
                        with lock:
                            status[1] = (f"selected {len(used_results)} chunks", status[1][1])
            except Exception:
                pass
        else:
            with lock:
                status[1] = ("BM25 retrieval (LLM re-rank disabled)", time.monotonic())
            self._emit("retriever", "Retriever", done=True, elapsed=0.0)

        # ── Agents 2 / 3 / 4 — parallel ──────────────────────────────────
        base_user = f"Context from HMMWV Technical Manuals:\n{context}\n\n{query_block}"
        with lock:
            status[2] = ("queued", time.monotonic())
            status[3] = ("queued", time.monotonic())
            status[4] = ("queued", time.monotonic())

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            f2 = pool.submit(self._run_agent, 2, AGENT2_PROCEDURE_PROMPT, base_user, status, lock)
            f3 = pool.submit(self._run_agent, 3, AGENT3_SAFETY_PROMPT,    base_user, status, lock)
            f4 = pool.submit(self._run_agent, 4, AGENT4_PARTS_PROMPT,     base_user, status, lock)
            procedure_raw = f2.result(timeout=300)
            safety_out    = f3.result(timeout=300)
            parts_out     = f4.result(timeout=300)

        # ── Agent 5: Simplifier ───────────────────────────────────────────
        if s.get("agent5_enabled", True):
            simplifier_user = f"Original procedure (written for experienced mechanics):\n\n{procedure_raw}"
            procedure_simple = self._run_agent(
                5, AGENT5_SIMPLIFIER_PROMPT, simplifier_user, status, lock
            )
        else:
            procedure_simple = procedure_raw
            with lock:
                status[5] = ("skipped (disabled)", time.monotonic())
            self._emit("simplifier", "Simplifier", done=True, elapsed=0.0)

        # ── Agent 6: Editor / Synthesizer ─────────────────────────────────
        editor_user = (
            f"Mechanic's Original Question:\n{self.query}\n\n"
            f"--- PROCEDURE AGENT ---\n{procedure_simple}\n\n"
            f"--- SAFETY AGENT ---\n{safety_out}\n\n"
            f"--- PARTS AGENT ---\n{parts_out}\n\n"
            f"Source Context (for citations):\n{context[:2500]}"
        )
        final_out = self._run_agent(6, AGENT6_EDITOR_PROMPT, editor_user, status, lock)

        # Compute timings
        timings = {}
        with lock:
            for num, (msg, t0) in status.items():
                m = re.search(r'\((\d+\.\d+)s\)', msg)
                timings[num] = float(m.group(1)) if m else 0.0

        return {
            "procedure":    procedure_raw,
            "simplified":   procedure_simple,
            "safety":       safety_out,
            "parts":        parts_out,
            "final":        final_out,
            "used_results": used_results,
            "timings":      timings,
        }
