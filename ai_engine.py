"""
HMMWV Technical Assistant — AI Engine
=======================================
Claude API integration with RAG context injection.
"""

import logging
import sys
from pathlib import Path
from typing import Generator, Optional

import importlib.util

if "config" not in sys.modules:
    _cfg = Path(__file__).resolve().parent.parent / "config.py"
    _sp = importlib.util.spec_from_file_location("config", str(_cfg))
    _m = importlib.util.module_from_spec(_sp)
    sys.modules["config"] = _m
    _sp.loader.exec_module(_m)

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
)

logger = logging.getLogger(__name__)


class AIEngine:
    """Handles Claude API calls with RAG context for HMMWV technical queries."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or ANTHROPIC_API_KEY
        self._client = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "Anthropic API key not configured. Set ANTHROPIC_API_KEY "
                    "environment variable or enter it in the sidebar."
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def set_api_key(self, key: str):
        """Update the API key and reset the client."""
        self._api_key = key
        self._client = None

    # ── Context Formatting ─────────────────────────────────────────────────

    def _format_context(self, search_results: list[dict]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        if not search_results:
            return (
                "<knowledge_base_context>\n"
                "No relevant technical manual content was found for this query.\n"
                "</knowledge_base_context>"
            )

        context_parts = ["<knowledge_base_context>"]
        for i, result in enumerate(search_results, 1):
            source = result.get("metadata", {}).get("source_file", "Unknown")
            page = result.get("metadata", {}).get("page_number", "?")
            distance = result.get("distance", 0)
            relevance = max(0, (1 - distance)) * 100

            context_parts.append(
                f"\n--- Reference {i} | Source: {source} | Page: {page} | "
                f"Relevance: {relevance:.0f}% ---\n"
                f"{result['text']}\n"
            )
        context_parts.append("</knowledge_base_context>")
        return "\n".join(context_parts)

    # ── Message Building ───────────────────────────────────────────────────

    def _build_user_message(
        self,
        query: str,
        context: str,
        vehicle_variant: str = "",
        maintenance_category: str = "",
    ) -> str:
        """Build the full user message with context and metadata."""
        parts = []

        # Add vehicle context if specified
        if vehicle_variant:
            parts.append(f"[Vehicle: {vehicle_variant}]")
        if maintenance_category:
            parts.append(f"[Category: {maintenance_category}]")

        parts.append(context)
        parts.append(f"\n## Mechanic's Question\n{query}")

        return "\n".join(parts)

    # ── Chat Completion ────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        search_results: list[dict],
        conversation_history: list[dict] = None,
        vehicle_variant: str = "",
        maintenance_category: str = "",
    ) -> str:
        """
        Send a query to Claude with RAG context.
        Returns the assistant's response text.
        """
        context = self._format_context(search_results)
        user_message = self._build_user_message(
            query, context, vehicle_variant, maintenance_category
        )

        # Build message list
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            return response.content[0].text
        except anthropic.AuthenticationError:
            return "❌ **Authentication Error**: Invalid API key. Please check your Anthropic API key in the sidebar."
        except anthropic.RateLimitError:
            return "⏳ **Rate Limited**: Too many requests. Please wait a moment and try again."
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"❌ **Error**: {str(e)}"

    def chat_stream(
        self,
        query: str,
        search_results: list[dict],
        conversation_history: list[dict] = None,
        vehicle_variant: str = "",
        maintenance_category: str = "",
    ) -> Generator[str, None, None]:
        """
        Stream a response from Claude with RAG context.
        Yields text chunks as they arrive.
        """
        context = self._format_context(search_results)
        user_message = self._build_user_message(
            query, context, vehicle_variant, maintenance_category
        )

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            with self.client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError:
            yield "❌ **Authentication Error**: Invalid API key. Please check your Anthropic API key in the sidebar."
        except anthropic.RateLimitError:
            yield "⏳ **Rate Limited**: Too many requests. Please wait a moment and try again."
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"❌ **Error**: {str(e)}"

    # ── Diagnostic Helper ──────────────────────────────────────────────────

    def diagnose(
        self,
        symptoms: str,
        search_results: list[dict],
        vehicle_variant: str = "",
    ) -> str:
        """
        Specialized diagnostic mode — mechanic describes symptoms
        and the AI provides troubleshooting steps.
        """
        diagnostic_prefix = (
            "The mechanic is reporting the following symptoms and needs help "
            "diagnosing the issue. Provide a structured troubleshooting guide "
            "starting with the most likely causes, organized from simplest to "
            "most complex checks:\n\n"
        )
        return self.chat(
            diagnostic_prefix + symptoms,
            search_results,
            vehicle_variant=vehicle_variant,
        )
