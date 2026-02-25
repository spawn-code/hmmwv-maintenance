"""
AIEngine — multi-provider streaming AI client extracted from app.py, Streamlit-free.
Supports Ollama, OpenAI-compatible, and Anthropic (Claude) via stdlib urllib only.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Generator

from config import (
    PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_ANTHROPIC,
    OLLAMA_DEFAULT_URL, OLLAMA_DEFAULT_MODEL,
    OPENAI_DEFAULT_URL, OPENAI_DEFAULT_MODEL,
    ANTHROPIC_DEFAULT_MODEL,
    SYSTEM_PROMPT, MAX_TOKENS, TEMPERATURE,
)

logger = logging.getLogger(__name__)


class AIEngine:
    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        self._cfg     = kwargs

    @staticmethod
    def check_ollama(base_url: str = OLLAMA_DEFAULT_URL) -> bool:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/api/tags"), timeout=3
            ):
                return True
        except Exception:
            return False

    @staticmethod
    def list_ollama_models(base_url: str = OLLAMA_DEFAULT_URL) -> list:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/api/tags"), timeout=5
            ) as resp:
                return sorted(m["name"] for m in json.loads(resp.read()).get("models", []))
        except Exception:
            return []

    def _format_context(self, results: list) -> str:
        if not results:
            return "<knowledge_base_context>\nNo relevant content found in the knowledge base.\n</knowledge_base_context>"
        parts = ["<knowledge_base_context>"]
        for i, r in enumerate(results, 1):
            meta    = r.get("metadata", {})
            src     = meta.get("source_file", "Unknown")
            pg      = meta.get("page_number", "?")
            section = meta.get("section_title", "")
            rel     = max(0, 1 - r.get("distance", 0)) * 100
            header_parts = [f"Ref {i}", src, f"Page {pg}"]
            if section:
                header_parts.append(f"Section: {section}")
            header_parts.append(f"Relevance: {rel:.0f}%")
            header = " | ".join(header_parts)
            parts.append(f"\n--- {header} ---\n{r['text']}\n")
        parts.append("</knowledge_base_context>")
        return "\n".join(parts)

    def _build_user_message(self, query, context, vehicle_variant="", maintenance_category=""):
        parts = []
        if vehicle_variant:      parts.append(f"[Vehicle: {vehicle_variant}]")
        if maintenance_category: parts.append(f"[Category: {maintenance_category}]")
        parts.append(context)
        parts.append(f"\n## Mechanic's Question\n{query}")
        return "\n".join(parts)

    def _stream_ollama(self, messages: list) -> Generator:
        url   = self._cfg.get("base_url", OLLAMA_DEFAULT_URL).rstrip("/")
        model = self._cfg.get("model", OLLAMA_DEFAULT_MODEL)
        payload = json.dumps({
            "model": model, "messages": messages, "stream": True,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
        }).encode()
        req = urllib.request.Request(
            f"{url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.strip()
                    if not line: continue
                    try:
                        chunk = json.loads(line.decode())
                        token = chunk.get("message", {}).get("content", "")
                        if token: yield token
                        if chunk.get("done"): break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as e:
            yield f"❌ **Cannot reach Ollama** at `{url}`\n\n{e.reason}"
        except Exception as e:
            yield f"❌ **Error**: {e}"

    def _stream_openai(self, messages: list) -> Generator:
        url    = self._cfg.get("base_url", OPENAI_DEFAULT_URL).rstrip("/")
        model  = self._cfg.get("model", OPENAI_DEFAULT_MODEL)
        apikey = self._cfg.get("api_key", "")
        payload = json.dumps({
            "model": model, "messages": messages, "stream": True,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if apikey: headers["Authorization"] = f"Bearer {apikey}"
        req = urllib.request.Request(
            f"{url}/chat/completions", data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    decoded = line.strip().decode("utf-8")
                    if decoded.startswith("data: "): decoded = decoded[6:]
                    if decoded == "[DONE]": break
                    if not decoded: continue
                    try:
                        token = json.loads(decoded).get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token: yield token
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode()
            except Exception: pass
            yield f"❌ **API Error {e.code}**: {e.reason}\n\n{body}"
        except Exception as e:
            yield f"❌ **Error**: {e}"

    def _stream_anthropic(self, messages: list) -> Generator:
        apikey = self._cfg.get("api_key", "")
        model  = self._cfg.get("model", ANTHROPIC_DEFAULT_MODEL)
        if not apikey:
            yield "❌ **Anthropic API key required.** Enter it in Settings."
            return
        sys_msg, filtered = "", []
        for m in messages:
            if m["role"] == "system": sys_msg = m["content"]
            else: filtered.append(m)
        payload = json.dumps({
            "model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "system": sys_msg or SYSTEM_PROMPT, "messages": filtered, "stream": True,
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": apikey,
                "anthropic-version": "2023-06-01",
            }, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    decoded = line.strip().decode("utf-8")
                    if decoded.startswith("data: "): decoded = decoded[6:]
                    if not decoded: continue
                    try:
                        event = json.loads(decoded)
                        if event.get("type") == "content_block_delta":
                            text = event.get("delta", {}).get("text", "")
                            if text: yield text
                        elif event.get("type") in ("message_stop", "error"):
                            if event.get("type") == "error":
                                yield f"❌ {event.get('error', {}).get('message', 'Error')}"
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode()
            except Exception: pass
            yield f"❌ **API Error {e.code}**: {e.reason}\n\n{body}"
        except Exception as e:
            yield f"❌ **Error**: {e}"

    def chat_stream(self, query, search_results, conversation_history=None,
                    vehicle_variant="", maintenance_category="") -> Generator:
        context  = self._format_context(search_results)
        user_msg = self._build_user_message(query, context, vehicle_variant, maintenance_category)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_msg})
        if self.provider == PROVIDER_OLLAMA:
            yield from self._stream_ollama(messages)
        elif self.provider == PROVIDER_OPENAI:
            yield from self._stream_openai(messages)
        elif self.provider == PROVIDER_ANTHROPIC:
            yield from self._stream_anthropic(messages)
        else:
            yield f"❌ Unknown provider: {self.provider}"

    def chat_complete(self, messages: list) -> str:
        """
        Non-streaming version — collects full response as a single string.
        Used by MultiAgentPipeline agents that run in parallel threads.
        Thread-safe: reads only from immutable self._cfg / self.provider.
        """
        parts: list = []
        try:
            if self.provider == PROVIDER_OLLAMA:
                gen = self._stream_ollama(messages)
            elif self.provider == PROVIDER_OPENAI:
                gen = self._stream_openai(messages)
            elif self.provider == PROVIDER_ANTHROPIC:
                gen = self._stream_anthropic(messages)
            else:
                return f"❌ Unknown provider: {self.provider}"
            for chunk in gen:
                parts.append(chunk)
        except Exception as e:
            parts.append(f"❌ Agent error: {e}")
        return "".join(parts)
