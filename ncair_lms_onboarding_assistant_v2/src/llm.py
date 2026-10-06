"""Thin LLM layer.

* `OllamaClient` talks to a local Ollama server over HTTP using only the
  standard library, so there is no Python-package version to break. It supports
  Ollama's structured outputs (`format` = JSON schema), which forces the router
  to emit valid JSON with only allowed tool names and page ids. That works with
  ANY model, including N-ATLaS, which does not need native tool-calling support.
* `FakeClient` replays scripted replies, so the whole pipeline can be tested
  without a model.
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import config


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str, host: str = config.OLLAMA_HOST, timeout: float = config.LLM_TIMEOUT_S):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def chat(self, messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]] = None,
             options: Optional[Dict[str, Any]] = None) -> str:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                "options": options or {}}
        if schema is not None:
            body["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            raise LLMError(f"Ollama call failed for model '{self.model}': {e}") from e
        if "error" in data:
            raise LLMError(f"Ollama error for model '{self.model}': {data['error']}")
        return (data.get("message") or {}).get("content", "").strip()

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read().decode())["models"]]
            return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)
        except Exception:
            return False


class FakeClient:
    """Returns scripted replies in order (or via a callable). Records every call."""

    def __init__(self, replies):
        self.replies = replies
        self.calls: List[Dict[str, Any]] = []
        self.model = "fake"

    def chat(self, messages, schema=None, options=None) -> str:
        self.calls.append({"messages": messages, "schema": schema})
        if callable(self.replies):
            out = self.replies(messages, schema)
        else:
            out = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(out, Exception):
            raise out
        return out
