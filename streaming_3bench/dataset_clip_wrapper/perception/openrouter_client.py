"""Shared OpenRouter HTTP client."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_BASE = "https://openrouter.ai/api/v1/chat/completions"


@contextmanager
def _total_timeout(seconds: int):
    if seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handle_timeout(signum, frame):  # type: ignore[no-untyped-def]
        raise TimeoutError(f"OpenRouter request exceeded {seconds}s total timeout")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def load_openrouter_api_key(*, keys_py_path: str | None = None, env_var: str = "OPENROUTER_API_KEY") -> str:
    env_key = os.environ.get(env_var)
    if env_key:
        return env_key
    if keys_py_path:
        path = Path(keys_py_path)
        if path.exists():
            spec = importlib.util.spec_from_file_location("openrouter_keys", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                key = getattr(module, "OPENROUTER_API_KEY", None)
                if key:
                    return key
    raise RuntimeError(f"Missing API key: set {env_var} or provide keys_py_path")


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    def _loads(candidate: str) -> dict[str, Any]:
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("model response must be a JSON object")
        return payload

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        for normalized in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return _loads(normalized)
            except Exception as exc:
                last_error = exc
    assert last_error is not None
    raise last_error


class OpenRouterClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning: dict[str, Any] | None = None,
        timeout_s: int = 180,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.timeout_s = timeout_s
        self.last_response_metadata: dict[str, Any] = {}

    @staticmethod
    def _message_text_chars(messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total += len(str(item.get("text") or ""))
                        elif item.get("type") == "image_url":
                            total += len(str(((item.get("image_url") or {}).get("url")) or ""))
                    else:
                        total += len(str(item))
            elif content is not None:
                total += len(str(content))
        return total

    def _base_request_metadata(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_chars = self._message_text_chars(messages)
        return {
            "model": self.model,
            "prompt_chars": prompt_chars,
            "prompt_approx_tokens": int(round(prompt_chars / 4.0)),
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "malformed_json": 0,
            "timeout_count": 0,
            "compact_retry_count": 0,
            "cache_hit": False,
        }

    def chat(self, messages: list[dict[str, Any]], *, response_format: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if response_format is not None and os.environ.get("VIDEO_SKILLS_DISABLE_RESPONSE_FORMAT") != "1":
            payload["response_format"] = response_format
        self.last_response_metadata = self._base_request_metadata(messages)
        try:
            with _total_timeout(self.timeout_s):
                response = requests.post(
                    self.api_base,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                response_payload = response.json()
        except TimeoutError:
            self.last_response_metadata["timeout_count"] = 1
            raise
        message = response_payload["choices"][0]["message"]
        content = message.get("content")
        usage = response_payload.get("usage") or {}
        self.last_response_metadata.update(
            {
                "finish_reason": response_payload["choices"][0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "output_chars": len(content.strip()) if isinstance(content, str) else 0,
            }
        )
        if content is None:
            raise ValueError(
                "OpenRouter response did not include assistant content; "
                f"finish_reason={response_payload['choices'][0].get('finish_reason')}, "
                f"reasoning_preview={repr(message.get('reasoning'))[:300]}"
            )
        return content.strip()

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload_format = response_format or {"type": "json_object"}
        text = self.chat(messages, response_format=payload_format)
        try:
            return parse_json_response(text)
        except Exception:
            self.last_response_metadata["malformed_json"] = 1
            raise
