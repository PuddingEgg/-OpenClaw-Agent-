from __future__ import annotations

import json
import re
from typing import Any
from urllib import error, request


class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: int = 45,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        last_error: Exception | None = None
        for use_json_mode in (True, False):
            for attempt in range(1, self.max_retries + 1):
                try:
                    response_data = self._request(messages, use_json_mode=use_json_mode)
                    content = response_data["choices"][0]["message"]["content"]
                    parsed = self._parse_json_content(content)
                    if not parsed:
                        raise DeepSeekError("Model returned empty content.")
                    return parsed
                except (DeepSeekError, KeyError, IndexError, TypeError) as exc:
                    last_error = exc
                    if attempt == self.max_retries:
                        break
        raise DeepSeekError(str(last_error) if last_error else "Unknown DeepSeek error.")

    def _request(self, messages: list[dict[str, str]], use_json_mode: bool) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 900,
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekError(f"HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise DeepSeekError(f"Network error: {exc.reason}") from exc

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        if not content or not content.strip():
            raise DeepSeekError("Model returned empty content.")

        raw = content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            return json.loads(snippet)

        raise DeepSeekError(f"Model did not return valid JSON: {content}")
