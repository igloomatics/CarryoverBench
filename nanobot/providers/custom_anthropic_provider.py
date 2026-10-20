from __future__ import annotations
"""Custom Anthropic-compatible provider for relay/gateway deployments."""

import asyncio
import json
from typing import Any

import requests

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomAnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "claude-sonnet-4-6",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self.provider_name = provider_name or "custom_anthropic"

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        headers.update(self.extra_headers)
        return headers

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
                continue
            if role == "tool":
                tool_name = msg.get("name", "tool")
                converted.append({
                    "role": "user",
                    "content": f"Tool result from {tool_name}: {content}",
                })
                continue
            converted.append({
                "role": role,
                "content": content,
            })

        system = "\n\n".join([p for p in system_parts if p]) or None
        return system, converted

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted = []
        for tool in tools:
            fn = tool.get("function", {})
            converted.append({
                "name": fn.get("name", tool.get("name", "tool")),
                "description": fn.get("description", tool.get("description", "")),
                "input_schema": fn.get("parameters", tool.get("parameters", {"type": "object", "properties": {}})),
            })
        return converted

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return str(content)

    def _parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        content_blocks = payload.get("content", []) or []
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for idx, block in enumerate(content_blocks):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.get("id", f"toolu_{idx}"),
                    name=block.get("name", "tool"),
                    arguments=block.get("input", {}) if isinstance(block.get("input", {}), dict) else {},
                ))

        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return LLMResponse(
            content="\n".join([p for p in text_parts if p]) or None,
            tool_calls=tool_calls,
            finish_reason=payload.get("stop_reason") or "stop",
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
        )

    def _messages_url(self) -> str:
        base = self.api_base.rstrip("/") if self.api_base else ""
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if not self.api_base:
            return LLMResponse(content="Error calling LLM: custom_anthropic api_base is not configured", finish_reason="error")

        system, converted_messages = self._convert_messages(messages)
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "messages": converted_messages,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        url = self._messages_url()
        headers = self._build_headers()

        def _do_request() -> requests.Response:
            return requests.post(url, headers=headers, json=payload, timeout=60)

        max_retries = 5
        base_delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.to_thread(_do_request)
                if response.status_code in (429, 502, 503, 504) and attempt < max_retries:
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
                    except ValueError:
                        delay = base_delay * (2 ** attempt)
                    import logging
                    logging.getLogger("custom_anthropic_retry").warning(
                        f"HTTP {response.status_code} (attempt {attempt+1}/{max_retries+1}). "
                        f"Sleeping {delay:.1f}s before retry."
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)
            except requests.exceptions.RequestException as e:
                last_exc = e
                is_retryable = isinstance(e, (requests.exceptions.ConnectionError,
                                              requests.exceptions.Timeout)) or any(
                    code in str(e) for code in ("429", "502", "503", "504")
                )
                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    import logging
                    logging.getLogger("custom_anthropic_retry").warning(
                        f"{type(e).__name__} (attempt {attempt+1}/{max_retries+1}): {e}. "
                        f"Sleeping {delay:.1f}s before retry."
                    )
                    await asyncio.sleep(delay)
                    continue
                return LLMResponse(content=f"Error calling LLM: {str(e)}", finish_reason="error")
            except Exception as e:
                return LLMResponse(content=f"Error calling LLM: {str(e)}", finish_reason="error")

        return LLMResponse(
            content=f"Error calling LLM: exhausted retries ({last_exc})",
            finish_reason="error",
        )

    def get_default_model(self) -> str:
        return self.default_model
