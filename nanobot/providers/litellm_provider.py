from __future__ import annotations
"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import json
import os
from typing import Any

import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


def _parse_bool_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = self._resolve_model(model or self.default_model)
        
        # OpenAI o-series and newer gpt-4o models require
        # 'max_completion_tokens' instead of 'max_tokens'.
        model_lower = model.lower()
        token_key = "max_tokens"
        if any(tag in model_lower for tag in ("gpt-4o", "o1", "o3", "o4")):
            token_key = "max_completion_tokens"

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            token_key: max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        enable_thinking = _parse_bool_env("BENCH_ENABLE_THINKING")
        if enable_thinking is not None and "qwen" in model_lower:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["enable_thinking"] = enable_thinking

        # Gemini safety filter mitigation: when routing Gemini models through
        # an OpenAI-compatible gateway (key77qiqi, apihy, aihubmix), Google's
        # safety layer probabilistically blocks responses to prompt-injection
        # content with finish_reason="content_filter". Pass safety_settings via
        # OpenAI extra_body — most proxies forward unknown JSON fields to the
        # upstream Gemini API. This is essential for §4.3 ablation runs where
        # attack content (user_attestation, urgent threats, emotional appeals)
        # would otherwise trigger Google's filter and erase agent responses.
        if (self._gateway
                and self._gateway.name in ("key77qiqi", "apihy", "aihubmix")
                and "gemini" in (self.default_model or "").lower()):
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["safety_settings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

        # Per-call timeout: when the provider hangs, fail fast and let the
        # retry loop (or the trial-level timeout) take over. Without this a
        # single stuck LLM call can block for the provider's default
        # (often 10 min), and a flaky-but-running upstream can pin the
        # process for hours.
        kwargs.setdefault("timeout", 120)

        max_retries = 4  # increased from 2 to cover Gemini content_filter probabilistic blocks
        base_delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = await acompletion(**kwargs)
                parsed = self._parse_response(response)

                # Gemini (via key77qiqi/apihy gateways) probabilistically returns
                # empty content + finish_reason="content_filter" — a soft refusal
                # from Google's safety layer. Same prompt often succeeds on retry.
                # Treat as retryable if we got nothing usable back.
                if (parsed.finish_reason == "content_filter"
                        and not parsed.content and not parsed.tool_calls
                        and attempt < max_retries):
                    delay = base_delay * (1.5 ** attempt)
                    import logging
                    logging.getLogger("litellm_retry").warning(
                        f"Gemini content_filter empty response (attempt {attempt+1}/{max_retries+1}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                return parsed
            except Exception as e:
                err_name = type(e).__name__
                is_retryable = any(kw in err_name for kw in (
                    "ServiceUnavailable", "BadGateway", "RateLimit",
                    "Timeout", "Connection", "APIError",
                )) or any(code in str(e) for code in ("502", "503", "429", "504"))

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    import logging
                    logging.getLogger("litellm_retry").warning(
                        f"LLM call failed (attempt {attempt+1}/{max_retries+1}): "
                        f"{err_name}. Retrying in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable or exhausted retries
                return LLMResponse(
                    content=f"Error calling LLM: {str(e)}",
                    finish_reason="error",
                )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None)
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
