from __future__ import annotations
"""Provider factory for nanobot runtime entrypoints."""

from nanobot.providers.base import LLMProvider
from nanobot.providers.custom_anthropic_provider import CustomAnthropicProvider
from nanobot.providers.litellm_provider import LiteLLMProvider


def make_provider(config, model: str | None = None) -> LLMProvider:
    selected_model = model or config.agents.defaults.model
    provider_cfg = config.get_provider(selected_model)
    provider_name = config.get_provider_name(selected_model)
    api_key = config.get_api_key(selected_model)
    api_base = config.get_api_base(selected_model)
    extra_headers = (
        provider_cfg.extra_headers
        if provider_cfg and provider_cfg.extra_headers
        else None
    )

    if provider_name == "custom_anthropic":
        return CustomAnthropicProvider(
            api_key=api_key,
            api_base=api_base,
            default_model=selected_model,
            extra_headers=extra_headers,
            provider_name=provider_name,
        )

    return LiteLLMProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=selected_model,
        extra_headers=extra_headers,
        provider_name=provider_name,
    )
