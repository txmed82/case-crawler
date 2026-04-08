from __future__ import annotations

from casecrawler.config import get_env
from casecrawler.llm.base import BaseLLMProvider

_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
}


def get_provider(provider: str, model: str, **kwargs) -> BaseLLMProvider:
    """Create an LLM provider instance from config."""
    if provider not in _PROVIDER_KEY_MAP:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(_PROVIDER_KEY_MAP.keys())}")

    required_key = _PROVIDER_KEY_MAP[provider]
    if required_key:
        api_key = get_env(required_key)
        if not api_key:
            raise ValueError(
                f"API key {required_key} not set. Add it to your .env file for provider '{provider}'."
            )
    else:
        api_key = None

    if provider == "anthropic":
        from casecrawler.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model)
    elif provider == "openai":
        from casecrawler.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model)
    elif provider == "openrouter":
        from casecrawler.llm.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(api_key=api_key, model=model)
    elif provider == "ollama":
        from casecrawler.llm.ollama_provider import OllamaProvider
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=base_url)

    raise ValueError(f"Unknown provider: {provider}")
