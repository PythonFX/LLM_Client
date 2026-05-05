import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv

from . import AnthropicClient, AzureClient, LLMClient, MlxClient, OpenAIClient
from .base import BaseLLMClient
from .models import AzureTokenProvider, Provider

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def create_anthropic_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 300.0,
) -> AnthropicClient:
    return AnthropicClient(
        api_key=api_key or os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL"),
        model=model or os.environ.get("ANTHROPIC_MODEL"),
        timeout=timeout,
    )


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 300.0,
) -> OpenAIClient:
    return OpenAIClient(
        api_key=api_key or os.environ["OPENAI_API_KEY"],
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        model=model or os.environ.get("OPENAI_MODEL"),
        timeout=timeout,
    )


def create_azure_client(
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_version: Optional[str] = None,
    timeout: float = 300.0,
) -> AzureClient:
    from .llm_helper import get_azure_ad_token

    ad_token_provider: AzureTokenProvider = get_azure_ad_token
    return AzureClient(
        deployment=deployment or os.environ["AZURE_DEPLOYMENT"],
        endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
        ad_token_provider=ad_token_provider,
        timeout=timeout,
    )


def create_doubao_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 300.0,
) -> AnthropicClient:
    return AnthropicClient(
        api_key=api_key or os.environ["DOUBAO_API_KEY"],
        base_url=base_url or os.environ.get("DOUBAO_ENDPOINT"),
        model=model or os.environ.get("DOUBAO_MODEL"),
        auth_mode="bearer",
        timeout=timeout,
    )


def create_kimi_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    thinking: bool = True,
    timeout: float = 300.0,
) -> AnthropicClient:
    thinking_config = {"type": "enabled", "budget_tokens": 10000} if thinking else None
    return AnthropicClient(
        api_key=api_key or os.environ["KIMI_API_KEY"],
        base_url=base_url or os.environ.get("KIMI_ENDPOINT"),
        model=model or os.environ.get("KIMI_MODEL", "kimi-k2.6"),
        auth_mode="bearer",
        thinking=thinking_config,
        timeout=timeout,
    )


def create_llm_client(
    default_provider: Optional[Union[Provider, str]] = None,
    timeout: float = 300.0,
) -> LLMClient:
    provider = default_provider or Provider(os.environ.get("DEFAULT_LLM_PROVIDER", "anthropic"))
    llm = LLMClient(default_provider=provider)

    if provider == Provider.ANTHROPIC:
        llm.add_client(Provider.ANTHROPIC, create_anthropic_client(timeout=timeout), default=True)
    elif provider == Provider.OPENAI:
        llm.add_client(Provider.OPENAI, create_openai_client(timeout=timeout), default=True)
    elif provider == Provider.AZURE:
        llm.add_client(Provider.AZURE, create_azure_client(timeout=timeout), default=True)
    elif provider == Provider.DOUBAO:
        llm.add_client(Provider.DOUBAO, create_doubao_client(timeout=timeout), default=True)
    elif provider == Provider.KIMI:
        llm.add_client(Provider.KIMI, create_kimi_client(timeout=timeout), default=True)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: {[p.value for p in Provider]}")

    return llm


def create_from_profiles(
    config_path: Optional[str] = None,
    default: Optional[str] = None,
    timeout: float = 300.0,
) -> LLMClient:
    path = Path(config_path) if config_path else Path.home() / ".llm_client_models.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    profiles = config.get("profiles", {})
    if not profiles:
        raise ValueError("No profiles defined in config file")

    default_name = default or config.get("default")
    llm = LLMClient()

    for name, profile in profiles.items():
        provider = profile.get("provider", "")
        client = _create_client_from_profile(provider, profile, timeout)
        is_default = name == default_name
        llm.add_client(name, client, default=is_default)

    if not llm.default_provider:
        llm.set_default_provider(next(iter(profiles)))

    return llm


def _create_client_from_profile(
    provider: str,
    profile: Dict[str, Any],
    timeout: float,
) -> BaseLLMClient:
    if provider == "openai":
        return OpenAIClient(
            api_key=profile["api_key"],
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            timeout=timeout,
        )
    elif provider == "anthropic":
        return AnthropicClient(
            api_key=profile["api_key"],
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            auth_mode=profile.get("auth_mode", "x-api-key"),
            thinking=profile.get("thinking"),
            timeout=timeout,
        )
    elif provider == "azure":
        from .llm_helper import get_azure_ad_token

        return AzureClient(
            deployment=profile["deployment"],
            endpoint=profile["endpoint"],
            api_version=profile.get("api_version", "2024-06-01"),
            api_key=profile.get("api_key"),
            ad_token_provider=get_azure_ad_token,
            timeout=timeout,
        )
    elif provider in ("doubao", "kimi"):
        return AnthropicClient(
            api_key=profile["api_key"],
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            auth_mode=profile.get("auth_mode", "bearer"),
            thinking=profile.get("thinking"),
            timeout=timeout,
        )
    elif provider == "mlx":
        return MlxClient(
            model_path=profile["model_path"],
            enable_thinking=profile.get("enable_thinking", True),
        )
    else:
        raise ValueError(f"Unknown provider in profile: {provider}")
