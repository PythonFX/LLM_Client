import os
from typing import Optional, Union

from dotenv import load_dotenv
from pathlib import Path

from . import AnthropicClient, AzureClient, LLMClient, OpenAIClient
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
