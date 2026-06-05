from llm_client.models import (
    AzureTokenProvider,
    LLMResponse,
    Message,
    Messages,
    Provider,
    StreamChunk,
    StreamEvent,
    ThinkingBlock,
    ToolDef,
    ToolUse,
    detect_provider,
)
from llm_client.base import BaseLLMClient
from llm_client.openai_client import OpenAIClient
from llm_client.azure_client import AzureClient
from llm_client.anthropic_client import AnthropicClient
from llm_client.llm_client import LLMClient
from llm_client.llm_factory import (
    create_anthropic_client, create_azure_client, create_doubao_client, create_from_profiles,
    create_kimi_client, create_llm_client, create_mlx_client, create_openai_client, create_zhipu_client,
    get_config, get_profile
)

__all__ = [
    "AzureClient",
    "AzureTokenProvider",
    "BaseLLMClient",
    "LLMClient",
    "LLMResponse",
    "Message",
    "Messages",
    "OpenAIClient",
    "AnthropicClient",
    "MlxClient",
    "Provider",
    "StreamChunk",
    "StreamEvent",
    "ThinkingBlock",
    "ToolDef",
    "ToolUse",
    "detect_provider",
    "create_anthropic_client",
    "create_zhipu_client",
    "create_azure_client",
    "create_doubao_client",
    "create_from_profiles",
    "create_kimi_client",
    "create_llm_client",
    "create_mlx_client",
    "create_openai_client",
    "get_config",
    "get_profile",
]


def __getattr__(name: str):
    if name == "MlxClient":
        from llm_client.mlx_client import MlxClient

        return MlxClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
