from .models import (
    AzureTokenProvider,
    LLMResponse,
    Message,
    Provider,
    StreamChunk,
    StreamEvent,
    ThinkingBlock,
    ToolDef,
    ToolUse,
    detect_provider,
)
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .azure_client import AzureClient
from .anthropic_client import AnthropicClient
from .llm_manager import LLMManager

# Backward-compatible alias
LLMClient = LLMManager

__all__ = [
    "AzureClient",
    "AzureTokenProvider",
    "BaseLLMClient",
    "LLMClient",
    "LLMManager",
    "LLMResponse",
    "Message",
    "OpenAIClient",
    "AnthropicClient",
    "Provider",
    "StreamChunk",
    "StreamEvent",
    "ThinkingBlock",
    "ToolDef",
    "ToolUse",
    "detect_provider",
]
