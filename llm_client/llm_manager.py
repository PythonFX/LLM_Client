from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .base import BaseLLMClient
from .models import LLMResponse, Message, Provider, StreamChunk, ToolDef


class LLMManager(BaseLLMClient):
    def __init__(self, default_provider: Optional[str] = None) -> None:
        super().__init__()
        self._clients: Dict[str, BaseLLMClient] = {}
        self._default_provider = default_provider

    def add_client(self, name: str, client: BaseLLMClient, default: bool = False) -> None:
        self._clients[name] = client
        if default or not self._default_provider:
            self._default_provider = name

    def remove_client(self, name: str) -> None:
        if name not in self._clients:
            raise KeyError(f"Client '{name}' not found")
        del self._clients[name]
        if self._default_provider == name:
            self._default_provider = next(iter(self._clients), None)

    def get_client(self, provider: Optional[str] = None) -> BaseLLMClient:
        name = provider or self._default_provider
        if not name:
            raise ValueError("No LLM client registered. Use add_client() first.")
        if name not in self._clients:
            raise KeyError(f"Client '{name}' not found. Available: {list(self._clients.keys())}")
        return self._clients[name]

    @property
    def default_provider(self) -> Optional[str]:
        return self._default_provider

    @property
    def available_providers(self) -> List[str]:
        return list(self._clients.keys())

    def completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self.get_client(provider)
        return client.completion(messages, model=model, system=system, tools=tools,
                                 max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def async_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self.get_client(provider)
        return await client.async_completion(messages, model=model, system=system, tools=tools,
                                             max_tokens=max_tokens, temperature=temperature, **kwargs)

    def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self.get_client(provider)
        return client.stream(messages, model=model, system=system, tools=tools,
                             max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def async_stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self.get_client(provider)
        return client.async_stream(messages, model=model, system=system, tools=tools,
                                   max_tokens=max_tokens, temperature=temperature, **kwargs)

    def close(self) -> None:
        for client in self._clients.values():
            client.close()

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
