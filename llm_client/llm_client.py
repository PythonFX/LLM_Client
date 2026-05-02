from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .base import BaseLLMClient
from .models import LLMResponse, Message, Provider, StreamChunk, ToolDef


def _to_provider(value: Union[Provider, str]) -> Provider:
    if isinstance(value, Provider):
        return value
    try:
        return Provider(value)
    except ValueError:
        raise KeyError(f"'{value}' is not a valid Provider. Valid: {[p.value for p in Provider]}")


class LLMClient(BaseLLMClient):
    def __init__(self, default_provider: Optional[Union[Provider, str]] = None) -> None:
        super().__init__()
        self._clients: Dict[Provider, BaseLLMClient] = {}
        self._default_provider: Optional[Provider] = _to_provider(default_provider) if default_provider else None

    def add_client(self, provider: Union[Provider, str], client: BaseLLMClient, default: bool = False) -> None:
        p = _to_provider(provider)
        self._clients[p] = client
        if default or not self._default_provider:
            self._default_provider = p

    def remove_client(self, provider: Union[Provider, str]) -> None:
        p = _to_provider(provider)
        if p not in self._clients:
            raise KeyError(f"Client '{p.value}' not found")
        del self._clients[p]
        if self._default_provider == p:
            self._default_provider = next(iter(self._clients), None)

    def get_client(self, provider: Optional[Union[Provider, str]] = None) -> BaseLLMClient:
        p = _to_provider(provider) if provider else self._default_provider
        if not p:
            raise ValueError("No LLM client registered. Use add_client() first.")
        if p not in self._clients:
            raise KeyError(f"Client '{p.value}' not found. Available: {[p.value for p in self._clients]}")
        return self._clients[p]

    @property
    def default_provider(self) -> Optional[Provider]:
        return self._default_provider

    def set_default_provider(self, provider: Union[Provider, str]) -> None:
        p = _to_provider(provider)
        if p not in self._clients:
            raise KeyError(f"Client '{p.value}' not found. Available: {[p.value for p in self._clients]}")
        self._default_provider = p

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
        yield from client.stream(messages, model=model, system=system, tools=tools,
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
        async for chunk in client.async_stream(messages, model=model, system=system, tools=tools,
                                               max_tokens=max_tokens, temperature=temperature, **kwargs):
            yield chunk

    def close(self) -> None:
        for client in self._clients.values():
            client.close()

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
