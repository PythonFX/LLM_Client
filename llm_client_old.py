from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable, Dict, Iterator, List, Optional, Union

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class ToolUse(BaseModel):
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ThinkingBlock(BaseModel):
    thinking: str = ""


class LLMResponse(BaseModel):
    content: str = ""
    thinking: str = ""
    tool_uses: List[ToolUse] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    raw: Any = Field(default=None, exclude=True)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_uses) > 0


class StreamEvent(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    DONE = "done"


@dataclass
class StreamChunk:
    event: StreamEvent
    data: Any = None


@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


AzureTokenProvider = Union[str, Callable[[], str], Callable[[], Awaitable[str]]]


def _detect_provider(model: str) -> Provider:
    if model.startswith("claude"):
        return Provider.ANTHROPIC
    if "minimax" in model.lower():
        return Provider.ANTHROPIC
    model_lower = model.strip().lower()
    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return Provider.OPENAI
    return Provider.ANTHROPIC


class LLMClient:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        anthropic_base_url: Optional[str] = None,
        anthropic_auth_mode: str = "x-api-key",
        azure_deployment: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-06-01",
        azure_api_key: Optional[str] = None,
        azure_ad_token_provider: Optional[AzureTokenProvider] = None,
        azure_ad_token_ttl: float = 1800.0,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self._openai_key = openai_api_key
        self._openai_base = openai_base_url
        self._anthropic_key = anthropic_api_key
        self._anthropic_base = anthropic_base_url or "https://api.anthropic.com"
        self._anthropic_auth_mode = anthropic_auth_mode
        self._azure_deployment = azure_deployment
        self._azure_endpoint = (azure_endpoint or "").rstrip("/")
        self._azure_api_version = azure_api_version
        self._azure_api_key = azure_api_key
        self._azure_ad_token_provider = azure_ad_token_provider
        self._azure_ad_token_ttl = azure_ad_token_ttl
        self._cached_ad_token: str = ""
        self._cached_ad_token_at: float = 0.0
        self._token_lock = threading.Lock()
        self._provider = provider
        self._default_model = model or (f"azure/{azure_deployment}" if azure_deployment else None)
        self._http = httpx.Client(timeout=300.0)
        self._ahttp = httpx.AsyncClient(timeout=300.0)

    def _resolve_provider(self, model: str) -> Provider:
        if self._provider:
            if self._provider in ("minimax",):
                return Provider.ANTHROPIC
            try:
                return Provider(self._provider)
            except ValueError:
                pass
        return _detect_provider(model)

    def _is_ad_token_expired(self) -> bool:
        if not self._cached_ad_token:
            return True
        return (time.monotonic() - self._cached_ad_token_at) >= self._azure_ad_token_ttl

    def _resolve_azure_token(self) -> str:
        if self._azure_api_key:
            return ""
        with self._token_lock:
            if not self._is_ad_token_expired():
                return self._cached_ad_token
        provider = self._azure_ad_token_provider
        if provider is None:
            return ""
        if isinstance(provider, str):
            token = provider
        else:
            token = provider()
        with self._token_lock:
            self._cached_ad_token = token
            self._cached_ad_token_at = time.monotonic()
        return token

    async def _async_resolve_azure_token(self) -> str:
        if self._azure_api_key:
            return ""
        with self._token_lock:
            if not self._is_ad_token_expired():
                return self._cached_ad_token
        provider = self._azure_ad_token_provider
        if provider is None:
            return ""
        if isinstance(provider, str):
            token = provider
        else:
            result = provider()
            if isinstance(result, Awaitable):
                token = await result
            else:
                token = result
        with self._token_lock:
            self._cached_ad_token = token
            self._cached_ad_token_at = time.monotonic()
        return token

    def _openai_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._openai_key}",
            "Content-Type": "application/json",
        }

    def _anthropic_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._anthropic_auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self._anthropic_key}"
        else:
            headers["x-api-key"] = self._anthropic_key
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def _azure_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._azure_api_key:
            headers["api-key"] = self._azure_api_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _azure_url(self, deployment: Optional[str] = None) -> str:
        dep = deployment or self._azure_deployment
        if not dep:
            raise ValueError("azure_deployment must be provided either at init or per-call")
        return (
            f"{self._azure_endpoint}/deployments/{dep}"
            f"/chat/completions?api-version={self._azure_api_version}"
        )

    def _build_openai_tools(self, tools: Optional[List[ToolDef]]) -> Optional[List[Dict]]:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _build_anthropic_tools(self, tools: Optional[List[ToolDef]]) -> Optional[List[Dict]]:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _messages_to_openai(self, messages: List[Message], system: Optional[str] = None) -> List[Dict]:
        result: List[Dict[str, Any]] = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            if msg.tool_call_id:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]) if isinstance(tc["input"], dict) else tc["input"],
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    def _messages_to_anthropic(self, messages: List[Message]) -> List[Dict]:
        result: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.tool_call_id:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                content_blocks: List[Dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"] if isinstance(tc["input"], dict) else json.loads(tc["input"]),
                    })
                result.append({"role": "assistant", "content": content_blocks})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    def _parse_openai_response(self, resp: Dict) -> LLMResponse:
        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        stop_reason = "end_turn"
        if finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"

        tool_uses: List[ToolUse] = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_uses.append(ToolUse(id=tc.get("id", ""), name=fn.get("name", ""), input=args))

        usage = resp.get("usage", {})
        return LLMResponse(
            content=content,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            raw=resp,
        )

    def _parse_anthropic_response(self, resp: Dict) -> LLMResponse:
        content_text = ""
        thinking_text = ""
        tool_uses: List[ToolUse] = []

        for block in resp.get("content", []):
            block_type = block.get("type", "")
            if block_type == "text":
                content_text += block.get("text", "")
            elif block_type == "thinking":
                thinking_text += block.get("thinking", "")
            elif block_type == "tool_use":
                tool_uses.append(ToolUse(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ))

        usage = resp.get("usage", {})
        return LLMResponse(
            content=content_text,
            thinking=thinking_text,
            tool_uses=tool_uses,
            stop_reason=resp.get("stop_reason"),
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            raw=resp,
        )

    def completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")
        provider = self._resolve_provider(model)

        if provider == Provider.ANTHROPIC:
            return self._anthropic_completion(model, messages, system, tools, max_tokens, temperature, **kwargs)
        elif provider == Provider.AZURE:
            return self._azure_completion(model, messages, system, tools, max_tokens, temperature, **kwargs)
        return self._openai_completion(model, messages, system, tools, max_tokens, temperature, **kwargs)

    def _openai_completion(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self._openai_base or 'https://api.openai.com'}/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        resp = self._http.post(url, headers=self._openai_headers(), json=body)
        resp.raise_for_status()
        return self._parse_openai_response(resp.json())

    def _azure_completion(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        token = self._resolve_azure_token()
        url = self._azure_url(model if model != self._azure_deployment else None)
        body: Dict[str, Any] = {
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        resp = self._http.post(url, headers=self._azure_headers(token), json=body)
        resp.raise_for_status()
        return self._parse_openai_response(resp.json())

    def _anthropic_completion(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self._anthropic_base}/v1/messages"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_anthropic(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if system:
            body["system"] = system
        ant_tools = self._build_anthropic_tools(tools)
        if ant_tools:
            body["tools"] = ant_tools
        body.update(kwargs)

        resp = self._http.post(url, headers=self._anthropic_headers(), json=body)
        resp.raise_for_status()
        return self._parse_anthropic_response(resp.json())

    async def async_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")
        provider = self._resolve_provider(model)

        if provider == Provider.ANTHROPIC:
            return await self._anthropic_async_completion(model, messages, system, tools, max_tokens, temperature, **kwargs)
        elif provider == Provider.AZURE:
            return await self._azure_async_completion(model, messages, system, tools, max_tokens, temperature, **kwargs)
        return await self._openai_async_completion(model, messages, system, tools, max_tokens, temperature, **kwargs)

    async def _openai_async_completion(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self._openai_base or 'https://api.openai.com'}/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        resp = await self._ahttp.post(url, headers=self._openai_headers(), json=body)
        resp.raise_for_status()
        return self._parse_openai_response(resp.json())

    async def _azure_async_completion(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        token = await self._async_resolve_azure_token()
        url = self._azure_url(model if model != self._azure_deployment else None)
        body: Dict[str, Any] = {
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        resp = await self._ahttp.post(url, headers=self._azure_headers(token), json=body)
        resp.raise_for_status()
        return self._parse_openai_response(resp.json())

    async def _anthropic_async_completion(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        url = f"{self._anthropic_base}/v1/messages"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_anthropic(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if system:
            body["system"] = system
        ant_tools = self._build_anthropic_tools(tools)
        if ant_tools:
            body["tools"] = ant_tools
        body.update(kwargs)

        resp = await self._ahttp.post(url, headers=self._anthropic_headers(), json=body)
        resp.raise_for_status()
        return self._parse_anthropic_response(resp.json())

    def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")
        provider = self._resolve_provider(model)

        if provider == Provider.ANTHROPIC:
            yield from self._anthropic_stream(model, messages, system, tools, max_tokens, temperature, **kwargs)
        elif provider == Provider.AZURE:
            yield from self._azure_stream(model, messages, system, tools, max_tokens, temperature, **kwargs)
        else:
            yield from self._openai_stream(model, messages, system, tools, max_tokens, temperature, **kwargs)

    def _openai_stream(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        url = f"{self._openai_base or 'https://api.openai.com'}/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        current_tools: Dict[int, Dict[str, Any]] = {}

        with self._http.stream("POST", url, headers=self._openai_headers(), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    yield StreamChunk(event=StreamEvent.DONE)
                    return

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                if delta.get("content"):
                    yield StreamChunk(event=StreamEvent.TEXT, data=delta["content"])

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in current_tools:
                        current_tools[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_START,
                            data={"id": current_tools[idx]["id"], "name": current_tools[idx]["name"], "index": idx},
                        )

                    fn_delta = tc.get("function", {})
                    if fn_delta.get("name"):
                        current_tools[idx]["name"] = fn_delta["name"]
                    if fn_delta.get("arguments"):
                        current_tools[idx]["arguments"] += fn_delta["arguments"]
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_DELTA,
                            data={"index": idx, "arguments": fn_delta["arguments"]},
                        )

                if finish_reason:
                    for idx in sorted(current_tools):
                        args_str = current_tools[idx]["arguments"]
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_END,
                            data={
                                "index": idx,
                                "id": current_tools[idx]["id"],
                                "name": current_tools[idx]["name"],
                                "input": args,
                            },
                        )
                    yield StreamChunk(event=StreamEvent.DONE)

    def _azure_stream(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        token = self._resolve_azure_token()
        url = self._azure_url(model if model != self._azure_deployment else None)
        body: Dict[str, Any] = {
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        current_tools: Dict[int, Dict[str, Any]] = {}

        with self._http.stream("POST", url, headers=self._azure_headers(token), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    yield StreamChunk(event=StreamEvent.DONE)
                    return

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                if delta.get("content"):
                    yield StreamChunk(event=StreamEvent.TEXT, data=delta["content"])

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in current_tools:
                        current_tools[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_START,
                            data={"id": current_tools[idx]["id"], "name": current_tools[idx]["name"], "index": idx},
                        )

                    fn_delta = tc.get("function", {})
                    if fn_delta.get("name"):
                        current_tools[idx]["name"] = fn_delta["name"]
                    if fn_delta.get("arguments"):
                        current_tools[idx]["arguments"] += fn_delta["arguments"]
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_DELTA,
                            data={"index": idx, "arguments": fn_delta["arguments"]},
                        )

                if finish_reason:
                    for idx in sorted(current_tools):
                        args_str = current_tools[idx]["arguments"]
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_END,
                            data={
                                "index": idx,
                                "id": current_tools[idx]["id"],
                                "name": current_tools[idx]["name"],
                                "input": args,
                            },
                        )
                    yield StreamChunk(event=StreamEvent.DONE)

    def _anthropic_stream(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        url = f"{self._anthropic_base}/v1/messages"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_anthropic(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            body["system"] = system
        ant_tools = self._build_anthropic_tools(tools)
        if ant_tools:
            body["tools"] = ant_tools
        body.update(kwargs)

        current_tool: Dict[str, Any] = {}
        current_tool_index: Optional[int] = None

        with self._http.stream("POST", url, headers=self._anthropic_headers(), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    block_type = block.get("type", "")
                    idx = event.get("index", 0)
                    if block_type == "text":
                        pass
                    elif block_type == "thinking":
                        pass
                    elif block_type == "tool_use":
                        current_tool = {"id": block.get("id", ""), "name": block.get("name", ""), "input": ""}
                        current_tool_index = idx
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_START,
                            data={"id": current_tool["id"], "name": current_tool["name"], "index": idx},
                        )

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")
                    if delta_type == "text_delta":
                        yield StreamChunk(event=StreamEvent.TEXT, data=delta.get("text", ""))
                    elif delta_type == "thinking_delta":
                        yield StreamChunk(event=StreamEvent.THINKING, data=delta.get("thinking", ""))
                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        if current_tool is not None:
                            current_tool["input"] += partial
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_DELTA,
                            data={"index": current_tool_index, "arguments": partial},
                        )

                elif event_type == "content_block_stop":
                    if current_tool and current_tool_index is not None:
                        try:
                            parsed = json.loads(current_tool["input"]) if current_tool["input"] else {}
                        except json.JSONDecodeError:
                            parsed = {"raw": current_tool["input"]}
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_END,
                            data={
                                "index": current_tool_index,
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "input": parsed,
                            },
                        )
                        current_tool = {}
                        current_tool_index = None

                elif event_type == "message_stop":
                    yield StreamChunk(event=StreamEvent.DONE)

    async def async_stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")
        provider = self._resolve_provider(model)

        if provider == Provider.ANTHROPIC:
            async for chunk in self._anthropic_async_stream(model, messages, system, tools, max_tokens, temperature, **kwargs):
                yield chunk
        elif provider == Provider.AZURE:
            async for chunk in self._azure_async_stream(model, messages, system, tools, max_tokens, temperature, **kwargs):
                yield chunk
        else:
            async for chunk in self._openai_async_stream(model, messages, system, tools, max_tokens, temperature, **kwargs):
                yield chunk

    async def _openai_async_stream(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        url = f"{self._openai_base or 'https://api.openai.com'}/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        current_tools: Dict[int, Dict[str, Any]] = {}

        async with self._ahttp.stream("POST", url, headers=self._openai_headers(), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    yield StreamChunk(event=StreamEvent.DONE)
                    return

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                if delta.get("content"):
                    yield StreamChunk(event=StreamEvent.TEXT, data=delta["content"])

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in current_tools:
                        current_tools[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_START,
                            data={"id": current_tools[idx]["id"], "name": current_tools[idx]["name"], "index": idx},
                        )

                    fn_delta = tc.get("function", {})
                    if fn_delta.get("name"):
                        current_tools[idx]["name"] = fn_delta["name"]
                    if fn_delta.get("arguments"):
                        current_tools[idx]["arguments"] += fn_delta["arguments"]
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_DELTA,
                            data={"index": idx, "arguments": fn_delta["arguments"]},
                        )

                if finish_reason:
                    for idx in sorted(current_tools):
                        args_str = current_tools[idx]["arguments"]
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_END,
                            data={
                                "index": idx,
                                "id": current_tools[idx]["id"],
                                "name": current_tools[idx]["name"],
                                "input": args,
                            },
                        )
                    yield StreamChunk(event=StreamEvent.DONE)

    async def _azure_async_stream(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        token = await self._async_resolve_azure_token()
        url = self._azure_url(model if model != self._azure_deployment else None)
        body: Dict[str, Any] = {
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        oai_tools = self._build_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)

        current_tools: Dict[int, Dict[str, Any]] = {}

        async with self._ahttp.stream("POST", url, headers=self._azure_headers(token), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    yield StreamChunk(event=StreamEvent.DONE)
                    return

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                if delta.get("content"):
                    yield StreamChunk(event=StreamEvent.TEXT, data=delta["content"])

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in current_tools:
                        current_tools[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_START,
                            data={"id": current_tools[idx]["id"], "name": current_tools[idx]["name"], "index": idx},
                        )

                    fn_delta = tc.get("function", {})
                    if fn_delta.get("name"):
                        current_tools[idx]["name"] = fn_delta["name"]
                    if fn_delta.get("arguments"):
                        current_tools[idx]["arguments"] += fn_delta["arguments"]
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_DELTA,
                            data={"index": idx, "arguments": fn_delta["arguments"]},
                        )

                if finish_reason:
                    for idx in sorted(current_tools):
                        args_str = current_tools[idx]["arguments"]
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_END,
                            data={
                                "index": idx,
                                "id": current_tools[idx]["id"],
                                "name": current_tools[idx]["name"],
                                "input": args,
                            },
                        )
                    yield StreamChunk(event=StreamEvent.DONE)

    async def _anthropic_async_stream(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        url = f"{self._anthropic_base}/v1/messages"
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_anthropic(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            body["system"] = system
        ant_tools = self._build_anthropic_tools(tools)
        if ant_tools:
            body["tools"] = ant_tools
        body.update(kwargs)

        current_tool: Dict[str, Any] = {}
        current_tool_index: Optional[int] = None

        async with self._ahttp.stream("POST", url, headers=self._anthropic_headers(), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    block_type = block.get("type", "")
                    idx = event.get("index", 0)
                    if block_type == "tool_use":
                        current_tool = {"id": block.get("id", ""), "name": block.get("name", ""), "input": ""}
                        current_tool_index = idx
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_START,
                            data={"id": current_tool["id"], "name": current_tool["name"], "index": idx},
                        )

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")
                    if delta_type == "text_delta":
                        yield StreamChunk(event=StreamEvent.TEXT, data=delta.get("text", ""))
                    elif delta_type == "thinking_delta":
                        yield StreamChunk(event=StreamEvent.THINKING, data=delta.get("thinking", ""))
                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        if current_tool is not None:
                            current_tool["input"] += partial
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_DELTA,
                            data={"index": current_tool_index, "arguments": partial},
                        )

                elif event_type == "content_block_stop":
                    if current_tool and current_tool_index is not None:
                        try:
                            parsed = json.loads(current_tool["input"]) if current_tool["input"] else {}
                        except json.JSONDecodeError:
                            parsed = {"raw": current_tool["input"]}
                        yield StreamChunk(
                            event=StreamEvent.TOOL_USE_END,
                            data={
                                "index": current_tool_index,
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "input": parsed,
                            },
                        )
                        current_tool = {}
                        current_tool_index = None

                elif event_type == "message_stop":
                    yield StreamChunk(event=StreamEvent.DONE)

    def collect_stream(self, chunks: Iterator[StreamChunk]) -> LLMResponse:
        content = ""
        thinking = ""
        tool_uses: List[ToolUse] = []
        stop_reason = "end_turn"

        for chunk in chunks:
            if chunk.event == StreamEvent.TEXT:
                content += chunk.data
            elif chunk.event == StreamEvent.THINKING:
                thinking += chunk.data
            elif chunk.event == StreamEvent.TOOL_USE_END:
                tool_uses.append(ToolUse(
                    id=chunk.data["id"],
                    name=chunk.data["name"],
                    input=chunk.data["input"],
                ))
                stop_reason = "tool_use"
            elif chunk.event == StreamEvent.DONE:
                break

        if tool_uses:
            stop_reason = "tool_use"

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
        )

    async def async_collect_stream(self, chunks: AsyncIterator[StreamChunk]) -> LLMResponse:
        content = ""
        thinking = ""
        tool_uses: List[ToolUse] = []
        stop_reason = "end_turn"

        try:
            async for chunk in chunks:
                if chunk.event == StreamEvent.TEXT:
                    content += chunk.data
                elif chunk.event == StreamEvent.THINKING:
                    thinking += chunk.data
                elif chunk.event == StreamEvent.TOOL_USE_END:
                    tool_uses.append(ToolUse(
                        id=chunk.data["id"],
                        name=chunk.data["name"],
                        input=chunk.data["input"],
                    ))
                    stop_reason = "tool_use"
                elif chunk.event == StreamEvent.DONE:
                    break
        finally:
            # Ensure the underlying httpx stream is fully drained/closed to prevent
            # "async generator ignored GeneratorExit" warnings from httpcore.
            if hasattr(chunks, "aclose"):
                await chunks.aclose()

        if tool_uses:
            stop_reason = "tool_use"

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
        )

    def close(self) -> None:
        self._http.close()

    async def aclose(self) -> None:
        await self._ahttp.aclose()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
