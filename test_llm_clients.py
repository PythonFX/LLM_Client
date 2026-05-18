"""Comprehensive tests for llm_client package.

Unit tests (no API) + Integration tests (require YAML config with credentials).
"""

import os
import asyncio
import subprocess
import sys
import json
import threading
from unittest.mock import MagicMock, patch

from llm_client import (
    LLMClient,
    AnthropicClient,
    OpenAIClient,
    AzureClient,
    MlxClient,
    BaseLLMClient,
    LLMResponse,
    Message,
    Provider,
    StreamChunk,
    StreamEvent,
    ToolDef,
    ToolUse,
    ThinkingBlock,
    AzureTokenProvider,
    detect_provider,
    get_config,
    get_profile,
    create_anthropic_client,
    create_openai_client,
    create_doubao_client,
    create_kimi_client,
    create_mlx_client,
    create_from_profiles,
)
from llm_client.llm_factory import create_zhipu_client

_cfg = get_config()
_profiles = _cfg.get("profiles", {})
MODEL = _profiles.get("minimax-anthropic", {}).get("model", "MiniMax-M2.7-highspeed")
MAX_TOKENS = 1024
HAS_API = bool(_profiles.get("minimax-anthropic", {}).get("api_key"))
HAS_DOUBAO = bool(_profiles.get("doubao-glm", {}).get("api_key"))
HAS_ZHIPU = bool(_profiles.get("zhipu-glm", {}).get("api_key"))
HAS_KIMI = bool(_profiles.get("kimi-k26", {}).get("api_key"))
HAS_MLX = os.path.exists("/Users/vincent/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit")


# ─── Unit Tests (no API calls) ─────────────────────────────────────────


class TestModels:
    def test_provider_enum(self):
        assert Provider.OPENAI == "openai"
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.AZURE == "azure"
        assert Provider.DOUBAO == "doubao"
        assert Provider.KIMI == "kimi"
        assert Provider.MLX == "mlx"

    def test_detect_provider(self):
        assert detect_provider("claude-3-opus") == Provider.ANTHROPIC
        assert detect_provider("claude-3.5-sonnet") == Provider.ANTHROPIC
        assert detect_provider("gpt-4") == Provider.OPENAI
        assert detect_provider("gpt-4o") == Provider.OPENAI
        assert detect_provider("o1-preview") == Provider.OPENAI
        assert detect_provider("o3-mini") == Provider.OPENAI
        assert detect_provider("o4-mini") == Provider.OPENAI
        assert detect_provider("MiniMax-M2.7") == Provider.ANTHROPIC
        assert detect_provider("minimax-01") == Provider.ANTHROPIC
        assert detect_provider("doubao-pro") == Provider.DOUBAO
        assert detect_provider("Doubao-1.5") == Provider.DOUBAO
        assert detect_provider("kimi-k2.6") == Provider.KIMI
        assert detect_provider("Kimi-latest") == Provider.KIMI
        assert detect_provider("unknown-model") == Provider.ANTHROPIC

    def test_llm_response(self):
        r = LLMResponse(content="hi", stop_reason="end_turn", usage={"input_tokens": 5, "output_tokens": 2})
        assert r.content == "hi"
        assert not r.has_tool_calls
        assert r.stop_reason == "end_turn"

    def test_llm_response_tool_calls(self):
        r = LLMResponse(tool_uses=[ToolUse(id="1", name="fn", input={"x": 1})])
        assert r.has_tool_calls

    def test_tool_use(self):
        t = ToolUse(id="t1", name="search", input={"q": "test"})
        assert t.name == "search"
        assert t.input == {"q": "test"}

    def test_message_basic(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"
        assert m.tool_calls is None
        assert m.tool_call_id is None

    def test_message_tool_call(self):
        m = Message(role="assistant", tool_calls=[{"id": "t1", "name": "fn", "input": {}}])
        assert m.tool_calls is not None

    def test_message_tool_result(self):
        m = Message(role="tool", tool_call_id="t1", content="result")
        assert m.tool_call_id == "t1"

    def test_stream_event_enum(self):
        assert StreamEvent.TEXT == "text"
        assert StreamEvent.THINKING == "thinking"
        assert StreamEvent.TOOL_USE_START == "tool_use_start"
        assert StreamEvent.TOOL_USE_DELTA == "tool_use_delta"
        assert StreamEvent.TOOL_USE_END == "tool_use_end"
        assert StreamEvent.DONE == "done"

    def test_stream_chunk(self):
        c = StreamChunk(event=StreamEvent.TEXT, data="hello")
        assert c.event == StreamEvent.TEXT
        assert c.data == "hello"

    def test_tool_def(self):
        t = ToolDef(name="search", description="Search", parameters={"type": "object"})
        assert t.name == "search"

    def test_thinking_block(self):
        t = ThinkingBlock(thinking="hmm")
        assert t.thinking == "hmm"

    def test_llm_response_raw_excluded(self):
        r = LLMResponse(content="hi", raw={"secret": True})
        d = r.model_dump()
        assert "raw" not in d


class TestBaseLLMClient:
    def test_abstract_cannot_instantiate(self):
        import inspect
        assert inspect.isabstract(BaseLLMClient)
        try:
            BaseLLMClient(model="test")
            assert False, "Should raise TypeError"
        except TypeError:
            pass

    def test_no_httpx_clients_in_base(self):
        mgr = LLMClient()
        assert not hasattr(mgr, "_http")
        assert not hasattr(mgr, "_ahttp")

    def test_close_aclose_safe_without_httpx(self):
        mgr = LLMClient()
        mgr.close()  # should not crash
        asyncio.run(mgr.aclose())  # should not crash

    def test_context_manager(self):
        with LLMClient() as mgr:
            assert isinstance(mgr, LLMClient)

    def test_async_context_manager(self):
        async def run():
            async with LLMClient() as mgr:
                assert isinstance(mgr, LLMClient)
        asyncio.run(run())


class TestCollectStream:
    def test_collect_stream_text(self):
        client = OpenAIClient(api_key="fake", model="gpt-4")
        chunks = [
            StreamChunk(event=StreamEvent.TEXT, data="Hello "),
            StreamChunk(event=StreamEvent.TEXT, data="World"),
            StreamChunk(event=StreamEvent.DONE),
        ]
        resp = client.collect_stream(iter(chunks))
        assert resp.content == "Hello World"
        assert resp.stop_reason == "end_turn"

    def test_collect_stream_thinking(self):
        client = AnthropicClient(api_key="fake", model="claude-3")
        chunks = [
            StreamChunk(event=StreamEvent.THINKING, data="hmm..."),
            StreamChunk(event=StreamEvent.TEXT, data="answer"),
            StreamChunk(event=StreamEvent.DONE),
        ]
        resp = client.collect_stream(iter(chunks))
        assert resp.thinking == "hmm..."
        assert resp.content == "answer"

    def test_collect_stream_tool_use(self):
        client = OpenAIClient(api_key="fake", model="gpt-4")
        chunks = [
            StreamChunk(event=StreamEvent.TEXT, data="Let me search"),
            StreamChunk(event=StreamEvent.TOOL_USE_END, data={
                "id": "t1", "name": "search", "input": {"q": "test"}
            }),
            StreamChunk(event=StreamEvent.DONE),
        ]
        resp = client.collect_stream(iter(chunks))
        assert resp.content == "Let me search"
        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_uses) == 1
        assert resp.tool_uses[0].name == "search"

    def test_collect_stream_no_done(self):
        client = OpenAIClient(api_key="fake", model="gpt-4")
        chunks = [
            StreamChunk(event=StreamEvent.TEXT, data="partial"),
        ]
        resp = client.collect_stream(iter(chunks))
        assert resp.content == "partial"

    def test_async_collect_stream(self):
        client = AnthropicClient(api_key="fake", model="claude-3")

        async def gen():
            yield StreamChunk(event=StreamEvent.TEXT, data="async hello")
            yield StreamChunk(event=StreamEvent.DONE)

        async def run():
            resp = await client.async_collect_stream(gen())
            assert resp.content == "async hello"

        asyncio.run(run())

    def test_async_collect_stream_tool_use(self):
        client = OpenAIClient(api_key="fake", model="gpt-4")

        async def gen():
            yield StreamChunk(event=StreamEvent.TOOL_USE_END, data={
                "id": "t2", "name": "calc", "input": {"x": 1}
            })
            yield StreamChunk(event=StreamEvent.DONE)

        async def run():
            resp = await client.async_collect_stream(gen())
            assert resp.stop_reason == "tool_use"
            assert resp.tool_uses[0].name == "calc"

        asyncio.run(run())


class TestOpenAIParsing:
    def test_messages_to_openai_basic(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        result = OpenAIClient._messages_to_openai(msgs)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "hi"}

    def test_messages_to_openai_with_system(self):
        msgs = [Message(role="user", content="hi")]
        result = OpenAIClient._messages_to_openai(msgs, system="You are helpful")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful"}

    def test_messages_to_openai_tool_calls(self):
        msgs = [Message(role="assistant", content="", tool_calls=[
            {"id": "t1", "name": "fn", "input": {"x": 1}}
        ])]
        result = OpenAIClient._messages_to_openai(msgs)
        assert result[0]["tool_calls"][0]["function"]["name"] == "fn"

    def test_messages_to_openai_tool_result(self):
        msgs = [Message(role="tool", tool_call_id="t1", content="result")]
        result = OpenAIClient._messages_to_openai(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "t1"

    def test_build_tools(self):
        tools = [ToolDef(name="fn1", description="desc", parameters={"type": "object"})]
        result = OpenAIClient._build_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "fn1"

    def test_build_tools_none(self):
        assert OpenAIClient._build_tools(None) is None
        assert OpenAIClient._build_tools([]) is None

    def test_parse_response(self):
        raw = {
            "choices": [{
                "message": {"content": "hello", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = OpenAIClient._parse_response(raw)
        assert resp.content == "hello"
        assert resp.stop_reason == "end_turn"
        assert resp.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_parse_response_tool_calls(self):
        raw = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {"name": "fn", "arguments": '{"x": 1}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = OpenAIClient._parse_response(raw)
        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_uses) == 1
        assert resp.tool_uses[0].input == {"x": 1}

    def test_parse_response_length(self):
        raw = {
            "choices": [{"message": {"content": "cut"}, "finish_reason": "length"}],
            "usage": {},
        }
        resp = OpenAIClient._parse_response(raw)
        assert resp.stop_reason == "max_tokens"

    def test_parse_stream_chunks(self):
        current_tools = {}
        line = 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}'
        chunks = OpenAIClient._parse_stream_chunks(line, current_tools)
        assert len(chunks) == 1
        assert chunks[0].event == StreamEvent.TEXT
        assert chunks[0].data == "hi"

    def test_parse_stream_chunks_done(self):
        current_tools = {}
        chunks = OpenAIClient._parse_stream_chunks("data: [DONE]", current_tools)
        assert chunks[0].event == StreamEvent.DONE

    def test_parse_stream_chunks_empty(self):
        current_tools = {}
        assert OpenAIClient._parse_stream_chunks("", current_tools) == []
        assert OpenAIClient._parse_stream_chunks("not data", current_tools) == []


class TestAnthropicParsing:
    def test_messages_to_anthropic_basic(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        result = AnthropicClient._messages_to_anthropic(msgs)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "hi"}

    def test_messages_to_anthropic_tool_use(self):
        msgs = [Message(role="assistant", content="", tool_calls=[
            {"id": "t1", "name": "fn", "input": {"x": 1}}
        ])]
        result = AnthropicClient._messages_to_anthropic(msgs)
        assert result[0]["role"] == "assistant"
        assert any(b["type"] == "tool_use" for b in result[0]["content"])

    def test_messages_to_anthropic_tool_result(self):
        msgs = [Message(role="tool", tool_call_id="t1", content="ok")]
        result = AnthropicClient._messages_to_anthropic(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"

    def test_build_tools(self):
        tools = [ToolDef(name="fn1", description="desc", parameters={"type": "object"})]
        result = AnthropicClient._build_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "fn1"
        assert "input_schema" in result[0]

    def test_build_tools_none(self):
        assert AnthropicClient._build_tools(None) is None

    def test_parse_response_text(self):
        raw = {
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        resp = AnthropicClient._parse_response(raw)
        assert resp.content == "hello"
        assert resp.stop_reason == "end_turn"

    def test_parse_response_thinking(self):
        raw = {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {},
        }
        resp = AnthropicClient._parse_response(raw)
        assert resp.thinking == "hmm"
        assert resp.content == "answer"

    def test_parse_response_tool_use(self):
        raw = {
            "content": [
                {"type": "text", "text": "calling fn"},
                {"type": "tool_use", "id": "t1", "name": "fn", "input": {"x": 1}},
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        resp = AnthropicClient._parse_response(raw)
        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_uses) == 1

    def test_parse_stream_event_text(self):
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}
        chunks, tool, idx = AnthropicClient._parse_stream_event(event, {}, None)
        assert chunks[0].event == StreamEvent.TEXT
        assert chunks[0].data == "hi"

    def test_parse_stream_event_thinking(self):
        event = {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}}
        chunks, _, _ = AnthropicClient._parse_stream_event(event, {}, None)
        assert chunks[0].event == StreamEvent.THINKING

    def test_parse_stream_event_tool_start(self):
        event = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "fn"},
        }
        chunks, tool, idx = AnthropicClient._parse_stream_event(event, {}, None)
        assert chunks[0].event == StreamEvent.TOOL_USE_START
        assert idx == 0

    def test_parse_stream_event_tool_delta(self):
        event = {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"x":'},
        }
        current_tool = {"id": "t1", "name": "fn", "input": ""}
        chunks, tool, idx = AnthropicClient._parse_stream_event(event, current_tool, 0)
        assert chunks[0].event == StreamEvent.TOOL_USE_DELTA
        assert tool["input"] == '{"x":'

    def test_parse_stream_event_tool_end(self):
        current_tool = {"id": "t1", "name": "fn", "input": '{"x":1}'}
        event = {"type": "content_block_stop", "index": 0}
        chunks, tool, idx = AnthropicClient._parse_stream_event(event, current_tool, 0)
        assert chunks[0].event == StreamEvent.TOOL_USE_END
        assert chunks[0].data["input"] == {"x": 1}

    def test_parse_stream_event_done(self):
        event = {"type": "message_stop"}
        chunks, _, _ = AnthropicClient._parse_stream_event(event, {}, None)
        assert chunks[0].event == StreamEvent.DONE


class TestAzureToken:
    def test_api_key_mode(self):
        client = AzureClient(deployment="d", endpoint="https://x.openai.azure.com", api_key="key")
        assert client._resolve_azure_token() == ""
        assert asyncio.run(client._async_resolve_azure_token()) == ""

    def test_string_token_provider(self):
        client = AzureClient(
            deployment="d", endpoint="https://x.openai.azure.com",
            ad_token_provider="static-token", ad_token_ttl=9999,
        )
        assert client._resolve_azure_token() == "static-token"

    def test_callable_token_provider(self):
        call_count = [0]
        def provider():
            call_count[0] += 1
            return f"token-{call_count[0]}"
        client = AzureClient(
            deployment="d", endpoint="https://x.openai.azure.com",
            ad_token_provider=provider, ad_token_ttl=9999,
        )
        t1 = client._resolve_azure_token()
        t2 = client._resolve_azure_token()
        assert t1 == "token-1"
        assert t2 == "token-1"  # cached

    def test_token_expiry(self):
        client = AzureClient(
            deployment="d", endpoint="https://x.openai.azure.com",
            ad_token_provider=lambda: "new-token", ad_token_ttl=0.0,
        )
        client._resolve_azure_token()
        # Token should be expired immediately with ttl=0
        assert client._is_ad_token_expired()

    def test_async_callable_token(self):
        async def provider():
            return "async-token"

        async def run():
            client = AzureClient(
                deployment="d", endpoint="https://x.openai.azure.com",
                ad_token_provider=provider, ad_token_ttl=9999,
            )
            t = await client._async_resolve_azure_token()
            assert t == "async-token"

        asyncio.run(run())

    def test_no_provider(self):
        client = AzureClient(deployment="d", endpoint="https://x.openai.azure.com")
        assert client._resolve_azure_token() == ""

    def test_url(self):
        client = AzureClient(
            deployment="my-deploy",
            endpoint="https://my.openai.azure.com",
            api_version="2024-06-01",
        )
        url = client._url()
        assert "my-deploy" in url
        assert "api-version=2024-06-01" in url

    def test_url_override_deployment(self):
        client = AzureClient(
            deployment="default-deploy",
            endpoint="https://my.openai.azure.com",
        )
        url = client._url("other-deploy")
        assert "other-deploy" in url

    def test_url_no_deployment(self):
        client = AzureClient(deployment="", endpoint="https://x.openai.azure.com")
        try:
            client._url()
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_headers_api_key(self):
        client = AzureClient(deployment="d", endpoint="https://x.openai.azure.com", api_key="k")
        h = client._headers()
        assert h["api-key"] == "k"

    def test_headers_bearer(self):
        client = AzureClient(deployment="d", endpoint="https://x.openai.azure.com")
        h = client._headers(token="my-token")
        assert h["Authorization"] == "Bearer my-token"


class TestMlxClient:
    MODEL_PATH = "/Users/vincent/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit"

    def test_init_defaults(self):
        c = MlxClient(model_path=self.MODEL_PATH)
        assert c._default_model == self.MODEL_PATH
        assert c._enable_thinking is True
        assert c._model is None
        assert c._tokenizer is None

    def test_init_disable_thinking(self):
        c = MlxClient(model_path=self.MODEL_PATH, enable_thinking=False)
        assert c._enable_thinking is False

    def test_strip_control_tokens(self):
        assert MlxClient._strip_control_tokens("<|turn|>") == ""
        assert MlxClient._strip_control_tokens("<turn|>") == ""
        assert MlxClient._strip_control_tokens("hello<|turn|>world") == "helloworld"
        assert MlxClient._strip_control_tokens("plain text") == "plain text"

    def test_parse_response_with_thinking(self):
        text = "<|channel>thoughtsome thinking<channel|>the answer"
        thinking, answer = MlxClient._parse_response(text)
        assert thinking == "some thinking"
        assert answer == "the answer"

    def test_parse_response_no_thinking(self):
        thinking, answer = MlxClient._parse_response("just an answer")
        assert thinking == ""
        assert answer == "just an answer"

    def test_parse_response_with_control_tokens(self):
        text = "<|channel>thoughtthink here<channel|>answer<|turn|>"
        thinking, answer = MlxClient._parse_response(text)
        assert thinking == "think here"
        assert answer == "answer"

    def test_close_resets_model(self):
        c = MlxClient(model_path=self.MODEL_PATH)
        c._model = "fake"
        c._tokenizer = "fake"
        c.close()
        assert c._model is None
        assert c._tokenizer is None

    def test_aclose_resets_model(self):
        async def run():
            c = MlxClient(model_path=self.MODEL_PATH)
            c._model = "fake"
            c._tokenizer = "fake"
            await c.aclose()
            assert c._model is None
            assert c._tokenizer is None
        asyncio.run(run())

    def test_lazy_load_not_called_on_init(self):
        c = MlxClient(model_path=self.MODEL_PATH)
        assert c._model is None
        assert c._tokenizer is None

    def test_mlx_in_manager(self):
        mgr = LLMClient()
        c = MlxClient(model_path=self.MODEL_PATH)
        mgr.add_client("gemma4-e4b", c, default=True)
        assert mgr.get_client() is c
        assert mgr.default_provider == "gemma4-e4b"

    def test_create_from_profiles_mlx(self):
        llm = create_from_profiles()
        assert "gemma4-e4b" in llm.available_profiles
        client = llm.get_client("gemma4-e4b")
        assert isinstance(client, MlxClient)
        assert client._enable_thinking is True


class TestLLMClientManager:
    def test_add_and_get(self):
        mgr = LLMClient()
        c1 = AnthropicClient(api_key="k1", model="claude-3")
        c2 = OpenAIClient(api_key="k2", model="gpt-4")
        mgr.add_client(Provider.ANTHROPIC, c1, default=True)
        mgr.add_client(Provider.OPENAI, c2)
        assert mgr.get_client() is c1
        assert mgr.get_client(Provider.OPENAI) is c2
        assert mgr.get_client("openai") is c2

    def test_default_provider(self):
        mgr = LLMClient()
        c = OpenAIClient(api_key="k", model="gpt-4")
        mgr.add_client(Provider.OPENAI, c)
        assert mgr.default_provider == Provider.OPENAI

    def test_available_providers(self):
        mgr = LLMClient()
        mgr.add_client(Provider.ANTHROPIC, AnthropicClient(api_key="k", model="c"))
        mgr.add_client(Provider.OPENAI, OpenAIClient(api_key="k", model="g"))
        assert set(mgr.available_providers) == {Provider.ANTHROPIC, Provider.OPENAI}

    def test_remove_client(self):
        mgr = LLMClient()
        c1 = AnthropicClient(api_key="k", model="c")
        c2 = OpenAIClient(api_key="k", model="g")
        mgr.add_client(Provider.ANTHROPIC, c1, default=True)
        mgr.add_client(Provider.OPENAI, c2)
        mgr.remove_client(Provider.ANTHROPIC)
        assert Provider.ANTHROPIC not in mgr.available_providers
        assert mgr.default_provider == Provider.OPENAI

    def test_remove_nonexistent(self):
        mgr = LLMClient()
        try:
            mgr.remove_client("nope")
            assert False
        except KeyError:
            pass

    def test_get_client_no_default(self):
        mgr = LLMClient()
        try:
            mgr.get_client()
            assert False
        except ValueError:
            pass

    def test_get_client_unknown_provider(self):
        mgr = LLMClient()
        mgr.add_client(Provider.ANTHROPIC, AnthropicClient(api_key="k", model="c"))
        try:
            mgr.get_client("nonexistent")
            assert False
        except KeyError:
            pass

    def test_close_closes_all(self):
        mgr = LLMClient()
        c1 = AnthropicClient(api_key="k", model="c")
        c2 = OpenAIClient(api_key="k", model="g")
        mgr.add_client(Provider.ANTHROPIC, c1)
        mgr.add_client(Provider.OPENAI, c2)
        mgr.close()  # should not crash

    def test_first_client_auto_default(self):
        mgr = LLMClient()
        mgr.add_client(Provider.OPENAI, OpenAIClient(api_key="k", model="g"))
        assert mgr.default_provider == Provider.OPENAI

    def test_set_default_provider(self):
        mgr = LLMClient()
        mgr.add_client(Provider.ANTHROPIC, AnthropicClient(api_key="k", model="c"))
        mgr.add_client(Provider.OPENAI, OpenAIClient(api_key="k", model="g"), default=True)
        assert mgr.default_provider == Provider.OPENAI
        mgr.set_default_provider(Provider.ANTHROPIC)
        assert mgr.default_provider == Provider.ANTHROPIC
        mgr.set_default_provider("openai")
        assert mgr.default_provider == Provider.OPENAI

    def test_set_default_provider_invalid(self):
        mgr = LLMClient()
        mgr.add_client(Provider.ANTHROPIC, AnthropicClient(api_key="k", model="c"))
        try:
            mgr.set_default_provider("not_a_provider")
            assert False
        except KeyError:
            pass

    def test_no_httpx_in_manager(self):
        mgr = LLMClient()
        assert not hasattr(mgr, "_http")
        assert not hasattr(mgr, "_ahttp")


class TestClientInit:
    def test_openai_default_base_url(self):
        c = OpenAIClient(api_key="k", model="gpt-4")
        assert c._base_url == "https://api.openai.com/v1"

    def test_openai_custom_base_url(self):
        c = OpenAIClient(api_key="k", model="gpt-4", base_url="https://custom.api.com/v1")
        assert c._base_url == "https://custom.api.com/v1"

    def test_openai_base_url_trailing_slash(self):
        c = OpenAIClient(api_key="k", model="gpt-4", base_url="https://custom.api.com/v1/")
        assert c._base_url == "https://custom.api.com/v1"

    def test_anthropic_default_base_url(self):
        c = AnthropicClient(api_key="k", model="claude-3")
        assert c._base_url == "https://api.anthropic.com"

    def test_anthropic_headers_x_api_key(self):
        c = AnthropicClient(api_key="my-key", model="claude-3")
        h = c._headers()
        assert h["x-api-key"] == "my-key"
        assert "anthropic-version" in h

    def test_anthropic_headers_bearer(self):
        c = AnthropicClient(api_key="my-key", model="claude-3", auth_mode="bearer")
        h = c._headers()
        assert h["Authorization"] == "Bearer my-key"
        assert "x-api-key" not in h

    def test_azure_default_model(self):
        c = AzureClient(deployment="my-deploy", endpoint="https://x.openai.azure.com", api_key="k")
        assert c._default_model == "azure/my-deploy"

    def test_model_required_for_completion(self):
        c = OpenAIClient(api_key="k")  # no model
        try:
            c.completion(messages=[Message(role="user", content="hi")])
            assert False
        except ValueError:
            pass

    def test_per_call_model_override(self):
        c = OpenAIClient(api_key="k", model="gpt-4")
        assert c._default_model == "gpt-4"


# ─── Integration Tests (require API credentials) ────────────────────────


def _run_subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
    )


def run_if_api(func):
    def wrapper():
        if not HAS_API:
            print(f"  SKIP (no API key)")
            return
        func()
    wrapper.__name__ = func.__name__
    return wrapper


@run_if_api
def test_anthropic_completion():
    print("=== Anthropic completion ===")
    with create_anthropic_client(profile_name="minimax-anthropic") as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        print(f"  Stop reason: {resp.stop_reason}")
        print(f"  Usage: {resp.usage}")
        assert resp.content, "Empty response"
        assert resp.stop_reason in ("end_turn", "max_tokens", "tool_use")
        assert "input_tokens" in resp.usage
        print("  PASS\n")


@run_if_api
def test_anthropic_completion_with_system():
    print("=== Anthropic completion with system ===")
    with create_anthropic_client(profile_name="minimax-anthropic") as client:
        resp = client.completion(
            messages=[Message(role="user", content="What is your name?")],
            system="You are a helpful assistant named Bot.",
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_anthropic_stream():
    print("=== Anthropic stream ===")
    with create_anthropic_client(profile_name="minimax-anthropic") as client:
        chunks = client.stream(
            messages=[Message(role="user", content="Count from 1 to 3.")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_anthropic_stream_events():
    print("=== Anthropic stream events ===")
    with create_anthropic_client(profile_name="minimax-anthropic") as client:
        events = []
        for chunk in client.stream(
            messages=[Message(role="user", content="Say hi")],
            max_tokens=MAX_TOKENS,
        ):
            events.append(chunk.event)
            if chunk.event == StreamEvent.DONE:
                break
        assert StreamEvent.DONE in events
        print(f"  Events: {[e.value for e in events]}")
        print("  PASS\n")


@run_if_api
def test_openai_completion():
    print("=== OpenAI completion ===")
    with create_openai_client(profile_name="minimax-openai") as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        print(f"  Stop reason: {resp.stop_reason}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_openai_stream():
    print("=== OpenAI stream ===")
    with create_openai_client(profile_name="minimax-openai") as client:
        chunks = client.stream(
            messages=[Message(role="user", content="Count from 1 to 3.")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_llm_client_manager():
    print("=== LLMClient manager ===")
    ant = create_anthropic_client(profile_name="minimax-anthropic")
    oai = create_openai_client(profile_name="minimax-openai")
    with LLMClient() as mgr:
        mgr.add_client("minimax-anthropic", ant, default=True)
        mgr.add_client("minimax-openai", oai)

        resp1 = mgr.completion(
            messages=[Message(role="user", content="Say hi")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  [default/anthropic] {resp1.content}")

        resp2 = mgr.completion(
            messages=[Message(role="user", content="Say hi")],
            max_tokens=MAX_TOKENS,
            provider="minimax-openai",
        )
        print(f"  [openai] {resp2.content}")
        assert resp1.content and resp2.content
        print("  PASS\n")


@run_if_api
def test_llm_client_manager_stream():
    print("=== LLMClient manager stream ===")
    ant = create_anthropic_client(profile_name="minimax-anthropic")
    with LLMClient() as mgr:
        mgr.add_client("minimax-anthropic", ant, default=True)
        chunks = mgr.stream(
            messages=[Message(role="user", content="Say hi")],
            max_tokens=MAX_TOKENS,
        )
        resp = mgr.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_async_completion():
    print("=== Async Anthropic completion ===")
    result = _run_subprocess("""
import asyncio
from llm_client import create_anthropic_client, Message

async def test():
    async with create_anthropic_client(profile_name="minimax-anthropic") as client:
        resp = await client.async_completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=1024,
        )
        print(f"Content: {resp.content}")
        assert resp.content, "Empty response"
        print("PASS")

asyncio.run(test())
""")
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  {result.stderr.strip()}")
        raise RuntimeError("Async test failed")
    print()


@run_if_api
def test_async_stream():
    print("=== Async Anthropic stream ===")
    result = _run_subprocess("""
import asyncio
from llm_client import create_anthropic_client, Message

async def test():
    async with create_anthropic_client(profile_name="minimax-anthropic") as client:
        chunks = client.async_stream(
            messages=[Message(role="user", content="Say hello")],
            max_tokens=1024,
        )
        resp = await client.async_collect_stream(chunks)
        print(f"Content: {resp.content}")
        assert resp.content, "Empty response"
        print("PASS")

asyncio.run(test())
""")
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  {result.stderr.strip()}")
        raise RuntimeError("Async stream test failed")
    print()


@run_if_api
def test_async_openai_completion():
    print("=== Async OpenAI completion ===")
    result = _run_subprocess("""
import asyncio
from llm_client import create_openai_client, Message

async def test():
    async with create_openai_client(profile_name="minimax-openai") as client:
        resp = await client.async_completion(
            messages=[Message(role="user", content="Say hello")],
            max_tokens=1024,
        )
        print(f"Content: {resp.content}")
        assert resp.content, "Empty response"
        print("PASS")

asyncio.run(test())
""")
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  {result.stderr.strip()}")
        raise RuntimeError("Async OpenAI test failed")
    print()


@run_if_api
def test_doubao_completion():
    if not HAS_DOUBAO:
        print("=== Doubao completion === SKIP (no doubao profile)")
        return
    print("=== Doubao completion ===")
    with create_doubao_client(profile_name="doubao-glm") as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content, "Empty response"
        print("  PASS\n")
        

@run_if_api
def test_zhipu_completion():
    if not HAS_ZHIPU:
        print("=== Zhipou completion === SKIP (no Zhipou profile)")
        return
    print("=== Zhipou completion ===")
    with create_zhipu_client() as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content, "Empty response"
        print("  PASS\n")


@run_if_api
def test_doubao_stream():
    if not HAS_DOUBAO:
        print("=== Doubao stream === SKIP (no doubao profile)")
        return
    print("=== Doubao stream ===")
    with create_doubao_client(profile_name="doubao-glm") as client:
        chunks = client.stream(
            messages=[Message(role="user", content="Count from 1 to 3.")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")
        

@run_if_api
def test_zhipu_stream():
    if not HAS_ZHIPU:
        print("=== Zhipou completion === SKIP (no Zhipou profile)")
        return
    print("=== Zhipou stream ===")
    with create_zhipu_client() as client:
        chunks = client.stream(
            messages=[Message(role="user", content="Count from 1 to 3.")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_doubao_completion_with_system():
    if not HAS_DOUBAO:
        print("=== Doubao completion with system === SKIP (no doubao profile)")
        return
    print("=== Doubao completion with system ===")
    with create_doubao_client(profile_name="doubao-glm") as client:
        resp = client.completion(
            messages=[Message(role="user", content="What is your name?")],
            system="You are a helpful assistant named Bot.",
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_doubao_stream_with_system():
    if not HAS_DOUBAO:
        print("=== Doubao stream with system === SKIP (no doubao profile)")
        return
    print("=== Doubao stream with system ===")
    with create_doubao_client(profile_name="doubao-glm") as client:
        chunks = client.stream(
            messages=[Message(role="user", content="What is your name?")],
            system="You are a helpful assistant named Bot.",
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_kimi_completion():
    if not HAS_KIMI:
        print("=== Kimi completion === SKIP (no kimi profile)")
        return
    print("=== Kimi completion ===")
    with create_kimi_client(profile_name="kimi-k26") as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content, "Empty response"
        print("  PASS\n")


@run_if_api
def test_kimi_stream():
    if not HAS_KIMI:
        print("=== Kimi stream === SKIP (no kimi profile)")
        return
    print("=== Kimi stream ===")
    with create_kimi_client(profile_name="kimi-k26") as client:
        chunks = client.stream(
            messages=[Message(role="user", content="Count from 1 to 3.")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_mlx_completion():
    if not HAS_MLX:
        print("=== MLX completion === SKIP (model not found)")
        return
    print("=== MLX completion ===")
    with create_mlx_client(profile_name="gemma4-e4b") as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        print(f"  Thinking: {resp.thinking[:100] if resp.thinking else '(none)'}")
        print(f"  Stop reason: {resp.stop_reason}")
        assert resp.content, "Empty response"
        assert resp.stop_reason == "end_turn"
        print("  PASS\n")


@run_if_api
def test_mlx_completion_with_system():
    if not HAS_MLX:
        print("=== MLX completion with system === SKIP (model not found)")
        return
    print("=== MLX completion with system ===")
    with create_mlx_client(profile_name="gemma4-e4b") as client:
        resp = client.completion(
            messages=[Message(role="user", content="What is your name?")],
            system="You are a helpful assistant named Bot.",
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_mlx_stream():
    if not HAS_MLX:
        print("=== MLX stream === SKIP (model not found)")
        return
    print("=== MLX stream ===")
    with create_mlx_client(profile_name="gemma4-e4b") as client:
        chunks = client.stream(
            messages=[Message(role="user", content="who is the president of USA?")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"  Content: {resp.content}")
        print(f"  Thinking: {resp.thinking[:100] if resp.thinking else '(none)'}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_mlx_no_thinking():
    if not HAS_MLX:
        print("=== MLX no thinking === SKIP (model not found)")
        return
    print("=== MLX no thinking ===")
    with create_mlx_client(profile_name="gemma4-e4b", enable_thinking=False) as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"  Content: {resp.content}")
        assert resp.content
        print("  PASS\n")


@run_if_api
def test_mlx_from_profiles():
    if not HAS_MLX:
        print("=== MLX from profiles === SKIP (model not found)")
        return
    print("=== MLX from profiles ===")
    llm = create_from_profiles()
    resp = llm.completion(
        messages=[Message(role="user", content="Say hi")],
        max_tokens=MAX_TOKENS,
        provider="gemma4-e4b",
    )
    print(f"  Content: {resp.content}")
    assert resp.content
    print("  PASS\n")


@run_if_api
def test_mlx_async_completion():
    if not HAS_MLX:
        print("=== MLX async completion === SKIP (model not found)")
        return
    print("=== MLX async completion ===")

    async def run():
        async with create_mlx_client(profile_name="gemma4-e4b") as client:
            resp = await client.async_completion(
                messages=[Message(role="user", content="Say hello.")],
                max_tokens=MAX_TOKENS,
            )
            print(f"  Content: {resp.content}")
            assert resp.content, "Empty response"

    asyncio.run(run())
    print("  PASS\n")


# ─── Runner ─────────────────────────────────────────────────────────────


def run_unit_tests():
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60 + "\n")

    test_classes = [
        TestModels(),
        TestBaseLLMClient(),
        TestCollectStream(),
        TestOpenAIParsing(),
        TestAnthropicParsing(),
        TestAzureToken(),
        TestMlxClient(),
        TestLLMClientManager(),
        TestClientInit(),
    ]

    total = 0
    passed = 0
    for cls in test_classes:
        name = cls.__class__.__name__
        methods = [m for m in dir(cls) if m.startswith("test_")]
        print(f"[{name}]")
        for method in methods:
            total += 1
            try:
                getattr(cls, method)()
                passed += 1
                print(f"  PASS: {method}")
            except Exception as e:
                print(f"  FAIL: {method} - {e}")
        print()

    print(f"Unit: {passed}/{total} passed\n")
    return passed == total


def run_integration_tests():
    print("=" * 60)
    print("INTEGRATION TESTS (API)")
    print("=" * 60 + "\n")

    tests = [
        test_anthropic_completion,
        test_anthropic_completion_with_system,
        test_anthropic_stream,
        test_anthropic_stream_events,
        test_openai_completion,
        test_openai_stream,
        test_llm_client_manager,
        test_llm_client_manager_stream,
        test_async_completion,
        test_async_stream,
        test_async_openai_completion,
        test_doubao_completion,
        test_doubao_stream,
        test_zhipu_completion,
        test_zhipu_stream,
        test_doubao_completion_with_system,
        test_doubao_stream_with_system,
        test_kimi_completion,
        test_kimi_stream,
        test_mlx_completion,
        test_mlx_completion_with_system,
        test_mlx_stream,
        test_mlx_no_thinking,
        test_mlx_from_profiles,
        test_mlx_async_completion,
    ]

    total = len(tests)
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__} - {e}\n")

    print(f"Integration: {passed}/{total} passed\n")
    return passed == total


if __name__ == "__main__":
    unit_ok = run_unit_tests()
    integ_ok = run_integration_tests()
    if unit_ok and integ_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
