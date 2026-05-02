"""Test MiniMax model via both Anthropic-compatible and OpenAI-compatible APIs."""

import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

from llm_client import LLMClient, AnthropicClient, OpenAIClient, Message

MODEL = os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.7-highspeed")
# MiniMax thinking blocks consume tokens; need generous limit for text to appear
MAX_TOKENS = 1024


def test_anthropic_completion():
    print("=== Anthropic completion ===")
    with AnthropicClient(
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        auth_mode="bearer",
        model=MODEL,
    ) as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"Content: {resp.content}")
        print(f"Stop reason: {resp.stop_reason}")
        print(f"Usage: {resp.usage}")
        assert resp.content, "Empty response"
        print("PASS\n")


def test_anthropic_stream():
    print("=== Anthropic stream ===")
    with AnthropicClient(
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        auth_mode="bearer",
        model=MODEL,
    ) as client:
        chunks = client.stream(
            messages=[Message(role="user", content="Count from 1 to 5.")],
            max_tokens=MAX_TOKENS,
        )
        resp = client.collect_stream(chunks)
        print(f"Content: {resp.content}")
        assert resp.content, "Empty response"
        print("PASS\n")


def test_openai_completion():
    print("=== OpenAI completion ===")
    with OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=MODEL,
    ) as client:
        resp = client.completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens=MAX_TOKENS,
        )
        print(f"Content: {resp.content}")
        print(f"Stop reason: {resp.stop_reason}")
        print(f"Usage: {resp.usage}")
        assert resp.content, "Empty response"
        print("PASS\n")


def test_llm_client():
    print("=== LLMClient (manager) ===")
    ant = AnthropicClient(
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        auth_mode="bearer",
        model=MODEL,
    )
    oai = OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=MODEL,
    )
    with LLMClient() as mgr:
        mgr.add_client("anthropic", ant, default=True)
        mgr.add_client("openai", oai)

        resp1 = mgr.completion(
            messages=[Message(role="user", content="Say hi")],
            max_tokens=MAX_TOKENS,
        )
        print(f"[default/anthropic] {resp1.content}")

        resp2 = mgr.completion(
            messages=[Message(role="user", content="Say hi")],
            max_tokens=MAX_TOKENS,
            provider="openai",
        )
        print(f"[openai] {resp2.content}")
        assert resp1.content and resp2.content
        print("PASS\n")


def test_async_completion():
    # Run in subprocess to avoid httpx sync/async event loop interference
    result = subprocess.run(
        [sys.executable, "-c", f"""
import os, asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
from llm_client import AnthropicClient, Message

async def test():
    async with AnthropicClient(
        api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        auth_mode="bearer",
        model={MODEL!r},
    ) as client:
        resp = await client.async_completion(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            max_tokens={MAX_TOKENS},
        )
        print(f"Content: {{resp.content}}")
        assert resp.content, "Empty response"
        print("PASS")

asyncio.run(test())
"""],
        capture_output=True,
        text=True,
    )
    print("=== Async Anthropic completion ===")
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Async test failed")
    if result.stderr:
        print(result.stderr)


if __name__ == "__main__":
    test_anthropic_completion()
    test_anthropic_stream()
    test_openai_completion()
    test_llm_client()
    test_async_completion()
    print("ALL TESTS PASSED")
