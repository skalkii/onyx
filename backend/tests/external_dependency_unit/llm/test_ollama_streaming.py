"""Live behavior test for Ollama streaming through LiteLLM.

`_patch_ollama_chunk_parser` (in `monkey_patches.py`) ensures reasoning
tokens are routed to `Delta.reasoning_content` and visible answer tokens
land on `Delta.content`, for both native `thinking` field chunks and
legacy `<think>...</think>`-tagged content chunks. Unit tests in
`backend/tests/unit/onyx/llm/test_litellm_monkey_patches.py` cover the
state machine against crafted chunk dicts; this test exercises the same
path against Ollama Cloud's live wire format so we catch upstream
protocol drift.

Skips when `OLLAMA_API_KEY` is unavailable.
"""

import os

import pytest

from onyx.llm.constants import LlmProviderNames
from onyx.llm.models import ChatCompletionMessage
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM


_OLLAMA_KEY_REQUIRED = "OLLAMA_API_KEY not configured"

# gpt-oss:120b-cloud emits native `thinking` tokens and is included in the
# nightly Ollama Cloud matrix, so the credential already has access.
_THINKING_MODEL = "gpt-oss:120b-cloud"


@pytest.mark.skipif(not os.environ.get("OLLAMA_API_KEY"), reason=_OLLAMA_KEY_REQUIRED)
def test_streaming_separates_reasoning_content_from_visible_content() -> None:
    """A thinking-capable Ollama model must stream its chain-of-thought on
    `Delta.reasoning_content` and its final answer on `Delta.content`, with
    no cross-contamination between the two fields in any single chunk.

    Without `_patch_ollama_chunk_parser`, LiteLLM's Ollama transformer drops
    reasoning tokens onto `Delta.content`, so they render in the visible
    answer stream. This test guards against regression in either the patch
    or upstream Ollama chunk shape.
    """
    llm = LitellmLLM(
        api_key=os.environ["OLLAMA_API_KEY"],
        model_provider=LlmProviderNames.OLLAMA_CHAT,
        model_name=_THINKING_MODEL,
        api_base="https://ollama.com",
        max_input_tokens=8192,
        timeout=120,
    )

    prompt: list[ChatCompletionMessage] = [
        UserMessage(
            role="user",
            content=(
                "Think briefly about what 12 * 7 is, then respond with just "
                "the number."
            ),
        )
    ]

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in llm.stream(prompt=prompt):
        delta = chunk.choice.delta
        rc = delta.reasoning_content
        content = delta.content
        if rc:
            assert not content, (
                f"Chunk leaked visible content onto a reasoning delta: "
                f"reasoning_content={rc!r} content={content!r}"
            )
            reasoning_parts.append(rc)
        if content:
            content_parts.append(content)

    full_reasoning = "".join(reasoning_parts)
    full_content = "".join(content_parts)

    if not full_reasoning:
        pytest.skip(
            "Model returned no reasoning tokens this run; cannot verify "
            "reasoning/content separation."
        )

    assert full_content.strip(), (
        f"Model produced reasoning but no visible answer tokens. "
        f"reasoning={full_reasoning!r}"
    )
    assert (
        "<think>" not in full_content and "</think>" not in full_content
    ), f"Raw <think> tags leaked into visible content: {full_content!r}"
    assert (
        "<think>" not in full_reasoning and "</think>" not in full_reasoning
    ), f"Raw <think> tags leaked into reasoning_content: {full_reasoning!r}"
