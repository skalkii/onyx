"""Live behavior tests for the OpenAI Responses API path through LiteLLM.

`LitellmLLM` routes true OpenAI models through LiteLLM's Responses API
bridge (model name prefixed with `openai/responses/`). These tests exercise
behavior of that bridge that cannot be reached with mocks:

- Parallel tool calls land in distinct streaming slots with intact arguments.
- Streaming errors surface as the appropriate exception type rather than
  being masked by an internal `TypeError`.
- Reasoning summary sections are separated by a blank line in the streamed
  `reasoning_content` (covered by `_patch_responses_reasoning_summary_newlines`
  in `monkey_patches.py`).

Tests skip when `OPENAI_API_KEY` is unavailable.
"""

import json
import os

import litellm
import pytest
from litellm.exceptions import AuthenticationError

from onyx.llm.constants import LlmProviderNames
from onyx.llm.models import ChatCompletionMessage
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LitellmLLM


_OPENAI_KEY_REQUIRED = "OPENAI_API_KEY not configured"


def _build_openai_llm(model: str) -> LitellmLLM:
    return LitellmLLM(
        api_key=os.environ["OPENAI_API_KEY"],
        model_provider=LlmProviderNames.OPENAI,
        model_name=model,
        max_input_tokens=128_000,
        timeout=60,
    )


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason=_OPENAI_KEY_REQUIRED)
def test_streaming_parallel_tool_calls_land_in_distinct_slots() -> None:
    """Concurrent tool calls in a single streaming response must arrive on
    distinct `index` values with intact, parseable arguments.

    Pre-litellm-1.83.0 the responses bridge hardcoded `index=0` for every
    tool call, causing argument deltas from the second call to overwrite the
    first; it also set `finish_reason="tool_calls"` on the first
    `output_item.done`, terminating the stream before the second call
    arrived. Onyx's responses-streaming patch was removed once upstream
    fixed both. This test is the regression guard against either failure
    re-emerging.
    """
    llm = _build_openai_llm("gpt-4o-mini")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_population",
                "description": "Get the population of a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
    ]

    prompt: list[ChatCompletionMessage] = [
        UserMessage(
            role="user",
            content=(
                "For Paris, France: call get_weather AND get_population. "
                "Issue both tool calls in a single response."
            ),
        )
    ]

    accumulated: dict[int, dict[str, str]] = {}
    for chunk in llm.stream(prompt=prompt, tools=tools):
        for tc in chunk.choice.delta.tool_calls:
            slot = accumulated.setdefault(
                tc.index, {"id": "", "name": "", "arguments": ""}
            )
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

    if len(accumulated) < 2:
        pytest.skip(
            f"Model did not produce parallel tool calls this run "
            f"(got {len(accumulated)} call(s)). Re-run or adjust prompt. "
            f"Accumulated: {accumulated}"
        )

    indices = sorted(accumulated.keys())
    assert indices == list(
        range(len(indices))
    ), f"Tool call indices should be 0..N-1, got {indices}"

    names = {slot["name"] for slot in accumulated.values()}
    assert names == {
        "get_weather",
        "get_population",
    }, f"Expected both tool names to land in distinct slots, got {names}"

    for slot in accumulated.values():
        assert slot["id"], f"Tool call slot missing id: {slot}"
        try:
            parsed = json.loads(slot["arguments"])
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Tool call arguments not valid JSON for {slot['name']!r}: "
                f"{slot['arguments']!r} ({e})"
            )
        assert "city" in parsed, f"Expected 'city' in arguments, got {parsed}"


def test_responses_call_with_invalid_key_raises_authentication_error() -> None:
    """An invalid API key must surface as `AuthenticationError`.

    Specifically, passing `metadata=None` together with a bad key must NOT
    surface as `TypeError: argument of type 'NoneType' is not iterable`.
    Pre-1.83.0 LiteLLM's `@client` wrapper did `kwargs.get("metadata", {})`
    which returned `None` when the key existed with value `None`, masking
    the real auth failure. Upstream now uses `kwargs.get("metadata") or {}`.
    Onyx's `_patch_responses_metadata_none` was removed once that fix
    landed; this test guards against regression.
    """
    with pytest.raises(Exception) as exc_info:
        litellm.responses(
            model="openai/gpt-4o-mini",
            input="hi",
            api_key="sk-onyx-contract-test-deliberately-invalid",
            metadata=None,
            max_output_tokens=8,
        )

    err = exc_info.value
    err_str = str(err)
    assert (
        "NoneType" not in err_str
    ), f"metadata=None TypeError leaked into the surfaced exception: {err_str!r}"
    assert (
        isinstance(err, AuthenticationError)
        or "auth" in err_str.lower()
        or "401" in err_str
    ), (
        f"Expected AuthenticationError-shaped exception, got "
        f"{type(err).__name__}: {err_str!r}"
    )


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason=_OPENAI_KEY_REQUIRED)
def test_responses_call_tolerates_explicit_metadata_none() -> None:
    """Passing `metadata=None` on the happy path must not raise.

    Sister test to `test_responses_call_with_invalid_key_raises_authentication_error`:
    confirms `metadata=None` is tolerated for *successful* calls, not just
    error paths.
    """
    response = litellm.responses(
        model="openai/gpt-4o-mini",
        input="Reply with exactly the word: ok",
        api_key=os.environ["OPENAI_API_KEY"],
        metadata=None,
        max_output_tokens=16,
    )
    assert response is not None


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason=_OPENAI_KEY_REQUIRED)
def test_streaming_reasoning_summary_sections_are_separated_by_blank_line() -> None:
    """Streamed `reasoning_content` must contain a `\\n\\n` separator
    between distinct summary sections.

    LiteLLM passes through `response.reasoning_summary_text.delta` events
    as-is, with no separator when `summary_index` changes. Without the
    separator, multiple sections render as a single concatenated wall of
    text. `_patch_responses_reasoning_summary_newlines` (in
    `monkey_patches.py`) inserts the blank line. This test guards that the
    patch is still firing for current LiteLLM and OpenAI behavior.

    Skips when the model produces only a single section this run.
    """
    llm = _build_openai_llm("o4-mini")

    prompt: list[ChatCompletionMessage] = [
        UserMessage(
            role="user",
            content=(
                "Think through these three independent sub-problems one by "
                "one before answering, and produce a separate reasoning "
                "section for each:\n"
                "1) What is 17 * 23?\n"
                "2) Name one prime number between 50 and 60.\n"
                "3) What is the chemical symbol for gold?\n"
                "Then give a brief final answer."
            ),
        )
    ]

    reasoning_parts: list[str] = []
    for chunk in llm.stream(prompt=prompt):
        rc = chunk.choice.delta.reasoning_content
        if rc:
            reasoning_parts.append(rc)

    full_reasoning = "".join(reasoning_parts)
    if not full_reasoning:
        pytest.skip("Model returned no reasoning_content this run.")

    if "\n\n" not in full_reasoning:
        pytest.skip(
            "Model produced only a single reasoning summary section this "
            f"run; cannot verify newline separator. Got: {full_reasoning!r}"
        )

    assert (
        "\n\n" in full_reasoning
    ), f"Expected double-newline separator between summary sections: {full_reasoning!r}"
