"""Execution-level differential: frozen SGLang Anthropic serving vs Miles codec.

Runs the *upstream* conversion code from the sglang source tree on this
interpreter's path against `miles.rollout.session.anthropic.codec` on identical
inputs and requires byte-equal dumps. This is the 1:1 tripwire the hand-derived
goldens in test_anthropic_codec.py cannot give: if the codec ever drifts from
the pinned upstream semantics, this fails.

Gate: the differential asserts only when the sglang tree on sys.path carries
exactly the `SOURCE_SHA` blobs recorded in
miles/rollout/session/anthropic/UPSTREAM.md. Anything else — an older lineage
without the anthropic entrypoint, or a moved sglang-miles head — skips with the
observed hashes so upstream drift routes to the UPSTREAM.md sync procedure
instead of failing unrelated PRs. In stage-b-cpu the sglang checkout defaults
to the sglang-miles head, so the assertion is live while the pin holds.

Scope: request/response/error conversion only. The fake-SSE builder has no
runtime-free upstream equivalent (upstream streams from the engine), and
thinking is rejected by the Miles launch profile, so neither is compared here.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=[])

import hashlib
import importlib
import importlib.util
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.rollout.session.anthropic import codec

# sha256 of the SOURCE_SHA blobs, copied from UPSTREAM.md.
_UPSTREAM_SHA256 = {
    "protocol": "ca5ee76a2f4fa02fbeb4f2bd43cab2914a36137a0f0c622aeefed81cef16b2dd",
    "serving": "18349a6c475dbbf1749b54fca251d078a3fe2b687ff9c71e528f76aead59da31",
}

_FIXED_UUID = uuid.UUID(int=0x1234)
_FIXED_MSG_ID = f"msg_{_FIXED_UUID.hex}"

# Permissive launch profile: the differential compares the frozen conversion
# semantics of every typed feature, so no gate may reject the fixtures.
_ALL_FEATURES = {
    "allow_images": True,
    "allow_output_config": True,
    "allow_beta_fields": True,
    "allow_tool_references": True,
    "allow_search_results": True,
    "allow_server_tools": True,
}


def _load_upstream():
    """Import the upstream anthropic modules, or return a skip reason."""
    module_names = {
        "protocol": "sglang.srt.entrypoints.anthropic.protocol",
        "serving": "sglang.srt.entrypoints.anthropic.serving",
    }
    origins = {}
    for key, name in module_names.items():
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError) as e:
            return None, f"{name} not importable: {e}"
        if spec is None or not spec.origin:
            return None, f"{name} not present in this sglang tree"
        origins[key] = spec.origin
    observed = {key: hashlib.sha256(Path(origin).read_bytes()).hexdigest() for key, origin in origins.items()}
    if observed != _UPSTREAM_SHA256:
        return None, (
            f"sglang anthropic sources drifted from the UPSTREAM.md pin (observed {observed}); "
            "re-vendor via the UPSTREAM.md sync procedure before trusting this differential"
        )
    try:
        modules = {key: importlib.import_module(name) for key, name in module_names.items()}
    except Exception as e:  # missing runtime deps despite matching sources
        return None, f"frozen upstream modules failed to import: {e}"
    return SimpleNamespace(**modules), None


@pytest.fixture(scope="module")
def upstream():
    modules, reason = _load_upstream()
    if modules is None:
        pytest.skip(f"upstream differential skipped: {reason}")
    return modules


def _upstream_serving(upstream, merge_inline_system: bool):
    """Frozen AnthropicServing without its runtime: only the inline-system
    policy the constructor would have probed is injected."""
    instance = upstream.serving.AnthropicServing.__new__(upstream.serving.AnthropicServing)
    instance._merge_inline_system = merge_inline_system
    return instance


def _payload(messages, **extra) -> dict:
    return {"model": "claude-diff", "max_tokens": 64, "messages": messages, **extra}


_TOOLS = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {}}}]

_REQUEST_CASES = {
    "text_system_sampling": _payload(
        [{"role": "user", "content": "hello"}],
        system="be brief",
        temperature=0.5,
        top_k=20,
        top_p=0.9,
        stop_sequences=["END", "STOP"],
    ),
    "system_blocks_and_inline_system": _payload(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": [{"type": "text", "text": " inline "}]},
        ],
        system=[{"type": "text", "text": "s1"}, {"type": "text", "text": "s2"}],
    ),
    "stream_with_options": _payload([{"role": "user", "content": "hi"}], stream=True),
    "tool_roundtrip": _payload(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "pre"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny", "is_error": True},
                    {"type": "text", "text": "post"},
                ],
            },
        ],
        tools=_TOOLS,
        tool_choice={"type": "auto"},
    ),
    "tool_result_variants": _payload(
        [
            {"role": "user", "content": [{"type": "tool_result", "id": "legacy_id", "content": None}]},
            {"role": "assistant", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "inner"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t3",
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}},
                            {"type": "tool_reference", "tool_name": "deferred_fn"},
                            {"type": "search_result", "title": "T", "source": "https://s"},
                        ],
                    }
                ],
            },
        ]
    ),
    "tool_choice_required": _payload([{"role": "user", "content": "q"}], tools=_TOOLS, tool_choice={"type": "any"}),
    "tool_choice_named": _payload(
        [{"role": "user", "content": "q"}], tools=_TOOLS, tool_choice={"type": "tool", "name": "get_weather"}
    ),
    "tool_choice_none": _payload([{"role": "user", "content": "q"}], tools=_TOOLS, tool_choice={"type": "none"}),
    "empty_placeholders": _payload(
        [
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": []},
            {"role": "user", "content": "q"},
        ]
    ),
    "images": _payload(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "eA=="}},
                    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                ],
            }
        ]
    ),
    "search_result_block": _payload(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "search_result",
                        "title": "T",
                        "source": "https://s",
                        "content": [{"type": "text", "text": "C"}],
                    }
                ],
            }
        ]
    ),
    "output_config_and_betas": _payload(
        [{"role": "user", "content": "q"}],
        output_config={"effort": "xhigh", "task_budget": {"type": "tokens", "total": 1000}},
        betas=["thinking-2025-08-04"],
    ),
    "server_tools_skipped": _payload(
        [{"role": "user", "content": "q"}],
        tools=[{"type": "web_search_20250305", "name": "web_search"}, *_TOOLS],
    ),
    "unknown_extra_keys": _payload([{"role": "user", "content": "hi", "unknown_msg_key": 1}], unknown_top_key="x"),
}


class TestRequestDifferential:
    @pytest.mark.parametrize("merge_inline_system", [True, False])
    @pytest.mark.parametrize("case", sorted(_REQUEST_CASES))
    def test_request_conversion_matches_upstream(self, upstream, case, merge_inline_system):
        payload = _REQUEST_CASES[case]
        upstream_request = upstream.protocol.AnthropicMessagesRequest.model_validate(payload)
        miles_request = codec.parse_anthropic_request(json.dumps(payload).encode())
        context = codec.AnthropicRequestContext(merge_inline_system=merge_inline_system, **_ALL_FEATURES)

        with patch("uuid.uuid4", return_value=_FIXED_UUID):
            upstream_chat = _upstream_serving(upstream, merge_inline_system)._convert_to_chat_completion_request(
                upstream_request
            )
            miles_chat = codec.to_openai_request(miles_request, context=context)

        assert miles_chat.model_dump() == upstream_chat.model_dump()

    @pytest.mark.parametrize(
        "payload",
        [
            _payload([{"role": "user", "content": "q"}], tools=_TOOLS, tool_choice={"type": "tool", "name": "nope"}),
            _payload([{"role": "user", "content": "q"}], tool_choice={"type": "any"}),
        ],
        ids=["named_tool_missing", "required_without_tools"],
    )
    def test_conversion_failures_match_upstream(self, upstream, payload):
        upstream_request = upstream.protocol.AnthropicMessagesRequest.model_validate(payload)
        miles_request = codec.parse_anthropic_request(json.dumps(payload).encode())
        context = codec.AnthropicRequestContext(merge_inline_system=True, **_ALL_FEATURES)

        with pytest.raises(ValueError) as upstream_error:
            _upstream_serving(upstream, True)._convert_to_chat_completion_request(upstream_request)
        with pytest.raises(codec.AnthropicRequestError) as miles_error:
            codec.to_openai_request(miles_request, context=context)
        assert str(miles_error.value) == str(upstream_error.value)


def _chat_response(message: dict, finish_reason: str = "stop", usage: dict | None = None):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "served-alias",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


_RESPONSE_CASES = {
    "text": ({"content": "hi"}, "stop", None),
    "reasoning_and_tools": (
        {
            "content": "text",
            "reasoning_content": "thought",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
                {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "not json"}},
            ],
        },
        "tool_calls",
        None,
    ),
    "empty_content": ({"content": ""}, "stop", None),
    "length_stop": ({"content": "x"}, "length", None),
    "unmapped_finish_reason": ({"content": "x"}, "content_filter", None),
    "cached_usage": (
        {"content": "hi"},
        "stop",
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    ),
}


class TestResponseDifferential:
    @pytest.mark.parametrize("case", sorted(_RESPONSE_CASES))
    def test_response_conversion_matches_upstream(self, upstream, case):
        message, finish_reason, usage = _RESPONSE_CASES[case]
        response = _chat_response(message, finish_reason, usage)

        with patch("uuid.uuid4", return_value=_FIXED_UUID):
            upstream_result = _upstream_serving(upstream, True)._convert_response(response)
        miles_result = codec.to_anthropic_response(response, id_factory=lambda: _FIXED_MSG_ID)

        assert miles_result.model_dump() == upstream_result.model_dump()


_ERROR_BODIES = [
    b'{"error": "boom"}',
    b'{"error": {"message": "m", "type": "custom_type"}}',
    b'{"message": "top-level"}',
    b"<html>gateway</html>",
    b"",
]


class TestErrorDifferential:
    # Statuses present in the frozen serving.py ERROR_TYPE_MAP; 413/422 are
    # deliberate Miles composite additions from the frozen http_server.py and
    # have no serving.py equivalent (asserted separately below).
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 429, 500, 502, 503, 504])
    @pytest.mark.parametrize("body_index", range(len(_ERROR_BODIES)))
    def test_error_conversion_matches_upstream(self, upstream, status, body_index):
        body = _ERROR_BODIES[body_index]
        upstream_response = _upstream_serving(upstream, True)._convert_openai_error_response(
            SimpleNamespace(status_code=status, body=body)
        )
        assert upstream_response.status_code == status

        miles_envelope = codec.to_anthropic_error(status, body)
        assert miles_envelope.model_dump() == json.loads(bytes(upstream_response.body))

    def test_composite_statuses_follow_http_server_policy(self):
        assert codec.ERROR_TYPE_MAP[413] == "request_too_large"
        assert codec.ERROR_TYPE_MAP[422] == "invalid_request_error"
