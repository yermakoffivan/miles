"""Differential tests for the SGLang-derived Anthropic codec.

Goldens are derived by reading the frozen source pinned in
``miles/rollout/session/anthropic/UPSTREAM.md``; they freeze the conversion
semantics the codec must preserve. The on-image differential against the
deployed runtime remains a separate Phase 0 obligation.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

from miles.rollout.session.anthropic import codec, protocol
from miles.utils.chat_template_utils.message_matcher_hub import resolve_session_message_matcher, strict_message_matches

# sha256 of the verbatim upstream blob recorded in UPSTREAM.md.
_PROTOCOL_SHA256 = "ca5ee76a2f4fa02fbeb4f2bd43cab2914a36137a0f0c622aeefed81cef16b2dd"
_MARKER = b"# --- BEGIN VERBATIM UPSTREAM SOURCE ---"

_CTX = codec.AnthropicRequestContext(merge_inline_system=True)


def _convert(payload: dict, context: codec.AnthropicRequestContext = _CTX) -> dict:
    request = codec.parse_anthropic_request(json.dumps(payload).encode())
    openai_request = codec.to_openai_request(request, context=context)
    return openai_request.model_dump(mode="json", exclude_none=True, by_alias=True)


def _base(messages, **extra) -> dict:
    return {"model": "claude-test", "max_tokens": 64, "messages": messages, **extra}


def _response(message: dict, finish_reason: str = "stop", usage: dict | None = None) -> ChatCompletionResponse:
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


class TestProtocolSnapshot:
    def test_vendored_protocol_is_byte_identical_below_marker(self):
        lines = Path(protocol.__file__).read_bytes().split(b"\n")
        body = b"\n".join(lines[lines.index(_MARKER) + 1 :])
        assert hashlib.sha256(body).hexdigest() == _PROTOCOL_SHA256

    def test_unknown_extra_keys_keep_frozen_ignore_behavior(self):
        payload = _base([{"role": "user", "content": "hi", "unknown_msg_key": 1}], unknown_top_key="x")
        request = codec.parse_anthropic_request(json.dumps(payload).encode())
        assert request.model == "claude-test"
        assert not hasattr(request, "unknown_top_key")


class TestRequestConversion:
    def test_text_system_and_sampling_params(self):
        dump = _convert(
            _base(
                [{"role": "user", "content": "hello"}],
                system="be brief",
                temperature=0.5,
                top_k=20,
                top_p=0.9,
                stop_sequences=["END"],
            )
        )
        assert dump["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
        assert dump["model"] == "claude-test"
        assert dump["max_tokens"] == 64
        assert (dump["temperature"], dump["top_k"], dump["top_p"], dump["stop"]) == (0.5, 20, 0.9, ["END"])
        assert dump["stream"] is False
        assert "stream_options" not in dump

    def test_system_blocks_and_inline_system_merge(self):
        payload = _base(
            [
                {"role": "user", "content": "q"},
                {"role": "system", "content": [{"type": "text", "text": " inline "}]},
            ],
            system=[{"type": "text", "text": "s1"}, {"type": "text", "text": "s2"}],
        )
        merged = _convert(payload, codec.AnthropicRequestContext(merge_inline_system=True))
        assert merged["messages"][0] == {"role": "system", "content": "s1\ns2\ninline"}
        assert [m["role"] for m in merged["messages"]] == ["system", "user"]

        unmerged = _convert(payload, codec.AnthropicRequestContext(merge_inline_system=False))
        assert unmerged["messages"][0] == {"role": "system", "content": "s1\ns2"}
        assert [m["role"] for m in unmerged["messages"]] == ["system", "user", "system"]

    def test_stream_true_sets_stream_options(self):
        dump = _convert(_base([{"role": "user", "content": "hi"}], stream=True))
        assert dump["stream"] is True
        assert dump["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}

    def test_tool_use_and_tool_result_roundtrip(self):
        dump = _convert(
            _base(
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
                tools=[{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
            )
        )
        assert dump["messages"][1] == {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
        # Wire order preserved: user(pre) → tool → user(post); is_error is
        # dropped from the OpenAI tool message (frozen behavior).
        assert dump["messages"][2] == {"role": "user", "content": "pre"}
        assert dump["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}
        assert dump["messages"][4] == {"role": "user", "content": "post"}
        assert dump["tools"][0]["function"]["name"] == "get_weather"
        assert dump["tool_choice"] == "auto"

    def test_tool_result_variants(self):
        dump = _convert(
            _base(
                [
                    {"role": "user", "content": [{"type": "tool_result", "id": "legacy_id", "content": None}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "inner"}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t3",
                                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                            }
                        ],
                    },
                ]
            )
        )
        # None content → ""; legacy ``id`` used as tool_call_id fallback.
        assert dump["messages"][0] == {"role": "tool", "tool_call_id": "legacy_id", "content": ""}
        # Assistant-role tool_result folds into text (frozen behavior).
        assert dump["messages"][1] == {"role": "assistant", "content": "Tool result: inner"}
        # Multi-text list keeps the parts list.
        assert dump["messages"][2]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    def test_empty_text_and_empty_assistant_placeholders(self):
        dump = _convert(
            _base(
                [
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                    {"role": "assistant", "content": []},
                    {"role": "user", "content": "q"},
                ]
            )
        )
        assert dump["messages"][0] == {"role": "assistant", "content": ""}
        assert dump["messages"][1] == {"role": "assistant", "content": ""}

    def test_tool_choice_mappings(self):
        tools = [{"name": "f", "input_schema": {"type": "object"}}]
        msgs = [{"role": "user", "content": "q"}]
        assert _convert(_base(msgs, tools=tools, tool_choice={"type": "any"}))["tool_choice"] == "required"
        assert _convert(_base(msgs, tools=tools, tool_choice={"type": "none"}))["tool_choice"] == "none"
        named = _convert(_base(msgs, tools=tools, tool_choice={"type": "tool", "name": "f"}))["tool_choice"]
        assert named == {"type": "function", "function": {"name": "f"}}

        with pytest.raises(codec.AnthropicRequestError, match="not in the forwarded tools list"):
            _convert(_base(msgs, tools=tools, tool_choice={"type": "tool", "name": "missing"}))
        with pytest.raises(codec.AnthropicRequestError, match="requires at least one custom tool"):
            _convert(_base(msgs, tool_choice={"type": "any"}))

    def test_parse_failures_are_request_errors(self):
        with pytest.raises(codec.AnthropicRequestError, match="invalid JSON body"):
            codec.parse_anthropic_request(b"{not json")
        with pytest.raises(codec.AnthropicRequestError):
            codec.parse_anthropic_request(json.dumps({"model": "m", "messages": []}).encode())  # no max_tokens


class TestFeatureGates:
    def test_disabled_features_fail_closed(self):
        msgs = [{"role": "user", "content": "q"}]
        cases = [
            _base(msgs, thinking={"type": "enabled", "budget_tokens": 2048}),
            _base(msgs, output_config={"effort": "high"}),
            _base(msgs, betas=["b-1"]),
            _base([{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "data": "eA=="}}]}]),
            _base([{"role": "assistant", "content": [{"type": "thinking", "thinking": "t"}]}]),
            _base([{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]),
            _base([{"role": "user", "content": [{"type": "search_result", "title": "t"}]}]),
            _base(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [{"type": "tool_reference", "tool_name": "f"}],
                            }
                        ],
                    }
                ]
            ),
            _base(msgs, tools=[{"type": "web_search_20250305", "name": "web_search"}]),
        ]
        for payload in cases:
            with pytest.raises(codec.AnthropicRequestError):
                _convert(payload)

    def test_enabled_image_conversion(self):
        ctx = codec.AnthropicRequestContext(merge_inline_system=True, allow_images=True)
        dump = _convert(
            _base(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": "eA=="},
                            },
                            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                        ],
                    }
                ]
            ),
            ctx,
        )
        # The runtime's typed message model may add its own defaults (e.g.
        # ``detail``) when dumped; assert only the Anthropic-derived fields.
        parts = dump["messages"][0]["content"]
        assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
        assert parts[0]["text"] == "look"
        assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,eA=="
        assert parts[2]["image_url"]["url"] == "https://x/y.png"

    def test_enabled_search_result_flattening(self):
        ctx = codec.AnthropicRequestContext(merge_inline_system=True, allow_search_results=True)
        dump = _convert(
            _base(
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
            ctx,
        )
        assert dump["messages"][0] == {"role": "user", "content": "Title: T\nSource: https://s\nContent: C"}

    def test_enabled_tool_reference_translation(self):
        ctx = codec.AnthropicRequestContext(merge_inline_system=True, allow_tool_references=True)
        dump = _convert(
            _base(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": [{"type": "tool_reference", "tool_name": "deferred_fn"}],
                            }
                        ],
                    }
                ]
            ),
            ctx,
        )
        assert dump["messages"][0]["content"] == [{"type": "tool_reference", "name": "deferred_fn"}]

    def test_enabled_output_config_effort_mapping(self):
        ctx = codec.AnthropicRequestContext(merge_inline_system=True, allow_output_config=True)
        msgs = [{"role": "user", "content": "q"}]
        assert _convert(_base(msgs, output_config={"effort": "high"}), ctx)["reasoning_effort"] == "high"
        assert _convert(_base(msgs, output_config={"effort": "xhigh"}), ctx)["reasoning_effort"] == "max"

    def test_enabled_betas_accepted(self):
        ctx = codec.AnthropicRequestContext(merge_inline_system=True, allow_beta_fields=True)
        dump = _convert(_base([{"role": "user", "content": "q"}], betas=["thinking-2025-08-04"]), ctx)
        assert dump["messages"] == [{"role": "user", "content": "q"}]

    def test_enabled_server_tools_are_skipped_not_forwarded(self):
        ctx = codec.AnthropicRequestContext(merge_inline_system=True, allow_server_tools=True)
        dump = _convert(
            _base(
                [{"role": "user", "content": "q"}],
                tools=[
                    {"type": "web_search_20250305", "name": "web_search"},
                    {"name": "f", "input_schema": {"type": "object"}},
                ],
            ),
            ctx,
        )
        assert [t["function"]["name"] for t in dump["tools"]] == ["f"]


class TestResponseConversion:
    def test_text_response_with_usage(self):
        result = codec.to_anthropic_response(
            _response(
                {"content": "hi"},
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            ),
            id_factory=lambda: "msg_fixed",
        )
        dump = result.model_dump(mode="json", exclude_none=True, by_alias=True)
        assert dump == {
            "id": "msg_fixed",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "served-alias",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 6, "output_tokens": 5, "cache_read_input_tokens": 4},
        }

    def test_reasoning_and_tool_calls(self):
        result = codec.to_anthropic_response(
            _response(
                {
                    "content": "text",
                    "reasoning_content": "thought",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
                        {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "not json"}},
                    ],
                },
                finish_reason="tool_calls",
            ),
            id_factory=lambda: "msg_fixed",
        )
        types = [(b.type, getattr(b, "name", None)) for b in result.content]
        assert types == [("thinking", None), ("text", None), ("tool_use", "f"), ("tool_use", "g")]
        assert result.content[2].input == {"a": 1}
        assert result.content[3].input == {}  # invalid JSON → empty input (frozen)
        assert result.stop_reason == "tool_use"

    def test_empty_and_unmapped_cases(self):
        empty = codec.to_anthropic_response(_response({"content": ""}), id_factory=lambda: "m")
        assert [b.type for b in empty.content] == ["text"] and empty.content[0].text == ""

        assert (
            codec.to_anthropic_response(_response({"content": "x"}, "length"), id_factory=lambda: "m").stop_reason
            == "max_tokens"
        )
        assert (
            codec.to_anthropic_response(
                _response({"content": "x"}, "content_filter"), id_factory=lambda: "m"
            ).stop_reason
            == "end_turn"
        )

        no_choices = ChatCompletionResponse.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
        result = codec.to_anthropic_response(no_choices, id_factory=lambda: "m")
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 0 and result.usage.output_tokens == 0


class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected_type"),
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (408, "request_timeout_error"),
            (409, "api_error"),  # unlisted status falls through (design table)
            (413, "request_too_large"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (502, "api_error"),
            (503, "overloaded_error"),
            (504, "api_error"),
        ],
    )
    def test_composite_status_map(self, status, expected_type):
        envelope = codec.to_anthropic_error(status, b'{"error": "boom"}')
        assert envelope.type == "error"
        assert envelope.error.type == expected_type
        assert envelope.error.message == ("Internal server error" if status >= 500 else "boom")

    def test_upstream_type_honored_for_4xx_only(self):
        body = json.dumps({"error": {"message": "m", "type": "custom_type"}}).encode()
        assert codec.to_anthropic_error(400, body).error.type == "custom_type"
        assert codec.to_anthropic_error(500, body).error.type == "api_error"

    def test_non_json_and_empty_bodies(self):
        envelope = codec.to_anthropic_error(400, b"<html>gateway</html>")
        assert envelope.error.message == "<html>gateway</html>"
        assert codec.to_anthropic_error(400, b"").error.message == "Request failed"
        assert codec.to_anthropic_error(502, b"").error.message == "Internal server error"

    def test_4xx_scrub_strips_traceback_lines_and_truncates(self):
        upstream = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "/app/handler.py", line 3, in run',
                "real cause: " + "x" * 600,
            ]
        )
        message = codec.to_anthropic_error(400, json.dumps({"error": {"message": upstream}}).encode()).error.message
        assert "Traceback" not in message and 'File "/' not in message
        assert message.startswith("real cause: ")
        assert len(message) == 501 and message.endswith("…")


class TestFakeSse:
    def test_text_event_sequence_uses_request_model(self):
        events = codec.to_anthropic_fake_sse_events(
            _response({"content": "hello"}), model="claude-test", id_factory=lambda: "msg_fixed"
        )
        assert [e.type for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        start = events[0].message
        # Model comes from the Anthropic request, not the backend alias.
        assert start.model == "claude-test" and start.id == "msg_fixed" and start.content == []
        assert start.usage.input_tokens == 10 and start.usage.output_tokens == 0
        assert events[2].delta.text == "hello"
        assert events[4].delta.stop_reason == "end_turn"
        assert events[4].usage.input_tokens is None and events[4].usage.output_tokens == 5

    def test_multi_block_index_accounting(self):
        events = codec.to_anthropic_fake_sse_events(
            _response(
                {
                    "content": "txt",
                    "reasoning_content": "think",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
                        {"id": "c2", "type": "function", "function": {"name": "g", "arguments": ""}},
                    ],
                },
                finish_reason="tool_calls",
            ),
            model="claude-test",
            id_factory=lambda: "m",
        )
        starts = [e for e in events if e.type == "content_block_start"]
        assert [(e.index, e.content_block.type) for e in starts] == [
            (0, "thinking"),
            (1, "text"),
            (2, "tool_use"),
            (3, "tool_use"),
        ]
        deltas = [e for e in events if e.type == "content_block_delta"]
        # Zero-argument tool call emits no input_json_delta (frozen stream).
        assert [(e.index, e.delta.type) for e in deltas] == [
            (0, "thinking_delta"),
            (1, "text_delta"),
            (2, "input_json_delta"),
        ]
        assert deltas[2].delta.partial_json == '{"a": 1}'
        stops = [e.index for e in events if e.type == "content_block_stop"]
        assert stops == [0, 1, 2, 3]
        assert events[-2].delta.stop_reason == "tool_use"
        assert events[-1].type == "message_stop"

    def test_unmapped_finish_reason_and_cached_usage(self):
        events = codec.to_anthropic_fake_sse_events(
            _response(
                {"content": "x"},
                finish_reason="content_filter",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            ),
            model="claude-test",
            id_factory=lambda: "m",
        )
        start_usage = events[0].message.usage
        assert (start_usage.input_tokens, start_usage.cache_read_input_tokens, start_usage.output_tokens) == (6, 4, 0)
        message_delta = next(e for e in events if e.type == "message_delta")
        assert message_delta.delta.stop_reason == "end_turn"  # content_filter is unmapped (frozen)
        assert message_delta.usage.cache_read_input_tokens is None and message_delta.usage.output_tokens == 5


class TestMatcherGate:
    """Design matcher gate: object → string re-serialization may change tool
    argument spelling; ``strict`` must reject it, ``loose_tool_call`` must
    accept it, so tool launch profiles default to ``loose_tool_call``."""

    def _replayed_assistant(self, arguments_object: dict) -> dict:
        dump = _convert(
            _base(
                [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call1", "name": "f", "input": arguments_object}],
                    },
                ]
            )
        )
        return dump["messages"][-1]

    def test_strict_rejects_and_loose_accepts_respelled_arguments(self):
        stored = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
            ],
        }
        replayed = self._replayed_assistant({"a": 1})
        assert replayed["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'  # json.dumps spelling
        assert strict_message_matches(stored, replayed) is False
        loose = resolve_session_message_matcher("loose_tool_call")
        assert loose(stored, replayed) is True

    def test_strict_accepts_identical_spelling(self):
        stored = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "index": 0, "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        }
        replayed = self._replayed_assistant({"a": 1})
        assert strict_message_matches(stored, replayed) is True


class TestImportHygiene:
    def test_codec_import_loads_no_serving_runtime(self):
        # Starlette/torch are pulled in by the sglang package root itself and
        # are out of the codec's control; the codec must not load the OpenAI
        # serving runtime, tokenizer manager, engine, or FastAPI.
        code = (
            "import sys\n"
            "import miles.rollout.session.anthropic.codec\n"
            "banned = ('fastapi', 'sglang.srt.entrypoints.openai.serving_chat',\n"
            "          'sglang.srt.managers.tokenizer_manager', 'sglang.srt.entrypoints.engine')\n"
            "loaded = [m for m in sys.modules for b in banned if m == b or m.startswith(b + '.')]\n"
            "assert not loaded, loaded\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=300)

    def test_codec_source_has_no_http_imports(self):
        source = Path(codec.__file__).read_text()
        for banned in ("fastapi", "starlette"):
            assert banned not in source
