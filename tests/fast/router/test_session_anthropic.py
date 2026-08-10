"""Integration tests for ``POST /sessions/{session_id}/v1/messages``.

The Anthropic route must reuse the OpenAI session path end to end: canonical
OpenAI ``SessionRecord``s, TITO ``input_ids``, matcher semantics, and the
same commit/skip decisions — clients only ever see Anthropic wire shapes.
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

import miles.rollout.session.core as core_module
import miles.rollout.session.v2.core as v2_core_module
from miles.rollout.session import sessions as sessions_module
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# Two-key arguments: the qwen25 parser re-serializes them in this key order,
# so a replay whose input object uses the REVERSED key order re-serializes to
# a different spelling — the matcher-gate scenario from the design doc.
_TOOL_CALL_TEXT = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}\n</tool_call>'
_TOOLS = [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {}}}]


def _process_fn(prompt: str) -> ProcessResult:
    if "RAISE" in prompt:
        raise RuntimeError("mock backend failure")
    if "sunny" in prompt:
        return ProcessResult(text="final-answer", finish_reason="stop")
    if "use the weather tool" in prompt:
        return ProcessResult(text=_TOOL_CALL_TEXT, finish_reason="stop")
    return ProcessResult(text="anthropic-echo", finish_reason="stop")


@contextmanager
def _anthropic_env(extra_args: dict | None = None, *, latency: float = 0.0):
    # The mock backend already emits choice.meta_info with
    # output_token_logprobs/completion_tokens in the session-server format.
    with with_mock_server(process_fn=_process_fn, latency=latency) as backend:
        args = SimpleNamespace(
            miles_router_timeout=30,
            hf_checkpoint="Qwen/Qwen3-0.6B",
            chat_template_path=None,
            apply_chat_template_kwargs={"enable_thinking": False},
            tito_model="default",
            sglang_speculative_algorithm=None,
            trajectory_manager="linear_trajectory",
            session_server_instance_id=uuid.uuid4().hex,
            save_debug_trajectory_data=None,
            **(extra_args or {}),
        )
        server_obj = SessionServer(args, backend_url=backend.url)
        port = find_available_port(31000)
        server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}


@pytest.fixture(scope="module", params=["v1", "v2"])
def anthropic_env(request):
    with _anthropic_env(_V2_ARGS if request.param == "v2" else None) as env:
        yield SimpleNamespace(version=request.param, **vars(env))


@pytest.fixture(scope="module", params=["v1", "v2"])
def anthropic_env_loose(request):
    extra = {"session_message_matcher": "loose_tool_call", **(_V2_ARGS if request.param == "v2" else {})}
    with _anthropic_env(extra) as env:
        yield SimpleNamespace(version=request.param, **vars(env))


def _create_session(url: str) -> str:
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post_messages(url: str, session_id: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{session_id}/v1/messages", json=payload, timeout=30.0)


def _records(url: str, session_id: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _payload(messages, **extra) -> dict:
    return {"model": "claude-test", "max_tokens": 64, "messages": messages, **extra}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_line, data_line = block.split("\n", 1)
        assert event_line.startswith("event: ") and data_line.startswith("data: ")
        events.append((event_line[len("event: ") :], json.loads(data_line[len("data: ") :])))
    return events


class TestAnthropicRoute:
    def test_non_stream_text_creates_canonical_openai_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], system="sys")
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["type"] == "message" and body["role"] == "assistant"
        assert body["id"].startswith("msg_")
        # Non-stream JSON keeps the backend response model (frozen behavior).
        assert body["model"] == "mock-model"
        assert body["content"] == [{"type": "text", "text": "anthropic-echo"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0

        records = _records(anthropic_env.url, session_id)
        assert len(records) == 1
        record = records[0]
        assert record["path"] == "/v1/chat/completions"
        assert record["request"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert isinstance(record["request"]["input_ids"], list) and record["request"]["input_ids"]
        assert record["response"]["object"] == "chat.completion"

    def test_stream_returns_eager_fake_sse(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(
            anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], stream=True)
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        message_start = events[0][1]["message"]
        # The stream's message model is the original Anthropic request model,
        # never the backend's served-model alias.
        assert message_start["model"] == "claude-test"
        assert message_start["usage"]["output_tokens"] == 0 and message_start["usage"]["input_tokens"] > 0
        assert events[2][1]["delta"] == {"type": "text_delta", "text": "anthropic-echo"}
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"
        assert events[4][1]["usage"]["output_tokens"] > 0
        assert len(_records(anthropic_env.url, session_id)) == 1

    def test_parity_with_equivalent_openai_request(self, anthropic_env):
        anthropic_session = _create_session(anthropic_env.url)
        openai_session = _create_session(anthropic_env.url)

        anthropic_resp = _post_messages(
            anthropic_env.url, anthropic_session, _payload([{"role": "user", "content": "hello"}], system="sys")
        )
        assert anthropic_resp.status_code == 200

        openai_resp = requests.post(
            f"{anthropic_env.url}/sessions/{openai_session}/v1/chat/completions",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            },
            timeout=30.0,
        )
        assert openai_resp.status_code == 200

        anthropic_record = _records(anthropic_env.url, anthropic_session)[0]
        openai_record = _records(anthropic_env.url, openai_session)[0]
        assert anthropic_record["request"]["messages"] == openai_record["request"]["messages"]
        assert anthropic_record["request"]["input_ids"] == openai_record["request"]["input_ids"]
        assert (
            anthropic_record["response"]["choices"][0]["message"] == openai_record["response"]["choices"][0]["message"]
        )

    def test_parity_with_equivalent_openai_request_with_tools(self, anthropic_env):
        """Tool-bearing parity pins the canonical tools spelling: the chat
        template renders tools in dict key order, so byte-identical TITO
        input_ids require the OpenAI client to write the codec's field order
        (function.description before function.name, no extra defaults)."""
        anthropic_session = _create_session(anthropic_env.url)
        openai_session = _create_session(anthropic_env.url)

        anthropic_resp = _post_messages(
            anthropic_env.url, anthropic_session, _payload([{"role": "user", "content": "hi"}], tools=_TOOLS)
        )
        assert anthropic_resp.status_code == 200

        openai_resp = requests.post(
            f"{anthropic_env.url}/sessions/{openai_session}/v1/chat/completions",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "description": "weather",
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
            timeout=30.0,
        )
        assert openai_resp.status_code == 200

        anthropic_record = _records(anthropic_env.url, anthropic_session)[0]
        openai_record = _records(anthropic_env.url, openai_session)[0]
        assert anthropic_record["request"]["messages"] == openai_record["request"]["messages"]
        assert anthropic_record["request"]["tools"] == openai_record["request"]["tools"]
        assert anthropic_record["request"]["input_ids"] == openai_record["request"]["input_ids"]

    def test_unexpected_core_exception_wears_anthropic_envelope(self, anthropic_env):
        """Unexpected (non-SessionError) processing failures keep the frozen
        wire behavior: a scrubbed generic Anthropic api_error envelope, not
        the framework's text/plain 500 page."""
        session_id = _create_session(anthropic_env.url)
        # patch.object on pre-imported modules: a string-target patch would
        # import v2.core lazily INSIDE the first patch's window and bake the
        # first mock into its from-import binding, leaking it after restore.
        with (
            patch.object(core_module, "extract_completion", side_effect=RuntimeError("boom")),
            patch.object(v2_core_module, "extract_completion", side_effect=RuntimeError("boom")),
        ):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        assert _records(anthropic_env.url, session_id) == []

    def test_nan_sampling_param_maps_to_invalid_request_error(self, anthropic_env):
        """json.loads admits the non-standard NaN literal and the wire models
        accept it, but ``_render_json`` (allow_nan=False) rejects it — that
        failure must stay a 400 request error, never a plain-text 500."""
        session_id = _create_session(anthropic_env.url)
        raw = (
            b'{"model": "claude-test", "max_tokens": 64, "temperature": NaN,'
            b' "messages": [{"role": "user", "content": "x"}]}'
        )
        resp = requests.post(
            f"{anthropic_env.url}/sessions/{session_id}/v1/messages",
            data=raw,
            headers={"content-type": "application/json"},
            timeout=10.0,
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert _records(anthropic_env.url, session_id) == []

    def test_error_envelopes_and_no_record_on_validation_failure(self, anthropic_env):
        payload = _payload([{"role": "user", "content": "hello"}])
        resp = _post_messages(anthropic_env.url, "nonexistent", payload)
        assert resp.status_code == 404
        assert resp.json() == {
            "type": "error",
            "error": {"type": "not_found_error", "message": "session not found: session_id=nonexistent"},
        }

        session_id = _create_session(anthropic_env.url)
        cases = [
            b"{not json",
            json.dumps({"model": "claude-test", "messages": [{"role": "user", "content": "x"}]}).encode(),
            json.dumps(_payload([{"role": "user", "content": "x"}], thinking={"type": "disabled"})).encode(),
            json.dumps(
                _payload([{"role": "user", "content": [{"type": "image", "source": {"url": "https://x"}}]}])
            ).encode(),
        ]
        for raw in cases:
            resp = requests.post(
                f"{anthropic_env.url}/sessions/{session_id}/v1/messages",
                data=raw,
                headers={"content-type": "application/json"},
                timeout=10.0,
            )
            assert resp.status_code == 400, raw
            assert resp.json()["error"]["type"] == "invalid_request_error"
        # Anthropic-side validation failures never reach the core.
        assert _records(anthropic_env.url, session_id) == []

    def test_backend_failure_maps_to_anthropic_error_without_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "RAISE"}]))
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "error" and body["error"]["type"] == "api_error"
        assert body["error"]["message"] == "Internal server error"
        assert _records(anthropic_env.url, session_id) == []

    def test_post_commit_conversion_failure_returns_500_and_keeps_record(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        with patch.object(sessions_module.anthropic_codec, "to_anthropic_response", side_effect=RuntimeError("boom")):
            resp = _post_messages(anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}]))
        assert resp.status_code == 500
        assert resp.json() == {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        # The accepted first-version boundary: core already committed.
        assert len(_records(anthropic_env.url, session_id)) == 1

    def test_sse_build_failure_returns_json_500_not_partial_stream(self, anthropic_env):
        session_id = _create_session(anthropic_env.url)
        with patch.object(
            sessions_module.anthropic_codec, "to_anthropic_fake_sse_events", side_effect=RuntimeError("boom")
        ):
            resp = _post_messages(
                anthropic_env.url, session_id, _payload([{"role": "user", "content": "hello"}], stream=True)
            )
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["type"] == "api_error"
        # Same post-commit boundary as the non-stream twin: the record stays.
        assert len(_records(anthropic_env.url, session_id)) == 1


def _tool_turn1(url: str, session_id: str) -> dict:
    """Run the tool-eliciting first turn; returns the tool_use block."""
    turn1 = _post_messages(
        url, session_id, _payload([{"role": "user", "content": "please use the weather tool"}], tools=_TOOLS)
    )
    assert turn1.status_code == 200
    body = turn1.json()
    assert body["stop_reason"] == "tool_use"
    tool_use = next(block for block in body["content"] if block["type"] == "tool_use")
    assert tool_use["name"] == "get_weather" and tool_use["input"] == {"city": "Paris", "unit": "C"}
    return tool_use


def _tool_turn2_payload(tool_use: dict) -> dict:
    """Replay with the tool_use input keys REVERSED: json.dumps preserves the
    object's key order, so the re-serialized arguments spelling differs from
    the stored one — accepted only by ``loose_tool_call``."""
    respelled_input = dict(reversed(list(tool_use["input"].items())))
    return _payload(
        [
            {"role": "user", "content": "please use the weather tool"},
            {"role": "assistant", "content": [{**tool_use, "input": respelled_input}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "sunny"}]},
        ],
        tools=_TOOLS,
    )


class TestAnthropicToolFlow:
    def test_multi_turn_tool_flow_reuses_stored_prefix(self, anthropic_env_loose):
        url = anthropic_env_loose.url
        session_id = _create_session(url)
        tool_use = _tool_turn1(url, session_id)

        turn2 = _post_messages(url, session_id, _tool_turn2_payload(tool_use))
        assert turn2.status_code == 200
        assert turn2.json()["content"] == [{"type": "text", "text": "final-answer"}]

        records = _records(url, session_id)
        assert len(records) == 2
        first_ids = records[0]["request"]["input_ids"]
        second_ids = records[1]["request"]["input_ids"]
        # The stored TITO prefix is reused: turn 2's prompt extends turn 1's.
        assert second_ids[: len(first_ids)] == first_ids
        replayed_assistant = next(m for m in records[1]["request"]["messages"] if m["role"] == "assistant")
        assert replayed_assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(replayed_assistant["tool_calls"][0]["function"]["arguments"]) == {
            "city": "Paris",
            "unit": "C",
        }

    def test_strict_matcher_rejects_respelled_tool_arguments(self, anthropic_env):
        """Design matcher gate, end to end: under the default ``strict``
        matcher the re-serialized tool arguments diverge from the stored
        assistant message, so v1 rolls back to the empty checkpoint —
        discarding turn 1's record — and re-renders from scratch instead of
        reusing the stored TITO prefix (contrast with the loose test above)."""
        if anthropic_env.version != "v1":
            pytest.skip("v2 branches to a new lineage instead of rolling back; the loose fixture covers v2")
        url = anthropic_env.url
        session_id = _create_session(url)
        tool_use = _tool_turn1(url, session_id)
        first_ids = _records(url, session_id)[0]["request"]["input_ids"]

        turn2 = _post_messages(url, session_id, _tool_turn2_payload(tool_use))
        assert turn2.status_code == 200

        records = _records(url, session_id)
        assert len(records) == 1
        assert [m["role"] for m in records[0]["request"]["messages"]] == ["user", "assistant", "tool"]
        second_ids = records[0]["request"]["input_ids"]
        assert second_ids[: len(first_ids)] != first_ids


class TestAnthropicCloseRace:
    def test_delete_during_inflight_chat_skips_update_gracefully(self):
        """Split-lock close race through the Anthropic route: DELETE lands
        while the chat is mid-proxy; Phase 3 sees closing=True and skips the
        commit, but the client still gets a well-formed Anthropic response —
        the same 200-with-skip outcome as the OpenAI route."""
        with _anthropic_env(latency=0.35) as env:
            session_id = _create_session(env.url)
            payload = _payload([{"role": "user", "content": "hello"}])

            with ThreadPoolExecutor(max_workers=2) as pool:
                inflight = pool.submit(_post_messages, env.url, session_id, payload)

                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if env.backend.request_log:
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("in-flight request did not reach backend in time")

                delete_resp = requests.delete(f"{env.url}/sessions/{session_id}", timeout=30.0)
                inflight_resp = inflight.result(timeout=30.0)

            assert delete_resp.status_code == 204
            assert inflight_resp.status_code == 200
            body = inflight_resp.json()
            assert body["type"] == "message"
            assert body["content"] == [{"type": "text", "text": "anthropic-echo"}]

            post_delete = _post_messages(env.url, session_id, payload)
            assert post_delete.status_code == 404
            assert post_delete.json()["error"]["type"] == "not_found_error"
