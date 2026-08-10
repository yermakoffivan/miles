"""Single-process FastAPI adapter for the session server.

Thin layer: converts each HTTP request to primitive inputs, calls
``SessionCore``. All session/TITO logic lives in ``core``.
"""

import json
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse
from starlette.responses import Response

from miles.rollout.session.anthropic import codec as anthropic_codec
from miles.rollout.session.core import JSON_MEDIA_TYPE, SessionCore, _render_json
from miles.rollout.session.errors import SessionError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.chat_template_utils.message_matcher_hub import (
    SessionMessageMatcherError,
    resolve_session_message_matcher,
)
from miles.utils.processing_utils import load_tokenizer

try:
    from sglang.srt.parser.template_detection import detect_inline_system_support
except ImportError:  # older sglang lineages keep it under managers/
    from sglang.srt.managers.template_detection import detect_inline_system_support

logger = logging.getLogger(__name__)

# End-to-end metadata kept on Anthropic error responses; everything else from
# the upstream response is dropped so framing headers describe the new body.
_ANTHROPIC_ERROR_HEADER_ALLOWLIST = ("www-authenticate", "retry-after", "x-request-id")
_ANTHROPIC_ERROR_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")


def _anthropic_wire_json(model) -> bytes:
    return _render_json(model.model_dump(mode="json", exclude_none=True, by_alias=True))


def _anthropic_error_response(status_code: int, body: bytes, headers: dict | None = None) -> Response:
    envelope = anthropic_codec.to_anthropic_error(status_code, body)
    kept_headers = {
        k: v
        for k, v in (headers or {}).items()
        if k.lower() in _ANTHROPIC_ERROR_HEADER_ALLOWLIST or k.lower().startswith(_ANTHROPIC_ERROR_HEADER_PREFIXES)
    }
    return Response(
        content=_anthropic_wire_json(envelope),
        status_code=status_code,
        headers=kept_headers,
        media_type=JSON_MEDIA_TYPE,
    )


def _anthropic_sse_body(events) -> bytes:
    return b"".join(
        f"event: {event.type}\ndata: ".encode() + _anthropic_wire_json(event) + b"\n\n" for event in events
    )


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)

    message_matcher_selector = getattr(args, "session_message_matcher", "strict")
    message_matcher = resolve_session_message_matcher(message_matcher_selector)
    logger.info("[session] Using message matcher selector=%r callable=%r", message_matcher_selector, message_matcher)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
    )

    use_v2 = getattr(args, "use_session_server", None) == "v2"
    if use_v2:
        from miles.rollout.session.v2.core import SessionCoreV2
        from miles.rollout.session.v2.session_state import SessionRegistryV2

        registry = SessionRegistryV2(args, tokenizer, tito_tokenizer=tito_tokenizer, message_matcher=message_matcher)
        core = SessionCoreV2(backend, registry, args, session_server_instance_id)
    else:
        registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer, message_matcher=message_matcher)
        core = SessionCore(backend, registry, args, session_server_instance_id)

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.exception_handler(SessionMessageMatcherError)
    async def session_message_matcher_error_handler(request: Request, exc: SessionMessageMatcherError):
        return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/health")
    async def health():
        return await core.health()

    @app.post("/sessions")
    async def create_session():
        return await core.create_session()

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return await core.get_session(session_id)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        return await core.delete_session(session_id)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        return await core.chat_completions(
            session_id,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )

    # Immutable launch-profile policy for the Anthropic codec, fixed at setup
    # like the frozen SGLang serving layer did in its constructor.
    anthropic_context = anthropic_codec.AnthropicRequestContext(
        merge_inline_system=not detect_inline_system_support(getattr(tokenizer, "chat_template", None))
    )

    @app.post("/sessions/{session_id}/v1/messages")
    async def anthropic_messages(request: Request, session_id: str):
        """Anthropic Messages wire wrapper over ``core.chat_completions``.

        The codec converts the wire formats; session/TITO/commit semantics,
        the canonical OpenAI ``SessionRecord``, and the backend path
        (``/v1/chat/completions``) are exactly the OpenAI route's. Registered
        before the catch-all ``session_proxy`` (Starlette matches in
        registration order).
        """
        body = await request.body()
        try:
            anthropic_request = anthropic_codec.parse_anthropic_request(body)
            openai_request = anthropic_codec.to_openai_request(anthropic_request, context=anthropic_context)
            # Force the core call non-streaming so it returns one complete
            # OpenAI JSON body; the client's stream intent is honored below
            # as eagerly materialized fake SSE.
            openai_request.stream = False
            openai_request.stream_options = None
            # exclude_unset keeps the body to Anthropic-derived fields so the
            # canonical record matches an equivalent client-written OpenAI
            # request instead of embedding vendored-model defaults.
            openai_body = _render_json(
                openai_request.model_dump(mode="json", exclude_none=True, exclude_unset=True, by_alias=True)
            )
        except ValueError as exc:
            # AnthropicRequestError plus non-encodable request values that
            # _render_json rejects (e.g. NaN sampling params): every request
            # construction failure is a 400 invalid_request_error.
            return _anthropic_error_response(400, _render_json({"error": str(exc)}))

        anthropic_stream = bool(anthropic_request.stream)

        try:
            core_response = await core.chat_completions(
                session_id,
                method=request.method,
                query=request.url.query,
                headers=dict(request.headers),
                body=openai_body,
            )
        except SessionError as exc:
            return _anthropic_error_response(exc.status_code, _render_json({"error": str(exc)}))
        except SessionMessageMatcherError as exc:
            return _anthropic_error_response(500, _render_json({"error": str(exc)}))
        except Exception:
            # Frozen behavior: unexpected processing failures still wear the
            # Anthropic error envelope (scrubbed generic 500) instead of the
            # framework's text/plain page. CancelledError still propagates.
            logger.exception("Anthropic chat processing failed for session %s", session_id)
            return _anthropic_error_response(500, b"")

        if core_response.status_code != 200:
            return _anthropic_error_response(
                core_response.status_code, core_response.body, dict(core_response.headers)
            )

        try:
            openai_response = ChatCompletionResponse.model_validate_json(core_response.body)
            if anthropic_stream:
                events = anthropic_codec.to_anthropic_fake_sse_events(
                    openai_response,
                    model=anthropic_request.model,
                    id_factory=anthropic_codec.anthropic_message_id,
                )
                return Response(
                    content=_anthropic_sse_body(events),
                    status_code=200,
                    headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
                    media_type="text/event-stream",
                )
            envelope = anthropic_codec.to_anthropic_response(
                openai_response, id_factory=anthropic_codec.anthropic_message_id
            )
            return Response(content=_anthropic_wire_json(envelope), status_code=200, media_type=JSON_MEDIA_TYPE)
        except Exception:
            # Post-commit conversion failure — the accepted first-version
            # boundary: core state is kept, the client gets a JSON 500,
            # never a partial SSE body. (CancelledError still propagates.)
            logger.exception("Anthropic response conversion failed for session %s", session_id)
            return _anthropic_error_response(500, b"")

    @app.post("/sessions/{session_id}/samples")
    async def collect_samples(request: Request, session_id: str):
        # Starlette matches routes in registration order; keep this before session_proxy.
        # Parse here so malformed input is not reported as an assembly error (422).
        body = await request.body()
        params = json.loads(body) if body else {}
        if use_v2:
            return await core.collect_samples(
                session_id, max_seq_len=params.get("max_seq_len"), agent_metadata=params.get("metadata")
            )
        return await core.collect_samples(session_id, max_seq_len=params.get("max_seq_len"))

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        return await core.proxy(
            session_id,
            path,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )
