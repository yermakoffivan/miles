"""SGLang-derived Anthropic ↔ OpenAI wire codec: pure conversion, no runtime.

Derived from sgl-project/sglang @ cb05a44f35a7c9e27e46d74112cc841ca674ef43
``python/sglang/srt/entrypoints/anthropic/serving.py`` (Apache-2.0); UPSTREAM.md
carries the symbol-level adaptation ledger. Only the protocol responsibilities
were kept — request/response/error conversion plus the Miles eager fake-SSE
event builder. Runtime orchestration (engine validation, real streaming, abort
tasks, count_tokens, tokenizer-manager access) was removed, and the two
instance dependencies of request conversion became explicit inputs: the
inline-system merge policy lives on ``AnthropicRequestContext`` and message IDs
come from an injected ``id_factory``.

Raw bytes enter only at the request/error parsing boundary; every other seam
consumes and returns DTOs. Wire serialization, HTTP status, headers, and eager
SSE materialization are owned by the route (``sessions.py``).
"""

import json
import logging
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)

from miles.rollout.session.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessageEndDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    is_server_tool,
)

logger = logging.getLogger(__name__)

# Frozen serving.py: OpenAI finish reasons → Anthropic stop reasons. Unmapped
# values (``content_filter``, ``abort``) fall through to ``end_turn`` with a
# WARNING at the call site.
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

# Miles composite policy (see UPSTREAM.md): 400/401/403/404/413/422/429/5xx
# from the frozen http_server.py ``/v1/messages`` exception handler; 408 from
# the frozen serving.py ``ERROR_TYPE_MAP``. Unlisted statuses → ``api_error``.
ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "request_timeout_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
}


class AnthropicRequestError(ValueError):
    """Anthropic request cannot be parsed, validated, or converted (HTTP 400)."""


@dataclass(frozen=True)
class AnthropicRequestContext:
    """Immutable launch-profile policy for request conversion.

    Known-but-disabled typed features are rejected with
    ``AnthropicRequestError`` before conversion. Validation walks only the
    typed request/content/tool models; ``tool_use.input``, custom-tool
    ``input_schema`` and ``metadata`` remain arbitrary-JSON boundaries.
    Thinking has no data-only ``ReasoningPolicy`` in this version and is
    always rejected.
    """

    merge_inline_system: bool
    allow_images: bool = False
    allow_output_config: bool = False
    allow_beta_fields: bool = False
    allow_tool_references: bool = False
    allow_search_results: bool = False
    allow_server_tools: bool = False


def anthropic_message_id() -> str:
    """Production message-ID factory, same format as the frozen source."""
    return f"msg_{uuid.uuid4().hex}"


def _cached_prompt_tokens(usage) -> int:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", 0) or 0


def _anthropic_input_tokens(usage) -> int:
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    cached = _cached_prompt_tokens(usage)
    if cached > prompt:
        # Upstream telemetry bug: cached cannot exceed the prompt it caches.
        # Clamping silently here would hide the discrepancy from billing
        # dashboards, so make it visible at WARNING level.
        logger.warning(
            "Cached tokens (%d) exceed prompt tokens (%d); clamping input_tokens to 0. "
            "This usually indicates an upstream telemetry bug.",
            cached,
            prompt,
        )
    return max(prompt - cached, 0)


def _anthropic_usage_from_openai(
    usage, *, include_input: bool, include_output: bool, force_zero_output: bool = False
) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage(
            input_tokens=0 if include_input else None,
            output_tokens=0 if include_output else None,
        )

    usage_fields: dict[str, int] = {}
    cached_tokens = _cached_prompt_tokens(usage)
    if include_input:
        usage_fields["input_tokens"] = _anthropic_input_tokens(usage)
        if cached_tokens:
            usage_fields["cache_read_input_tokens"] = cached_tokens
    if include_output:
        usage_fields["output_tokens"] = 0 if force_zero_output else (getattr(usage, "completion_tokens", 0) or 0)
    return AnthropicUsage(**usage_fields)


def _extract_system_text(content: str | list[AnthropicContentBlock]) -> str | None:
    """Flatten a system message's content to a trimmed string, or ``None``."""
    if isinstance(content, str):
        return content.strip() or None
    texts = []
    for block in content:
        if isinstance(block, BaseModel) and getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
        else:
            continue
        text = (text or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else None


def _scrub_error_message(message: str, status_code: int) -> str:
    """Return a safe outward-facing error message.

    5xx is always generic — never echo upstream ``str(e)`` payloads, which may
    contain stack frames, file paths, or PII. 4xx keeps the original message
    (truncated and with obvious traceback lines stripped) so callers see the
    real validation failure.
    """
    if status_code >= 500:
        return "Internal server error"
    if not message:
        return "Request failed"
    safe_lines = [ln for ln in message.splitlines() if not ln.startswith("Traceback") and 'File "/' not in ln]
    cleaned = "\n".join(safe_lines).strip()
    if len(cleaned) > 500:
        cleaned = cleaned[:500] + "…"
    return cleaned or "Request failed"


def parse_anthropic_request(body: bytes) -> AnthropicMessagesRequest:
    """Parse the raw request body; failures map to HTTP 400."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise AnthropicRequestError(f"invalid JSON body: {e}") from e
    try:
        return AnthropicMessagesRequest.model_validate(payload)
    except ValidationError as e:
        raise AnthropicRequestError(str(e)) from e


def _iter_typed_content_blocks(request: AnthropicMessagesRequest) -> Iterator[Any]:
    """Yield every typed content block reachable from the request: top-level
    ``system`` blocks, message blocks, and one level of ``tool_result`` nested
    content (the depth the frozen conversion reads)."""
    if request.system is not None and not isinstance(request.system, str):
        yield from request.system
    for msg in request.messages:
        if isinstance(msg.content, str):
            continue
        for block in msg.content:
            yield block
            if getattr(block, "type", None) == "tool_result" and isinstance(block.content, list):
                yield from block.content


def _validate_known_features(request: AnthropicMessagesRequest, context: AnthropicRequestContext) -> None:
    """Reject known-but-disabled typed features (fail closed, HTTP 400).

    Only typed models are walked; arbitrary-JSON boundaries are not recursed.
    Unknown extra keys inside known models keep the frozen Pydantic ignore
    behavior and are not checked here.
    """
    if request.thinking is not None:
        raise AnthropicRequestError("thinking is not supported by this endpoint")
    if request.output_config is not None and not context.allow_output_config:
        raise AnthropicRequestError("output_config is not enabled for this deployment")
    if request.betas and not context.allow_beta_fields:
        raise AnthropicRequestError("betas is not enabled for this deployment")
    if request.tools and not context.allow_server_tools:
        for tool in request.tools:
            if is_server_tool(tool):
                raise AnthropicRequestError(
                    f"server tool {tool.name!r} (type={tool.type!r}) is not enabled for this deployment"
                )
    for block in _iter_typed_content_blocks(request):
        block_type = getattr(block, "type", None)
        if block_type in ("thinking", "redacted_thinking"):
            raise AnthropicRequestError("thinking content blocks are not supported by this endpoint")
        if block_type == "image" and not context.allow_images:
            raise AnthropicRequestError("image content blocks are not enabled for this deployment")
        if block_type == "tool_reference" and not context.allow_tool_references:
            raise AnthropicRequestError("tool_reference content blocks are not enabled for this deployment")
        if block_type == "search_result" and not context.allow_search_results:
            raise AnthropicRequestError("search_result content blocks are not enabled for this deployment")


def _convert_anthropic_image_source_to_openai_part(source: Any) -> dict | None:
    # Source may arrive as a Pydantic model (typed ImageBlock.source)
    # or as a raw dict when parsed from a nested tool_result payload.
    if isinstance(source, BaseModel):
        source = source.model_dump(exclude_none=True)
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}

    url = source.get("url")
    if url:
        return {"type": "image_url", "image_url": {"url": url}}

    return None


def _text_from_search_result(item: dict[str, Any]) -> str:
    search_parts = []
    title = item.get("title")
    if title:
        search_parts.append(f"Title: {title}")

    source = item.get("source")
    if isinstance(source, dict):
        source_text = source.get("url") or source.get("text")
        if source_text:
            search_parts.append(f"Source: {source_text}")
    elif source:
        search_parts.append(f"Source: {source}")

    content = item.get("content")
    content_parts = []
    if isinstance(content, str):
        content_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                content_parts.append(part["text"])
    if content_parts:
        search_parts.append("Content: " + "\n".join(content_parts))

    return "\n".join(search_parts)


def _convert_tool_result_content(content: Any) -> tuple[str | list[dict], str]:
    if isinstance(content, list):
        tool_content_parts = []
        tool_text_parts = []

        for raw_item in content:
            # Items may be typed Pydantic blocks (after request validation)
            # or raw dicts (from legacy callers). Coerce to dict so the
            # existing key-based logic still works.
            if isinstance(raw_item, BaseModel):
                item = raw_item.model_dump(exclude_none=True)
            elif isinstance(raw_item, dict):
                item = raw_item
            else:
                continue

            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                if text:
                    tool_text_parts.append(text)
                    tool_content_parts.append({"type": "text", "text": text})
            elif item_type == "image":
                image_part = _convert_anthropic_image_source_to_openai_part(item.get("source"))
                if image_part is not None:
                    tool_content_parts.append(image_part)
            elif item_type == "tool_reference":
                # Anthropic uses `tool_name`; the SGLang chat template
                # matches on `name`. Translate at the boundary.
                ref_name = item.get("tool_name") or item.get("name")
                if ref_name:
                    tool_content_parts.append({"type": "tool_reference", "name": ref_name})
            elif item_type == "search_result":
                search_text = _text_from_search_result(item)
                if search_text:
                    tool_text_parts.append(search_text)
                    tool_content_parts.append({"type": "text", "text": search_text})

        tool_text = "\n".join(tool_text_parts)
        if len(tool_content_parts) == 1 and tool_content_parts[0]["type"] == "text":
            return tool_content_parts[0]["text"], tool_text
        if tool_content_parts:
            return tool_content_parts, tool_text
        return "", tool_text

    tool_text = str(content) if content else ""
    return tool_text, tool_text


def _convert_messages(request: AnthropicMessagesRequest, merge_inline_system: bool) -> list[dict]:
    """Frozen message conversion: leading merged system block, then per-message
    block conversion. ``thinking``/``redacted_thinking`` never reach this point
    (rejected by validation); the skip below is kept as a structural guard."""
    openai_messages: list[dict] = []

    system_parts: list[str] = []
    if request.system:
        if isinstance(request.system, str):
            if request.system.strip():
                system_parts.append(request.system)
        else:
            for block in request.system:
                if block.type == "text" and block.text:
                    system_parts.append(block.text)

    if merge_inline_system:
        for msg in request.messages:
            if msg.role != "system":
                continue
            text = _extract_system_text(msg.content)
            if text:
                system_parts.append(text)

    if system_parts:
        openai_messages.append({"role": "system", "content": "\n".join(system_parts)})

    def _emit_user_message(parts: list[dict]) -> None:
        """Append accumulated parts as a user message, then clear them.

        Used to flush content collected BEFORE a tool_result so the wire order
        stays user(pre) → tool → user(post)."""
        if not parts:
            return
        if len(parts) == 1 and parts[0]["type"] == "text":
            openai_messages.append({"role": "user", "content": parts[0]["text"]})
        else:
            openai_messages.append({"role": "user", "content": list(parts)})
        parts.clear()

    for msg in request.messages:
        if msg.role == "system" and merge_inline_system:
            continue
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
            continue

        # Complex content with blocks
        openai_msg = {"role": msg.role}
        content_parts: list[dict] = []
        tool_calls: list[dict] = []

        for block in msg.content:
            if block.type in ("thinking", "redacted_thinking"):
                continue

            # ``is not None`` (not truthy) so an empty-string text block
            # still produces a placeholder text part — without it, an
            # assistant turn whose only content is "" vanishes and
            # subsequent user→user pairs trip strict chat templates.
            if block.type == "text" and block.text is not None:
                content_parts.append({"type": "text", "text": block.text})

            elif block.type == "image" and block.source:
                image_part = _convert_anthropic_image_source_to_openai_part(block.source)
                if image_part is not None:
                    content_parts.append(image_part)

            elif block.type == "search_result":
                search_text = _text_from_search_result(block.model_dump())
                if search_text:
                    content_parts.append({"type": "text", "text": search_text})

            elif block.type == "tool_use":
                tool_call = {
                    "id": block.id or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": block.name or "", "arguments": json.dumps(block.input or {})},
                }
                tool_calls.append(tool_call)

            elif block.type == "tool_result":
                tool_content, tool_text = _convert_tool_result_content(block.content)

                # Use tool_use_id (per spec) with fallback to id
                tool_call_id = block.tool_use_id or block.id or ""

                # Tool results from user become separate tool messages.
                # Flush any pending text/image first so the wire order is
                # preserved (a tool_result that arrived AFTER a text block
                # must come AFTER that text in OpenAI form too).
                if msg.role == "user":
                    _emit_user_message(content_parts)
                    openai_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_content})
                else:
                    content_parts.append({"type": "text", "text": f"Tool result: {tool_text}"})

        # Attach tool calls to assistant messages
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls

        # Attach content
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts
            openai_messages.append(openai_msg)
        elif tool_calls:
            openai_messages.append(openai_msg)
        elif msg.role == "user":
            # User turn that was entirely tool_results — the tool messages
            # were already emitted above, nothing left.
            continue
        else:
            # Assistant turn with no content and no tool_calls: emit an
            # empty-string placeholder so strict templates still see a
            # valid role-alternation sequence.
            openai_msg["content"] = ""
            openai_messages.append(openai_msg)

    return openai_messages


def _convert_tools(request: AnthropicMessagesRequest, chat_request: ChatCompletionRequest) -> None:
    """Frozen tool conversion. Deferred tools stay in the list with
    defer_loading=True; server tools are skipped with a visible log (they
    reach here only when ``allow_server_tools`` admitted them)."""
    if not request.tools:
        return
    converted_tools = []
    for tool in request.tools:
        if is_server_tool(tool):
            # Anthropic server-side tools have no client-side input_schema
            # because Anthropic provides the implementation. We can't forward
            # them to the OpenAI tools array (which requires a schema).
            logger.info(
                "Skipping built-in Anthropic server tool %r (type=%r): "
                "no native support in the OpenAI-compatible backend",
                tool.name,
                tool.type,
            )
            continue

        # Custom tools always have a validated input_schema.
        converted_tools.append(
            Tool(
                type="function",
                defer_loading=tool.defer_loading,
                function={
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            )
        )

    if converted_tools:
        chat_request.tools = converted_tools


def _apply_tool_choice(request: AnthropicMessagesRequest, chat_request: ChatCompletionRequest) -> None:
    """Frozen tool-choice conversion. ``any``/``tool`` are hard requirements;
    if every requested tool was a skipped server tool, silently downgrading to
    "no tool" would deceive the caller, so raise an explicit 400."""
    if request.tool_choice is not None:
        tc_type = request.tool_choice.type
        if tc_type == "none":
            chat_request.tool_choice = "none"
        elif chat_request.tools:
            if tc_type == "auto":
                chat_request.tool_choice = "auto"
            elif tc_type == "any":
                chat_request.tool_choice = "required"
            elif tc_type == "tool":
                tool_name = request.tool_choice.name
                if not any(t.function.name == tool_name for t in chat_request.tools):
                    raise ValueError(
                        f"tool_choice references tool {tool_name!r} but it is not in the forwarded tools list "
                        f"(server-side Anthropic tools cannot be selected)"
                    )
                chat_request.tool_choice = ToolChoice(type="function", function=ToolChoiceFuncName(name=tool_name))
        elif tc_type in ("any", "tool"):
            raise ValueError(
                f"tool_choice={tc_type!r} requires at least one custom tool; all supplied tools were "
                f"server-side Anthropic built-ins which the OpenAI-compatible backend cannot invoke"
            )
    elif chat_request.tools:
        chat_request.tool_choice = "auto"


def to_openai_request(request: AnthropicMessagesRequest, *, context: AnthropicRequestContext) -> ChatCompletionRequest:
    """Convert an Anthropic Messages request to an OpenAI ChatCompletion
    request under the launch-profile ``context``. Any validation or conversion
    failure raises ``AnthropicRequestError`` (frozen behavior: HTTP 400)."""
    _validate_known_features(request, context)
    try:
        openai_messages = _convert_messages(request, context.merge_inline_system)

        request_data = {
            "messages": openai_messages,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "stream": request.stream or False,
        }
        if request.temperature is not None:
            request_data["temperature"] = request.temperature
        if request.top_p is not None:
            request_data["top_p"] = request.top_p
        if request.top_k is not None:
            request_data["top_k"] = request.top_k
        if request.stop_sequences is not None:
            request_data["stop"] = request.stop_sequences

        # Enable usage in stream so we can report it
        if request.stream:
            request_data["stream_options"] = StreamOptions(include_usage=True, continuous_usage_stats=True)

        chat_request = ChatCompletionRequest(**request_data)

        # Claude 4.7 ``output_config``: map ``effort`` onto the OpenAI
        # ``reasoning_effort`` knob. ``xhigh`` collapses to ``max`` because
        # the OpenAI Literal does not include the Anthropic-only ``xhigh``.
        # ``task_budget`` is a soft hint (``max_tokens`` stays the hard cap).
        if request.output_config is not None:
            oc = request.output_config
            if oc.effort is not None:
                chat_request.reasoning_effort = "max" if oc.effort == "xhigh" else oc.effort
            if oc.task_budget is not None:
                logger.info(
                    "Anthropic output_config.task_budget hint: %d %s", oc.task_budget.total, oc.task_budget.type
                )

        # ``betas`` is the Anthropic SDK's opt-in feature list. The backend
        # has no equivalent beta system; accept-and-log so requests don't 400.
        if request.betas:
            logger.info("Anthropic request opted into betas %s — no-op locally", request.betas)

        _convert_tools(request, chat_request)
        _apply_tool_choice(request, chat_request)
    except AnthropicRequestError:
        raise
    except Exception as e:
        # Frozen behavior: every conversion failure is a 400
        # invalid_request_error, logged with its traceback server-side.
        logger.exception("Error converting Anthropic request: %s", e)
        raise AnthropicRequestError(str(e)) from e

    return chat_request


def to_anthropic_response(
    response: ChatCompletionResponse, *, id_factory: Callable[[], str]
) -> AnthropicMessagesResponse:
    """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages response."""
    if not response.choices:
        return AnthropicMessagesResponse(
            id=id_factory(),
            content=[TextBlock(text="")],
            model=response.model,
            stop_reason="end_turn",
            usage=_anthropic_usage_from_openai(None, include_input=True, include_output=True),
        )

    choice = response.choices[0]
    content: list[AnthropicContentBlock] = []

    # Add reasoning content as a thinking block. signature is omitted
    # entirely when the backend doesn't provide one — empty strings would
    # fail downstream Anthropic signature verifiers.
    if choice.message.reasoning_content:
        content.append(ThinkingBlock(thinking=choice.message.reasoning_content))

    # Add text content
    if choice.message.content:
        content.append(TextBlock(text=choice.message.content))

    # Add tool calls
    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            raw_args = tool_call.function.arguments
            try:
                tool_input = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                # Surface invalid tool arguments so an empty-dict tool call
                # is never indistinguishable from a real one when something
                # downstream goes wrong.
                logger.warning(
                    "Tool %r emitted invalid JSON arguments: %r — defaulting to empty input",
                    tool_call.function.name,
                    (raw_args or "")[:200],
                )
                tool_input = {}

            content.append(ToolUseBlock(id=tool_call.id, name=tool_call.function.name, input=tool_input))

    # Map stop reason
    finish_reason = choice.finish_reason or "stop"
    if finish_reason not in STOP_REASON_MAP:
        logger.warning("Unmapped OpenAI finish_reason %r; defaulting to end_turn", finish_reason)
    stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

    # Anthropic requires ``content`` to contain at least one block. Empty
    # string completions would otherwise ship ``content=[]`` and break
    # strict SDK parsers.
    if not content:
        content.append(TextBlock(text=""))

    return AnthropicMessagesResponse(
        id=id_factory(),
        content=content,
        model=response.model,
        stop_reason=stop_reason,
        usage=_anthropic_usage_from_openai(response.usage, include_input=True, include_output=True),
    )


def to_anthropic_error(status_code: int, body: bytes) -> AnthropicErrorResponse:
    """Map an upstream/core error status and body to an Anthropic error
    envelope. 4xx keeps a sanitized upstream message and honors an upstream
    ``error.type``; 5xx is always the generic scrubbed message."""
    body = body or b""
    error_type = ERROR_TYPE_MAP.get(status_code, "api_error")

    upstream_message: str | None = None
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Non-JSON body (HTML gateway error, plain text, ...). Use a bounded
        # slice of the raw body so the client still has a useful hint.
        upstream_message = body.decode("utf-8", errors="replace")[:500]
    else:
        if isinstance(payload, dict):
            error_payload = payload.get("error", payload)
            if isinstance(error_payload, dict):
                upstream_message = error_payload.get("message") or payload.get("message")
                # Honor the upstream error.type only for 4xx; 5xx is
                # normalized by the scrub below.
                if status_code < 500:
                    upstream_type = error_payload.get("type")
                    if isinstance(upstream_type, str) and upstream_type:
                        error_type = upstream_type
            elif isinstance(error_payload, str):
                upstream_message = error_payload
            elif isinstance(payload.get("message"), str):
                upstream_message = payload["message"]

    message = _scrub_error_message(upstream_message or "", status_code)
    return AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message))


def to_anthropic_fake_sse_events(
    response: ChatCompletionResponse, *, model: str, id_factory: Callable[[], str]
) -> tuple[AnthropicStreamEvent, ...]:
    """Eagerly synthesize the Anthropic SSE event sequence from one complete
    OpenAI response.

    Event schema, block ordering (thinking → text → tool_use), index
    accounting, usage split (input on ``message_start``, output on
    ``message_delta``) and stop-reason mapping follow the frozen SGLang
    stream; collapsing each block into a single delta is the documented
    Miles fake-streaming difference. ``model`` must be the original Anthropic
    request model, not the backend's possibly-aliased response model.
    """
    events: list[AnthropicStreamEvent] = [
        MessageStartEvent(
            message=AnthropicMessagesResponse(
                id=id_factory(),
                content=[],
                model=model,
                usage=_anthropic_usage_from_openai(
                    response.usage, include_input=True, include_output=True, force_zero_output=True
                ),
            )
        )
    ]

    message = response.choices[0].message if response.choices else None
    index = 0

    if message is not None and message.reasoning_content:
        events.append(ContentBlockStartEvent(index=index, content_block=ThinkingBlock(thinking="")))
        events.append(ContentBlockDeltaEvent(index=index, delta=ThinkingDelta(thinking=message.reasoning_content)))
        events.append(ContentBlockStopEvent(index=index))
        index += 1

    if message is not None and message.content:
        events.append(ContentBlockStartEvent(index=index, content_block=TextBlock(text="")))
        events.append(ContentBlockDeltaEvent(index=index, delta=TextDelta(text=message.content)))
        events.append(ContentBlockStopEvent(index=index))
        index += 1

    if message is not None and message.tool_calls:
        for tool_call in message.tool_calls:
            events.append(
                ContentBlockStartEvent(
                    index=index,
                    content_block=ToolUseBlock(
                        id=tool_call.id or f"toolu_{uuid.uuid4().hex}",
                        name=tool_call.function.name,
                        input={},
                    ),
                )
            )
            if tool_call.function.arguments:
                events.append(
                    ContentBlockDeltaEvent(
                        index=index, delta=InputJsonDelta(partial_json=tool_call.function.arguments)
                    )
                )
            events.append(ContentBlockStopEvent(index=index))
            index += 1

    finish_reason = (response.choices[0].finish_reason or "stop") if response.choices else "stop"
    if finish_reason not in STOP_REASON_MAP:
        logger.warning("Unmapped OpenAI finish_reason %r; defaulting to end_turn", finish_reason)
    events.append(
        MessageDeltaEvent(
            delta=AnthropicMessageEndDelta(stop_reason=STOP_REASON_MAP.get(finish_reason, "end_turn")),
            usage=_anthropic_usage_from_openai(response.usage, include_input=False, include_output=True),
        )
    )
    events.append(MessageStopEvent())
    return tuple(events)
