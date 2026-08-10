# Upstream provenance and adaptation ledger

This package is a derivative of SGLang's Anthropic Messages entrypoint. It exists so Miles does not maintain a second hand-written Anthropic ↔ OpenAI converter; see `docs/developer/anthropic-session-api.md` for the governing design.

## Source pin

- Upstream repository: <https://github.com/sgl-project/sglang> (Apache-2.0, same license as Miles).
- `SOURCE_SHA`: `cb05a44f35a7c9e27e46d74112cc841ca674ef43` (the `sglang-miles` runtime candidate recorded in the design doc, fetched 2026-08-10).
- Copied source files at that commit:

| Upstream path | Git blob SHA | sha256 |
|---|---|---|
| `python/sglang/srt/entrypoints/anthropic/protocol.py` | `95217d6b399e0dd1b215760d97235085b8818288` | `ca5ee76a2f4fa02fbeb4f2bd43cab2914a36137a0f0c622aeefed81cef16b2dd` |
| `python/sglang/srt/entrypoints/anthropic/serving.py` | `d846142104465380cfd6d574d6c8282f87bc7fec` | `18349a6c475dbbf1749b54fca251d078a3fe2b687ff9c71e528f76aead59da31` |

- `RUNTIME_SHA` verification against the deployed image digest is **pending Phase 0**; the design rule `SOURCE_SHA = RUNTIME_SHA` has not been discharged yet. The development checkout this branch was built against installs a different sglang lineage, so on-image import/differential runs remain mandatory before rollout.
- Deliberately excluded upstream patches: `09193bf36fbec930bd54649ac64a7a5ede76d46b` (main-only `tool_reference` grouping/order fix); adopt only together with a runtime upgrade and its own behavior review.

## File mapping

- `protocol.py`: byte-identical to the upstream blob below the `# --- BEGIN VERBATIM UPSTREAM SOURCE ---` marker; only the provenance header above the marker was added (checked by `tests/fast/router/test_anthropic_codec.py`).
- `codec.py`: derived from `serving.py`. The upstream file is not retained; the ledger below accounts for every upstream symbol.
- The upstream tests were not vendored; `tests/fast/router/test_anthropic_codec.py` holds Miles-authored goldens derived by reading the frozen source.

## Adaptation ledger for `codec.py`

Kept with identical semantics (mechanical moves; nested closures promoted to module functions where they were stateless):

- `STOP_REASON_MAP`, `_cached_prompt_tokens`, `_anthropic_input_tokens`, `_anthropic_usage_from_openai`, `_extract_system_text`, `_scrub_error_message`.
- `_convert_to_chat_completion_request` → `to_openai_request` + `_convert_messages` + `_convert_tools` + `_apply_tool_choice`, with `self._merge_inline_system` replaced by `AnthropicRequestContext.merge_inline_system`. The blanket except-Exception→400 wrapper reproduces the frozen `handle_messages` conversion-error path as a typed `AnthropicRequestError`.
- `_convert_anthropic_image_source_to_openai_part`, `_text_from_search_result`, `_convert_tool_result_content` (were nested defs, now module-level).
- `_convert_response` → `to_anthropic_response`, message ID now from the injected `id_factory` (production factory `anthropic_message_id` keeps the frozen `msg_<uuid4hex>` format).
- `_convert_openai_error_response` + `_error_response` → `to_anthropic_error(status_code, body)`: same payload parsing, 4xx `error.type` honoring, and `_scrub_error_message` policy, but it returns the envelope DTO and leaves HTTP response construction to the route. The inner `try/except` around the lossless `errors="replace"` decode was dropped (cannot raise).
- `ERROR_TYPE_MAP` is a documented Miles composite: 400/401/403/404/413/422/429/500/502/503/504 from the frozen `http_server.py` `/v1/messages` exception handler; 408 `request_timeout_error` from the frozen `serving.py::ERROR_TYPE_MAP`; anything unlisted → `api_error` (both sources agree on that default).

Removed (non-migratable runtime responsibilities per the design):

- `AnthropicServing` class shell, `handle_messages`, `_handle_non_streaming`, `_handle_streaming`, `_generate_anthropic_stream`, `handle_count_tokens`, `_chat_template`, and every `openai_serving_chat`/tokenizer-manager/`monotonic_time`/abort-task access. Miles serves requests through the existing `SessionCore.chat_completions`.
- `_wrap_sse_event`: SSE framing moved to the route, which renders each event DTO with the session server's `_render_json`.

Replaced by Miles behavior:

- Real token streaming → `to_anthropic_fake_sse_events`: one eager event tuple built from the final `ChatCompletionResponse` (block order thinking → text → tool_use, per-block index accounting, usage split and stop-reason mapping as in the frozen stream; one delta per block is the documented fake-streaming difference; `message_start.message.model` comes from the original Anthropic request). The frozen stream-only error/ping paths (`ErrorEvent`, `PingEvent`, `SignatureDelta`, upstream stream-error envelope parsing) have no counterpart because rendering happens before the HTTP response exists.
- `wrap_reasoning_history` / `apply_reasoning_enabled` / `_convert_assistant_thinking_blocks`: not ported. Thinking (request param and history blocks, including `redacted_thinking`) is outside the first launch profile and is rejected by `_validate_known_features` with `invalid_request_error`; the loop-level skip of thinking blocks is kept only as a structural guard. A future data-only `ReasoningPolicy` (deferred, see design doc) must be verified against the runtime parser/template before thinking can be enabled.
- New `_validate_known_features` + `AnthropicRequestContext` feature gates (images, `output_config`, `betas`, `tool_reference`, `search_result`, server tools): known-but-disabled typed features fail closed with 400 instead of the frozen accept-and-log/skip behavior. When a gate is enabled, conversion follows the frozen semantics unchanged. Unknown extra keys inside known models keep the frozen Pydantic ignore behavior.
- Serialization contract: routes dump wire models with `model_dump(mode="json", exclude_none=True, by_alias=True)` + `_render_json` (compact UTF-8), replacing the frozen `JSONResponse(...model_dump(exclude_none=True))` / `model_dump_json(exclude_none=True)` split. The **request** body additionally uses `exclude_unset=True`: the frozen path handed the `ChatCompletionRequest` object in-process and never re-serialized it, whereas Miles must produce wire bytes — dropping unset vendored-model defaults (`strict`, `n`, `min_tokens`, …) keeps the canonical `SessionRecord.request` and the TITO-rendered prompt identical to an equivalent client-written OpenAI request (cross-protocol parity is byte-order-sensitive because chat templates render tools in dict key order).
- Unexpected-exception envelope: the frozen `handle_messages`/`_handle_non_streaming` wrapped all processing in `except Exception → api_error 500` JSON. The Miles route preserves that wire guarantee with a broad `except Exception` around the core call (scrubbed generic envelope; `CancelledError` propagates), rather than letting Starlette's text/plain 500 page reach Anthropic SDK clients.

## Sync procedure

Update the runtime pin first, re-copy both files at the new `SOURCE_SHA`, diff against this snapshot, then replay each ledger entry; never edit `codec.py` directly from a different upstream revision. Any upstream patch that changes a conversion semantic branch is a behavior change requiring its own review, not a sync.
