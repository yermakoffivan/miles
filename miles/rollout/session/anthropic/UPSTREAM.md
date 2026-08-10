# Upstream provenance

This package is a derivative of SGLang's Anthropic Messages entrypoint. It exists so Miles does not maintain a second hand-written Anthropic ↔ OpenAI converter; see `docs/developer/anthropic-session-api.md` for the governing design.

## Source pin

- Upstream repository: <https://github.com/sgl-project/sglang> (Apache-2.0, same license as Miles).
- `SOURCE_SHA`: `cb05a44f35a7c9e27e46d74112cc841ca674ef43` (the `sglang-miles` runtime candidate recorded in the design doc, fetched 2026-08-10).
- Copied source files at that commit:

| Upstream path | Git blob SHA | sha256 |
|---|---|---|
| `python/sglang/srt/entrypoints/anthropic/protocol.py` | `95217d6b399e0dd1b215760d97235085b8818288` | `ca5ee76a2f4fa02fbeb4f2bd43cab2914a36137a0f0c622aeefed81cef16b2dd` |
| `python/sglang/srt/entrypoints/anthropic/serving.py` | `d846142104465380cfd6d574d6c8282f87bc7fec` | `18349a6c475dbbf1749b54fca251d078a3fe2b687ff9c71e528f76aead59da31` |

- `RUNTIME_SHA` verification against the deployed image digest is **pending Phase 0**; the design rule `SOURCE_SHA = RUNTIME_SHA` has not been discharged yet.
- Deliberately excluded upstream patches: `09193bf36fbec930bd54649ac64a7a5ede76d46b` (main-only `tool_reference` grouping/order fix); adopt only together with a runtime upgrade and its own behavior review.

## Snapshot state

This commit is the verbatim source import only: `protocol.py` is byte-identical to the upstream blob below its provenance-header marker, `serving.py` is byte-identical in full, nothing is wired into Miles, and `serving.py` is not importable here (its sglang-internal imports are absent by design). The safe refactor to a pure codec, the Miles route, and the symbol-level adaptation ledger land in the follow-up commit so every kept/dropped symbol stays mechanically auditable against these blobs.
