# MiniMax-M3.1 tool-call parser fixes: build and test kit

Branch `tt/m31-toolcall-fixes-v0.9.6` of `philei-tt/frontend-crates` = `dynamo-parsers-v2` v0.9.6 (the version
`ai-dynamo/dynamo` main pins) + upstream #525 + #526 + commit `0022bb16`:

- A required key whose schema provably admits `null` and that the model left out is emitted as `null`
  (MiniMax's chat template omits `None` arguments, so M3.1 omits nullable keys: 34/135 OpenRouter
  `toolcall_formats` failures).
- Exactly one `\n` / `\r\n` just inside a string parameter's tags is dropped (5/135 failures), in both the batch
  converter and the streaming emitter.

Unit tests: `cargo test -p dynamo-parsers-v2 --test tt_m31_repro` (18 cases, every split and one char per chunk).
The same fix on current frontend-crates main is branch `tt/m31-toolcall-fixes`.

## Build the frontend image

`Dockerfile.m31-frontend` (build from this directory): dynamo main `945f7850` + `dynamo-patches/` (ai-dynamo/dynamo
#16476, unpaired tool results -> 400) + this parser branch, layered on the deployed M3.1 frontend
`ghcr.io/tenstorrent/tt-d-gen/dynamo-frontend:sha-e025a59d` (ai-dynamo 1.5.0 -> 1.7.0). About 7 min on a 64-core host.

Untested: the 1.7.0 frontend against 1.5.0 workers (`dgen-worker:sha-e025a59d`). Smoke-test that first.
For `reasoning_split` clients, run the frontend with `DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS=true`.

## Test

`run_suites.sh <label> <frontend_url>` runs, against one frontend, with no OpenRouter key:

- OpenRouter `benchmark-harness` (`OpenRouterTeam/benchmark-harness@c6a05aa`) `toolcall_formats` with streaming and
  non-streaming upstream calls, and `toolcall_schema_fuzz`. `responses_adapter.py` turns the harness's Responses API
  calls into `/v1/chat/completions` calls, as OpenRouter does for a provider.
- `MiniMax-AI/MiniMax-Provider-Verifier@336ccc7` M3.1 text, stream and reasoning_effort suites.

It expects the harness, the verifier and bun under `~/m31-toolcall/tests/` (paths at the top of the script).
Run it once against the stock frontend and once against the patched one on the same workers, and compare.
