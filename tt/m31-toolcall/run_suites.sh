#!/bin/bash
# Tool-call quality suites against one Dynamo frontend, for baseline-vs-canary comparison on Virtu.
#   run_suites.sh <label> <frontend_base_url, e.g. http://10.43.x.y:8000> [model]
# Writes everything under ~/m31-toolcall/results/<label>/. No OpenRouter key needed: the harness talks to
# responses_adapter.py, which calls the frontend's /v1/chat/completions the way OpenRouter calls a provider.
set -uo pipefail
LABEL=$1; FE=${2%/}; MODEL=${3:-MiniMaxAI/MiniMax-M3.1-Flash}
T=~/m31-toolcall/tests; OUT=~/m31-toolcall/results/$LABEL; mkdir -p "$OUT"
BUN=~/.bun/bin/bun; VPY=~/tis-verifier/.workflow_venvs/.venv_minimax_provider_verifier/bin/python
log(){ echo "$(date -u +%FT%TZ) [$LABEL] $*" | tee -a "$OUT/timeline.txt"; }

curl -sf -m 20 "$FE/v1/models" | grep -q "$MODEL" || { log "STOP: $MODEL not served at $FE"; exit 1; }

run_adapter(){ # port stream logfile
  UPSTREAM=$FE MODEL=$MODEL STREAM=$2 LOG=$3 PORT=$1 nohup python3 "$T/responses_adapter.py" > "$3.stdout" 2>&1 &
  echo $!
}
harness(){ # name benchmark port concurrency
  ( cd "$T/benchmark-harness" && OPENROUTER_BASE_URL=http://127.0.0.1:$3/api/v1 OPENROUTER_API_KEY=direct-no-key \
    $BUN run bench -- --benchmark "$2" --model minimax/minimax-m3.1-flash --concurrency "$4" \
      --artifact-dir "$OUT/$1" > "$OUT/$1.log" 2>&1 )
  log "$1 done (exit $?): $(grep -E -i 'accuracy|score|correct|pass' "$OUT/$1.log" | tail -2 | tr '\n' ' ')"
}

log "start against $FE"
A1=$(run_adapter 18081 1 "$OUT/raw_formats_stream.jsonl"); A2=$(run_adapter 18082 0 "$OUT/raw_formats_nonstream.jsonl")
A3=$(run_adapter 18083 1 "$OUT/raw_fuzz_stream.jsonl"); sleep 2
harness formats_stream toolcall_formats 18081 8 &
harness formats_nonstream toolcall_formats 18082 8 &
wait
harness fuzz_stream toolcall_schema_fuzz 18083 16
kill $A1 $A2 $A3 2>/dev/null

log "MiniMax verifier M3.1 suites"
cd "$T/MiniMax-Provider-Verifier/m3_format_check" || exit 1
for s in m3_1_flash_text_test.py m3_1_flash_stream_test.py m3_1_flash_reasoning_effort_tests.py; do
  M3_BASE_URL=$FE/v1 M3_API_KEY=direct-no-key M3_MODEL=$MODEL M3_RUN_LOG="$OUT/verifier_${s%.py}.runlog" \
    timeout 5400 $VPY -m pytest "$s" -q -p no:cacheprovider --junitxml="$OUT/verifier_${s%.py}.xml" \
    > "$OUT/verifier_${s%.py}.log" 2>&1
  log "verifier $s: $(tail -1 "$OUT/verifier_${s%.py}.log")"
done
log "ALL DONE"
