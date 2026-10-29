#!/usr/bin/env bash
# track_daily.sh — one context-rot snapshot per day.
#
# Runs the pipeline against the live OpenCode DB, then appends ONE row to
# history/context_rot_history.csv so the trend survives the outputs/ directory being
# regenerated. The whole point is the column over time: the day the score at a fixed
# token depth stops being noise is the day the compaction threshold has evidence.
#
# Cron/systemd safe: paths are absolute, the pipeline cds to its own project root, and a
# failed run still leaves a row (rc + reason) instead of silently skipping a day.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DB="${OPENCODE_DB:-$HOME/.local/share/opencode/opencode.db}"
MEM_MD="${MEMORY_MD:-$HOME/.config/opencode/AGENTS.md}"
HIST="$ROOT/history/context_rot_history.csv"
LOG="$ROOT/history/last_run.log"

mkdir -p "$ROOT/history"
[ -x "$PY" ] || { printf '%s\n' "$(date -Is),ERROR,no-venv,,,," >>"$HIST"; exit 1; }
[ -f "$DB" ] || { printf '%s\n' "$(date -Is),ERROR,no-db,,,," >>"$HIST"; exit 1; }
[ -f "$HIST" ] || printf '%s\n' "run_at,status,sessions,steps,max_tokens_since_compaction,recommended_T_tokens,knee_tokens,score_at_64k,score_at_128k,score_at_256k" >"$HIST"

cd "$ROOT" || exit 1
started=$(date -Is)
"$PY" run_pipeline.py --db-path "$DB" --agents-md "$MEM_MD" --bootstrap >"$LOG" 2>&1
rc=$?

row=$("$PY" "$ROOT/tools/track_extract.py" "$ROOT" "$rc" "$started")
printf '%s\n' "$row" >>"$HIST"
printf '%s\n' "$row"
exit "$rc"
