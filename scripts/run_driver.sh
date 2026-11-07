#!/usr/bin/env bash
# Usage: run_driver.sh pilot|full [--effort high] [--model gpt-6.1-sol]
#                      [--wall DURATION] [--stop-at N] [--reset-views]
#        run_driver.sh resume pilot|full [--wall 30m] [--stop-at N] [--effort high] [--model gpt-6.1-sol]
#   resume           continue the Codex session recorded in out/<mode>/session_id. It falls back to a fresh
#                    `codex exec` with brief_rendered.md only if codex exited non-zero within 30 s AND emitted no
#                    thread.started event (never after a monitor/wall-cap stop); the fallback thread id is written to
#                    out/<mode>/session_id_fallback, session_id is never overwritten.
#   Defaults         model gpt-6.1-sol, reasoning effort high. There are NO view budgets: the looker daemon serves
#                    unlimited views, and every val image viewed through look.py is moved by the daemon to
#                    data/train (look and it moves). out/looker_val_cap.json is deleted at the start of every run.
#   pilot            fresh pilot runs temporarily replace workspace/data with a 5 train + 5 val subset (one image
#                    each from the first five classes alphabetically; first file by sorted name). The full data is
#                    stashed in .pilot_stash/data and restored when the codex run ends (and by an EXIT trap).
#                    Full mode never touches the data.
#   fresh run        after the pre-run commit, workspace/ root is cleaned of leftovers of earlier runs (classify.py,
#                    NOTES.md, *.py except check.py/look.py, *.json/*.csv/*.txt/*.md except AGENTS.md/BRIEF.md,
#                    __pycache__/). Not done on resume.
#   --wall DURATION  override the wall-clock limit, as for timeout(1), e.g. 20m, 2h (default: pilot 15m, full 45m; a safety net,
#                    the budget monitor is the real limit)
#   --stop-at N      stop threshold for scripts/budget_monitor.py, in percent of the Codex 5-hour window
#                    (default: $STOP_AT or 90). The monitor is started automatically for every run and
#                    resume with --kill (SIGINT to `codex exec` when crossed) and stopped at the end;
#                    its output goes to out/<mode>/budget.stdout and its log to out/<mode>/budget.jsonl.
#   --extra-prompt T (resume only) append text T to the base resume prompt (saved in out/<mode>/brief_resume_N.md)
#   --reset-views    write {"views_used": 0, "val_views_used": 0} to out/looker_state.json and clear
#                    out/looker_viewed_val.json (only if the looker daemon is running). The counters are records only.
#
# Notes:
#  - Codex's workspace-write sandbox has network access OFF by default, so no extra flag is
#    needed to keep the agent offline.
#  - The looker daemon (looker/daemon.py) and the judge watcher (judge/watch.py) must already
#    be running before this script starts; it only warns if it cannot find them.
set -u

ACTION=run
if [ "${1:-}" = resume ]; then ACTION=resume; shift; fi
MODE="${1:-}"
[ $# -gt 0 ] && shift
EFFORT="high"
MODEL="gpt-6.1-sol"
WALL_OVERRIDE=""
STOP_AT="${STOP_AT:-90}"
RESET_VIEWS=0
EXTRA_PROMPT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --effort) EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --model)  MODEL="${2:?--model needs a value}"; shift 2 ;;
    --wall)   WALL_OVERRIDE="${2:?--wall needs a value}"; shift 2 ;;
    --stop-at) STOP_AT="${2:?--stop-at needs a value}"; shift 2 ;;
    --reset-views) RESET_VIEWS=1; shift ;;
    --extra-prompt) EXTRA_PROMPT="${2:?--extra-prompt needs a value}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  pilot) WALL=15m; NTRAIN=5;   NVAL=5
         EXTRA="This is a short pilot with 5 train + 5 val images; produce a working classify.py and NOTES.md that follow AGENTS.md." ;;
  full)  WALL=45m;  NTRAIN=500;  NVAL=500; EXTRA="" ;;
  *) echo "usage: $0 [resume] pilot|full [--effort high] [--model gpt-6.1-sol] [--wall DURATION] [--stop-at N] [--reset-views]" >&2; exit 2 ;;
esac
[ -n "$WALL_OVERRIDE" ] && WALL="$WALL_OVERRIDE"
case "$STOP_AT" in ''|*[!0-9.]*) echo "--stop-at must be a number (percent)" >&2; exit 2 ;; esac
if [ "$ACTION" = resume ] && [ "$RESET_VIEWS" = 1 ]; then echo "--reset-views is not allowed with resume" >&2; exit 2; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# The original driver committed and pushed the whole project here; this copy only logs the checkpoint.
commit_push() {
  echo "checkpoint: $1"
}

# Budget monitor: started for every run/resume, killed when the driver exits.
MONITOR_PID=""
start_monitor() {  # $1 = out dir
  rm -f "$1/budget_stop"   # stale marker of an earlier run; the monitor re-creates it when it kills this run
  STOP_LINES_BEFORE=$(grep -c 'budget stop' "$1/budget.stdout" 2>/dev/null || true); STOP_LINES_BEFORE=${STOP_LINES_BEFORE:-0}
  nohup python3 scripts/budget_monitor.py --kill --stop-at "$STOP_AT" --log "$1/budget.jsonl" \
    --marker "$1/budget_stop" >> "$1/budget.stdout" 2>&1 &
  MONITOR_PID=$!
  echo "budget monitor started (pid $MONITOR_PID, stop at ${STOP_AT}% of the 5h window, log $1/budget.jsonl)"
}
# True if the budget monitor stopped the current run (marker file, or a new "budget stop" line in budget.stdout).
monitor_stopped() {  # $1 = out dir
  [ -e "$1/budget_stop" ] && return 0
  local now; now=$(grep -c 'budget stop' "$1/budget.stdout" 2>/dev/null || true)
  [ "${now:-0}" -gt "${STOP_LINES_BEFORE:-0}" ]
}
stop_monitor() { [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null; MONITOR_PID=""; }

# Pilot data subset: stash the full workspace/data and build a 5 train + 5 val copy (first five classes
# alphabetically, first file by sorted name in each of train and val). Full mode never calls these.
PILOT_STASHED=0
build_pilot_subset() {  # $1 = workspace dir, $2 = stash dir (must not exist)
  local ws="$1" stash="$2"
  if [ -e "$stash" ]; then echo "ERROR: $stash already exists (leftover of an interrupted pilot?); refusing to continue" >&2; return 1; fi
  [ -d "$ws/data" ] || { echo "ERROR: $ws/data not found" >&2; return 1; }
  mkdir -p "$stash" && mv "$ws/data" "$stash/data" || return 1
  PILOT_STASHED=1
  python3 - "$stash/data" "$ws/data" <<'PY' || return 1
import os, shutil, sys
src, dst = sys.argv[1:]
classes = sorted(os.listdir(os.path.join(src, "train")))[:5]
for split in ("train", "val"):
    for c in classes:
        files = sorted(os.listdir(os.path.join(src, split, c)))
        os.makedirs(os.path.join(dst, split, c), exist_ok=True)
        shutil.copy2(os.path.join(src, split, c, files[0]), os.path.join(dst, split, c, files[0]))
print("pilot subset built in %s: classes %s, 1 train + 1 val image each" % (dst, ", ".join(classes)))
PY
}
restore_pilot_subset() {  # $1 = workspace dir, $2 = stash dir; no-op unless this run stashed the data
  local ws="$1" stash="$2"
  [ "$PILOT_STASHED" = 1 ] && [ -d "$stash/data" ] || return 0
  rm -rf "$ws/data" && mv "$stash/data" "$ws/data" && rmdir "$stash" && PILOT_STASHED=0 \
    && echo "pilot data restored: $ws/data ($(find "$ws/data" -name '*.JPEG' | wc -l) images)"
}
trap 'stop_monitor; restore_pilot_subset "$ROOT/workspace" "$ROOT/.pilot_stash"' EXIT
budget_once() { python3 scripts/budget_monitor.py --once 2>&1 | tail -n 1; }
json_str() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# Duration (as for timeout(1): 90, 30s, 30m, 2h) -> whole minutes, rounded up.
to_minutes() {
  python3 - "$1" <<'PY'
import sys, math
v = sys.argv[1]
mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
n = float(v[:-1]) * mult[v[-1]] if v[-1] in mult else float(v)
print(max(1, math.ceil(n / 60)))
PY
}

# Print the Codex thread id (thread.started event) of an events file.
extract_session_id() {
  python3 - "$1" <<'PY'
import sys, json
for i, line in enumerate(open(sys.argv[1])):
    if i > 50: break
    try: e = json.loads(line)
    except ValueError: continue
    if e.get("type") == "thread.started" and e.get("thread_id"):
        print(e["thread_id"]); break
PY
}

if [ "$ACTION" = resume ]; then
  OUT="out/$MODE"
  [ -f "$OUT/session_id" ] || { echo "no $OUT/session_id; cannot resume" >&2; exit 1; }
  [ -f "$OUT/brief_rendered.md" ] || { echo "no $OUT/brief_rendered.md; cannot resume" >&2; exit 1; }
  SID=$(cat "$OUT/session_id")
  N=1
  while [ -e "$OUT/brief_resume_$N.md" ]; do N=$((N + 1)); done
  rm -f out/looker_val_cap.json   # no view budgets any more
  WALL_MIN=$(to_minutes "$WALL")
  RBRIEF="$OUT/brief_resume_$N.md"
  printf '%s\n' "Continue the same task. classify.py and NOTES.md on disk are exactly as you left them. Re-read AGENTS.md and NOTES.md first." > "$RBRIEF"
  [ -n "$EXTRA_PROMPT" ] && printf '\n%s\n' "$EXTRA_PROMPT" >> "$RBRIEF"
  echo "resume #$N of $MODE: session $SID, wall $WALL (${WALL_MIN} min), views unlimited"

  commit_push "pre-run resume $MODE $N"
  start_monitor "$OUT"
  BUDGET_BEFORE=$(budget_once); echo "budget before: $BUDGET_BEFORE"
  START_EPOCH=$(date +%s); START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  # `codex exec resume` accepts neither -s nor -C: cd into workspace/ and set the sandbox via -c.
  EV="$OUT/events_resume_$N.jsonl"; ERR="$OUT/codex_stderr_resume_$N.log"; LAST="$OUT/last_message_resume_$N.md"
  ( cd workspace && timeout --signal=INT --kill-after=60 "$WALL" codex exec resume "$SID" -m "$MODEL" \
      -c model_reasoning_effort="$EFFORT" -c sandbox_mode='"workspace-write"' --skip-git-repo-check --json \
      -o "../$LAST" "$(cat "../$RBRIEF")" > "../$EV" 2> "../$ERR" < /dev/null )
  EXIT=$?
  END_EPOCH=$(date +%s); ELAPSED=$((END_EPOCH - START_EPOCH))

  FALLBACK=false; NEWSID=""
  # Fall back to a fresh session ONLY if the resume itself failed to start: codex exited non-zero within
  # 30 s AND emitted no thread.started event. Never after a monitor stop, a wall-cap stop (124/130) or a
  # run longer than 30 s (stderr noise such as "failed to record rollout items: thread ... not found"
  # after a SIGINT is NOT a resume failure).
  RESUME_TID=$(extract_session_id "$EV")
  if [ "$EXIT" -ne 0 ] && [ "$EXIT" -ne 124 ] && [ "$EXIT" -ne 130 ] && [ "$ELAPSED" -lt 30 ] \
     && [ -z "$RESUME_TID" ] && ! monitor_stopped "$OUT"; then
    FALLBACK=true
    echo "resume failed (exit $EXIT after ${ELAPSED}s); falling back to a fresh codex exec" >&2
    FBRIEF="$OUT/brief_resume_${N}_fallback.md"
    { cat "$OUT/brief_rendered.md"; printf '\n%s\n' "You are continuing a previous attempt by another session; read NOTES.md and classify.py first."; } > "$FBRIEF"
    EV="$OUT/events_resume_${N}_fallback.jsonl"; ERR="$OUT/codex_stderr_resume_${N}_fallback.log"; LAST="$OUT/last_message_resume_${N}_fallback.md"
    timeout --signal=INT --kill-after=60 "$WALL" codex exec -C workspace -m "$MODEL" \
      -c model_reasoning_effort="$EFFORT" -s workspace-write --skip-git-repo-check --json \
      -o "$LAST" "$(cat "$FBRIEF")" > "$EV" 2> "$ERR" < /dev/null
    EXIT=$?
    NEWSID=$(extract_session_id "$EV")
    # The original session id in $OUT/session_id is never overwritten; the fallback thread goes to its own file.
    [ -n "$NEWSID" ] && printf '%s\n' "$NEWSID" > "$OUT/session_id_fallback"
    END_EPOCH=$(date +%s); ELAPSED=$((END_EPOCH - START_EPOCH))
  fi
  END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  BUDGET_AFTER=$(budget_once); echo "budget after: $BUDGET_AFTER"
  stop_monitor

  python3 - "$OUT/run_meta.json" "$N" "$START_ISO" "$END_ISO" "$EXIT" "$ELAPSED" "$WALL" "$SID" "$FALLBACK" "$NEWSID" "$BUDGET_BEFORE" "$BUDGET_AFTER" <<'PY'
import os, sys, json
p, n, st, en, ex, el, wall, sid, fb, nsid, bb, ba = sys.argv[1:]
m = json.load(open(p))
r = {"n": int(n), "start": st, "end": en, "exit_code": int(ex), "elapsed_seconds": int(el),
     "wall_limit": wall, "resumed_session": sid,
     "fallback_to_fresh_session": fb == "true", "stopped_by_monitor": os.path.exists(os.path.join(os.path.dirname(p), "budget_stop")), "budget_before": bb, "budget_after": ba}
if nsid: r["new_session_id"] = nsid
m.setdefault("resumes", []).append(r)
json.dump(m, open(p, "w"), indent=2); open(p, "a").write("\n")
PY
  commit_push "post-run resume $MODE $N"
  echo "== run_driver resume summary =="
  echo "mode: $MODE  resume #$N  fallback: $FALLBACK  exit code: $EXIT (124 = hit wall-clock limit)  elapsed: ${ELAPSED}s"
  echo "events: $(wc -l < "$EV") lines  classify.py: $( [ -f workspace/classify.py ] && stat -c %s workspace/classify.py || echo missing ) bytes"
  exit 0
fi

OUT="out/$MODE"
TS=$(date +%Y%m%d_%H%M%S)

# Preserve the artifacts of an earlier run of the same mode.
if [ -e "$OUT" ]; then
  mv "$OUT" "out/${MODE}_${TS}" && echo "moved previous $OUT -> out/${MODE}_${TS}"
fi
mkdir -p "$OUT"

# Warnings about background services.
[ -f out/looker_state.json ] || echo "WARNING: out/looker_state.json not found; looker daemon may not be running." >&2
pgrep -f 'daemon\.py' >/dev/null || echo "WARNING: no daemon.py process found (looker daemon)." >&2
pgrep -f 'watch\.py'  >/dev/null || echo "WARNING: no watch.py process found (judge watcher)." >&2

# Housekeeping: reset the looker cap (only on request, only if the daemon is up).
if [ "$RESET_VIEWS" = 1 ]; then
  if pgrep -f 'daemon\.py' >/dev/null; then
    printf '{"views_used": 0, "val_views_used": 0}' > out/looker_state.json
    printf '[]' > out/looker_viewed_val.json
    echo "reset-views: wrote {\"views_used\": 0, \"val_views_used\": 0} to out/looker_state.json and cleared out/looker_viewed_val.json (daemon is running)"
  else
    echo "WARNING: --reset-views given but no daemon.py process found; NOT resetting out/looker_state.json." >&2
  fi
fi
# No view budgets: make sure no stale val cap file is left for the daemon.
rm -f out/looker_val_cap.json
# Rotate looker/judge logs (if non-empty) into the run dir.
for f in looker judge; do
  if [ -s "out/$f.jsonl" ]; then
    mv "out/$f.jsonl" "$OUT/${f}_prev_${TS}.jsonl" && echo "rotated out/$f.jsonl -> $OUT/${f}_prev_${TS}.jsonl"
  fi
done

# Pilot only: swap workspace/data for a 5 train + 5 val subset (restored by restore_pilot_subset / the EXIT trap).
if [ "$MODE" = pilot ]; then
  build_pilot_subset "$ROOT/workspace" "$ROOT/.pilot_stash" || exit 1
fi

# Render the brief (BRIEF.md itself is not modified). It states no time or view numbers.
sed -e "s/{{N_TRAIN}}/$NTRAIN/g" -e "s/{{N_VAL}}/$NVAL/g" \
    workspace/BRIEF.md > "$OUT/brief_rendered.md"
if [ -n "$EXTRA" ]; then
  printf '\n%s\n' "$EXTRA" >> "$OUT/brief_rendered.md"
fi

commit_push "pre-run $MODE"

# Clean leftovers of earlier runs from the workspace root (everything is in git by now).
echo "cleaning workspace/ root:"
( cd workspace && find . -maxdepth 1 \( -type f \( -name classify.py -o -name NOTES.md \
    -o \( \( -name '*.py' -o -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.md' \) \
         ! -name check.py ! -name look.py ! -name AGENTS.md ! -name BRIEF.md \) \) \
    -o \( -type d ! -name . ! -name data ! -name '.looker' ! -name '.agents' ! -name '.aws' ! -name '.codex' ! -name '.git' \) \) -print -exec rm -rf {} + | sed 's|^\./|  removed |' )

start_monitor "$OUT"
BUDGET_BEFORE=$(budget_once); echo "budget before: $BUDGET_BEFORE"

START_EPOCH=$(date +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

timeout --signal=INT --kill-after=60 "$WALL" codex exec -C workspace -m "$MODEL" \
  -c model_reasoning_effort="$EFFORT" -s workspace-write --skip-git-repo-check --json \
  -o "$OUT/last_message.md" "$(cat "$OUT/brief_rendered.md")" \
  > "$OUT/events.jsonl" 2> "$OUT/codex_stderr.log" < /dev/null
EXIT=$?

END_EPOCH=$(date +%s)
END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$((END_EPOCH - START_EPOCH))
BUDGET_AFTER=$(budget_once); echo "budget after: $BUDGET_AFTER"
stop_monitor
restore_pilot_subset "$ROOT/workspace" "$ROOT/.pilot_stash"   # before the post-run commit (the trap is a safety net)

SESSION_ID=$(extract_session_id "$OUT/events.jsonl")
[ -n "$SESSION_ID" ] && printf '%s\n' "$SESSION_ID" > "$OUT/session_id" || echo "WARNING: no thread.started event found in $OUT/events.jsonl" >&2

cat > "$OUT/run_meta.json" <<JSON
{
  "mode": "$MODE",
  "model": "$MODEL",
  "effort": "$EFFORT",
  "wall_limit": "$WALL",
  "views": "unlimited (val images viewed are moved to data/train)",
  "n_train": $NTRAIN,
  "n_val": $NVAL,
  "start": "$START_ISO",
  "end": "$END_ISO",
  "elapsed_seconds": $ELAPSED,
  "exit_code": $EXIT,
  "session_id": "$SESSION_ID",
  "budget_before": $(json_str "$BUDGET_BEFORE"),
  "budget_after": $(json_str "$BUDGET_AFTER")
}
JSON

commit_push "post-run $MODE"

echo "== run_driver summary =="
echo "mode: $MODE  model: $MODEL  effort: $EFFORT  views: unlimited (val views move to data/train)"
echo "exit code: $EXIT (124 = hit wall-clock limit)  elapsed: ${ELAPSED}s"
echo "events: $(wc -l < "$OUT/events.jsonl") lines  classify.py: $( [ -f workspace/classify.py ] && stat -c %s workspace/classify.py || echo missing ) bytes"
echo "outputs in $OUT/"
exit 0
