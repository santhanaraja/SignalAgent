#!/usr/bin/env bash
# THE BAKE, AS ONE RE-RUNNABLE UNIT.
#
# It exists so the commit step can throw a bake away and produce another one
# from a newer base. Rebasing a generated artifact onto a newer generated
# artifact cannot work: both sides rewrite the same bytes, and a merge that
# succeeds is worse than one that fails, because it produces a file matching
# neither input. So the workflow never merges. It resets and re-bakes, and
# the newest bake from the newest base wins.
#
# FAILURE SEMANTICS MIRROR THE WORKFLOW STEPS THIS REPLACES, EXACTLY:
#   required, a failure fails the run  : signal_engine.py, history_manager.py
#   non-blocking, logged and continued : fear_greed_engine.py,
#                                        framework.framework_runner,
#                                        notify_intraday.py
# A framework failure leaves the previous artifact in place and must not
# block signal history; an F&G or Slack hiccup must never block the pipeline.
#
# --no-notify is for RETRIES. The post-close Slack push lives in its own
# workflow step and must fire at most once per run: on 2026-09-28 the same
# close report went out twice, at 21:43 and 22:49, because each attempt
# re-posted and the data/last_notified marker never rode a commit. A retry
# re-bakes the data and says nothing.
set -uo pipefail

NOTIFY=1
for arg in "$@"; do
  case "$arg" in
    --no-notify) NOTIFY=0 ;;
    *) echo "[bake] unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

run_required() {
  echo "[bake] $1 (required)"
  if ! python "${@:2}"; then
    echo "[bake] FAILED: $1 — required, failing the bake" >&2
    exit 1
  fi
}

run_optional() {
  echo "[bake] $1 (non-blocking)"
  python "${@:2}" || echo "[bake] $1 failed — continuing, as the workflow step did"
}

run_required "signal engine" signal_engine.py
run_optional "fear & greed" fear_greed_engine.py
# Framework BEFORE the history manager: history_manager reads
# public/framework.json to log regime_change events, which must reflect THIS
# run's regime rather than the previous one's.
run_optional "framework engine" -m framework.framework_runner
run_required "history manager" history_manager.py
run_optional "intraday stop-breach alerts" notify_intraday.py
if [ "$NOTIFY" -eq 1 ]; then
  run_optional "post-close assessment" notify_assessment.py
else
  echo "[bake] post-close assessment SKIPPED (retry — it already ran this job)"
fi
echo "[bake] complete"
