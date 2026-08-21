#!/usr/bin/env bash
# Runs a command, then commits+pushes its output dir so artifacts survive sandbox resets.
set -u
OUTDIR="$1"; shift
"$@"
RC=$?
echo "[run_and_commit] command exit=$RC; committing artifacts in $OUTDIR"
cd /home/user/next-paper
git add -A "paper-ideas/CoalGameRec/code/results/journal_runs/$OUTDIR" 2>/dev/null || true
git add -A paper-ideas/CoalGameRec/code/results/journal_runs/_notebook_logs 2>/dev/null || true
git -c user.email="arena-agent@users.noreply.github.com" -c user.name="arena-agent" \
    commit -q -m "auto-commit run artifacts: $OUTDIR (exit=$RC)" 2>/dev/null || echo "nothing to commit"
git push -q origin arena/019fdd75-next-paper 2>/dev/null || echo "push failed (will retry later)"
exit $RC
