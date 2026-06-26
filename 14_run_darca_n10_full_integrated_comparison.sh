#!/usr/bin/env bash
# 14_run_darca_n10_full_integrated_comparison.sh
# DARCA N10 reviewer-level comparison:
# FULL_INTEGRATED vs DARCA_ONLY vs fixed baselines / ablations.
#
# Fixed principle:
# - n_agents is kept at 10.
# - The fixed scenario battery is kept unchanged.
# - The only conceptual change from the DARCA_ONLY-centered run is that FULL_INTEGRATED is restored.
# - seeds and steps are increased for reviewer-level stability.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

ENGINE_FILE="${ENGINE_FILE:-$HOME/Downloads/11_darca_emergent_intelligence_adaptive_reroute_evidence_all_in_one.py}"
DARCA_FILE="${DARCA_FILE:-$HOME/Downloads/darca_v24_.py}"
AGENT_CORE="${AGENT_CORE:-$HOME/Downloads/02_darca_v24_integrated_agent_core_gravity_fixed.py}"

OUTDIR="${OUTDIR:-$HOME/Desktop/14_darca_N10_full_integrated_vs_darca_only}"
SEEDS="${SEEDS:-64}"
STEPS="${STEPS:-5000}"
N_AGENTS="${N_AGENTS:-10}"
WORKERS="${WORKERS:-auto}"
LOG_STRIDE="${LOG_STRIDE:-50}"

CONDITIONS="RANDOM_POLICY,REFLEX_SAFE_POLICY,GREEDY_RESOURCE_POLICY,DARCA_ONLY,FULL_INTEGRATED,SHUFFLED_INTERNAL_STATE,FROZEN_INTERNAL_STATE,NO_Q,NO_PHYSICS,NO_MEMORY"
SCENARIOS="unknown_risk_resource,safe_vs_risky_resource,adaptive_reroute_after_stable,delayed_hazard_after_stable,vertical_shortcut_tradeoff"

mkdir -p "$OUTDIR"

if [[ ! -f "$ENGINE_FILE" ]]; then
  echo "[ERROR] Engine file not found: $ENGINE_FILE" >&2
  exit 2
fi

if [[ ! -f "$DARCA_FILE" ]]; then
  echo "[ERROR] DARCA file not found: $DARCA_FILE" >&2
  exit 2
fi

if [[ ! -f "$AGENT_CORE" ]]; then
  echo "[WARN] Agent core file not found at: $AGENT_CORE" >&2
  echo "[WARN] Continuing without --agent-core if the engine does not require it." >&2
fi

echo "[INFO] Engine:     $ENGINE_FILE"
echo "[INFO] DARCA file: $DARCA_FILE"
echo "[INFO] Agent core: $AGENT_CORE"
echo "[INFO] Outdir:     $OUTDIR"
echo "[INFO] seeds=$SEEDS steps=$STEPS n_agents=$N_AGENTS log_stride=$LOG_STRIDE workers=$WORKERS"
echo "[INFO] conditions=$CONDITIONS"
echo "[INFO] scenarios=$SCENARIOS"

CMD=(
  "$PYTHON_BIN" -u "$ENGINE_FILE"
  --seeds "$SEEDS"
  --steps "$STEPS"
  --n-agents "$N_AGENTS"
  --workers "$WORKERS"
  --log-stride "$LOG_STRIDE"
  --conditions "$CONDITIONS"
  --scenarios "$SCENARIOS"
  --outdir "$OUTDIR"
  --darca-file "$DARCA_FILE"
  --clean
)

# Add --agent-core only if the engine supports it.
HELP_TEXT="$("$PYTHON_BIN" "$ENGINE_FILE" --help 2>&1 || true)"
if grep -q -- "--agent-core" <<< "$HELP_TEXT"; then
  if [[ -f "$AGENT_CORE" ]]; then
    CMD+=(--agent-core "$AGENT_CORE")
  else
    echo "[ERROR] Engine supports --agent-core, but agent core file is missing: $AGENT_CORE" >&2
    exit 2
  fi
fi

RUN_LOG="$OUTDIR/00_N10_FULL_VS_DARCA_ONLY_RUN.log"

echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo
echo "[INFO] Running..."

"${CMD[@]}" 2>&1 | tee "$RUN_LOG"

echo "[INFO] Done."
echo "[INFO] Main report should be:"
echo "       $OUTDIR/07_INTRINSIC_INTELLIGENCE_REPORT.md"
echo "[INFO] Key CSV files:"
echo "       $OUTDIR/05_aggregate_episode_summary.csv"
echo "       $OUTDIR/05b_intelligence_emergence_summary.csv"
echo "       $OUTDIR/05g_evidence_matrix.csv"
echo "       $OUTDIR/05i_pareto_tradeoff_diagnostics.csv"
