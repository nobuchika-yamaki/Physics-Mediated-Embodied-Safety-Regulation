#!/usr/bin/env bash
# 15_run_darca_density_sweep_N1_N3_N5_N10.sh
#
# Density sweep for DARCA / FULL_INTEGRATED validation.
#
# Fixed principle:
# - Only n_agents is varied: 1, 3, 5, 10.
# - Conditions are identical across densities.
# - Scenarios are identical across densities.
# - Seeds, steps, and metrics are identical across densities.
# - This separates individual-control, minimal-social, moderate-density,
#   and N10 crowding conditions without changing the task battery.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

ENGINE_FILE="${ENGINE_FILE:-$HOME/Downloads/11_darca_emergent_intelligence_adaptive_reroute_evidence_all_in_one.py}"
DARCA_FILE="${DARCA_FILE:-$HOME/Downloads/darca_v24_.py}"
AGENT_CORE="${AGENT_CORE:-$HOME/Downloads/02_darca_v24_integrated_agent_core_gravity_fixed.py}"

BASE_OUTDIR="${BASE_OUTDIR:-$HOME/Desktop/15_darca_density_sweep_N1_N3_N5_N10}"
SEEDS="${SEEDS:-64}"
STEPS="${STEPS:-5000}"
WORKERS="${WORKERS:-auto}"
LOG_STRIDE="${LOG_STRIDE:-50}"

DENSITIES="${DENSITIES:-1 3 5 10}"

CONDITIONS="RANDOM_POLICY,REFLEX_SAFE_POLICY,GREEDY_RESOURCE_POLICY,DARCA_ONLY,FULL_INTEGRATED,SHUFFLED_INTERNAL_STATE,FROZEN_INTERNAL_STATE,NO_Q,NO_PHYSICS,NO_MEMORY"
SCENARIOS="unknown_risk_resource,safe_vs_risky_resource,adaptive_reroute_after_stable,delayed_hazard_after_stable,vertical_shortcut_tradeoff"

mkdir -p "$BASE_OUTDIR"

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

HELP_TEXT="$("$PYTHON_BIN" "$ENGINE_FILE" --help 2>&1 || true)"

MASTER_LOG="$BASE_OUTDIR/00_DENSITY_SWEEP_MASTER.log"
: > "$MASTER_LOG"

{
  echo "[INFO] Density sweep started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[INFO] Engine:     $ENGINE_FILE"
  echo "[INFO] DARCA file: $DARCA_FILE"
  echo "[INFO] Agent core: $AGENT_CORE"
  echo "[INFO] Base outdir: $BASE_OUTDIR"
  echo "[INFO] densities=$DENSITIES"
  echo "[INFO] seeds=$SEEDS steps=$STEPS log_stride=$LOG_STRIDE workers=$WORKERS"
  echo "[INFO] conditions=$CONDITIONS"
  echo "[INFO] scenarios=$SCENARIOS"
} | tee -a "$MASTER_LOG"

for N_AGENTS in $DENSITIES; do
  OUTDIR="$BASE_OUTDIR/N${N_AGENTS}_seeds${SEEDS}_steps${STEPS}"
  mkdir -p "$OUTDIR"

  RUN_LOG="$OUTDIR/00_RUN_N${N_AGENTS}.log"

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

  if grep -q -- "--agent-core" <<< "$HELP_TEXT"; then
    if [[ -f "$AGENT_CORE" ]]; then
      CMD+=(--agent-core "$AGENT_CORE")
    else
      echo "[ERROR] Engine supports --agent-core, but agent core file is missing: $AGENT_CORE" | tee -a "$MASTER_LOG" >&2
      exit 2
    fi
  fi

  {
    echo ""
    echo "================================================================"
    echo "[INFO] START N=${N_AGENTS}: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[INFO] OUTDIR=$OUTDIR"
    echo "[INFO] Command:"
    printf ' %q' "${CMD[@]}"
    echo
    echo "================================================================"
  } | tee -a "$MASTER_LOG"

  set +e
  "${CMD[@]}" 2>&1 | tee "$RUN_LOG"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [[ "$EXIT_CODE" -ne 0 ]]; then
    {
      echo "[ERROR] FAILED N=${N_AGENTS}: exit_code=$EXIT_CODE"
      echo "[ERROR] See: $RUN_LOG"
      echo "[ERROR] Density sweep stopped."
    } | tee -a "$MASTER_LOG" >&2
    exit "$EXIT_CODE"
  fi

  {
    echo "[INFO] DONE N=${N_AGENTS}: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[INFO] Report: $OUTDIR/07_INTRINSIC_INTELLIGENCE_REPORT.md"
    echo "[INFO] Summary CSV: $OUTDIR/05b_intelligence_emergence_summary.csv"
    echo "[INFO] Evidence matrix: $OUTDIR/05g_evidence_matrix.csv"
  } | tee -a "$MASTER_LOG"
done

# Aggregate the density-level summary CSVs into one file.
AGG_SCRIPT="$BASE_OUTDIR/01_collect_density_summaries.py"
cat > "$AGG_SCRIPT" <<'PY'
#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import sys

base = Path(sys.argv[1]).expanduser().resolve()
rows = []
for p in sorted(base.glob("N*_seeds*_steps*/05b_intelligence_emergence_summary.csv")):
    m = re.search(r"N(\d+)_seeds(\d+)_steps(\d+)", str(p.parent.name))
    if not m:
        continue
    n_agents, seeds, steps = m.groups()
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["n_agents"] = n_agents
            row["seeds"] = seeds
            row["steps"] = steps
            row["source_file"] = str(p)
            rows.append(row)

out = base / "02_density_sweep_intelligence_emergence_summary.csv"
if rows:
    fields = []
    seen = set()
    preferred = ["n_agents", "seeds", "steps", "condition"]
    for k in preferred:
        if any(k in r for r in rows) and k not in seen:
            fields.append(k); seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] wrote {out}")
else:
    print("[WARN] no summary rows found")
PY
chmod +x "$AGG_SCRIPT"

"$PYTHON_BIN" "$AGG_SCRIPT" "$BASE_OUTDIR" | tee -a "$MASTER_LOG"

echo "[INFO] Density sweep completed: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "[INFO] Aggregated summary: $BASE_OUTDIR/02_density_sweep_intelligence_emergence_summary.csv" | tee -a "$MASTER_LOG"
