#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_darca_robustness_analysis_from_density_outputs.py

Robustness analysis for the DARCA density-sweep and causal analysis.

This script does NOT rerun simulations and does NOT modify the validation battery.
It reads existing density-sweep outputs, then tests whether the main causal conclusions
survive alternative aggregation choices.

Main claims tested
------------------
Claim 1:
    FULL_INTEGRATED corrects DARCA_ONLY's unsafe survival extension.

Operational test:
    FULL_INTEGRATED vs DARCA_ONLY should show positive signed benefit for
    damage_per_alive_step, collision_per_alive_step, and fall_per_alive_step,
    without a large survival/resource collapse.

Claim 2:
    The correction is mainly associated with physics / embodied safety coupling,
    not uniquely with active internal-state dynamics.

Operational test:
    FULL_INTEGRATED vs NO_PHYSICS should be consistently positive for safety metrics,
    while FULL vs NO_Q / NO_MEMORY / SHUFFLED / FROZEN should be weaker or mixed.

Default input
-------------
    ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10

Default output
--------------
    ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10/91_robustness_analysis

Run
---
    python3 -u 17_darca_robustness_analysis_from_density_outputs.py

or

    python3 -u 17_darca_robustness_analysis_from_density_outputs.py \
      --base-dir ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# =============================================================================
# Configuration
# =============================================================================

PRIMARY_METRICS = [
    "survival_steps",
    "resource_gain_per_alive_step",
    "damage_per_alive_step",
    "collision_per_alive_step",
    "fall_per_alive_step",
]

SAFETY_METRICS = [
    "damage_per_alive_step",
    "collision_per_alive_step",
    "fall_per_alive_step",
]

EFFICIENCY_METRICS = [
    "survival_steps",
    "resource_gain_per_alive_step",
]

DIAGNOSTIC_METRICS = [
    "mean_q",
    "mean_physics_score",
    "signal_rate",
    "signal_entropy_bits",
]

ALL_METRICS = PRIMARY_METRICS + DIAGNOSTIC_METRICS

HIGHER_IS_BETTER = {
    "survival_steps": True,
    "resource_gain_per_alive_step": True,
    "damage_per_alive_step": False,
    "collision_per_alive_step": False,
    "fall_per_alive_step": False,
    "mean_q": False,
    "mean_physics_score": True,
    "signal_rate": None,
    "signal_entropy_bits": None,
}

CORE_CONTRASTS = [
    ("FULL_INTEGRATED", "DARCA_ONLY", "full_vs_darca_only"),
    ("FULL_INTEGRATED", "NO_PHYSICS", "full_vs_no_physics"),
    ("FULL_INTEGRATED", "FROZEN_INTERNAL_STATE", "full_vs_frozen"),
    ("FULL_INTEGRATED", "SHUFFLED_INTERNAL_STATE", "full_vs_shuffled"),
    ("FULL_INTEGRATED", "NO_Q", "full_vs_no_q"),
    ("FULL_INTEGRATED", "NO_MEMORY", "full_vs_no_memory"),
]

NEGATIVE_CONTROL_CONTRASTS = [
    ("FULL_INTEGRATED", "RANDOM_POLICY", "full_vs_random"),
    ("FULL_INTEGRATED", "REFLEX_SAFE_POLICY", "full_vs_reflex_safe"),
    ("FULL_INTEGRATED", "GREEDY_RESOURCE_POLICY", "full_vs_greedy_resource"),
    ("DARCA_ONLY", "RANDOM_POLICY", "darca_only_vs_random"),
]

ALL_CONTRASTS = CORE_CONTRASTS + NEGATIVE_CONTROL_CONTRASTS

BATTERY_SCENARIO = "_battery_mean"


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_density_from_path(path: Path) -> Optional[int]:
    candidates = [path.name] + [p.name for p in path.parents]
    for part in candidates:
        m = re.search(r"(?:^|_)N(\d+)(?:_|$)", part)
        if m:
            return int(m.group(1))
    return None


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_num(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def signed_benefit(left: float, right: float, metric: str) -> float:
    """
    Positive means left condition is better than right condition.
    For lower-is-better metrics, benefit = right - left.
    """
    direction = HIGHER_IS_BETTER.get(metric, True)
    if direction is True:
        return float(left) - float(right)
    if direction is False:
        return float(right) - float(left)
    return float(left) - float(right)


def robust_round(x, nd: int = 6) -> str:
    x = to_num(x)
    if not np.isfinite(x):
        return ""
    return f"{x:.{nd}g}"


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.median(arr))


def trimmed_mean(values: Sequence[float], proportion: float = 0.20) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) < 5:
        return float(np.mean(arr))
    arr = np.sort(arr)
    k = int(math.floor(len(arr) * proportion))
    if 2 * k >= len(arr):
        return float(np.mean(arr))
    return float(np.mean(arr[k:len(arr)-k]))


def winsorized_mean(values: Sequence[float], proportion: float = 0.20) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) < 5:
        return float(np.mean(arr))
    lo, hi = np.quantile(arr, [proportion, 1.0 - proportion])
    arr = np.clip(arr, lo, hi)
    return float(np.mean(arr))


def sign_fraction_positive(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr > 0))


def bootstrap_ci(values: Sequence[float], n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def standardized_mean_difference(left_vals: Sequence[float], right_vals: Sequence[float], metric: str) -> float:
    """
    Signed standardized contrast using pooled SD. Positive means left is better.
    Used only for composite safety metrics.
    """
    l = np.asarray(left_vals, dtype=float)
    r = np.asarray(right_vals, dtype=float)
    mask_l = np.isfinite(l)
    mask_r = np.isfinite(r)
    l = l[mask_l]
    r = r[mask_r]
    if len(l) == 0 or len(r) == 0:
        return float("nan")
    pooled = np.concatenate([l, r])
    sd = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
    if sd <= 1e-12:
        return 0.0
    raw = float(np.mean(l) - np.mean(r))
    if HIGHER_IS_BETTER.get(metric) is False:
        raw = -raw
    return raw / sd


# =============================================================================
# Data loading and standardization
# =============================================================================

def standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    mappings = {
        "survival_steps": [
            "survival_steps",
            "mean_survival_steps",
            "across_scenario_mean_survival_steps",
        ],
        "resource_gain_per_alive_step": [
            "resource_gain_per_alive_step",
            "mean_resource_gain_per_alive_step",
            "across_scenario_mean_resource_gain_per_alive_step",
        ],
        "damage_per_alive_step": [
            "damage_per_alive_step",
            "mean_damage_per_alive_step",
            "across_scenario_mean_damage_per_alive_step",
        ],
        "collision_per_alive_step": [
            "collision_per_alive_step",
            "mean_collision_per_alive_step",
            "across_scenario_mean_collision_per_alive_step",
        ],
        "fall_per_alive_step": [
            "fall_per_alive_step",
            "mean_fall_per_alive_step",
            "across_scenario_mean_fall_per_alive_step",
        ],
        "mean_q": [
            "mean_q",
            "mean_mean_q",
            "across_scenario_mean_mean_q",
        ],
        "mean_physics_score": [
            "mean_physics_score",
            "mean_mean_physics_score",
            "across_scenario_mean_mean_physics_score",
        ],
        "signal_rate": [
            "signal_rate",
            "mean_signal_rate",
            "across_scenario_mean_signal_rate",
        ],
        "signal_entropy_bits": [
            "signal_entropy_bits",
            "mean_signal_entropy_bits",
            "across_scenario_mean_signal_entropy_bits",
        ],
    }

    for target, candidates in mappings.items():
        src = first_existing_column(out, candidates)
        if src is not None:
            out[target] = pd.to_numeric(out[src], errors="coerce")

    if "condition" in out.columns:
        out["condition"] = out["condition"].astype(str)

    if "scenario" not in out.columns:
        out["scenario"] = BATTERY_SCENARIO

    return out


def load_raw_episode_summary(base_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in sorted(base_dir.glob("N*_seeds*_steps*/01_episode_summary.csv")):
        n = parse_density_from_path(p.parent)
        if n is None:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] cannot read {p}: {e}", file=sys.stderr)
            continue
        df["n_agents"] = n
        df["source_file"] = str(p)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = standardize_metric_columns(df)

    if "seed" not in df.columns:
        for c in ["base_seed", "episode_seed", "replicate", "rep", "world_seed"]:
            if c in df.columns:
                df["seed"] = df[c]
                break
    if "seed" not in df.columns:
        # Fallback is not ideal. It avoids crashing but disables meaningful pairing.
        df["seed"] = np.arange(len(df), dtype=int)

    if "scenario" not in df.columns:
        df["scenario"] = BATTERY_SCENARIO

    keep = ["n_agents", "condition", "scenario", "seed", "source_file"]
    for m in ALL_METRICS:
        if m in df.columns:
            keep.append(m)

    out = df[keep].copy()
    for c in ["n_agents", "seed"] + [m for m in ALL_METRICS if m in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_scenario_summary(base_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in sorted(base_dir.glob("N*_seeds*_steps*/05_aggregate_episode_summary.csv")):
        n = parse_density_from_path(p.parent)
        if n is None:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] cannot read {p}: {e}", file=sys.stderr)
            continue
        df["n_agents"] = n
        df["source_file"] = str(p)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = standardize_metric_columns(df)

    keep = ["n_agents", "condition", "scenario", "source_file"]
    for m in ALL_METRICS:
        if m in df.columns:
            keep.append(m)
    out = df[keep].copy()
    for c in ["n_agents"] + [m for m in ALL_METRICS if m in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_battery_summary(base_dir: Path, summary_csv: Optional[Path]) -> pd.DataFrame:
    candidates: List[Path] = []
    if summary_csv:
        candidates.append(summary_csv)
    candidates.append(base_dir / "02_density_sweep_intelligence_emergence_summary.csv")

    rows: List[pd.DataFrame] = []
    loaded_combined = False

    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] cannot read {p}: {e}", file=sys.stderr)
                continue
            df["source_file"] = str(p)
            rows.append(df)
            loaded_combined = True
            break

    if not loaded_combined:
        for p in sorted(base_dir.glob("N*_seeds*_steps*/05b_intelligence_emergence_summary.csv")):
            n = parse_density_from_path(p.parent)
            if n is None:
                continue
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] cannot read {p}: {e}", file=sys.stderr)
                continue
            df["n_agents"] = n
            df["source_file"] = str(p)
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df = standardize_metric_columns(df)
    df["scenario"] = BATTERY_SCENARIO

    keep = ["n_agents", "condition", "scenario", "source_file"]
    extras = [
        "nonrandom_competence_score_0_to_5",
        "mean_delta_survival_steps_vs_RANDOM_POLICY",
        "mean_delta_resource_gain_per_alive_step_vs_RANDOM_POLICY",
        "mean_delta_damage_per_alive_step_vs_RANDOM_POLICY",
        "mean_delta_collision_per_alive_step_vs_RANDOM_POLICY",
        "mean_delta_fall_per_alive_step_vs_RANDOM_POLICY",
        "mean_delta_survival_steps_vs_DARCA_ONLY",
        "mean_delta_resource_gain_per_alive_step_vs_DARCA_ONLY",
        "mean_delta_damage_per_alive_step_vs_DARCA_ONLY",
        "mean_delta_collision_per_alive_step_vs_DARCA_ONLY",
        "mean_delta_fall_per_alive_step_vs_DARCA_ONLY",
    ]
    for m in ALL_METRICS:
        if m in df.columns:
            keep.append(m)
    for e in extras:
        if e in df.columns:
            keep.append(e)

    out = df[keep].copy()
    for c in ["n_agents"] + [m for m in ALL_METRICS if m in out.columns] + [e for e in extras if e in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =============================================================================
# Core contrast calculations
# =============================================================================

def collapse_raw_to_seed_means(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    metrics = [m for m in ALL_METRICS if m in raw.columns]
    if not metrics:
        return pd.DataFrame()
    return (
        raw.groupby(["n_agents", "scenario", "condition", "seed"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )


def paired_seed_differences(seed_df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    metrics = [m for m in ALL_METRICS if m in seed_df.columns]
    l = seed_df[seed_df["condition"] == left].copy()
    r = seed_df[seed_df["condition"] == right].copy()
    if l.empty or r.empty:
        return pd.DataFrame()
    merged = l.merge(
        r,
        on=["n_agents", "scenario", "seed"],
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return pd.DataFrame()

    out = merged[["n_agents", "scenario", "seed"]].copy()
    out["left_condition"] = left
    out["right_condition"] = right
    for m in metrics:
        lv = pd.to_numeric(merged[f"{m}_left"], errors="coerce")
        rv = pd.to_numeric(merged[f"{m}_right"], errors="coerce")
        out[f"{m}_raw_diff"] = lv - rv
        out[f"{m}_signed_benefit"] = [
            signed_benefit(a, b, m) if np.isfinite(a) and np.isfinite(b) else np.nan
            for a, b in zip(lv, rv)
        ]
    return out


def all_paired_seed_differences(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right, label in ALL_CONTRASTS:
        d = paired_seed_differences(seed_df, left, right)
        if d.empty:
            continue
        d["contrast"] = label
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def seed_level_bootstrap(pair_df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    rows = []
    for (contrast, n_agents, scenario), g in pair_df.groupby(["contrast", "n_agents", "scenario"], dropna=False):
        row = {
            "contrast": contrast,
            "n_agents": n_agents,
            "scenario": scenario,
            "n_paired_seeds": int(g["seed"].nunique()),
        }
        for m in PRIMARY_METRICS:
            col = f"{m}_signed_benefit"
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(vals, n_boot, rng)
            row[f"{m}_mean_signed_benefit"] = mean
            row[f"{m}_ci95_low"] = lo
            row[f"{m}_ci95_high"] = hi
            row[f"{m}_positive_seed_fraction"] = sign_fraction_positive(vals)
            row[f"{m}_median_signed_benefit"] = safe_median(vals)
        safety_cols = [f"{m}_signed_benefit" for m in SAFETY_METRICS if f"{m}_signed_benefit" in g.columns]
        if safety_cols:
            smat = g[safety_cols].apply(pd.to_numeric, errors="coerce")
            row["all_three_safety_positive_seed_fraction"] = float((smat.gt(0).all(axis=1)).mean())
            row["any_safety_negative_seed_fraction"] = float((smat.lt(0).any(axis=1)).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def mean_level_contrasts(df: pd.DataFrame, level: str, contrasts: Sequence[Tuple[str, str, str]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    metrics = [m for m in ALL_METRICS if m in df.columns]
    keys = ["n_agents", "scenario"]
    rows = []
    for key_values, g in df.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_dict = dict(zip(keys, key_values))
        conds = {str(r["condition"]): r for _, r in g.iterrows()}
        for left, right, label in contrasts:
            if left not in conds or right not in conds:
                continue
            lrow = conds[left]
            rrow = conds[right]
            out = {
                "analysis_level": level,
                **key_dict,
                "contrast": label,
                "left_condition": left,
                "right_condition": right,
            }
            for m in metrics:
                lv = to_num(lrow.get(m))
                rv = to_num(rrow.get(m))
                if not np.isfinite(lv) or not np.isfinite(rv):
                    continue
                out[f"{m}_left"] = lv
                out[f"{m}_right"] = rv
                out[f"{m}_raw_diff_left_minus_right"] = lv - rv
                out[f"{m}_signed_benefit"] = signed_benefit(lv, rv, m)
            safety_b = [out.get(f"{m}_signed_benefit", np.nan) for m in SAFETY_METRICS]
            safety_b = [x for x in safety_b if np.isfinite(to_num(x))]
            out["all_three_safety_positive"] = int(len(safety_b) == 3 and all(x > 0 for x in safety_b))
            out["safety_positive_count"] = int(sum(1 for x in safety_b if x > 0))
            out["safety_available_count"] = int(len(safety_b))
            rows.append(out)

    return pd.DataFrame(rows)


# =============================================================================
# Robustness modules
# =============================================================================

def leave_one_density_analysis(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute scenario-pooled contrasts while excluding each density.
    Uses scenario-level aggregate rows; no raw rerun.
    """
    if scenario_summary.empty:
        return pd.DataFrame()

    densities = sorted(pd.to_numeric(scenario_summary["n_agents"], errors="coerce").dropna().astype(int).unique())
    metrics = [m for m in PRIMARY_METRICS if m in scenario_summary.columns]
    rows = []

    for excluded in densities:
        sub = scenario_summary[scenario_summary["n_agents"] != excluded].copy()
        if sub.empty:
            continue
        # Equal weight by density-scenario-condition cell.
        grouped = sub.groupby(["condition"], dropna=False)[metrics].mean().reset_index()
        grouped["n_agents"] = -1
        grouped["scenario"] = f"leave_out_N{excluded}"
        ctab = mean_level_contrasts(grouped, "leave_one_density", ALL_CONTRASTS)
        if ctab.empty:
            continue
        ctab["excluded_density"] = excluded
        rows.append(ctab)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def leave_one_scenario_analysis(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute density-pooled contrasts while excluding each scenario.
    """
    if scenario_summary.empty or "scenario" not in scenario_summary.columns:
        return pd.DataFrame()

    scenarios = sorted(str(x) for x in scenario_summary["scenario"].dropna().unique())
    metrics = [m for m in PRIMARY_METRICS if m in scenario_summary.columns]
    rows = []

    for excluded in scenarios:
        sub = scenario_summary[scenario_summary["scenario"].astype(str) != excluded].copy()
        if sub.empty:
            continue
        # Keep density separate; average over remaining scenarios.
        grouped = sub.groupby(["n_agents", "condition"], dropna=False)[metrics].mean().reset_index()
        grouped["scenario"] = f"leave_out_{excluded}"
        ctab = mean_level_contrasts(grouped, "leave_one_scenario", ALL_CONTRASTS)
        if ctab.empty:
            continue
        ctab["excluded_scenario"] = excluded
        rows.append(ctab)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def alternative_safety_composites(pair_df: pd.DataFrame, scenario_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create robust safety composite summaries:
    - equal_safety_benefit_mean: mean of signed safety benefits.
    - all_three_safety_positive_fraction: strict sign consistency.
    - pareto_safe_extension_fraction: all safety positive and survival/resource not strongly negative.
    """
    rows = []

    # Seed-level composite if raw paired data exist.
    if not pair_df.empty:
        for (contrast, n_agents, scenario), g in pair_df.groupby(["contrast", "n_agents", "scenario"], dropna=False):
            row = {
                "analysis_level": "seed_level",
                "contrast": contrast,
                "n_agents": n_agents,
                "scenario": scenario,
                "n_rows": int(len(g)),
            }
            safety_cols = [f"{m}_signed_benefit" for m in SAFETY_METRICS if f"{m}_signed_benefit" in g.columns]
            if safety_cols:
                smat = g[safety_cols].apply(pd.to_numeric, errors="coerce")
                row["equal_safety_benefit_mean"] = float(smat.mean(axis=1, skipna=True).mean(skipna=True))
                row["equal_safety_benefit_median"] = float(smat.mean(axis=1, skipna=True).median(skipna=True))
                row["all_three_safety_positive_fraction"] = float(smat.gt(0).all(axis=1).mean())
                row["two_or_more_safety_positive_fraction"] = float((smat.gt(0).sum(axis=1) >= 2).mean())
            surv = pd.to_numeric(g.get("survival_steps_signed_benefit", pd.Series(dtype=float)), errors="coerce")
            res = pd.to_numeric(g.get("resource_gain_per_alive_step_signed_benefit", pd.Series(dtype=float)), errors="coerce")
            if safety_cols and len(surv) == len(g) and len(res) == len(g):
                smat = g[safety_cols].apply(pd.to_numeric, errors="coerce")
                # Conservative: safety all positive, survival not below -10 steps, resource not below -1e-4.
                pareto = smat.gt(0).all(axis=1) & (surv >= -10.0) & (res >= -1e-4)
                row["pareto_safe_extension_fraction"] = float(pareto.mean())
            rows.append(row)

    # Aggregate-level standardized safety contrast by density/scenario.
    if not scenario_summary.empty:
        metrics = [m for m in SAFETY_METRICS if m in scenario_summary.columns]
        for (n_agents, scenario), g in scenario_summary.groupby(["n_agents", "scenario"], dropna=False):
            for left, right, label in ALL_CONTRASTS:
                l = g[g["condition"] == left]
                r = g[g["condition"] == right]
                if l.empty or r.empty:
                    continue
                row = {
                    "analysis_level": "scenario_mean_standardized",
                    "contrast": label,
                    "n_agents": n_agents,
                    "scenario": scenario,
                    "n_rows": 1,
                }
                signed = []
                for m in metrics:
                    lv = to_num(l.iloc[0].get(m))
                    rv = to_num(r.iloc[0].get(m))
                    if np.isfinite(lv) and np.isfinite(rv):
                        signed.append(signed_benefit(lv, rv, m))
                        row[f"{m}_signed_benefit"] = signed_benefit(lv, rv, m)
                row["equal_safety_benefit_mean"] = safe_mean(signed)
                row["all_three_safety_positive_fraction"] = float(len(signed) == 3 and all(x > 0 for x in signed))
                row["two_or_more_safety_positive_fraction"] = float(sum(1 for x in signed if x > 0) >= 2)
                rows.append(row)

    return pd.DataFrame(rows)


def outlier_robustness(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    rows = []
    for (contrast, n_agents, scenario), g in pair_df.groupby(["contrast", "n_agents", "scenario"], dropna=False):
        row_base = {
            "contrast": contrast,
            "n_agents": n_agents,
            "scenario": scenario,
            "n_paired_rows": int(len(g)),
        }
        for m in PRIMARY_METRICS:
            col = f"{m}_signed_benefit"
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            row = dict(row_base)
            row["metric"] = m
            row["mean"] = safe_mean(vals)
            row["median"] = safe_median(vals)
            row["trimmed_mean_20pct"] = trimmed_mean(vals, 0.20)
            row["winsorized_mean_20pct"] = winsorized_mean(vals, 0.20)
            row["positive_fraction"] = sign_fraction_positive(vals)
            row["min"] = float(np.min(vals))
            row["max"] = float(np.max(vals))
            row["q05"] = float(np.quantile(vals, 0.05))
            row["q95"] = float(np.quantile(vals, 0.95))
            # Robust sign agreement across estimators
            estimators = [row["mean"], row["median"], row["trimmed_mean_20pct"], row["winsorized_mean_20pct"]]
            row["all_estimators_positive"] = int(all(np.isfinite(x) and x > 0 for x in estimators))
            rows.append(row)

    return pd.DataFrame(rows)


def negative_control_robustness(battery_contrasts: pd.DataFrame, seed_boot: pd.DataFrame) -> pd.DataFrame:
    rows = []

    wanted = {label for _, _, label in NEGATIVE_CONTROL_CONTRASTS}
    if not battery_contrasts.empty:
        g = battery_contrasts[battery_contrasts["contrast"].isin(wanted)].copy()
        for _, r in g.iterrows():
            row = {
                "analysis_level": "battery_mean",
                "contrast": r.get("contrast"),
                "n_agents": r.get("n_agents"),
                "scenario": r.get("scenario"),
            }
            for m in PRIMARY_METRICS:
                col = f"{m}_signed_benefit"
                if col in r:
                    row[f"{m}_signed_benefit"] = to_num(r[col])
            safety = [row.get(f"{m}_signed_benefit", np.nan) for m in SAFETY_METRICS]
            safety = [x for x in safety if np.isfinite(to_num(x))]
            row["safety_positive_count"] = sum(1 for x in safety if x > 0)
            row["all_three_safety_positive"] = int(len(safety) == 3 and all(x > 0 for x in safety))
            rows.append(row)

    if not seed_boot.empty:
        g = seed_boot[seed_boot["contrast"].isin(wanted)].copy()
        for _, r in g.iterrows():
            row = {
                "analysis_level": "seed_bootstrap",
                "contrast": r.get("contrast"),
                "n_agents": r.get("n_agents"),
                "scenario": r.get("scenario"),
                "n_paired_seeds": r.get("n_paired_seeds"),
            }
            for m in PRIMARY_METRICS:
                col = f"{m}_mean_signed_benefit"
                if col in r:
                    row[f"{m}_mean_signed_benefit"] = to_num(r[col])
                    row[f"{m}_ci95_low"] = to_num(r.get(f"{m}_ci95_low"))
                    row[f"{m}_ci95_high"] = to_num(r.get(f"{m}_ci95_high"))
                    row[f"{m}_positive_seed_fraction"] = to_num(r.get(f"{m}_positive_seed_fraction"))
            rows.append(row)

    return pd.DataFrame(rows)


def component_attribution_robustness(
    battery_contrasts: pd.DataFrame,
    scenario_contrasts: pd.DataFrame,
    seed_boot: pd.DataFrame,
    leave_density: pd.DataFrame,
    leave_scenario: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize whether component attribution survives multiple robustness layers.
    """
    component_labels = {
        "full_vs_no_physics": "physics_or_embodied_safety_coupling",
        "full_vs_no_q": "q_layer",
        "full_vs_no_memory": "memory_layer",
        "full_vs_shuffled": "ordered_internal_state",
        "full_vs_frozen": "active_internal_dynamics",
        "full_vs_darca_only": "integrated_layers_total_effect",
    }

    sources = [
        ("battery_mean", battery_contrasts),
        ("scenario_mean", scenario_contrasts),
        ("seed_bootstrap", seed_boot),
        ("leave_one_density", leave_density),
        ("leave_one_scenario", leave_scenario),
    ]

    rows = []
    for contrast, component in component_labels.items():
        row = {"contrast": contrast, "component_interpretation": component}
        all_values = []
        source_summaries = {}
        for source_name, df in sources:
            if df.empty or "contrast" not in df.columns:
                continue
            g = df[df["contrast"] == contrast].copy()
            if g.empty:
                continue

            vals = []
            for m in SAFETY_METRICS:
                for col in [
                    f"{m}_signed_benefit",
                    f"{m}_mean_signed_benefit",
                    f"{m}_signed_benefit_left_over_right",
                ]:
                    if col in g.columns:
                        vals.extend(pd.to_numeric(g[col], errors="coerce").dropna().tolist())
                        break
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                all_values.extend(vals)
                source_summaries[f"{source_name}_mean_safety_benefit"] = float(np.mean(vals))
                source_summaries[f"{source_name}_positive_fraction"] = float(np.mean(np.asarray(vals) > 0))
                source_summaries[f"{source_name}_n_values"] = int(len(vals))

        row.update(source_summaries)
        if all_values:
            arr = np.asarray(all_values, dtype=float)
            row["overall_mean_safety_benefit"] = float(np.mean(arr))
            row["overall_median_safety_benefit"] = float(np.median(arr))
            row["overall_positive_fraction"] = float(np.mean(arr > 0))
            row["overall_abs_mean_safety_benefit"] = float(np.mean(np.abs(arr)))
            if row["overall_positive_fraction"] >= 0.80 and row["overall_mean_safety_benefit"] > 0:
                row["robustness_class"] = "robust_positive"
            elif 0.55 <= row["overall_positive_fraction"] < 0.80 and row["overall_mean_safety_benefit"] > 0:
                row["robustness_class"] = "moderate_positive"
            elif 0.45 <= row["overall_positive_fraction"] < 0.55:
                row["robustness_class"] = "mixed_neutral"
            else:
                row["robustness_class"] = "weak_or_negative"
        else:
            row["robustness_class"] = "not_available"
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Reporting and plots
# =============================================================================

def write_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def summarize_main_claim(seed_boot: pd.DataFrame, leave_density: pd.DataFrame, leave_scenario: pd.DataFrame, composites: pd.DataFrame, components: pd.DataFrame) -> List[str]:
    lines: List[str] = []

    def source_claim(df: pd.DataFrame, name: str) -> str:
        if df.empty or "contrast" not in df.columns:
            return f"{name}: not available."
        g = df[df["contrast"] == "full_vs_darca_only"].copy()
        if g.empty:
            return f"{name}: full_vs_darca_only not available."
        vals = []
        for m in SAFETY_METRICS:
            for col in [f"{m}_mean_signed_benefit", f"{m}_signed_benefit"]:
                if col in g.columns:
                    vals.extend(pd.to_numeric(g[col], errors="coerce").dropna().tolist())
                    break
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return f"{name}: safety values not available."
        return (
            f"{name}: mean safety benefit={np.mean(vals):.6g}, "
            f"positive fraction={np.mean(np.asarray(vals) > 0):.3f}, "
            f"n_values={len(vals)}."
        )

    lines.append(source_claim(seed_boot, "Seed-level bootstrap"))
    lines.append(source_claim(leave_density, "Leave-one-density"))
    lines.append(source_claim(leave_scenario, "Leave-one-scenario"))

    if not composites.empty:
        g = composites[composites["contrast"] == "full_vs_darca_only"]
        if not g.empty and "all_three_safety_positive_fraction" in g.columns:
            vals = pd.to_numeric(g["all_three_safety_positive_fraction"], errors="coerce").dropna()
            if len(vals):
                lines.append(
                    f"Alternative safety composite: mean all-three-safety-positive fraction={vals.mean():.3f}."
                )

    if not components.empty:
        for contrast in ["full_vs_no_physics", "full_vs_no_q", "full_vs_no_memory", "full_vs_frozen", "full_vs_shuffled"]:
            g = components[components["contrast"] == contrast]
            if not g.empty:
                r = g.iloc[0]
                lines.append(
                    f"Component {contrast}: class={r.get('robustness_class')}, "
                    f"overall_mean_safety_benefit={robust_round(r.get('overall_mean_safety_benefit'), 6)}, "
                    f"overall_positive_fraction={robust_round(r.get('overall_positive_fraction'), 4)}."
                )

    return lines


def save_plots(outdir: Path, seed_boot: pd.DataFrame, components: pd.DataFrame, leave_density: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        return
    figdir = outdir / "figures"
    ensure_dir(figdir)

    # Plot core safety benefit by density from seed bootstrap.
    if not seed_boot.empty:
        for contrast in ["full_vs_darca_only", "full_vs_no_physics"]:
            g = seed_boot[seed_boot["contrast"] == contrast].copy()
            if g.empty:
                continue
            for metric in SAFETY_METRICS:
                col = f"{metric}_mean_signed_benefit"
                lo = f"{metric}_ci95_low"
                hi = f"{metric}_ci95_high"
                if col not in g.columns:
                    continue
                # Average over scenarios for display only.
                plot_df = g.groupby("n_agents", dropna=False).agg(
                    mean=(col, "mean"),
                    low=(lo, "mean") if lo in g.columns else (col, "mean"),
                    high=(hi, "mean") if hi in g.columns else (col, "mean"),
                ).reset_index().sort_values("n_agents")
                x = pd.to_numeric(plot_df["n_agents"], errors="coerce").to_numpy()
                y = pd.to_numeric(plot_df["mean"], errors="coerce").to_numpy()
                if np.isfinite(y).sum() < 2:
                    continue
                plt.figure(figsize=(7, 4.5))
                plt.plot(x, y, marker="o")
                plt.axhline(0, linewidth=1)
                plt.xlabel("n_agents")
                plt.ylabel(f"{metric} signed benefit")
                plt.title(f"{contrast}: seed-level safety robustness")
                plt.tight_layout()
                plt.savefig(figdir / f"{contrast}_{metric}_seed_robustness.png", dpi=180)
                plt.close()

    # Component attribution bar plot.
    if not components.empty and "overall_mean_safety_benefit" in components.columns:
        g = components.copy()
        g = g.sort_values("overall_mean_safety_benefit", ascending=False)
        labels = g["contrast"].astype(str).tolist()
        vals = pd.to_numeric(g["overall_mean_safety_benefit"], errors="coerce").to_numpy()
        if len(vals) > 0 and np.isfinite(vals).any():
            plt.figure(figsize=(9, 4.8))
            plt.bar(range(len(vals)), vals)
            plt.axhline(0, linewidth=1)
            plt.xticks(range(len(vals)), labels, rotation=35, ha="right")
            plt.ylabel("overall mean safety benefit")
            plt.title("Component attribution robustness")
            plt.tight_layout()
            plt.savefig(figdir / "component_attribution_robustness.png", dpi=180)
            plt.close()


def write_report(
    outdir: Path,
    raw: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    battery_summary: pd.DataFrame,
    seed_boot: pd.DataFrame,
    leave_density: pd.DataFrame,
    leave_scenario: pd.DataFrame,
    composites: pd.DataFrame,
    outlier: pd.DataFrame,
    negative: pd.DataFrame,
    components: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# DARCA robustness analysis report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This robustness analysis does not rerun simulations and does not modify the task battery.")
    lines.append("It tests whether the causal conclusions survive seed-level pairing, leave-one-density, leave-one-scenario, alternative safety composites, outlier-resistant estimators, negative controls, and component-attribution checks.")
    lines.append("")
    lines.append("## Data loaded")
    lines.append("")
    lines.append(f"- Raw episode rows: {len(raw)}")
    lines.append(f"- Scenario-summary rows: {len(scenario_summary)}")
    lines.append(f"- Battery-summary rows: {len(battery_summary)}")
    if not raw.empty:
        dens = sorted(pd.to_numeric(raw["n_agents"], errors="coerce").dropna().astype(int).unique())
        lines.append(f"- Densities in raw data: {', '.join('N'+str(x) for x in dens)}")
        scenarios = sorted(str(x) for x in raw["scenario"].dropna().unique())
        lines.append(f"- Scenarios in raw data: {', '.join(scenarios)}")
    lines.append("")
    lines.append("## Main robustness checks")
    lines.append("")
    for x in summarize_main_claim(seed_boot, leave_density, leave_scenario, composites, components):
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Interpretation rule")
    lines.append("")
    lines.append("- Strong support: FULL_INTEGRATED vs DARCA_ONLY remains positive for damage, collision, and fall across seed-level, leave-one-density, and leave-one-scenario analyses.")
    lines.append("- Density-limited support: the effect remains for N3/N5/N10 but weakens at N1.")
    lines.append("- Scenario-limited support: the effect disappears when a specific scenario is removed.")
    lines.append("- Component attribution: a component is treated as a robust contributor only if FULL vs lesion remains positive across multiple robustness layers.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `01_seed_level_bootstrap.csv`")
    lines.append("- `02_leave_one_density.csv`")
    lines.append("- `03_leave_one_scenario.csv`")
    lines.append("- `04_alternative_safety_composites.csv`")
    lines.append("- `05_outlier_robustness.csv`")
    lines.append("- `06_negative_control_robustness.csv`")
    lines.append("- `07_component_attribution_robustness.csv`")
    lines.append("- `08_seed_level_pairwise_differences.csv`")
    lines.append("- `09_battery_mean_contrasts.csv`")
    lines.append("- `10_scenario_mean_contrasts.csv`")
    lines.append("- `figures/`")
    lines.append("")
    lines.append("## Caution")
    lines.append("")
    lines.append("These are robustness checks on already-generated experimental contrasts. They do not prove a component is mechanistically sufficient; they test whether the observed contrast is stable under alternative summaries and exclusions.")
    lines.append("")

    (outdir / "00_ROBUSTNESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="~/Desktop/15_darca_density_sweep_N1_N3_N5_N10",
        help="Base directory containing N*_seeds*_steps* folders.",
    )
    ap.add_argument(
        "--summary-csv",
        default=None,
        help="Optional combined density summary CSV. Default: base-dir/02_density_sweep_intelligence_emergence_summary.csv",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: base-dir/91_robustness_analysis",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap iterations for seed-level paired contrasts.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap.",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve() if args.summary_csv else None
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else base_dir / "91_robustness_analysis"
    ensure_dir(outdir)

    rng = np.random.default_rng(args.seed)

    raw = load_raw_episode_summary(base_dir)
    scenario_summary = load_scenario_summary(base_dir)
    battery_summary = load_battery_summary(base_dir, summary_csv)

    if raw.empty and scenario_summary.empty and battery_summary.empty:
        raise SystemExit(f"No usable input files found under {base_dir}")

    seed_means = collapse_raw_to_seed_means(raw)
    pair_df = all_paired_seed_differences(seed_means)

    seed_boot = seed_level_bootstrap(pair_df, args.bootstrap, rng)
    battery_contrasts = mean_level_contrasts(battery_summary, "battery_mean", ALL_CONTRASTS)
    scenario_contrasts = mean_level_contrasts(scenario_summary, "scenario_mean", ALL_CONTRASTS)
    leave_density = leave_one_density_analysis(scenario_summary)
    leave_scenario = leave_one_scenario_analysis(scenario_summary)
    composites = alternative_safety_composites(pair_df, scenario_summary)
    outlier = outlier_robustness(pair_df)
    negative = negative_control_robustness(battery_contrasts, seed_boot)
    components = component_attribution_robustness(
        battery_contrasts=battery_contrasts,
        scenario_contrasts=scenario_contrasts,
        seed_boot=seed_boot,
        leave_density=leave_density,
        leave_scenario=leave_scenario,
    )

    # Save audit inputs.
    if not raw.empty and len(raw) <= 700_000:
        write_df(raw, outdir / "input_standardized_raw_episode_summary.csv")
    if not scenario_summary.empty:
        write_df(scenario_summary, outdir / "input_standardized_scenario_summary.csv")
    if not battery_summary.empty:
        write_df(battery_summary, outdir / "input_standardized_battery_summary.csv")
    if not seed_means.empty:
        write_df(seed_means, outdir / "input_seed_means.csv")

    # Save outputs.
    write_df(seed_boot, outdir / "01_seed_level_bootstrap.csv")
    write_df(leave_density, outdir / "02_leave_one_density.csv")
    write_df(leave_scenario, outdir / "03_leave_one_scenario.csv")
    write_df(composites, outdir / "04_alternative_safety_composites.csv")
    write_df(outlier, outdir / "05_outlier_robustness.csv")
    write_df(negative, outdir / "06_negative_control_robustness.csv")
    write_df(components, outdir / "07_component_attribution_robustness.csv")
    write_df(pair_df, outdir / "08_seed_level_pairwise_differences.csv")
    write_df(battery_contrasts, outdir / "09_battery_mean_contrasts.csv")
    write_df(scenario_contrasts, outdir / "10_scenario_mean_contrasts.csv")

    save_plots(outdir, seed_boot, components, leave_density)

    write_report(
        outdir=outdir,
        raw=raw,
        scenario_summary=scenario_summary,
        battery_summary=battery_summary,
        seed_boot=seed_boot,
        leave_density=leave_density,
        leave_scenario=leave_scenario,
        composites=composites,
        outlier=outlier,
        negative=negative,
        components=components,
    )

    print(f"[DONE] robustness analysis written to: {outdir}")
    print(f"[REPORT] {outdir / '00_ROBUSTNESS_REPORT.md'}")
    print(f"[CSV] {outdir / '01_seed_level_bootstrap.csv'}")
    print(f"[CSV] {outdir / '02_leave_one_density.csv'}")
    print(f"[CSV] {outdir / '03_leave_one_scenario.csv'}")
    print(f"[CSV] {outdir / '04_alternative_safety_composites.csv'}")
    print(f"[CSV] {outdir / '05_outlier_robustness.csv'}")
    print(f"[CSV] {outdir / '06_negative_control_robustness.csv'}")
    print(f"[CSV] {outdir / '07_component_attribution_robustness.csv'}")


if __name__ == "__main__":
    main()
