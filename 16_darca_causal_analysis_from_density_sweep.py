#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_darca_causal_analysis_from_density_sweep.py

Causal / lesion-contrast analysis for the DARCA density sweep.

Purpose
-------
This script does not change the task battery and does not rerun simulations.
It analyzes existing outputs from:

    ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10

The causal logic is experimental contrast analysis:

    FULL_INTEGRATED vs DARCA_ONLY
        -> effect of adding integrated embodiment layers to the DARCA core

    FULL_INTEGRATED vs NO_Q
        -> necessity / contribution of the Q body-valence layer

    FULL_INTEGRATED vs NO_MEMORY
        -> necessity / contribution of memory-like internal state

    FULL_INTEGRATED vs NO_PHYSICS
        -> necessity / contribution of physical-law / gravity layer

    FULL_INTEGRATED vs SHUFFLED_INTERNAL_STATE
        -> necessity / contribution of ordered internal state

    FULL_INTEGRATED vs FROZEN_INTERNAL_STATE
        -> necessity / contribution of active internal-state dynamics

    FULL_INTEGRATED vs RANDOM / REFLEX / GREEDY
        -> non-random competence and relation to simple baselines

Design constraints
------------------
- n_agents is treated as an experimental density factor, not as a sample-size knob.
- N1, N3, N5, N10 are analyzed separately and then compared.
- The script prefers scenario-level aggregate files if available.
- If raw episode summaries are available, it also computes seed-level paired bootstrap CIs.
- If only the combined 05b summary is available, it still performs battery-level contrasts.

Default input
-------------
    ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10

Default output
--------------
    ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10/90_causal_analysis

Run
---
    python3 -u 16_darca_causal_analysis_from_density_sweep.py

or

    python3 -u 16_darca_causal_analysis_from_density_sweep.py \
      --base-dir ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10 \
      --outdir ~/Desktop/15_darca_density_sweep_N1_N3_N5_N10/90_causal_analysis
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

HIGHER_IS_BETTER = {
    "survival_steps": True,
    "resource_gain_per_alive_step": True,
    "mean_q": False,  # lower q is interpreted as lower distress/risk load
    "mean_physics_score": True,
    "signal_rate": None,  # diagnostic only; no fixed good/bad direction
    "signal_entropy_bits": None,
    "damage_per_alive_step": False,
    "collision_per_alive_step": False,
    "fall_per_alive_step": False,
}

PRIMARY_METRICS = [
    "survival_steps",
    "resource_gain_per_alive_step",
    "damage_per_alive_step",
    "collision_per_alive_step",
    "fall_per_alive_step",
]

DIAGNOSTIC_METRICS = [
    "mean_q",
    "mean_physics_score",
    "signal_rate",
    "signal_entropy_bits",
]

CONTRASTS = [
    ("FULL_INTEGRATED", "DARCA_ONLY", "integrated_layers_vs_darca_core"),
    ("FULL_INTEGRATED", "RANDOM_POLICY", "full_vs_random_policy"),
    ("DARCA_ONLY", "RANDOM_POLICY", "darca_core_vs_random_policy"),
    ("FULL_INTEGRATED", "REFLEX_SAFE_POLICY", "full_vs_reflex_safe_policy"),
    ("FULL_INTEGRATED", "GREEDY_RESOURCE_POLICY", "full_vs_greedy_resource_policy"),
    ("FULL_INTEGRATED", "NO_Q", "q_layer_contribution"),
    ("FULL_INTEGRATED", "NO_MEMORY", "memory_contribution"),
    ("FULL_INTEGRATED", "NO_PHYSICS", "physics_layer_contribution"),
    ("FULL_INTEGRATED", "SHUFFLED_INTERNAL_STATE", "ordered_internal_state_contribution"),
    ("FULL_INTEGRATED", "FROZEN_INTERNAL_STATE", "active_internal_dynamics_contribution"),
]

SCENARIO_BATTERY_NAME = "_battery_mean"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_density_from_path(path: Path) -> Optional[int]:
    for part in [path.name, *[p.name for p in path.parents]]:
        m = re.search(r"(?:^|_)N(\d+)(?:_|$)", part)
        if m:
            return int(m.group(1))
    return None


def numeric_or_nan(x) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def metric_candidates(metric: str) -> List[str]:
    return [
        metric,
        f"mean_{metric}",
        f"across_scenario_mean_{metric}",
        f"mean_mean_{metric}",
    ]


def standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known aggregate/raw column names into canonical metric names.
    Keeps original columns as well.
    """
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

    if "scenario" not in out.columns:
        out["scenario"] = SCENARIO_BATTERY_NAME

    if "condition" in out.columns:
        out["condition"] = out["condition"].astype(str)

    return out


def signed_benefit(left_value: float, right_value: float, metric: str) -> float:
    """
    Positive means left condition is better than right condition for this metric.
    For lower-is-better metrics, benefit is right - left.
    """
    direction = HIGHER_IS_BETTER.get(metric, True)
    if direction is True:
        return left_value - right_value
    if direction is False:
        return right_value - left_value
    return left_value - right_value


def raw_difference(left_value: float, right_value: float) -> float:
    return left_value - right_value


def safe_round(x: float, nd: int = 6) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.{nd}g}"


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1 or n_boot <= 0:
        return mean, float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@dataclass
class LoadedData:
    scenario_summary: pd.DataFrame
    battery_summary: pd.DataFrame
    raw_episode_summary: pd.DataFrame


def load_scenario_summaries(base_dir: Path) -> pd.DataFrame:
    rows = []
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
    keep_metrics = PRIMARY_METRICS + DIAGNOSTIC_METRICS
    existing = ["n_agents", "condition", "scenario", "source_file"] + [m for m in keep_metrics if m in df.columns]
    return df[existing].copy()


def load_battery_summary(base_dir: Path, summary_csv: Optional[Path]) -> pd.DataFrame:
    candidates = []
    if summary_csv:
        candidates.append(summary_csv)
    candidates.append(base_dir / "02_density_sweep_intelligence_emergence_summary.csv")

    rows = []
    loaded_any_combined = False
    for p in candidates:
        if p and p.exists():
            try:
                df = pd.read_csv(p)
                df["source_file"] = str(p)
                rows.append(df)
                loaded_any_combined = True
                break
            except Exception as e:
                print(f"[WARN] cannot read {p}: {e}", file=sys.stderr)

    if not loaded_any_combined:
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
    df["scenario"] = SCENARIO_BATTERY_NAME
    keep_metrics = PRIMARY_METRICS + DIAGNOSTIC_METRICS
    extra = [
        "n_scenarios",
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
        "source_file",
    ]
    existing = ["n_agents", "condition", "scenario"] + [m for m in keep_metrics if m in df.columns] + [c for c in extra if c in df.columns]
    return df[existing].copy()


def load_raw_episode_summaries(base_dir: Path) -> pd.DataFrame:
    rows = []
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

    # Detect seed column and scenario/condition
    if "seed" not in df.columns:
        for c in ["base_seed", "episode_seed", "replicate", "rep"]:
            if c in df.columns:
                df["seed"] = df[c]
                break
    if "seed" not in df.columns:
        df["seed"] = np.arange(len(df))

    if "scenario" not in df.columns:
        df["scenario"] = SCENARIO_BATTERY_NAME

    keep = ["n_agents", "condition", "scenario", "seed", "source_file"]
    for c in PRIMARY_METRICS + DIAGNOSTIC_METRICS:
        if c in df.columns:
            keep.append(c)
    return df[keep].copy()


def load_all(base_dir: Path, summary_csv: Optional[Path]) -> LoadedData:
    scenario = load_scenario_summaries(base_dir)
    battery = load_battery_summary(base_dir, summary_csv)
    raw = load_raw_episode_summaries(base_dir)
    return LoadedData(scenario_summary=scenario, battery_summary=battery, raw_episode_summary=raw)


# ---------------------------------------------------------------------
# Contrast calculations
# ---------------------------------------------------------------------

def contrast_table_from_means(
    df: pd.DataFrame,
    level_name: str,
    include_scenario: bool = True,
) -> pd.DataFrame:
    """
    Contrast table from mean-level data.
    Each row compares left vs right at a given density and scenario/battery.
    """
    if df.empty:
        return pd.DataFrame()

    metrics = [m for m in PRIMARY_METRICS + DIAGNOSTIC_METRICS if m in df.columns]
    keys = ["n_agents"]
    if include_scenario and "scenario" in df.columns:
        keys.append("scenario")

    rows = []
    for key_values, g in df.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        key_dict = dict(zip(keys, key_values))
        by_condition = {str(r["condition"]): r for _, r in g.iterrows() if "condition" in r}
        for left, right, label in CONTRASTS:
            if left not in by_condition or right not in by_condition:
                continue
            lrow = by_condition[left]
            rrow = by_condition[right]
            out = {
                "analysis_level": level_name,
                **key_dict,
                "contrast": label,
                "left_condition": left,
                "right_condition": right,
            }
            positive_count = 0
            available_count = 0
            safety_positive_count = 0
            safety_available_count = 0
            for m in metrics:
                lv = numeric_or_nan(lrow.get(m))
                rv = numeric_or_nan(rrow.get(m))
                if not np.isfinite(lv) or not np.isfinite(rv):
                    continue
                raw = raw_difference(lv, rv)
                benefit = signed_benefit(lv, rv, m)
                out[f"{m}_left"] = lv
                out[f"{m}_right"] = rv
                out[f"{m}_raw_diff_left_minus_right"] = raw
                out[f"{m}_signed_benefit_left_over_right"] = benefit
                if HIGHER_IS_BETTER.get(m) is not None:
                    available_count += 1
                    if benefit > 0:
                        positive_count += 1
                if m in ["damage_per_alive_step", "collision_per_alive_step", "fall_per_alive_step"]:
                    safety_available_count += 1
                    if benefit > 0:
                        safety_positive_count += 1
            out["positive_metric_count"] = positive_count
            out["available_directed_metric_count"] = available_count
            out["safety_positive_metric_count"] = safety_positive_count
            out["safety_available_metric_count"] = safety_available_count
            # unsafe survival extension: left lives longer but has worse damage/collision/fall.
            surv_b = out.get("survival_steps_signed_benefit_left_over_right", np.nan)
            safety_benefits = [
                out.get("damage_per_alive_step_signed_benefit_left_over_right", np.nan),
                out.get("collision_per_alive_step_signed_benefit_left_over_right", np.nan),
                out.get("fall_per_alive_step_signed_benefit_left_over_right", np.nan),
            ]
            worse_safety = [b for b in safety_benefits if np.isfinite(b) and b < 0]
            out["unsafe_survival_extension_flag"] = int(np.isfinite(surv_b) and surv_b > 0 and len(worse_safety) >= 2)
            rows.append(out)

    return pd.DataFrame(rows)


def seed_level_paired_contrasts(raw: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    If raw episode data are available, aggregate agents within seed and scenario,
    then compute paired seed-level contrasts.
    """
    if raw.empty:
        return pd.DataFrame()

    metrics = [m for m in PRIMARY_METRICS + DIAGNOSTIC_METRICS if m in raw.columns]
    if not metrics:
        return pd.DataFrame()

    work = raw.copy()
    work = coerce_numeric(work, ["n_agents", "seed"] + metrics)
    # Agent rows within the same seed are not independent. Collapse them first.
    seed_mean = (
        work.groupby(["n_agents", "scenario", "condition", "seed"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )

    rows = []
    for (n_agents, scenario), g in seed_mean.groupby(["n_agents", "scenario"], dropna=False):
        for left, right, label in CONTRASTS:
            l = g[g["condition"] == left]
            r = g[g["condition"] == right]
            if l.empty or r.empty:
                continue
            merged = l.merge(r, on=["n_agents", "scenario", "seed"], suffixes=("_left", "_right"))
            if merged.empty:
                continue
            base = {
                "analysis_level": "seed_paired_raw_episode",
                "n_agents": n_agents,
                "scenario": scenario,
                "contrast": label,
                "left_condition": left,
                "right_condition": right,
                "n_paired_seeds": int(merged["seed"].nunique()),
            }
            for m in metrics:
                lv = pd.to_numeric(merged[f"{m}_left"], errors="coerce").to_numpy()
                rv = pd.to_numeric(merged[f"{m}_right"], errors="coerce").to_numpy()
                raw_diff = lv - rv
                signed = np.array([signed_benefit(a, b, m) for a, b in zip(lv, rv)], dtype=float)
                raw_mean, raw_lo, raw_hi = bootstrap_ci(raw_diff, n_boot, rng)
                ben_mean, ben_lo, ben_hi = bootstrap_ci(signed, n_boot, rng)
                base[f"{m}_raw_diff_mean"] = raw_mean
                base[f"{m}_raw_diff_ci95_low"] = raw_lo
                base[f"{m}_raw_diff_ci95_high"] = raw_hi
                base[f"{m}_signed_benefit_mean"] = ben_mean
                base[f"{m}_signed_benefit_ci95_low"] = ben_lo
                base[f"{m}_signed_benefit_ci95_high"] = ben_hi
            rows.append(dict(base))

    return pd.DataFrame(rows)


def summarize_contrasts(contrast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize contrast consistency across densities and scenarios.
    """
    if contrast_df.empty:
        return pd.DataFrame()

    metric_cols = [f"{m}_signed_benefit_left_over_right" for m in PRIMARY_METRICS if f"{m}_signed_benefit_left_over_right" in contrast_df.columns]
    rows = []
    for contrast, g in contrast_df.groupby("contrast", dropna=False):
        row = {"contrast": contrast}
        row["n_rows"] = len(g)
        row["densities"] = ";".join(str(int(x)) for x in sorted(pd.to_numeric(g["n_agents"], errors="coerce").dropna().unique()))
        row["scenarios"] = ";".join(str(x) for x in sorted(g["scenario"].dropna().astype(str).unique())) if "scenario" in g else ""
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            metric = col.replace("_signed_benefit_left_over_right", "")
            if len(vals) == 0:
                continue
            row[f"{metric}_mean_signed_benefit"] = float(vals.mean())
            row[f"{metric}_median_signed_benefit"] = float(vals.median())
            row[f"{metric}_positive_fraction"] = float((vals > 0).mean())
            row[f"{metric}_negative_fraction"] = float((vals < 0).mean())
        if "positive_metric_count" in g.columns and "available_directed_metric_count" in g.columns:
            denom = pd.to_numeric(g["available_directed_metric_count"], errors="coerce").replace(0, np.nan)
            frac = pd.to_numeric(g["positive_metric_count"], errors="coerce") / denom
            row["mean_positive_metric_fraction"] = float(frac.mean(skipna=True))
        if "safety_positive_metric_count" in g.columns:
            denom2 = pd.to_numeric(g["safety_available_metric_count"], errors="coerce").replace(0, np.nan)
            frac2 = pd.to_numeric(g["safety_positive_metric_count"], errors="coerce") / denom2
            row["mean_safety_positive_fraction"] = float(frac2.mean(skipna=True))
        rows.append(row)

    return pd.DataFrame(rows)


def density_trends(contrast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate simple density trends for each contrast and metric.
    Slope is not a causal estimator by itself; it is a density-dependence diagnostic.
    """
    if contrast_df.empty:
        return pd.DataFrame()

    rows = []
    metrics = [m for m in PRIMARY_METRICS if f"{m}_signed_benefit_left_over_right" in contrast_df.columns]
    for (contrast, scenario), g in contrast_df.groupby(["contrast", "scenario"], dropna=False):
        x = pd.to_numeric(g["n_agents"], errors="coerce").to_numpy(dtype=float)
        for m in metrics:
            y = pd.to_numeric(g[f"{m}_signed_benefit_left_over_right"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2:
                continue
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            corr = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() >= 3 else float("nan")
            rows.append({
                "contrast": contrast,
                "scenario": scenario,
                "metric": m,
                "density_slope_per_agent_signed_benefit": float(slope),
                "density_intercept": float(intercept),
                "density_correlation": float(corr),
                "n_density_points": int(mask.sum()),
                "min_density": float(np.min(x[mask])),
                "max_density": float(np.max(x[mask])),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting and report
# ---------------------------------------------------------------------

def save_density_plots(contrast_df: pd.DataFrame, outdir: Path) -> None:
    if not HAS_MATPLOTLIB or contrast_df.empty:
        return

    plot_dir = outdir / "figures"
    ensure_dir(plot_dir)

    selected_contrasts = [
        "integrated_layers_vs_darca_core",
        "q_layer_contribution",
        "memory_contribution",
        "physics_layer_contribution",
        "active_internal_dynamics_contribution",
    ]

    for contrast in selected_contrasts:
        g = contrast_df[(contrast_df["contrast"] == contrast) & (contrast_df["scenario"] == SCENARIO_BATTERY_NAME)]
        if g.empty:
            continue
        for metric in PRIMARY_METRICS:
            col = f"{metric}_signed_benefit_left_over_right"
            if col not in g.columns:
                continue
            gg = g.sort_values("n_agents")
            x = pd.to_numeric(gg["n_agents"], errors="coerce")
            y = pd.to_numeric(gg[col], errors="coerce")
            if y.notna().sum() < 2:
                continue
            plt.figure(figsize=(7, 4.5))
            plt.plot(x, y, marker="o")
            plt.axhline(0, linewidth=1)
            plt.xlabel("n_agents")
            plt.ylabel(f"{metric}: signed benefit")
            plt.title(f"{contrast}: {metric}")
            plt.tight_layout()
            safe_name = f"{contrast}__{metric}.png".replace("/", "_")
            plt.savefig(plot_dir / safe_name, dpi=180)
            plt.close()


def write_csv_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def strongest_rows_text(summary: pd.DataFrame, n: int = 8) -> str:
    if summary.empty:
        return "No contrast summary was available."

    cols = [
        "contrast",
        "mean_positive_metric_fraction",
        "mean_safety_positive_fraction",
        "survival_steps_mean_signed_benefit",
        "resource_gain_per_alive_step_mean_signed_benefit",
        "damage_per_alive_step_mean_signed_benefit",
        "collision_per_alive_step_mean_signed_benefit",
        "fall_per_alive_step_mean_signed_benefit",
    ]
    cols = [c for c in cols if c in summary.columns]
    if not cols:
        return "No standard contrast columns were available."

    df = summary.copy()
    sort_col = "mean_safety_positive_fraction" if "mean_safety_positive_fraction" in df.columns else cols[-1]
    df = df.sort_values(sort_col, ascending=False).head(n)
    lines = []
    for _, r in df.iterrows():
        items = [f"{c}={safe_round(numeric_or_nan(r[c]), 4) if c != 'contrast' else r[c]}" for c in cols]
        lines.append("- " + "; ".join(items))
    return "\n".join(lines)


def interpret_key_results(battery_contrasts: pd.DataFrame, contrast_summary: pd.DataFrame) -> List[str]:
    lines = []

    def get_battery(contrast: str) -> pd.DataFrame:
        return battery_contrasts[
            (battery_contrasts["contrast"] == contrast)
            & (battery_contrasts["scenario"] == SCENARIO_BATTERY_NAME)
        ].copy()

    full_vs_darca = get_battery("integrated_layers_vs_darca_core")
    if not full_vs_darca.empty:
        safety_cols = [
            "damage_per_alive_step_signed_benefit_left_over_right",
            "collision_per_alive_step_signed_benefit_left_over_right",
            "fall_per_alive_step_signed_benefit_left_over_right",
        ]
        all_safety_positive_by_density = []
        for _, r in full_vs_darca.iterrows():
            vals = [numeric_or_nan(r.get(c)) for c in safety_cols]
            if all(np.isfinite(v) and v > 0 for v in vals):
                all_safety_positive_by_density.append(int(r["n_agents"]))
        if all_safety_positive_by_density:
            lines.append(
                "FULL_INTEGRATED improved all three safety-rate metrics relative to DARCA_ONLY "
                f"at densities: {', '.join('N'+str(x) for x in sorted(all_safety_positive_by_density))}."
            )
        else:
            lines.append(
                "FULL_INTEGRATED did not consistently improve all safety-rate metrics relative to DARCA_ONLY."
            )

    for contrast, label in [
        ("q_layer_contribution", "Q layer"),
        ("memory_contribution", "memory"),
        ("physics_layer_contribution", "physics layer"),
        ("ordered_internal_state_contribution", "ordered internal state"),
        ("active_internal_dynamics_contribution", "active internal dynamics"),
    ]:
        g = get_battery(contrast)
        if g.empty:
            continue
        safety_cols = [
            "damage_per_alive_step_signed_benefit_left_over_right",
            "collision_per_alive_step_signed_benefit_left_over_right",
            "fall_per_alive_step_signed_benefit_left_over_right",
        ]
        vals = []
        for c in safety_cols:
            vals.extend(pd.to_numeric(g[c], errors="coerce").dropna().tolist() if c in g.columns else [])
        if vals:
            mean_abs = float(np.mean(np.abs(vals)))
            mean_signed = float(np.mean(vals))
            lines.append(
                f"{label}: mean signed safety benefit of FULL over lesion/control = "
                f"{mean_signed:.6g}; mean absolute safety contrast = {mean_abs:.6g}."
            )

    if not contrast_summary.empty:
        rows = contrast_summary[contrast_summary["contrast"].isin([
            "q_layer_contribution",
            "memory_contribution",
            "physics_layer_contribution",
            "ordered_internal_state_contribution",
            "active_internal_dynamics_contribution",
        ])].copy()
        if not rows.empty and "mean_safety_positive_fraction" in rows.columns:
            weak = rows[pd.to_numeric(rows["mean_safety_positive_fraction"], errors="coerce") < 0.67]
            if not weak.empty:
                lines.append(
                    "Some component-lesion contrasts were weak or mixed; this means the FULL advantage should not yet be attributed to a single internal component without further targeted intervention."
                )

    return lines


def write_report(
    outdir: Path,
    data: LoadedData,
    scenario_contrasts: pd.DataFrame,
    battery_contrasts: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    trend_df: pd.DataFrame,
    seed_contrasts: pd.DataFrame,
) -> None:
    report = outdir / "00_CAUSAL_ANALYSIS_REPORT.md"
    lines: List[str] = []
    lines.append("# DARCA density-sweep causal analysis report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This analysis does not modify the validation tasks. It analyzes existing N1, N3, N5, and N10 outputs as experimental density conditions.")
    lines.append("")
    lines.append("The causal interpretation is based on pre-existing interventions: DARCA_ONLY, FULL_INTEGRATED, NO_Q, NO_MEMORY, NO_PHYSICS, SHUFFLED_INTERNAL_STATE, and FROZEN_INTERNAL_STATE.")
    lines.append("")
    lines.append("Positive signed benefit means the left condition is better than the right condition. For damage, collision, and fall rates, lower raw values are treated as better.")
    lines.append("")
    lines.append("## Data loaded")
    lines.append("")
    lines.append(f"- Scenario-level rows: {len(data.scenario_summary)}")
    lines.append(f"- Battery-level rows: {len(data.battery_summary)}")
    lines.append(f"- Raw episode rows: {len(data.raw_episode_summary)}")
    lines.append("")
    if not data.battery_summary.empty:
        dens = sorted(pd.to_numeric(data.battery_summary["n_agents"], errors="coerce").dropna().astype(int).unique())
        lines.append(f"- Densities: {', '.join('N'+str(x) for x in dens)}")
        conds = sorted(data.battery_summary["condition"].dropna().astype(str).unique())
        lines.append(f"- Conditions: {', '.join(conds)}")
    lines.append("")
    lines.append("## Key causal contrasts")
    lines.append("")
    lines.append(strongest_rows_text(contrast_summary, n=12))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    interp = interpret_key_results(battery_contrasts, contrast_summary)
    if interp:
        for x in interp:
            lines.append(f"- {x}")
    else:
        lines.append("- No automatic interpretation was possible because the expected contrasts were not available.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `01_battery_level_causal_contrasts.csv`")
    lines.append("- `02_scenario_level_causal_contrasts.csv`")
    lines.append("- `03_causal_contrast_summary.csv`")
    lines.append("- `04_density_trend_diagnostics.csv`")
    lines.append("- `05_seed_paired_raw_episode_contrasts.csv` if raw episode files were available")
    lines.append("- `figures/` if matplotlib was available")
    lines.append("")
    lines.append("## Caution")
    lines.append("")
    lines.append("Lesion contrasts identify whether a component is necessary under the implemented intervention. Similar performance between FULL_INTEGRATED and a lesion does not prove the component has no mechanistic role; it means the present behavioral metrics do not require that component under this task battery.")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="~/Desktop/15_darca_density_sweep_N1_N3_N5_N10",
        help="Base directory containing N*_seeds*_steps* result folders.",
    )
    ap.add_argument(
        "--summary-csv",
        default=None,
        help="Optional combined 02_density_sweep_intelligence_emergence_summary.csv. If omitted, the script searches base-dir.",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: base-dir/90_causal_analysis",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap iterations for raw episode seed-level paired contrasts when raw data are available.",
    )
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for bootstrap.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    summary_csv = Path(args.summary_csv).expanduser().resolve() if args.summary_csv else None
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else base_dir / "90_causal_analysis"
    ensure_dir(outdir)

    rng = np.random.default_rng(args.seed)

    data = load_all(base_dir, summary_csv)

    if data.scenario_summary.empty and data.battery_summary.empty and data.raw_episode_summary.empty:
        raise SystemExit(
            f"No readable density-sweep outputs found under {base_dir}. "
            "Check --base-dir or pass --summary-csv."
        )

    # Write standardized inputs for audit.
    if not data.scenario_summary.empty:
        write_csv_df(data.scenario_summary, outdir / "input_standardized_scenario_summary.csv")
    if not data.battery_summary.empty:
        write_csv_df(data.battery_summary, outdir / "input_standardized_battery_summary.csv")
    if not data.raw_episode_summary.empty:
        # Avoid extremely large audit file unless manageable.
        if len(data.raw_episode_summary) <= 500_000:
            write_csv_df(data.raw_episode_summary, outdir / "input_standardized_raw_episode_summary.csv")

    scenario_contrasts = contrast_table_from_means(data.scenario_summary, "scenario_mean", include_scenario=True)
    battery_contrasts = contrast_table_from_means(data.battery_summary, "battery_mean", include_scenario=True)

    all_contrasts_for_summary = pd.concat(
        [df for df in [scenario_contrasts, battery_contrasts] if not df.empty],
        ignore_index=True,
    ) if (not scenario_contrasts.empty or not battery_contrasts.empty) else pd.DataFrame()

    contrast_summary = summarize_contrasts(all_contrasts_for_summary)
    trend_df = density_trends(all_contrasts_for_summary)

    seed_contrasts = seed_level_paired_contrasts(data.raw_episode_summary, args.bootstrap, rng)

    write_csv_df(battery_contrasts, outdir / "01_battery_level_causal_contrasts.csv")
    write_csv_df(scenario_contrasts, outdir / "02_scenario_level_causal_contrasts.csv")
    write_csv_df(contrast_summary, outdir / "03_causal_contrast_summary.csv")
    write_csv_df(trend_df, outdir / "04_density_trend_diagnostics.csv")
    write_csv_df(seed_contrasts, outdir / "05_seed_paired_raw_episode_contrasts.csv")

    save_density_plots(battery_contrasts, outdir)

    write_report(
        outdir=outdir,
        data=data,
        scenario_contrasts=scenario_contrasts,
        battery_contrasts=battery_contrasts,
        contrast_summary=contrast_summary,
        trend_df=trend_df,
        seed_contrasts=seed_contrasts,
    )

    print(f"[DONE] causal analysis written to: {outdir}")
    print(f"[REPORT] {outdir / '00_CAUSAL_ANALYSIS_REPORT.md'}")
    print(f"[CSV] {outdir / '01_battery_level_causal_contrasts.csv'}")
    print(f"[CSV] {outdir / '02_scenario_level_causal_contrasts.csv'}")
    print(f"[CSV] {outdir / '03_causal_contrast_summary.csv'}")
    print(f"[CSV] {outdir / '04_density_trend_diagnostics.csv'}")
    if not seed_contrasts.empty:
        print(f"[CSV] {outdir / '05_seed_paired_raw_episode_contrasts.csv'}")


if __name__ == "__main__":
    main()
