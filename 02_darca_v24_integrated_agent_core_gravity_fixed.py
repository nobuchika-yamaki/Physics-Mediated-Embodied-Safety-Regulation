#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
darca_v24_integrated_agent_core_v2.py
====================================

Independent agent core for the MuJoCo viewer series.

This file separates the AGENT from the ENVIRONMENT / MuJoCo viewer.
The agent is the integrated model used in the latest viewer workflow:

    fixed DARCA v24 core
    + gravity / physical-law understanding layer
    + qualitative valence (Q / qualia-style body-state layer)
    + Phase-4c style anonymous social signal layer

Important boundary:
    - This file does not create terrain, bridges, rivers, enemies, walls, cameras,
      XML, sounds, or MuJoCo bodies.
    - It accepts an observation vector from any environment and returns an action
      intent plus diagnostic variables.
    - It does not use LMM, API, language prompts, LAUGH/HUMOR labels, or a
      benign-violation controller variable.

Typical use:

    from darca_v24_integrated_agent_core_v2 import IntegratedDARCAAgent, AgentObservation

    agent = IntegratedDARCAAgent(darca_file="./darca_v24_.py", seed=1)
    obs = AgentObservation(x=0.0, y=0.0, resource_pressure=0.5, danger_pressure=0.1)
    action = agent.step(obs)
    print(action.desired_vx, action.desired_vy, action.signal_channel)

Run standalone smoke test:

    python3 -u darca_v24_integrated_agent_core_v2.py \
        --darca-file ./darca_v24_.py \
        --steps 500 \
        --outdir ~/Desktop/darca_integrated_agent_smoke
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# General utilities
# =============================================================================


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, float(x))))


def clip01(x: float) -> float:
    return clip(x, 0.0, 1.0)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def norm2(x: float, y: float, eps: float = 1e-9) -> Tuple[float, float, float]:
    n = math.sqrt(float(x) * float(x) + float(y) * float(y))
    if n < eps:
        return 0.0, 0.0, 0.0
    return float(x) / n, float(y) / n, n


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============================================================================
# External interface
# =============================================================================

SIGNAL_LABELS = {
    -1: "NONE",
    0: "RESOURCE",
    1: "DANGER",
    2: "WALL_BLOCK",
    3: "TRAP_SURFACE",
    4: "PURSUER",
}


@dataclass
class AgentObservation:
    """Environment-to-agent observation.

    The viewer/environment should fill these fields. Missing fields default to zero.
    The agent does not know about MuJoCo XML, bridges, cameras, or object layout.
    """

    # time and body state
    step: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    heading_x: float = 1.0
    heading_y: float = 0.0
    grounded: float = 1.0
    slope: float = 0.0
    height: float = 0.0
    vertical_drop: float = 0.0
    # Directional gravity affordances supplied by the environment.
    # fall_dx/fall_dy point toward the locally more dangerous downhill/drop direction.
    # uphill_dx/uphill_dy point toward the locally safer uphill direction.
    fall_dx: float = 0.0
    fall_dy: float = 0.0
    uphill_dx: float = 0.0
    uphill_dy: float = 0.0

    # local environmental pressures, all 0..1 unless noted
    resource_pressure: float = 0.0
    danger_pressure: float = 0.0
    trap_pressure: float = 0.0
    wall_pressure: float = 0.0
    pursuer_pressure: float = 0.0
    friction_pressure: float = 0.0
    current_pressure: float = 0.0
    novelty: float = 0.0
    social_density: float = 0.0

    # directional affordances. These are optional hints from the environment.
    resource_dx: float = 0.0
    resource_dy: float = 0.0
    danger_dx: float = 0.0
    danger_dy: float = 0.0
    trap_dx: float = 0.0
    trap_dy: float = 0.0
    wall_dx: float = 0.0
    wall_dy: float = 0.0
    pursuer_dx: float = 0.0
    pursuer_dy: float = 0.0
    current_dx: float = 0.0
    current_dy: float = 0.0
    target_hint_dx: float = 0.0
    target_hint_dy: float = 0.0

    # event and outcome flags supplied by the environment
    damage: float = 0.0
    resource_gain: float = 0.0
    recovery_gain: float = 0.0
    hit_wall: float = 0.0
    hit_danger: float = 0.0
    hit_trap: float = 0.0
    hit_pursuer: float = 0.0
    can_jump: float = 0.0

    # optional external scalar sensory stream, if caller already computed it
    scalar_y: Optional[float] = None

    def clipped(self) -> "AgentObservation":
        d = asdict(self)
        for k in [
            "grounded", "resource_pressure", "danger_pressure", "trap_pressure", "wall_pressure",
            "pursuer_pressure", "friction_pressure", "current_pressure", "novelty", "social_density",
            "damage", "resource_gain", "recovery_gain", "hit_wall", "hit_danger", "hit_trap",
            "hit_pursuer", "can_jump",
        ]:
            d[k] = clip01(float(d[k]))
        return AgentObservation(**d)


@dataclass
class AgentAction:
    """Agent-to-environment output."""

    desired_vx: float = 0.0
    desired_vy: float = 0.0
    speed_scale: float = 1.0
    wants_jump: float = 0.0
    wants_scan: float = 0.0
    wants_rest: float = 0.0
    wants_regulate: float = 0.0

    signal_channel: int = -1
    signal_label: str = "NONE"
    signal_strength: float = 0.0

    # main diagnostics used by viewers / logs
    q: float = 0.0
    life_h: float = 0.0
    autonomy: float = 0.0
    identity: float = 0.0
    darca_action_name: str = "NA"
    darca_u: float = 0.0
    scalar_y: float = 0.0
    gravity_fall_risk: float = 0.0
    physics_score: float = 0.0
    self_appraisal_gap: float = 0.0
    relief: float = 0.0
    safe_surprise: float = 0.0
    receiver_recovery: float = 0.0
    n_heard: float = 0.0
    social_spread: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        out = asdict(self)
        dbg = out.pop("debug", {}) or {}
        for k, v in dbg.items():
            if k not in out:
                out[k] = v
        return out


@dataclass
class NeighborState:
    """Minimal neighbor state for the Phase-4c listener layer."""

    x: float
    y: float
    listener_value: Optional[np.ndarray] = None
    social_tension: float = 0.2
    social_sync: float = 0.45
    exploration_drive: float = 0.60


# =============================================================================
# DARCA v24 wrapper
# =============================================================================


class DarcaV24Wrapper:
    """Thin loader around the original DARCA v24 Agent.

    This wrapper does not rewrite the DARCA core. It imports Agent, Params, and
    Condition from the supplied file and calls Agent.step(y, env_info).
    """

    def __init__(self, darca_file: str, seed: int = 0, theta: Optional[float] = None):
        self.darca_file = str(Path(darca_file).expanduser())
        self.module = self._load_module(self.darca_file)
        Params = getattr(self.module, "Params")
        Condition = getattr(self.module, "Condition")
        Agent = getattr(self.module, "Agent")
        params = Params() if theta is None else Params(theta=float(theta))
        condition = Condition("Full")
        self.agent = Agent(params, condition, int(seed))

    @staticmethod
    def _load_module(path: str) -> Any:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"DARCA file not found: {p}")
        name = "_darca_v24_loaded_core"
        spec = importlib.util.spec_from_file_location(name, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DARCA module from {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def step(self, scalar_y: float, external_shock: float) -> Dict[str, Any]:
        env_info = {
            "external_shock": clip01(float(external_shock)),
            "d_dyn": 0.0,
            "coupling_t": 0.0,
            "sigma_t": 0.0,
        }
        return dict(self.agent.step(float(scalar_y), env_info))


# =============================================================================
# Gravity / physical-law understanding layer
# =============================================================================


class OnlineActionOutcomePredictor:
    """Small online predictor for action consequences.

    It is intentionally generic: it learns per-action expected damage, gain,
    wall/block, fall, and slope cost. It is independent from the environment.
    """

    ACTIONS = ["REST", "SCAN", "MOVE", "AVOID", "APPROACH", "JUMP"]

    def __init__(self, lesion: bool = False):
        self.lesion = bool(lesion)
        self.stats: Dict[str, Dict[str, float]] = {
            a: {"n": 1e-6, "damage": 0.0, "gain": 0.0, "wall": 0.0, "fall": 0.0, "surface": 0.0, "pred_err": 0.0}
            for a in self.ACTIONS
        }
        self.global_pred_error = 0.08
        self.score = 0.0

    def predict(self, action: str) -> Dict[str, float]:
        st = self.stats.get(action, self.stats["MOVE"])
        return {
            "pred_damage": float(st["damage"]),
            "pred_gain": float(st["gain"]),
            "pred_wall": float(st["wall"]),
            "pred_fall": float(st["fall"]),
            "pred_surface": float(st["surface"]),
        }

    def update(self, action: str, obs: AgentObservation, fall_risk: float) -> Dict[str, float]:
        action = action if action in self.stats else "MOVE"
        pred = self.predict(action)
        actual_damage = clip01(float(obs.damage) + 0.12 * float(obs.hit_danger) + 0.14 * float(obs.hit_pursuer) + 0.04 * float(obs.hit_trap))
        actual_gain = clip01(float(obs.resource_gain) + float(obs.recovery_gain))
        actual_wall = clip01(float(obs.hit_wall) + float(obs.wall_pressure))
        actual_fall = clip01(fall_risk + max(0.0, -float(obs.vz)) * (1.0 - float(obs.grounded)))
        actual_surface = clip01(float(obs.trap_pressure) + float(obs.friction_pressure) + float(obs.current_pressure))
        err = (
            abs(pred["pred_damage"] - actual_damage)
            + 0.5 * abs(pred["pred_gain"] - actual_gain)
            + 0.5 * abs(pred["pred_wall"] - actual_wall)
            + 0.5 * abs(pred["pred_fall"] - actual_fall)
            + 0.3 * abs(pred["pred_surface"] - actual_surface)
        )
        if self.lesion:
            self.global_pred_error = 0.98 * self.global_pred_error + 0.02 * err
            return {"physics_pred_error": err, "physics_score": 0.0, "physics_lesion": 1}
        st = self.stats[action]
        alpha = 0.08
        st["n"] += 1.0
        st["damage"] = (1 - alpha) * st["damage"] + alpha * actual_damage
        st["gain"] = (1 - alpha) * st["gain"] + alpha * actual_gain
        st["wall"] = (1 - alpha) * st["wall"] + alpha * actual_wall
        st["fall"] = (1 - alpha) * st["fall"] + alpha * actual_fall
        st["surface"] = (1 - alpha) * st["surface"] + alpha * actual_surface
        st["pred_err"] = (1 - alpha) * st["pred_err"] + alpha * err
        self.global_pred_error = 0.98 * self.global_pred_error + 0.02 * err
        self.score = clip(1.0 - self.global_pred_error / 0.22, 0.0, 1.0)
        return {"physics_pred_error": err, "physics_score": self.score, "physics_lesion": 0}


class GravityUnderstandingLayer:
    """Gravity-bound physical relation tracker.

    The viewer can still move the body however it wants; this layer only estimates
    fall risk, jump affordance, and action-consequence prediction from body-state
    observations. It does not directly move the agent.
    """

    def __init__(self):
        self.prev_vz = 0.0
        self.prev_z = 0.0
        self.g_est = 0.0
        self.g_conf = 0.0
        self.predictor = OnlineActionOutcomePredictor(lesion=False)

    def step(self, obs: AgentObservation, action_hint: str) -> Dict[str, float]:
        dz = float(obs.z) - self.prev_z
        dvz = float(obs.vz) - self.prev_vz
        if float(obs.grounded) < 0.5:
            # Negative vertical acceleration while airborne is evidence for gravity.
            sample = max(0.0, -dvz)
            self.g_est = 0.98 * self.g_est + 0.02 * sample
            self.g_conf = clip01(0.995 * self.g_conf + 0.04 * min(1.0, sample / 0.20))
        else:
            self.g_conf = clip01(0.997 * self.g_conf)
        self.prev_vz = float(obs.vz)
        self.prev_z = float(obs.z)

        fall_risk = clip01(
            0.42 * max(0.0, -float(obs.vz))
            + 0.20 * max(0.0, float(obs.height) - 0.7)
            + 0.18 * float(obs.vertical_drop)
            + 0.12 * float(obs.slope)
            + 0.22 * (1.0 - float(obs.grounded)) * self.g_conf
        )
        jump_affordance = clip01(float(obs.can_jump) * (0.25 + 0.75 * self.g_conf) * (1.0 - 0.55 * fall_risk))
        fall_dx, fall_dy, fall_vec_mag = norm2(float(getattr(obs, "fall_dx", 0.0)), float(getattr(obs, "fall_dy", 0.0)))
        uphill_dx, uphill_dy, uphill_vec_mag = norm2(float(getattr(obs, "uphill_dx", 0.0)), float(getattr(obs, "uphill_dy", 0.0)))
        pred = self.predictor.update(action_hint, obs, fall_risk)
        return {
            "gravity_estimate": float(self.g_est),
            "gravity_confidence": float(self.g_conf),
            "gravity_fall_risk": float(fall_risk),
            "jump_affordance": float(jump_affordance),
            "gravity_fall_dx": float(fall_dx),
            "gravity_fall_dy": float(fall_dy),
            "gravity_fall_vector_mag": float(fall_vec_mag),
            "gravity_uphill_dx": float(uphill_dx),
            "gravity_uphill_dy": float(uphill_dy),
            "gravity_uphill_vector_mag": float(uphill_vec_mag),
            **pred,
        }


# =============================================================================
# Qualitative valence / Q layer
# =============================================================================


@dataclass
class QualiaState:
    integrity: float = 0.96
    energy: float = 0.94
    fatigue: float = 0.10
    stability: float = 0.84
    pain_memory: float = 0.02
    danger_memory: float = 0.06
    comfort_memory: float = 0.52
    action_possibility: float = 0.55
    agency_lag_memory: float = 0.0
    q: float = 0.0


class QualitativeValenceLayer:
    """Q / qualia-style body-state layer.

    This layer keeps the variables that were used as qualitative valence support:
    body-relative danger, comfort, pain, memory, action possibility, and agency-like
    delay attribution. It does not replace DARCA's internal viability; it supplies
    additional body-state valuation and diagnostics.
    """

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.state = QualiaState(
            integrity=float(rng.uniform(0.92, 1.00)),
            energy=float(rng.uniform(0.88, 1.00)),
            fatigue=float(rng.uniform(0.05, 0.15)),
            stability=float(rng.uniform(0.74, 0.92)),
        )
        self.prev_action_intent = np.zeros(2, dtype=float)
        self.prev_motion = np.zeros(2, dtype=float)

    def step(self, obs: AgentObservation, darca_out: Dict[str, Any], gravity: Dict[str, float]) -> Dict[str, float]:
        s = self.state
        damage = clip01(float(obs.damage) + 0.08 * float(obs.hit_danger) + 0.12 * float(obs.hit_pursuer) + 0.035 * float(obs.hit_trap))
        risk = clip01(
            0.32 * float(obs.danger_pressure)
            + 0.30 * float(obs.pursuer_pressure)
            + 0.16 * float(obs.trap_pressure)
            + 0.12 * float(obs.wall_pressure)
            + 0.10 * float(gravity.get("gravity_fall_risk", 0.0))
        )
        comfort = clip01(0.52 + 0.30 * float(obs.resource_pressure) + 0.16 * float(obs.recovery_gain) - 0.22 * risk)
        surface_cost = clip01(float(obs.friction_pressure) + float(obs.current_pressure) + float(obs.trap_pressure))
        action_poss = clip01(1.0 - 0.45 * float(obs.wall_pressure) - 0.25 * surface_cost - 0.35 * float(gravity.get("gravity_fall_risk", 0.0)))

        # Body dynamics.
        recovery = 0.025 * float(obs.recovery_gain) + 0.018 * float(obs.resource_gain)
        s.integrity = clip01(s.integrity - 0.52 * damage + 0.04 * recovery)
        s.energy = clip01(s.energy - 0.00035 - 0.004 * surface_cost - 0.010 * risk + recovery)
        s.fatigue = clip01(0.975 * s.fatigue + 0.015 * surface_cost + 0.012 * risk - 0.05 * float(obs.recovery_gain))
        s.stability = clip01(s.stability - 0.035 * damage - 0.018 * float(obs.slope) - 0.020 * float(gravity.get("gravity_fall_risk", 0.0)) + 0.016 * action_poss)
        s.pain_memory = clip01(0.988 * s.pain_memory + 0.18 * damage)
        s.danger_memory = clip01(0.988 * s.danger_memory + 0.10 * risk + 0.06 * float(obs.danger_pressure) + 0.04 * float(obs.pursuer_pressure))
        s.comfort_memory = clip01(0.988 * s.comfort_memory + 0.10 * comfort)
        s.action_possibility = action_poss

        # Agency-like delay attribution: compare intended direction with resulting motion.
        motion = np.array([float(obs.vx), float(obs.vy)], dtype=float)
        mn = np.linalg.norm(motion)
        if mn > 1e-6:
            motion = motion / mn
        prev_n = np.linalg.norm(self.prev_action_intent)
        if prev_n > 1e-6:
            align = float(np.dot(self.prev_action_intent / prev_n, motion))
        else:
            align = 0.0
        s.agency_lag_memory = clip01(0.97 * s.agency_lag_memory + 0.03 * max(0.0, align))

        vulnerability = 1.0 - min(s.integrity, s.energy)
        q = (
            0.28 * risk
            + 0.25 * damage
            + 0.15 * s.danger_memory
            + 0.10 * s.pain_memory
            + 0.07 * s.fatigue
            + 0.07 * (1.0 - s.stability)
            + 0.07 * vulnerability
            + 0.06 * (1.0 - action_poss)
            - 0.11 * s.comfort_memory
            - 0.05 * s.agency_lag_memory
        )
        s.q = float(np.clip(q, -0.2, 1.2))
        return {
            "Q": s.q,
            "Q_integrity": s.integrity,
            "Q_energy": s.energy,
            "Q_fatigue": s.fatigue,
            "Q_stability": s.stability,
            "Q_pain": s.pain_memory,
            "Q_learned_danger": s.danger_memory,
            "Q_learned_comfort": s.comfort_memory,
            "Q_action_possibility": s.action_possibility,
            "Q_agency": s.agency_lag_memory,
            "Q_rho": risk,
            "Q_high_contact": damage,
        }

    def set_last_intent(self, vx: float, vy: float) -> None:
        self.prev_action_intent = np.array([float(vx), float(vy)], dtype=float)


# =============================================================================
# Phase-4c style anonymous social signal layer
# =============================================================================


@dataclass
class Phase4cConfig:
    n_signals: int = 5
    signal_threshold: float = 0.455
    signal_bias: float = -0.62
    signal_noise: float = 0.060
    signal_exploration_rate: float = 0.055
    signal_lr: float = 0.016
    weight_decay: float = 0.00045
    refractory_steps: int = 4
    signal_energy_cost: float = 0.00006
    listener_lr: float = 0.060
    signal_radius: float = 3.6
    listener_effect_strength: float = 0.070
    contagion_strength: float = 0.030
    base_social_relaxation: float = 0.0040
    social_decay: float = 0.007
    memory_decay: float = 0.012
    predictor_lr: float = 0.045
    safe_pe_cutoff: float = 0.07
    safe_damage_cutoff: float = 0.018
    safe_risk_cutoff: float = 0.38
    past_threat_cutoff: float = 0.34
    danger_damage_cutoff: float = 0.040
    danger_risk_cutoff: float = 0.64


class Phase4cAgentState:
    def __init__(self, rng: np.random.Generator, cfg: Phase4cConfig):
        self.integrity = float(rng.uniform(0.92, 1.00))
        self.energy = float(rng.uniform(0.88, 1.00))
        self.fatigue = float(rng.uniform(0.05, 0.15))
        self.stability = float(rng.uniform(0.74, 0.92))
        self.pain_memory = 0.02
        self.danger_memory = 0.06
        self.comfort_memory = 0.52
        self.social_tension = float(rng.uniform(0.14, 0.28))
        self.social_sync = float(rng.uniform(0.35, 0.55))
        self.exploration_drive = float(rng.uniform(0.50, 0.70))
        self.last_appraisal = 0.0
        self.last_actual_risk = 0.0
        self.last_q = 0.0
        self.refractory = 0
        self.x = 0.0
        self.y = 0.0
        self.listener_value = rng.normal(0.0, 0.02, size=cfg.n_signals)


class Phase4cSignalLayer:
    """Phase-4c anonymous social signal layer adapted to real observations.

    It keeps the source constraints:
        - no LAUGH/HUMOR action label,
        - no benign-violation variable in the controller,
        - no external prompt / LMM / API,
        - anonymous signal_0..signal_4 channels,
        - relief, safe-surprise, and self-appraisal gap are analysis variables.
    """

    def __init__(self, seed: int = 0, cfg: Optional[Phase4cConfig] = None):
        self.cfg = cfg or Phase4cConfig()
        self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.state = Phase4cAgentState(self.rng, self.cfg)
        self.n_features = 14
        self.signal_weights = self._make_initial_signal_weights()
        self.pred_intensity = 0.35
        self.last_probs = np.zeros(self.cfg.n_signals, dtype=float)

    def _make_initial_signal_weights(self) -> np.ndarray:
        cfg = self.cfg
        w = self.rng.normal(0.0, 0.08, size=(cfg.n_signals, self.n_features))
        w[:, 2] += self.rng.normal(0.02, 0.03, cfg.n_signals)      # prediction error
        w[:, 6] += self.rng.normal(0.08, 0.03, cfg.n_signals)      # relief raw
        w[:, 7] += self.rng.normal(0.07, 0.03, cfg.n_signals)      # novelty
        w[:, 9] += self.rng.normal(0.035, 0.025, cfg.n_signals)    # social tension
        w[:, 3] += self.rng.normal(-0.22, 0.035, cfg.n_signals)    # risk suppression
        w[:, 4] += self.rng.normal(-0.18, 0.035, cfg.n_signals)    # damage suppression
        return w

    def _observation_event_outcome(self, obs: AgentObservation, q_value: float) -> Dict[str, float]:
        actual_risk = clip01(
            0.34 * float(obs.danger_pressure)
            + 0.30 * float(obs.pursuer_pressure)
            + 0.14 * float(obs.trap_pressure)
            + 0.12 * float(obs.wall_pressure)
            + 0.10 * float(obs.social_density)
        )
        damage = clip01(float(obs.damage) + 0.06 * float(obs.hit_danger) + 0.10 * float(obs.hit_pursuer) + 0.03 * float(obs.hit_trap))
        intensity = clip01(
            0.22 * float(obs.novelty)
            + 0.24 * actual_risk
            + 0.18 * float(obs.current_pressure)
            + 0.18 * float(obs.friction_pressure)
            + 0.18 * min(1.0, abs(float(obs.vx)) + abs(float(obs.vy)))
        )
        appraisal = clip01(0.62 * actual_risk + 0.26 * q_value + 0.12 * float(obs.novelty))
        agency = clip01(0.58 + 0.22 * float(obs.resource_pressure) - 0.22 * actual_risk - 0.14 * float(obs.wall_pressure))
        return {
            "intensity": float(intensity),
            "actual_risk": float(actual_risk),
            "damage": float(damage),
            "initial_appraisal": float(appraisal),
            "novelty": float(obs.novelty),
            "agency": float(agency),
        }

    def _compute_q_source(self, out: Dict[str, float]) -> float:
        a = self.state
        vulnerability = 1.0 - min(a.integrity, a.energy)
        q = (
            0.28 * out["actual_risk"]
            + 0.30 * clip01(out["damage"] * 12.0)
            + 0.16 * a.danger_memory
            + 0.10 * a.pain_memory
            + 0.08 * a.fatigue
            + 0.08 * (1.0 - a.stability)
            + 0.08 * vulnerability
            - 0.12 * a.comfort_memory
            - 0.06 * a.social_sync
        )
        return float(np.clip(q, -0.2, 1.2))

    def _feature_vector(self, out: Dict[str, float], q: float) -> np.ndarray:
        a = self.state
        pe = abs(out["intensity"] - self.pred_intensity)
        relief_raw = max(0.0, a.last_appraisal - out["actual_risk"])
        x = np.array([
            1.0,
            q,
            pe,
            out["actual_risk"],
            clip01(out["damage"] * 12.0),
            a.last_appraisal,
            relief_raw,
            out["novelty"],
            out["agency"],
            a.social_tension,
            a.social_sync,
            a.exploration_drive,
            a.danger_memory,
            1.0 - min(a.integrity, a.energy),
        ], dtype=float)
        return x

    def _select_signal(self, features: np.ndarray, forced_channel: int = -1) -> Tuple[int, np.ndarray]:
        cfg = self.cfg
        a = self.state
        if forced_channel >= 0:
            return int(forced_channel), np.ones(cfg.n_signals, dtype=float)
        if a.refractory > 0:
            a.refractory -= 1
            return -1, np.zeros(cfg.n_signals, dtype=float)
        risk_inhibition = 0.55 * features[3] + 0.42 * features[4]
        logits = self.signal_weights @ features + cfg.signal_bias - risk_inhibition + self.rng.normal(0, cfg.signal_noise, cfg.n_signals)
        p = sigmoid(logits)
        if self.rng.random() < cfg.signal_exploration_rate:
            ch = int(self.rng.integers(0, cfg.n_signals))
            if self.rng.random() < 0.55:
                return ch, np.asarray(p, dtype=float)
        ch = int(np.argmax(p))
        if p[ch] > cfg.signal_threshold:
            return ch, np.asarray(p, dtype=float)
        return -1, np.asarray(p, dtype=float)

    def _analysis_variables(self, out: Dict[str, float], q_before: float, q_after: float) -> Dict[str, float]:
        cfg = self.cfg
        a = self.state
        pe = abs(out["intensity"] - self.pred_intensity)
        past_threat = a.last_appraisal
        current_safety = 1.0 - max(out["actual_risk"], clip01(out["damage"] * 10.0))
        relief = max(0.0, past_threat - out["actual_risk"])
        safe_surprise = float((pe >= cfg.safe_pe_cutoff) and (out["actual_risk"] <= cfg.safe_risk_cutoff) and (out["damage"] <= cfg.safe_damage_cutoff))
        self_gap = float(past_threat * current_safety * abs(past_threat - out["actual_risk"]))
        q_relief = max(0.0, q_before - q_after)
        danger_context = float((out["actual_risk"] >= cfg.danger_risk_cutoff) or (out["damage"] >= cfg.danger_damage_cutoff))
        safe_context = float((past_threat >= cfg.past_threat_cutoff) and (current_safety >= 0.60) and (pe >= cfg.safe_pe_cutoff) and (danger_context < 0.5))
        return {
            "prediction_error_social": float(pe),
            "relief": float(relief),
            "safe_surprise": safe_surprise,
            "self_appraisal_gap": self_gap,
            "q_relief": float(q_relief),
            "safe_context": safe_context,
            "danger_context": danger_context,
            "past_threat": float(past_threat),
            "current_safety": float(current_safety),
        }

    def _update_body(self, out: Dict[str, float], q: float, signal_emitted: bool) -> None:
        cfg = self.cfg
        a = self.state
        damage = out["damage"]
        risk = out["actual_risk"]
        appraisal = out["initial_appraisal"]
        energy_cost = 0.00008 + 0.00055 * risk + (cfg.signal_energy_cost if signal_emitted else 0.0)
        safe_recovery = 0.0025 if (risk < 0.30 and damage < 0.006 and appraisal > 0.40) else 0.0
        passive_repair = 0.00035 if damage < 0.012 else 0.0
        a.integrity = clip01(a.integrity - 0.62 * damage + safe_recovery + passive_repair)
        a.energy = clip01(a.energy - energy_cost + 0.25 * safe_recovery)
        a.fatigue = clip01((1 - 0.024) * a.fatigue + 0.010 * out["intensity"] + 0.004 * risk)
        a.stability = clip01(a.stability - 0.055 * damage - 0.012 * risk + 0.020 * out["agency"] + 0.010 * safe_recovery)
        a.pain_memory = clip01((1 - cfg.memory_decay) * a.pain_memory + 0.18 * clip01(damage * 10.0))
        a.danger_memory = clip01((1 - cfg.memory_decay) * a.danger_memory + 0.10 * risk + 0.06 * appraisal)
        a.comfort_memory = clip01((1 - cfg.memory_decay) * a.comfort_memory + 0.10 * (1.0 - risk))
        a.social_tension = clip01((1 - cfg.social_decay) * a.social_tension + 0.050 * risk + 0.025 * appraisal - cfg.base_social_relaxation)
        a.social_sync = clip01((1 - 0.004) * a.social_sync - 0.008 * risk)
        a.exploration_drive = clip01(0.995 * a.exploration_drive + 0.006 * (1.0 - risk) - 0.012 * q)

    def _apply_to_neighbors(self, neighbors: Optional[List[NeighborState]], channel: int, out: Dict[str, float]) -> Tuple[float, float, float]:
        if channel < 0 or not neighbors:
            return 0.0, 0.0, 0.0
        cfg = self.cfg
        a = self.state
        recoveries: List[float] = []
        spread = 0
        heard = 0
        for nb in neighbors:
            d = math.sqrt((a.x - float(nb.x)) ** 2 + (a.y - float(nb.y)) ** 2)
            if d > cfg.signal_radius:
                continue
            heard += 1
            lv = nb.listener_value if nb.listener_value is not None else np.zeros(cfg.n_signals, dtype=float)
            before_tension = float(nb.social_tension)
            before_explore = float(nb.exploration_drive)
            risk_penalty = max(0.0, out["actual_risk"] - 0.45) + clip01(out["damage"] * 8.0)
            effect = cfg.listener_effect_strength * sigmoid(2.5 * float(lv[channel])) * max(0.0, 1.0 - 1.7 * risk_penalty)
            nb.social_tension = clip01(float(nb.social_tension) - effect)
            nb.social_sync = clip01(float(nb.social_sync) + 0.75 * effect)
            nb.exploration_drive = clip01(float(nb.exploration_drive) + 0.55 * effect)
            rec = (before_tension - nb.social_tension) + 0.5 * (nb.exploration_drive - before_explore)
            recoveries.append(rec)
            if self.rng.random() < cfg.contagion_strength * sigmoid(2.0 * float(lv[channel])) * max(0.0, 1.0 - risk_penalty):
                spread += 1
        return float(np.mean(recoveries)) if recoveries else 0.0, float(spread), float(heard)

    def _update_learning(self, features: np.ndarray, channel: int, benefit: float, receiver_recovery: float, out: Dict[str, float]) -> None:
        if channel < 0:
            return
        cfg = self.cfg
        risk_cost = 0.035 * max(0.0, out["actual_risk"] - 0.42) + 0.030 * clip01(out["damage"] * 10.0)
        delta = benefit - risk_cost - cfg.signal_energy_cost
        self.signal_weights *= (1.0 - cfg.weight_decay)
        self.signal_weights[channel] += cfg.signal_lr * np.clip(delta, -0.05, 0.05) * features
        target = receiver_recovery - (0.05 * max(0.0, out["actual_risk"] - 0.45) + 0.05 * clip01(out["damage"] * 8.0))
        self.state.listener_value[channel] += cfg.listener_lr * np.clip(target, -0.08, 0.08)
        self.state.listener_value[channel] = np.clip(self.state.listener_value[channel], -1.5, 1.5)

    def forced_channel_from_observation(self, obs: AgentObservation) -> int:
        if obs.hit_pursuer > 0.5 or obs.pursuer_pressure > 0.82:
            return 4
        if obs.hit_danger > 0.5 or obs.danger_pressure > 0.88:
            return 1
        if obs.hit_wall > 0.5 or obs.wall_pressure > 0.92:
            return 2
        if obs.hit_trap > 0.5 or obs.trap_pressure > 0.88 or obs.friction_pressure > 0.92 or obs.current_pressure > 0.92:
            return 3
        if obs.resource_pressure > 0.90 and obs.danger_pressure < 0.45 and obs.pursuer_pressure < 0.45:
            return 0
        return -1

    def step(self, obs: AgentObservation, q_value: float, neighbors: Optional[List[NeighborState]] = None) -> Dict[str, Any]:
        a = self.state
        a.x = float(obs.x)
        a.y = float(obs.y)
        out = self._observation_event_outcome(obs, q_value)
        q_before = a.last_q
        q_source = self._compute_q_source(out)
        features = self._feature_vector(out, q_source)
        forced = self.forced_channel_from_observation(obs)
        ch, probs = self._select_signal(features, forced_channel=forced)
        if ch >= 0:
            a.refractory = self.cfg.refractory_steps
        receiver_recovery, spread, heard = self._apply_to_neighbors(neighbors, ch, out)
        benefit = max(0.0, a.social_tension - (a.social_tension - 0.01 * receiver_recovery)) + 0.5 * receiver_recovery
        self._update_learning(features, ch, benefit, receiver_recovery, out)
        self._update_body(out, q_source, signal_emitted=(ch >= 0))
        q_after = self._compute_q_source(out)
        ana = self._analysis_variables(out, q_before, q_after)
        self.pred_intensity = (1.0 - self.cfg.predictor_lr) * self.pred_intensity + self.cfg.predictor_lr * out["intensity"]
        a.last_appraisal = out["initial_appraisal"]
        a.last_actual_risk = out["actual_risk"]
        a.last_q = q_after
        self.last_probs = np.asarray(probs, dtype=float)
        return {
            "signal_channel": int(ch),
            "signal_label": SIGNAL_LABELS.get(int(ch), "UNKNOWN"),
            "signal_strength": float(probs[ch]) if ch >= 0 and len(probs) > ch else 0.0,
            "signal_probs": probs.tolist(),
            "receiver_recovery": float(receiver_recovery),
            "social_spread": float(spread),
            "n_heard": float(heard),
            "social_q_source": float(q_source),
            **out,
            **ana,
        }


# =============================================================================
# Integrated agent
# =============================================================================


class IntegratedDarcaV24Agent:
    """Independent integrated agent.

    The environment calls step(observation) and receives a motor/signal intent.
    Nothing in this class constructs terrain or viewer objects.
    """

    def __init__(self, darca_file: str, seed: int = 0, theta: Optional[float] = None):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed & 0xFFFFFFFF)
        self.darca = DarcaV24Wrapper(darca_file=darca_file, seed=seed, theta=theta)
        self.gravity = GravityUnderstandingLayer()
        self.q_layer = QualitativeValenceLayer(seed=seed + 101)
        self.social = Phase4cSignalLayer(seed=seed + 202)
        self.last_action_class = "MOVE"
        self.last_row: Dict[str, Any] = {}
        self._last_gravity_action_bias = 0.0
        self._last_gravity_fall_vector_mag = 0.0
        self._last_gravity_uphill_vector_mag = 0.0

    def _scalar_for_darca(self, obs: AgentObservation) -> Tuple[float, float]:
        # Positive = resource/novelty/controlled ascent; negative = danger/damage/block.
        if obs.scalar_y is not None:
            y = float(obs.scalar_y)
        else:
            y = (
                1.10 * float(obs.resource_pressure)
                + 0.18 * float(obs.novelty)
                + 0.06 * max(0.0, float(obs.height))
                - 1.05 * float(obs.danger_pressure)
                - 1.10 * float(obs.pursuer_pressure)
                - 0.62 * float(obs.trap_pressure)
                - 0.54 * float(obs.wall_pressure)
                - 0.34 * float(obs.damage)
            )
        external_shock = clip01(
            0.30 * abs(y)
            + 0.26 * float(obs.danger_pressure)
            + 0.30 * float(obs.pursuer_pressure)
            + 0.16 * float(obs.trap_pressure)
            + 0.18 * float(obs.damage)
            + 0.12 * float(obs.wall_pressure)
        )
        return float(np.clip(y, -2.0, 2.0)), float(external_shock)

    def _action_class_from_darca(self, darca_out: Dict[str, Any], obs: AgentObservation, q: float, gravity: Dict[str, float]) -> str:
        name = str(darca_out.get("action_name", ""))
        fall_risk = clip01(float(gravity.get("gravity_fall_risk", 0.0)))
        # Gravity must enter the action class, not only the diagnostic score.
        # A fall-risky state is treated as AVOID unless a confident jump affordance is available.
        if fall_risk > 0.20 and (float(obs.vertical_drop) > 0.12 or float(obs.slope) > 0.14 or float(obs.grounded) < 0.75):
            if float(obs.can_jump) > 0.5 and float(gravity.get("jump_affordance", 0.0)) > 0.35:
                return "JUMP"
            return "AVOID"
        if float(obs.can_jump) > 0.5 and float(gravity.get("jump_affordance", 0.0)) > 0.35:
            return "JUMP"
        if "SCAN" in name or "PROBE" in name:
            return "SCAN"
        if "REGULATE" in name or q > 0.82 or float(darca_out.get("h", 1.0)) < 0.38:
            return "REST"
        if float(obs.danger_pressure) + float(obs.pursuer_pressure) + float(obs.wall_pressure) > 1.0:
            return "AVOID"
        if float(obs.resource_pressure) > 0.35:
            return "APPROACH"
        return "MOVE"

    def _motor_intent(self, obs: AgentObservation, darca_out: Dict[str, Any], qd: Dict[str, float], gravity: Dict[str, float], action_class: str) -> Tuple[float, float, float, float, float, float, float]:
        # Directional hints. Positive attract, negative repel. If no hints are supplied,
        # default to current heading / target hint.
        hx, hy, _ = norm2(float(obs.heading_x), float(obs.heading_y))
        tx, ty, tn = norm2(float(obs.target_hint_dx), float(obs.target_hint_dy))
        if tn <= 0:
            tx, ty = hx, hy
        rdx, rdy, _ = norm2(float(obs.resource_dx), float(obs.resource_dy))
        ddx, ddy, _ = norm2(float(obs.danger_dx), float(obs.danger_dy))
        tdx, tdy, _ = norm2(float(obs.trap_dx), float(obs.trap_dy))
        wdx, wdy, _ = norm2(float(obs.wall_dx), float(obs.wall_dy))
        pdx, pdy, _ = norm2(float(obs.pursuer_dx), float(obs.pursuer_dy))
        cdx, cdy, _ = norm2(float(obs.current_dx), float(obs.current_dy))

        vx = 0.58 * tx + 0.55 * float(obs.resource_pressure) * rdx
        vy = 0.58 * ty + 0.55 * float(obs.resource_pressure) * rdy
        # Repel from known danger vectors. The vectors are assumed to point from agent to object.
        vx -= 0.92 * float(obs.danger_pressure) * ddx
        vy -= 0.92 * float(obs.danger_pressure) * ddy
        vx -= 1.10 * float(obs.pursuer_pressure) * pdx
        vy -= 1.10 * float(obs.pursuer_pressure) * pdy
        vx -= 0.62 * float(obs.trap_pressure) * tdx
        vy -= 0.62 * float(obs.trap_pressure) * tdy
        vx -= 0.70 * float(obs.wall_pressure) * wdx
        vy -= 0.70 * float(obs.wall_pressure) * wdy
        # Current pushes body; agent can partially compensate.
        vx -= 0.20 * float(obs.current_pressure) * cdx
        vy -= 0.20 * float(obs.current_pressure) * cdy

        # Gravity-fixed coupling: previous versions only slowed the agent when
        # gravity_fall_risk was high. That made NO_GRAVITY look almost identical.
        # Here the gravity layer contributes a directional bias: repel from the
        # locally worse fall/drop direction and bias toward uphill when available.
        fall_risk = clip01(float(gravity.get("gravity_fall_risk", 0.0)))
        fdx, fdy, fmag = norm2(float(gravity.get("gravity_fall_dx", getattr(obs, "fall_dx", 0.0))), float(gravity.get("gravity_fall_dy", getattr(obs, "fall_dy", 0.0))))
        udx, udy, umag = norm2(float(gravity.get("gravity_uphill_dx", getattr(obs, "uphill_dx", 0.0))), float(gravity.get("gravity_uphill_dy", getattr(obs, "uphill_dy", 0.0))))
        gravity_action_bias = 0.0
        if fall_risk > 0.02:
            if fmag > 1e-9:
                vx -= 1.30 * fall_risk * fdx
                vy -= 1.30 * fall_risk * fdy
                gravity_action_bias += 1.30 * fall_risk
            if umag > 1e-9:
                vx += 0.85 * fall_risk * udx
                vy += 0.85 * fall_risk * udy
                gravity_action_bias += 0.85 * fall_risk

        if action_class == "AVOID":
            vx *= 1.18
            vy *= 1.18
        elif action_class == "APPROACH":
            vx += 0.25 * rdx
            vy += 0.25 * rdy
        elif action_class == "SCAN":
            # slow, rotate-ish drift; environment may ignore this and animate scan.
            vx = 0.30 * vx - 0.18 * hy
            vy = 0.30 * vy + 0.18 * hx
        elif action_class == "REST":
            vx *= 0.15
            vy *= 0.15

        ux, uy, mag = norm2(vx, vy)
        if mag <= 0:
            ux, uy = hx, hy
        h = float(darca_out.get("h", 0.7))
        darca_active = float(darca_out.get("active", 0.0))
        base_speed = 0.35 + 0.45 * clip01(h) + 0.10 * clip01(darca_active)
        surface_slow = 1.0 - 0.34 * clip01(float(obs.friction_pressure) + 0.6 * float(obs.current_pressure) + 0.5 * float(obs.trap_pressure))
        q_slow = 1.0 - 0.32 * clip01(float(qd.get("Q", 0.0)))
        fall_slow = 1.0 - 0.25 * clip01(float(gravity.get("gravity_fall_risk", 0.0)))
        speed_scale = clip(base_speed * surface_slow * q_slow * fall_slow, 0.05, 1.25)
        wants_jump = float(action_class == "JUMP")
        wants_scan = float(action_class == "SCAN")
        wants_rest = float(action_class == "REST")
        wants_regulate = float("REGULATE" in str(darca_out.get("action_name", "")))
        self._last_gravity_action_bias = float(locals().get("gravity_action_bias", 0.0))
        self._last_gravity_fall_vector_mag = float(locals().get("fmag", 0.0))
        self._last_gravity_uphill_vector_mag = float(locals().get("umag", 0.0))
        return ux, uy, speed_scale, wants_jump, wants_scan, wants_rest, wants_regulate

    def step(self, observation: AgentObservation, neighbors: Optional[List[NeighborState]] = None) -> AgentAction:
        obs = observation.clipped()
        scalar_y, shock = self._scalar_for_darca(obs)
        darca_out = self.darca.step(scalar_y, shock)
        # preliminary action class for physics update
        prelim_action = "MOVE"
        gravity = self.gravity.step(obs, prelim_action)
        qd = self.q_layer.step(obs, darca_out, gravity)
        action_class = self._action_class_from_darca(darca_out, obs, float(qd.get("Q", 0.0)), gravity)
        # update gravity predictor with final action class too, so per-action stats track real intent.
        gravity = {**gravity, **self.gravity.predictor.update(action_class, obs, float(gravity.get("gravity_fall_risk", 0.0)))}
        social = self.social.step(obs, float(qd.get("Q", 0.0)), neighbors=neighbors)
        vx, vy, speed_scale, wants_jump, wants_scan, wants_rest, wants_regulate = self._motor_intent(obs, darca_out, qd, gravity, action_class)
        self.q_layer.set_last_intent(vx, vy)

        signal_channel = int(social.get("signal_channel", -1))
        out = AgentAction(
            desired_vx=float(vx),
            desired_vy=float(vy),
            speed_scale=float(speed_scale),
            wants_jump=float(wants_jump),
            wants_scan=float(wants_scan),
            wants_rest=float(wants_rest),
            wants_regulate=float(wants_regulate),
            signal_channel=signal_channel,
            signal_label=SIGNAL_LABELS.get(signal_channel, "UNKNOWN"),
            signal_strength=float(social.get("signal_strength", 0.0)),
            q=float(qd.get("Q", 0.0)),
            life_h=float(darca_out.get("h", 0.0)),
            autonomy=float(darca_out.get("autonomy", 0.0)),
            identity=float(darca_out.get("identity", 0.0)),
            darca_action_name=str(darca_out.get("action_name", "NA")),
            darca_u=float(darca_out.get("u", 0.0)),
            scalar_y=float(scalar_y),
            gravity_fall_risk=float(gravity.get("gravity_fall_risk", 0.0)),
            physics_score=float(gravity.get("physics_score", 0.0)),
            self_appraisal_gap=float(social.get("self_appraisal_gap", 0.0)),
            relief=float(social.get("relief", 0.0)),
            safe_surprise=float(social.get("safe_surprise", 0.0)),
            receiver_recovery=float(social.get("receiver_recovery", 0.0)),
            n_heard=float(social.get("n_heard", 0.0)),
            social_spread=float(social.get("social_spread", 0.0)),
            debug={
                "action_class": action_class,
                "gravity_action_bias": float(self._last_gravity_action_bias),
                "gravity_fall_vector_used": float(self._last_gravity_fall_vector_mag),
                "gravity_uphill_vector_used": float(self._last_gravity_uphill_vector_mag),
                **{k: v for k, v in darca_out.items() if isinstance(v, (int, float, str))},
                **qd,
                **gravity,
                **{k: v for k, v in social.items() if k != "signal_probs"},
            },
        )
        self.last_action_class = action_class
        self.last_row = out.to_row()
        return out


# Public alias using the user's preferred all-caps spelling.
IntegratedDARCAAgent = IntegratedDarcaV24Agent


class IntegratedDarcaPopulation:
    """Convenience multi-agent container using the same independent agent class."""

    def __init__(self, darca_file: str, n_agents: int = 10, seed: int = 0):
        self.agents = [IntegratedDarcaV24Agent(darca_file=darca_file, seed=seed + 1009 * i) for i in range(int(n_agents))]
        self.neighbor_states = [NeighborState(x=0.0, y=0.0) for _ in self.agents]

    def step(self, observations: Sequence[AgentObservation]) -> List[AgentAction]:
        actions: List[AgentAction] = []
        # update neighbor locations before stepping
        for i, obs in enumerate(observations[: len(self.neighbor_states)]):
            self.neighbor_states[i].x = float(obs.x)
            self.neighbor_states[i].y = float(obs.y)
            self.neighbor_states[i].listener_value = self.agents[i].social.state.listener_value.copy()
            self.neighbor_states[i].social_tension = self.agents[i].social.state.social_tension
            self.neighbor_states[i].social_sync = self.agents[i].social.state.social_sync
            self.neighbor_states[i].exploration_drive = self.agents[i].social.state.exploration_drive
        for i, agent in enumerate(self.agents):
            obs = observations[i] if i < len(observations) else AgentObservation(step=0)
            neighbors = [n for j, n in enumerate(self.neighbor_states) if j != i]
            actions.append(agent.step(obs, neighbors=neighbors))
        return actions


# =============================================================================
# Smoke test CLI
# =============================================================================


def make_mock_observation(step: int, rng: np.random.Generator, x: float, y: float) -> AgentObservation:
    danger = clip01(0.2 + 0.45 * max(0.0, math.sin(step * 0.031)) + rng.normal(0, 0.04))
    resource = clip01(0.35 + 0.35 * max(0.0, math.sin(step * 0.017 + 1.2)) + rng.normal(0, 0.04))
    pursuer = clip01(0.75 if (step % 120) in range(42, 58) else 0.12 + rng.normal(0, 0.03))
    trap = clip01(0.62 if (step % 90) in range(20, 28) else 0.10 + rng.normal(0, 0.03))
    wall = clip01(0.82 if (step % 160) in range(70, 76) else 0.06 + rng.normal(0, 0.03))
    angle = 0.025 * step
    return AgentObservation(
        step=step,
        x=x,
        y=y,
        z=0.6 + 0.2 * math.sin(0.015 * step),
        vx=0.2 * math.cos(angle),
        vy=0.2 * math.sin(angle),
        vz=0.03 * math.cos(0.015 * step),
        heading_x=math.cos(angle),
        heading_y=math.sin(angle),
        grounded=1.0,
        slope=clip01(0.12 + 0.2 * max(0.0, math.sin(0.04 * step))),
        height=0.3 + 0.3 * max(0.0, math.sin(0.015 * step)),
        resource_pressure=resource,
        danger_pressure=danger,
        trap_pressure=trap,
        wall_pressure=wall,
        pursuer_pressure=pursuer,
        friction_pressure=0.2 * trap,
        current_pressure=0.1 * trap,
        novelty=clip01(0.25 + 0.25 * abs(math.sin(0.05 * step))),
        resource_dx=math.cos(angle + 0.3),
        resource_dy=math.sin(angle + 0.3),
        danger_dx=math.cos(angle - 1.0),
        danger_dy=math.sin(angle - 1.0),
        trap_dx=math.cos(angle + 1.7),
        trap_dy=math.sin(angle + 1.7),
        wall_dx=math.cos(angle - 2.0),
        wall_dy=math.sin(angle - 2.0),
        pursuer_dx=math.cos(angle + 2.6),
        pursuer_dy=math.sin(angle + 2.6),
        target_hint_dx=math.cos(angle),
        target_hint_dy=math.sin(angle),
        damage=0.06 if pursuer > 0.7 else 0.0,
        hit_pursuer=1.0 if pursuer > 0.72 else 0.0,
        hit_wall=1.0 if wall > 0.8 else 0.0,
        hit_trap=1.0 if trap > 0.6 else 0.0,
        resource_gain=0.03 if resource > 0.65 else 0.0,
    )


def run_smoke(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(int(args.seed) & 0xFFFFFFFF)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    n_agents = max(1, int(args.n_agents))
    if n_agents == 1:
        agent = IntegratedDARCAAgent(darca_file=args.darca_file, seed=args.seed)
        x, y = 0.0, 0.0
        for step in range(int(args.steps)):
            obs = make_mock_observation(step, rng, x, y)
            act = agent.step(obs)
            x += 0.18 * act.speed_scale * act.desired_vx
            y += 0.18 * act.speed_scale * act.desired_vy
            row = {"agent_id": 0, "step": step, "x_after": x, "y_after": y, **asdict(obs), **act.to_row()}
            rows.append(row)
            if step % max(1, int(args.log_every)) == 0:
                print(
                    f"[step={step:5d}] agent=0 action={act.darca_action_name:14s} "
                    f"class={act.debug.get('action_class','NA'):8s} h={act.life_h:.3f} Q={act.q:.3f} "
                    f"sig={act.signal_channel}:{act.signal_label} v=({act.desired_vx:+.2f},{act.desired_vy:+.2f})",
                    flush=True,
                )
    else:
        pop = IntegratedDarcaPopulation(darca_file=args.darca_file, n_agents=n_agents, seed=args.seed)
        xy = []
        for i in range(n_agents):
            angle = 2.0 * math.pi * i / max(1, n_agents)
            xy.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
        for step in range(int(args.steps)):
            observations: List[AgentObservation] = []
            for i in range(n_agents):
                x, y = xy[i]
                obs = make_mock_observation(step + 13 * i, rng, x, y)
                # give each agent a slightly different directional context; this is only smoke input,
                # not an internal agent rule.
                obs.step = step
                obs.x = x
                obs.y = y
                observations.append(obs)
            actions = pop.step(observations)
            for i, (obs, act) in enumerate(zip(observations, actions)):
                xy[i][0] += 0.18 * act.speed_scale * act.desired_vx
                xy[i][1] += 0.18 * act.speed_scale * act.desired_vy
                row = {"agent_id": i, "step": step, "x_after": xy[i][0], "y_after": xy[i][1], **asdict(obs), **act.to_row()}
                rows.append(row)
            if step % max(1, int(args.log_every)) == 0:
                sig_counts: Dict[int, int] = {}
                mean_q = 0.0
                for act in actions:
                    sig_counts[act.signal_channel] = sig_counts.get(act.signal_channel, 0) + 1
                    mean_q += float(act.q)
                mean_q /= max(1, len(actions))
                print(
                    f"[step={step:5d}] population n={n_agents} mean_Q={mean_q:.3f} "
                    f"signals={sig_counts}",
                    flush=True,
                )

    csv_path = outdir / "integrated_agent_smoke.csv"
    write_csv(csv_path, rows)
    report = outdir / "README_integrated_agent_core.txt"
    report.write_text(
        "DARCA v24 integrated independent agent smoke test\n"
        "================================================\n\n"
        f"steps: {args.steps}\n"
        f"n_agents: {n_agents}\n"
        f"csv: {csv_path}\n\n"
        "The file under test contains no MuJoCo viewer/environment construction.\n"
        "It loads DARCA v24 from --darca-file and adds gravity/physics, Q, and Phase-4c anonymous signals.\n"
        "For external projects, import IntegratedDARCAAgent or IntegratedDarcaPopulation and call step(observation).\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {csv_path}", flush=True)
    return 0

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Independent DARCA v24 integrated agent core smoke test.")
    ap.add_argument("--darca-file", required=True, help="Path to darca_v24_.py")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7200)
    ap.add_argument("--outdir", default=str(Path.home() / "Desktop" / "darca_integrated_agent_core_smoke"))
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--n-agents", type=int, default=1, help="Smoke-test population size. Use 1 for a single integrated agent.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
