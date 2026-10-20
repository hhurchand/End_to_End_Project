"""
team_metrics.py — Team aggregate metrics and grade formulas.

Implements (per grading-methodology-2026-10-17.md):
  §4    starter_PAR, bench_depth_PAR, weakness_rate, combined_score
  §4.1  Team grade formula (tanh-scaled)
  §4.2  Individual player display grade (clamp-scaled)

Does NOT implement: synthetic draft simulator, schedule generator,
PAR engine (see par_engine.py), projection baselines (see baselines.py),
or any validation experiment.

Canonical identifier: player_id (string). Never use player_name for matching.

Input contract:
  - assigned: Dict[slot_index, player_id] — output of compute_per_team_lineup().assigned
  - n_total_slots: total starting slots for the team (filled + unsaturated)
  - par_values: player_id → PAR float (genuine-B3 players only; others absent or None)
  - genuine_b3_set: frozenset of player_ids with non-NaN compute_b3_all() output
  - roster_player_ids: all player_ids on the team (starters + bench)
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Optional, Tuple


# ---------------------------------------------------------------------------
# §4 — Team aggregate metrics
# ---------------------------------------------------------------------------

def compute_starter_par(
    assigned: Dict[int, str],
    par_values: Dict[str, Optional[float]],
    genuine_b3_set: FrozenSet[str],
) -> float:
    """
    Sum of PAR for optimal-lineup starters with genuine B3 projections (§4).

    Starters assigned to a slot via fallback projection (B1 or B0 only) are
    not in the genuine-B3 pool. Their contribution to starter_PAR is 0.
    Unsaturated slots (no eligible starter) also contribute 0.
    PAR values that are None (position exhausted stop condition) contribute 0.

    Parameters
    ----------
    assigned         : slot_index → player_id for every filled slot
    par_values       : player_id → PAR float (None if undefined)
    genuine_b3_set   : player_ids with non-NaN compute_b3_all() output
    """
    total = 0.0
    for pid in assigned.values():
        if pid in genuine_b3_set:
            par = par_values.get(pid)
            if par is not None:
                total += par
    return total


def compute_bench_depth_par(
    roster_player_ids: list[str],
    assigned: Dict[int, str],
    par_values: Dict[str, Optional[float]],
    genuine_b3_set: FrozenSet[str],
) -> float:
    """
    Sum of positive PAR among bench players with genuine B3 projections (§4).

    Bench = all rostered players not in the optimal starting lineup.
    Only positive PAR contributions are counted; players with PAR ≤ 0
    contribute 0. Players without genuine B3 projections contribute 0.
    PAR values that are None contribute 0.

    Parameters
    ----------
    roster_player_ids : all player_ids on the team (starters + bench)
    assigned          : slot_index → player_id for starters
    par_values        : player_id → PAR float (None if undefined)
    genuine_b3_set    : player_ids with non-NaN compute_b3_all() output
    """
    starter_pids = set(assigned.values())
    total = 0.0
    for pid in roster_player_ids:
        if pid in starter_pids:
            continue
        if pid not in genuine_b3_set:
            continue
        par = par_values.get(pid)
        if par is not None and par > 0.0:
            total += par
    return total


def compute_weakness_rate(
    assigned: Dict[int, str],
    n_total_slots: int,
    par_values: Dict[str, Optional[float]],
    genuine_b3_set: FrozenSet[str],
) -> float:
    """
    Fraction of starting slots that are "weak" (§4, §7 Experiment 2b).

    A slot is weak if:
      - It is unsaturated (no eligible player for that slot type)
      - Its starter is a fallback player (not in genuine_b3_set)
      - Its starter has a genuine B3 projection with PAR < 0
      - Its starter has a genuine B3 projection with PAR = None (stop condition)

    Ranges from 0.0 (no weak starters) to 1.0 (all starters weak).
    Returns 0.0 when n_total_slots == 0.

    Parameters
    ----------
    assigned        : slot_index → player_id for filled slots only
    n_total_slots   : total starting slots (filled + unsaturated)
    par_values      : player_id → PAR float (None if undefined)
    genuine_b3_set  : player_ids with non-NaN compute_b3_all() output
    """
    if n_total_slots == 0:
        return 0.0

    # Unsaturated slots are always weak.
    weak = n_total_slots - len(assigned)

    for pid in assigned.values():
        if pid not in genuine_b3_set:
            # Fallback starter (B1 or B0) — always weak per §4.
            weak += 1
        else:
            par = par_values.get(pid)
            # PAR < 0 or undefined (None) → weak.
            if par is None or par < 0.0:
                weak += 1

    return weak / n_total_slots


# ---------------------------------------------------------------------------
# §7 Experiment 2b — norm() and combined_score
# ---------------------------------------------------------------------------

def _norm(x: float, p5: float, p95: float) -> float:
    """
    Percentile-normalise x into [0, 1] using pre-computed P5 and P95 bounds.

    norm(x, P5, P95) = clamp((x − P5) / (P95 − P5), 0.0, 1.0)

    Divide-by-zero guard: if P95 == P5, return 0.5 for all x.

    Parameters
    ----------
    x   : value to normalise
    p5  : 5th-percentile bound from calibration population
    p95 : 95th-percentile bound from calibration population
    """
    if p95 == p5:
        return 0.5
    raw = (x - p5) / (p95 - p5)
    return max(0.0, min(1.0, raw))


def compute_combined_score(
    starter_par: float,
    bench_depth_par: float,
    weakness_rate: float,
    starter_par_bounds: Tuple[float, float],
    bench_depth_par_bounds: Tuple[float, float],
) -> float:
    """
    Pre-registered candidate combined_score formula (§7, Experiment 2b).

    combined_score = 50 + 25 × norm(starter_PAR)
                        + 15 × norm(bench_depth_PAR)
                        + 10 × (1 − weakness_rate)

    Weights are pre-registered candidates. They are validated (not confirmed)
    by Experiment 2b before being used in the production grade formula.

    Parameters
    ----------
    starter_par              : output of compute_starter_par()
    bench_depth_par          : output of compute_bench_depth_par()
    weakness_rate            : output of compute_weakness_rate(), in [0, 1]
    starter_par_bounds       : (P5, P95) for starter_PAR from 2016–2020 window
    bench_depth_par_bounds   : (P5, P95) for bench_depth_PAR from 2016–2020 window
    """
    sp_norm = _norm(starter_par, *starter_par_bounds)
    bd_norm = _norm(bench_depth_par, *bench_depth_par_bounds)
    return 50.0 + 25.0 * sp_norm + 15.0 * bd_norm + 10.0 * (1.0 - weakness_rate)


# ---------------------------------------------------------------------------
# §4.1 — Team grade formula
# ---------------------------------------------------------------------------

def compute_team_grade(
    combined_score: float,
    mu: float,
    sigma: float,
) -> float:
    """
    Convert combined_score to 0–100 grade via tanh scaling (§4.1).

    grade = 50 + 50 × tanh((combined_score − μ) / σ)

    Raises ValueError if sigma == 0 (§4.1 hard stop: formula undefined).

    In the validation experiments μ and σ are frozen calibration constants
    derived from the 2016–2020 window before any 2021–2022 data is examined.

    Parameters
    ----------
    combined_score : output of compute_combined_score()
    mu             : calibration mean (μ_cs from 2016–2020 population)
    sigma          : calibration std  (σ_cs from 2016–2020 population); must be > 0
    """
    if sigma == 0.0:
        raise ValueError(
            "sigma == 0: team grade formula is undefined (§4.1 hard stop). "
            "combined_score has no variation in the calibration population."
        )
    return 50.0 + 50.0 * math.tanh((combined_score - mu) / sigma)


# ---------------------------------------------------------------------------
# §4.1 — Fallback team grade (starter_PAR only)
# ---------------------------------------------------------------------------

def compute_team_grade_starter_par(
    starter_par: float,
    mu: float,
    sigma: float,
) -> float:
    """
    Fallback team grade using starter_PAR only (§4.1 Outcome 2).

    Used when combined_score fails Experiment 2b retention.

    grade = 50 + 50 × tanh((starter_PAR − μ_sp) / σ_sp)

    Raises ValueError if sigma == 0.

    Parameters
    ----------
    starter_par : output of compute_starter_par()
    mu          : calibration mean (μ_sp from 2016–2020 population)
    sigma       : calibration std  (σ_sp from 2016–2020 population); must be > 0
    """
    if sigma == 0.0:
        raise ValueError(
            "sigma == 0: team grade formula is undefined (§4.1 hard stop). "
            "starter_PAR has no variation in the calibration population."
        )
    return 50.0 + 50.0 * math.tanh((starter_par - mu) / sigma)


# ---------------------------------------------------------------------------
# §4.2 — Individual player display grade
# ---------------------------------------------------------------------------

def compute_player_grade(
    par: float,
    par_95th: float,
) -> float:
    """
    Convert a player's PAR to a 0–100 display grade (§4.2).

    player_grade = 50 + 50 × clamp(PAR / PAR_95th, −1, 1)

    - PAR = 0          → grade 50  (mid-range, C)
    - PAR = PAR_95th   → grade 100 (A)
    - PAR = −PAR_95th  → grade 0   (F)

    Raises ValueError if par_95th ≤ 0 (§4.2 stop condition: invalid constant).

    Parameters
    ----------
    par      : player's PAR value (output of compute_par())
    par_95th : 95th-percentile PAR for the player's position (frozen constant
               derived from 2012–2015 calibration window)
    """
    if par_95th <= 0.0 or not math.isfinite(par_95th):
        raise ValueError(
            f"PAR_95th must be a positive finite number; got {par_95th!r}. "
            "§4.2 stop condition: grade conversion is undefined."
        )
    clamped = max(-1.0, min(1.0, par / par_95th))
    return 50.0 + 50.0 * clamped


# ---------------------------------------------------------------------------
# §4.1 — Letter grade threshold
# ---------------------------------------------------------------------------

#: Pre-specified letter grade thresholds (§4.1).
LETTER_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (80.0, "A"),
    (65.0, "B"),
    (45.0, "C"),
    (30.0, "D"),
    (0.0,  "F"),
]


def numeric_to_letter(grade: float) -> str:
    """
    Convert a numeric grade (0–100) to a letter grade per §4.1 thresholds.

    | Letter | Numeric range |
    |--------|---------------|
    | A      | 80–100        |
    | B      | 65–79         |
    | C      | 45–64         |
    | D      | 30–44         |
    | F      | 0–29          |
    """
    for threshold, letter in LETTER_GRADE_THRESHOLDS:
        if grade >= threshold:
            return letter
    return "F"
