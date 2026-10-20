"""
grade_engine.py — Per-week grade computation orchestrator.

Ties together baselines.py, par_engine.py, and team_metrics.py to compute
all metrics for one team-week.

Implements the walk-forward computation pipeline:
  1. Compute genuine B3 projections (compute_b3_all)
  2. Apply fallback chain (apply_fallbacks) for per-team lineup
  3. Run league-wide matching → assigned_set, replacement_pts
  4. Compute per-team lineup score
  5. Compute PAR for each genuine-B3 player
  6. Compute team aggregate metrics (starter_PAR, bench_depth_PAR, weakness_rate)
  7. Compute combined_score and grades

This module does NOT own the calibration constants (PAR_95th, μ, σ, P5/P95
bounds). Those are passed in by the caller (experiment scripts), which derive
them from the appropriate calibration windows.

Canonical identifier: player_id (string).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

from baselines import (
    compute_b0,
    compute_b1_all,
    compute_b3_all,
    apply_fallbacks,
)
from par_engine import (
    compute_league_wide_matching,
    compute_par,
    compute_per_team_lineup,
    DEFAULT_N_TEAMS,
    DEFAULT_SLOTS_PER_TEAM,
)
from team_metrics import (
    compute_bench_depth_par,
    compute_combined_score,
    compute_player_grade,
    compute_starter_par,
    compute_team_grade,
    compute_team_grade_starter_par,
    compute_weakness_rate,
    numeric_to_letter,
)


# ---------------------------------------------------------------------------
# Calibration constants container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationConstants:
    """
    Frozen calibration constants derived from the appropriate pre-evaluation
    window before any evaluation data is examined.

    All constants are position-keyed or global floats.
    """
    # PAR_95th per position (§4.2) — derived from 2012–2015 window
    par_95th: Dict[str, float]

    # Team grade calibration (§4.1) — derived from 2016–2020 window
    mu_cs: float       # mean of combined_score
    sigma_cs: float    # std of combined_score

    # Fallback team grade calibration (§4.1 Outcome 2)
    mu_sp: float       # mean of starter_PAR
    sigma_sp: float    # std of starter_PAR

    # combined_score normalization bounds (§7 Exp 2b) — 2016–2020 window
    starter_par_p5: float
    starter_par_p95: float
    bench_depth_par_p5: float
    bench_depth_par_p95: float


# ---------------------------------------------------------------------------
# Per-team week result
# ---------------------------------------------------------------------------

@dataclass
class TeamWeekResult:
    """
    All computed metrics for one team-week.

    If league_wide_success is False, PAR, starter_par, bench_depth_par,
    weakness_rate, combined_score, and grade fields are all None.
    Inspect stop_reason for the cause.
    """
    team_index: int
    season: int
    week: int

    # Per-team lineup
    lineup_score: float
    saturated: bool
    unsaturated_slot_types: List[str]

    # League-wide matching outcome
    league_wide_success: bool
    stop_reason: Optional[str]

    # Per-player PAR (genuine-B3 players only)
    par_values: Dict[str, Optional[float]]   # player_id → PAR or None

    # Team aggregate metrics
    starter_par: Optional[float]
    bench_depth_par: Optional[float]
    weakness_rate: Optional[float]
    combined_score: Optional[float]

    # Grades
    team_grade_numeric: Optional[float]
    team_grade_letter: Optional[str]
    team_grade_fallback_numeric: Optional[float]   # using starter_par only


@dataclass
class PlayerWeekResult:
    """
    Grade for one player-week (used in Experiment 1 — monotonicity)."""
    player_id: str
    season: int
    week: int
    position: str
    genuine_b3: float
    par: Optional[float]
    grade_numeric: Optional[float]
    grade_letter: Optional[str]


# ---------------------------------------------------------------------------
# League-week computation
# ---------------------------------------------------------------------------

def compute_league_week(
    season: int,
    week: int,
    historical: pd.DataFrame,
    rosters: Dict[int, List[str]],   # team_index → [player_id, ...]
    positions: Dict[str, str],       # player_id → position
    calibration: CalibrationConstants,
    format_col: str = "pts_ppr",
    n_teams: int = DEFAULT_N_TEAMS,
    slots_per_team: Dict[str, int] = DEFAULT_SLOTS_PER_TEAM,
) -> Tuple[List[TeamWeekResult], List[PlayerWeekResult]]:
    """
    Compute all metrics for all teams in one league-week.

    Walk-forward: `historical` contains only data strictly before this week.
    No information from this week or any future week is used.

    Parameters
    ----------
    season      : NFL season year (e.g. 2021)
    week        : NFL week number (1-indexed)
    historical  : features DataFrame with data strictly before this week
    rosters     : team_index → list of player_ids
    positions   : player_id → primary position string
    calibration : frozen calibration constants
    format_col  : scoring column ('pts_ppr', 'pts_std', 'pts_half')
    n_teams     : number of teams (default 12)
    slots_per_team : starting slot configuration

    Returns
    -------
    (team_results, player_results)
    team_results   : one TeamWeekResult per team
    player_results : one PlayerWeekResult per genuine-B3 player-week
    """
    # ── Step 1: Compute projections ──────────────────────────────────────
    b3_series = compute_b3_all(historical, format_col)
    b1_series = compute_b1_all(historical, format_col, season)
    b0_dict   = compute_b0(historical, format_col, season)

    # genuine_b3_projections: only players with non-NaN B3
    genuine_b3_projections: Dict[str, float] = {
        pid: float(b3_series[pid])
        for pid in b3_series.index
        if pd.notna(b3_series[pid])
    }
    genuine_b3_set: FrozenSet[str] = frozenset(genuine_b3_projections.keys())

    # Player positions series for apply_fallbacks
    pos_series = pd.Series(positions)

    # Effective projections (post-fallback) for per-team lineup matching
    effective_series = apply_fallbacks(
        b3_series, b1_series, b0_dict, pos_series, name="eff_proj"
    )
    effective_projections: Dict[str, float] = {
        pid: float(v) for pid, v in effective_series.items() if pd.notna(v)
    }

    # ── Step 2: League-wide matching ────────────────────────────────────
    all_genuine_ids = sorted(genuine_b3_projections.keys())
    lw_result = compute_league_wide_matching(
        genuine_b3_player_ids=all_genuine_ids,
        genuine_b3_projections=genuine_b3_projections,
        positions=positions,
        n_teams=n_teams,
        slots_per_team=slots_per_team,
    )

    # ── Step 3: PAR for all genuine-B3 players ───────────────────────────
    par_values: Dict[str, Optional[float]] = {}
    if lw_result.success:
        for pid in genuine_b3_set:
            par_values[pid] = compute_par(
                player_id=pid,
                genuine_b3_projection=genuine_b3_projections[pid],
                primary_position=positions.get(pid, ""),
                replacement_pts=lw_result.replacement_pts,
            )

    # ── Step 4: Per-team metrics ─────────────────────────────────────────
    all_rostered = set(pid for pids in rosters.values() for pid in pids)
    team_results: List[TeamWeekResult] = []

    for team_idx, roster_ids in rosters.items():
        pt_result = compute_per_team_lineup(
            roster_player_ids=roster_ids,
            effective_projections=effective_projections,
            positions=positions,
            slots_per_team=slots_per_team,
        )

        n_total_slots = sum(slots_per_team.values())

        if not lw_result.success:
            team_results.append(TeamWeekResult(
                team_index=team_idx,
                season=season,
                week=week,
                lineup_score=pt_result.lineup_score,
                saturated=pt_result.saturated,
                unsaturated_slot_types=pt_result.unsaturated_slot_types,
                league_wide_success=False,
                stop_reason=lw_result.stop_reason,
                par_values={},
                starter_par=None,
                bench_depth_par=None,
                weakness_rate=None,
                combined_score=None,
                team_grade_numeric=None,
                team_grade_letter=None,
                team_grade_fallback_numeric=None,
            ))
            continue

        sp = compute_starter_par(pt_result.assigned, par_values, genuine_b3_set)
        bd = compute_bench_depth_par(roster_ids, pt_result.assigned, par_values, genuine_b3_set)
        wr = compute_weakness_rate(pt_result.assigned, n_total_slots, par_values, genuine_b3_set)
        cs = compute_combined_score(
            starter_par=sp,
            bench_depth_par=bd,
            weakness_rate=wr,
            starter_par_bounds=(calibration.starter_par_p5, calibration.starter_par_p95),
            bench_depth_par_bounds=(calibration.bench_depth_par_p5, calibration.bench_depth_par_p95),
        )

        # Grade (combined_score path — Outcome 1)
        try:
            tg = compute_team_grade(cs, calibration.mu_cs, calibration.sigma_cs)
            tg_letter = numeric_to_letter(tg)
        except ValueError:
            tg = None
            tg_letter = None

        # Fallback grade (starter_PAR path — Outcome 2)
        try:
            tg_fb = compute_team_grade_starter_par(sp, calibration.mu_sp, calibration.sigma_sp)
        except ValueError:
            tg_fb = None

        team_results.append(TeamWeekResult(
            team_index=team_idx,
            season=season,
            week=week,
            lineup_score=pt_result.lineup_score,
            saturated=pt_result.saturated,
            unsaturated_slot_types=pt_result.unsaturated_slot_types,
            league_wide_success=True,
            stop_reason=None,
            par_values={pid: par_values.get(pid) for pid in roster_ids if pid in par_values},
            starter_par=sp,
            bench_depth_par=bd,
            weakness_rate=wr,
            combined_score=cs,
            team_grade_numeric=tg,
            team_grade_letter=tg_letter,
            team_grade_fallback_numeric=tg_fb,
        ))

    # ── Step 5: Player-week results ──────────────────────────────────────
    player_results: List[PlayerWeekResult] = []
    if lw_result.success:
        for pid in genuine_b3_set:
            pos = positions.get(pid, "")
            if not pos:
                continue
            par = par_values.get(pid)
            par_95th_pos = calibration.par_95th.get(pos)
            if par is not None and par_95th_pos is not None and par_95th_pos > 0:
                try:
                    pg = compute_player_grade(par, par_95th_pos)
                    pg_letter = numeric_to_letter(pg)
                except ValueError:
                    pg = None
                    pg_letter = None
            else:
                pg = None
                pg_letter = None
            player_results.append(PlayerWeekResult(
                player_id=pid,
                season=season,
                week=week,
                position=pos,
                genuine_b3=genuine_b3_projections[pid],
                par=par,
                grade_numeric=pg,
                grade_letter=pg_letter,
            ))

    return team_results, player_results
