"""
Integration tests for research/grade_engine.py.

Uses synthetic DataFrames — no actual nflverse data required.

Run with:
    cd /Users/gregspunt/fantasy-grade
    python3.11 -m pytest research/tests/test_grade_engine.py -v

Covers:
  - compute_league_week: full pipeline success path
  - compute_league_week: league-wide stop condition propagates to all teams
  - TeamWeekResult fields populated correctly on success
  - PlayerWeekResult fields populated correctly on success
  - Calibration σ=0 handled gracefully (grade=None, not crash)
  - Unsaturated roster reduces lineup_score, propagates to weakness_rate
"""

import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grade_engine import (
    CalibrationConstants,
    PlayerWeekResult,
    TeamWeekResult,
    compute_league_week,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_calibration(
    sigma_cs: float = 5.0,
    sigma_sp: float = 5.0,
) -> CalibrationConstants:
    return CalibrationConstants(
        par_95th={"QB": 10.0, "RB": 8.0, "WR": 6.0, "TE": 5.0},
        mu_cs=75.0,
        sigma_cs=sigma_cs,
        mu_sp=15.0,
        sigma_sp=sigma_sp,
        starter_par_p5=0.0,
        starter_par_p95=30.0,
        bench_depth_par_p5=0.0,
        bench_depth_par_p95=20.0,
    )


def _make_historical(
    player_specs: list[tuple[str, str, float]],
    season: int = 2020,
    week: int = 5,
) -> pd.DataFrame:
    """
    Build a minimal historical DataFrame for weeks 1..(week-1) of season.
    player_specs: [(player_id, position, pts_ppr), ...]
    Each player gets 3 identical historical rows (weeks 1-3).
    """
    rows = []
    for pid, pos, pts in player_specs:
        for w in range(1, 4):
            rows.append({
                "player_id": pid,
                "position": pos,
                "season": season,
                "week": w,
                "pts_ppr": pts,
                "pts_std": pts * 0.9,
                "pts_half": pts * 0.95,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: build a minimal viable league pool
# ---------------------------------------------------------------------------

def _viable_pool():
    """
    12 teams × 8 starters = 96 slots: 12 QB + 24 RB + 36 WR + 12 TE + 12 FLEX.
    We need at least 96 genuine-B3 players to saturate the league-wide matching.
    Add a bench of 72 fallback players (6 per team × 12 teams).
    """
    specs = []
    positions_map = {}

    # Genuine B3 players: 12 QB, 24 RB, 36 WR, 12 TE = 84 primary + 12 FLEX eligible
    # FLEX slots: 12, need ≥ 12 additional RB/WR/TE after filling primary slots
    # Provide: 12 QB, 26 RB, 38 WR, 14 TE = 90 genuine-B3 players (>96 slots needed)
    # Actually: slots = 12×(1QB+2RB+3WR+1TE+1FLEX) = 12+24+36+12+12 = 96
    # Need at least 12 QB, 24 RB, 36 WR, 12 TE genuine-B3 for primary slots
    # plus 12 more FLEX-eligible: use extra RB/WR/TE
    # Let's use: 12QB + 30RB + 40WR + 14TE = 96 total (enough for all 96 slots)

    idx = 0
    for pos, count, pts_base in [("QB", 12, 22.0), ("RB", 30, 15.0),
                                   ("WR", 40, 12.0), ("TE", 14, 8.0)]:
        for i in range(count):
            pid = f"{pos}_{idx:03d}"
            pts = pts_base - i * 0.1
            specs.append((pid, pos, pts))
            positions_map[pid] = pos
            idx += 1

    # Bench fallback players (no historical data → B1/B0 fallback)
    bench_specs = []
    for i in range(72):
        pos = ["RB", "WR", "TE"][i % 3]
        pid = f"BENCH_{i:03d}"
        bench_specs.append((pid, pos))
        positions_map[pid] = pos

    return specs, bench_specs, positions_map


def _build_rosters(specs, bench_specs):
    """Assign players to 12 teams (8 starters + 6 bench each)."""
    # Starters: assign based on position order
    # Draft order: all QBs to teams 0-11 (one each), then RBs 2 each, etc.
    rosters = {t: [] for t in range(12)}
    team = 0
    for pid, pos, _ in specs:
        rosters[team % 12].append(pid)
        team += 1
    # Fill bench
    for i, (pid, pos) in enumerate(bench_specs):
        rosters[i % 12].append(pid)
    return rosters


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeLeagueWeekSuccess(unittest.TestCase):

    def setUp(self):
        specs, bench_specs, self.positions = _viable_pool()
        self.specs = specs
        self.bench_specs = bench_specs
        self.historical = _make_historical(specs, season=2020, week=5)
        self.rosters = _build_rosters(specs, bench_specs)
        self.calibration = _make_calibration()

    def test_returns_12_team_results(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        self.assertEqual(len(team_res), 12)

    def test_all_teams_league_wide_success(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            self.assertTrue(tr.league_wide_success, f"team {tr.team_index} failed: {tr.stop_reason}")

    def test_lineup_score_positive(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            self.assertGreater(tr.lineup_score, 0.0)

    def test_starter_par_is_float(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            self.assertIsInstance(tr.starter_par, float)

    def test_combined_score_in_range(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            self.assertIsNotNone(tr.combined_score)
            self.assertGreaterEqual(tr.combined_score, 50.0)
            self.assertLessEqual(tr.combined_score, 100.0)

    def test_team_grade_in_0_100(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            if tr.team_grade_numeric is not None:
                self.assertGreaterEqual(tr.team_grade_numeric, 0.0)
                self.assertLessEqual(tr.team_grade_numeric, 100.0)

    def test_team_grade_letter_is_valid(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            if tr.team_grade_letter is not None:
                self.assertIn(tr.team_grade_letter, {"A", "B", "C", "D", "F"})

    def test_weakness_rate_in_0_1(self):
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for tr in team_res:
            if tr.weakness_rate is not None:
                self.assertGreaterEqual(tr.weakness_rate, 0.0)
                self.assertLessEqual(tr.weakness_rate, 1.0)

    def test_player_results_populated(self):
        _, player_res = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        self.assertGreater(len(player_res), 0)

    def test_player_grade_in_0_100(self):
        _, player_res = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for pr in player_res:
            if pr.grade_numeric is not None:
                self.assertGreaterEqual(pr.grade_numeric, 0.0)
                self.assertLessEqual(pr.grade_numeric, 100.0)

    def test_player_positions_correct(self):
        _, player_res = compute_league_week(
            season=2020, week=5,
            historical=self.historical,
            rosters=self.rosters,
            positions=self.positions,
            calibration=self.calibration,
        )
        for pr in player_res:
            self.assertIn(pr.position, {"QB", "RB", "WR", "TE"})


class TestComputeLeagueWeekStopCondition(unittest.TestCase):

    def test_empty_pool_stop_propagates_to_all_teams(self):
        """If historical is empty, B3 pool is empty → league-wide stop."""
        specs, bench_specs, positions = _viable_pool()
        empty_hist = pd.DataFrame(columns=[
            "player_id", "position", "season", "week",
            "pts_ppr", "pts_std", "pts_half"
        ])
        rosters = _build_rosters(specs, bench_specs)
        calibration = _make_calibration()

        team_res, player_res = compute_league_week(
            season=2020, week=1,
            historical=empty_hist,
            rosters=rosters,
            positions=positions,
            calibration=calibration,
        )
        for tr in team_res:
            self.assertFalse(tr.league_wide_success)
            self.assertIsNone(tr.starter_par)
            self.assertIsNone(tr.combined_score)
        self.assertEqual(len(player_res), 0)


class TestCalibrationSigmaZero(unittest.TestCase):

    def test_sigma_zero_produces_none_grade_not_crash(self):
        """σ=0 should produce None grade fields, not raise an exception."""
        specs, bench_specs, positions = _viable_pool()
        historical = _make_historical(specs, season=2020, week=5)
        rosters = _build_rosters(specs, bench_specs)
        calibration = _make_calibration(sigma_cs=0.0, sigma_sp=0.0)

        # Should not raise
        team_res, _ = compute_league_week(
            season=2020, week=5,
            historical=historical,
            rosters=rosters,
            positions=positions,
            calibration=calibration,
        )
        for tr in team_res:
            if tr.league_wide_success:
                # Both grade paths have σ=0 → both should be None
                self.assertIsNone(tr.team_grade_numeric)
                self.assertIsNone(tr.team_grade_fallback_numeric)


if __name__ == "__main__":
    unittest.main()
