"""
Unit tests for research/team_metrics.py.

Run with:
    cd /Users/gregspunt/fantasy-grade
    python3.11 -m pytest research/tests/test_team_metrics.py -v

Covers:
  - compute_starter_par: genuine-B3 starters, fallback starters, unsaturated slots
  - compute_bench_depth_par: positive PAR only, fallback exclusion
  - compute_weakness_rate: weak categories (PAR<0, fallback, unsaturated, None)
  - compute_combined_score: formula, norm() clamp/fallback
  - compute_team_grade: tanh scaling, sigma==0 hard stop
  - compute_team_grade_starter_par: same contract as team grade
  - compute_player_grade: clamp scaling, invalid par_95th
  - numeric_to_letter: all thresholds including boundaries
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from team_metrics import (
    compute_starter_par,
    compute_bench_depth_par,
    compute_weakness_rate,
    compute_combined_score,
    compute_team_grade,
    compute_team_grade_starter_par,
    compute_player_grade,
    numeric_to_letter,
    _norm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assigned(*pids):
    """Build an assigned dict {0: pids[0], 1: pids[1], ...}."""
    return {i: pid for i, pid in enumerate(pids)}


# ---------------------------------------------------------------------------
# compute_starter_par
# ---------------------------------------------------------------------------

class TestStarterPAR(unittest.TestCase):

    def test_all_genuine_positive(self):
        assigned = _assigned("QB1", "RB1", "RB2")
        par = {"QB1": 5.0, "RB1": 3.0, "RB2": 2.0}
        genuine = frozenset({"QB1", "RB1", "RB2"})
        result = compute_starter_par(assigned, par, genuine)
        self.assertAlmostEqual(result, 10.0)

    def test_fallback_starter_contributes_zero(self):
        # RB2 is a fallback player — not in genuine_b3_set.
        assigned = _assigned("QB1", "RB1", "RB2")
        par = {"QB1": 5.0, "RB1": 3.0, "RB2": 7.0}
        genuine = frozenset({"QB1", "RB1"})  # RB2 excluded
        result = compute_starter_par(assigned, par, genuine)
        self.assertAlmostEqual(result, 8.0)  # QB1+RB1 only

    def test_unsaturated_slot_contributes_zero(self):
        # Only 2 of 3 slots filled.
        assigned = _assigned("QB1", "RB1")  # third slot unsaturated
        par = {"QB1": 5.0, "RB1": 3.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_starter_par(assigned, par, genuine)
        self.assertAlmostEqual(result, 8.0)

    def test_par_none_contributes_zero(self):
        # Genuine B3 starter but PAR is None (position exhausted).
        assigned = _assigned("QB1")
        par = {"QB1": None}
        genuine = frozenset({"QB1"})
        result = compute_starter_par(assigned, par, genuine)
        self.assertAlmostEqual(result, 0.0)

    def test_negative_par_included(self):
        # Negative PAR should still be summed (not clamped to 0).
        assigned = _assigned("QB1", "RB1")
        par = {"QB1": -2.0, "RB1": 5.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_starter_par(assigned, par, genuine)
        self.assertAlmostEqual(result, 3.0)

    def test_empty_lineup_returns_zero(self):
        result = compute_starter_par({}, {}, frozenset())
        self.assertAlmostEqual(result, 0.0)

    def test_mixed_genuine_fallback_unsaturated(self):
        # 4 slots total: 2 genuine, 1 fallback, 1 unsaturated
        assigned = _assigned("QB1", "RB1", "WR_fb")  # 3 filled, 1 unsaturated
        par = {"QB1": 4.0, "RB1": 6.0, "WR_fb": 10.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_starter_par(assigned, par, genuine)
        self.assertAlmostEqual(result, 10.0)  # QB1 + RB1 only


# ---------------------------------------------------------------------------
# compute_bench_depth_par
# ---------------------------------------------------------------------------

class TestBenchDepthPAR(unittest.TestCase):

    def test_positive_bench_par_summed(self):
        roster = ["QB1", "RB1", "RB2", "WR1"]
        assigned = _assigned("QB1", "RB1")
        par = {"QB1": 5.0, "RB1": 3.0, "RB2": 4.0, "WR1": 2.0}
        genuine = frozenset({"QB1", "RB1", "RB2", "WR1"})
        result = compute_bench_depth_par(roster, assigned, par, genuine)
        self.assertAlmostEqual(result, 6.0)  # RB2 + WR1

    def test_negative_bench_par_excluded(self):
        roster = ["QB1", "RB1", "RB2"]
        assigned = _assigned("QB1")
        par = {"QB1": 5.0, "RB1": 3.0, "RB2": -1.0}
        genuine = frozenset({"QB1", "RB1", "RB2"})
        result = compute_bench_depth_par(roster, assigned, par, genuine)
        self.assertAlmostEqual(result, 3.0)  # RB1 only (RB2 negative)

    def test_zero_par_excluded(self):
        roster = ["QB1", "RB1"]
        assigned = _assigned("QB1")
        par = {"QB1": 5.0, "RB1": 0.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_bench_depth_par(roster, assigned, par, genuine)
        self.assertAlmostEqual(result, 0.0)  # PAR=0 not included (> 0 required)

    def test_fallback_bench_excluded(self):
        roster = ["QB1", "RB1", "WR_fb"]
        assigned = _assigned("QB1")
        par = {"QB1": 5.0, "RB1": 3.0, "WR_fb": 10.0}
        genuine = frozenset({"QB1", "RB1"})  # WR_fb is fallback
        result = compute_bench_depth_par(roster, assigned, par, genuine)
        self.assertAlmostEqual(result, 3.0)  # RB1 only

    def test_par_none_excluded(self):
        roster = ["QB1", "RB1"]
        assigned = _assigned("QB1")
        par = {"QB1": 5.0, "RB1": None}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_bench_depth_par(roster, assigned, par, genuine)
        self.assertAlmostEqual(result, 0.0)

    def test_starters_excluded_from_bench(self):
        # QB1 is a starter — should not contribute to bench depth
        roster = ["QB1", "RB1"]
        assigned = _assigned("QB1", "RB1")
        par = {"QB1": 10.0, "RB1": 10.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_bench_depth_par(roster, assigned, par, genuine)
        self.assertAlmostEqual(result, 0.0)


# ---------------------------------------------------------------------------
# compute_weakness_rate
# ---------------------------------------------------------------------------

class TestWeaknessRate(unittest.TestCase):

    def test_no_weak_starters(self):
        assigned = _assigned("QB1", "RB1", "WR1")
        par = {"QB1": 1.0, "RB1": 0.5, "WR1": 0.1}
        genuine = frozenset({"QB1", "RB1", "WR1"})
        result = compute_weakness_rate(assigned, 3, par, genuine)
        self.assertAlmostEqual(result, 0.0)

    def test_all_slots_negative_par(self):
        assigned = _assigned("QB1", "RB1")
        par = {"QB1": -1.0, "RB1": -2.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_weakness_rate(assigned, 2, par, genuine)
        self.assertAlmostEqual(result, 1.0)

    def test_unsaturated_slots_counted_weak(self):
        # 4 total slots, 2 filled (both strong), 2 unsaturated
        assigned = _assigned("QB1", "RB1")
        par = {"QB1": 5.0, "RB1": 3.0}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_weakness_rate(assigned, 4, par, genuine)
        self.assertAlmostEqual(result, 0.5)  # 2 unsaturated / 4 total

    def test_fallback_starter_counted_weak(self):
        # 3 slots: 1 genuine strong, 1 genuine negative, 1 fallback
        assigned = _assigned("QB1", "RB1", "WR_fb")
        par = {"QB1": 5.0, "RB1": -1.0, "WR_fb": 8.0}
        genuine = frozenset({"QB1", "RB1"})  # WR_fb is fallback
        result = compute_weakness_rate(assigned, 3, par, genuine)
        # RB1 (PAR<0) + WR_fb (fallback) = 2 weak out of 3
        self.assertAlmostEqual(result, 2.0 / 3.0)

    def test_par_none_counted_weak(self):
        # Genuine B3 but PAR is None → weak
        assigned = _assigned("QB1", "RB1")
        par = {"QB1": 5.0, "RB1": None}
        genuine = frozenset({"QB1", "RB1"})
        result = compute_weakness_rate(assigned, 2, par, genuine)
        self.assertAlmostEqual(result, 0.5)  # RB1 → weak

    def test_zero_slots_returns_zero(self):
        result = compute_weakness_rate({}, 0, {}, frozenset())
        self.assertAlmostEqual(result, 0.0)

    def test_mixed_categories(self):
        # 8 slots: 4 genuine+positive, 1 genuine+negative, 1 fallback, 1 None PAR, 1 unsaturated
        assigned = _assigned("p1", "p2", "p3", "p4", "p5", "p6", "p7")
        par = {"p1": 5.0, "p2": 3.0, "p3": 1.0, "p4": 0.1,
               "p5": -1.0, "p6": 8.0, "p7": None}
        genuine = frozenset({"p1", "p2", "p3", "p4", "p5", "p7"})  # p6 is fallback
        result = compute_weakness_rate(assigned, 8, par, genuine)
        # weak: p5 (neg), p6 (fallback), p7 (None), 1 unsaturated = 4/8
        self.assertAlmostEqual(result, 4.0 / 8.0)


# ---------------------------------------------------------------------------
# _norm and compute_combined_score
# ---------------------------------------------------------------------------

class TestNorm(unittest.TestCase):

    def test_at_p5(self):
        self.assertAlmostEqual(_norm(10.0, 10.0, 20.0), 0.0)

    def test_at_p95(self):
        self.assertAlmostEqual(_norm(20.0, 10.0, 20.0), 1.0)

    def test_midpoint(self):
        self.assertAlmostEqual(_norm(15.0, 10.0, 20.0), 0.5)

    def test_below_p5_clamped_to_zero(self):
        self.assertAlmostEqual(_norm(0.0, 10.0, 20.0), 0.0)

    def test_above_p95_clamped_to_one(self):
        self.assertAlmostEqual(_norm(30.0, 10.0, 20.0), 1.0)

    def test_degenerate_p95_equals_p5_returns_half(self):
        self.assertAlmostEqual(_norm(15.0, 10.0, 10.0), 0.5)
        self.assertAlmostEqual(_norm(0.0, 5.0, 5.0), 0.5)


class TestCombinedScore(unittest.TestCase):

    def test_all_at_p95_no_weakness(self):
        # norm(sp)=1, norm(bd)=1, weakness=0 → 50+25+15+10 = 100
        result = compute_combined_score(
            starter_par=20.0, bench_depth_par=10.0, weakness_rate=0.0,
            starter_par_bounds=(0.0, 20.0), bench_depth_par_bounds=(0.0, 10.0),
        )
        self.assertAlmostEqual(result, 100.0)

    def test_all_at_p5_full_weakness(self):
        # norm(sp)=0, norm(bd)=0, weakness=1 → 50+0+0+0 = 50
        result = compute_combined_score(
            starter_par=0.0, bench_depth_par=0.0, weakness_rate=1.0,
            starter_par_bounds=(0.0, 20.0), bench_depth_par_bounds=(0.0, 10.0),
        )
        self.assertAlmostEqual(result, 50.0)

    def test_midpoint_no_weakness(self):
        # norm(sp)=0.5, norm(bd)=0.5, weakness=0 → 50+12.5+7.5+10 = 80
        result = compute_combined_score(
            starter_par=10.0, bench_depth_par=5.0, weakness_rate=0.0,
            starter_par_bounds=(0.0, 20.0), bench_depth_par_bounds=(0.0, 10.0),
        )
        self.assertAlmostEqual(result, 80.0)

    def test_degenerate_bounds_use_half(self):
        # norm() returns 0.5 when P5==P95 → 50+12.5+7.5+10 = 80
        result = compute_combined_score(
            starter_par=100.0, bench_depth_par=100.0, weakness_rate=0.0,
            starter_par_bounds=(5.0, 5.0), bench_depth_par_bounds=(5.0, 5.0),
        )
        self.assertAlmostEqual(result, 80.0)

    def test_weakness_penalty(self):
        # norm(sp)=1, norm(bd)=1, weakness=0.5 → 50+25+15+5 = 95
        result = compute_combined_score(
            starter_par=20.0, bench_depth_par=10.0, weakness_rate=0.5,
            starter_par_bounds=(0.0, 20.0), bench_depth_par_bounds=(0.0, 10.0),
        )
        self.assertAlmostEqual(result, 95.0)


# ---------------------------------------------------------------------------
# compute_team_grade
# ---------------------------------------------------------------------------

class TestTeamGrade(unittest.TestCase):

    def test_at_mean_returns_50(self):
        result = compute_team_grade(combined_score=75.0, mu=75.0, sigma=10.0)
        self.assertAlmostEqual(result, 50.0)

    def test_large_positive_z_approaches_100(self):
        result = compute_team_grade(combined_score=150.0, mu=75.0, sigma=10.0)
        self.assertGreater(result, 99.0)

    def test_large_negative_z_approaches_0(self):
        result = compute_team_grade(combined_score=0.0, mu=75.0, sigma=10.0)
        self.assertLess(result, 1.0)

    def test_output_in_range(self):
        for cs in [0, 50, 75, 100]:
            result = compute_team_grade(cs, mu=75.0, sigma=10.0)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 100.0)

    def test_sigma_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_team_grade(75.0, mu=75.0, sigma=0.0)

    def test_symmetry(self):
        # grade at +x above mean and −x below mean should be symmetric around 50
        above = compute_team_grade(85.0, mu=75.0, sigma=10.0)
        below = compute_team_grade(65.0, mu=75.0, sigma=10.0)
        self.assertAlmostEqual(above + below, 100.0, places=6)


class TestTeamGradeStarterPAR(unittest.TestCase):

    def test_at_mean_returns_50(self):
        result = compute_team_grade_starter_par(starter_par=10.0, mu=10.0, sigma=5.0)
        self.assertAlmostEqual(result, 50.0)

    def test_sigma_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_team_grade_starter_par(10.0, mu=10.0, sigma=0.0)


# ---------------------------------------------------------------------------
# compute_player_grade
# ---------------------------------------------------------------------------

class TestPlayerGrade(unittest.TestCase):

    def test_par_zero_returns_50(self):
        result = compute_player_grade(par=0.0, par_95th=10.0)
        self.assertAlmostEqual(result, 50.0)

    def test_par_at_par_95th_returns_100(self):
        result = compute_player_grade(par=10.0, par_95th=10.0)
        self.assertAlmostEqual(result, 100.0)

    def test_par_at_neg_par_95th_returns_0(self):
        result = compute_player_grade(par=-10.0, par_95th=10.0)
        self.assertAlmostEqual(result, 0.0)

    def test_par_above_par_95th_clamped_to_100(self):
        result = compute_player_grade(par=20.0, par_95th=10.0)
        self.assertAlmostEqual(result, 100.0)

    def test_par_below_neg_par_95th_clamped_to_0(self):
        result = compute_player_grade(par=-20.0, par_95th=10.0)
        self.assertAlmostEqual(result, 0.0)

    def test_midpoint(self):
        result = compute_player_grade(par=5.0, par_95th=10.0)
        self.assertAlmostEqual(result, 75.0)

    def test_par_95th_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_player_grade(par=5.0, par_95th=0.0)

    def test_par_95th_negative_raises(self):
        with self.assertRaises(ValueError):
            compute_player_grade(par=5.0, par_95th=-1.0)

    def test_par_95th_nan_raises(self):
        with self.assertRaises(ValueError):
            compute_player_grade(par=5.0, par_95th=float("nan"))

    def test_par_95th_inf_raises(self):
        with self.assertRaises(ValueError):
            compute_player_grade(par=5.0, par_95th=float("inf"))

    def test_output_always_in_0_100(self):
        for par in [-50, -10, -5, 0, 5, 10, 50]:
            result = compute_player_grade(par=float(par), par_95th=10.0)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 100.0)


# ---------------------------------------------------------------------------
# numeric_to_letter
# ---------------------------------------------------------------------------

class TestNumericToLetter(unittest.TestCase):

    def test_a_at_80(self):
        self.assertEqual(numeric_to_letter(80.0), "A")

    def test_a_at_100(self):
        self.assertEqual(numeric_to_letter(100.0), "A")

    def test_a_at_90(self):
        self.assertEqual(numeric_to_letter(90.0), "A")

    def test_b_at_65(self):
        self.assertEqual(numeric_to_letter(65.0), "B")

    def test_b_at_79(self):
        self.assertEqual(numeric_to_letter(79.9), "B")

    def test_c_at_45(self):
        self.assertEqual(numeric_to_letter(45.0), "C")

    def test_c_at_64(self):
        self.assertEqual(numeric_to_letter(64.9), "C")

    def test_d_at_30(self):
        self.assertEqual(numeric_to_letter(30.0), "D")

    def test_d_at_44(self):
        self.assertEqual(numeric_to_letter(44.9), "D")

    def test_f_at_29(self):
        self.assertEqual(numeric_to_letter(29.9), "F")

    def test_f_at_0(self):
        self.assertEqual(numeric_to_letter(0.0), "F")

    def test_boundary_a_b(self):
        # 80 is A; 79.99 is B
        self.assertEqual(numeric_to_letter(80.0), "A")
        self.assertEqual(numeric_to_letter(79.99), "B")

    def test_boundary_b_c(self):
        self.assertEqual(numeric_to_letter(65.0), "B")
        self.assertEqual(numeric_to_letter(64.99), "C")

    def test_boundary_c_d(self):
        self.assertEqual(numeric_to_letter(45.0), "C")
        self.assertEqual(numeric_to_letter(44.99), "D")

    def test_boundary_d_f(self):
        self.assertEqual(numeric_to_letter(30.0), "D")
        self.assertEqual(numeric_to_letter(29.99), "F")


if __name__ == "__main__":
    unittest.main()
