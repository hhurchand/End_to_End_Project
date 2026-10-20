"""
Unit tests for research/synthetic_league.py.

Run with:
    cd /Users/gregspunt/fantasy-grade
    python3.11 -m pytest research/tests/test_synthetic_league.py -v

Covers:
  - run_snake_draft: ordering, snake direction, NaN handling, short pool
  - berger_schedule: round count, matchup count, full coverage, no self-play
  - get_week_matchups: correct round mapping, wrap-around weeks 12-14
  - generate_leagues: roster count, schedule variation, determinism, team coverage
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthetic_league import (
    run_snake_draft,
    berger_schedule,
    get_week_matchups,
    generate_leagues,
    N_TEAMS,
    N_ROUNDS,
    EVAL_WEEKS,
)


# ---------------------------------------------------------------------------
# run_snake_draft
# ---------------------------------------------------------------------------

class TestSnakeDraft(unittest.TestCase):

    def _pool(self, n):
        """n players: p00..pNN with b1 values n..1 (p00 is best)."""
        ids = [f"p{i:02d}" for i in range(n)]
        projs = {pid: float(n - i) for i, pid in enumerate(ids)}
        return ids, projs

    def test_first_pick_is_best_b1(self):
        ids, projs = self._pool(24)
        rosters = run_snake_draft(ids, projs, n_teams=12, n_rounds=2)
        self.assertEqual(rosters[0][0], "p00")  # team 0 picks first overall

    def test_snake_direction_round2(self):
        # Round 1: teams 0-11 pick in order; round 2: teams 11-0 pick
        ids, projs = self._pool(24)
        rosters = run_snake_draft(ids, projs, n_teams=12, n_rounds=2)
        # In round 2 (even), team 11 picks 13th overall (index 12)
        # i.e., rosters[11][1] is the 13th player picked = ids[12]
        self.assertEqual(rosters[11][1], "p12")

    def test_each_player_drafted_once(self):
        ids, projs = self._pool(168)
        rosters = run_snake_draft(ids, projs, n_teams=12, n_rounds=14)
        all_picks = [pid for team in rosters.values() for pid in team]
        self.assertEqual(len(all_picks), len(set(all_picks)))

    def test_total_picks_correct(self):
        ids, projs = self._pool(168)
        rosters = run_snake_draft(ids, projs, n_teams=12, n_rounds=14)
        total = sum(len(v) for v in rosters.values())
        self.assertEqual(total, 12 * 14)

    def test_each_team_gets_n_rounds_players(self):
        ids, projs = self._pool(168)
        rosters = run_snake_draft(ids, projs, n_teams=12, n_rounds=14)
        for team, picks in rosters.items():
            self.assertEqual(len(picks), 14, f"team {team} has {len(picks)} picks")

    def test_nan_projection_sorted_to_end(self):
        ids = ["p0", "p1", "p2"]
        projs = {"p0": float("nan"), "p1": 10.0, "p2": 5.0}
        rosters = run_snake_draft(ids, projs, n_teams=1, n_rounds=3)
        # p1 (10.0) first, p2 (5.0) second, p0 (NaN) last
        self.assertEqual(rosters[0], ["p1", "p2", "p0"])

    def test_short_pool_no_error(self):
        # Pool smaller than n_teams × n_rounds — should not raise
        ids, projs = self._pool(10)
        rosters = run_snake_draft(ids, projs, n_teams=12, n_rounds=14)
        total = sum(len(v) for v in rosters.values())
        self.assertEqual(total, 10)

    def test_deterministic(self):
        ids, projs = self._pool(168)
        r1 = run_snake_draft(ids, projs, n_teams=12, n_rounds=14)
        r2 = run_snake_draft(ids, projs, n_teams=12, n_rounds=14)
        self.assertEqual(r1, r2)

    def test_tiebreak_by_player_id_ascending(self):
        # Two players with same b1 → earlier player_id wins
        ids = ["pb", "pa"]
        projs = {"pa": 5.0, "pb": 5.0}
        rosters = run_snake_draft(ids, projs, n_teams=1, n_rounds=2)
        self.assertEqual(rosters[0][0], "pa")


# ---------------------------------------------------------------------------
# berger_schedule
# ---------------------------------------------------------------------------

class TestBergerSchedule(unittest.TestCase):

    def test_n_rounds_equals_n_teams_minus_1(self):
        sched = berger_schedule(12)
        self.assertEqual(len(sched), 11)

    def test_n_matchups_per_round(self):
        sched = berger_schedule(12)
        for rnd in sched:
            self.assertEqual(len(rnd), 6)

    def test_every_team_appears_once_per_round(self):
        sched = berger_schedule(12)
        for r, rnd in enumerate(sched):
            teams_in_round = set()
            for a, b in rnd:
                teams_in_round.add(a)
                teams_in_round.add(b)
            self.assertEqual(teams_in_round, set(range(12)), f"round {r}")

    def test_no_self_play(self):
        sched = berger_schedule(12)
        for rnd in sched:
            for a, b in rnd:
                self.assertNotEqual(a, b)

    def test_all_matchups_involve_valid_teams(self):
        sched = berger_schedule(12)
        for rnd in sched:
            for a, b in rnd:
                self.assertIn(a, range(12))
                self.assertIn(b, range(12))

    def test_each_pair_plays_exactly_once_in_11_rounds(self):
        # Over 11 rounds every pair of teams meets exactly once
        sched = berger_schedule(12)
        seen = set()
        for rnd in sched:
            for a, b in rnd:
                pair = frozenset({a, b})
                self.assertNotIn(pair, seen, f"pair {pair} plays twice")
                seen.add(pair)
        self.assertEqual(len(seen), 12 * 11 // 2)

    def test_odd_n_teams_raises(self):
        with self.assertRaises(ValueError):
            berger_schedule(11)

    def test_deterministic(self):
        s1 = berger_schedule(12)
        s2 = berger_schedule(12)
        self.assertEqual(s1, s2)

    def test_works_for_various_even_sizes(self):
        for n in [4, 6, 8, 10, 12, 14]:
            sched = berger_schedule(n)
            self.assertEqual(len(sched), n - 1)


# ---------------------------------------------------------------------------
# get_week_matchups
# ---------------------------------------------------------------------------

class TestGetWeekMatchups(unittest.TestCase):

    def setUp(self):
        self.sched = berger_schedule(12)  # 11 rounds

    def test_week1_returns_round0(self):
        self.assertEqual(get_week_matchups(self.sched, 1), self.sched[0])

    def test_week11_returns_round10(self):
        self.assertEqual(get_week_matchups(self.sched, 11), self.sched[10])

    def test_week12_wraps_to_round0(self):
        self.assertEqual(get_week_matchups(self.sched, 12), self.sched[0])

    def test_week13_wraps_to_round1(self):
        self.assertEqual(get_week_matchups(self.sched, 13), self.sched[1])

    def test_week14_wraps_to_round2(self):
        self.assertEqual(get_week_matchups(self.sched, 14), self.sched[2])

    def test_all_eval_weeks_return_nonempty(self):
        for w in EVAL_WEEKS:
            matchups = get_week_matchups(self.sched, w)
            self.assertTrue(len(matchups) > 0, f"week {w} empty")


# ---------------------------------------------------------------------------
# generate_leagues
# ---------------------------------------------------------------------------

class TestGenerateLeagues(unittest.TestCase):

    def _small_setup(self, n_players=24):
        ids = [f"p{i:03d}" for i in range(n_players)]
        projs = {pid: float(n_players - i) for i, pid in enumerate(ids)}
        return ids, projs

    def test_returns_correct_number_of_leagues(self):
        ids, projs = self._small_setup(168)
        rosters_list, schedules_list = generate_leagues(
            ids, projs, n_leagues=10, n_teams=12, n_rounds=14
        )
        self.assertEqual(len(rosters_list), 10)
        self.assertEqual(len(schedules_list), 10)

    def test_all_rosters_identical(self):
        # Rosters should be identical across leagues (same deterministic draft)
        ids, projs = self._small_setup(168)
        rosters_list, _ = generate_leagues(
            ids, projs, n_leagues=5, n_teams=12, n_rounds=14
        )
        for lg in rosters_list[1:]:
            self.assertEqual(
                sorted(p for t in lg.values() for p in t),
                sorted(p for t in rosters_list[0].values() for p in t),
            )

    def test_schedules_differ_across_leagues(self):
        ids, projs = self._small_setup(168)
        _, schedules_list = generate_leagues(
            ids, projs, n_leagues=10, n_teams=12, n_rounds=14
        )
        # Not all schedules should be identical
        week1_matchups = [frozenset(frozenset(m) for m in s[0]) for s in schedules_list]
        unique = set(
            tuple(sorted(tuple(sorted(m)) for m in s[0]))
            for s in schedules_list
        )
        self.assertGreater(len(unique), 1, "All 10 leagues have identical schedules")

    def test_each_league_has_correct_team_count(self):
        ids, projs = self._small_setup(168)
        rosters_list, schedules_list = generate_leagues(
            ids, projs, n_leagues=3, n_teams=12, n_rounds=14
        )
        for rosters in rosters_list:
            self.assertEqual(len(rosters), 12)
        for sched in schedules_list:
            for rnd in sched:
                teams = set()
                for a, b in rnd:
                    teams.add(a)
                    teams.add(b)
                self.assertEqual(teams, set(range(12)))

    def test_deterministic_with_same_seed(self):
        ids, projs = self._small_setup(168)
        r1, s1 = generate_leagues(ids, projs, n_leagues=3, n_teams=12, n_rounds=14, base_seed=42)
        r2, s2 = generate_leagues(ids, projs, n_leagues=3, n_teams=12, n_rounds=14, base_seed=42)
        self.assertEqual(r1, r2)
        self.assertEqual(s1, s2)

    def test_different_seeds_produce_different_schedules(self):
        ids, projs = self._small_setup(168)
        _, s1 = generate_leagues(ids, projs, n_leagues=1, n_teams=12, n_rounds=14, base_seed=0)
        _, s2 = generate_leagues(ids, projs, n_leagues=1, n_teams=12, n_rounds=14, base_seed=99)
        self.assertNotEqual(s1, s2)

    def test_all_players_appear_exactly_once_per_league(self):
        ids, projs = self._small_setup(168)
        rosters_list, _ = generate_leagues(
            ids, projs, n_leagues=3, n_teams=12, n_rounds=14
        )
        for rosters in rosters_list:
            all_picks = [pid for t in rosters.values() for pid in t]
            self.assertEqual(len(all_picks), len(set(all_picks)), "duplicate picks")
            self.assertEqual(len(all_picks), 168)


if __name__ == "__main__":
    unittest.main()
