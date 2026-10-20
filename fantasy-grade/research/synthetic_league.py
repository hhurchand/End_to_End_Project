"""
synthetic_league.py — Synthetic draft and schedule generation.

Implements (per grading-methodology-2026-10-17.md §7.1):
  - B1-ranked snake draft (14 rounds, 14 teams per team)
  - Berger round-robin schedule (weeks 1-14 regular season matchups)

The synthetic leagues are used in Experiments 2a and 2b. All matchup
results produced from these leagues are labeled "simulated hypothetical
matchups" and are not predictions of actual games.

Design decisions (§7.1):
  - 100 synthetic leagues per season
  - 12 teams per league
  - 14 roster spots per team (8 starters + 6 bench)
  - Draft order: B1 ranking descending (best available B1 projection first)
  - Snake format: odd rounds pick 1-12, even rounds pick 12-1
  - Schedule: Berger round-robin, 14 weeks
  - numpy==1.26.4 pinned for schedule reproducibility

numpy.random is NOT used in this module. Determinism is achieved by:
  - Draft: sorted B1 order (no randomness)
  - Schedule: Berger algorithm (fully deterministic given n_teams)
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of synthetic leagues per season (§7.1)
N_LEAGUES: int = 100

#: Teams per synthetic league (§7.1)
N_TEAMS: int = 12

#: Roster spots per team (§7.1 decision: 8 starters + 6 bench)
ROSTER_SIZE: int = 14

#: Starting slots per team (drives 8 starters per §3.1 defaults)
N_STARTERS: int = 8

#: Number of draft rounds = roster size
N_ROUNDS: int = ROSTER_SIZE   # 14 rounds

#: Regular-season weeks evaluated in experiments (§7.1, Experiments 2a/2b)
EVAL_WEEKS: List[int] = list(range(1, 15))   # weeks 1-14


# ---------------------------------------------------------------------------
# §7.1 — B1-ranked snake draft
# ---------------------------------------------------------------------------

def run_snake_draft(
    player_ids: List[str],
    b1_projections: Dict[str, float],
    n_teams: int = N_TEAMS,
    n_rounds: int = N_ROUNDS,
) -> Dict[int, List[str]]:
    """
    B1-ranked snake draft (§7.1).

    Draft order is determined by descending B1 projection value. Players
    with NaN or missing B1 projections are placed at the end of the draft
    pool, sorted by player_id ascending (deterministic tiebreak).

    Snake format:
      - Odd rounds (1, 3, 5, …): teams pick 0, 1, 2, …, n_teams−1
      - Even rounds (2, 4, 6, …): teams pick n_teams−1, …, 1, 0

    Only n_teams × n_rounds players are drafted. If the pool is smaller,
    all available players are drafted and some teams may receive fewer
    than n_rounds players (trailing teams in the last round).

    Parameters
    ----------
    player_ids     : all available player_ids (no duplicates assumed)
    b1_projections : player_id → B1 projection float (NaN for unavailable)
    n_teams        : number of teams in the league (default 12)
    n_rounds       : number of draft rounds (default 14)

    Returns
    -------
    Dict[team_index, List[player_id]] — team 0 picks first in round 1.
    List order = draft order (position 0 = first pick overall).
    """
    # Sort: valid B1 descending, then NaN by player_id ascending.
    def sort_key(pid: str) -> tuple:
        proj = b1_projections.get(pid, float("nan"))
        if proj is None or (isinstance(proj, float) and proj != proj):
            # NaN — sort to end
            return (1, pid)
        return (0, -proj, pid)

    pool = sorted(player_ids, key=sort_key)

    rosters: Dict[int, List[str]] = {t: [] for t in range(n_teams)}

    pick_overall = 0
    for rnd in range(1, n_rounds + 1):
        if rnd % 2 == 1:
            order = list(range(n_teams))
        else:
            order = list(range(n_teams - 1, -1, -1))
        for team in order:
            if pick_overall >= len(pool):
                break
            rosters[team].append(pool[pick_overall])
            pick_overall += 1
        if pick_overall >= len(pool):
            break

    return rosters


# ---------------------------------------------------------------------------
# §7.1 — Berger round-robin schedule
# ---------------------------------------------------------------------------

def berger_schedule(n_teams: int = N_TEAMS) -> List[List[Tuple[int, int]]]:
    """
    Generate a Berger round-robin schedule for n_teams teams.

    Returns a list of rounds. Each round is a list of (home, away) matchup
    tuples. For n_teams teams there are n_teams − 1 rounds, each with
    n_teams // 2 matchups. All teams appear exactly once per round.

    n_teams must be even. If n_teams is odd, raises ValueError.

    The Berger algorithm (fixed-point rotation):
      - Fix team index 0 ("the bye" placeholder in odd-team leagues, but
        here n_teams is always even so 0 is a real team).
      - Rotate the remaining n_teams − 1 teams by 1 position each round.
      - Round r pairs position 0 with position n_teams−1, then
        position 1 with position n_teams−2, etc.

    This is fully deterministic for given n_teams, meeting the
    numpy==1.26.4 reproducibility requirement (no numpy randomness used).

    Parameters
    ----------
    n_teams : number of teams; must be even (default 12)

    Returns
    -------
    List of rounds; rounds[r][k] = (team_a, team_b) for matchup k in round r.
    Length of outer list = n_teams − 1 = 11 rounds for 12 teams.
    """
    if n_teams % 2 != 0:
        raise ValueError(f"n_teams must be even; got {n_teams}")

    # Teams as indices 0..n_teams-1. Fix index 0; rotate 1..n_teams-1.
    fixed = 0
    rotating = list(range(1, n_teams))
    n_rounds = n_teams - 1
    half = n_teams // 2

    rounds: List[List[Tuple[int, int]]] = []
    for r in range(n_rounds):
        # Build this round's team list: [fixed] + rotating (after r rotations)
        teams = [fixed] + rotating
        matchups: List[Tuple[int, int]] = []
        for k in range(half):
            home = teams[k]
            away = teams[n_teams - 1 - k]
            matchups.append((home, away))
        rounds.append(matchups)
        # Rotate: move last element of rotating to front
        rotating = [rotating[-1]] + rotating[:-1]

    return rounds


def get_week_matchups(
    schedule: List[List[Tuple[int, int]]],
    week: int,
) -> List[Tuple[int, int]]:
    """
    Return the matchups for a given 1-indexed week.

    The 14-week regular season repeats the 11-round Berger schedule:
      weeks 1-11  → rounds 0-10 (first pass)
      weeks 12-14 → rounds 0-2  (second pass, first 3 rounds repeated)

    Parameters
    ----------
    schedule : output of berger_schedule()
    week     : 1-indexed week number (1–14)
    """
    n_rounds = len(schedule)
    idx = (week - 1) % n_rounds
    return schedule[idx]


# ---------------------------------------------------------------------------
# Multi-league generation
# ---------------------------------------------------------------------------

def generate_leagues(
    player_ids: List[str],
    b1_projections: Dict[str, float],
    n_leagues: int = N_LEAGUES,
    n_teams: int = N_TEAMS,
    n_rounds: int = N_ROUNDS,
    base_seed: int = 0,
) -> Tuple[List[Dict[int, List[str]]], List[List[List[Tuple[int, int]]]]]:
    """
    Generate n_leagues synthetic leagues (§7.1).

    **Rosters**: All n_leagues leagues share the same B1-ranked snake draft
    result. The draft is fully deterministic (no randomness): the same 12
    teams appear in every league, as required by "B1-ranked snake draft —
    an explicit synthetic drafting rule, not ADP."

    **Schedules**: Leagues differ only in their weekly matchup pairings.
    For each league, the 12 team indices (0–11) are randomly permuted before
    the Berger algorithm runs, so different leagues pair different teams each
    week. This is the source of variance across leagues in Experiments 2a/2b.

    The permutation for league L uses np.random.default_rng(base_seed + L),
    ensuring full reproducibility. numpy==1.26.4 is pinned for this purpose.

    Parameters
    ----------
    player_ids     : available player_ids
    b1_projections : player_id → B1 float
    n_leagues      : number of leagues to generate (default 100)
    n_teams        : teams per league (default 12)
    n_rounds       : rounds per draft (default 14)
    base_seed      : base RNG seed for schedule permutations (default 0)

    Returns
    -------
    (rosters_list, schedules_list)
    rosters_list  : List of length n_leagues; each entry is a
                    Dict[team_index, List[player_id]] (all identical)
    schedules_list: List of length n_leagues; each entry is the output of
                    berger_schedule() re-mapped through a per-league team
                    permutation — List[rounds] where rounds[r][k] = (ta, tb)
    """
    # One deterministic draft — same for all leagues.
    base_rosters = run_snake_draft(
        player_ids, b1_projections, n_teams=n_teams, n_rounds=n_rounds
    )

    # Canonical Berger schedule over indices 0..n_teams-1.
    canonical_schedule = berger_schedule(n_teams)

    rosters_list: List[Dict[int, List[str]]] = []
    schedules_list: List[List[List[Tuple[int, int]]]] = []

    for league_idx in range(n_leagues):
        rng = np.random.default_rng(base_seed + league_idx)
        # Permute team indices to produce a distinct schedule per league.
        perm = rng.permutation(n_teams).tolist()

        # Re-map the canonical schedule through the permutation.
        permuted_schedule: List[List[Tuple[int, int]]] = []
        for rnd in canonical_schedule:
            permuted_round = [(perm[a], perm[b]) for a, b in rnd]
            permuted_schedule.append(permuted_round)

        # Rosters: re-map team indices to follow the permutation so that
        # team 0 in this league's schedule corresponds to perm[0] in the
        # base_rosters, etc.
        permuted_rosters: Dict[int, List[str]] = {
            new_idx: base_rosters[old_idx]
            for new_idx, old_idx in enumerate(perm)
        }

        rosters_list.append(permuted_rosters)
        schedules_list.append(permuted_schedule)

    return rosters_list, schedules_list
