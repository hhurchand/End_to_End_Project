# FantasyGrade — Grading Methodology Specification
**Project:** FantasyGrade
**Date:** 2026-10-17
**Status:** Approved methodology — implementation not yet authorized
**Constraint:** $0 spend, no new downloads, no outreach, no paid APIs, no deployment, no spending

---

## 1. Phase 1 Result: Projection Baseline

Phase 1 backtesting evaluated the v0 shrinkage model and four baselines (B0–B3) against the 2023 dev-test season using walk-forward evaluation on nflverse data. All 18 pre-registered acceptance criteria failed for v0.

**Conclusion:** B3 is the production projection baseline. The 2024 holdout remains sealed.

**Pre-registered baseline parameters:**

| Parameter | Value |
|---|---|
| Decay factor (d) | 0.80 |
| Max active games (B3_MAX_WEEKS) | 6 |
| Min active games required (B3_MIN_ACTIVE) | 2 |
| Starter population QB | 16 |
| Starter population RB | 32 |
| Starter population WR | 40 |
| Starter population TE | 16 |

**Phase 1 MAE results (PPR, 2023 dev-test):**

| Position | B3 MAE | v0 MAE |
|---|---|---|
| QB | 6.286 | 7.741 |
| RB | 4.843 | 6.607 |
| WR | 5.110 | 7.003 |
| TE | 3.974 | 5.315 |

Root cause of v0 failure: position mean (B0) is a weak shrinkage anchor. The B3 → B1 fallback chain already handles sparse-history players correctly without shrinking toward a noisy population mean.

---

## 2. Projection Definitions

### 2.1 B3 — Primary Projection

Exponentially weighted average over the player's most recent active games.

```
pts_wtd = sum(d^(i-1) * score_i for i in 1..n) / sum(d^(i-1) for i in 1..n)
```

where `i=1` is the most recent active game, `d=0.80`, `n = min(actual_active_games, 6)`.

A game is "active" if the player recorded at least one counting stat (attempt, carry, target, or snap). Games missed due to injury, bye, or inactive designation do not count and do not break the window — the window spans season boundaries to find the most recent 6 active games regardless of calendar distance.

Minimum `n=2` active games required; if fewer, fall through to the fallback chain.

### 2.2 Fallback Chain

```
B3 (n >= 2 active games in rolling window)
  → B1 current season (season average if n < 2)
  → B1 prior season (prior season average if no current-season data)
  → B0 (position mean if no individual history available)
```

### 2.3 B2 — Recency Baseline (Used in Experiment 3)

B2 is the player's **single most recent prior active-game actual fantasy score**, regardless of calendar-week distance. It is not "last week's score" — it is the score from the most recent game the player appeared in, however many weeks ago that was.

B2 is used only in Experiment 3 (B3 vs B2 disagreement comparison). It is not used in the grading engine.

---

## 3. Core Metrics: Three Distinct Concepts

These three metrics are not interchangeable. Each answers a different question.

### 3.1 lineup_score

**Definition:** The projected output of the team's optimal starting lineup.

**Computation:** Maximum-weight bipartite matching (Hungarian algorithm / `scipy.optimize.linear_sum_assignment`) assigns players to roster slots to maximize total projected points. The sum of assigned player projections is the lineup_score.

**What it answers:** How many projected points will this team score if they set their lineup optimally?

**Important:** `sum(MLC values) ≠ lineup_score`. MLC values are counterfactual differences and do not sum to the lineup total.

### 3.2 PAR — Points Above Replacement

**Definition:** League-relative player value.

```
PAR(player) = B3(player) − replacement_pts(P)
```

where:

```
replacement_pts(P) = B3( best non-starter in the league-wide optimal matching
                         whose primary position is P )
```

The league-wide matching fills `league_size × lineup_slots` from the eligible player pool. Players assigned to a starting slot by that matching are starters. The highest-projected unassigned player at primary position P is the replacement player for that position.

**What it answers:** How much does this player contribute above freely available talent at their position?

**PAR is league-relative.** It does not vary by individual roster composition. The same player has the same PAR regardless of which team they are on.

**Provisional treatment for FLEX/SuperFLEX:** The "best non-starter at primary position P" heuristic may understate replacement value in configurations where a player could cascade into a FLEX or SuperFLEX slot. The exact replacement level in those configurations requires running the optimizer without the player to determine roster impact — but that is MLC, not PAR. PAR values for FLEX and SuperFLEX configurations are provisional until the league-wide matching procedure is extended to handle multi-eligible-position players correctly.

### 3.3 MLC — Marginal Lineup Contribution

**Definition:** Roster-specific marginal contribution.

```
MLC(player) = optimal_score(roster WITH player) − optimal_score(roster WITHOUT player)
```

**What it answers:** How much does this specific player contribute to this specific team's projected output?

**MLC is roster-specific.** The same player will have different MLC values on different teams because their presence or absence affects how other players fill remaining slots.

**Use cases for MLC:**
- Start/sit decisions: `swap_value = optimal_score(A starts, B bench) − optimal_score(B starts, A bench)`
- Exact trade value for a specific roster configuration

Do not use MLC as a league-wide player ranking signal; use PAR for that.

---

## 4. Team Grade Components

The team grade is derived from the following components, each computed separately:

| Component | Metric used | What it measures |
|---|---|---|
| **lineup_score** | Bipartite matching output | Projected starter output |
| **starter_PAR** | Sum of PAR for optimal-lineup starters | League-relative starter quality |
| **bench_depth** | Sum of positive PAR among bench players | Value available beyond optimal lineup |
| **positional_balance** | Distribution of PAR across positions | Roster construction risk |

**combined_score** is a weighted combination of these components. The specific weights are selected by Experiment 2b (formula selection, 2016–2020). Weights are not set until Experiment 2b is run.

### 4.1 Grade Scale

Grades are 0–100. The conversion from combined_score to a grade uses:

```
grade = 50 + 50 × tanh(team_vorp_zscore)
```

where `team_vorp_zscore` is the z-score of the team's combined_score relative to the reference population (all teams in the league if league-connected; all simulated active-player rosters if manual entry).

**Letter grade thresholds (pre-specified):**

| Letter | Numeric range |
|---|---|
| A | 80–100 |
| B | 65–79 |
| C | 45–64 |
| D | 30–44 |
| F | 0–29 |

All grades are labeled **PROVISIONAL — unvalidated** until the validation experiments (§7) are complete and reviewed.

---

## 5. League Context

The following configurable parameters affect replacement level and grade calculations:

| Parameter | Default | Notes |
|---|---|---|
| League size | 12 | Affects replacement rank |
| Scoring format | PPR | PPR, half-PPR, or standard |
| Roster slots | Standard (1QB/2RB/3WR/1TE/1FLEX) | Drives bipartite matching structure |
| FLEX eligibility | RB/WR/TE | Configurable |
| SuperFLEX | Off | If enabled, PAR treatment is provisional (see §3.2) |

**Scoring format computation:**

```
pts_ppr  = fantasy_points_ppr  (direct from nflverse)
pts_std  = fantasy_points       (direct from nflverse)
pts_half = fantasy_points_ppr − 0.5 × receptions
```

The half-PPR formula has been verified exact (max diff 0.0000 on nflverse data).

---

## 6. Recommendations

Three recommendation types are defined. Each has a specific metric basis:

### 6.1 Start/Sit Swap

**Metric:** MLC-based swap value.

```
swap_value(A→B) = optimal_score(A starts, B bench) − optimal_score(B starts, A bench)
```

A positive swap_value means starting A over B increases projected output. The recommendation is issued when swap_value exceeds a minimum threshold (to suppress noise from near-zero differences).

Do not compare standalone MLC values to make start/sit decisions. The correct computation runs the optimizer twice per candidate pair.

### 6.2 Upgrade Available (Add/Drop)

**Metric:** PAR comparison.

A waiver-wire upgrade is recommended when a free-agent player at position P has PAR significantly higher than a rostered player at the same position. "Significantly higher" is defined as a PAR difference exceeding one positional standard deviation in the reference population.

### 6.3 Bye-Week Clustering Flag

Flagged when two or more starters at the same position share a bye week, reducing lineup_score by more than one standard deviation from the team's non-bye projected score. No projection metric is used for this flag; it is a schedule check only.

---

## 7. Validation Experiments

### 7.1 Data Windows

| Period | Purpose |
|---|---|
| 2012–2015 | Fallback constants derivation only |
| 2016–2020 | Experiment 2b formula selection only |
| 2021–2022 | Evaluation: Experiments 1, 2a, 3, 4 |
| 2023 | Projection dev-test — spent |
| 2024 | Sealed holdout — do not open |

**The 2024 holdout must not be opened until at least one validation experiment from the 2021–2022 window produces results and those results are reviewed.**

Synthetic leagues use B1-ranked snake draft — an explicit synthetic drafting rule, not ADP. All synthetic matchup results are labeled "simulated hypothetical matchups."

---

### Experiment 1 — Player-Grade Monotonicity

**Question:** Do higher player grades correspond to higher subsequent actual fantasy points?

**Data:** 2021–2022 player-weeks. Each observation: grade computed from information available before that week's game; outcome is that week's actual PPR score.

**Unit:** Player-week.

**Grade buckets:** Pre-specified letter-grade thresholds (A/B/C/D/F per §4.1). Monotonicity predicts mean actual points increase weakly from F to A.

**Dependence structure:** Player-weeks are not independent.
1. Same player appears in multiple weeks (within-player autocorrelation).
2. All players in the same NFL week share game-environment effects.

No single permutation scheme handles both dependence sources simultaneously. A block-within-week permutation addresses source (2) but not (1); a within-player permutation addresses (1) but not (2).

**Treatment:** Descriptive only. No formal p-value or permutation test.

**Reporting:** For each grade bucket, for each of 2021 and 2022 separately: mean actual PPR points, median, interquartile range, sample size. Visual monotonicity check is the primary result.

**Failure signal:** Reversals at adjacent grade buckets or inconsistent direction across the two seasons indicate the grading thresholds require revision.

---

### Experiment 2a — Starter PAR as Matchup Predictor

**Question:** Does the synthetic team with higher pre-game `starter_PAR` win the weekly synthetic matchup more often than the lower-`starter_PAR` team?

**Data:** 2021–2022. 100 synthetic leagues per season (B1-ranked snake draft). Regular-season matchups (~13 weeks). `starter_PAR` computed before each week's games.

**Dependence:** All 100 synthetic leagues in the same season draw from the same historical NFL player outcomes. Matchup results within any week are not independent across leagues.

**Aggregation:** Within each week W, compute:
```
w_W = fraction of week-W matchups won by the higher-starter_PAR team
      across all 100 leagues
```
This collapses within-week cross-league dependence into one weekly observation.

**Treatment:** Descriptive only. No t-test or p-value.

**Reporting for each of 2021 and 2022 separately:**
- Total synthetic matchups evaluated
- Number of matchups where the higher-`starter_PAR` team won
- Number of actual-score ties (excluded)
- Number of missing-outcome cases (excluded)
- Higher-`starter_PAR` matchup win rate for that season

**Pooled rate:** Report pooled matchup win rate across both seasons, explicitly labeled:
> *Pooled descriptive rate — synthetic leagues reuse the same underlying NFL outcomes; observations are not independent*

No power claims are made.

---

### Experiment 2b — Formula Selection (2016–2020)

**Question:** Does combined_score (starter_PAR + bench/balance components) correlate with win rate better than starter_PAR alone?

**Data:** 2016–2020 (5 seasons). Synthetic leagues as in Experiment 2a.

**True independent unit:** NFL season (N=5).

**Candidate formula (pre-registered — not confirmed or validated):**

```
combined_score = 50 + 25 × norm(starter_PAR)
                    + 15 × norm(bench_depth_PAR)
                    + 10 × (1 − weakness_rate)
```

**Component definitions (pre-registered):**

- **`starter_PAR`** — sum of PAR for the players in the optimal starting lineup, as defined in §3.2 and §4.
- **`bench_depth_PAR`** — the existing `bench_depth` component (§4): the sum of positive PAR among all bench players. Players with PAR ≤ 0 on the bench contribute zero.
- **`weakness_rate`** — fraction of optimal-lineup starting slots whose starter has PAR < 0:
  ```
  weakness_rate = (number of starting slots where starter PAR < 0)
                  / (total number of starting slots)
  ```
  Ranges from 0 (no weak starters) to 1 (all starters below replacement). Used directly as the penalty term `(1 − weakness_rate)`. Not percentile-normalized because it is already bounded to [0, 1].

**Normalization rule for `norm()` (applied to `starter_PAR` and `bench_depth_PAR` only):**

```
norm(x, P5, P95) = clamp((x − P5) / (P95 − P5), 0.0, 1.0)

Fallback (divide-by-zero guard): if P95 = P5, norm(x) = 0.5 for all x.

clamp(v, 0, 1): v < 0 → 0.0; v > 1 → 1.0; else v.
```

- **Bounds population:** all team-week observations from the 2016–2020 selection window.
- **Timing:** P5 and P95 are computed once from the 2016–2020 population before any T_y is calculated. Bounds are not recalculated per season and are not adjusted after examining results.
- **Per component:** P5 and P95 are computed separately for `starter_PAR` and `bench_depth_PAR`. Each component has its own bounds.
- **Consistency:** the same bounds are reused across all five seasons (2016–2020).

This formula is the candidate to be evaluated by the selection rule below. It is not confirmed as the production formula until that rule is satisfied.

**Tie handling:** Matchups with an absolute actual-score difference of less than 0.5 points are excluded from the discordant-matchup comparison. This is consistent with the tie-exclusion rule applied in Experiments 2a and 3.

**Estimand:** For each season Y in 2016–2020, let:
- c_y = number of eligible matchups won by the higher-combined_score team
- b_y = number of eligible matchups won by the lower-combined_score team
- n_y = c_y + b_y (total eligible matchups; ties excluded)
- T_y = c_y / n_y (fraction of eligible matchups won by the higher-combined_score team)
- T_obs = mean(T_y) across all 5 seasons (equal weight per season, not per matchup)

Compute the same statistics using starter_PAR alone as the ranking criterion. Experiment 2b evaluates whether combined_score produces higher T_y values than starter_PAR alone, season by season.

**Descriptive retention rule for combined_score:**
- T_y > 0.52 in at least 4 of 5 seasons
- T_obs (pooled) > 0.52
- No season has T_y < 0.48

If this rule is not met, starter_PAR alone is used as the team grade basis and the bench/balance components are removed from the formula.

**No formal inference.** N=5 truly independent observations (NFL seasons) does not support a calibrated statistical test. This experiment is exploratory and prescribes a formula selection rule only.

**This window is formula selection only. Results from 2016–2020 do not constitute validation.**

---

### Experiment 3 — B3 vs B2 on Start/Sit Disagreement Cases

**Question:** On same-position start/sit decisions where B3 and B2 recommend different players, does the B3-preferred player subsequently score more actual fantasy points than the B2-preferred player?

**Data:** 2021–2022.

**B2 definition:** Single most recent prior active-game actual fantasy score, regardless of calendar-week distance.

**Case eligibility:**
- Same team, same position, same week
- B3 ranks one player above the other AND B2 ranks them in the opposite order
- Ties in either projection: excluded
- Actual-score difference < 0.5 points: tie, excluded
- Missing actual outcome for either player: excluded

**Scoring:**
- Hit = B3-preferred player scores strictly more
- Miss = B2-preferred player scores strictly more

**Primary estimand:**
```
H = total B3 hits / (total B3 hits + total B2 hits)
```
computed after all exclusions. Case-weighted: each eligible disagreement pair contributes one observation.

**Secondary measure (supplementary):**
```
h̄ = mean(h_W)  where  h_W = hits / (hits + misses)  within week W
```
Equal weight per eligible week. Reported separately and labeled:
> *Equal-weight-per-week summary — secondary only*

**Dependence:** Cases are not independent (shared players, weeks, teams, and NFL-season outcomes). No valid independent-trial test has been specified.

**Treatment:** Descriptive only. No p-value or power claims.

**Reporting for each of 2021 and 2022, and for each position within each season:**
- Number of eligible cases
- Hits
- Misses
- Ties excluded
- Missing outcomes excluded
- Case-level hit rate H for that season-position cell

**Pooled H:** Report pooled case-level hit rate, labeled:
> *Pooled descriptive rate — cases share players, weeks, and NFL-season outcomes; not a statistically validated estimate*

**Exclusion tracking:** Ties and missing outcomes must be counted as separate exclusion categories before computing the denominator. If the case-generation process drops ineligible pairs before logging, disclose:
> *Tie and missing-outcome exclusion counts are not available separately; the case-generation process discarded ineligible pairs before logging. Total eligible cases are reported; exclusion breakdown requires a revised case-generation pass.*

Do not invent counts.

---

### Experiment 4 — Season-Average Team Grade vs Synthetic Season Record

**Question:** Does season-average team grade correlate with synthetic regular-season win record?

**Data:** 2021–2022 synthetic leagues.

**Unit:** Team-season (one observation per team per season).

**Estimand:** Spearman rank correlation between season-average team grade and regular-season win total, computed within each synthetic league, then averaged across leagues for each season.

**Treatment:** Descriptive. Report correlation separately for 2021 and 2022. No formal inference given the synthetic nature of the leagues and the shared underlying player outcomes.

---

## 8. Remaining Limitations

1. **All validation experiments are exploratory and descriptive.** No formal inferential procedure has been specified that accounts for all operative sources of dependence (within-player autocorrelation, within-week shared outcomes, within-team correlated pairs). Results support or challenge the methodology descriptively; they do not constitute statistical validation.

2. **N=2 truly independent NFL seasons** (2021–2022) is the evaluation window. Consistent results across both seasons and across positions constitute supporting evidence; contradictory patterns across seasons are a meaningful negative signal. Neither outcome is conclusive.

3. **Replacement level for FLEX/SuperFLEX is provisional.** The "best non-starter at primary position P" heuristic does not fully capture replacement value when a player could fill a FLEX or SuperFLEX slot. PAR values in those configurations should not be treated as definitive until the league-wide matching procedure is revised.

4. **No power estimates are stated** for any experiment. The intraclass correlation of weekly matchup outcomes and of weekly hit rates is unknown before the experiments run.

5. **Synthetic leagues use B1-ranked snake draft, not actual ADP.** All results are labeled "simulated hypothetical matchups." Findings in synthetic leagues may not generalize to actual draft behaviors or league compositions.

6. **Commercial data rights are unresolved.** nflverse data is used under CC BY 4.0 in a private research environment only. The underlying NFL data has independent rights. **Commercial use requires a legal opinion before launch.** No paid provider has been contracted. The $0 budget constraint remains in force.

7. **Grade outputs are PROVISIONAL until validation is complete.** All displayed grades must carry the label "PROVISIONAL — unvalidated" until Experiments 1–4 are run, results are reviewed, and a determination is made.

---

## 9. What This Specification Does Not Authorize

- Implementation of any grading, matching, or recommendation logic
- Running any validation experiment
- Modifying application code, schemas, or sample data
- Opening the 2024 sealed holdout
- Downloading additional data
- Contacting data providers
- Spending money
- Deploying anything

Each of these steps requires separate, explicit authorization.

---

*Related documents:*
- `docs/free-data-model-memo-2026-10-16.md` — nflverse field audit, data licensing by tier, baseline model design
- `docs/data-provider-memo-2026-10-16.md` — provider-by-provider investigation, outreach drafts
- `docs/data-provider-decision-2026-10-16.md` — provider decision matrix and recommendation
- `research/README.md` — Phase 1 run order, file structure, holdout protection, licensing
