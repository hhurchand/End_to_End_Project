# FantasyGrade — Sleeper + Yahoo Fantasy API Integration

**Date:** 2026-10-20
**Status:** Approved
**Scope:** Connect real fantasy league rosters and compute real B3 projections from recent stats, fed into the existing validated grading engine.

---

## Overview

Replace the demo data layer with live data from Sleeper (no auth) and Yahoo Fantasy (OAuth 2.0). Both providers implement a shared `FantasyProvider` interface so the grading logic is completely platform-agnostic. The entire computation — fetch roster, compute B3, compute replacement level, grade — happens server-side in one pass before the page renders.

**Deployment target:** Vercel
**Auth model:** Stateless — Yahoo token stored in a signed httpOnly cookie (7-day), no user accounts, no database.

---

## Connection Flows

```
/ (landing page)
│
├── Connect Sleeper
│     /connect/sleeper
│       enter username → server fetches leagues → pick league → pick team
│       → /grade?platform=sleeper&leagueId=xxx&teamKey=xxx
│
└── Connect Yahoo
      /api/auth/yahoo  →  Yahoo OAuth  →  /api/auth/yahoo/callback
        exchange code → set httpOnly cookie → /connect/yahoo
        server fetches leagues → pick league → pick team
        → /grade?platform=yahoo&leagueKey=xxx&teamKey=xxx
```

The grade page at `/grade` detects platform params in the query string. When present, it fetches real data. When absent, it falls back to the existing demo (`DEMO_TEAM`).

---

## Provider Interface

Both Sleeper and Yahoo implement:

```ts
interface FantasyProvider {
  // identifier = Sleeper username | Yahoo: ignored (user derived from token)
  getLeagues(identifier: string | null): Promise<League[]>
  getRoster(leagueId: string, teamKey: string, week: number): Promise<RosterPlayer[]>
  getPlayerStats(playerIds: string[], weeks: number[]): Promise<Map<string, WeeklyScore[]>>
  // Returns all rostered players + top 100 available free agents by recent scoring average
  getPlayerPool(leagueId: string): Promise<PoolPlayer[]>
  getLeagueSettings(leagueId: string): Promise<LeagueSettings>
}
```

The grader (`lib/providers/grader.ts`) only speaks `FantasyProvider`. It is never aware of Sleeper or Yahoo specifics.

---

## B3 Computation

Identical algorithm to the research (`research/exp_utils.py`):

- Exponentially weighted average of the player's last ≤6 **active** weeks
- Decay factor d = 0.80 (most recent week weight = 1.0, next = 0.80, next = 0.64, …)
- **Active** = player actually played that week (not bye, not DNP/IR with 0 snaps)
  - Sleeper: detected via non-zero `player_score` or snap counts
  - Yahoo: detected via `is_starting` flag and non-zero stats
- Minimum 2 active weeks required; returns `null` if not met
- `null` B3 triggers fallback chain: season average → position mean → `grade: null` (displayed as `—`)

```ts
// Signature
function computeB3(scores: WeeklyScore[]): number | null

interface WeeklyScore {
  week: number
  pts: number
  active: boolean
}
```

---

## Replacement Level

Instead of the hardcoded `REPLACEMENT_BASELINE` constants, compute dynamically from the actual league player pool:

1. `getPlayerPool` returns all rostered players + top 100 free agents by recent scoring average
2. Compute B3 for every pool player
3. For each position (QB/RB/WR/TE): find the best-B3 player **not on any roster** — this is the replacement player
4. Their B3 is the replacement baseline for that position
5. Fallback: if no genuine-B3 player (< 2 active weeks) is available at a position, use hardcoded `REPLACEMENT_BASELINE[pos]`

---

## Grader Orchestration

`lib/providers/grader.ts` — `gradeTeam(provider, leagueId, teamKey, week)`:

1. `getRoster` → player list with slot assignments
2. `getPlayerPool` → full league pool (rostered + top FAs)
3. `getPlayerStats(allPlayerIds, last6Weeks)` → weekly scores
4. `computeB3` per player → projections map
5. `computeReplacementBaselines(pool, b3Map)` → replacement B3 per position
6. `computePar` per rostered player: `B3(player) − replacementB3(pos)`
7. `computeOptimalLineup` (existing optimizer, unchanged) → starter assignments
8. `teamGrade(sum of starter PARs)` + `playerGrade(par, pos)` per player
9. Return `ComputedTeamGrade` (same shape as `SampleTeamGrade` — no UI changes needed)

---

## Yahoo OAuth

**Env vars required:**
- `YAHOO_CLIENT_ID`
- `YAHOO_CLIENT_SECRET`
- `SESSION_SECRET` (for cookie signing)

**Flow:**
1. `GET /api/auth/yahoo` — builds Yahoo authorization URL with `scope=fspt-r`, random `state` stored in a short-lived cookie for CSRF protection, redirects user to Yahoo
2. Yahoo redirects to `GET /api/auth/yahoo/callback?code=xxx&state=xxx`
3. Server verifies `state`, POSTs to Yahoo token endpoint with credentials + code
4. Stores `{ access_token, refresh_token, expires_at }` as encrypted JSON in signed httpOnly cookie `yf_session` (7-day max-age)
5. Redirects to `/connect/yahoo`

**Token refresh:**
- `yahoo/auth.ts` checks `expires_at` before every API call
- Yahoo tokens expire in 1 hour; if within 5 minutes of expiry (or expired), silently refresh using `refresh_token`
- If refresh fails (refresh token revoked), redirect to `/api/auth/yahoo` to re-auth

---

## File Structure

```
lib/providers/
  types.ts                      — FantasyProvider interface + all shared types
  b3.ts                         — computeB3(), detectActive()
  replacement.ts                — computeReplacementBaselines()
  grader.ts                     — gradeTeam() orchestrator
  sleeper/
    client.ts                   — Sleeper REST API calls (no auth)
    adapter.ts                  — SleeperProvider implements FantasyProvider
  yahoo/
    client.ts                   — Yahoo Fantasy API calls (bearer token)
    adapter.ts                  — YahooProvider implements FantasyProvider
    auth.ts                     — getTokenFromCookie(), refreshIfNeeded()

app/api/auth/yahoo/
  route.ts                      — GET: initiate OAuth redirect
  callback/route.ts             — GET: exchange code, set cookie, redirect

app/connect/
  sleeper/page.tsx              — username input + league/team picker (server component)
  yahoo/page.tsx                — league/team picker (server component, reads cookie)

app/grade/page.tsx              — updated: detects platform params, calls gradeTeam()
```

Existing files that do not change: `lib/grading-engine/*`, `lib/lineup-optimizer/*`, all UI components.

---

## Error Handling

| Error | Handling |
|---|---|
| Yahoo token expired | Silent refresh attempt; if fails, redirect to `/api/auth/yahoo` |
| Sleeper username not found | Inline error on `/connect/sleeper`, user retries |
| Player has < 2 active weeks | B3 = null → fallback chain → `grade: null` shown as `—` |
| No genuine-B3 pool players at a position | Fall back to hardcoded `REPLACEMENT_BASELINE` |
| API 429 rate limit | Error page: "Too many requests — try again shortly" |
| League/team not found | Redirect back to picker with error message |
| Missing env vars at startup | Throw with clear message identifying the missing var |

---

## Testing

| File | What it tests |
|---|---|
| `lib/providers/b3.test.ts` | `computeB3`: correct weighting, min-2 guard, bye exclusion, d=0.80 decay |
| `lib/providers/replacement.test.ts` | `computeReplacementBaselines`: best-unrostered selection, fallback |
| `lib/providers/sleeper/adapter.test.ts` | Sleeper adapter with fixture JSON (no live calls) |
| `lib/providers/yahoo/adapter.test.ts` | Yahoo adapter with fixture JSON (no live calls) |
| `lib/providers/grader.test.ts` | Full `gradeTeam()` pipeline with a mock `FantasyProvider` |

No live API calls in tests. All external I/O is behind the `FantasyProvider` interface.

---

## Constraints

- No external data providers beyond Sleeper and Yahoo APIs
- No database — stateless cookie session only
- No scraping
- Grading formula is frozen — `teamGrade()` and `playerGrade()` in `lib/grading-engine/formulas.ts` are not modified
- All projections labeled as estimates; injury/waiver status sourced directly from the platform API, never invented
