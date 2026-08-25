# Fantasy League Broadcast System — Design Spec

**Date:** 2026-08-25
**Status:** Approved, pending implementation plan
**Author:** Brainstorming session — Greg Spunt + Claude

---

## 1. Product Overview

A standalone AI broadcast system for a private Yahoo Fantasy Football league. The product is a personalized sports media destination: an AI-produced weekly podcast with named host personas, backed by a persistent historical league knowledge database. The written recap is a companion to the audio, not the primary experience.

The system is **not** a fantasy assistant or a start/sit tool. It is a narrator and historian for a specific league. It should feel like the league has had its own broadcast team following it for years.

### Primary experience

1. Commissioner connects Yahoo once (OAuth). No Yahoo connection required for other managers.
2. The system ingests the current season's data and the prior season's full history.
3. Every week, the system produces an AI-generated podcast episode featuring multiple named AI hosts discussing what happened in the league — scores, storylines, rivalries, history.
4. Managers access the site via a shared link (no login). They hit play. They hear their league covered like a real sports broadcast.
5. Below the player, a written recap and league information serves as the companion.

### V1 scope

Single-league. One commissioner, one Yahoo league, one broadcast destination. Architecture is designed to support multi-league expansion without requiring refactoring.

### What is explicitly out of V1 scope

- Per-manager Yahoo OAuth
- Per-manager accounts or personalized dashboards
- Live scoring during games
- Push delivery (email, Slack, Discord)
- Native mobile app
- Multi-league support (architecture supports it; not built)
- Trade analyzer or start/sit advice

---

## 2. Yahoo API — Verified Data Availability

### Confirmed available

| Endpoint | Data provided |
|---|---|
| `/league/{league_key}` | League name, season, current week, scoring type, `renew` field, `renewed` field |
| `/league/{league_key}/teams` | All teams: team name, manager nickname, Yahoo GUID (when exposed), logo URL |
| `/league/{league_key}/standings` | W/L/T, points for/against, streak, waiver priority, playoff seed |
| `/league/{league_key}/scoreboard;week={n}` | All matchups for a given week with team scores |
| `/team/{team_key}/matchups` | All matchups a team played across a full season |
| `/league/{league_key}/transactions` | Trades, adds, drops (limited count per pull; must paginate) |
| `/league/{league_key}/draftresults` | Full draft order with player names and team assignments |
| `/team/{team_key}/stats;type=season` | Season-level stat totals per team |

### Prior season access mechanism

Yahoo returns `renew` (format: `{game_id}_{league_id}`) in the current season's league metadata. This is reconstructed as `{game_id}.l.{league_id}` to form the prior season's `league_key`. The commissioner's OAuth token covers the prior season because they were a member. This is confirmed working in the wild.

Known NFL game IDs: 2022 = 414, 2023 = 423, 2024 = 449, 2025 = 453.

### Known limitations

**Per-player weekly stat lines for completed seasons** return zeros (confirmed Reddit report, 2025). This affects individual player stat breakdowns only — not team matchup scores. Team-level scores per week via the scoreboard endpoint survive for completed seasons. The broadcast uses team matchup scores and season-level stats; per-player weekly breakdowns are not required.

**Transaction history** is paginated and capped (~25 per pull). Full historical transaction ingestion requires paginated pulls during the one-time historical ingestion.

### Two ingestion paths (both supported)

| Path | Trigger | Mechanism |
|---|---|---|
| Auto-discovery | `renew` field is populated | Reconstruct prior league_key from `{game_id}_{league_id}` |
| Manual entry | `renew` field is empty (fresh league) | Commissioner enters prior league key in setup UI |

If neither path produces a prior league key, the system operates with current season data only. The broadcast degrades gracefully — no historical claims are made that cannot be sourced.

---

## 3. Architecture Overview

```
Yahoo Fantasy API (commissioner OAuth)
         │
         ▼
 Data Ingestion Service
 (historical ingestion on setup; weekly sync every Tuesday)
         │
         ▼
  Supabase PostgreSQL
  (source-of-truth facts: leagues, teams, managers,
   matchups, standings, transactions, draft picks)
         │
         ▼
  Context Builder
  (structured fact object — the hard factual boundary.
   Claude never sees raw Yahoo data.
   Every node includes a source reference.)
         │
         ▼
  Claude Opus 4.6
  (broadcast script as structured JSON.
   Factual segments carry source references.
   Commentary labeled is_fact: false.)
         │
    ┌────┴─────┐
    ▼          ▼
ElevenLabs   Claude Sonnet 4.6
(multi-voice  (written recap —
 audio →       markdown,
 MP3 →         independent
 Storage)      from script)
         │
         ▼
  Next.js Web App (Vercel)
  Audio player (hero) + transcript + history + admin
```

### Weekly rhythm

- **Tuesday 9 AM** — Vercel Cron triggers weekly sync (scores, transactions, standings) → recap episode generation
- **Friday 9 AM** — weekly preview episode generation (V1: manual trigger only via admin panel; cron scheduling is a post-V1 addition)
- **Manual** — commissioner can trigger any episode or re-sync from admin panel

### One-time setup

Commissioner connects Yahoo → system reads `renew` field → confirms or manually sets prior league key → historical ingestion pulls full prior season + current season to date → preseason episode generated.

---

## 4. Database Schema

### Design principles

- **One row per sourced fact.** Scores, records, and transactions come from Yahoo and are stored verbatim. AI commentary lives only in `episodes`.
- **Manager identity is stable and separate from team identity.** The `managers` table represents real people across seasons. `teams` represents their entry in a given season. These are not conflated.
- **Season chain is explicit.** `league_seasons.prior_league_key` links backward; `next_league_key` links forward. The chain is traversable in both directions.
- **Incomplete data is tolerated.** Score fields are nullable until finalized. Ingestion never fails on missing per-player stats.
- **Manager identity is never silently merged.** Mismatches are flagged for commissioner review. Historical claims about a manager with unresolved identity are withheld from the broadcast until confirmed.

---

### `league_seasons`

One row per season of the league.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_key          text UNIQUE NOT NULL        -- "449.l.123456"
league_id           text NOT NULL               -- "123456"
game_id             text NOT NULL               -- "449"
season              integer NOT NULL            -- 2025
name                text NOT NULL               -- set during setup, commissioner-editable
num_teams           integer
scoring_type        text                        -- "head", "headpoint"
start_week          integer
end_week            integer                     -- last regular season week
playoff_start_week  integer
is_finished         boolean DEFAULT false
prior_league_key    text                        -- from "renew" field OR manual entry; nullable
next_league_key     text                        -- from "renewed" field; nullable
champion_team_key   text                        -- FK to teams.team_key; set post-season; nullable
ingested_at         timestamptz
created_at          timestamptz DEFAULT now()
```

---

### `managers`

Real people, stable across seasons.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
primary_nickname    text NOT NULL               -- most recent confirmed nickname
yahoo_guid          text UNIQUE                 -- stable Yahoo GUID when API provides it; nullable
identity_status     text NOT NULL DEFAULT 'confirmed'
                    -- 'confirmed' | 'needs_review' | 'unmatched'
is_active           boolean DEFAULT true        -- in the current season
first_season        integer                     -- earliest season they appeared
created_at          timestamptz DEFAULT now()
```

**Identity matching logic (on ingestion):**

1. If `yahoo_guid` matches an existing manager → auto-confirm. No further check needed.
2. If no GUID, and a single existing manager shares the exact nickname with no prior-season conflict → flag as `needs_review` (not auto-confirmed). Commissioner resolves in admin.
3. No match on GUID or nickname → create new manager row with `identity_status: 'unmatched'`. Commissioner confirms or merges.

The broadcast never makes cross-season historical claims about a manager whose `identity_status` is not `confirmed`.

---

### `manager_nicknames`

All known nicknames per manager per season. Supports "Greg's team was called X in 2024 and Y in 2025" type history.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
manager_id          uuid NOT NULL REFERENCES managers(id)
nickname            text NOT NULL
season              integer NOT NULL
league_season_id    uuid NOT NULL REFERENCES league_seasons(id)
is_primary          boolean DEFAULT false       -- true = the nickname Yahoo returned this season

UNIQUE (manager_id, season)
```

---

### `teams`

One row per team per season.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
team_key            text UNIQUE NOT NULL        -- "449.l.123456.t.3"
league_season_id    uuid NOT NULL REFERENCES league_seasons(id)
manager_id          uuid REFERENCES managers(id)     -- null if identity unresolved
team_name           text NOT NULL               -- team name for this season
manager_nickname    text NOT NULL               -- raw from Yahoo this season
manager_yahoo_guid  text                        -- raw from Yahoo; nullable
logo_url            text
draft_position      integer
final_standing      integer                     -- 1 = champion
wins                integer NOT NULL DEFAULT 0
losses              integer NOT NULL DEFAULT 0
ties                integer NOT NULL DEFAULT 0
points_for          numeric NOT NULL DEFAULT 0
points_against      numeric NOT NULL DEFAULT 0
playoff_seed        integer
made_playoffs       boolean DEFAULT false
is_champion         boolean DEFAULT false
created_at          timestamptz DEFAULT now()
```

---

### `matchups`

Every game, every week, every season. The core fact table for all broadcast history.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id    uuid NOT NULL REFERENCES league_seasons(id)
season              integer NOT NULL            -- denormalized for fast cross-season queries
week                integer NOT NULL
team1_key           text NOT NULL               -- no FK constraint; team may be from prior season
team2_key           text NOT NULL
team1_score         numeric                     -- null until game finalizes
team2_score         numeric
winner_team_key     text                        -- null if incomplete or tie
is_tie              boolean DEFAULT false
margin              numeric                     -- ABS(team1_score - team2_score); null until complete
is_playoff          boolean DEFAULT false
is_consolation      boolean DEFAULT false
is_championship     boolean DEFAULT false
is_complete         boolean DEFAULT false
raw_yahoo_json      jsonb                       -- raw Yahoo response; for recovery/debugging

created_at          timestamptz DEFAULT now()
updated_at          timestamptz DEFAULT now()

UNIQUE (league_season_id, week, team1_key, team2_key)
INDEX  (season, team1_key, team2_key)           -- H2H lookups
INDEX  (league_season_id, week)                 -- weekly pulls
INDEX  (is_championship)                        -- championship history
INDEX  (margin DESC)                            -- blowout/closest queries
```

**Queries this table supports (examples):**

- H2H all-time record: join `teams` twice on `manager_id`, filter both sides
- Current win streak: matchups for team ordered by week DESC, count unbroken W/L
- Biggest blowout: `ORDER BY margin DESC LIMIT 1`
- Closest game: `ORDER BY margin ASC WHERE is_complete = true LIMIT 1`
- Championship history: `WHERE is_championship = true` join `teams` for winner name
- Returning vs new managers: `teams` for current season LEFT JOIN prior season on `manager_id`

---

### `standings_snapshots`

League standings captured after each week finalizes. Enables "Greg was in first place heading into Week 8" storylines.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id    uuid NOT NULL REFERENCES league_seasons(id)
team_key            text NOT NULL
season              integer NOT NULL
week                integer NOT NULL
wins                integer NOT NULL
losses              integer NOT NULL
ties                integer NOT NULL DEFAULT 0
points_for          numeric NOT NULL
points_against      numeric NOT NULL
streak_type         text NOT NULL               -- 'W' or 'L'
streak_count        integer NOT NULL DEFAULT 1
playoff_seed        integer

UNIQUE (league_season_id, team_key, week)
```

---

### `transactions`

Trades, adds, drops. Historical record for broadcast storylines ("Greg traded Mahomes in Week 5").

```sql
id                      uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id        uuid NOT NULL REFERENCES league_seasons(id)
yahoo_transaction_key   text UNIQUE             -- nullable; for dedup
type                    text NOT NULL           -- 'trade' | 'add' | 'drop' | 'add/drop'
status                  text NOT NULL           -- 'successful' | 'failed' | 'pending'
week                    integer
occurred_at             timestamptz
involved_team_keys      text[]                  -- all team_keys in the transaction
players_sent            jsonb
-- [{player_name, position, from_team_key, to_team_key}]
players_received        jsonb
-- same shape; from the perspective of the receiving team
raw_yahoo_json          jsonb

created_at              timestamptz DEFAULT now()
```

---

### `draft_picks`

Full draft order per season.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id    uuid NOT NULL REFERENCES league_seasons(id)
season              integer NOT NULL
pick_number         integer NOT NULL
round               integer NOT NULL
team_key            text NOT NULL
player_name         text NOT NULL
player_id           text                        -- Yahoo player ID; nullable
position            text
nfl_team            text

UNIQUE (league_season_id, pick_number)
```

---

### `episodes`

AI-generated broadcast content. Strictly separated from source facts.

```sql
id                      uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id        uuid NOT NULL REFERENCES league_seasons(id)
season                  integer NOT NULL
week                    integer                 -- null for preseason / postseason specials
episode_type            text NOT NULL
-- 'preseason' | 'weekly_preview' | 'weekly_recap' | 'playoff_special' | 'championship'
title                   text NOT NULL           -- commissioner-editable after generation
status                  text NOT NULL DEFAULT 'pending'
-- 'pending' | 'generating_script' | 'generating_audio' | 'ready' | 'partial' | 'failed'
script_json             jsonb                   -- structured segments (see schema below)
transcript_md           text                    -- written recap companion
audio_url               text                    -- Supabase Storage path; null until ready
audio_duration_secs     integer
published_at            timestamptz
generation_metadata     jsonb
-- { model, tokens_in, tokens_out, elevenlabs_chars, elapsed_ms, cost_usd_estimate }
error_message           text

created_at              timestamptz DEFAULT now()
updated_at              timestamptz DEFAULT now()
```

**`script_json` schema — fact/commentary separation enforced at this level:**

```json
{
  "episode": {
    "title": "Week 5 Recap — The Blowout Edition",
    "week": 5,
    "season": 2025,
    "type": "weekly_recap",
    "estimated_duration_secs": 480
  },
  "segments": [
    {
      "index": 0,
      "host": "troy",
      "text": "Welcome back to [LEAGUE SHOW NAME]. Week 5 is in the books.",
      "is_fact": false,
      "type": "intro"
    },
    {
      "index": 1,
      "host": "troy",
      "text": "Greg's team put up 147 points to Mike's 89 — a 58-point margin, the biggest blowout of the season.",
      "is_fact": true,
      "source": "matchups:week5:max_margin",
      "type": "score_report"
    },
    {
      "index": 2,
      "host": "lisa",
      "text": "And Troy, that's now three straight wins for Greg over Mike after losing the first four meetings between them all-time.",
      "is_fact": true,
      "source": "matchups:h2h_query:team3_vs_team7",
      "type": "historical_context"
    },
    {
      "index": 3,
      "host": "danny",
      "text": "The student has become the teacher. I love to see it.",
      "is_fact": false,
      "type": "commentary"
    }
  ],
  "sections": ["intro", "matchup_recap", "standings", "storylines", "closing"]
}
```

Every `is_fact: true` segment carries a `source` string tracing to a named context builder query. This is enforced by the Claude system prompt and validated before the episode is marked `ready`.

---

### `league_config`

Per-league configuration: show branding, access control, host voices. One row per league. When a season renews, config is copied forward and updated.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id    uuid NOT NULL UNIQUE REFERENCES league_seasons(id)
show_name           text NOT NULL               -- commissioner-set broadcast name, e.g. "The Spunt Report"
share_token         text NOT NULL UNIQUE        -- random token in the share link; stored in localStorage after first visit
accent_color        text NOT NULL DEFAULT '#00ff87'  -- hex; used for player button, winner highlight
host_troy_voice_id  text                        -- ElevenLabs voice ID
host_lisa_voice_id  text                        -- ElevenLabs voice ID
host_danny_voice_id text                        -- ElevenLabs voice ID
episode_artwork_url text                        -- optional custom artwork; falls back to Yahoo league logo
created_at          timestamptz DEFAULT now()
updated_at          timestamptz DEFAULT now()
```

`show_name` is the broadcast show name set during commissioner setup — distinct from `league_seasons.name` (the Yahoo league name). A Yahoo league named "Fantasy Bros 2025" might have a show called "The Spunt Report."

---

### `ingestion_log`

Audit trail for every data pull.

```sql
id                  uuid PRIMARY KEY DEFAULT gen_random_uuid()
league_season_id    uuid NOT NULL REFERENCES league_seasons(id)
ingestion_type      text NOT NULL
-- 'historical_full' | 'weekly_sync' | 'transaction_sync' | 'draft_sync'
week                integer                     -- null for full-season pulls
status              text NOT NULL               -- 'running' | 'complete' | 'partial' | 'failed'
records_upserted    integer DEFAULT 0
warnings            text[]                      -- e.g. "week 3 scores null, stored incomplete"
error_message       text
started_at          timestamptz DEFAULT now()
completed_at        timestamptz
```

---

## 5. Content Generation Pipeline

### The hard factual boundary

The Context Builder is the only system that reads from the database. Claude receives a structured fact object — never raw Yahoo data, never raw SQL results. Every leaf node in the fact object that represents a verifiable claim carries a `source` field (a named query identifier). Claude's script output must include this source on every `is_fact: true` segment. The system validates this before marking an episode ready.

### Context object structure (weekly recap example)

```json
{
  "meta": {
    "episode_type": "weekly_recap",
    "week": 5,
    "season": 2025,
    "league_name": "[commissioner-set name]",
    "generated_at": "2025-10-14T10:00:00Z"
  },
  "this_week": {
    "matchups": [
      {
        "id": "uuid",
        "team_a": {
          "team_key": "...",
          "team_name": "...",
          "manager": "Greg",
          "record_before": "3-1",
          "points_for_season": 512.4,
          "last_3_scores": [134.2, 98.1, 112.6]
        },
        "team_b": {
          "team_key": "...",
          "team_name": "...",
          "manager": "Mike",
          "record_before": "2-2",
          "points_for_season": 432.1,
          "last_3_scores": [98.4, 110.2, 89.1]
        },
        "result": {
          "team_a_score": 147.3,
          "team_b_score": 89.1,
          "winner_manager": "Greg",
          "margin": 58.2
        },
        "h2h": {
          "all_time_team_a_wins": 4,
          "all_time_team_b_wins": 2,
          "this_season_team_a_wins": 1,
          "this_season_team_b_wins": 0,
          "last_meeting": {
            "week": 2, "season": 2025,
            "winner": "Greg", "margin": 22.9
          },
          "source": "matchups:h2h_query:team3_vs_team7"
        }
      }
    ],
    "week_records": {
      "highest_score":   { "score": 147.3, "manager": "Greg", "source": "matchups:week5:max_score" },
      "lowest_score":    { "score": 71.2,  "manager": "Mike", "source": "matchups:week5:min_score" },
      "closest_game":    { "margin": 3.1,  "source": "matchups:week5:min_margin" },
      "biggest_blowout": { "margin": 58.2, "source": "matchups:week5:max_margin" }
    }
  },
  "standings": [
    {
      "rank": 1,
      "manager": "Greg",
      "team_name": "...",
      "record": "4-1",
      "points_for": 646.7,
      "streak": "W3",
      "source": "standings_snapshots:week5:team3"
    }
  ],
  "all_time_records": {
    "highest_score_ever":   { "score": 234.1, "manager": "Mike", "week": 7, "season": 2024, "source": "matchups:all_time:max_score" },
    "biggest_blowout_ever": { "margin": 89.3, "winner": "Greg", "loser": "Tim", "source": "matchups:all_time:max_margin" },
    "championship_history": [
      { "season": 2024, "champion": "Mike", "runner_up": "Sarah", "championship_score": "162.4-134.1", "source": "matchups:championship:2024" }
    ]
  },
  "storylines": {
    "active_streaks":        [{ "manager": "Greg", "streak": "W3", "source": "matchups:streak:team3" }],
    "milestone_approaching": [{ "description": "Greg is 1 win from matching his 2024 season-best of W4", "source": "matchups:streak_history:team3" }],
    "h2h_narratives":        [{ "description": "Greg has beaten Mike 3 straight after losing the first 4 meetings", "source": "matchups:h2h_query:team3_vs_team7" }]
  },
  "manager_continuity": {
    "returning":  ["Greg", "Mike", "Sarah", "Tim"],
    "new_this_season": ["Jordan"],
    "departed":   ["Chris"],
    "source": "teams:season_comparison:2024_vs_2025"
  },
  "recent_transactions": [
    {
      "type": "trade",
      "week": 4,
      "description": "Greg traded Davante Adams to Mike for Stefon Diggs",
      "teams": ["Greg", "Mike"],
      "source": "transactions:trade:uuid"
    }
  ]
}
```

The context object is scoped to what's relevant — the broadcaster identifies the 3–5 most interesting stories automatically during context building (not all matchups are equally interesting; blowouts, upsets, streak games, rivalries, milestone weeks get surfaced). Historical context appears when it's directly relevant to this week's events — not as a data dump on every episode.

### Script generator

**Model:** Claude Opus 4.6

**Host personas:**

| Host | Role | Voice |
|---|---|---|
| Troy | Lead anchor | Authoritative, warm, factual. Delivers scores, records, results. |
| Lisa | Analyst | Pattern-finder, historically-minded. Connects this week to league history. |
| Danny | Color | Banter, personality, hot takes, reactions. The group chat voice. |

Hosts interact — this is a conversation, not turns at a podium. Troy sets up a result, Lisa adds historical weight, Danny reacts. They can disagree, joke, and call back to earlier segments. Only factual claims are source-bound; the conversational texture is free.

**System prompt core constraints:**

```
FACT INTEGRITY — these override all other instructions:
1. You may ONLY state facts that appear in the provided context object.
2. Every segment with is_fact: true MUST copy the "source" value from
   the corresponding context node.
3. Never invent, extrapolate, or assume scores, records, names, or
   history not present in the context.
4. Absent or null context fields must not be referenced.
5. Commentary is allowed but MUST be labeled is_fact: false.
6. When uncertain, default to is_fact: false.

EDITORIAL — shape the broadcast around 3–5 stories, not every matchup.
Pick the most interesting narrative threads and build the episode around those.
Historical context should appear when relevant to this week's events,
not as a standing segment in every episode.
```

**Episode types and structures:**

| Type | Sections |
|---|---|
| `preseason` | Welcome + league intro, returning vs new managers, championship history, major rivalries (H2H records), notable records, predictions (labeled commentary) |
| `weekly_preview` | This week's matchups, H2H history per matchup, stakes and playoff implications |
| `weekly_recap` | Scores (hero stories first, not scoreboard order), standings shift, ongoing storylines, closing |
| `playoff_special` | Bracket, manager history, what's at stake, path to championship |
| `championship` | Build-up, result, season summary, new champion narrative |

**Preseason episode standard:** The preseason episode must feel substantial. It uses the historical database to introduce the entire league: which managers are back, who won last year, the full championship timeline, the top rivalries with all-time H2H records, notable historical records, and clearly labeled predictions/commentary for the upcoming season. It is the orientation episode — a new manager joining the league should be able to listen to it and understand everything.

### Audio producer

**Service:** ElevenLabs TTS

Three voice IDs are configured per league during setup. Commissioner selects from ElevenLabs voice library. Voice IDs stored in `league_config` (`host_troy_voice_id`, `host_lisa_voice_id`, `host_danny_voice_id`). The audio producer reads these at generation time — not hardcoded.

**Generation flow:**

```
For each segment (sequential, not parallel — preserves order):
  POST /v1/text-to-speech/{voice_id}
    model: eleven_turbo_v2_5
    text: segment.text
    voice_settings: { stability: 0.6, similarity_boost: 0.8 }

  On failure: retry up to 2×
  On 2× failure: generate silence buffer of equivalent duration,
                 log warning, continue — do not abort

  Store MP3 buffer

After all segments:
  - 300ms silence between same-host consecutive segments
  - 600ms silence between different-host switches
  - 1200ms silence between major sections
  Concatenate → final MP3
  Upload to Supabase Storage: /{season}/week-{week}/episode.mp3
```

If a segment generates silence instead of audio, the transition handling is unchanged — the episode does not sound broken from a missing segment, it just sounds like a pause. This is acceptable; the transcript is always complete.

If >20% of segments fail, abort audio generation entirely. Episode publishes as transcript-only (`partial` status). The admin dashboard surfaces this as an alert badge on the episodes page — no email or push notification in V1; commissioner checks the dashboard.

**Note on pricing:** ElevenLabs pricing and model availability should be verified at implementation time. The architecture assumes per-character billing with a turbo-class model; if pricing or model names have changed, update configuration accordingly. Do not hard-code cost estimates.

### Transcript generator

**Model:** Claude Sonnet 4.6

Receives the same context object as the script generator — not the script itself. Produces clean markdown independently. This prevents the transcript from being a degraded version of the audio.

Every generated transcript ends with:

```
---
*Match scores and league records sourced from Yahoo Fantasy.
Commentary and analysis are AI-generated.*
```

No exception. This line appears on every episode document.

### Publishing and error handling

```
Episode status flow:
pending → generating_script → generating_audio → ready
                          ↘ generating_transcript ↗
                                               → partial (audio failed, transcript available)
                           → failed (script failed; transcript and audio not attempted)
```

Script generation completes first (Claude Opus). Transcript (Claude Sonnet) and audio (ElevenLabs) then run in parallel. Episode is `ready` only when both are complete. If audio fails but transcript succeeds, status is `partial` — the episode is still published.

| Failure | Behavior |
|---|---|
| Yahoo pull fails | Abort pipeline, log to ingestion_log, retry next cron |
| Context builder returns no data | Abort, log error, do not generate |
| Claude script fails | Retry once with same context. On second failure → `failed`. |
| Script fails source validation | Regenerate with stronger constraint prompt once. On failure → `failed`. |
| Individual ElevenLabs segment fails | Retry 2×, insert silence, log warning, continue |
| >20% segments fail | Abort audio, publish transcript-only as `partial` |
| Audio upload fails | Retry once. On failure → publish transcript-only as `partial` |

---

## 6. Web App

### Access model

- The site is deployed at a single URL (no league_slug routing in V1; architecture supports adding it)
- All public pages (player, archive, history, standings) are accessible to anyone with the URL
- A `share_token` (stored in Supabase `league_config`) is included when the commissioner shares the link: `https://app.url?join=abc123`
- On first visit with a valid token, the token is stored in `localStorage`. Subsequent visits don't require the token in the URL.
- Admin routes (`/admin/*`) require Supabase Auth (commissioner email / magic link)
- No per-manager accounts. No per-manager Yahoo OAuth.

### Page structure

```
app/
├── (public)/
│   ├── layout.tsx                      # Dark theme, persistent audio player context
│   ├── page.tsx                        # Home — latest episode + this week's results
│   ├── episode/[season]/[week]/
│   │   └── page.tsx                    # Individual episode with player + full transcript
│   ├── archive/
│   │   └── page.tsx                    # All episodes, filterable by season
│   ├── history/
│   │   └── page.tsx                    # League history — the permanent record book
│   └── standings/
│       └── page.tsx                    # Current standings + recent form
├── admin/
│   ├── layout.tsx                      # Supabase Auth gate
│   ├── page.tsx                        # Commissioner dashboard
│   ├── setup/page.tsx                  # Initial setup flow
│   ├── episodes/page.tsx               # Episode management + generation triggers
│   ├── managers/page.tsx               # Manager identity review
│   └── ingestion/page.tsx             # Ingestion log + manual sync
└── api/
    ├── yahoo/callback/route.ts
    ├── ingest/historical/route.ts
    ├── ingest/weekly/route.ts
    ├── episode/generate/route.ts
    └── cron/weekly/route.ts
```

### Home page (`/`)

The audio player is the first element. A manager who opens this on their phone from a group chat link immediately sees and can play the latest episode.

Layout (top to bottom, mobile-first):

```
[League name + show branding]

[Episode artwork or league logo — square]
[Episode title and date]
[──────────────────────────────────] ← progress bar
[◀◀] [▶] [▶▶]  1×  ↓ share          ← controls
[00:00 / 48:12]

─────────────────────────────
THIS WEEK'S RESULTS
[Matchup score cards — scrollable horizontal row on mobile]
[Greg 147.3] [Mike 89.1]  WINNER +58.2
─────────────────────────────
STANDINGS
1. Greg    4-1  W3
2. Sarah   3-2  W1
...
─────────────────────────────
PAST EPISODES  [Archive →]
[Week 4] [Week 3] [Week 2] ...
```

A persistent mini-player docks at the bottom of the screen when the user scrolls past the hero player — identical to how podcast apps work. This is implemented via React Context with a global audio state.

### Individual episode page (`/episode/[season]/[week]`)

```
[← Back]
[Episode title]
[Player — same controls as home]
─────────────────────────────
RECAP
[Full transcript_md rendered as article prose]
[Scores table]
[Standings after this week]
[Transactions this week (if any)]
─────────────────────────────
[← Previous episode]  [Next episode →]
```

The transcript renders below the player. Audio is the hero; transcript is the companion. Headers, tables, and prose are all rendered from markdown.

### Archive page (`/archive`)

Season tabs across the top. Episodes listed as cards, newest first within each season.

Special episodes (preseason, championship, playoff) are visually tagged. Cards show: episode title, date, duration, [Play] button that navigates to episode page.

### League History page (`/history`)

This is the permanent record book. It is a real feature — not a static page. All data is live from the database. Only managers with `identity_status: 'confirmed'` appear in cross-season history.

Sections:

**Championship Roll** — one card per season, champion + runner-up + championship score. Scrolls back through every completed season in the DB.

**All-Time Records** — a table of league records sourced directly from the `matchups` table:
- Highest single-week score ever
- Lowest single-week score ever
- Biggest blowout (margin)
- Closest game (margin)
- Most points in a season
- Longest win streak
- Most championships

**Manager Career Stats** — one row per confirmed manager. Columns: seasons played, total record (W-L-T), career points for, championships, playoff appearances, best season record. Sortable.

**Head-to-Head Matrix** — a grid showing all-time W-L between every pair of confirmed managers. Mobile version collapses to a per-manager detail view (tap a manager to see their H2H record vs everyone).

**Rivalry Spotlight** — the 3–5 most-played matchups by total games, with all-time record and most recent result.

### Standings page (`/standings`)

Clean current standings table. Columns: Rank, Team, Manager, W-L, Pts For, Pts Against, Streak, Last 3 (W/L icons). Playoff cutline highlighted visually (dotted line between positions in and out of playoffs). Links to latest episode above the table.

### Admin panel

**Setup flow (`/admin/setup`)** — one-time, 6 steps:

1. Connect Yahoo (OAuth)
2. Select league from discovered leagues on commissioner's account
3. Set league/show name and branding (commissioner configures the show name — not "The Gridiron Gazette"; whatever they want)
4. Prior season confirmation: auto-discovered via `renew` field with commissioner confirmation, or manual entry if not found
5. Configure host voices (ElevenLabs voice picker for Troy, Lisa, Danny)
6. Run historical ingestion → generate preseason episode

**Dashboard (`/admin`)** — ongoing operations:

- Ingestion status: last sync time, next scheduled sync
- Episode pipeline status: latest episode, current status
- Manager identity alerts: count of managers flagged `needs_review` or `unmatched` with link to manager page
- Quick actions: [Sync now], [Generate this week's episode], [View ingestion log]

**Manager identity review (`/admin/managers`):**

Table of all managers across all seasons. Columns: Primary nickname, all known nicknames (with seasons), Yahoo GUID (if known), identity status badge, actions.

Actions: [Confirm same person as X], [This is a different person], [Merge with →]. Commissioner can resolve all flags here. Until a manager is confirmed, cross-season historical claims about them are withheld from broadcast context.

**Episode management (`/admin/episodes`):**

Table of all episodes with status badges. Per-episode actions:
- [Regenerate] — full pipeline re-run
- [Regenerate audio only] — re-run ElevenLabs if script is intact
- [View script JSON] — read-only, for debugging source attribution
- [Edit transcript] — manual override of written recap
- [Edit title] — episode title only

**Ingestion log (`/admin/ingestion`):**

Table from `ingestion_log`. Columns: type, week, status, records upserted, warnings, timestamp. [Force re-sync] button for any week or full historical re-pull.

### Visual design direction

- **Dark theme** — deep navy/charcoal (#0f1117 range), not pure black. Sports broadcast aesthetic.
- **Typography** — bold display font for episode titles, scores, and record numbers. Clean legible sans-serif for body text and recaps.
- **Accent** — one strong primary color (commissioner can configure; default: electric green). Used for: play button, winning team highlight, active streak indicators, CTAs.
- **Score cards** — winner gets a subtle left-border highlight in accent color; loser is muted.
- **No stock photography** — league team logos from Yahoo + episode artwork as the only images.
- **Mobile-first layout** — the player, score cards, and standings all work at 390px. No horizontal scrolling. Scores collapse to vertical stack on narrow screens.

### Tech stack

| Concern | Choice |
|---|---|
| Framework | Next.js 14+ App Router |
| Styling | Tailwind CSS |
| Audio player | Custom (`<audio>` + React Context) — no library |
| Persistent mini-player | React Context with global audio state + fixed footer |
| DB access | Supabase JS (server components via service role) |
| Auth (admin only) | Supabase Auth — magic link |
| Audio storage | Supabase Storage |
| Deployment | Vercel |
| Cron | Vercel Cron Jobs |
| Script generation | Claude Opus 4.6 via Anthropic SDK |
| Transcript generation | Claude Sonnet 4.6 via Anthropic SDK |
| Audio generation | ElevenLabs REST API |

---

## 7. Multi-League Considerations (deferred)

Architecture supports multi-league expansion without refactoring. When building multi-league:

- Add `slug` to `league_seasons`. Routes become `/{slug}/...`.
- Add league-level `config` table (show name, share_token, host voice IDs, accent color).
- Commissioner auth becomes multi-league aware (one commissioner can own multiple leagues).
- Context Builder and episode pipeline are already stateless relative to league — they operate on `league_season_id`.
- Web app public routes add the slug prefix. Everything else is the same.

---

## 8. Open Questions (resolved before implementation)

| Question | Resolution |
|---|---|
| Does `renew` field populate on football leagues in 2025? | Confirmed working; verified via Yahoo API documentation and community reports. |
| Per-player weekly stats for completed seasons? | Not relied upon. Team matchup scores survive; player-level breakdowns are not used. |
| ElevenLabs model/pricing at implementation time | Verify at implementation. `eleven_turbo_v2_5` assumed; update if deprecated. |
| League name/branding | Set by commissioner during setup. No default placeholder used in production. |
| Commissioner account type | Single commissioner per league in V1. Multi-owner deferred. |

---

## 9. V1 Delivery Summary

**Included:**
- Yahoo OAuth (commissioner only) with token refresh
- Historical ingestion (auto-discovery via `renew` + manual fallback)
- Full database schema as specified
- Manager identity matching with commissioner review flow
- Context Builder with named queries and source references
- Claude Opus script generation with fact/commentary separation
- ElevenLabs multi-voice audio production with graceful failure handling
- Claude Sonnet transcript generation
- Next.js web app: home, episode, archive, history, standings, admin
- Persistent mini audio player
- Vercel Cron for weekly automation
- Commissioner setup flow including show branding

**Excluded from V1:**
- Per-manager login or Yahoo OAuth
- Live scoring
- Push delivery (email, Slack, Discord)
- Native mobile app
- Multi-league routing
- Trade analyzer or lineup advice
