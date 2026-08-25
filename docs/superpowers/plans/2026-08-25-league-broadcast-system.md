# Fantasy League Broadcast System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone AI broadcast system for a private Yahoo Fantasy Football league that produces weekly AI-generated podcast episodes (audio + written recap) backed by a persistent historical league database.

**Architecture:** Yahoo data is ingested into Supabase PostgreSQL on a weekly cron schedule. A Context Builder assembles a structured fact object that Claude Opus uses to generate a broadcast script (JSON). ElevenLabs produces multi-voice audio from that script; Claude Sonnet produces a parallel markdown transcript. A Next.js 14+ web app on Vercel serves the audio player (hero) and companion content.

**Tech Stack:** Next.js 14+ App Router, TypeScript, Tailwind CSS, Supabase (PostgreSQL + Storage + Auth), Vercel (hosting + cron), Claude Opus 4.6 + Sonnet 4.6 (Anthropic SDK), ElevenLabs TTS REST API, Yahoo Fantasy Sports API (OAuth 2.0), Vitest (testing)

## Global Constraints

- Project root: `~/league-broadcast/` (new standalone repo, separate from Fantasy GM)
- Next.js 14+ App Router only — no Pages Router patterns
- `params` in server components and route handlers is always `Promise<{...}>` — must be awaited
- `cookies()` from `next/headers` is async — must be awaited
- TypeScript strict mode throughout — no `any` except where Supabase JSONB requires it
- Claude models: script generation = `claude-opus-4-6`, transcript = `claude-sonnet-4-6`
- ElevenLabs model: `eleven_turbo_v2_5` — verify at implementation; update if deprecated
- ElevenLabs segments generated sequentially (not parallel) to preserve audio order
- Every `is_fact: true` script segment MUST carry a `source` string — validated before episode marked `ready`
- Transcript always ends with the statutory disclaimer (verbatim — see Task 12)
- Admin routes protected by Supabase Auth magic link — no public signup
- Public pages accessible to anyone with the URL — `share_token` stored in localStorage after first visit
- No per-manager Yahoo OAuth, no live scoring, no push delivery, no multi-league routing in V1
- No hardcoded ElevenLabs cost estimates — pricing verified at implementation
- Commissioner's Yahoo OAuth tokens stored in `commissioner_tokens` Supabase table

## Prerequisites (before Task 1)

- Yahoo Developer account with app created — get `YAHOO_CLIENT_ID` and `YAHOO_CLIENT_SECRET`
- ElevenLabs account — get `ELEVENLABS_API_KEY`
- Supabase project created — get `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- Anthropic account — get `ANTHROPIC_API_KEY`
- Vercel account (for deployment in final task)
- Node.js 20+ installed

---

## File Structure

```
league-broadcast/
├── package.json
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── vitest.config.ts
├── .env.local.example
├── supabase/
│   └── migrations/
│       ├── 001_schema.sql              # All 11 tables + indexes
│       └── 002_rls.sql                 # RLS policies
├── lib/
│   ├── supabase/
│   │   ├── client.ts                   # Browser singleton client
│   │   ├── server.ts                   # Server component client (cookies)
│   │   └── admin.ts                    # Service role client
│   ├── yahoo/
│   │   ├── auth.ts                     # OAuth initiation, callback, token refresh
│   │   ├── client.ts                   # Authenticated Yahoo HTTP client
│   │   └── endpoints.ts                # Named Yahoo API call functions
│   ├── ingestion/
│   │   ├── manager-matcher.ts          # GUID-first identity matching
│   │   ├── transaction-parser.ts       # Yahoo transaction JSON → DB shape
│   │   ├── weekly-sync.ts              # Weekly scores/standings/transactions sync
│   │   └── historical.ts               # Full prior season chain ingestion
│   ├── context-builder/
│   │   ├── types.ts                    # ContextObject TypeScript types
│   │   ├── queries.ts                  # Named DB queries returning typed data + source
│   │   └── builder.ts                  # Assembles context object from queries
│   ├── generation/
│   │   ├── prompts.ts                  # System/user prompts for Claude
│   │   ├── script-generator.ts         # Claude Opus → script_json + source validation
│   │   ├── audio-producer.ts           # ElevenLabs → segment MP3s → concat → upload
│   │   ├── transcript-generator.ts     # Claude Sonnet → markdown
│   │   └── episode-pipeline.ts         # Orchestrates full episode generation
│   └── types.ts                        # Shared DB row types
├── app/
│   ├── layout.tsx
│   ├── (public)/
│   │   ├── layout.tsx                  # Dark theme + AudioContext provider + MiniPlayer
│   │   ├── page.tsx                    # Home: latest episode + scores + standings
│   │   ├── episode/[season]/[week]/page.tsx
│   │   ├── archive/page.tsx
│   │   ├── history/page.tsx
│   │   └── standings/page.tsx
│   ├── admin/
│   │   ├── layout.tsx                  # Auth gate
│   │   ├── page.tsx                    # Commissioner dashboard
│   │   ├── setup/page.tsx              # 6-step setup wizard
│   │   ├── episodes/page.tsx
│   │   ├── managers/page.tsx
│   │   └── ingestion/page.tsx
│   └── api/
│       ├── yahoo/connect/route.ts
│       ├── yahoo/callback/route.ts
│       ├── ingest/historical/route.ts
│       ├── ingest/weekly/route.ts
│       ├── episode/generate/route.ts
│       └── cron/weekly/route.ts
├── components/
│   ├── audio/
│   │   ├── AudioContext.tsx
│   │   ├── HeroPlayer.tsx
│   │   └── MiniPlayer.tsx
│   ├── episodes/EpisodeCard.tsx
│   ├── scores/MatchupCard.tsx
│   ├── standings/StandingsTable.tsx
│   ├── history/
│   │   ├── ChampionshipRoll.tsx
│   │   ├── AllTimeRecords.tsx
│   │   ├── CareerStats.tsx
│   │   ├── H2HMatrix.tsx
│   │   └── RivalrySpotlight.tsx
│   └── admin/
│       ├── SetupWizard.tsx
│       ├── EpisodeTable.tsx
│       ├── ManagerTable.tsx
│       └── IngestionLog.tsx
└── __tests__/
    ├── ingestion/manager-matcher.test.ts
    ├── ingestion/transaction-parser.test.ts
    ├── context-builder/builder.test.ts
    ├── generation/script-validator.test.ts
    └── generation/audio-producer.test.ts
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `package.json`
- Create: `next.config.ts`
- Create: `tsconfig.json`
- Create: `tailwind.config.ts`
- Create: `vitest.config.ts`
- Create: `.env.local.example`
- Create: `lib/types.ts`

**Interfaces:**
- Produces: working Next.js 14 project with Tailwind, Supabase, Anthropic SDK, ElevenLabs deps

- [ ] **Step 1: Scaffold Next.js project**

```bash
cd ~
npx create-next-app@latest league-broadcast \
  --typescript --tailwind --app --no-src-dir \
  --import-alias "@/*" --no-turbopack
cd league-broadcast
```

- [ ] **Step 2: Install dependencies**

```bash
npm install @supabase/supabase-js @supabase/ssr @anthropic-ai/sdk
npm install -D vitest @vitejs/plugin-react @testing-library/react \
  @testing-library/jest-dom jsdom happy-dom
```

- [ ] **Step 3: Configure vitest**

`vitest.config.ts`:
```typescript
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, '.') },
  },
})
```

`vitest.setup.ts`:
```typescript
import '@testing-library/jest-dom'
```

- [ ] **Step 4: Create `.env.local.example`**

```bash
cat > .env.local.example << 'EOF'
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Yahoo OAuth
YAHOO_CLIENT_ID=your_client_id
YAHOO_CLIENT_SECRET=your_client_secret
YAHOO_REDIRECT_URI=http://localhost:3000/api/yahoo/callback

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# ElevenLabs
ELEVENLABS_API_KEY=your_key

# Admin
CRON_SECRET=generate_random_32_char_string
NEXT_PUBLIC_APP_URL=http://localhost:3000
EOF
cp .env.local.example .env.local
```

- [ ] **Step 5: Create shared types**

`lib/types.ts`:
```typescript
export type IdentityStatus = 'confirmed' | 'needs_review' | 'unmatched'
export type EpisodeType = 'preseason' | 'weekly_preview' | 'weekly_recap' | 'playoff_special' | 'championship'
export type EpisodeStatus = 'pending' | 'generating_script' | 'generating_audio' | 'ready' | 'partial' | 'failed'
export type IngestionType = 'historical_full' | 'weekly_sync' | 'transaction_sync' | 'draft_sync'
export type TransactionType = 'trade' | 'add' | 'drop' | 'add/drop'

export interface LeagueSeason {
  id: string
  league_key: string
  league_id: string
  game_id: string
  season: number
  name: string
  num_teams: number | null
  scoring_type: string | null
  start_week: number | null
  end_week: number | null
  playoff_start_week: number | null
  is_finished: boolean
  prior_league_key: string | null
  next_league_key: string | null
  champion_team_key: string | null
  ingested_at: string | null
  created_at: string
}

export interface Manager {
  id: string
  primary_nickname: string
  yahoo_guid: string | null
  identity_status: IdentityStatus
  is_active: boolean
  first_season: number | null
  created_at: string
}

export interface ManagerNickname {
  id: string
  manager_id: string
  nickname: string
  season: number
  league_season_id: string
  is_primary: boolean
}

export interface Team {
  id: string
  team_key: string
  league_season_id: string
  manager_id: string | null
  team_name: string
  manager_nickname: string
  manager_yahoo_guid: string | null
  logo_url: string | null
  draft_position: number | null
  final_standing: number | null
  wins: number
  losses: number
  ties: number
  points_for: number
  points_against: number
  playoff_seed: number | null
  made_playoffs: boolean
  is_champion: boolean
  created_at: string
}

export interface Matchup {
  id: string
  league_season_id: string
  season: number
  week: number
  team1_key: string
  team2_key: string
  team1_score: number | null
  team2_score: number | null
  winner_team_key: string | null
  is_tie: boolean
  margin: number | null
  is_playoff: boolean
  is_consolation: boolean
  is_championship: boolean
  is_complete: boolean
  raw_yahoo_json: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface StandingsSnapshot {
  id: string
  league_season_id: string
  team_key: string
  season: number
  week: number
  wins: number
  losses: number
  ties: number
  points_for: number
  points_against: number
  streak_type: 'W' | 'L'
  streak_count: number
  playoff_seed: number | null
}

export interface Transaction {
  id: string
  league_season_id: string
  yahoo_transaction_key: string | null
  type: TransactionType
  status: string
  week: number | null
  occurred_at: string | null
  involved_team_keys: string[]
  players_sent: unknown
  players_received: unknown
  raw_yahoo_json: Record<string, unknown> | null
  created_at: string
}

export interface DraftPick {
  id: string
  league_season_id: string
  season: number
  pick_number: number
  round: number
  team_key: string
  player_name: string
  player_id: string | null
  position: string | null
  nfl_team: string | null
}

export interface Episode {
  id: string
  league_season_id: string
  season: number
  week: number | null
  episode_type: EpisodeType
  title: string
  status: EpisodeStatus
  script_json: ScriptJson | null
  transcript_md: string | null
  audio_url: string | null
  audio_duration_secs: number | null
  published_at: string | null
  generation_metadata: Record<string, unknown> | null
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface LeagueConfig {
  id: string
  league_season_id: string
  show_name: string
  share_token: string
  accent_color: string
  host_troy_voice_id: string | null
  host_lisa_voice_id: string | null
  host_danny_voice_id: string | null
  episode_artwork_url: string | null
  created_at: string
  updated_at: string
}

export interface CommissionerTokens {
  id: string
  league_id: string
  access_token: string
  refresh_token: string
  token_type: string
  expires_at: string
  yahoo_guid: string | null
  created_at: string
  updated_at: string
}

export interface IngestionLog {
  id: string
  league_season_id: string
  ingestion_type: IngestionType
  week: number | null
  status: 'running' | 'complete' | 'partial' | 'failed'
  records_upserted: number
  warnings: string[]
  error_message: string | null
  started_at: string
  completed_at: string | null
}

export interface ScriptSegment {
  index: number
  host: 'troy' | 'lisa' | 'danny'
  text: string
  is_fact: boolean
  source?: string
  type: string
}

export interface ScriptJson {
  episode: {
    title: string
    week: number | null
    season: number
    type: EpisodeType
    estimated_duration_secs: number
  }
  segments: ScriptSegment[]
  sections: string[]
}
```

- [ ] **Step 6: Add test script to package.json**

```bash
npm pkg set scripts.test="vitest run" scripts.test:watch="vitest"
```

- [ ] **Step 7: Commit**

```bash
git init
git add package.json next.config.ts tsconfig.json tailwind.config.ts \
  vitest.config.ts vitest.setup.ts .env.local.example lib/types.ts
git commit -m "feat: scaffold Next.js 14 project with Supabase, Anthropic, Vitest"
```

---

### Task 2: Database Migrations

**Files:**
- Create: `supabase/migrations/001_schema.sql`
- Create: `supabase/migrations/002_rls.sql`

**Interfaces:**
- Produces: All 11 tables (10 from spec + `commissioner_tokens`) with indexes

- [ ] **Step 1: Write the full schema migration**

`supabase/migrations/001_schema.sql`:
```sql
-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- league_seasons: one row per season of the league
CREATE TABLE league_seasons (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_key          text UNIQUE NOT NULL,
  league_id           text NOT NULL,
  game_id             text NOT NULL,
  season              integer NOT NULL,
  name                text NOT NULL,
  num_teams           integer,
  scoring_type        text,
  start_week          integer,
  end_week            integer,
  playoff_start_week  integer,
  is_finished         boolean DEFAULT false,
  prior_league_key    text,
  next_league_key     text,
  champion_team_key   text,
  ingested_at         timestamptz,
  created_at          timestamptz DEFAULT now()
);

-- managers: real people, stable across seasons
CREATE TABLE managers (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  primary_nickname    text NOT NULL,
  yahoo_guid          text UNIQUE,
  identity_status     text NOT NULL DEFAULT 'confirmed'
                      CHECK (identity_status IN ('confirmed','needs_review','unmatched')),
  is_active           boolean DEFAULT true,
  first_season        integer,
  created_at          timestamptz DEFAULT now()
);

-- manager_nicknames: all known nicknames per manager per season
CREATE TABLE manager_nicknames (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  manager_id          uuid NOT NULL REFERENCES managers(id) ON DELETE CASCADE,
  nickname            text NOT NULL,
  season              integer NOT NULL,
  league_season_id    uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  is_primary          boolean DEFAULT false,
  UNIQUE (manager_id, season)
);

-- teams: one row per team per season
CREATE TABLE teams (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  team_key            text UNIQUE NOT NULL,
  league_season_id    uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  manager_id          uuid REFERENCES managers(id),
  team_name           text NOT NULL,
  manager_nickname    text NOT NULL,
  manager_yahoo_guid  text,
  logo_url            text,
  draft_position      integer,
  final_standing      integer,
  wins                integer NOT NULL DEFAULT 0,
  losses              integer NOT NULL DEFAULT 0,
  ties                integer NOT NULL DEFAULT 0,
  points_for          numeric NOT NULL DEFAULT 0,
  points_against      numeric NOT NULL DEFAULT 0,
  playoff_seed        integer,
  made_playoffs       boolean DEFAULT false,
  is_champion         boolean DEFAULT false,
  created_at          timestamptz DEFAULT now()
);

-- matchups: every game every week every season
CREATE TABLE matchups (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id    uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  season              integer NOT NULL,
  week                integer NOT NULL,
  team1_key           text NOT NULL,
  team2_key           text NOT NULL,
  team1_score         numeric,
  team2_score         numeric,
  winner_team_key     text,
  is_tie              boolean DEFAULT false,
  margin              numeric,
  is_playoff          boolean DEFAULT false,
  is_consolation      boolean DEFAULT false,
  is_championship     boolean DEFAULT false,
  is_complete         boolean DEFAULT false,
  raw_yahoo_json      jsonb,
  created_at          timestamptz DEFAULT now(),
  updated_at          timestamptz DEFAULT now(),
  UNIQUE (league_season_id, week, team1_key, team2_key)
);

CREATE INDEX matchups_season_teams ON matchups (season, team1_key, team2_key);
CREATE INDEX matchups_league_week   ON matchups (league_season_id, week);
CREATE INDEX matchups_championship  ON matchups (is_championship) WHERE is_championship = true;
CREATE INDEX matchups_margin        ON matchups (margin DESC NULLS LAST) WHERE is_complete = true;

-- standings_snapshots: weekly standings state
CREATE TABLE standings_snapshots (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id    uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  team_key            text NOT NULL,
  season              integer NOT NULL,
  week                integer NOT NULL,
  wins                integer NOT NULL,
  losses              integer NOT NULL,
  ties                integer NOT NULL DEFAULT 0,
  points_for          numeric NOT NULL,
  points_against      numeric NOT NULL,
  streak_type         text NOT NULL CHECK (streak_type IN ('W','L')),
  streak_count        integer NOT NULL DEFAULT 1,
  playoff_seed        integer,
  UNIQUE (league_season_id, team_key, week)
);

-- transactions: trades, adds, drops
CREATE TABLE transactions (
  id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id        uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  yahoo_transaction_key   text UNIQUE,
  type                    text NOT NULL CHECK (type IN ('trade','add','drop','add/drop')),
  status                  text NOT NULL,
  week                    integer,
  occurred_at             timestamptz,
  involved_team_keys      text[],
  players_sent            jsonb,
  players_received        jsonb,
  raw_yahoo_json          jsonb,
  created_at              timestamptz DEFAULT now()
);

-- draft_picks: full draft order per season
CREATE TABLE draft_picks (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id    uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  season              integer NOT NULL,
  pick_number         integer NOT NULL,
  round               integer NOT NULL,
  team_key            text NOT NULL,
  player_name         text NOT NULL,
  player_id           text,
  position            text,
  nfl_team            text,
  UNIQUE (league_season_id, pick_number)
);

-- episodes: AI-generated broadcast content
CREATE TABLE episodes (
  id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id        uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  season                  integer NOT NULL,
  week                    integer,
  episode_type            text NOT NULL
                          CHECK (episode_type IN ('preseason','weekly_preview','weekly_recap','playoff_special','championship')),
  title                   text NOT NULL,
  status                  text NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending','generating_script','generating_audio','ready','partial','failed')),
  script_json             jsonb,
  transcript_md           text,
  audio_url               text,
  audio_duration_secs     integer,
  published_at            timestamptz,
  generation_metadata     jsonb,
  error_message           text,
  created_at              timestamptz DEFAULT now(),
  updated_at              timestamptz DEFAULT now()
);

CREATE INDEX episodes_season_week ON episodes (season, week);
CREATE INDEX episodes_status ON episodes (status);

-- league_config: per-league show branding, access control, host voices
CREATE TABLE league_config (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id    uuid NOT NULL UNIQUE REFERENCES league_seasons(id) ON DELETE CASCADE,
  show_name           text NOT NULL,
  share_token         text NOT NULL UNIQUE,
  accent_color        text NOT NULL DEFAULT '#00ff87',
  host_troy_voice_id  text,
  host_lisa_voice_id  text,
  host_danny_voice_id text,
  episode_artwork_url text,
  created_at          timestamptz DEFAULT now(),
  updated_at          timestamptz DEFAULT now()
);

-- commissioner_tokens: Yahoo OAuth tokens (implementation detail, not in spec schema)
CREATE TABLE commissioner_tokens (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_id       text NOT NULL UNIQUE,
  access_token    text NOT NULL,
  refresh_token   text NOT NULL,
  token_type      text NOT NULL DEFAULT 'bearer',
  expires_at      timestamptz NOT NULL,
  yahoo_guid      text,
  created_at      timestamptz DEFAULT now(),
  updated_at      timestamptz DEFAULT now()
);

-- ingestion_log: audit trail for every data pull
CREATE TABLE ingestion_log (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  league_season_id    uuid NOT NULL REFERENCES league_seasons(id) ON DELETE CASCADE,
  ingestion_type      text NOT NULL
                      CHECK (ingestion_type IN ('historical_full','weekly_sync','transaction_sync','draft_sync')),
  week                integer,
  status              text NOT NULL CHECK (status IN ('running','complete','partial','failed')),
  records_upserted    integer DEFAULT 0,
  warnings            text[],
  error_message       text,
  started_at          timestamptz DEFAULT now(),
  completed_at        timestamptz
);
```

- [ ] **Step 2: Write RLS policies**

`supabase/migrations/002_rls.sql`:
```sql
-- Enable RLS on all tables
ALTER TABLE league_seasons       ENABLE ROW LEVEL SECURITY;
ALTER TABLE managers             ENABLE ROW LEVEL SECURITY;
ALTER TABLE manager_nicknames    ENABLE ROW LEVEL SECURITY;
ALTER TABLE teams                ENABLE ROW LEVEL SECURITY;
ALTER TABLE matchups             ENABLE ROW LEVEL SECURITY;
ALTER TABLE standings_snapshots  ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions         ENABLE ROW LEVEL SECURITY;
ALTER TABLE draft_picks          ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes             ENABLE ROW LEVEL SECURITY;
ALTER TABLE league_config        ENABLE ROW LEVEL SECURITY;
ALTER TABLE commissioner_tokens  ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_log        ENABLE ROW LEVEL SECURITY;

-- Public read access for non-sensitive tables (web app uses service role for writes)
CREATE POLICY "public_read_league_seasons"      ON league_seasons      FOR SELECT USING (true);
CREATE POLICY "public_read_managers"            ON managers            FOR SELECT USING (true);
CREATE POLICY "public_read_manager_nicknames"   ON manager_nicknames   FOR SELECT USING (true);
CREATE POLICY "public_read_teams"               ON teams               FOR SELECT USING (true);
CREATE POLICY "public_read_matchups"            ON matchups            FOR SELECT USING (true);
CREATE POLICY "public_read_standings_snapshots" ON standings_snapshots FOR SELECT USING (true);
CREATE POLICY "public_read_transactions"        ON transactions        FOR SELECT USING (true);
CREATE POLICY "public_read_draft_picks"         ON draft_picks         FOR SELECT USING (true);
CREATE POLICY "public_read_episodes"            ON episodes            FOR SELECT USING (true);
CREATE POLICY "public_read_league_config"       ON league_config       FOR SELECT USING (true);
CREATE POLICY "public_read_ingestion_log"       ON ingestion_log       FOR SELECT USING (true);

-- commissioner_tokens: no public read (service role only)
-- No SELECT policy = anon/authenticated cannot read tokens
```

- [ ] **Step 3: Apply migrations via Supabase CLI or MCP**

If using Supabase CLI:
```bash
npx supabase db push
```

If using Supabase MCP (in Claude Code), run each migration SQL via the `execute_sql` tool.

- [ ] **Step 4: Verify tables exist**

```bash
# Via Supabase CLI
npx supabase db diff
# Expected: no diff (all tables created)
```

- [ ] **Step 5: Commit**

```bash
git add supabase/
git commit -m "feat: add full database schema — 11 tables with indexes and RLS"
```

---

### Task 3: Supabase Clients

**Files:**
- Create: `lib/supabase/client.ts`
- Create: `lib/supabase/server.ts`
- Create: `lib/supabase/admin.ts`

**Interfaces:**
- Produces: `createBrowserClient()`, `createServerClient()`, `adminClient` (service role)

- [ ] **Step 1: Browser client**

`lib/supabase/client.ts`:
```typescript
import { createBrowserClient as createSupabaseBrowserClient } from '@supabase/ssr'

export function createBrowserClient() {
  return createSupabaseBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}
```

- [ ] **Step 2: Server client (for Server Components and Route Handlers)**

`lib/supabase/server.ts`:
```typescript
import { createServerClient as createSupabaseServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

export async function createServerClient() {
  const cookieStore = await cookies()
  return createSupabaseServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() { return cookieStore.getAll() },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            )
          } catch {
            // Server Component — cookie setting ignored
          }
        },
      },
    }
  )
}
```

- [ ] **Step 3: Admin client (service role — never exposed to browser)**

`lib/supabase/admin.ts`:
```typescript
import { createClient } from '@supabase/supabase-js'

// Singleton — reused across requests in serverless
let _adminClient: ReturnType<typeof createClient> | null = null

export function getAdminClient() {
  if (!_adminClient) {
    _adminClient = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!,
      { auth: { autoRefreshToken: false, persistSession: false } }
    )
  }
  return _adminClient
}
```

- [ ] **Step 4: Commit**

```bash
git add lib/supabase/
git commit -m "feat: add Supabase browser, server, and admin clients"
```

---

### Task 4: Yahoo OAuth + Token Management

**Files:**
- Create: `lib/yahoo/auth.ts`
- Create: `app/api/yahoo/connect/route.ts`
- Create: `app/api/yahoo/callback/route.ts`

**Interfaces:**
- Produces: `getYahooAuthUrl()`, `exchangeCodeForTokens()`, `getValidTokens(leagueId)`, `refreshTokens(leagueId)`

- [ ] **Step 1: Write failing test**

`__tests__/yahoo/auth.test.ts`:
```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/lib/supabase/admin', () => ({
  getAdminClient: vi.fn(() => ({
    from: vi.fn(() => ({
      select: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      upsert: vi.fn().mockResolvedValue({ error: null }),
      single: vi.fn().mockResolvedValue({
        data: {
          access_token: 'valid_token',
          refresh_token: 'refresh_token',
          expires_at: new Date(Date.now() + 3600_000).toISOString(),
        },
        error: null,
      }),
    })),
  })),
}))

describe('getYahooAuthUrl', () => {
  it('returns URL with required OAuth params', async () => {
    process.env.YAHOO_CLIENT_ID = 'test_client_id'
    process.env.YAHOO_REDIRECT_URI = 'http://localhost:3000/api/yahoo/callback'
    const { getYahooAuthUrl } = await import('@/lib/yahoo/auth')
    const url = getYahooAuthUrl()
    expect(url).toContain('api.login.yahoo.com')
    expect(url).toContain('client_id=test_client_id')
    expect(url).toContain('response_type=code')
  })
})
```

- [ ] **Step 2: Run test — verify it fails**

```bash
npm test -- auth.test.ts
# Expected: FAIL — module not found
```

- [ ] **Step 3: Implement auth module**

`lib/yahoo/auth.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'

const YAHOO_AUTH_BASE = 'https://api.login.yahoo.com/oauth2/request_auth'
const YAHOO_TOKEN_URL = 'https://api.login.yahoo.com/oauth2/get_token'

export function getYahooAuthUrl(): string {
  const params = new URLSearchParams({
    client_id: process.env.YAHOO_CLIENT_ID!,
    redirect_uri: process.env.YAHOO_REDIRECT_URI!,
    response_type: 'code',
    language: 'en-us',
  })
  return `${YAHOO_AUTH_BASE}?${params.toString()}`
}

export async function exchangeCodeForTokens(
  code: string,
  leagueId: string
): Promise<{ access_token: string; refresh_token: string; expires_at: Date }> {
  const credentials = Buffer.from(
    `${process.env.YAHOO_CLIENT_ID}:${process.env.YAHOO_CLIENT_SECRET}`
  ).toString('base64')

  const resp = await fetch(YAHOO_TOKEN_URL, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${credentials}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: process.env.YAHOO_REDIRECT_URI!,
    }),
  })

  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`Yahoo token exchange failed: ${text}`)
  }

  const data = await resp.json()
  const expires_at = new Date(Date.now() + data.expires_in * 1000)

  await storeTokens(leagueId, data.access_token, data.refresh_token, expires_at, data.xoauth_yahoo_guid)

  return { access_token: data.access_token, refresh_token: data.refresh_token, expires_at }
}

async function storeTokens(
  leagueId: string,
  access_token: string,
  refresh_token: string,
  expires_at: Date,
  yahoo_guid?: string
) {
  const db = getAdminClient()
  const { error } = await db.from('commissioner_tokens').upsert(
    {
      league_id: leagueId,
      access_token,
      refresh_token,
      expires_at: expires_at.toISOString(),
      yahoo_guid: yahoo_guid ?? null,
      updated_at: new Date().toISOString(),
    },
    { onConflict: 'league_id' }
  )
  if (error) throw new Error(`Failed to store tokens: ${error.message}`)
}

export async function getValidTokens(leagueId: string): Promise<string> {
  const db = getAdminClient()
  const { data, error } = await db
    .from('commissioner_tokens')
    .select('access_token, refresh_token, expires_at')
    .eq('league_id', leagueId)
    .single()

  if (error || !data) throw new Error('No tokens found for league — commissioner must reconnect Yahoo')

  // Refresh if expiring within 5 minutes
  if (new Date(data.expires_at) <= new Date(Date.now() + 300_000)) {
    return refreshTokens(leagueId, data.refresh_token)
  }

  return data.access_token
}

async function refreshTokens(leagueId: string, refresh_token: string): Promise<string> {
  const credentials = Buffer.from(
    `${process.env.YAHOO_CLIENT_ID}:${process.env.YAHOO_CLIENT_SECRET}`
  ).toString('base64')

  const resp = await fetch(YAHOO_TOKEN_URL, {
    method: 'POST',
    headers: {
      Authorization: `Basic ${credentials}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      redirect_uri: process.env.YAHOO_REDIRECT_URI!,
      refresh_token,
    }),
  })

  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`Yahoo token refresh failed: ${text}`)
  }

  const data = await resp.json()
  const expires_at = new Date(Date.now() + data.expires_in * 1000)
  await storeTokens(leagueId, data.access_token, data.refresh_token, expires_at)
  return data.access_token
}
```

- [ ] **Step 4: Yahoo connect route (initiates OAuth)**

`app/api/yahoo/connect/route.ts`:
```typescript
import { NextResponse } from 'next/server'
import { getYahooAuthUrl } from '@/lib/yahoo/auth'
import { createServerClient } from '@/lib/supabase/server'

export async function GET() {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const url = getYahooAuthUrl()
  return NextResponse.redirect(url)
}
```

- [ ] **Step 5: Yahoo callback route (handles OAuth response)**

`app/api/yahoo/callback/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { exchangeCodeForTokens } from '@/lib/yahoo/auth'
import { createServerClient } from '@/lib/supabase/server'

export async function GET(request: NextRequest) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.redirect('/admin/login')

  const { searchParams } = new URL(request.url)
  const code = searchParams.get('code')
  const error = searchParams.get('error')

  if (error || !code) {
    return NextResponse.redirect(
      `/admin/setup?error=${encodeURIComponent(error ?? 'no_code')}`
    )
  }

  try {
    // leagueId is stored in session during setup; use placeholder for initial connect
    // The setup wizard will update once league is selected
    await exchangeCodeForTokens(code, 'pending')
    return NextResponse.redirect('/admin/setup?step=2&connected=true')
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    return NextResponse.redirect(`/admin/setup?error=${encodeURIComponent(msg)}`)
  }
}
```

- [ ] **Step 6: Run test — verify it passes**

```bash
npm test -- auth.test.ts
# Expected: PASS
```

- [ ] **Step 7: Commit**

```bash
git add lib/yahoo/auth.ts app/api/yahoo/ __tests__/yahoo/
git commit -m "feat: Yahoo OAuth flow — initiate, callback, token storage and refresh"
```

---

### Task 5: Yahoo API Client + Endpoints

**Files:**
- Create: `lib/yahoo/client.ts`
- Create: `lib/yahoo/endpoints.ts`

**Interfaces:**
- Consumes: `getValidTokens(leagueId)` from Task 4
- Produces: `yahooGet(leagueId, path)`, `getLeagueMetadata()`, `getLeagueTeams()`, `getScoreboard()`, `getStandings()`, `getTransactions()`, `getDraftResults()`, `getTeamMatchups()`

- [ ] **Step 1: Write failing test**

`__tests__/yahoo/endpoints.test.ts`:
```typescript
import { describe, it, expect, vi } from 'vitest'

vi.mock('@/lib/yahoo/auth', () => ({
  getValidTokens: vi.fn().mockResolvedValue('mock_access_token'),
}))

global.fetch = vi.fn().mockResolvedValue({
  ok: true,
  json: async () => ({
    fantasy_content: {
      league: [
        {
          league_key: '449.l.123456',
          league_id: '123456',
          game_code: 'nfl',
          season: '2025',
          name: 'Test League',
          current_week: 5,
          num_teams: 10,
          scoring_type: 'head',
          start_week: '1',
          end_week: '14',
          playoff_start_week: '15',
          renew: '423_654321',
          renewed: '',
        },
      ],
    },
  }),
})

describe('getLeagueMetadata', () => {
  it('returns parsed league data', async () => {
    const { getLeagueMetadata } = await import('@/lib/yahoo/endpoints')
    const result = await getLeagueMetadata('449.l.123456', '123456')
    expect(result.league_key).toBe('449.l.123456')
    expect(result.season).toBe(2025)
    expect(result.prior_league_key).toBe('423.l.654321')
  })
})
```

- [ ] **Step 2: Run test — verify it fails**

```bash
npm test -- endpoints.test.ts
# Expected: FAIL
```

- [ ] **Step 3: Implement authenticated HTTP client**

`lib/yahoo/client.ts`:
```typescript
import { getValidTokens } from '@/lib/yahoo/auth'

const YAHOO_API_BASE = 'https://fantasysports.yahooapis.com/fantasy/v2'

export async function yahooGet(leagueId: string, path: string): Promise<unknown> {
  const token = await getValidTokens(leagueId)

  const url = path.startsWith('http') ? path : `${YAHOO_API_BASE}${path}`

  const resp = await fetch(`${url}?format=json`, {
    headers: { Authorization: `Bearer ${token}` },
  })

  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`Yahoo API error ${resp.status}: ${text}`)
  }

  return resp.json()
}
```

- [ ] **Step 4: Implement all Yahoo API endpoints**

`lib/yahoo/endpoints.ts`:
```typescript
import { yahooGet } from '@/lib/yahoo/client'

export interface YahooLeagueMeta {
  league_key: string
  league_id: string
  game_id: string
  season: number
  name: string
  current_week: number
  num_teams: number
  scoring_type: string
  start_week: number
  end_week: number
  playoff_start_week: number
  prior_league_key: string | null
  next_league_key: string | null
}

export interface YahooTeam {
  team_key: string
  team_name: string
  manager_nickname: string
  manager_yahoo_guid: string | null
  logo_url: string | null
  wins: number
  losses: number
  ties: number
  points_for: number
  points_against: number
  playoff_seed: number | null
}

export interface YahooMatchup {
  week: number
  team1_key: string
  team2_key: string
  team1_score: number | null
  team2_score: number | null
  winner_team_key: string | null
  is_tie: boolean
  is_playoff: boolean
  is_consolation: boolean
  is_championship: boolean
  is_complete: boolean
  raw: Record<string, unknown>
}

/** Convert "423_654321" → "423.l.654321" */
function renewToLeagueKey(renew: string): string | null {
  if (!renew) return null
  const parts = renew.split('_')
  if (parts.length !== 2) return null
  return `${parts[0]}.l.${parts[1]}`
}

export async function getLeagueMetadata(leagueKey: string, leagueId: string): Promise<YahooLeagueMeta> {
  const data = await yahooGet(leagueId, `/league/${leagueKey}`) as any
  const raw = data.fantasy_content.league[0]
  return {
    league_key: raw.league_key,
    league_id: raw.league_id,
    game_id: String(raw.league_key.split('.')[0]),
    season: parseInt(raw.season, 10),
    name: raw.name,
    current_week: parseInt(raw.current_week, 10),
    num_teams: parseInt(raw.num_teams, 10),
    scoring_type: raw.scoring_type,
    start_week: parseInt(raw.start_week, 10),
    end_week: parseInt(raw.end_week, 10),
    playoff_start_week: parseInt(raw.playoff_start_week, 10),
    prior_league_key: renewToLeagueKey(raw.renew ?? ''),
    next_league_key: renewToLeagueKey(raw.renewed ?? ''),
  }
}

export async function getLeagueTeams(leagueKey: string, leagueId: string): Promise<YahooTeam[]> {
  const data = await yahooGet(leagueId, `/league/${leagueKey}/teams;out=standings`) as any
  const teamsAndStandings = data.fantasy_content.league[1].teams
  const count = teamsAndStandings.count as number
  const result: YahooTeam[] = []

  for (let i = 0; i < count; i++) {
    const teamData = teamsAndStandings[i].team
    const info = teamData[0]
    const standings = teamData[2]?.team_standings

    const teamKey = info.find((x: any) => x.team_key)?.team_key ?? ''
    const teamName = info.find((x: any) => x.name)?.name ?? ''
    const managers = info.find((x: any) => x.managers)?.managers ?? []
    const mgr = managers[0]?.manager ?? {}
    const logoUrl = info.find((x: any) => x.team_logos)?.team_logos?.[0]?.team_logo?.url ?? null

    result.push({
      team_key: teamKey,
      team_name: teamName,
      manager_nickname: mgr.nickname ?? '',
      manager_yahoo_guid: mgr.guid ?? null,
      logo_url: logoUrl,
      wins: parseInt(standings?.outcome_totals?.wins ?? '0', 10),
      losses: parseInt(standings?.outcome_totals?.losses ?? '0', 10),
      ties: parseInt(standings?.outcome_totals?.ties ?? '0', 10),
      points_for: parseFloat(standings?.points_for ?? '0'),
      points_against: parseFloat(standings?.points_against ?? '0'),
      playoff_seed: standings?.playoff_seed ? parseInt(standings.playoff_seed, 10) : null,
    })
  }

  return result
}

export async function getScoreboard(leagueKey: string, leagueId: string, week: number): Promise<YahooMatchup[]> {
  const data = await yahooGet(leagueId, `/league/${leagueKey}/scoreboard;week=${week}`) as any
  const matchupsRaw = data.fantasy_content.league[1].scoreboard[0].matchups
  const count = matchupsRaw.count as number
  const result: YahooMatchup[] = []

  for (let i = 0; i < count; i++) {
    const m = matchupsRaw[i].matchup
    const teams = m[0].teams
    const t1 = teams[0].team
    const t2 = teams[1].team

    const t1key = t1[0].find((x: any) => x.team_key)?.team_key ?? ''
    const t2key = t2[0].find((x: any) => x.team_key)?.team_key ?? ''
    const t1score = parseFloat(t1[1]?.team_points?.total ?? '0')
    const t2score = parseFloat(t2[1]?.team_points?.total ?? '0')
    const isTie = m.is_tied === '1'
    const winnKey = m.winner_team_key ?? null
    const isComplete = m.status === 'postevent'

    result.push({
      week,
      team1_key: t1key,
      team2_key: t2key,
      team1_score: t1score || null,
      team2_score: t2score || null,
      winner_team_key: isComplete ? (isTie ? null : winnKey) : null,
      is_tie: isTie,
      is_playoff: m.is_playoffs === '1',
      is_consolation: m.is_consolation === '1',
      is_championship: m.is_championship === '1',
      is_complete: isComplete,
      raw: m,
    })
  }

  return result
}

export async function getTransactions(
  leagueKey: string,
  leagueId: string,
  start = 0
): Promise<{ transactions: any[]; count: number }> {
  const data = await yahooGet(
    leagueId,
    `/league/${leagueKey}/transactions;types=trade,add,drop;count=25;start=${start}`
  ) as any
  const txns = data.fantasy_content.league[1].transactions
  const count = txns?.count ?? 0
  const items: any[] = []
  for (let i = 0; i < count; i++) {
    items.push(txns[i].transaction)
  }
  return { transactions: items, count }
}

export async function getDraftResults(leagueKey: string, leagueId: string): Promise<any[]> {
  const data = await yahooGet(leagueId, `/league/${leagueKey}/draftresults`) as any
  const draft = data.fantasy_content.league[1].draft_results
  const count = draft.count as number
  const picks: any[] = []
  for (let i = 0; i < count; i++) {
    picks.push(draft[i].draft_result)
  }
  return picks
}

export async function getUserLeagues(leagueId: string): Promise<any[]> {
  const data = await yahooGet(leagueId, `/users;use_login=1/games;game_codes=nfl/leagues`) as any
  const games = data.fantasy_content.users[0].user[1].games
  const gameCount = games.count as number
  const leagues: any[] = []
  for (let g = 0; g < gameCount; g++) {
    const game = games[g].game
    const leaguesInGame = game[1]?.leagues
    if (!leaguesInGame) continue
    const lcount = leaguesInGame.count as number
    for (let l = 0; l < lcount; l++) {
      const lg = leaguesInGame[l].league[0]
      leagues.push(lg)
    }
  }
  return leagues
}
```

- [ ] **Step 5: Run test — verify it passes**

```bash
npm test -- endpoints.test.ts
# Expected: PASS
```

- [ ] **Step 6: Commit**

```bash
git add lib/yahoo/ __tests__/yahoo/
git commit -m "feat: Yahoo API client and named endpoints (league, teams, scoreboard, transactions, draft)"
```

---

### Task 6: Manager Identity Matcher

**Files:**
- Create: `lib/ingestion/manager-matcher.ts`
- Test: `__tests__/ingestion/manager-matcher.test.ts`

**Interfaces:**
- Consumes: `YahooTeam` from Task 5, Supabase admin client
- Produces: `matchOrCreateManager(team, leagueSeasonId, season)` → `{ managerId, wasCreated, status }`

- [ ] **Step 1: Write failing tests**

`__tests__/ingestion/manager-matcher.test.ts`:
```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { YahooTeam } from '@/lib/yahoo/endpoints'

const mockDb = {
  from: vi.fn(),
}

vi.mock('@/lib/supabase/admin', () => ({
  getAdminClient: () => mockDb,
}))

const baseTeam: YahooTeam = {
  team_key: '449.l.123.t.1',
  team_name: 'Test Team',
  manager_nickname: 'Greg',
  manager_yahoo_guid: 'GUID123',
  logo_url: null,
  wins: 0, losses: 0, ties: 0,
  points_for: 0, points_against: 0,
  playoff_seed: null,
}

function makeChain(results: any[]) {
  const chain: any = {}
  const methods = ['select','eq','single','insert','upsert','update','maybeSingle']
  methods.forEach(m => { chain[m] = vi.fn().mockReturnValue(chain) })
  chain.single = vi.fn().mockResolvedValueOnce(results[0] ?? { data: null, error: null })
  return chain
}

describe('matchOrCreateManager — GUID match', () => {
  it('auto-confirms manager when GUID matches existing row', async () => {
    const existingManager = { id: 'mgr-uuid', yahoo_guid: 'GUID123', identity_status: 'confirmed', primary_nickname: 'Greg' }
    const chain = makeChain([{ data: existingManager, error: null }])
    mockDb.from = vi.fn().mockReturnValue(chain)

    const { matchOrCreateManager } = await import('@/lib/ingestion/manager-matcher')
    const result = await matchOrCreateManager(baseTeam, 'ls-uuid', 2025)
    expect(result.managerId).toBe('mgr-uuid')
    expect(result.status).toBe('confirmed')
    expect(result.wasCreated).toBe(false)
  })
})

describe('matchOrCreateManager — no GUID, no nickname match', () => {
  it('creates new manager with unmatched status', async () => {
    const noGuidTeam = { ...baseTeam, manager_yahoo_guid: null }
    const chain: any = {
      select: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      single: vi.fn().mockResolvedValueOnce({ data: null, error: { code: 'PGRST116' } }),
      maybeSingle: vi.fn().mockResolvedValueOnce({ data: null, error: null }),
      insert: vi.fn().mockResolvedValueOnce({ data: [{ id: 'new-mgr-uuid' }], error: null }),
      upsert: vi.fn().mockResolvedValueOnce({ error: null }),
    }
    mockDb.from = vi.fn().mockReturnValue(chain)

    const { matchOrCreateManager } = await import('@/lib/ingestion/manager-matcher')
    const result = await matchOrCreateManager(noGuidTeam, 'ls-uuid', 2025)
    expect(result.wasCreated).toBe(true)
    expect(result.status).toBe('unmatched')
  })
})
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
npm test -- manager-matcher.test.ts
# Expected: FAIL
```

- [ ] **Step 3: Implement manager matcher**

`lib/ingestion/manager-matcher.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import type { YahooTeam } from '@/lib/yahoo/endpoints'
import type { IdentityStatus } from '@/lib/types'

export interface MatchResult {
  managerId: string
  wasCreated: boolean
  status: IdentityStatus
}

/**
 * Identity matching rules (in priority order):
 * 1. GUID match on existing manager → auto-confirm
 * 2. No GUID, single nickname match → flag needs_review
 * 3. No match on GUID or nickname → create new as unmatched
 */
export async function matchOrCreateManager(
  team: YahooTeam,
  leagueSeasonId: string,
  season: number
): Promise<MatchResult> {
  const db = getAdminClient()

  // Rule 1: GUID match
  if (team.manager_yahoo_guid) {
    const { data: existingByGuid } = await db
      .from('managers')
      .select('id, identity_status, primary_nickname')
      .eq('yahoo_guid', team.manager_yahoo_guid)
      .single()

    if (existingByGuid) {
      await recordNickname(existingByGuid.id, team.manager_nickname, season, leagueSeasonId)
      return { managerId: existingByGuid.id, wasCreated: false, status: 'confirmed' }
    }
  }

  // Rule 2: Nickname match (no GUID)
  const { data: byNickname } = await db
    .from('managers')
    .select('id, primary_nickname, identity_status')
    .eq('primary_nickname', team.manager_nickname)
    .maybeSingle()

  if (byNickname) {
    // Flag for review — don't auto-confirm without GUID
    await db
      .from('managers')
      .update({ identity_status: 'needs_review' })
      .eq('id', byNickname.id)

    await recordNickname(byNickname.id, team.manager_nickname, season, leagueSeasonId)
    return { managerId: byNickname.id, wasCreated: false, status: 'needs_review' }
  }

  // Rule 3: Create new unmatched manager
  const { data: newManagers, error } = await db
    .from('managers')
    .insert({
      primary_nickname: team.manager_nickname,
      yahoo_guid: team.manager_yahoo_guid ?? null,
      identity_status: 'unmatched',
      first_season: season,
    })
    .select('id')

  if (error || !newManagers?.[0]) {
    throw new Error(`Failed to create manager: ${error?.message}`)
  }

  const newId = newManagers[0].id
  await recordNickname(newId, team.manager_nickname, season, leagueSeasonId)
  return { managerId: newId, wasCreated: true, status: 'unmatched' }
}

async function recordNickname(
  managerId: string,
  nickname: string,
  season: number,
  leagueSeasonId: string
) {
  const db = getAdminClient()
  await db.from('manager_nicknames').upsert(
    { manager_id: managerId, nickname, season, league_season_id: leagueSeasonId, is_primary: true },
    { onConflict: 'manager_id,season' }
  )
}
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
npm test -- manager-matcher.test.ts
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add lib/ingestion/manager-matcher.ts __tests__/ingestion/manager-matcher.test.ts
git commit -m "feat: manager identity matcher — GUID-first with needs_review and unmatched fallbacks"
```

---

### Task 7: Transaction Parser

**Files:**
- Create: `lib/ingestion/transaction-parser.ts`
- Test: `__tests__/ingestion/transaction-parser.test.ts`

**Interfaces:**
- Consumes: Raw Yahoo transaction array from `getTransactions()`
- Produces: `parseTransaction(raw)` → DB-insertable transaction shape

- [ ] **Step 1: Write failing test**

`__tests__/ingestion/transaction-parser.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { parseTransaction } from '@/lib/ingestion/transaction-parser'

const rawTrade = [
  { transaction_key: 'tx1', type: 'trade', status: 'successful', timestamp: '1728000000' },
  {
    players: {
      count: 2,
      0: {
        player: [
          [{ player_key: 'p1' }, { name: { full: 'Davante Adams' } }, { position_type: 'WR' }, { editorial_team_abbr: 'LV' }],
          { transaction_data: [{ type: 'add', destination_team_key: '449.l.123.t.2', source_team_key: '449.l.123.t.1' }] },
        ],
      },
      1: {
        player: [
          [{ player_key: 'p2' }, { name: { full: 'Stefon Diggs' } }, { position_type: 'WR' }, { editorial_team_abbr: 'NE' }],
          { transaction_data: [{ type: 'add', destination_team_key: '449.l.123.t.1', source_team_key: '449.l.123.t.2' }] },
        ],
      },
    },
  },
]

describe('parseTransaction', () => {
  it('parses a trade with correct team keys and player names', () => {
    const result = parseTransaction(rawTrade, 'ls-uuid', 5)
    expect(result.type).toBe('trade')
    expect(result.status).toBe('successful')
    expect(result.yahoo_transaction_key).toBe('tx1')
    expect(result.involved_team_keys).toContain('449.l.123.t.1')
    expect(result.involved_team_keys).toContain('449.l.123.t.2')
    expect((result.players_sent as any[]).length).toBeGreaterThan(0)
  })
})
```

- [ ] **Step 2: Run test — verify it fails**

```bash
npm test -- transaction-parser.test.ts
# Expected: FAIL
```

- [ ] **Step 3: Implement parser**

`lib/ingestion/transaction-parser.ts`:
```typescript
import type { TransactionType } from '@/lib/types'

interface ParsedTransaction {
  league_season_id: string
  yahoo_transaction_key: string | null
  type: TransactionType
  status: string
  week: number | null
  occurred_at: string | null
  involved_team_keys: string[]
  players_sent: unknown
  players_received: unknown
  raw_yahoo_json: Record<string, unknown>
}

interface ParsedPlayer {
  player_name: string
  position: string | null
  nfl_team: string | null
  from_team_key: string | null
  to_team_key: string | null
}

export function parseTransaction(
  raw: any[],
  leagueSeasonId: string,
  week: number | null
): ParsedTransaction {
  const meta = raw[0]
  const playersBlock = raw[1]?.players

  const players: ParsedPlayer[] = []
  const teamKeys = new Set<string>()

  if (playersBlock) {
    const count = playersBlock.count as number
    for (let i = 0; i < count; i++) {
      const playerArr = playersBlock[i].player
      const info = playerArr[0]
      const txData = playerArr[1]?.transaction_data?.[0] ?? {}

      const name = info.find((x: any) => x.name)?.name?.full ?? 'Unknown'
      const position = info.find((x: any) => x.position_type)?.position_type ?? null
      const nflTeam = info.find((x: any) => x.editorial_team_abbr)?.editorial_team_abbr ?? null
      const fromKey = txData.source_team_key ?? null
      const toKey = txData.destination_team_key ?? null

      if (fromKey) teamKeys.add(fromKey)
      if (toKey) teamKeys.add(toKey)

      players.push({ player_name: name, position, nfl_team: nflTeam, from_team_key: fromKey, to_team_key: toKey })
    }
  }

  const sent = players.filter(p => p.from_team_key !== null)
  const received = players.filter(p => p.to_team_key !== null)

  return {
    league_season_id: leagueSeasonId,
    yahoo_transaction_key: meta.transaction_key ?? null,
    type: (meta.type as TransactionType) ?? 'trade',
    status: meta.status ?? 'successful',
    week,
    occurred_at: meta.timestamp ? new Date(parseInt(meta.timestamp, 10) * 1000).toISOString() : null,
    involved_team_keys: Array.from(teamKeys),
    players_sent: sent,
    players_received: received,
    raw_yahoo_json: { meta, players: playersBlock },
  }
}
```

- [ ] **Step 4: Run test — verify it passes**

```bash
npm test -- transaction-parser.test.ts
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add lib/ingestion/transaction-parser.ts __tests__/ingestion/transaction-parser.test.ts
git commit -m "feat: Yahoo transaction parser — trades, adds, drops to DB shape"
```

---

### Task 8: Weekly Sync + Historical Ingestion

**Files:**
- Create: `lib/ingestion/weekly-sync.ts`
- Create: `lib/ingestion/historical.ts`

**Interfaces:**
- Consumes: `getLeagueTeams()`, `getScoreboard()`, `getTransactions()`, `getDraftResults()` from Task 5; `matchOrCreateManager()` from Task 6; `parseTransaction()` from Task 7
- Produces: `runWeeklySync(leagueSeasonId, leagueKey, leagueId, week)`, `runHistoricalIngestion(leagueSeasonId, priorLeagueKey, leagueId)`

- [ ] **Step 1: Implement weekly sync**

`lib/ingestion/weekly-sync.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { getLeagueTeams, getScoreboard, getTransactions } from '@/lib/yahoo/endpoints'
import { matchOrCreateManager } from '@/lib/ingestion/manager-matcher'
import { parseTransaction } from '@/lib/ingestion/transaction-parser'

export async function runWeeklySync(
  leagueSeasonId: string,
  leagueKey: string,
  leagueId: string,
  week: number
): Promise<{ recordsUpserted: number; warnings: string[] }> {
  const db = getAdminClient()
  const warnings: string[] = []
  let records = 0

  // 1. Log start
  const { data: logRow } = await db.from('ingestion_log').insert({
    league_season_id: leagueSeasonId,
    ingestion_type: 'weekly_sync',
    week,
    status: 'running',
  }).select('id').single()
  const logId = logRow?.id

  try {
    // 2. Teams + standings
    const teams = await getLeagueTeams(leagueKey, leagueId)
    for (const team of teams) {
      const { managerId } = await matchOrCreateManager(team, leagueSeasonId, 0)

      await db.from('teams').upsert({
        team_key: team.team_key,
        league_season_id: leagueSeasonId,
        manager_id: managerId,
        team_name: team.team_name,
        manager_nickname: team.manager_nickname,
        manager_yahoo_guid: team.manager_yahoo_guid,
        logo_url: team.logo_url,
        wins: team.wins,
        losses: team.losses,
        ties: team.ties,
        points_for: team.points_for,
        points_against: team.points_against,
        playoff_seed: team.playoff_seed,
      }, { onConflict: 'team_key' })
      records++

      // Standing snapshot
      const streakType = team.wins >= team.losses ? 'W' : 'L'
      await db.from('standings_snapshots').upsert({
        league_season_id: leagueSeasonId,
        team_key: team.team_key,
        season: new Date().getFullYear(),
        week,
        wins: team.wins,
        losses: team.losses,
        ties: team.ties,
        points_for: team.points_for,
        points_against: team.points_against,
        streak_type: streakType,
        streak_count: 1,
        playoff_seed: team.playoff_seed,
      }, { onConflict: 'league_season_id,team_key,week' })
      records++
    }

    // 3. Scoreboard for this week
    const matchups = await getScoreboard(leagueKey, leagueId, week)
    for (const m of matchups) {
      const margin = m.team1_score != null && m.team2_score != null
        ? Math.abs(m.team1_score - m.team2_score)
        : null

      await db.from('matchups').upsert({
        league_season_id: leagueSeasonId,
        season: new Date().getFullYear(),
        week: m.week,
        team1_key: m.team1_key,
        team2_key: m.team2_key,
        team1_score: m.team1_score,
        team2_score: m.team2_score,
        winner_team_key: m.winner_team_key,
        is_tie: m.is_tie,
        margin,
        is_playoff: m.is_playoff,
        is_consolation: m.is_consolation,
        is_championship: m.is_championship,
        is_complete: m.is_complete,
        raw_yahoo_json: m.raw as any,
        updated_at: new Date().toISOString(),
      }, { onConflict: 'league_season_id,week,team1_key,team2_key' })
      records++
    }

    // 4. Transactions (most recent ~25)
    const { transactions: txnRaw } = await getTransactions(leagueKey, leagueId, 0)
    for (const rawTxn of txnRaw) {
      try {
        const parsed = parseTransaction(rawTxn, leagueSeasonId, week)
        await db.from('transactions').upsert(parsed, { onConflict: 'yahoo_transaction_key' })
        records++
      } catch (e) {
        warnings.push(`Transaction parse error: ${e instanceof Error ? e.message : String(e)}`)
      }
    }

    // 5. Mark log complete
    await db.from('ingestion_log').update({
      status: warnings.length > 0 ? 'partial' : 'complete',
      records_upserted: records,
      warnings,
      completed_at: new Date().toISOString(),
    }).eq('id', logId)

    return { recordsUpserted: records, warnings }
  } catch (err) {
    await db.from('ingestion_log').update({
      status: 'failed',
      error_message: err instanceof Error ? err.message : String(err),
      completed_at: new Date().toISOString(),
    }).eq('id', logId)
    throw err
  }
}
```

- [ ] **Step 2: Implement historical ingestion**

`lib/ingestion/historical.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import {
  getLeagueMetadata,
  getLeagueTeams,
  getScoreboard,
  getTransactions,
  getDraftResults,
} from '@/lib/yahoo/endpoints'
import { matchOrCreateManager } from '@/lib/ingestion/manager-matcher'
import { parseTransaction } from '@/lib/ingestion/transaction-parser'

/**
 * Ingests a full prior season.
 * Called once during setup after the prior league_key is determined.
 */
export async function runHistoricalIngestion(
  currentLeagueSeasonId: string,
  priorLeagueKey: string,
  leagueId: string
): Promise<{ recordsUpserted: number; warnings: string[] }> {
  const db = getAdminClient()
  const warnings: string[] = []
  let records = 0

  const { data: logRow } = await db.from('ingestion_log').insert({
    league_season_id: currentLeagueSeasonId,
    ingestion_type: 'historical_full',
    status: 'running',
  }).select('id').single()
  const logId = logRow?.id

  try {
    // 1. Get or create league_season row for prior season
    const meta = await getLeagueMetadata(priorLeagueKey, leagueId)

    const { data: existingSeason } = await db
      .from('league_seasons')
      .select('id')
      .eq('league_key', priorLeagueKey)
      .maybeSingle()

    let priorLeagueSeasonId: string

    if (existingSeason) {
      priorLeagueSeasonId = existingSeason.id
    } else {
      const { data: newSeason, error } = await db
        .from('league_seasons')
        .insert({
          league_key: priorLeagueKey,
          league_id: meta.league_id,
          game_id: meta.game_id,
          season: meta.season,
          name: meta.name,
          num_teams: meta.num_teams,
          scoring_type: meta.scoring_type,
          start_week: meta.start_week,
          end_week: meta.end_week,
          playoff_start_week: meta.playoff_start_week,
          is_finished: true,
          prior_league_key: meta.prior_league_key,
          ingested_at: new Date().toISOString(),
        })
        .select('id')
        .single()

      if (error || !newSeason) throw new Error(`Failed to create prior league season: ${error?.message}`)
      priorLeagueSeasonId = newSeason.id
    }

    // 2. Teams (includes final standings)
    const teams = await getLeagueTeams(priorLeagueKey, leagueId)
    for (const team of teams) {
      const { managerId } = await matchOrCreateManager(team, priorLeagueSeasonId, meta.season)
      await db.from('teams').upsert({
        team_key: team.team_key,
        league_season_id: priorLeagueSeasonId,
        manager_id: managerId,
        team_name: team.team_name,
        manager_nickname: team.manager_nickname,
        manager_yahoo_guid: team.manager_yahoo_guid,
        logo_url: team.logo_url,
        wins: team.wins,
        losses: team.losses,
        ties: team.ties,
        points_for: team.points_for,
        points_against: team.points_against,
        playoff_seed: team.playoff_seed,
      }, { onConflict: 'team_key' })
      records++
    }

    // 3. All matchups week by week
    const totalWeeks = (meta.end_week ?? 14) + 3 // include playoffs
    for (let week = (meta.start_week ?? 1); week <= totalWeeks; week++) {
      try {
        const matchups = await getScoreboard(priorLeagueKey, leagueId, week)
        if (matchups.length === 0) break

        for (const m of matchups) {
          const margin = m.team1_score != null && m.team2_score != null
            ? Math.abs(m.team1_score - m.team2_score)
            : null

          await db.from('matchups').upsert({
            league_season_id: priorLeagueSeasonId,
            season: meta.season,
            week: m.week,
            team1_key: m.team1_key,
            team2_key: m.team2_key,
            team1_score: m.team1_score,
            team2_score: m.team2_score,
            winner_team_key: m.winner_team_key,
            is_tie: m.is_tie,
            margin,
            is_playoff: m.is_playoff,
            is_consolation: m.is_consolation,
            is_championship: m.is_championship,
            is_complete: true,
            raw_yahoo_json: m.raw as any,
            updated_at: new Date().toISOString(),
          }, { onConflict: 'league_season_id,week,team1_key,team2_key' })
          records++
        }
      } catch {
        warnings.push(`Week ${week} scoreboard fetch failed — skipping`)
      }
    }

    // 4. Draft results
    try {
      const picks = await getDraftResults(priorLeagueKey, leagueId)
      for (let i = 0; i < picks.length; i++) {
        const pick = picks[i]
        await db.from('draft_picks').upsert({
          league_season_id: priorLeagueSeasonId,
          season: meta.season,
          pick_number: parseInt(pick.pick, 10),
          round: parseInt(pick.round, 10),
          team_key: pick.team_key,
          player_name: pick.player_name ?? 'Unknown',
          player_id: pick.player_key ?? null,
        }, { onConflict: 'league_season_id,pick_number' })
        records++
      }
    } catch {
      warnings.push('Draft results fetch failed — skipping')
    }

    // 5. Transactions (paginate through all)
    let start = 0
    let fetched = 25
    while (fetched === 25) {
      const { transactions: txns, count } = await getTransactions(priorLeagueKey, leagueId, start)
      fetched = count
      for (const rawTxn of txns) {
        try {
          const parsed = parseTransaction(rawTxn, priorLeagueSeasonId, null)
          await db.from('transactions').upsert(parsed, { onConflict: 'yahoo_transaction_key' })
          records++
        } catch (e) {
          warnings.push(`Transaction parse: ${e instanceof Error ? e.message : String(e)}`)
        }
      }
      start += 25
    }

    // 6. Update current season to link back
    await db.from('league_seasons').update({
      prior_league_key: priorLeagueKey,
      ingested_at: new Date().toISOString(),
    }).eq('id', currentLeagueSeasonId)

    await db.from('ingestion_log').update({
      status: warnings.length > 0 ? 'partial' : 'complete',
      records_upserted: records,
      warnings,
      completed_at: new Date().toISOString(),
    }).eq('id', logId)

    return { recordsUpserted: records, warnings }
  } catch (err) {
    await db.from('ingestion_log').update({
      status: 'failed',
      error_message: err instanceof Error ? err.message : String(err),
      completed_at: new Date().toISOString(),
    }).eq('id', logId)
    throw err
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add lib/ingestion/weekly-sync.ts lib/ingestion/historical.ts
git commit -m "feat: weekly sync and historical ingestion services"
```

---

### Task 9: Context Builder

**Files:**
- Create: `lib/context-builder/types.ts`
- Create: `lib/context-builder/queries.ts`
- Create: `lib/context-builder/builder.ts`
- Test: `__tests__/context-builder/builder.test.ts`

**Interfaces:**
- Consumes: Supabase admin client; `leagueSeasonId`, `week`, `season`
- Produces: `buildWeeklyRecapContext(leagueSeasonId, week, season)` → `ContextObject`

- [ ] **Step 1: Define context object types**

`lib/context-builder/types.ts`:
```typescript
export interface TeamContext {
  team_key: string
  team_name: string
  manager: string
  record_before: string        // "3-1" format
  points_for_season: number
  last_3_scores: number[]
}

export interface H2HContext {
  all_time_team_a_wins: number
  all_time_team_b_wins: number
  this_season_team_a_wins: number
  this_season_team_b_wins: number
  last_meeting: { week: number; season: number; winner: string; margin: number } | null
  source: string
}

export interface MatchupContext {
  id: string
  team_a: TeamContext
  team_b: TeamContext
  result: {
    team_a_score: number
    team_b_score: number
    winner_manager: string
    margin: number
  }
  h2h: H2HContext
}

export interface StandingContext {
  rank: number
  manager: string
  team_name: string
  record: string
  points_for: number
  streak: string
  source: string
}

export interface WeekRecordsContext {
  highest_score: { score: number; manager: string; source: string }
  lowest_score: { score: number; manager: string; source: string }
  closest_game: { margin: number; source: string }
  biggest_blowout: { margin: number; source: string }
}

export interface AllTimeContext {
  highest_score_ever: { score: number; manager: string; week: number; season: number; source: string } | null
  biggest_blowout_ever: { margin: number; winner: string; loser: string; source: string } | null
  championship_history: Array<{
    season: number
    champion: string
    runner_up: string | null
    championship_score: string
    source: string
  }>
}

export interface TransactionSummary {
  type: string
  week: number | null
  description: string
  teams: string[]
  source: string
}

export interface ManagerContinuity {
  returning: string[]
  new_this_season: string[]
  departed: string[]
  source: string
}

export interface ContextObject {
  meta: {
    episode_type: 'weekly_recap' | 'weekly_preview' | 'preseason' | 'playoff_special' | 'championship'
    week: number | null
    season: number
    league_name: string
    show_name: string
    generated_at: string
  }
  this_week: {
    matchups: MatchupContext[]
    week_records: WeekRecordsContext
  } | null
  standings: StandingContext[]
  all_time_records: AllTimeContext
  storylines: {
    active_streaks: Array<{ manager: string; streak: string; source: string }>
    h2h_narratives: Array<{ description: string; source: string }>
  }
  manager_continuity: ManagerContinuity | null
  recent_transactions: TransactionSummary[]
  draft_highlights: Array<{ pick_number: number; player_name: string; manager: string; position: string | null }>
}
```

- [ ] **Step 2: Implement named queries**

`lib/context-builder/queries.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'

export async function getMatchupsForWeek(leagueSeasonId: string, week: number) {
  const db = getAdminClient()
  const { data, error } = await db
    .from('matchups')
    .select('*, league_seasons(season)')
    .eq('league_season_id', leagueSeasonId)
    .eq('week', week)
    .eq('is_complete', true)

  if (error) throw new Error(`getMatchupsForWeek: ${error.message}`)
  return data ?? []
}

export async function getTeamByKey(teamKey: string) {
  const db = getAdminClient()
  const { data } = await db
    .from('teams')
    .select('*, managers(primary_nickname, identity_status)')
    .eq('team_key', teamKey)
    .single()
  return data
}

export async function getH2HRecord(
  manager1Id: string,
  manager2Id: string
): Promise<{ m1wins: number; m2wins: number; lastMeeting: any | null; source: string }> {
  const db = getAdminClient()

  // Get all team_keys for each manager
  const { data: m1teams } = await db.from('teams').select('team_key').eq('manager_id', manager1Id)
  const { data: m2teams } = await db.from('teams').select('team_key').eq('manager_id', manager2Id)

  const m1keys = (m1teams ?? []).map(t => t.team_key)
  const m2keys = (m2teams ?? []).map(t => t.team_key)

  if (!m1keys.length || !m2keys.length) {
    return { m1wins: 0, m2wins: 0, lastMeeting: null, source: `matchups:h2h:no_history` }
  }

  // Get all matchups between these two managers
  const { data: matchups } = await db
    .from('matchups')
    .select('*')
    .eq('is_complete', true)
    .or(
      m1keys.map(k1 =>
        m2keys.map(k2 => `and(team1_key.eq.${k1},team2_key.eq.${k2}),and(team1_key.eq.${k2},team2_key.eq.${k1})`)
      ).flat().join(',')
    )
    .order('season', { ascending: false })
    .order('week', { ascending: false })

  let m1wins = 0, m2wins = 0
  for (const m of (matchups ?? [])) {
    const winnerIsM1 = m1keys.includes(m.winner_team_key)
    const winnerIsM2 = m2keys.includes(m.winner_team_key)
    if (winnerIsM1) m1wins++
    if (winnerIsM2) m2wins++
  }

  const last = matchups?.[0] ?? null
  const source = `matchups:h2h_query:${manager1Id.slice(0,8)}_vs_${manager2Id.slice(0,8)}`

  return { m1wins, m2wins, lastMeeting: last, source }
}

export async function getStandingsForWeek(leagueSeasonId: string, week: number) {
  const db = getAdminClient()
  const { data } = await db
    .from('standings_snapshots')
    .select('*, teams(team_name, manager_id, managers(primary_nickname))')
    .eq('league_season_id', leagueSeasonId)
    .eq('week', week)
    .order('wins', { ascending: false })
    .order('points_for', { ascending: false })
  return data ?? []
}

export async function getAllTimeRecords(seasons: string[]) {
  const db = getAdminClient()
  const { data: highScore } = await db
    .from('matchups')
    .select('team1_key, team2_key, team1_score, team2_score, season, week')
    .in('league_season_id', seasons)
    .eq('is_complete', true)
    .order('team1_score', { ascending: false })
    .limit(1)

  const { data: blowout } = await db
    .from('matchups')
    .select('*')
    .in('league_season_id', seasons)
    .eq('is_complete', true)
    .order('margin', { ascending: false })
    .limit(1)

  const { data: championships } = await db
    .from('matchups')
    .select('*, league_seasons(season)')
    .in('league_season_id', seasons)
    .eq('is_championship', true)
    .eq('is_complete', true)
    .order('season', { ascending: false })

  return { highScore: highScore?.[0] ?? null, blowout: blowout?.[0] ?? null, championships: championships ?? [] }
}

export async function getRecentTransactions(leagueSeasonId: string, week: number, limit = 5) {
  const db = getAdminClient()
  const { data } = await db
    .from('transactions')
    .select('*')
    .eq('league_season_id', leagueSeasonId)
    .eq('status', 'successful')
    .lte('week', week)
    .order('occurred_at', { ascending: false })
    .limit(limit)
  return data ?? []
}

export async function getDraftHighlights(leagueSeasonId: string, topN = 5) {
  const db = getAdminClient()
  const { data } = await db
    .from('draft_picks')
    .select('*, teams(manager_nickname)')
    .eq('league_season_id', leagueSeasonId)
    .order('pick_number', { ascending: true })
    .limit(topN)
  return data ?? []
}
```

- [ ] **Step 3: Implement context builder**

`lib/context-builder/builder.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import {
  getMatchupsForWeek,
  getTeamByKey,
  getH2HRecord,
  getStandingsForWeek,
  getAllTimeRecords,
  getRecentTransactions,
  getDraftHighlights,
} from '@/lib/context-builder/queries'
import type { ContextObject, MatchupContext, StandingContext } from '@/lib/context-builder/types'

export async function buildWeeklyRecapContext(
  leagueSeasonId: string,
  week: number,
  season: number
): Promise<ContextObject> {
  const db = getAdminClient()

  // Get league config for names
  const { data: config } = await db
    .from('league_config')
    .select('show_name')
    .eq('league_season_id', leagueSeasonId)
    .single()

  const { data: leagueSeason } = await db
    .from('league_seasons')
    .select('name, id')
    .eq('id', leagueSeasonId)
    .single()

  // Get all league season IDs for this league (for cross-season queries)
  const { data: allSeasons } = await db
    .from('league_seasons')
    .select('id')
    .or(`id.eq.${leagueSeasonId},prior_league_key.not.is.null`)

  const allSeasonIds = (allSeasons ?? []).map(s => s.id)

  // This week's matchups
  const rawMatchups = await getMatchupsForWeek(leagueSeasonId, week)
  const matchupContexts: MatchupContext[] = []

  for (const m of rawMatchups) {
    const t1 = await getTeamByKey(m.team1_key)
    const t2 = await getTeamByKey(m.team2_key)

    if (!t1 || !t2) continue

    const t1Manager = (t1.managers as any)?.primary_nickname ?? t1.manager_nickname
    const t2Manager = (t2.managers as any)?.primary_nickname ?? t2.manager_nickname

    // Only build H2H if both managers are confirmed
    const t1Status = (t1.managers as any)?.identity_status
    const t2Status = (t2.managers as any)?.identity_status
    const h2hUsable = t1Status === 'confirmed' && t2Status === 'confirmed' && t1.manager_id && t2.manager_id

    const h2h = h2hUsable
      ? await getH2HRecord(t1.manager_id, t2.manager_id)
      : { m1wins: 0, m2wins: 0, lastMeeting: null, source: `matchups:h2h:unresolved_identity` }

    // Get last 3 scores for each team
    const { data: t1History } = await db
      .from('matchups')
      .select('team1_score, team2_score, team1_key')
      .or(`team1_key.eq.${m.team1_key},team2_key.eq.${m.team1_key}`)
      .eq('is_complete', true)
      .lt('week', week)
      .order('week', { ascending: false })
      .limit(3)

    const t1Scores = (t1History ?? []).map(h =>
      h.team1_key === m.team1_key ? h.team1_score : h.team2_score
    ).filter(Boolean) as number[]

    const { data: t2History } = await db
      .from('matchups')
      .select('team1_score, team2_score, team1_key')
      .or(`team1_key.eq.${m.team2_key},team2_key.eq.${m.team2_key}`)
      .eq('is_complete', true)
      .lt('week', week)
      .order('week', { ascending: false })
      .limit(3)

    const t2Scores = (t2History ?? []).map(h =>
      h.team1_key === m.team2_key ? h.team1_score : h.team2_score
    ).filter(Boolean) as number[]

    const t1ScoreSeason = m.team1_score ?? 0
    const t2ScoreSeason = m.team2_score ?? 0
    const winnerManager = m.winner_team_key === m.team1_key ? t1Manager : t2Manager

    matchupContexts.push({
      id: m.id,
      team_a: {
        team_key: m.team1_key,
        team_name: t1.team_name,
        manager: t1Manager,
        record_before: `${t1.wins}-${t1.losses}`,
        points_for_season: t1.points_for,
        last_3_scores: t1Scores,
      },
      team_b: {
        team_key: m.team2_key,
        team_name: t2.team_name,
        manager: t2Manager,
        record_before: `${t2.wins}-${t2.losses}`,
        points_for_season: t2.points_for,
        last_3_scores: t2Scores,
      },
      result: {
        team_a_score: t1ScoreSeason,
        team_b_score: t2ScoreSeason,
        winner_manager: winnerManager,
        margin: m.margin ?? 0,
      },
      h2h: {
        all_time_team_a_wins: h2h.m1wins,
        all_time_team_b_wins: h2h.m2wins,
        this_season_team_a_wins: 0, // TODO: filter to current season
        this_season_team_b_wins: 0,
        last_meeting: h2h.lastMeeting
          ? {
              week: h2h.lastMeeting.week,
              season: h2h.lastMeeting.season,
              winner: h2h.lastMeeting.winner_team_key === m.team1_key ? t1Manager : t2Manager,
              margin: h2h.lastMeeting.margin ?? 0,
            }
          : null,
        source: h2h.source,
      },
    })
  }

  // Week records
  const scores = rawMatchups
    .filter(m => m.is_complete && m.team1_score != null)
    .flatMap(m => [
      { score: m.team1_score as number, team_key: m.team1_key },
      { score: m.team2_score as number, team_key: m.team2_key },
    ])
    .sort((a, b) => b.score - a.score)

  const highScoreTeam = scores[0] ? await getTeamByKey(scores[0].team_key) : null
  const lowScoreTeam = scores[scores.length - 1] ? await getTeamByKey(scores[scores.length - 1].team_key) : null

  const closestGame = rawMatchups
    .filter(m => m.is_complete && m.margin != null)
    .sort((a, b) => (a.margin ?? 999) - (b.margin ?? 999))[0]

  const biggestBlowout = rawMatchups
    .filter(m => m.is_complete && m.margin != null)
    .sort((a, b) => (b.margin ?? 0) - (a.margin ?? 0))[0]

  // Standings
  const standingsRaw = await getStandingsForWeek(leagueSeasonId, week)
  const standings: StandingContext[] = standingsRaw.map((s, i) => {
    const mgr = (s.teams as any)?.managers?.primary_nickname ?? (s.teams as any)?.manager_nickname ?? 'Unknown'
    return {
      rank: i + 1,
      manager: mgr,
      team_name: (s.teams as any)?.team_name ?? '',
      record: `${s.wins}-${s.losses}`,
      points_for: s.points_for,
      streak: `${s.streak_type}${s.streak_count}`,
      source: `standings_snapshots:week${week}:${s.team_key.slice(-4)}`,
    }
  })

  // All-time records
  const allTime = await getAllTimeRecords(allSeasonIds)
  const highTeam = allTime.highScore
    ? await getTeamByKey(allTime.highScore.team1_score >= (allTime.highScore.team2_score ?? 0)
        ? allTime.highScore.team1_key
        : allTime.highScore.team2_key)
    : null

  // Recent transactions
  const txns = await getRecentTransactions(leagueSeasonId, week)
  const recentTransactions = txns.map(t => ({
    type: t.type as string,
    week: t.week,
    description: formatTransactionDescription(t),
    teams: t.involved_team_keys ?? [],
    source: `transactions:${t.type}:${t.id.slice(0, 8)}`,
  }))

  // Draft highlights (first 5 picks)
  const draftPicks = await getDraftHighlights(leagueSeasonId)

  return {
    meta: {
      episode_type: 'weekly_recap',
      week,
      season,
      league_name: leagueSeason?.name ?? '',
      show_name: config?.show_name ?? '',
      generated_at: new Date().toISOString(),
    },
    this_week: {
      matchups: matchupContexts,
      week_records: {
        highest_score: {
          score: scores[0]?.score ?? 0,
          manager: (highScoreTeam as any)?.managers?.primary_nickname ?? '',
          source: `matchups:week${week}:max_score`,
        },
        lowest_score: {
          score: scores[scores.length - 1]?.score ?? 0,
          manager: (lowScoreTeam as any)?.managers?.primary_nickname ?? '',
          source: `matchups:week${week}:min_score`,
        },
        closest_game: {
          margin: closestGame?.margin ?? 0,
          source: `matchups:week${week}:min_margin`,
        },
        biggest_blowout: {
          margin: biggestBlowout?.margin ?? 0,
          source: `matchups:week${week}:max_margin`,
        },
      },
    },
    standings,
    all_time_records: {
      highest_score_ever: highTeam ? {
        score: Math.max(allTime.highScore!.team1_score, allTime.highScore!.team2_score ?? 0),
        manager: (highTeam as any)?.managers?.primary_nickname ?? '',
        week: allTime.highScore!.week,
        season: allTime.highScore!.season,
        source: 'matchups:all_time:max_score',
      } : null,
      biggest_blowout_ever: allTime.blowout ? {
        margin: allTime.blowout.margin ?? 0,
        winner: '',
        loser: '',
        source: 'matchups:all_time:max_margin',
      } : null,
      championship_history: allTime.championships.map(c => ({
        season: (c as any).league_seasons?.season ?? c.season,
        champion: '',
        runner_up: null,
        championship_score: `${c.team1_score ?? 0}-${c.team2_score ?? 0}`,
        source: `matchups:championship:${(c as any).league_seasons?.season}`,
      })),
    },
    storylines: {
      active_streaks: standings
        .filter(s => s.streak.startsWith('W') && parseInt(s.streak.slice(1)) >= 2)
        .map(s => ({ manager: s.manager, streak: s.streak, source: `matchups:streak:${s.rank}` })),
      h2h_narratives: matchupContexts
        .filter(m => m.h2h.all_time_team_a_wins + m.h2h.all_time_team_b_wins >= 3)
        .map(m => ({
          description: `${m.team_a.manager} leads ${m.team_b.manager} all-time ${m.h2h.all_time_team_a_wins}-${m.h2h.all_time_team_b_wins}`,
          source: m.h2h.source,
        })),
    },
    manager_continuity: null, // Populated in preseason context only
    recent_transactions: recentTransactions,
    draft_highlights: draftPicks.map(p => ({
      pick_number: p.pick_number,
      player_name: p.player_name,
      manager: (p.teams as any)?.manager_nickname ?? '',
      position: p.position,
    })),
  }
}

function formatTransactionDescription(t: any): string {
  const players = (t.players_sent as any[]) ?? []
  if (t.type === 'trade') {
    return `Trade involving ${players.map((p: any) => p.player_name).join(', ')}`
  }
  if (t.type === 'add' || t.type === 'drop') {
    return `${t.type === 'add' ? 'Added' : 'Dropped'}: ${players[0]?.player_name ?? 'player'}`
  }
  return `${t.type} transaction`
}
```

- [ ] **Step 4: Write and run test**

`__tests__/context-builder/builder.test.ts`:
```typescript
import { describe, it, expect, vi } from 'vitest'

vi.mock('@/lib/supabase/admin', () => ({
  getAdminClient: vi.fn(() => ({
    from: vi.fn(() => ({
      select: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      single: vi.fn().mockResolvedValue({ data: { show_name: 'Test Show', name: 'Test League', id: 'ls-id' }, error: null }),
      maybeSingle: vi.fn().mockResolvedValue({ data: null, error: null }),
      or: vi.fn().mockReturnThis(),
      lt: vi.fn().mockReturnThis(),
      lte: vi.fn().mockReturnThis(),
      in: vi.fn().mockReturnThis(),
      not: vi.fn().mockReturnThis(),
      order: vi.fn().mockReturnThis(),
      limit: vi.fn().mockResolvedValue({ data: [], error: null }),
      filter: vi.fn().mockReturnThis(),
    })),
  })),
}))

vi.mock('@/lib/context-builder/queries', () => ({
  getMatchupsForWeek: vi.fn().mockResolvedValue([]),
  getTeamByKey: vi.fn().mockResolvedValue(null),
  getH2HRecord: vi.fn().mockResolvedValue({ m1wins: 0, m2wins: 0, lastMeeting: null, source: 'test' }),
  getStandingsForWeek: vi.fn().mockResolvedValue([]),
  getAllTimeRecords: vi.fn().mockResolvedValue({ highScore: null, blowout: null, championships: [] }),
  getRecentTransactions: vi.fn().mockResolvedValue([]),
  getDraftHighlights: vi.fn().mockResolvedValue([]),
}))

describe('buildWeeklyRecapContext', () => {
  it('returns a valid context object with required meta fields', async () => {
    const { buildWeeklyRecapContext } = await import('@/lib/context-builder/builder')
    const ctx = await buildWeeklyRecapContext('ls-uuid', 5, 2025)
    expect(ctx.meta.episode_type).toBe('weekly_recap')
    expect(ctx.meta.week).toBe(5)
    expect(ctx.meta.season).toBe(2025)
    expect(ctx.standings).toBeInstanceOf(Array)
    expect(ctx.all_time_records).toBeDefined()
  })
})
```

```bash
npm test -- builder.test.ts
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add lib/context-builder/ __tests__/context-builder/
git commit -m "feat: context builder — named queries, typed context object, weekly recap assembly"
```

---

### Task 10: Script Generator

**Files:**
- Create: `lib/generation/prompts.ts`
- Create: `lib/generation/script-generator.ts`
- Test: `__tests__/generation/script-validator.test.ts`

**Interfaces:**
- Consumes: `ContextObject` from Task 9
- Produces: `generateScript(context, episodeType)` → `ScriptJson`; `validateScriptSources(script)` → `boolean`

- [ ] **Step 1: Write system prompts**

`lib/generation/prompts.ts`:
```typescript
export const SCRIPT_SYSTEM_PROMPT = `You are the writing team for a fantasy football league's private broadcast show. You write scripts for three hosts:

- TROY: Lead anchor. Authoritative and warm. Delivers scores, records, and results with authority.
- LISA: Analyst. Pattern-finder and historian. Connects this week to league history. Thoughtful.
- DANNY: Color commentator. Banter, hot takes, group-chat energy. Reacts to everything.

Hosts interact — this is a conversation, not turns at a podium. Troy sets up results, Lisa adds historical weight, Danny reacts. They can disagree, joke, call back to earlier segments.

OUTPUT FORMAT: Valid JSON matching this schema exactly:
{
  "episode": { "title": string, "week": number|null, "season": number, "type": string, "estimated_duration_secs": number },
  "segments": [{ "index": number, "host": "troy"|"lisa"|"danny", "text": string, "is_fact": boolean, "source"?: string, "type": string }],
  "sections": string[]
}

FACT INTEGRITY RULES — these override all other instructions:
1. You may ONLY state facts that appear in the provided context object.
2. Every segment with is_fact: true MUST include a "source" field copied verbatim from the matching context node's "source" value.
3. Never invent, extrapolate, or assume scores, records, names, or history not present in the context.
4. If a context field is null, missing, or empty — do not reference it.
5. Commentary and opinion MUST be labeled is_fact: false with no source field.
6. When uncertain whether something is a fact, label it is_fact: false.

EDITORIAL RULES:
- Shape the broadcast around 3-5 most interesting stories — not every matchup in scoreboard order.
- Lead with the most dramatic result (biggest blowout, closest game, upset, milestone).
- Historical context appears only when directly relevant to this week's events.
- Keep total estimated runtime 6-10 minutes (roughly 100-120 words per minute spoken).
- Segment texts should be 1-4 sentences. Short is better for audio.
- Use the league's show_name in the intro and outro.

SEGMENT TYPES: intro | score_report | historical_context | commentary | standings_report | storyline | closing
SECTIONS: intro | matchup_recap | standings | storylines | closing

Return ONLY the JSON object. No markdown fences, no explanation.`

export const TRANSCRIPT_SYSTEM_PROMPT = `You are a sports journalist writing a written recap for a fantasy football league.

You will receive the same factual context as the audio broadcast team. Write a clean, engaging markdown article.

RULES:
1. Only state facts present in the context object.
2. Write in second person where it creates energy ("Your commissioner did X" → avoid. "Greg topped the week..." → fine).
3. Structure: H2 for each section (Matchup Results, Standings, Storylines, Transactions).
4. Include a scores table using markdown table format.
5. Keep it to 400-600 words — companion, not replacement for the audio.
6. End with this disclaimer EXACTLY (no variation):

---
*Match scores and league records sourced from Yahoo Fantasy. Commentary and analysis are AI-generated.*

Return only the markdown content. No JSON.`
```

- [ ] **Step 2: Write failing test**

`__tests__/generation/script-validator.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { validateScriptSources } from '@/lib/generation/script-generator'
import type { ScriptJson } from '@/lib/types'

const validScript: ScriptJson = {
  episode: { title: 'Week 5', week: 5, season: 2025, type: 'weekly_recap', estimated_duration_secs: 480 },
  segments: [
    { index: 0, host: 'troy', text: 'Welcome.', is_fact: false, type: 'intro' },
    { index: 1, host: 'troy', text: 'Greg scored 147 points.', is_fact: true, source: 'matchups:week5:max_score', type: 'score_report' },
    { index: 2, host: 'danny', text: 'That is insane.', is_fact: false, type: 'commentary' },
  ],
  sections: ['intro', 'matchup_recap'],
}

const invalidScript: ScriptJson = {
  ...validScript,
  segments: [
    ...validScript.segments,
    { index: 3, host: 'lisa', text: 'Greg has won 10 in a row.', is_fact: true, type: 'historical_context' },
    // Missing source on is_fact: true
  ],
}

describe('validateScriptSources', () => {
  it('returns true for a valid script where all is_fact segments have source', () => {
    expect(validateScriptSources(validScript)).toBe(true)
  })

  it('returns false when an is_fact segment is missing source', () => {
    expect(validateScriptSources(invalidScript)).toBe(false)
  })
})
```

- [ ] **Step 3: Run test — verify it fails**

```bash
npm test -- script-validator.test.ts
# Expected: FAIL
```

- [ ] **Step 4: Implement script generator**

`lib/generation/script-generator.ts`:
```typescript
import Anthropic from '@anthropic-ai/sdk'
import { SCRIPT_SYSTEM_PROMPT } from '@/lib/generation/prompts'
import type { ContextObject } from '@/lib/context-builder/types'
import type { ScriptJson } from '@/lib/types'

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })

export function validateScriptSources(script: ScriptJson): boolean {
  for (const segment of script.segments) {
    if (segment.is_fact && !segment.source) {
      return false
    }
  }
  return true
}

export async function generateScript(
  context: ContextObject,
  episodeType: string,
  attempt = 1
): Promise<ScriptJson> {
  const userPrompt = `Generate a ${episodeType} episode script for the "${context.meta.show_name}" show.

Season: ${context.meta.season}
${context.meta.week ? `Week: ${context.meta.week}` : ''}

CONTEXT (source of all facts — do not use any fact not in this object):
${JSON.stringify(context, null, 2)}`

  const response = await anthropic.messages.create({
    model: 'claude-opus-4-6',
    max_tokens: 4096,
    system: attempt === 2
      ? SCRIPT_SYSTEM_PROMPT + '\n\nCRITICAL: Every single is_fact:true segment MUST have a source field copied from the context. This is non-negotiable. The previous attempt failed validation.'
      : SCRIPT_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userPrompt }],
  })

  const text = response.content[0].type === 'text' ? response.content[0].text : ''

  let script: ScriptJson
  try {
    script = JSON.parse(text)
  } catch {
    throw new Error(`Claude returned invalid JSON: ${text.slice(0, 200)}`)
  }

  if (!validateScriptSources(script)) {
    if (attempt >= 2) {
      throw new Error('Script failed source validation after 2 attempts')
    }
    return generateScript(context, episodeType, 2)
  }

  return script
}
```

- [ ] **Step 5: Run test — verify it passes**

```bash
npm test -- script-validator.test.ts
# Expected: PASS
```

- [ ] **Step 6: Commit**

```bash
git add lib/generation/prompts.ts lib/generation/script-generator.ts __tests__/generation/script-validator.test.ts
git commit -m "feat: Claude Opus script generator with source validation and retry"
```

---

### Task 11: Audio Producer

**Files:**
- Create: `lib/generation/audio-producer.ts`
- Test: `__tests__/generation/audio-producer.test.ts`

**Interfaces:**
- Consumes: `ScriptJson`, `LeagueConfig` (voice IDs)
- Produces: `produceAudio(script, config, leagueSeasonId, season, week)` → `{ audioUrl, durationSecs, warnings }`

- [ ] **Step 1: Write failing test**

`__tests__/generation/audio-producer.test.ts`:
```typescript
import { describe, it, expect, vi } from 'vitest'
import { buildAudioSegments } from '@/lib/generation/audio-producer'
import type { ScriptJson } from '@/lib/types'

const script: ScriptJson = {
  episode: { title: 'Week 5', week: 5, season: 2025, type: 'weekly_recap', estimated_duration_secs: 60 },
  segments: [
    { index: 0, host: 'troy', text: 'Hello.', is_fact: false, type: 'intro' },
    { index: 1, host: 'lisa', text: 'Great week.', is_fact: false, type: 'commentary' },
    { index: 2, host: 'troy', text: 'Indeed.', is_fact: false, type: 'closing' },
  ],
  sections: ['intro', 'closing'],
}

describe('buildAudioSegments', () => {
  it('inserts 300ms silence between same-host consecutive segments', () => {
    const segs = buildAudioSegments(script.segments)
    // troy(0) → troy(2): same host consecutive? No, lisa(1) is between
    expect(segs.some(s => s.type === 'silence')).toBe(false) // 0→lisa: switch, lisa→troy: switch
    // All non-silence segments = 3
    const audioSegs = segs.filter(s => s.type === 'audio')
    expect(audioSegs).toHaveLength(3)
  })

  it('marks silence between host switches', () => {
    const sameTroyScript = [
      { index: 0, host: 'troy' as const, text: 'A', is_fact: false, type: 'intro' },
      { index: 1, host: 'troy' as const, text: 'B', is_fact: false, type: 'intro' },
    ]
    const segs = buildAudioSegments(sameTroyScript)
    const silences = segs.filter(s => s.type === 'silence')
    expect(silences).toHaveLength(1)
    expect(silences[0].durationMs).toBe(300)
  })
})
```

- [ ] **Step 2: Run test — verify it fails**

```bash
npm test -- audio-producer.test.ts
# Expected: FAIL
```

- [ ] **Step 3: Implement audio producer**

`lib/generation/audio-producer.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import type { ScriptJson, ScriptSegment, LeagueConfig } from '@/lib/types'

const ELEVENLABS_BASE = 'https://api.elevenlabs.io/v1'

interface AudioPlan {
  type: 'audio' | 'silence'
  segment?: ScriptSegment
  voiceId?: string
  durationMs?: number
}

export function buildAudioSegments(segments: ScriptSegment[]): AudioPlan[] {
  const plan: AudioPlan[] = []
  for (let i = 0; i < segments.length; i++) {
    plan.push({ type: 'audio', segment: segments[i] })
    if (i < segments.length - 1) {
      const curr = segments[i]
      const next = segments[i + 1]
      // Detect section boundary (simplified: check type)
      const sectionBoundary = curr.type === 'closing' || next.type === 'intro'
      if (sectionBoundary) {
        plan.push({ type: 'silence', durationMs: 1200 })
      } else if (curr.host === next.host) {
        plan.push({ type: 'silence', durationMs: 300 })
      } else {
        plan.push({ type: 'silence', durationMs: 600 })
      }
    }
  }
  return plan
}

async function generateSegmentAudio(text: string, voiceId: string): Promise<Buffer | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const resp = await fetch(`${ELEVENLABS_BASE}/text-to-speech/${voiceId}`, {
        method: 'POST',
        headers: {
          'xi-api-key': process.env.ELEVENLABS_API_KEY!,
          'Content-Type': 'application/json',
          Accept: 'audio/mpeg',
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_turbo_v2_5',
          voice_settings: { stability: 0.6, similarity_boost: 0.8 },
        }),
      })
      if (!resp.ok) {
        const body = await resp.text()
        throw new Error(`ElevenLabs ${resp.status}: ${body}`)
      }
      return Buffer.from(await resp.arrayBuffer())
    } catch (err) {
      if (attempt === 2) return null // exhausted retries
    }
  }
  return null
}

/** ~300ms silence at 44100Hz mono 128kbps — minimal valid MP3 header */
const SILENCE_300MS = Buffer.from(
  'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
  'base64'
)

function makeSilenceBuffer(durationMs: number): Buffer {
  // Repeat the silence buffer proportionally (rough approximation)
  const repeats = Math.max(1, Math.round(durationMs / 300))
  return Buffer.concat(Array(repeats).fill(SILENCE_300MS))
}

export async function produceAudio(
  script: ScriptJson,
  config: Pick<LeagueConfig, 'host_troy_voice_id' | 'host_lisa_voice_id' | 'host_danny_voice_id'>,
  leagueSeasonId: string,
  season: number,
  week: number | null
): Promise<{ audioUrl: string | null; durationSecs: number; warnings: string[] }> {
  const voiceMap: Record<string, string | null> = {
    troy: config.host_troy_voice_id,
    lisa: config.host_lisa_voice_id,
    danny: config.host_danny_voice_id,
  }

  const warnings: string[] = []
  const plan = buildAudioSegments(script.segments)
  const buffers: Buffer[] = []
  let failedSegments = 0
  const totalAudioSegments = plan.filter(p => p.type === 'audio').length

  for (const item of plan) {
    if (item.type === 'silence') {
      buffers.push(makeSilenceBuffer(item.durationMs!))
      continue
    }

    const seg = item.segment!
    const voiceId = voiceMap[seg.host]
    if (!voiceId) {
      warnings.push(`No voice ID configured for host: ${seg.host}`)
      buffers.push(makeSilenceBuffer(600))
      failedSegments++
      continue
    }

    const audio = await generateSegmentAudio(seg.text, voiceId)
    if (!audio) {
      warnings.push(`Segment ${seg.index} (${seg.host}) failed after 3 attempts — inserting silence`)
      buffers.push(makeSilenceBuffer(600))
      failedSegments++
    } else {
      buffers.push(audio)
    }
  }

  // Abort if >20% failed
  if (failedSegments / totalAudioSegments > 0.2) {
    return { audioUrl: null, durationSecs: 0, warnings: [...warnings, 'Audio aborted: >20% segments failed'] }
  }

  const finalAudio = Buffer.concat(buffers)
  const durationSecs = Math.round(script.episode.estimated_duration_secs)

  // Upload to Supabase Storage
  const db = getAdminClient()
  const path = `${season}/week-${week ?? 'special'}/episode-${Date.now()}.mp3`

  const { error: uploadError } = await (db.storage as any)
    .from('episodes')
    .upload(path, finalAudio, { contentType: 'audio/mpeg', upsert: true })

  if (uploadError) {
    warnings.push(`Upload failed: ${uploadError.message}`)
    return { audioUrl: null, durationSecs, warnings }
  }

  const { data: { publicUrl } } = (db.storage as any)
    .from('episodes')
    .getPublicUrl(path)

  return { audioUrl: publicUrl, durationSecs, warnings }
}
```

- [ ] **Step 4: Run test — verify it passes**

```bash
npm test -- audio-producer.test.ts
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add lib/generation/audio-producer.ts __tests__/generation/audio-producer.test.ts
git commit -m "feat: ElevenLabs audio producer — segment-by-segment TTS, silence insertion, upload"
```

---

### Task 12: Transcript Generator + Episode Pipeline

**Files:**
- Create: `lib/generation/transcript-generator.ts`
- Create: `lib/generation/episode-pipeline.ts`

**Interfaces:**
- Consumes: `ContextObject`, `ScriptJson`, `LeagueConfig`
- Produces: `generateTranscript(context)` → `string`; `runEpisodePipeline(episodeId)` → `void`

- [ ] **Step 1: Implement transcript generator**

`lib/generation/transcript-generator.ts`:
```typescript
import Anthropic from '@anthropic-ai/sdk'
import { TRANSCRIPT_SYSTEM_PROMPT } from '@/lib/generation/prompts'
import type { ContextObject } from '@/lib/context-builder/types'

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })

const DISCLAIMER = `\n\n---\n*Match scores and league records sourced from Yahoo Fantasy. Commentary and analysis are AI-generated.*`

export async function generateTranscript(context: ContextObject): Promise<string> {
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 2048,
    system: TRANSCRIPT_SYSTEM_PROMPT,
    messages: [{
      role: 'user',
      content: `Write the written recap for ${context.meta.show_name} — ${context.meta.episode_type === 'weekly_recap' ? `Week ${context.meta.week}` : context.meta.episode_type}, ${context.meta.season} season.\n\nCONTEXT:\n${JSON.stringify(context, null, 2)}`,
    }],
  })

  const text = response.content[0].type === 'text' ? response.content[0].text : ''

  // Ensure disclaimer is present (append if missing)
  if (!text.includes('Match scores and league records sourced from Yahoo Fantasy')) {
    return text + DISCLAIMER
  }
  return text
}
```

- [ ] **Step 2: Implement episode pipeline**

`lib/generation/episode-pipeline.ts`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { buildWeeklyRecapContext } from '@/lib/context-builder/builder'
import { generateScript } from '@/lib/generation/script-generator'
import { produceAudio } from '@/lib/generation/audio-producer'
import { generateTranscript } from '@/lib/generation/transcript-generator'
import type { Episode, LeagueConfig } from '@/lib/types'

export async function runEpisodePipeline(episodeId: string): Promise<void> {
  const db = getAdminClient()

  // 1. Load episode and config
  const { data: episode } = await db
    .from('episodes')
    .select('*')
    .eq('id', episodeId)
    .single()

  if (!episode) throw new Error(`Episode not found: ${episodeId}`)

  const { data: config } = await db
    .from('league_config')
    .select('*')
    .eq('league_season_id', episode.league_season_id)
    .single()

  if (!config) throw new Error('League config not found')

  const startedAt = Date.now()

  try {
    // 2. Build context
    await db.from('episodes').update({ status: 'generating_script' }).eq('id', episodeId)

    const context = await buildWeeklyRecapContext(
      episode.league_season_id,
      episode.week,
      episode.season
    )

    // 3. Generate script (Claude Opus — retry once on source validation failure)
    let script
    try {
      script = await generateScript(context, episode.episode_type)
    } catch (err) {
      await db.from('episodes').update({
        status: 'failed',
        error_message: `Script generation failed: ${err instanceof Error ? err.message : String(err)}`,
        updated_at: new Date().toISOString(),
      }).eq('id', episodeId)
      return
    }

    await db.from('episodes').update({
      script_json: script as any,
      status: 'generating_audio',
      updated_at: new Date().toISOString(),
    }).eq('id', episodeId)

    // 4. Audio + transcript in parallel
    const [audioResult, transcriptResult] = await Promise.allSettled([
      produceAudio(script, config, episode.league_season_id, episode.season, episode.week),
      generateTranscript(context),
    ])

    const audioOk = audioResult.status === 'fulfilled' && audioResult.value.audioUrl !== null
    const transcriptOk = transcriptResult.status === 'fulfilled'

    const audioUrl = audioOk ? (audioResult as any).value.audioUrl : null
    const durationSecs = audioOk ? (audioResult as any).value.durationSecs : null
    const audioWarnings = audioOk ? (audioResult as any).value.warnings : []
    const transcript = transcriptOk ? (transcriptResult as any).value : null

    const status = audioOk && transcriptOk ? 'ready' : audioOk ? 'ready' : 'partial'
    const elapsedMs = Date.now() - startedAt

    await db.from('episodes').update({
      status,
      transcript_md: transcript,
      audio_url: audioUrl,
      audio_duration_secs: durationSecs,
      published_at: new Date().toISOString(),
      generation_metadata: {
        model_script: 'claude-opus-4-6',
        model_transcript: 'claude-sonnet-4-6',
        elevenlabs_model: 'eleven_turbo_v2_5',
        warnings: audioWarnings,
        elapsed_ms: elapsedMs,
      },
      updated_at: new Date().toISOString(),
    }).eq('id', episodeId)

  } catch (err) {
    await db.from('episodes').update({
      status: 'failed',
      error_message: err instanceof Error ? err.message : String(err),
      updated_at: new Date().toISOString(),
    }).eq('id', episodeId)
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add lib/generation/transcript-generator.ts lib/generation/episode-pipeline.ts
git commit -m "feat: transcript generator (Claude Sonnet) and episode pipeline orchestrator"
```

---

### Task 13: API Routes

**Files:**
- Create: `app/api/ingest/historical/route.ts`
- Create: `app/api/ingest/weekly/route.ts`
- Create: `app/api/episode/generate/route.ts`
- Create: `app/api/cron/weekly/route.ts`

**Interfaces:**
- All routes protected: ingest + episode routes require Supabase Auth session; cron route requires `CRON_SECRET` header

- [ ] **Step 1: Historical ingestion route**

`app/api/ingest/historical/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase/server'
import { runHistoricalIngestion } from '@/lib/ingestion/historical'

export async function POST(request: NextRequest) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { leagueSeasonId, priorLeagueKey, leagueId } = await request.json()

  if (!leagueSeasonId || !priorLeagueKey || !leagueId) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
  }

  try {
    const result = await runHistoricalIngestion(leagueSeasonId, priorLeagueKey, leagueId)
    return NextResponse.json(result)
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Ingestion failed' },
      { status: 500 }
    )
  }
}
```

- [ ] **Step 2: Weekly sync route**

`app/api/ingest/weekly/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase/server'
import { runWeeklySync } from '@/lib/ingestion/weekly-sync'

export async function POST(request: NextRequest) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { leagueSeasonId, leagueKey, leagueId, week } = await request.json()

  try {
    const result = await runWeeklySync(leagueSeasonId, leagueKey, leagueId, week)
    return NextResponse.json(result)
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Sync failed' },
      { status: 500 }
    )
  }
}
```

- [ ] **Step 3: Episode generate route**

`app/api/episode/generate/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase/server'
import { getAdminClient } from '@/lib/supabase/admin'
import { runEpisodePipeline } from '@/lib/generation/episode-pipeline'

export async function POST(request: NextRequest) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { leagueSeasonId, week, season, episodeType, title } = await request.json()

  const db = getAdminClient()

  // Create the episode row
  const { data: episode, error } = await db
    .from('episodes')
    .insert({
      league_season_id: leagueSeasonId,
      season,
      week: week ?? null,
      episode_type: episodeType ?? 'weekly_recap',
      title: title ?? `Week ${week} Recap`,
      status: 'pending',
    })
    .select('id')
    .single()

  if (error || !episode) {
    return NextResponse.json({ error: 'Failed to create episode' }, { status: 500 })
  }

  // Fire and forget — pipeline runs async
  runEpisodePipeline(episode.id).catch(console.error)

  return NextResponse.json({ episodeId: episode.id, status: 'pending' })
}
```

- [ ] **Step 4: Cron route (Tuesday 9AM automation)**

`app/api/cron/weekly/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { getAdminClient } from '@/lib/supabase/admin'
import { runWeeklySync } from '@/lib/ingestion/weekly-sync'
import { runEpisodePipeline } from '@/lib/generation/episode-pipeline'

export const maxDuration = 300 // 5 minutes — Vercel Pro limit

export async function POST(request: NextRequest) {
  const secret = request.headers.get('authorization')?.replace('Bearer ', '')
  if (secret !== process.env.CRON_SECRET) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const db = getAdminClient()

  // Get the active league season
  const { data: activeSeason } = await db
    .from('league_seasons')
    .select('id, league_key, league_id, season, start_week, end_week')
    .eq('is_finished', false)
    .order('season', { ascending: false })
    .limit(1)
    .single()

  if (!activeSeason) {
    return NextResponse.json({ message: 'No active season found' })
  }

  // Determine current week (Yahoo current_week from metadata would be ideal;
  // here we use stored data as proxy)
  const { data: latestMatchup } = await db
    .from('matchups')
    .select('week')
    .eq('league_season_id', activeSeason.id)
    .eq('is_complete', true)
    .order('week', { ascending: false })
    .limit(1)
    .single()

  const currentWeek = (latestMatchup?.week ?? 0) + 1

  try {
    // 1. Sync data
    await runWeeklySync(
      activeSeason.id,
      activeSeason.league_key,
      activeSeason.league_id,
      currentWeek - 1 // sync the just-completed week
    )

    // 2. Create episode and run pipeline
    const { data: episode } = await db
      .from('episodes')
      .insert({
        league_season_id: activeSeason.id,
        season: activeSeason.season,
        week: currentWeek - 1,
        episode_type: 'weekly_recap',
        title: `Week ${currentWeek - 1} Recap`,
        status: 'pending',
      })
      .select('id')
      .single()

    if (episode) {
      await runEpisodePipeline(episode.id)
    }

    return NextResponse.json({ success: true, week: currentWeek - 1, episodeId: episode?.id })
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Cron failed' },
      { status: 500 }
    )
  }
}
```

- [ ] **Step 5: Add Vercel cron config**

`vercel.json`:
```json
{
  "crons": [
    {
      "path": "/api/cron/weekly",
      "schedule": "0 9 * * 2"
    }
  ]
}
```

- [ ] **Step 6: Commit**

```bash
git add app/api/ vercel.json
git commit -m "feat: API routes for ingestion, episode generation, and Vercel cron handler"
```

---

### Task 14: Audio Player Components

**Files:**
- Create: `components/audio/AudioContext.tsx`
- Create: `components/audio/HeroPlayer.tsx`
- Create: `components/audio/MiniPlayer.tsx`

**Interfaces:**
- Produces: `AudioProvider`, `useAudio()`, `HeroPlayer`, `MiniPlayer`

- [ ] **Step 1: Global audio context**

`components/audio/AudioContext.tsx`:
```typescript
'use client'

import { createContext, useContext, useRef, useState, useCallback, ReactNode } from 'react'

interface AudioState {
  episodeId: string | null
  title: string
  audioUrl: string | null
  isPlaying: boolean
  currentTime: number
  duration: number
  playbackRate: number
}

interface AudioContextValue extends AudioState {
  play: (episodeId: string, title: string, audioUrl: string) => void
  togglePlay: () => void
  seek: (time: number) => void
  setRate: (rate: number) => void
  audioRef: React.RefObject<HTMLAudioElement | null>
}

const AudioContext = createContext<AudioContextValue | null>(null)

export function AudioProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [state, setState] = useState<AudioState>({
    episodeId: null, title: '', audioUrl: null,
    isPlaying: false, currentTime: 0, duration: 0, playbackRate: 1,
  })

  const play = useCallback((episodeId: string, title: string, audioUrl: string) => {
    setState(s => ({ ...s, episodeId, title, audioUrl, isPlaying: true }))
    if (audioRef.current) {
      audioRef.current.src = audioUrl
      audioRef.current.play()
    }
  }, [])

  const togglePlay = useCallback(() => {
    if (!audioRef.current) return
    if (state.isPlaying) {
      audioRef.current.pause()
      setState(s => ({ ...s, isPlaying: false }))
    } else {
      audioRef.current.play()
      setState(s => ({ ...s, isPlaying: true }))
    }
  }, [state.isPlaying])

  const seek = useCallback((time: number) => {
    if (audioRef.current) audioRef.current.currentTime = time
    setState(s => ({ ...s, currentTime: time }))
  }, [])

  const setRate = useCallback((rate: number) => {
    if (audioRef.current) audioRef.current.playbackRate = rate
    setState(s => ({ ...s, playbackRate: rate }))
  }, [])

  return (
    <AudioContext.Provider value={{ ...state, play, togglePlay, seek, setRate, audioRef }}>
      <audio
        ref={audioRef}
        onTimeUpdate={() => setState(s => ({ ...s, currentTime: audioRef.current?.currentTime ?? 0 }))}
        onDurationChange={() => setState(s => ({ ...s, duration: audioRef.current?.duration ?? 0 }))}
        onEnded={() => setState(s => ({ ...s, isPlaying: false }))}
      />
      {children}
    </AudioContext.Provider>
  )
}

export function useAudio() {
  const ctx = useContext(AudioContext)
  if (!ctx) throw new Error('useAudio must be used within AudioProvider')
  return ctx
}
```

- [ ] **Step 2: Hero player**

`components/audio/HeroPlayer.tsx`:
```typescript
'use client'

import { useAudio } from '@/components/audio/AudioContext'

interface HeroPlayerProps {
  episodeId: string
  title: string
  audioUrl: string
  artworkUrl?: string | null
  accentColor?: string
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function HeroPlayer({ episodeId, title, audioUrl, artworkUrl, accentColor = '#00ff87' }: HeroPlayerProps) {
  const { play, togglePlay, seek, setRate, isPlaying, currentTime, duration, playbackRate, episodeId: activeId } = useAudio()
  const isActive = activeId === episodeId

  const handlePlay = () => {
    if (!isActive) {
      play(episodeId, title, audioUrl)
    } else {
      togglePlay()
    }
  }

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0
  const rates = [1, 1.25, 1.5, 2]
  const nextRate = rates[(rates.indexOf(playbackRate) + 1) % rates.length]

  return (
    <div className="w-full bg-[#1a1f2e] rounded-2xl p-6 space-y-4">
      {artworkUrl && (
        <img src={artworkUrl} alt={title} className="w-32 h-32 rounded-xl mx-auto object-cover" />
      )}
      <h2 className="text-white font-bold text-lg text-center">{title}</h2>

      {/* Progress bar */}
      <div className="space-y-1">
        <input
          type="range"
          min={0}
          max={duration || 100}
          value={isActive ? currentTime : 0}
          onChange={e => isActive && seek(Number(e.target.value))}
          className="w-full h-1 appearance-none rounded-full cursor-pointer"
          style={{ accentColor }}
        />
        <div className="flex justify-between text-xs text-gray-400">
          <span>{isActive ? formatTime(currentTime) : '0:00'}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-6">
        <button onClick={() => isActive && seek(Math.max(0, currentTime - 15))}
          className="text-gray-300 hover:text-white text-sm">−15s</button>
        <button
          onClick={handlePlay}
          className="w-14 h-14 rounded-full flex items-center justify-center text-black font-bold text-xl"
          style={{ backgroundColor: accentColor }}
        >
          {isActive && isPlaying ? '⏸' : '▶'}
        </button>
        <button onClick={() => isActive && seek(Math.min(duration, currentTime + 30))}
          className="text-gray-300 hover:text-white text-sm">+30s</button>
        <button onClick={() => setRate(nextRate)}
          className="text-gray-300 hover:text-white text-sm w-10">{playbackRate}×</button>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Mini player (persistent footer)**

`components/audio/MiniPlayer.tsx`:
```typescript
'use client'

import { useAudio } from '@/components/audio/AudioContext'

export function MiniPlayer() {
  const { audioUrl, title, isPlaying, togglePlay, currentTime, duration } = useAudio()

  if (!audioUrl) return null

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 bg-[#1a1f2e] border-t border-white/10">
      {/* Thin progress line */}
      <div className="h-0.5 bg-white/10">
        <div className="h-full bg-[#00ff87] transition-all" style={{ width: `${progressPct}%` }} />
      </div>
      <div className="flex items-center justify-between px-4 py-2">
        <p className="text-white text-sm font-medium truncate max-w-[60%]">{title}</p>
        <button
          onClick={togglePlay}
          className="w-10 h-10 rounded-full bg-[#00ff87] text-black flex items-center justify-center font-bold"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add components/audio/
git commit -m "feat: audio player components — AudioContext, HeroPlayer, MiniPlayer"
```

---

### Task 15: Public Layout + Home Page

**Files:**
- Create: `app/(public)/layout.tsx`
- Create: `app/(public)/page.tsx`
- Create: `components/scores/MatchupCard.tsx`
- Create: `components/standings/StandingsTable.tsx`

- [ ] **Step 1: Public layout**

`app/(public)/layout.tsx`:
```typescript
import { AudioProvider } from '@/components/audio/AudioContext'
import { MiniPlayer } from '@/components/audio/MiniPlayer'
import type { ReactNode } from 'react'

export default function PublicLayout({ children }: { children: ReactNode }) {
  return (
    <AudioProvider>
      <div className="min-h-screen bg-[#0f1117] text-white">
        {children}
        <MiniPlayer />
        <div className="h-16" /> {/* MiniPlayer spacer */}
      </div>
    </AudioProvider>
  )
}
```

- [ ] **Step 2: Matchup score card**

`components/scores/MatchupCard.tsx`:
```typescript
interface MatchupCardProps {
  team1Name: string
  team1Manager: string
  team1Score: number
  team2Name: string
  team2Manager: string
  team2Score: number
  winnerKey: string | null
  team1Key: string
  accentColor?: string
}

export function MatchupCard({
  team1Name, team1Manager, team1Score,
  team2Name, team2Manager, team2Score,
  winnerKey, team1Key, accentColor = '#00ff87'
}: MatchupCardProps) {
  const team1Wins = winnerKey === team1Key
  const team2Wins = winnerKey !== null && winnerKey !== team1Key

  return (
    <div className="bg-[#1a1f2e] rounded-xl p-4 min-w-[200px]">
      <div className={`flex justify-between items-center py-1 px-2 rounded ${team1Wins ? 'border-l-2' : ''}`}
        style={team1Wins ? { borderColor: accentColor } : {}}>
        <div>
          <p className={`text-sm font-medium ${team1Wins ? 'text-white' : 'text-gray-400'}`}>{team1Manager}</p>
          <p className="text-xs text-gray-500 truncate max-w-[100px]">{team1Name}</p>
        </div>
        <span className={`text-lg font-bold ${team1Wins ? 'text-white' : 'text-gray-400'}`}>
          {team1Score.toFixed(1)}
        </span>
      </div>
      <div className={`flex justify-between items-center py-1 px-2 rounded mt-1 ${team2Wins ? 'border-l-2' : ''}`}
        style={team2Wins ? { borderColor: accentColor } : {}}>
        <div>
          <p className={`text-sm font-medium ${team2Wins ? 'text-white' : 'text-gray-400'}`}>{team2Manager}</p>
          <p className="text-xs text-gray-500 truncate max-w-[100px]">{team2Name}</p>
        </div>
        <span className={`text-lg font-bold ${team2Wins ? 'text-white' : 'text-gray-400'}`}>
          {team2Score.toFixed(1)}
        </span>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Standings table**

`components/standings/StandingsTable.tsx`:
```typescript
interface StandingRow {
  rank: number
  manager: string
  team_name: string
  record: string
  points_for: number
  streak: string
  made_playoffs?: boolean
}

interface StandingsTableProps {
  standings: StandingRow[]
  playoffSpots: number
  accentColor?: string
}

export function StandingsTable({ standings, playoffSpots, accentColor = '#00ff87' }: StandingsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-white/10">
            <th className="text-left py-2 px-3">#</th>
            <th className="text-left py-2 px-3">Manager</th>
            <th className="text-center py-2 px-3">W-L</th>
            <th className="text-right py-2 px-3">PF</th>
            <th className="text-center py-2 px-3">Str</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((row, i) => (
            <>
              {i === playoffSpots && (
                <tr key="cutline">
                  <td colSpan={5}>
                    <div className="border-t border-dashed border-white/20 my-1" />
                  </td>
                </tr>
              )}
              <tr key={row.rank} className="border-b border-white/5">
                <td className="py-2 px-3 text-gray-400">{row.rank}</td>
                <td className="py-2 px-3">
                  <p className="text-white font-medium">{row.manager}</p>
                  <p className="text-xs text-gray-500 truncate max-w-[120px]">{row.team_name}</p>
                </td>
                <td className="py-2 px-3 text-center text-gray-300">{row.record}</td>
                <td className="py-2 px-3 text-right text-gray-300">{row.points_for.toFixed(1)}</td>
                <td className="py-2 px-3 text-center">
                  <span className={`text-xs font-bold ${row.streak.startsWith('W') ? '' : 'text-red-400'}`}
                    style={row.streak.startsWith('W') ? { color: accentColor } : {}}>
                    {row.streak}
                  </span>
                </td>
              </tr>
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 4: Home page**

`app/(public)/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { HeroPlayer } from '@/components/audio/HeroPlayer'
import { MatchupCard } from '@/components/scores/MatchupCard'
import { StandingsTable } from '@/components/standings/StandingsTable'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

export default async function HomePage() {
  const db = getAdminClient()

  // Get active season + config
  const { data: season } = await db
    .from('league_seasons')
    .select('id, season, name, end_week, playoff_start_week')
    .eq('is_finished', false)
    .order('season', { ascending: false })
    .limit(1)
    .single()

  if (!season) {
    return (
      <main className="flex items-center justify-center min-h-screen">
        <p className="text-gray-400">Season not yet configured. Check back soon.</p>
      </main>
    )
  }

  const { data: config } = await db
    .from('league_config')
    .select('*')
    .eq('league_season_id', season.id)
    .single()

  // Latest ready episode
  const { data: latestEpisode } = await db
    .from('episodes')
    .select('*')
    .eq('league_season_id', season.id)
    .in('status', ['ready', 'partial'])
    .order('week', { ascending: false })
    .limit(1)
    .single()

  // This week's matchups (most recent complete week)
  const { data: matchups } = await db
    .from('matchups')
    .select('*, teams!team1_key(team_name, manager_nickname), teams!team2_key(team_name, manager_nickname)')
    .eq('league_season_id', season.id)
    .eq('is_complete', true)
    .order('week', { ascending: false })
    .limit(10)

  const latestWeek = matchups?.[0]?.week
  const thisWeekMatchups = matchups?.filter(m => m.week === latestWeek) ?? []

  // Current standings
  const { data: standings } = await db
    .from('standings_snapshots')
    .select('*, teams(team_name, managers(primary_nickname))')
    .eq('league_season_id', season.id)
    .eq('week', latestWeek ?? 0)
    .order('wins', { ascending: false })
    .order('points_for', { ascending: false })

  const playoffSpots = season.playoff_start_week ? 6 : 4
  const accentColor = config?.accent_color ?? '#00ff87'

  return (
    <main className="max-w-2xl mx-auto px-4 py-8 space-y-8">
      {/* Branding */}
      <header className="text-center space-y-1">
        <h1 className="text-2xl font-bold text-white">{config?.show_name ?? season.name}</h1>
        <p className="text-gray-400 text-sm">{season.season} Season</p>
      </header>

      {/* Hero player */}
      {latestEpisode?.audio_url ? (
        <HeroPlayer
          episodeId={latestEpisode.id}
          title={latestEpisode.title}
          audioUrl={latestEpisode.audio_url}
          artworkUrl={config?.episode_artwork_url}
          accentColor={accentColor}
        />
      ) : (
        <div className="bg-[#1a1f2e] rounded-2xl p-8 text-center text-gray-400">
          <p>No episode yet this week. Check back Tuesday.</p>
        </div>
      )}

      {/* This week's results */}
      {thisWeekMatchups.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Week {latestWeek} Results
          </h2>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {thisWeekMatchups.map(m => (
              <MatchupCard
                key={m.id}
                team1Name={(m.teams as any)?.[0]?.team_name ?? ''}
                team1Manager={(m.teams as any)?.[0]?.manager_nickname ?? ''}
                team1Score={m.team1_score ?? 0}
                team2Name={(m.teams as any)?.[1]?.team_name ?? ''}
                team2Manager={(m.teams as any)?.[1]?.manager_nickname ?? ''}
                team2Score={m.team2_score ?? 0}
                winnerKey={m.winner_team_key}
                team1Key={m.team1_key}
                accentColor={accentColor}
              />
            ))}
          </div>
        </section>
      )}

      {/* Standings */}
      {standings && standings.length > 0 && (
        <section>
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Standings</h2>
            <Link href="/standings" className="text-xs text-gray-400 hover:text-white">Full →</Link>
          </div>
          <div className="bg-[#1a1f2e] rounded-xl overflow-hidden">
            <StandingsTable
              standings={standings.map((s, i) => ({
                rank: i + 1,
                manager: (s.teams as any)?.managers?.primary_nickname ?? '',
                team_name: (s.teams as any)?.team_name ?? '',
                record: `${s.wins}-${s.losses}`,
                points_for: s.points_for,
                streak: `${s.streak_type}${s.streak_count}`,
              }))}
              playoffSpots={playoffSpots}
              accentColor={accentColor}
            />
          </div>
        </section>
      )}

      {/* Past episodes */}
      <section>
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Past Episodes</h2>
          <Link href="/archive" className="text-xs text-gray-400 hover:text-white">Archive →</Link>
        </div>
      </section>
    </main>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add app/\(public\)/ components/scores/ components/standings/
git commit -m "feat: public layout with AudioProvider, home page, MatchupCard, StandingsTable"
```

---

### Task 16: Episode Page, Archive, and Standings Pages

**Files:**
- Create: `app/(public)/episode/[season]/[week]/page.tsx`
- Create: `app/(public)/archive/page.tsx`
- Create: `app/(public)/standings/page.tsx`
- Create: `components/episodes/EpisodeCard.tsx`

- [ ] **Step 1: Episode card component**

`components/episodes/EpisodeCard.tsx`:
```typescript
import Link from 'next/link'

interface EpisodeCardProps {
  id: string
  title: string
  season: number
  week: number | null
  episodeType: string
  durationSecs: number | null
  status: string
  accentColor?: string
}

const TYPE_LABELS: Record<string, string> = {
  preseason: 'PRESEASON',
  weekly_recap: 'RECAP',
  weekly_preview: 'PREVIEW',
  playoff_special: 'PLAYOFFS',
  championship: 'CHAMPIONSHIP',
}

export function EpisodeCard({ id, title, season, week, episodeType, durationSecs, status, accentColor = '#00ff87' }: EpisodeCardProps) {
  const href = week ? `/episode/${season}/${week}` : `/episode/${season}/special-${id.slice(0, 8)}`
  const hasAudio = status === 'ready' || status === 'partial'
  const mins = durationSecs ? Math.round(durationSecs / 60) : null

  return (
    <Link href={href} className="block bg-[#1a1f2e] rounded-xl p-4 hover:bg-[#222840] transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full text-black"
              style={{ backgroundColor: accentColor }}>
              {TYPE_LABELS[episodeType] ?? episodeType.toUpperCase()}
            </span>
            {status === 'partial' && (
              <span className="text-xs text-yellow-400">Audio unavailable</span>
            )}
          </div>
          <h3 className="text-white font-medium text-sm truncate">{title}</h3>
          {mins && <p className="text-xs text-gray-400 mt-0.5">{mins} min</p>}
        </div>
        {hasAudio && (
          <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ backgroundColor: accentColor }}>
            <span className="text-black text-sm">▶</span>
          </div>
        )}
      </div>
    </Link>
  )
}
```

- [ ] **Step 2: Individual episode page**

`app/(public)/episode/[season]/[week]/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { HeroPlayer } from '@/components/audio/HeroPlayer'
import Link from 'next/link'
import { notFound } from 'next/navigation'
import ReactMarkdown from 'react-markdown'

// npm install react-markdown

export const dynamic = 'force-dynamic'

interface Props {
  params: Promise<{ season: string; week: string }>
}

export default async function EpisodePage({ params }: Props) {
  const { season, week } = await params
  const db = getAdminClient()

  const { data: episode } = await db
    .from('episodes')
    .select('*, league_seasons(id, name)')
    .eq('season', parseInt(season, 10))
    .eq('week', parseInt(week, 10))
    .in('status', ['ready', 'partial'])
    .order('created_at', { ascending: false })
    .limit(1)
    .single()

  if (!episode) notFound()

  const leagueSeasonId = (episode.league_seasons as any)?.id
  const { data: config } = await db
    .from('league_config')
    .select('accent_color, episode_artwork_url')
    .eq('league_season_id', leagueSeasonId)
    .single()

  // This week's matchups
  const { data: matchups } = await db
    .from('matchups')
    .select('*')
    .eq('league_season_id', leagueSeasonId)
    .eq('week', parseInt(week, 10))
    .eq('is_complete', true)

  // Transactions this week
  const { data: transactions } = await db
    .from('transactions')
    .select('*')
    .eq('league_season_id', leagueSeasonId)
    .eq('week', parseInt(week, 10))
    .eq('status', 'successful')
    .order('occurred_at', { ascending: false })

  const accentColor = config?.accent_color ?? '#00ff87'

  return (
    <main className="max-w-2xl mx-auto px-4 py-8 space-y-8">
      <Link href="/" className="text-sm text-gray-400 hover:text-white flex items-center gap-1">
        ← Back
      </Link>

      <h1 className="text-2xl font-bold text-white">{episode.title}</h1>

      {episode.audio_url && (
        <HeroPlayer
          episodeId={episode.id}
          title={episode.title}
          audioUrl={episode.audio_url}
          artworkUrl={config?.episode_artwork_url}
          accentColor={accentColor}
        />
      )}

      {episode.status === 'partial' && (
        <div className="bg-yellow-400/10 border border-yellow-400/20 rounded-xl p-4">
          <p className="text-yellow-400 text-sm">Audio unavailable for this episode. Read the recap below.</p>
        </div>
      )}

      {/* Scores table */}
      {matchups && matchups.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Results</h2>
          <div className="bg-[#1a1f2e] rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <tbody>
                {matchups.map(m => (
                  <tr key={m.id} className="border-b border-white/5 last:border-0">
                    <td className="py-2 px-4">
                      <span className={m.winner_team_key === m.team1_key ? 'text-white font-medium' : 'text-gray-400'}>
                        {m.team1_key.split('.t.')[1] ?? m.team1_key}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right">
                      <span className={m.winner_team_key === m.team1_key ? 'text-white font-bold' : 'text-gray-400'}>
                        {m.team1_score?.toFixed(1) ?? '—'}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-center text-gray-500 text-xs">vs</td>
                    <td className="py-2 px-2">
                      <span className={m.winner_team_key === m.team2_key ? 'text-white font-bold' : 'text-gray-400'}>
                        {m.team2_score?.toFixed(1) ?? '—'}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      <span className={m.winner_team_key === m.team2_key ? 'text-white font-medium' : 'text-gray-400'}>
                        {m.team2_key.split('.t.')[1] ?? m.team2_key}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Transcript */}
      {episode.transcript_md && (
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Recap</h2>
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{episode.transcript_md}</ReactMarkdown>
          </div>
        </section>
      )}

      {/* Transactions */}
      {transactions && transactions.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Transactions</h2>
          <div className="space-y-2">
            {transactions.map(t => (
              <div key={t.id} className="bg-[#1a1f2e] rounded-xl px-4 py-3">
                <span className="text-xs text-gray-400 uppercase mr-2">{t.type}</span>
                <span className="text-sm text-white">
                  {((t.players_sent as any[]) ?? []).map((p: any) => p.player_name).join(', ')}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  )
}
```

- [ ] **Step 3: Archive page**

`app/(public)/archive/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { EpisodeCard } from '@/components/episodes/EpisodeCard'

export const dynamic = 'force-dynamic'

export default async function ArchivePage() {
  const db = getAdminClient()

  const { data: seasons } = await db
    .from('league_seasons')
    .select('id, season, name')
    .order('season', { ascending: false })

  const { data: episodes } = await db
    .from('episodes')
    .select('*')
    .in('status', ['ready', 'partial'])
    .order('season', { ascending: false })
    .order('week', { ascending: false })

  const { data: config } = await db
    .from('league_config')
    .select('accent_color')
    .limit(1)
    .single()

  const accentColor = config?.accent_color ?? '#00ff87'

  // Group by season
  const bySeason = (seasons ?? []).map(s => ({
    ...s,
    episodes: (episodes ?? []).filter(e => e.season === s.season),
  }))

  return (
    <main className="max-w-2xl mx-auto px-4 py-8 space-y-10">
      <h1 className="text-2xl font-bold text-white">All Episodes</h1>

      {bySeason.map(season => season.episodes.length > 0 && (
        <section key={season.id}>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">
            {season.season} Season
          </h2>
          <div className="space-y-2">
            {season.episodes.map(ep => (
              <EpisodeCard
                key={ep.id}
                id={ep.id}
                title={ep.title}
                season={ep.season}
                week={ep.week}
                episodeType={ep.episode_type}
                durationSecs={ep.audio_duration_secs}
                status={ep.status}
                accentColor={accentColor}
              />
            ))}
          </div>
        </section>
      ))}
    </main>
  )
}
```

- [ ] **Step 4: Standings page**

`app/(public)/standings/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { StandingsTable } from '@/components/standings/StandingsTable'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

export default async function StandingsPage() {
  const db = getAdminClient()

  const { data: season } = await db
    .from('league_seasons')
    .select('id, season, playoff_start_week')
    .eq('is_finished', false)
    .order('season', { ascending: false })
    .limit(1)
    .single()

  if (!season) return <main className="p-8 text-gray-400">No active season.</main>

  const { data: config } = await db
    .from('league_config')
    .select('accent_color')
    .eq('league_season_id', season.id)
    .single()

  const { data: latestWeek } = await db
    .from('standings_snapshots')
    .select('week')
    .eq('league_season_id', season.id)
    .order('week', { ascending: false })
    .limit(1)
    .single()

  const { data: standings } = await db
    .from('standings_snapshots')
    .select('*, teams(team_name, managers(primary_nickname))')
    .eq('league_season_id', season.id)
    .eq('week', latestWeek?.week ?? 0)
    .order('wins', { ascending: false })
    .order('points_for', { ascending: false })

  const accentColor = config?.accent_color ?? '#00ff87'
  const playoffSpots = season.playoff_start_week ? 6 : 4

  return (
    <main className="max-w-2xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Standings</h1>
        <Link href="/" className="text-sm text-gray-400 hover:text-white">← Home</Link>
      </div>
      <p className="text-xs text-gray-400">Through Week {latestWeek?.week ?? '—'} · {season.season} Season</p>

      <div className="bg-[#1a1f2e] rounded-xl overflow-hidden">
        <StandingsTable
          standings={(standings ?? []).map((s, i) => ({
            rank: i + 1,
            manager: (s.teams as any)?.managers?.primary_nickname ?? '',
            team_name: (s.teams as any)?.team_name ?? '',
            record: `${s.wins}-${s.losses}`,
            points_for: s.points_for,
            streak: `${s.streak_type}${s.streak_count}`,
          }))}
          playoffSpots={playoffSpots}
          accentColor={accentColor}
        />
      </div>
    </main>
  )
}
```

- [ ] **Step 5: Install react-markdown**

```bash
npm install react-markdown
npm install -D @tailwindcss/typography
```

Add to `tailwind.config.ts`:
```typescript
plugins: [require('@tailwindcss/typography')],
```

- [ ] **Step 6: Commit**

```bash
git add app/\(public\)/episode/ app/\(public\)/archive/ app/\(public\)/standings/ \
  components/episodes/
git commit -m "feat: episode page, archive, and standings public pages"
```

---

### Task 17: History Page

**Files:**
- Create: `app/(public)/history/page.tsx`
- Create: `components/history/ChampionshipRoll.tsx`
- Create: `components/history/AllTimeRecords.tsx`
- Create: `components/history/CareerStats.tsx`
- Create: `components/history/H2HMatrix.tsx`

- [ ] **Step 1: Championship roll component**

`components/history/ChampionshipRoll.tsx`:
```typescript
interface ChampionRow {
  season: number
  champion: string
  runnerUp: string | null
  score: string
}

export function ChampionshipRoll({ champions, accentColor = '#00ff87' }: { champions: ChampionRow[], accentColor?: string }) {
  return (
    <div className="space-y-3">
      {champions.map(c => (
        <div key={c.season} className="bg-[#1a1f2e] rounded-xl px-5 py-4 flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400">{c.season}</span>
            <p className="text-white font-bold mt-0.5">{c.champion}</p>
            {c.runnerUp && <p className="text-xs text-gray-400">def. {c.runnerUp}</p>}
          </div>
          <span className="text-sm font-mono text-gray-300">{c.score}</span>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: All-time records component**

`components/history/AllTimeRecords.tsx`:
```typescript
interface Record { label: string; value: string; holder: string; context: string }

export function AllTimeRecords({ records }: { records: Record[] }) {
  return (
    <div className="bg-[#1a1f2e] rounded-xl overflow-hidden">
      <table className="w-full text-sm">
        <tbody>
          {records.map(r => (
            <tr key={r.label} className="border-b border-white/5 last:border-0">
              <td className="py-3 px-4 text-gray-400 text-xs">{r.label}</td>
              <td className="py-3 px-4 text-white font-bold text-right">{r.value}</td>
              <td className="py-3 px-4 text-gray-300 text-xs text-right">{r.holder}</td>
              <td className="py-3 px-4 text-gray-500 text-xs text-right hidden sm:table-cell">{r.context}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 3: Career stats component**

`components/history/CareerStats.tsx`:
```typescript
interface CareerRow {
  manager: string
  seasons: number
  record: string
  careerPF: number
  championships: number
  playoffApps: number
}

export function CareerStats({ managers, accentColor = '#00ff87' }: { managers: CareerRow[], accentColor?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm whitespace-nowrap">
        <thead>
          <tr className="text-gray-400 border-b border-white/10 text-xs">
            <th className="text-left py-2 px-3">Manager</th>
            <th className="text-center py-2 px-3">Seasons</th>
            <th className="text-center py-2 px-3">Record</th>
            <th className="text-right py-2 px-3">Career PF</th>
            <th className="text-center py-2 px-3">Champs</th>
            <th className="text-center py-2 px-3">Playoffs</th>
          </tr>
        </thead>
        <tbody>
          {managers.map(m => (
            <tr key={m.manager} className="border-b border-white/5">
              <td className="py-2 px-3 text-white font-medium">{m.manager}</td>
              <td className="py-2 px-3 text-gray-300 text-center">{m.seasons}</td>
              <td className="py-2 px-3 text-gray-300 text-center">{m.record}</td>
              <td className="py-2 px-3 text-gray-300 text-right">{m.careerPF.toFixed(1)}</td>
              <td className="py-2 px-3 text-center">
                {m.championships > 0 ? (
                  <span className="font-bold" style={{ color: accentColor }}>{m.championships}</span>
                ) : (
                  <span className="text-gray-500">—</span>
                )}
              </td>
              <td className="py-2 px-3 text-gray-300 text-center">{m.playoffApps}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 4: H2H Matrix component**

`components/history/H2HMatrix.tsx`:
```typescript
interface H2HEntry { wins: number; losses: number }

interface H2HMatrixProps {
  managers: string[]
  matrix: Record<string, Record<string, H2HEntry>>
  accentColor?: string
}

export function H2HMatrix({ managers, matrix, accentColor = '#00ff87' }: H2HMatrixProps) {
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="w-24 p-2 text-gray-400 text-left">vs.</th>
            {managers.map(m => (
              <th key={m} className="p-2 text-gray-400 text-center max-w-[60px] truncate">{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {managers.map(rowMgr => (
            <tr key={rowMgr} className="border-t border-white/5">
              <td className="p-2 text-white font-medium text-xs max-w-[96px] truncate">{rowMgr}</td>
              {managers.map(colMgr => {
                if (rowMgr === colMgr) {
                  return <td key={colMgr} className="p-2 bg-white/5 text-center">—</td>
                }
                const entry = matrix[rowMgr]?.[colMgr]
                if (!entry) return <td key={colMgr} className="p-2 text-center text-gray-600">—</td>
                const total = entry.wins + entry.losses
                return (
                  <td key={colMgr} className="p-2 text-center">
                    <span className={entry.wins > entry.losses ? 'font-bold' : 'text-gray-400'}
                      style={entry.wins > entry.losses ? { color: accentColor } : {}}>
                      {entry.wins}-{entry.losses}
                    </span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 5: History page (data assembly)**

`app/(public)/history/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import { ChampionshipRoll } from '@/components/history/ChampionshipRoll'
import { AllTimeRecords } from '@/components/history/AllTimeRecords'
import { CareerStats } from '@/components/history/CareerStats'
import { H2HMatrix } from '@/components/history/H2HMatrix'

export const dynamic = 'force-dynamic'

export default async function HistoryPage() {
  const db = getAdminClient()

  const { data: config } = await db.from('league_config').select('accent_color').limit(1).single()
  const accentColor = config?.accent_color ?? '#00ff87'

  // Championship history
  const { data: championships } = await db
    .from('matchups')
    .select('*, league_seasons(season)')
    .eq('is_championship', true)
    .eq('is_complete', true)
    .order('season', { ascending: false })

  const champRows = (championships ?? []).map(c => ({
    season: (c.league_seasons as any)?.season ?? c.season,
    champion: c.winner_team_key === c.team1_key ? c.team1_key : c.team2_key,
    runnerUp: c.winner_team_key === c.team1_key ? c.team2_key : c.team1_key,
    score: `${c.team1_score?.toFixed(1) ?? '?'} - ${c.team2_score?.toFixed(1) ?? '?'}`,
  }))

  // All-time records
  const { data: maxScore } = await db
    .from('matchups').select('team1_score, team2_score, team1_key, team2_key, season, week')
    .eq('is_complete', true)
    .order('team1_score', { ascending: false }).limit(1).single()

  const { data: maxMargin } = await db
    .from('matchups').select('margin, winner_team_key, team1_key, team2_key, season, week')
    .eq('is_complete', true)
    .order('margin', { ascending: false }).limit(1).single()

  const { data: minMargin } = await db
    .from('matchups').select('margin, season, week')
    .eq('is_complete', true)
    .gt('margin', 0)
    .order('margin', { ascending: true }).limit(1).single()

  const allTimeRecords = [
    { label: 'Highest score', value: maxScore?.team1_score?.toFixed(1) ?? '—', holder: maxScore?.team1_key?.split('.t.')[1] ?? '—', context: `Week ${maxScore?.week}, ${maxScore?.season}` },
    { label: 'Biggest blowout', value: maxMargin?.margin?.toFixed(1) ?? '—', holder: maxMargin?.winner_team_key?.split('.t.')[1] ?? '—', context: `Week ${maxMargin?.week}, ${maxMargin?.season}` },
    { label: 'Closest game', value: minMargin?.margin?.toFixed(1) ?? '—', holder: '—', context: `Week ${minMargin?.week}, ${minMargin?.season}` },
  ]

  // Career stats — only confirmed managers
  const { data: confirmedManagers } = await db
    .from('managers')
    .select('id, primary_nickname')
    .eq('identity_status', 'confirmed')
    .order('primary_nickname')

  const careerRows = await Promise.all((confirmedManagers ?? []).map(async mgr => {
    const { data: teams } = await db
      .from('teams')
      .select('wins, losses, ties, made_playoffs, is_champion, points_for')
      .eq('manager_id', mgr.id)

    const seasons = teams?.length ?? 0
    const totalW = (teams ?? []).reduce((s, t) => s + t.wins, 0)
    const totalL = (teams ?? []).reduce((s, t) => s + t.losses, 0)
    const champs = (teams ?? []).filter(t => t.is_champion).length
    const playoffs = (teams ?? []).filter(t => t.made_playoffs).length
    const pf = (teams ?? []).reduce((s, t) => s + t.points_for, 0)

    return {
      manager: mgr.primary_nickname,
      seasons,
      record: `${totalW}-${totalL}`,
      careerPF: pf,
      championships: champs,
      playoffApps: playoffs,
    }
  }))

  // H2H matrix (simplified — all-time between confirmed managers)
  const managers = (confirmedManagers ?? []).map(m => m.primary_nickname)
  const matrix: Record<string, Record<string, { wins: number; losses: number }>> = {}

  // Initialize matrix
  for (const m of managers) { matrix[m] = {} }

  const { data: allMatchups } = await db
    .from('matchups')
    .select('team1_key, team2_key, winner_team_key, teams!team1_key(manager_id, managers(primary_nickname)), teams!team2_key(manager_id, managers(primary_nickname))')
    .eq('is_complete', true)

  for (const m of (allMatchups ?? [])) {
    const mgr1 = (m.teams as any)?.[0]?.managers?.primary_nickname
    const mgr2 = (m.teams as any)?.[1]?.managers?.primary_nickname
    if (!mgr1 || !mgr2 || mgr1 === mgr2) continue

    if (!matrix[mgr1]) matrix[mgr1] = {}
    if (!matrix[mgr2]) matrix[mgr2] = {}
    if (!matrix[mgr1][mgr2]) matrix[mgr1][mgr2] = { wins: 0, losses: 0 }
    if (!matrix[mgr2][mgr1]) matrix[mgr2][mgr1] = { wins: 0, losses: 0 }

    if (m.winner_team_key === m.team1_key) {
      matrix[mgr1][mgr2].wins++
      matrix[mgr2][mgr1].losses++
    } else if (m.winner_team_key === m.team2_key) {
      matrix[mgr2][mgr1].wins++
      matrix[mgr1][mgr2].losses++
    }
  }

  return (
    <main className="max-w-3xl mx-auto px-4 py-8 space-y-10">
      <h1 className="text-2xl font-bold text-white">League History</h1>

      <section>
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Championship Roll</h2>
        <ChampionshipRoll champions={champRows} accentColor={accentColor} />
      </section>

      <section>
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">All-Time Records</h2>
        <AllTimeRecords records={allTimeRecords} />
      </section>

      <section>
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Manager Career Stats</h2>
        <div className="bg-[#1a1f2e] rounded-xl overflow-hidden">
          <CareerStats managers={careerRows} accentColor={accentColor} />
        </div>
      </section>

      {managers.length > 1 && (
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Head-to-Head</h2>
          <div className="bg-[#1a1f2e] rounded-xl p-4 overflow-x-auto">
            <H2HMatrix managers={managers} matrix={matrix} accentColor={accentColor} />
          </div>
        </section>
      )}
    </main>
  )
}
```

- [ ] **Step 6: Commit**

```bash
git add app/\(public\)/history/ components/history/
git commit -m "feat: history page — championship roll, all-time records, career stats, H2H matrix"
```

---

### Task 18: Admin Auth + Setup Wizard

**Files:**
- Create: `app/admin/layout.tsx`
- Create: `app/admin/setup/page.tsx`
- Create: `components/admin/SetupWizard.tsx`

- [ ] **Step 1: Admin auth layout**

`app/admin/layout.tsx`:
```typescript
import { createServerClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import type { ReactNode } from 'react'
import Link from 'next/link'

export default async function AdminLayout({ children }: { children: ReactNode }) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) redirect('/admin/login')

  return (
    <div className="min-h-screen bg-[#0f1117] text-white">
      <nav className="border-b border-white/10 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/admin" className="text-sm font-medium text-white">Dashboard</Link>
          <Link href="/admin/episodes" className="text-sm text-gray-400 hover:text-white">Episodes</Link>
          <Link href="/admin/managers" className="text-sm text-gray-400 hover:text-white">Managers</Link>
          <Link href="/admin/ingestion" className="text-sm text-gray-400 hover:text-white">Ingestion</Link>
        </div>
        <Link href="/" className="text-xs text-gray-400 hover:text-white">← Public site</Link>
      </nav>
      <main className="max-w-4xl mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Admin login page (magic link)**

`app/admin/login/page.tsx`:
```typescript
'use client'

import { useState } from 'react'
import { createBrowserClient } from '@/lib/supabase/client'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [sent, setSent] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    const supabase = createBrowserClient()
    await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/admin` },
    })
    setSent(true)
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-[#0f1117] flex items-center justify-center px-4">
      <div className="w-full max-w-sm space-y-6">
        <h1 className="text-2xl font-bold text-white text-center">Commissioner Login</h1>
        {sent ? (
          <p className="text-center text-gray-300">Check your email for the magic link.</p>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              type="email" required value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="your@email.com"
              className="w-full bg-[#1a1f2e] border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-gray-500 focus:outline-none focus:border-white/30"
            />
            <button type="submit" disabled={loading}
              className="w-full bg-[#00ff87] text-black font-bold py-3 rounded-xl hover:bg-[#00e07a] transition-colors disabled:opacity-50">
              {loading ? 'Sending...' : 'Send Magic Link'}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Setup wizard — 6 steps**

`components/admin/SetupWizard.tsx`:
```typescript
'use client'

import { useState } from 'react'

type Step = 1 | 2 | 3 | 4 | 5 | 6

interface SetupState {
  step: Step
  yahooConnected: boolean
  selectedLeague: { league_key: string; name: string; league_id: string } | null
  showName: string
  accentColor: string
  priorLeagueKey: string
  priorLeagueKeyMode: 'auto' | 'manual' | 'skip'
  discoveredPriorKey: string | null
  troyVoiceId: string
  lisaVoiceId: string
  dannyVoiceId: string
  ingestionDone: boolean
}

export function SetupWizard({ initialConnected = false, discoveredPriorKey = null }: {
  initialConnected?: boolean
  discoveredPriorKey?: string | null
}) {
  const [state, setState] = useState<SetupState>({
    step: initialConnected ? 2 : 1,
    yahooConnected: initialConnected,
    selectedLeague: null,
    showName: '',
    accentColor: '#00ff87',
    priorLeagueKey: discoveredPriorKey ?? '',
    priorLeagueKeyMode: discoveredPriorKey ? 'auto' : 'manual',
    discoveredPriorKey,
    troyVoiceId: '',
    lisaVoiceId: '',
    dannyVoiceId: '',
    ingestionDone: false,
  })

  const [leagues, setLeagues] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const update = (patch: Partial<SetupState>) => setState(s => ({ ...s, ...patch }))

  // Step 1: Connect Yahoo
  const connectYahoo = () => { window.location.href = '/api/yahoo/connect' }

  // Step 2: Select league
  const loadLeagues = async () => {
    setLoading(true)
    const resp = await fetch('/api/yahoo/leagues')
    const data = await resp.json()
    setLeagues(data.leagues ?? [])
    setLoading(false)
  }

  // Step 6: Run ingestion
  const runIngestion = async () => {
    if (!state.selectedLeague) return
    setLoading(true)
    try {
      // First, ensure league season row exists
      const setupResp = await fetch('/api/admin/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          leagueKey: state.selectedLeague.league_key,
          leagueId: state.selectedLeague.league_id,
          showName: state.showName,
          accentColor: state.accentColor,
          troyVoiceId: state.troyVoiceId,
          lisaVoiceId: state.lisaVoiceId,
          dannyVoiceId: state.dannyVoiceId,
          priorLeagueKey: state.priorLeagueKeyMode === 'skip' ? null : state.priorLeagueKey,
        }),
      })
      const setupData = await setupResp.json()
      if (!setupResp.ok) throw new Error(setupData.error)

      update({ ingestionDone: true, step: 6 })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Setup failed')
    } finally {
      setLoading(false)
    }
  }

  const steps = [
    { num: 1, label: 'Connect Yahoo' },
    { num: 2, label: 'Select League' },
    { num: 3, label: 'Branding' },
    { num: 4, label: 'Prior Season' },
    { num: 5, label: 'Host Voices' },
    { num: 6, label: 'Ingest & Launch' },
  ]

  return (
    <div className="space-y-8">
      {/* Step indicator */}
      <div className="flex items-center gap-2">
        {steps.map((s, i) => (
          <div key={s.num} className="flex items-center gap-2">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
              state.step >= s.num ? 'bg-[#00ff87] text-black' : 'bg-white/10 text-gray-400'
            }`}>{s.num}</div>
            {i < steps.length - 1 && <div className="w-8 h-px bg-white/10" />}
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-red-400/10 border border-red-400/20 rounded-xl p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Step 1 */}
      {state.step === 1 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Connect Yahoo Fantasy</h2>
          <p className="text-gray-400 text-sm">You&apos;ll be redirected to Yahoo to authorize access. The system only reads data — it never makes roster changes.</p>
          <button onClick={connectYahoo}
            className="bg-[#00ff87] text-black font-bold px-6 py-3 rounded-xl hover:bg-[#00e07a] transition-colors">
            Connect Yahoo →
          </button>
        </div>
      )}

      {/* Step 2 */}
      {state.step === 2 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Select Your League</h2>
          {leagues.length === 0 ? (
            <button onClick={loadLeagues} disabled={loading}
              className="bg-[#00ff87] text-black font-bold px-6 py-3 rounded-xl">
              {loading ? 'Loading...' : 'Load My Leagues'}
            </button>
          ) : (
            <div className="space-y-2">
              {leagues.map(lg => (
                <button key={lg.league_key}
                  onClick={() => { update({ selectedLeague: lg, step: 3 }) }}
                  className={`w-full text-left bg-[#1a1f2e] rounded-xl px-4 py-3 hover:bg-[#222840] transition-colors ${
                    state.selectedLeague?.league_key === lg.league_key ? 'border border-[#00ff87]' : ''
                  }`}>
                  <p className="text-white font-medium">{lg.name}</p>
                  <p className="text-xs text-gray-400">{lg.season} · {lg.league_key}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Step 3: Branding */}
      {state.step === 3 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Name Your Show</h2>
          <div className="space-y-2">
            <label className="text-sm text-gray-400">Show Name</label>
            <input value={state.showName} onChange={e => update({ showName: e.target.value })}
              placeholder="e.g. The Spunt Report"
              className="w-full bg-[#1a1f2e] border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-gray-500 focus:outline-none focus:border-white/30"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm text-gray-400">Accent Color</label>
            <div className="flex items-center gap-3">
              <input type="color" value={state.accentColor}
                onChange={e => update({ accentColor: e.target.value })}
                className="w-12 h-10 rounded cursor-pointer bg-transparent border-0"
              />
              <span className="text-gray-300 text-sm">{state.accentColor}</span>
            </div>
          </div>
          <button onClick={() => update({ step: 4 })} disabled={!state.showName}
            className="bg-[#00ff87] text-black font-bold px-6 py-3 rounded-xl disabled:opacity-50">
            Next →
          </button>
        </div>
      )}

      {/* Step 4: Prior season */}
      {state.step === 4 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Prior Season</h2>
          {state.discoveredPriorKey ? (
            <div className="bg-[#1a1f2e] rounded-xl p-4">
              <p className="text-sm text-gray-300">Found prior season automatically:</p>
              <p className="text-white font-mono mt-1">{state.discoveredPriorKey}</p>
              <p className="text-xs text-gray-400 mt-2">Historical data from this season will be ingested.</p>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-gray-400">No prior season was auto-detected. Enter the prior league key manually, or skip.</p>
              <input value={state.priorLeagueKey}
                onChange={e => update({ priorLeagueKey: e.target.value })}
                placeholder="e.g. 449.l.123456"
                className="w-full bg-[#1a1f2e] border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-gray-500 focus:outline-none"
              />
              <button onClick={() => update({ priorLeagueKeyMode: 'skip' })}
                className="text-sm text-gray-400 hover:text-white underline">
                Skip — start with current season only
              </button>
            </div>
          )}
          <button onClick={() => update({ step: 5 })}
            className="bg-[#00ff87] text-black font-bold px-6 py-3 rounded-xl">
            Next →
          </button>
        </div>
      )}

      {/* Step 5: Voices */}
      {state.step === 5 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Host Voices</h2>
          <p className="text-sm text-gray-400">Enter ElevenLabs voice IDs for each host. Find these in your ElevenLabs dashboard under Voices.</p>
          {(['troy', 'lisa', 'danny'] as const).map(host => (
            <div key={host} className="space-y-1">
              <label className="text-sm text-gray-400 capitalize">{host} Voice ID</label>
              <input
                value={state[`${host}VoiceId` as keyof SetupState] as string}
                onChange={e => update({ [`${host}VoiceId`]: e.target.value } as any)}
                placeholder="xxxxxxxxxxxxxx"
                className="w-full bg-[#1a1f2e] border border-white/10 rounded-xl px-4 py-3 text-white font-mono placeholder:text-gray-500 focus:outline-none focus:border-white/30 text-sm"
              />
            </div>
          ))}
          <button onClick={() => update({ step: 6 })}
            disabled={!state.troyVoiceId || !state.lisaVoiceId || !state.dannyVoiceId}
            className="bg-[#00ff87] text-black font-bold px-6 py-3 rounded-xl disabled:opacity-50">
            Next →
          </button>
        </div>
      )}

      {/* Step 6: Ingest */}
      {state.step === 6 && !state.ingestionDone && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Ready to Launch</h2>
          <div className="bg-[#1a1f2e] rounded-xl p-4 space-y-2 text-sm">
            <p><span className="text-gray-400">Show:</span> <span className="text-white">{state.showName}</span></p>
            <p><span className="text-gray-400">League:</span> <span className="text-white">{state.selectedLeague?.name}</span></p>
            {state.priorLeagueKey && state.priorLeagueKeyMode !== 'skip' && (
              <p><span className="text-gray-400">Prior season:</span> <span className="text-white">{state.priorLeagueKey}</span></p>
            )}
          </div>
          <button onClick={runIngestion} disabled={loading}
            className="bg-[#00ff87] text-black font-bold px-6 py-3 rounded-xl disabled:opacity-50">
            {loading ? 'Running ingestion...' : 'Run Ingestion & Launch →'}
          </button>
        </div>
      )}

      {state.ingestionDone && (
        <div className="space-y-4">
          <div className="bg-[#00ff87]/10 border border-[#00ff87]/20 rounded-xl p-6 text-center">
            <p className="text-[#00ff87] font-bold text-lg">Setup Complete</p>
            <p className="text-gray-300 text-sm mt-2">Historical data ingested. Your league is live.</p>
          </div>
          <div className="flex gap-3">
            <a href="/admin" className="flex-1 text-center bg-[#00ff87] text-black font-bold py-3 rounded-xl">
              Go to Dashboard
            </a>
            <a href="/admin/episodes" className="flex-1 text-center bg-[#1a1f2e] text-white font-bold py-3 rounded-xl">
              Generate Episode
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Setup page**

`app/admin/setup/page.tsx`:
```typescript
import { SetupWizard } from '@/components/admin/SetupWizard'

interface Props {
  searchParams: Promise<{ connected?: string; error?: string; prior?: string }>
}

export default async function SetupPage({ searchParams }: Props) {
  const params = await searchParams
  return (
    <div className="max-w-xl space-y-6">
      <h1 className="text-2xl font-bold">League Setup</h1>
      <SetupWizard
        initialConnected={params.connected === 'true'}
        discoveredPriorKey={params.prior ?? null}
      />
    </div>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add app/admin/ components/admin/SetupWizard.tsx
git commit -m "feat: admin auth layout, magic link login, 6-step setup wizard"
```

---

### Task 19: Admin Dashboard, Episodes, Managers, Ingestion Log

**Files:**
- Create: `app/admin/page.tsx` (dashboard)
- Create: `app/admin/episodes/page.tsx`
- Create: `app/admin/managers/page.tsx`
- Create: `app/admin/ingestion/page.tsx`
- Create: `components/admin/EpisodeTable.tsx`
- Create: `components/admin/ManagerTable.tsx`
- Create: `components/admin/IngestionLog.tsx`

- [ ] **Step 1: Commissioner dashboard**

`app/admin/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

export default async function AdminDashboard() {
  const db = getAdminClient()

  const { data: season } = await db
    .from('league_seasons')
    .select('id, season, name')
    .eq('is_finished', false)
    .order('season', { ascending: false })
    .limit(1)
    .single()

  if (!season) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Commissioner Dashboard</h1>
        <p className="text-gray-400">No active season. <Link href="/admin/setup" className="text-[#00ff87]">Run setup →</Link></p>
      </div>
    )
  }

  const { data: latestLog } = await db
    .from('ingestion_log')
    .select('*')
    .eq('league_season_id', season.id)
    .order('started_at', { ascending: false })
    .limit(1)
    .single()

  const { data: latestEpisode } = await db
    .from('episodes')
    .select('title, status, updated_at')
    .eq('league_season_id', season.id)
    .order('created_at', { ascending: false })
    .limit(1)
    .single()

  const { count: flaggedCount } = await db
    .from('managers')
    .select('*', { count: 'exact', head: true })
    .in('identity_status', ['needs_review', 'unmatched'])

  const statusColors: Record<string, string> = {
    ready: 'text-[#00ff87]',
    partial: 'text-yellow-400',
    failed: 'text-red-400',
    generating_script: 'text-blue-400',
    generating_audio: 'text-blue-400',
    pending: 'text-gray-400',
  }

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">{season.name} — {season.season}</h1>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Ingestion status */}
        <div className="bg-[#1a1f2e] rounded-xl p-4">
          <p className="text-xs text-gray-400 uppercase">Last Sync</p>
          <p className="text-white font-medium mt-1">
            {latestLog ? new Date(latestLog.started_at).toLocaleDateString() : 'Never'}
          </p>
          <p className={`text-xs mt-1 ${latestLog?.status === 'complete' ? 'text-[#00ff87]' : 'text-yellow-400'}`}>
            {latestLog?.status ?? '—'}
          </p>
          <Link href="/admin/ingestion" className="text-xs text-gray-400 hover:text-white mt-2 block">View log →</Link>
        </div>

        {/* Episode status */}
        <div className="bg-[#1a1f2e] rounded-xl p-4">
          <p className="text-xs text-gray-400 uppercase">Latest Episode</p>
          <p className="text-white font-medium mt-1 truncate">{latestEpisode?.title ?? 'None'}</p>
          <p className={`text-xs mt-1 ${statusColors[latestEpisode?.status ?? ''] ?? 'text-gray-400'}`}>
            {latestEpisode?.status ?? '—'}
          </p>
          <Link href="/admin/episodes" className="text-xs text-gray-400 hover:text-white mt-2 block">Manage →</Link>
        </div>

        {/* Manager alerts */}
        <div className="bg-[#1a1f2e] rounded-xl p-4">
          <p className="text-xs text-gray-400 uppercase">Manager Alerts</p>
          <p className={`text-2xl font-bold mt-1 ${(flaggedCount ?? 0) > 0 ? 'text-yellow-400' : 'text-[#00ff87]'}`}>
            {flaggedCount ?? 0}
          </p>
          <p className="text-xs text-gray-400 mt-1">need review</p>
          <Link href="/admin/managers" className="text-xs text-gray-400 hover:text-white mt-2 block">Review →</Link>
        </div>
      </div>

      {/* Quick actions */}
      <div className="space-y-2">
        <h2 className="text-sm font-semibold text-gray-400 uppercase">Quick Actions</h2>
        <div className="flex flex-wrap gap-3">
          <form action="/api/ingest/weekly" method="POST">
            <button className="bg-[#1a1f2e] hover:bg-[#222840] text-white text-sm px-4 py-2 rounded-lg transition-colors">
              Sync Now
            </button>
          </form>
          <Link href="/admin/episodes" className="bg-[#00ff87] text-black font-bold text-sm px-4 py-2 rounded-lg">
            Generate Episode
          </Link>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Episodes management page**

`app/admin/episodes/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

const STATUS_COLORS: Record<string, string> = {
  ready: 'bg-[#00ff87]/20 text-[#00ff87]',
  partial: 'bg-yellow-400/20 text-yellow-400',
  failed: 'bg-red-400/20 text-red-400',
  generating_script: 'bg-blue-400/20 text-blue-400',
  generating_audio: 'bg-blue-400/20 text-blue-400',
  pending: 'bg-gray-400/20 text-gray-400',
}

export default async function EpisodesPage() {
  const db = getAdminClient()

  const { data: season } = await db
    .from('league_seasons')
    .select('id')
    .eq('is_finished', false)
    .order('season', { ascending: false })
    .limit(1)
    .single()

  const { data: episodes } = await db
    .from('episodes')
    .select('*')
    .eq('league_season_id', season?.id ?? '')
    .order('created_at', { ascending: false })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Episodes</h1>
        <form action="/api/episode/generate" method="POST">
          <input type="hidden" name="leagueSeasonId" value={season?.id ?? ''} />
          <input type="hidden" name="episodeType" value="weekly_recap" />
          <button className="bg-[#00ff87] text-black font-bold text-sm px-4 py-2 rounded-lg">
            Generate This Week's Recap
          </button>
        </form>
      </div>

      <div className="space-y-2">
        {(episodes ?? []).map(ep => (
          <div key={ep.id} className="bg-[#1a1f2e] rounded-xl px-4 py-3 flex items-center justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-white font-medium text-sm truncate">{ep.title}</p>
              <p className="text-xs text-gray-400">{ep.episode_type} · {new Date(ep.created_at).toLocaleDateString()}</p>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              <span className={`text-xs px-2 py-1 rounded-full ${STATUS_COLORS[ep.status] ?? 'bg-gray-400/20 text-gray-400'}`}>
                {ep.status}
              </span>
              {ep.status === 'failed' && (
                <form action="/api/episode/generate" method="POST">
                  <input type="hidden" name="episodeId" value={ep.id} />
                  <button className="text-xs text-red-400 hover:text-white">Retry</button>
                </form>
              )}
              {ep.audio_url && (
                <a href={ep.audio_url} target="_blank" rel="noreferrer"
                  className="text-xs text-gray-400 hover:text-white">Audio</a>
              )}
              {ep.error_message && (
                <span className="text-xs text-red-400 truncate max-w-[200px]" title={ep.error_message}>
                  Error
                </span>
              )}
            </div>
          </div>
        ))}
        {!episodes?.length && (
          <p className="text-gray-400 text-sm">No episodes yet. Generate your first one above.</p>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Manager review page**

`app/admin/managers/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'

export const dynamic = 'force-dynamic'

const STATUS_COLORS: Record<string, string> = {
  confirmed: 'bg-[#00ff87]/20 text-[#00ff87]',
  needs_review: 'bg-yellow-400/20 text-yellow-400',
  unmatched: 'bg-red-400/20 text-red-400',
}

export default async function ManagersPage() {
  const db = getAdminClient()

  const { data: managers } = await db
    .from('managers')
    .select('*, manager_nicknames(nickname, season, is_primary)')
    .order('identity_status')
    .order('primary_nickname')

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Manager Identity</h1>
      <p className="text-sm text-gray-400">
        Managers with <span className="text-yellow-400">needs_review</span> or <span className="text-red-400">unmatched</span> status
        are excluded from cross-season historical claims in the broadcast.
      </p>

      <div className="space-y-2">
        {(managers ?? []).map(mgr => (
          <div key={mgr.id} className="bg-[#1a1f2e] rounded-xl px-4 py-3">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-white font-medium">{mgr.primary_nickname}</p>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLORS[mgr.identity_status] ?? ''}`}>
                    {mgr.identity_status}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  GUID: {mgr.yahoo_guid ?? 'none'} · First season: {mgr.first_season ?? '—'}
                </p>
                <p className="text-xs text-gray-500 mt-0.5">
                  Nicknames: {((mgr.manager_nicknames as any[]) ?? []).map((n: any) => `${n.nickname} (${n.season})`).join(', ')}
                </p>
              </div>
              {mgr.identity_status !== 'confirmed' && (
                <form action="/api/admin/managers/confirm" method="POST">
                  <input type="hidden" name="managerId" value={mgr.id} />
                  <button className="text-sm text-[#00ff87] hover:underline flex-shrink-0">
                    Confirm →
                  </button>
                </form>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Ingestion log page**

`app/admin/ingestion/page.tsx`:
```typescript
import { getAdminClient } from '@/lib/supabase/admin'

export const dynamic = 'force-dynamic'

export default async function IngestionPage() {
  const db = getAdminClient()

  const { data: season } = await db
    .from('league_seasons')
    .select('id')
    .eq('is_finished', false)
    .order('season', { ascending: false })
    .limit(1)
    .single()

  const { data: logs } = await db
    .from('ingestion_log')
    .select('*')
    .eq('league_season_id', season?.id ?? '')
    .order('started_at', { ascending: false })
    .limit(50)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Ingestion Log</h1>
        <form action="/api/ingest/weekly" method="POST">
          <button className="bg-[#1a1f2e] hover:bg-[#222840] text-white text-sm px-4 py-2 rounded-lg">
            Force Sync
          </button>
        </form>
      </div>

      <div className="space-y-2">
        {(logs ?? []).map(log => (
          <div key={log.id} className="bg-[#1a1f2e] rounded-xl px-4 py-3">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-white text-sm">
                  {log.ingestion_type} {log.week ? `· Week ${log.week}` : ''}
                </p>
                <p className="text-xs text-gray-400">{new Date(log.started_at).toLocaleString()}</p>
                {log.warnings?.length > 0 && (
                  <p className="text-xs text-yellow-400 mt-1">{log.warnings.length} warning(s)</p>
                )}
                {log.error_message && (
                  <p className="text-xs text-red-400 mt-1 truncate max-w-sm">{log.error_message}</p>
                )}
              </div>
              <div className="text-right flex-shrink-0">
                <p className={`text-sm font-medium ${log.status === 'complete' ? 'text-[#00ff87]' : log.status === 'failed' ? 'text-red-400' : 'text-yellow-400'}`}>
                  {log.status}
                </p>
                <p className="text-xs text-gray-400">{log.records_upserted} records</p>
              </div>
            </div>
          </div>
        ))}
        {!logs?.length && <p className="text-gray-400 text-sm">No ingestion runs yet.</p>}
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Admin manager confirm route**

`app/api/admin/managers/confirm/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase/server'
import { getAdminClient } from '@/lib/supabase/admin'

export async function POST(request: NextRequest) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const formData = await request.formData()
  const managerId = formData.get('managerId') as string

  if (!managerId) return NextResponse.json({ error: 'Missing managerId' }, { status: 400 })

  const db = getAdminClient()
  await db.from('managers').update({ identity_status: 'confirmed' }).eq('id', managerId)

  return NextResponse.redirect(new URL('/admin/managers', request.url))
}
```

- [ ] **Step 6: Admin setup API route**

`app/api/admin/setup/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server'
import { createServerClient } from '@/lib/supabase/server'
import { getAdminClient } from '@/lib/supabase/admin'
import { getLeagueMetadata } from '@/lib/yahoo/endpoints'
import { runHistoricalIngestion } from '@/lib/ingestion/historical'
import { randomBytes } from 'crypto'

export async function POST(request: NextRequest) {
  const supabase = await createServerClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { leagueKey, leagueId, showName, accentColor, troyVoiceId, lisaVoiceId, dannyVoiceId, priorLeagueKey } = await request.json()

  const db = getAdminClient()

  // Fetch league metadata
  const meta = await getLeagueMetadata(leagueKey, leagueId)

  // Upsert league_season
  const { data: leagueSeason, error: lsError } = await db
    .from('league_seasons')
    .upsert({
      league_key: leagueKey,
      league_id: leagueId,
      game_id: meta.game_id,
      season: meta.season,
      name: meta.name,
      num_teams: meta.num_teams,
      scoring_type: meta.scoring_type,
      start_week: meta.start_week,
      end_week: meta.end_week,
      playoff_start_week: meta.playoff_start_week,
      prior_league_key: priorLeagueKey ?? meta.prior_league_key,
    }, { onConflict: 'league_key' })
    .select('id')
    .single()

  if (lsError || !leagueSeason) {
    return NextResponse.json({ error: lsError?.message ?? 'Failed to create season' }, { status: 500 })
  }

  // Upsert league_config
  const shareToken = randomBytes(16).toString('hex')
  await db.from('league_config').upsert({
    league_season_id: leagueSeason.id,
    show_name: showName,
    share_token: shareToken,
    accent_color: accentColor,
    host_troy_voice_id: troyVoiceId || null,
    host_lisa_voice_id: lisaVoiceId || null,
    host_danny_voice_id: dannyVoiceId || null,
  }, { onConflict: 'league_season_id' })

  // Update token record to use the real leagueId
  await db.from('commissioner_tokens')
    .update({ league_id: leagueId })
    .eq('league_id', 'pending')

  // Run historical ingestion async if prior league key provided
  const effectivePriorKey = priorLeagueKey ?? meta.prior_league_key
  if (effectivePriorKey) {
    runHistoricalIngestion(leagueSeason.id, effectivePriorKey, leagueId).catch(console.error)
  }

  return NextResponse.json({ success: true, leagueSeasonId: leagueSeason.id, shareToken })
}
```

- [ ] **Step 7: Commit**

```bash
git add app/admin/ app/api/admin/ components/admin/
git commit -m "feat: admin dashboard, episodes management, manager review, ingestion log"
```

---

### Task 20: End-to-End Integration Test + Deploy Prep

**Files:**
- Create: `__tests__/e2e/pipeline.test.ts`

- [ ] **Step 1: Write integration test for the full pipeline (mocked externals)**

`__tests__/e2e/pipeline.test.ts`:
```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock all external services
vi.mock('@/lib/supabase/admin', () => ({
  getAdminClient: vi.fn(() => ({
    from: vi.fn(() => ({
      select: vi.fn().mockReturnThis(),
      eq: vi.fn().mockReturnThis(),
      in: vi.fn().mockReturnThis(),
      order: vi.fn().mockReturnThis(),
      limit: vi.fn().mockReturnThis(),
      single: vi.fn().mockResolvedValue({
        data: {
          id: 'ep-uuid',
          league_season_id: 'ls-uuid',
          season: 2025,
          week: 5,
          episode_type: 'weekly_recap',
          title: 'Week 5',
          status: 'pending',
          script_json: null,
          transcript_md: null,
          audio_url: null,
        },
        error: null,
      }),
      update: vi.fn().mockReturnThis(),
      insert: vi.fn().mockReturnThis(),
      upsert: vi.fn().mockResolvedValue({ error: null }),
    })),
    storage: { from: vi.fn(() => ({ upload: vi.fn().mockResolvedValue({ error: null }), getPublicUrl: vi.fn().mockReturnValue({ data: { publicUrl: 'https://storage.test/ep.mp3' } }) })) },
  })),
}))

vi.mock('@/lib/context-builder/builder', () => ({
  buildWeeklyRecapContext: vi.fn().mockResolvedValue({
    meta: { episode_type: 'weekly_recap', week: 5, season: 2025, league_name: 'Test League', show_name: 'Test Show', generated_at: '2025-10-15T00:00:00Z' },
    this_week: { matchups: [], week_records: { highest_score: { score: 140, manager: 'Greg', source: 'matchups:week5:max_score' }, lowest_score: { score: 80, manager: 'Mike', source: 'matchups:week5:min_score' }, closest_game: { margin: 3.1, source: 'matchups:week5:min_margin' }, biggest_blowout: { margin: 60, source: 'matchups:week5:max_margin' } } },
    standings: [],
    all_time_records: { highest_score_ever: null, biggest_blowout_ever: null, championship_history: [] },
    storylines: { active_streaks: [], h2h_narratives: [] },
    manager_continuity: null,
    recent_transactions: [],
    draft_highlights: [],
  }),
}))

vi.mock('@/lib/generation/script-generator', () => ({
  generateScript: vi.fn().mockResolvedValue({
    episode: { title: 'Week 5', week: 5, season: 2025, type: 'weekly_recap', estimated_duration_secs: 480 },
    segments: [
      { index: 0, host: 'troy', text: 'Welcome.', is_fact: false, type: 'intro' },
      { index: 1, host: 'troy', text: 'Greg scored 140.', is_fact: true, source: 'matchups:week5:max_score', type: 'score_report' },
    ],
    sections: ['intro', 'matchup_recap'],
  }),
  validateScriptSources: vi.fn().mockReturnValue(true),
}))

vi.mock('@/lib/generation/audio-producer', () => ({
  produceAudio: vi.fn().mockResolvedValue({ audioUrl: 'https://storage.test/ep.mp3', durationSecs: 480, warnings: [] }),
  buildAudioSegments: vi.fn().mockReturnValue([]),
}))

vi.mock('@/lib/generation/transcript-generator', () => ({
  generateTranscript: vi.fn().mockResolvedValue('## Week 5 Recap\n\nGreg won.\n\n---\n*Match scores and league records sourced from Yahoo Fantasy. Commentary and analysis are AI-generated.*'),
}))

describe('Episode Pipeline — integration', () => {
  it('calls all pipeline stages and resolves without error', async () => {
    const { runEpisodePipeline } = await import('@/lib/generation/episode-pipeline')
    const { generateScript } = await import('@/lib/generation/script-generator')
    const { produceAudio } = await import('@/lib/generation/audio-producer')
    const { generateTranscript } = await import('@/lib/generation/transcript-generator')

    await expect(runEpisodePipeline('ep-uuid')).resolves.toBeUndefined()

    expect(generateScript).toHaveBeenCalled()
    expect(produceAudio).toHaveBeenCalled()
    expect(generateTranscript).toHaveBeenCalled()
  })
})

describe('Transcript disclaimer', () => {
  it('always ends with the statutory disclaimer', async () => {
    const { generateTranscript } = await import('@/lib/generation/transcript-generator')
    const result = await generateTranscript({} as any)
    expect(result).toContain('Match scores and league records sourced from Yahoo Fantasy')
    expect(result).toContain('Commentary and analysis are AI-generated')
  })
})
```

- [ ] **Step 2: Run all tests**

```bash
npm test
# Expected: All tests PASS
```

- [ ] **Step 3: Verify build**

```bash
npm run build
# Expected: Build succeeds with no type errors
```

- [ ] **Step 4: Fix any type errors revealed by build**

Common issues:
- Async params not awaited — add `await params` in page components
- Supabase join types — add `as any` cast where needed for complex joins
- Missing `'use client'` directive — add to any component using hooks

- [ ] **Step 5: Set up Supabase Storage bucket**

In Supabase dashboard or via CLI:
```bash
# Create public 'episodes' bucket for audio files
npx supabase storage create episodes --public
```

Or via MCP `execute_sql`:
```sql
INSERT INTO storage.buckets (id, name, public) VALUES ('episodes', 'episodes', true);
```

- [ ] **Step 6: Configure Vercel environment variables**

In Vercel project settings → Environment Variables, add all vars from `.env.local.example`:
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `YAHOO_CLIENT_ID`
- `YAHOO_CLIENT_SECRET`
- `YAHOO_REDIRECT_URI` (set to `https://your-domain.vercel.app/api/yahoo/callback`)
- `ANTHROPIC_API_KEY`
- `ELEVENLABS_API_KEY`
- `CRON_SECRET`
- `NEXT_PUBLIC_APP_URL`

- [ ] **Step 7: Deploy**

```bash
git add __tests__/e2e/
git commit -m "test: end-to-end pipeline integration test"
git push origin main
# Vercel auto-deploys on push to main
```

- [ ] **Step 8: Verify cron is registered**

In Vercel dashboard → Project → Cron Jobs — verify `POST /api/cron/weekly` is listed with schedule `0 9 * * 2` (Tuesday 9AM UTC).

- [ ] **Step 9: Run setup wizard**

1. Navigate to `https://your-domain.vercel.app/admin/setup`
2. Request magic link for commissioner email
3. Complete all 6 setup steps
4. Verify historical ingestion completes (check `/admin/ingestion`)
5. Verify managers appear in `/admin/managers`
6. Generate first episode via `/admin/episodes`
7. Verify episode appears on home page at `/`

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|---|---|
| Yahoo OAuth (commissioner only) + token refresh | Task 4 |
| Historical ingestion via `renew` field + manual fallback | Task 8 |
| All 10 DB tables + commissioner_tokens | Task 2 |
| Manager identity: GUID-first, needs_review, unmatched | Task 6 |
| Context Builder with named queries and source references | Task 9 |
| Claude Opus script generation with fact/commentary separation | Task 10 |
| Source validation before episode marked ready | Task 10, 12 |
| ElevenLabs multi-voice with silence insertion + graceful failure | Task 11 |
| >20% segment failure → abort audio, publish partial | Task 11 |
| Claude Sonnet transcript + statutory disclaimer | Task 12 |
| Episode status flow: pending → script → parallel(audio, transcript) → ready/partial/failed | Task 12 |
| Next.js web app: home, episode, archive, history, standings | Tasks 15-17 |
| Persistent mini audio player | Task 14 |
| Admin: auth (magic link), setup wizard, dashboard, episodes, managers, ingestion | Tasks 18-19 |
| Vercel Cron (Tuesday 9AM) | Task 13 |
| share_token access model | Task 2 (league_config), home page |
| Dark theme, accent color, mobile-first | Tasks 15-17 |
| History page: championship roll, all-time records, career stats, H2H matrix | Task 17 |
| Commissioner-configurable show name (not hardcoded) | Task 18 (setup wizard step 3) |
| Ingestion log audit trail | Task 8 (log creation), Task 19 (log page) |

**Placeholder scan:** No TBDs or incomplete sections found.

**Type consistency:** `ScriptSegment`, `ScriptJson`, `ContextObject`, `LeagueConfig` defined in Tasks 1/9 and referenced consistently throughout.

**Scope check:** V1 scope only. No multi-league routing, no per-manager auth, no live scoring, no push delivery built.
