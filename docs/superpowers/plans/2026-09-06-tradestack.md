# TradeStack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build TradeStack — a Whop-distributed Next.js app with 3 AI tools for home service businesses: Review Rocket (SMS), AdForge (ad copy), EstimateAI (estimates).

**Architecture:** Next.js App Router deployed to Vercel, embedded in Whop iframe. Whop handles all billing and user identity via JWT token headers. Supabase stores app data only. Claude API powers all 3 tools.

**Tech Stack:** Next.js 16+ App Router, `@whop/sdk`, Supabase (`@supabase/ssr`), Anthropic SDK, Twilio, Resend, `@react-pdf/renderer`, Vitest, Playwright, Tailwind CSS, shadcn/ui

## Global Constraints

- Next.js App Router — `params` is always `Promise<{...}>`, must be awaited in pages and route handlers
- `cookies()` from `next/headers` is async — must be awaited
- All Whop token verification happens server-side — never trust client-side claims about tier
- `getTier(headers)` is the single source of truth for tier — never re-implement inline
- Supabase queries use service role key server-side (no user sessions — Whop owns identity)
- `whop_user_id` (string) is the foreign key across all app data tables
- All API routes return `Response` objects, not `NextResponse`
- Claude model: `claude-sonnet-4-6`
- No `console.log` in production code — use `console.error` for errors only

---

## File Map

```
tradestack/
├── app/
│   ├── layout.tsx                          # Root layout, fonts, providers
│   ├── page.tsx                            # Entry: check onboarding → redirect
│   ├── onboarding/
│   │   └── page.tsx                        # Business profile setup
│   ├── dashboard/
│   │   └── page.tsx                        # Tool hub with tier-gated tool cards
│   ├── tools/
│   │   ├── review-rocket/
│   │   │   └── page.tsx                    # Review Rocket tool UI
│   │   ├── adforge/
│   │   │   └── page.tsx                    # AdForge tool UI
│   │   └── estimateai/
│   │       └── page.tsx                    # EstimateAI tool UI
│   ├── settings/
│   │   └── page.tsx                        # Business profile + Agency locations
│   └── api/
│       ├── ai/
│       │   ├── review/route.ts             # POST: generate SMS via Claude
│       │   ├── ad/route.ts                 # POST: generate ad copy via Claude
│       │   └── estimate/route.ts           # POST: generate estimate via Claude
│       ├── sms/
│       │   └── send/route.ts               # POST: send SMS via Twilio
│       ├── email/
│       │   └── send/route.ts               # POST: send estimate email via Resend
│       └── pdf/
│           └── route.ts                    # POST: generate estimate PDF
├── lib/
│   ├── whop.ts                             # Whop SDK client + getTier()
│   ├── supabase.ts                         # Supabase service role client
│   ├── anthropic.ts                        # Anthropic client
│   ├── twilio.ts                           # Twilio client
│   ├── resend.ts                           # Resend client
│   └── prompts/
│       ├── review.ts                       # Review SMS prompt builder
│       ├── ad.ts                           # Ad copy prompt builder
│       └── estimate.ts                     # Estimate prompt builder
├── components/
│   ├── tool-gate.tsx                       # Tier lock wrapper — shows upgrade CTA
│   ├── upgrade-cta.tsx                     # Upgrade prompt with Whop link
│   └── nav.tsx                             # Top navigation
├── types/
│   └── index.ts                            # Shared TypeScript types
├── supabase/
│   └── migrations/
│       ├── 001_schema.sql                  # Tables: businesses, locations, sms_log, estimates, ad_generations
│       └── 002_rls.sql                     # RLS: service role bypass only
├── proxy.ts                                # Middleware: iframe headers (X-Frame-Options)
├── next.config.ts
├── package.json
├── .env.local.example
└── vitest.config.ts
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `package.json`
- Create: `next.config.ts`
- Create: `proxy.ts`
- Create: `.env.local.example`
- Create: `vitest.config.ts`
- Create: `app/layout.tsx`
- Create: `app/globals.css`

**Interfaces:**
- Produces: runnable Next.js dev server with Whop SDK installed

- [ ] **Step 1: Scaffold project from Whop template**

```bash
pnpm create next-app@latest -e https://github.com/whopio/whop-nextjs-app-template tradestack
cd tradestack
```

- [ ] **Step 2: Install all additional dependencies**

```bash
pnpm add @supabase/supabase-js @anthropic-ai/sdk twilio resend @react-pdf/renderer zod
pnpm add -D vitest @vitejs/plugin-react jsdom @vitest/coverage-v8 playwright @playwright/test
```

- [ ] **Step 3: Create `.env.local.example`**

```bash
# .env.local.example
WHOP_API_KEY=
WHOP_EXPERIENCE_STARTER=
WHOP_EXPERIENCE_PRO=
WHOP_EXPERIENCE_AGENCY=

NEXT_PUBLIC_SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=

ANTHROPIC_API_KEY=

TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

RESEND_API_KEY=
RESEND_FROM_EMAIL=

NEXT_PUBLIC_APP_URL=http://localhost:3000
```

Copy to `.env.local` and fill in values before proceeding.

- [ ] **Step 4: Update `next.config.ts`** to allow iframe embedding

```ts
// next.config.ts
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options', value: 'ALLOWALL' },
          { key: 'Content-Security-Policy', value: "frame-ancestors 'self' https://whop.com https://*.whop.com" },
        ],
      },
    ]
  },
}

export default nextConfig
```

- [ ] **Step 5: Create `proxy.ts`** (middleware — no auth redirect, just iframe headers passthrough)

```ts
// proxy.ts
import { NextResponse, type NextRequest } from 'next/server'

export function proxy(request: NextRequest) {
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)'],
}
```

- [ ] **Step 6: Create `vitest.config.ts`**

```ts
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, '.') },
  },
})
```

- [ ] **Step 7: Verify dev server starts**

```bash
pnpm dev
```

Expected: `http://localhost:3000` loads without errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: scaffold TradeStack with Whop template and deps"
```

---

## Task 2: Types + Supabase Schema

**Files:**
- Create: `types/index.ts`
- Create: `supabase/migrations/001_schema.sql`
- Create: `supabase/migrations/002_rls.sql`

**Interfaces:**
- Produces: `Tier`, `Business`, `SmsLog`, `Estimate`, `AdGeneration` types used across all tasks

- [ ] **Step 1: Create `types/index.ts`**

```ts
// types/index.ts

export type Tier = 'agency' | 'pro' | 'starter' | 'none'

export type BusinessType = 'hvac' | 'plumbing' | 'auto' | 'roofing' | 'other'

export interface Business {
  id: string
  whop_user_id: string
  name: string
  type: BusinessType
  phone: string
  google_review_url: string
  starter_tool: 'review-rocket' | 'adforge' | 'estimateai' | null
  onboarded_at: string
  created_at: string
}

export interface Location {
  id: string
  whop_user_id: string
  name: string
  phone: string
  google_review_url: string
  created_at: string
}

export interface SmsLog {
  id: string
  whop_user_id: string
  customer_name: string
  customer_phone: string
  message: string
  twilio_sid: string | null
  status: 'queued' | 'sent' | 'delivered' | 'failed'
  location_id: string | null
  sent_at: string
}

export interface Estimate {
  id: string
  whop_user_id: string
  customer_name: string
  customer_email: string
  job_type: string
  scope: string
  line_items: LineItem[]
  total: number
  terms: string
  status: 'draft' | 'sent' | 'accepted' | 'declined'
  location_id: string | null
  sent_at: string | null
  created_at: string
}

export interface LineItem {
  description: string
  quantity: number
  unit_price: number
  amount: number
}

export interface AdGeneration {
  id: string
  whop_user_id: string
  inputs: AdInputs
  variations: AdVariation[]
  platform: 'facebook' | 'google'
  created_at: string
}

export interface AdInputs {
  business_type: string
  service: string
  audience: string
  hook: string
  tone: 'professional' | 'casual' | 'urgent'
  platform: 'facebook' | 'google'
}

export interface AdVariation {
  headline: string
  body: string
  cta: string
}
```

- [ ] **Step 2: Create `supabase/migrations/001_schema.sql`**

```sql
-- supabase/migrations/001_schema.sql

create table if not exists businesses (
  id uuid primary key default gen_random_uuid(),
  whop_user_id text unique not null,
  name text not null,
  type text not null check (type in ('hvac','plumbing','auto','roofing','other')),
  phone text not null,
  google_review_url text not null,
  starter_tool text check (starter_tool in ('review-rocket','adforge','estimateai')),
  onboarded_at timestamptz default now(),
  created_at timestamptz default now()
);

create table if not exists locations (
  id uuid primary key default gen_random_uuid(),
  whop_user_id text not null,
  name text not null,
  phone text not null,
  google_review_url text not null,
  created_at timestamptz default now()
);

create table if not exists sms_log (
  id uuid primary key default gen_random_uuid(),
  whop_user_id text not null,
  customer_name text not null,
  customer_phone text not null,
  message text not null,
  twilio_sid text,
  status text not null default 'queued' check (status in ('queued','sent','delivered','failed')),
  location_id uuid references locations(id) on delete set null,
  sent_at timestamptz default now()
);

create table if not exists estimates (
  id uuid primary key default gen_random_uuid(),
  whop_user_id text not null,
  customer_name text not null,
  customer_email text not null,
  job_type text not null,
  scope text not null,
  line_items jsonb not null default '[]',
  total numeric(10,2) not null default 0,
  terms text not null default '',
  status text not null default 'draft' check (status in ('draft','sent','accepted','declined')),
  location_id uuid references locations(id) on delete set null,
  sent_at timestamptz,
  created_at timestamptz default now()
);

create table if not exists ad_generations (
  id uuid primary key default gen_random_uuid(),
  whop_user_id text not null,
  inputs jsonb not null,
  variations jsonb not null default '[]',
  platform text not null check (platform in ('facebook','google')),
  created_at timestamptz default now()
);
```

- [ ] **Step 3: Create `supabase/migrations/002_rls.sql`**

```sql
-- supabase/migrations/002_rls.sql
-- All queries go through service role key (server-side only). Disable RLS.

alter table businesses disable row level security;
alter table locations disable row level security;
alter table sms_log disable row level security;
alter table estimates disable row level security;
alter table ad_generations disable row level security;
```

- [ ] **Step 4: Apply migrations**

```bash
# In Supabase dashboard → SQL editor, run 001_schema.sql then 002_rls.sql
# Or via CLI if supabase CLI is configured:
supabase db push
```

Expected: All 5 tables visible in Supabase table editor.

- [ ] **Step 5: Commit**

```bash
git add types/index.ts supabase/migrations/
git commit -m "feat: add TypeScript types and Supabase schema migrations"
```

---

## Task 3: Whop + Supabase + AI Clients

**Files:**
- Create: `lib/whop.ts`
- Create: `lib/supabase.ts`
- Create: `lib/anthropic.ts`
- Create: `lib/twilio.ts`
- Create: `lib/resend.ts`

**Interfaces:**
- Produces:
  - `getTier(headers: Headers): Promise<Tier>` — used by every page and API route
  - `getWhopUserId(headers: Headers): Promise<string>` — used by every API route
  - `supabase` — Supabase service role client
  - `anthropic` — Anthropic client
  - `twilioClient` — Twilio client
  - `resendClient` — Resend client

- [ ] **Step 1: Write failing test for `getTier`**

```ts
// lib/__tests__/whop.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock @whop/sdk
vi.mock('@whop/sdk', () => ({
  WhopServerSdk: vi.fn(() => ({
    verifyUserToken: vi.fn(),
    users: {
      checkAccess: vi.fn(),
    },
  })),
}))

describe('getTier', () => {
  it('returns "none" when token is invalid', async () => {
    const { getTier } = await import('../whop')
    const headers = new Headers({ 'x-whop-user-token': 'bad-token' })
    const tier = await getTier(headers)
    expect(tier).toBe('none')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pnpm vitest run lib/__tests__/whop.test.ts
```

Expected: FAIL — `getTier` not defined.

- [ ] **Step 3: Create `lib/whop.ts`**

```ts
// lib/whop.ts
import { WhopServerSdk } from '@whop/sdk'
import type { Tier } from '@/types'

const whop = WhopServerSdk({ apiKey: process.env.WHOP_API_KEY! })

export async function getWhopUserId(headers: Headers): Promise<string> {
  const { userId } = await whop.verifyUserToken({ headers })
  return userId
}

export async function getTier(headers: Headers): Promise<Tier> {
  try {
    const { userId } = await whop.verifyUserToken({ headers })

    // Check highest tier first
    const agencyCheck = await whop.users.checkAccess({
      userId,
      experienceId: process.env.WHOP_EXPERIENCE_AGENCY!,
    })
    if (agencyCheck.hasAccess) return 'agency'

    const proCheck = await whop.users.checkAccess({
      userId,
      experienceId: process.env.WHOP_EXPERIENCE_PRO!,
    })
    if (proCheck.hasAccess) return 'pro'

    const starterCheck = await whop.users.checkAccess({
      userId,
      experienceId: process.env.WHOP_EXPERIENCE_STARTER!,
    })
    if (starterCheck.hasAccess) return 'starter'

    return 'none'
  } catch {
    return 'none'
  }
}

export { whop }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pnpm vitest run lib/__tests__/whop.test.ts
```

Expected: PASS.

- [ ] **Step 5: Create `lib/supabase.ts`**

```ts
// lib/supabase.ts
import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
  { auth: { persistSession: false } }
)
```

- [ ] **Step 6: Create `lib/anthropic.ts`**

```ts
// lib/anthropic.ts
import Anthropic from '@anthropic-ai/sdk'

export const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY!,
})
```

- [ ] **Step 7: Create `lib/twilio.ts`**

```ts
// lib/twilio.ts
import twilio from 'twilio'

export const twilioClient = twilio(
  process.env.TWILIO_ACCOUNT_SID!,
  process.env.TWILIO_AUTH_TOKEN!
)

export const TWILIO_FROM = process.env.TWILIO_PHONE_NUMBER!
```

- [ ] **Step 8: Create `lib/resend.ts`**

```ts
// lib/resend.ts
import { Resend } from 'resend'

export const resendClient = new Resend(process.env.RESEND_API_KEY!)

export const FROM_EMAIL = process.env.RESEND_FROM_EMAIL!
```

- [ ] **Step 9: Commit**

```bash
git add lib/
git commit -m "feat: add Whop, Supabase, Anthropic, Twilio, Resend clients"
```

---

## Task 4: AI Prompt Builders

**Files:**
- Create: `lib/prompts/review.ts`
- Create: `lib/prompts/ad.ts`
- Create: `lib/prompts/estimate.ts`
- Create: `lib/__tests__/prompts.test.ts`

**Interfaces:**
- Consumes: `AdInputs`, `LineItem` from `@/types`
- Produces:
  - `buildReviewPrompt({ customerName, jobType, note, businessName, reviewUrl }): string`
  - `buildAdPrompt(inputs: AdInputs): string`
  - `buildEstimatePrompt({ customerName, jobType, scope, materials, laborHours, markupPercent, businessName }): string`

- [ ] **Step 1: Write failing tests**

```ts
// lib/__tests__/prompts.test.ts
import { describe, it, expect } from 'vitest'
import { buildReviewPrompt } from '../prompts/review'
import { buildAdPrompt } from '../prompts/ad'
import { buildEstimatePrompt } from '../prompts/estimate'

describe('buildReviewPrompt', () => {
  it('includes customer name, job type, and review URL', () => {
    const prompt = buildReviewPrompt({
      customerName: 'John',
      jobType: 'furnace repair',
      note: 'very happy, heat is working great',
      businessName: 'ABC HVAC',
      reviewUrl: 'https://g.page/r/abc',
    })
    expect(prompt).toContain('John')
    expect(prompt).toContain('furnace repair')
    expect(prompt).toContain('https://g.page/r/abc')
  })
})

describe('buildAdPrompt', () => {
  it('includes platform, service, and tone', () => {
    const prompt = buildAdPrompt({
      business_type: 'hvac',
      service: 'AC tune-up',
      audience: 'homeowners',
      hook: '$49 special',
      tone: 'urgent',
      platform: 'facebook',
    })
    expect(prompt).toContain('AC tune-up')
    expect(prompt).toContain('facebook')
    expect(prompt).toContain('urgent')
  })
})

describe('buildEstimatePrompt', () => {
  it('includes customer name, job type, and markup', () => {
    const prompt = buildEstimatePrompt({
      customerName: 'Jane Smith',
      jobType: 'furnace replacement',
      scope: 'Remove old unit, install new 96% efficiency furnace',
      materials: 'Carrier 96% furnace, flue pipe, fittings',
      laborHours: 8,
      markupPercent: 20,
      businessName: 'ABC HVAC',
    })
    expect(prompt).toContain('Jane Smith')
    expect(prompt).toContain('furnace replacement')
    expect(prompt).toContain('20')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pnpm vitest run lib/__tests__/prompts.test.ts
```

Expected: FAIL — modules not found.

- [ ] **Step 3: Create `lib/prompts/review.ts`**

```ts
// lib/prompts/review.ts

interface ReviewPromptInput {
  customerName: string
  jobType: string
  note: string
  businessName: string
  reviewUrl: string
}

export function buildReviewPrompt(input: ReviewPromptInput): string {
  return `You are writing a short, friendly, personalized SMS message from a home service business to a satisfied customer asking them to leave a Google review.

Business name: ${input.businessName}
Customer first name: ${input.customerName}
Job completed: ${input.jobType}
Additional context: ${input.note}
Google review link: ${input.reviewUrl}

Write a single SMS message. Requirements:
- Maximum 160 characters
- Sound natural and human, not robotic
- Include the customer's first name
- Reference the job briefly
- Include the review link
- Do not use emojis
- Do not use formal language like "Dear" or "Sincerely"
- Output ONLY the SMS text, nothing else`
}
```

- [ ] **Step 4: Create `lib/prompts/ad.ts`**

```ts
// lib/prompts/ad.ts
import type { AdInputs, AdVariation } from '@/types'

export function buildAdPrompt(inputs: AdInputs): string {
  return `You are an expert direct-response copywriter for local home service businesses.

Platform: ${inputs.platform} ads
Business type: ${inputs.business_type}
Service to promote: ${inputs.service}
Target audience: ${inputs.audience}
Offer / hook: ${inputs.hook}
Tone: ${inputs.tone}

Write exactly 3 ad copy variations. Each must have:
- headline: max 40 characters (for ${inputs.platform === 'google' ? 'Google' : 'Facebook'})
- body: max 125 characters, punchy, benefit-focused
- cta: max 20 characters (e.g. "Book Now", "Get a Free Quote")

Return ONLY a JSON array in this exact format, no markdown, no extra text:
[
  { "headline": "...", "body": "...", "cta": "..." },
  { "headline": "...", "body": "...", "cta": "..." },
  { "headline": "...", "body": "...", "cta": "..." }
]`
}

export function parseAdVariations(raw: string): AdVariation[] {
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed) || parsed.length !== 3) throw new Error('Invalid format')
    return parsed as AdVariation[]
  } catch {
    throw new Error('Claude returned malformed ad variations')
  }
}
```

- [ ] **Step 5: Create `lib/prompts/estimate.ts`**

```ts
// lib/prompts/estimate.ts
import type { LineItem } from '@/types'

interface EstimatePromptInput {
  customerName: string
  jobType: string
  scope: string
  materials: string
  laborHours: number
  markupPercent: number
  businessName: string
}

interface EstimateOutput {
  intro: string
  line_items: LineItem[]
  subtotal: number
  markup_amount: number
  total: number
  terms: string
}

export function buildEstimatePrompt(input: EstimatePromptInput): string {
  return `You are generating a professional customer-facing estimate for a home service business.

Business: ${input.businessName}
Customer: ${input.customerName}
Job type: ${input.jobType}
Scope of work: ${input.scope}
Materials needed: ${input.materials}
Labor hours: ${input.laborHours}
Markup percentage: ${input.markupPercent}%

Generate a professional estimate. Return ONLY valid JSON in this exact format, no markdown:
{
  "intro": "A 2-3 sentence professional intro addressed to the customer about the job",
  "line_items": [
    { "description": "Labor (X hours @ $Y/hr)", "quantity": 1, "unit_price": 0, "amount": 0 },
    { "description": "Materials - [item name]", "quantity": 1, "unit_price": 0, "amount": 0 }
  ],
  "subtotal": 0,
  "markup_amount": 0,
  "total": 0,
  "terms": "Net 30. 50% deposit required before work begins. Estimate valid for 30 days."
}

Rules:
- Use realistic market rates for labor ($85-$120/hr for trades)
- Subtotal = sum of line_items amounts
- markup_amount = subtotal * ${input.markupPercent} / 100
- total = subtotal + markup_amount
- All numbers must be numeric (not strings)
- Do not include tax (customer handles that)`
}

export function parseEstimateOutput(raw: string): EstimateOutput {
  try {
    return JSON.parse(raw) as EstimateOutput
  } catch {
    throw new Error('Claude returned malformed estimate JSON')
  }
}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pnpm vitest run lib/__tests__/prompts.test.ts
```

Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add lib/prompts/ lib/__tests__/prompts.test.ts
git commit -m "feat: add AI prompt builders for Review Rocket, AdForge, EstimateAI"
```

---

## Task 5: API Routes (AI + SMS + Email)

**Files:**
- Create: `app/api/ai/review/route.ts`
- Create: `app/api/ai/ad/route.ts`
- Create: `app/api/ai/estimate/route.ts`
- Create: `app/api/sms/send/route.ts`
- Create: `app/api/email/send/route.ts`
- Create: `app/api/pdf/route.ts`

**Interfaces:**
- Consumes: `getTier`, `getWhopUserId` from `@/lib/whop`, `anthropic` from `@/lib/anthropic`, `supabase` from `@/lib/supabase`, all prompt builders
- Produces: REST endpoints consumed by tool UI pages

- [ ] **Step 1: Create `app/api/ai/review/route.ts`**

```ts
// app/api/ai/review/route.ts
import { NextRequest } from 'next/server'
import { getTier, getWhopUserId } from '@/lib/whop'
import { anthropic } from '@/lib/anthropic'
import { buildReviewPrompt } from '@/lib/prompts/review'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  customerName: z.string().min(1),
  jobType: z.string().min(1),
  note: z.string(),
  businessName: z.string().min(1),
  reviewUrl: z.string().url(),
})

export async function POST(request: NextRequest) {
  const tier = await getTier(request.headers)
  if (tier === 'none' || tier === 'starter') {
    return Response.json({ error: 'Upgrade to Pro or Agency to use Review Rocket' }, { status: 403 })
  }

  const userId = await getWhopUserId(request.headers)

  // Check monthly SMS limit
  const thisMonth = new Date()
  thisMonth.setDate(1)
  thisMonth.setHours(0, 0, 0, 0)
  const { count } = await supabase
    .from('sms_log')
    .select('*', { count: 'exact', head: true })
    .eq('whop_user_id', userId)
    .gte('sent_at', thisMonth.toISOString())

  const limit = tier === 'agency' ? 1000 : 200
  if ((count ?? 0) >= limit) {
    return Response.json({ error: `Monthly SMS limit of ${limit} reached. Upgrade to send more.` }, { status: 429 })
  }

  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input', details: parsed.error.flatten() }, { status: 400 })
  }

  const prompt = buildReviewPrompt(parsed.data)

  const message = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 200,
    messages: [{ role: 'user', content: prompt }],
  })

  const smsText = (message.content[0] as { type: string; text: string }).text.trim()

  return Response.json({ sms: smsText })
}
```

- [ ] **Step 2: Create `app/api/ai/ad/route.ts`**

```ts
// app/api/ai/ad/route.ts
import { NextRequest } from 'next/server'
import { getTier, getWhopUserId } from '@/lib/whop'
import { anthropic } from '@/lib/anthropic'
import { buildAdPrompt, parseAdVariations } from '@/lib/prompts/ad'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  business_type: z.string().min(1),
  service: z.string().min(1),
  audience: z.string().min(1),
  hook: z.string().min(1),
  tone: z.enum(['professional', 'casual', 'urgent']),
  platform: z.enum(['facebook', 'google']),
})

export async function POST(request: NextRequest) {
  const tier = await getTier(request.headers)
  if (tier === 'none') {
    return Response.json({ error: 'No active subscription' }, { status: 403 })
  }

  const userId = await getWhopUserId(request.headers)

  // Check generation limit for starter (100/mo)
  if (tier === 'starter') {
    const thisMonth = new Date()
    thisMonth.setDate(1)
    thisMonth.setHours(0, 0, 0, 0)
    const { count } = await supabase
      .from('ad_generations')
      .select('*', { count: 'exact', head: true })
      .eq('whop_user_id', userId)
      .gte('created_at', thisMonth.toISOString())

    if ((count ?? 0) >= 100) {
      return Response.json({ error: 'Monthly generation limit of 100 reached. Upgrade to Pro for unlimited.' }, { status: 429 })
    }
  }

  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input', details: parsed.error.flatten() }, { status: 400 })
  }

  const prompt = buildAdPrompt(parsed.data)

  const message = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 600,
    messages: [{ role: 'user', content: prompt }],
  })

  const raw = (message.content[0] as { type: string; text: string }).text.trim()
  const variations = parseAdVariations(raw)

  // Save to history
  await supabase.from('ad_generations').insert({
    whop_user_id: userId,
    inputs: parsed.data,
    variations,
    platform: parsed.data.platform,
  })

  return Response.json({ variations })
}
```

- [ ] **Step 3: Create `app/api/ai/estimate/route.ts`**

```ts
// app/api/ai/estimate/route.ts
import { NextRequest } from 'next/server'
import { getTier, getWhopUserId } from '@/lib/whop'
import { anthropic } from '@/lib/anthropic'
import { buildEstimatePrompt, parseEstimateOutput } from '@/lib/prompts/estimate'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  customerName: z.string().min(1),
  customerEmail: z.string().email(),
  jobType: z.string().min(1),
  scope: z.string().min(1),
  materials: z.string().min(1),
  laborHours: z.number().positive(),
  markupPercent: z.number().min(0).max(100),
  businessName: z.string().min(1),
})

export async function POST(request: NextRequest) {
  const tier = await getTier(request.headers)
  if (tier === 'none') {
    return Response.json({ error: 'No active subscription' }, { status: 403 })
  }

  const userId = await getWhopUserId(request.headers)
  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input', details: parsed.error.flatten() }, { status: 400 })
  }

  const prompt = buildEstimatePrompt(parsed.data)

  const message = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1000,
    messages: [{ role: 'user', content: prompt }],
  })

  const raw = (message.content[0] as { type: string; text: string }).text.trim()
  const estimateData = parseEstimateOutput(raw)

  // Save draft estimate
  const { data: estimate } = await supabase
    .from('estimates')
    .insert({
      whop_user_id: userId,
      customer_name: parsed.data.customerName,
      customer_email: parsed.data.customerEmail,
      job_type: parsed.data.jobType,
      scope: parsed.data.scope,
      line_items: estimateData.line_items,
      total: estimateData.total,
      terms: estimateData.terms,
      status: 'draft',
    })
    .select('id')
    .single()

  return Response.json({ estimate: estimateData, estimateId: estimate?.id })
}
```

- [ ] **Step 4: Create `app/api/sms/send/route.ts`**

```ts
// app/api/sms/send/route.ts
import { NextRequest } from 'next/server'
import { getWhopUserId } from '@/lib/whop'
import { twilioClient, TWILIO_FROM } from '@/lib/twilio'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  customerName: z.string().min(1),
  customerPhone: z.string().min(10),
  message: z.string().min(1).max(160),
  locationId: z.string().uuid().optional(),
})

export async function POST(request: NextRequest) {
  const userId = await getWhopUserId(request.headers)
  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input' }, { status: 400 })
  }

  // Insert log entry as queued
  const { data: logEntry } = await supabase
    .from('sms_log')
    .insert({
      whop_user_id: userId,
      customer_name: parsed.data.customerName,
      customer_phone: parsed.data.customerPhone,
      message: parsed.data.message,
      status: 'queued',
      location_id: parsed.data.locationId ?? null,
    })
    .select('id')
    .single()

  try {
    const msg = await twilioClient.messages.create({
      body: parsed.data.message,
      from: TWILIO_FROM,
      to: parsed.data.customerPhone,
    })

    await supabase
      .from('sms_log')
      .update({ status: 'sent', twilio_sid: msg.sid })
      .eq('id', logEntry!.id)

    return Response.json({ success: true, sid: msg.sid })
  } catch (err) {
    await supabase
      .from('sms_log')
      .update({ status: 'failed' })
      .eq('id', logEntry!.id)

    console.error('Twilio send failed:', err)
    return Response.json({ error: 'Failed to send SMS' }, { status: 500 })
  }
}
```

- [ ] **Step 5: Create `app/api/email/send/route.ts`**

```ts
// app/api/email/send/route.ts
import { NextRequest } from 'next/server'
import { getTier, getWhopUserId } from '@/lib/whop'
import { resendClient, FROM_EMAIL } from '@/lib/resend'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  estimateId: z.string().uuid(),
  businessName: z.string().min(1),
})

export async function POST(request: NextRequest) {
  const tier = await getTier(request.headers)
  const userId = await getWhopUserId(request.headers)
  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input' }, { status: 400 })
  }

  const { data: estimate } = await supabase
    .from('estimates')
    .select('*')
    .eq('id', parsed.data.estimateId)
    .eq('whop_user_id', userId)
    .single()

  if (!estimate) {
    return Response.json({ error: 'Estimate not found' }, { status: 404 })
  }

  const fromName = tier === 'agency' ? parsed.data.businessName : `${parsed.data.businessName} via TradeStack`

  const lineItemsHtml = (estimate.line_items as Array<{ description: string; amount: number }>)
    .map((item) => `<tr><td>${item.description}</td><td style="text-align:right">$${item.amount.toFixed(2)}</td></tr>`)
    .join('')

  const html = `
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:24px">
      <h2 style="margin-bottom:4px">${parsed.data.businessName}</h2>
      <p style="color:#666;margin-top:0">Estimate for ${estimate.customer_name}</p>
      <hr/>
      <p>${estimate.scope}</p>
      <table style="width:100%;border-collapse:collapse;margin:16px 0">
        <thead><tr style="background:#f5f5f5"><th style="text-align:left;padding:8px">Description</th><th style="text-align:right;padding:8px">Amount</th></tr></thead>
        <tbody>${lineItemsHtml}</tbody>
        <tfoot>
          <tr><td style="padding:8px;font-weight:bold">Total</td><td style="text-align:right;padding:8px;font-weight:bold">$${Number(estimate.total).toFixed(2)}</td></tr>
        </tfoot>
      </table>
      <p style="color:#666;font-size:14px">${estimate.terms}</p>
    </div>`

  try {
    await resendClient.emails.send({
      from: `${fromName} <${FROM_EMAIL}>`,
      to: estimate.customer_email,
      subject: `Estimate for ${estimate.job_type} — ${parsed.data.businessName}`,
      html,
    })

    await supabase
      .from('estimates')
      .update({ status: 'sent', sent_at: new Date().toISOString() })
      .eq('id', estimate.id)

    return Response.json({ success: true })
  } catch (err) {
    console.error('Resend failed, retrying once:', err)
    try {
      await resendClient.emails.send({
        from: `${fromName} <${FROM_EMAIL}>`,
        to: estimate.customer_email,
        subject: `Estimate for ${estimate.job_type} — ${parsed.data.businessName}`,
        html,
      })
      await supabase
        .from('estimates')
        .update({ status: 'sent', sent_at: new Date().toISOString() })
        .eq('id', estimate.id)
      return Response.json({ success: true })
    } catch (err2) {
      console.error('Resend retry failed:', err2)
      return Response.json({ error: 'Failed to send email after retry' }, { status: 500 })
    }
  }
}
```

- [ ] **Step 6: Commit**

```bash
git add app/api/
git commit -m "feat: add AI, SMS, and email API routes"
```

---

## Task 6: Layout + Navigation + Tool Gate

**Files:**
- Create: `app/layout.tsx`
- Create: `components/nav.tsx`
- Create: `components/tool-gate.tsx`
- Create: `components/upgrade-cta.tsx`

**Interfaces:**
- Consumes: `Tier` from `@/types`
- Produces: `<ToolGate tier={tier} required={['pro','agency']}>{children}</ToolGate>` — used by all tool pages

- [ ] **Step 1: Create `app/layout.tsx`**

```tsx
// app/layout.tsx
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'TradeStack',
  description: 'AI tools built for the trades',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
      </body>
    </html>
  )
}
```

- [ ] **Step 2: Create `components/upgrade-cta.tsx`**

```tsx
// components/upgrade-cta.tsx

interface UpgradeCTAProps {
  message: string
  whopUpgradeUrl: string
}

export function UpgradeCTA({ message, whopUpgradeUrl }: UpgradeCTAProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-8 text-center border-2 border-dashed border-gray-200 rounded-xl">
      <div className="text-4xl mb-4">🔒</div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">Upgrade Required</h3>
      <p className="text-gray-500 mb-6 max-w-sm">{message}</p>
      <a
        href={whopUpgradeUrl}
        target="_top"
        className="bg-black text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-800 transition-colors"
      >
        Upgrade on Whop
      </a>
    </div>
  )
}
```

- [ ] **Step 3: Create `components/tool-gate.tsx`**

```tsx
// components/tool-gate.tsx
import type { Tier } from '@/types'
import { UpgradeCTA } from './upgrade-cta'

const TIER_RANK: Record<Tier, number> = { none: 0, starter: 1, pro: 2, agency: 3 }

interface ToolGateProps {
  tier: Tier
  required: Tier[]
  upgradeMessage: string
  children: React.ReactNode
}

export function ToolGate({ tier, required, upgradeMessage, children }: ToolGateProps) {
  const userRank = TIER_RANK[tier]
  const hasAccess = required.some((r) => userRank >= TIER_RANK[r])

  if (!hasAccess) {
    return (
      <UpgradeCTA
        message={upgradeMessage}
        whopUpgradeUrl="https://whop.com/tradestack"
      />
    )
  }

  return <>{children}</>
}
```

- [ ] **Step 4: Create `components/nav.tsx`**

```tsx
// components/nav.tsx
import Link from 'next/link'
import type { Tier } from '@/types'

interface NavProps {
  tier: Tier
}

const TIER_LABELS: Record<Tier, string> = {
  agency: 'Agency',
  pro: 'Pro',
  starter: 'Starter',
  none: 'No Plan',
}

const TIER_COLORS: Record<Tier, string> = {
  agency: 'bg-purple-100 text-purple-800',
  pro: 'bg-blue-100 text-blue-800',
  starter: 'bg-gray-100 text-gray-700',
  none: 'bg-red-100 text-red-700',
}

export function Nav({ tier }: NavProps) {
  return (
    <nav className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
      <div className="flex items-center gap-8">
        <Link href="/dashboard" className="font-bold text-lg tracking-tight">
          TradeStack
        </Link>
        <div className="flex items-center gap-4 text-sm">
          <Link href="/tools/review-rocket" className="text-gray-600 hover:text-gray-900">Review Rocket</Link>
          <Link href="/tools/adforge" className="text-gray-600 hover:text-gray-900">AdForge</Link>
          <Link href="/tools/estimateai" className="text-gray-600 hover:text-gray-900">EstimateAI</Link>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${TIER_COLORS[tier]}`}>
          {TIER_LABELS[tier]}
        </span>
        <Link href="/settings" className="text-sm text-gray-500 hover:text-gray-900">Settings</Link>
      </div>
    </nav>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add app/layout.tsx components/
git commit -m "feat: add layout, nav, and tier gate components"
```

---

## Task 7: Entry Point + Onboarding

**Files:**
- Create: `app/page.tsx`
- Create: `app/onboarding/page.tsx`
- Create: `app/dashboard/page.tsx`

**Interfaces:**
- Consumes: `getTier`, `getWhopUserId` from `@/lib/whop`, `supabase` from `@/lib/supabase`
- Produces: onboarded user lands on `/dashboard` with `tier` available

- [ ] **Step 1: Create `app/page.tsx`** (entry — redirect based on onboarding status)

```tsx
// app/page.tsx
import { headers } from 'next/headers'
import { redirect } from 'next/navigation'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'

export default async function EntryPage() {
  const h = await headers()
  const tier = await getTier(h)

  if (tier === 'none') {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-2">Welcome to TradeStack</h1>
          <p className="text-gray-500">Purchase a plan on Whop to get started.</p>
          <a
            href="https://whop.com/tradestack"
            target="_top"
            className="mt-6 inline-block bg-black text-white px-6 py-3 rounded-lg font-medium"
          >
            Get Started on Whop
          </a>
        </div>
      </div>
    )
  }

  const userId = await getWhopUserId(h)
  const { data: business } = await supabase
    .from('businesses')
    .select('id')
    .eq('whop_user_id', userId)
    .single()

  if (!business) {
    redirect('/onboarding')
  }

  redirect('/dashboard')
}
```

- [ ] **Step 2: Create `app/onboarding/page.tsx`**

```tsx
// app/onboarding/page.tsx
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

const BUSINESS_TYPES = ['hvac', 'plumbing', 'auto', 'roofing', 'other'] as const
const TOOLS = [
  { id: 'review-rocket', label: 'Review Rocket', description: 'Send SMS to get Google reviews' },
  { id: 'adforge', label: 'AdForge', description: 'Generate Facebook & Google ad copy' },
  { id: 'estimateai', label: 'EstimateAI', description: 'Create and send professional estimates' },
] as const

export default function OnboardingPage() {
  const router = useRouter()
  const [step, setStep] = useState<'profile' | 'tool'>('profile')
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState({
    name: '',
    type: 'hvac' as string,
    phone: '',
    google_review_url: '',
    starter_tool: '' as string,
  })

  async function handleProfileSubmit(e: React.FormEvent) {
    e.preventDefault()
    // If starter, pick tool next — otherwise save directly
    // We don't know tier client-side, so always show tool picker (server action will ignore if not starter)
    setStep('tool')
  }

  async function handleComplete(tool?: string) {
    setLoading(true)
    const res = await fetch('/api/onboarding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, starter_tool: tool ?? null }),
    })
    if (res.ok) {
      router.push('/dashboard')
    } else {
      setLoading(false)
    }
  }

  if (step === 'tool') {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <div className="max-w-md w-full">
          <h1 className="text-2xl font-bold mb-2">Which tool do you want first?</h1>
          <p className="text-gray-500 mb-8">Starter plan includes 1 tool. You can upgrade anytime to unlock all three.</p>
          <div className="space-y-3">
            {TOOLS.map((tool) => (
              <button
                key={tool.id}
                onClick={() => handleComplete(tool.id)}
                disabled={loading}
                className="w-full text-left p-4 border-2 border-gray-200 rounded-xl hover:border-black transition-colors"
              >
                <div className="font-semibold">{tool.label}</div>
                <div className="text-sm text-gray-500">{tool.description}</div>
              </button>
            ))}
          </div>
          <button
            onClick={() => handleComplete()}
            className="mt-4 text-sm text-gray-400 underline"
          >
            Skip — I have Pro or Agency
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-8">
      <div className="max-w-md w-full">
        <h1 className="text-2xl font-bold mb-2">Set up your business</h1>
        <p className="text-gray-500 mb-8">Takes 60 seconds. You can edit this anytime.</p>
        <form onSubmit={handleProfileSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Business name</label>
            <input
              required
              className="w-full border border-gray-200 rounded-lg px-3 py-2"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Business type</label>
            <select
              className="w-full border border-gray-200 rounded-lg px-3 py-2"
              value={form.type}
              onChange={(e) => setForm({ ...form, type: e.target.value })}
            >
              {BUSINESS_TYPES.map((t) => (
                <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Business phone</label>
            <input
              required
              type="tel"
              className="w-full border border-gray-200 rounded-lg px-3 py-2"
              value={form.phone}
              onChange={(e) => setForm({ ...form, phone: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Google review link</label>
            <input
              required
              type="url"
              placeholder="https://g.page/r/..."
              className="w-full border border-gray-200 rounded-lg px-3 py-2"
              value={form.google_review_url}
              onChange={(e) => setForm({ ...form, google_review_url: e.target.value })}
            />
          </div>
          <button
            type="submit"
            className="w-full bg-black text-white py-3 rounded-lg font-medium"
          >
            Continue
          </button>
        </form>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create `app/api/onboarding/route.ts`**

```ts
// app/api/onboarding/route.ts
import { NextRequest } from 'next/server'
import { getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  name: z.string().min(1),
  type: z.enum(['hvac', 'plumbing', 'auto', 'roofing', 'other']),
  phone: z.string().min(10),
  google_review_url: z.string().url(),
  starter_tool: z.enum(['review-rocket', 'adforge', 'estimateai']).nullable(),
})

export async function POST(request: NextRequest) {
  const userId = await getWhopUserId(request.headers)
  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input' }, { status: 400 })
  }

  await supabase.from('businesses').upsert(
    {
      whop_user_id: userId,
      name: parsed.data.name,
      type: parsed.data.type,
      phone: parsed.data.phone,
      google_review_url: parsed.data.google_review_url,
      starter_tool: parsed.data.starter_tool,
      onboarded_at: new Date().toISOString(),
    },
    { onConflict: 'whop_user_id' }
  )

  return Response.json({ success: true })
}
```

- [ ] **Step 4: Create `app/dashboard/page.tsx`**

```tsx
// app/dashboard/page.tsx
import { headers } from 'next/headers'
import Link from 'next/link'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { Nav } from '@/components/nav'
import type { Tier } from '@/types'

const TOOLS = [
  {
    id: 'review-rocket',
    label: 'Review Rocket',
    description: 'Send personalized SMS to customers → get Google reviews',
    href: '/tools/review-rocket',
    requiredTiers: ['pro', 'agency'] as Tier[],
    emoji: '⭐',
  },
  {
    id: 'adforge',
    label: 'AdForge',
    description: 'Generate 3 ad copy variations for Facebook or Google',
    href: '/tools/adforge',
    requiredTiers: ['starter', 'pro', 'agency'] as Tier[],
    emoji: '📣',
  },
  {
    id: 'estimateai',
    label: 'EstimateAI',
    description: 'Create professional estimates and email them instantly',
    href: '/tools/estimateai',
    requiredTiers: ['starter', 'pro', 'agency'] as Tier[],
    emoji: '📋',
  },
]

const TIER_RANK: Record<Tier, number> = { none: 0, starter: 1, pro: 2, agency: 3 }

export default async function DashboardPage() {
  const h = await headers()
  const tier = await getTier(h)
  const userId = await getWhopUserId(h)

  const { data: business } = await supabase
    .from('businesses')
    .select('name, starter_tool')
    .eq('whop_user_id', userId)
    .single()

  return (
    <div className="min-h-screen bg-gray-50">
      <Nav tier={tier} />
      <div className="max-w-4xl mx-auto p-8">
        <h1 className="text-2xl font-bold mb-1">
          {business?.name ? `Welcome, ${business.name}` : 'Dashboard'}
        </h1>
        <p className="text-gray-500 mb-8">Your AI tools for the trades.</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {TOOLS.map((tool) => {
            const hasAccess =
              tier !== 'none' &&
              (tier !== 'starter' || business?.starter_tool === tool.id || tool.requiredTiers.includes('starter'))
            const locked = !hasAccess

            return (
              <Link
                key={tool.id}
                href={locked ? '#' : tool.href}
                className={`block p-6 bg-white rounded-xl border-2 transition-all ${
                  locked
                    ? 'border-gray-100 opacity-60 cursor-default'
                    : 'border-gray-200 hover:border-black'
                }`}
              >
                <div className="text-3xl mb-3">{tool.emoji}</div>
                <div className="font-semibold mb-1">{tool.label}</div>
                <div className="text-sm text-gray-500">{tool.description}</div>
                {locked && (
                  <div className="mt-3 text-xs text-gray-400">🔒 Upgrade to unlock</div>
                )}
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add app/page.tsx app/onboarding/ app/dashboard/ app/api/onboarding/
git commit -m "feat: add entry point, onboarding flow, and dashboard"
```

---

## Task 8: Review Rocket Tool UI

**Files:**
- Create: `app/tools/review-rocket/page.tsx`

**Interfaces:**
- Consumes: `POST /api/ai/review`, `POST /api/sms/send`, `getTier`, `getWhopUserId`, `supabase`

- [ ] **Step 1: Create `app/tools/review-rocket/page.tsx`**

```tsx
// app/tools/review-rocket/page.tsx
import { headers } from 'next/headers'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { Nav } from '@/components/nav'
import { ToolGate } from '@/components/tool-gate'
import { ReviewRocketClient } from './client'
import type { SmsLog } from '@/types'

export default async function ReviewRocketPage() {
  const h = await headers()
  const tier = await getTier(h)
  const userId = await getWhopUserId(h)

  const { data: business } = await supabase
    .from('businesses')
    .select('name, google_review_url')
    .eq('whop_user_id', userId)
    .single()

  const { data: history } = await supabase
    .from('sms_log')
    .select('*')
    .eq('whop_user_id', userId)
    .order('sent_at', { ascending: false })
    .limit(20)

  return (
    <div className="min-h-screen bg-gray-50">
      <Nav tier={tier} />
      <div className="max-w-3xl mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-1">⭐ Review Rocket</h1>
          <p className="text-gray-500">Send a personalized SMS to get a Google review.</p>
        </div>
        <ToolGate
          tier={tier}
          required={['pro', 'agency']}
          upgradeMessage="Review Rocket is included in Pro and Agency plans. Upgrade to start collecting reviews automatically."
        >
          <ReviewRocketClient
            businessName={business?.name ?? ''}
            reviewUrl={business?.google_review_url ?? ''}
            history={(history as SmsLog[]) ?? []}
          />
        </ToolGate>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `app/tools/review-rocket/client.tsx`**

```tsx
// app/tools/review-rocket/client.tsx
'use client'

import { useState } from 'react'
import type { SmsLog } from '@/types'

interface Props {
  businessName: string
  reviewUrl: string
  history: SmsLog[]
}

export function ReviewRocketClient({ businessName, reviewUrl, history: initialHistory }: Props) {
  const [tab, setTab] = useState<'send' | 'history'>('send')
  const [form, setForm] = useState({ customerName: '', customerPhone: '', jobType: '', note: '' })
  const [preview, setPreview] = useState('')
  const [loading, setLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState(initialHistory)
  const [sent, setSent] = useState(false)

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    setPreview('')

    const res = await fetch('/api/ai/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, businessName, reviewUrl }),
    })

    const data = await res.json()
    if (!res.ok) {
      setError(data.error ?? 'Failed to generate SMS')
    } else {
      setPreview(data.sms)
    }
    setLoading(false)
  }

  async function handleSend() {
    setSending(true)
    setError('')

    const res = await fetch('/api/sms/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customerName: form.customerName,
        customerPhone: form.customerPhone,
        message: preview,
      }),
    })

    const data = await res.json()
    if (!res.ok) {
      setError(data.error ?? 'Failed to send SMS')
    } else {
      setSent(true)
      setForm({ customerName: '', customerPhone: '', jobType: '', note: '' })
      setPreview('')
      setTimeout(() => setSent(false), 3000)
    }
    setSending(false)
  }

  return (
    <div>
      <div className="flex gap-4 mb-6 border-b border-gray-200">
        {(['send', 'history'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`pb-3 text-sm font-medium capitalize border-b-2 transition-colors ${
              tab === t ? 'border-black text-black' : 'border-transparent text-gray-400'
            }`}
          >
            {t === 'history' ? `History (${history.length})` : 'Send Review Request'}
          </button>
        ))}
      </div>

      {tab === 'send' && (
        <div className="space-y-6">
          <form onSubmit={handleGenerate} className="bg-white p-6 rounded-xl border border-gray-200 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Customer name</label>
                <input
                  required
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.customerName}
                  onChange={(e) => setForm({ ...form, customerName: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Customer phone</label>
                <input
                  required
                  type="tel"
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.customerPhone}
                  onChange={(e) => setForm({ ...form, customerPhone: e.target.value })}
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Job type</label>
              <input
                required
                placeholder="e.g. AC tune-up, pipe repair, oil change"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.jobType}
                onChange={(e) => setForm({ ...form, jobType: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Quick note (optional)</label>
              <input
                placeholder="e.g. customer was very happy, problem solved quickly"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.note}
                onChange={(e) => setForm({ ...form, note: e.target.value })}
              />
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white py-2.5 rounded-lg text-sm font-medium disabled:opacity-50"
            >
              {loading ? 'Generating...' : 'Generate SMS'}
            </button>
          </form>

          {preview && (
            <div className="bg-white p-6 rounded-xl border border-gray-200">
              <label className="block text-sm font-medium mb-2">Preview — edit if needed</label>
              <textarea
                rows={4}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={preview}
                onChange={(e) => setPreview(e.target.value)}
              />
              <p className="text-xs text-gray-400 mt-1">{preview.length}/160 characters</p>
              <button
                onClick={handleSend}
                disabled={sending || sent}
                className="mt-3 w-full bg-green-600 text-white py-2.5 rounded-lg text-sm font-medium disabled:opacity-50"
              >
                {sent ? '✓ Sent!' : sending ? 'Sending...' : `Send to ${form.customerPhone}`}
              </button>
            </div>
          )}
        </div>
      )}

      {tab === 'history' && (
        <div className="space-y-3">
          {history.length === 0 && (
            <p className="text-gray-400 text-sm py-8 text-center">No messages sent yet.</p>
          )}
          {history.map((log) => (
            <div key={log.id} className="bg-white p-4 rounded-xl border border-gray-200">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm">{log.customer_name}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  log.status === 'delivered' ? 'bg-green-100 text-green-700' :
                  log.status === 'sent' ? 'bg-blue-100 text-blue-700' :
                  log.status === 'failed' ? 'bg-red-100 text-red-700' :
                  'bg-gray-100 text-gray-500'
                }`}>{log.status}</span>
              </div>
              <p className="text-sm text-gray-500">{log.message}</p>
              <p className="text-xs text-gray-400 mt-1">{new Date(log.sent_at).toLocaleDateString()}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add app/tools/review-rocket/
git commit -m "feat: add Review Rocket tool UI"
```

---

## Task 9: AdForge Tool UI

**Files:**
- Create: `app/tools/adforge/page.tsx`
- Create: `app/tools/adforge/client.tsx`

- [ ] **Step 1: Create `app/tools/adforge/page.tsx`**

```tsx
// app/tools/adforge/page.tsx
import { headers } from 'next/headers'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { Nav } from '@/components/nav'
import { AdForgeClient } from './client'
import type { AdGeneration } from '@/types'

export default async function AdForgePage() {
  const h = await headers()
  const tier = await getTier(h)
  const userId = await getWhopUserId(h)

  const { data: history } = await supabase
    .from('ad_generations')
    .select('*')
    .eq('whop_user_id', userId)
    .order('created_at', { ascending: false })
    .limit(10)

  return (
    <div className="min-h-screen bg-gray-50">
      <Nav tier={tier} />
      <div className="max-w-3xl mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-1">📣 AdForge</h1>
          <p className="text-gray-500">Generate 3 ad copy variations for Facebook or Google.</p>
        </div>
        <AdForgeClient history={(history as AdGeneration[]) ?? []} />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `app/tools/adforge/client.tsx`**

```tsx
// app/tools/adforge/client.tsx
'use client'

import { useState } from 'react'
import type { AdGeneration, AdVariation } from '@/types'

interface Props {
  history: AdGeneration[]
}

export function AdForgeClient({ history }: Props) {
  const [tab, setTab] = useState<'generate' | 'history'>('generate')
  const [form, setForm] = useState({
    business_type: 'hvac',
    service: '',
    audience: 'local homeowners',
    hook: '',
    tone: 'professional' as 'professional' | 'casual' | 'urgent',
    platform: 'facebook' as 'facebook' | 'google',
  })
  const [variations, setVariations] = useState<AdVariation[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState<number | null>(null)

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    setVariations([])

    const res = await fetch('/api/ai/ad', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    })

    const data = await res.json()
    if (!res.ok) {
      setError(data.error ?? 'Failed to generate ads')
    } else {
      setVariations(data.variations)
    }
    setLoading(false)
  }

  function handleCopy(idx: number, text: string) {
    navigator.clipboard.writeText(text)
    setCopied(idx)
    setTimeout(() => setCopied(null), 2000)
  }

  return (
    <div>
      <div className="flex gap-4 mb-6 border-b border-gray-200">
        {(['generate', 'history'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`pb-3 text-sm font-medium capitalize border-b-2 transition-colors ${
              tab === t ? 'border-black text-black' : 'border-transparent text-gray-400'
            }`}
          >
            {t === 'history' ? `History (${history.length})` : 'Generate Ads'}
          </button>
        ))}
      </div>

      {tab === 'generate' && (
        <div className="space-y-6">
          <form onSubmit={handleGenerate} className="bg-white p-6 rounded-xl border border-gray-200 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Business type</label>
                <select
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.business_type}
                  onChange={(e) => setForm({ ...form, business_type: e.target.value })}
                >
                  {['hvac', 'plumbing', 'auto', 'roofing', 'other'].map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Platform</label>
                <select
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.platform}
                  onChange={(e) => setForm({ ...form, platform: e.target.value as 'facebook' | 'google' })}
                >
                  <option value="facebook">Facebook</option>
                  <option value="google">Google</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Service to promote</label>
              <input
                required
                placeholder="e.g. AC tune-up, drain cleaning, oil change"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.service}
                onChange={(e) => setForm({ ...form, service: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Offer / hook</label>
              <input
                required
                placeholder="e.g. $49 special, free estimate, same-day service"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.hook}
                onChange={(e) => setForm({ ...form, hook: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Tone</label>
              <div className="flex gap-3">
                {(['professional', 'casual', 'urgent'] as const).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setForm({ ...form, tone: t })}
                    className={`px-4 py-2 text-sm rounded-lg border-2 capitalize transition-colors ${
                      form.tone === t ? 'border-black bg-black text-white' : 'border-gray-200 text-gray-600'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white py-2.5 rounded-lg text-sm font-medium disabled:opacity-50"
            >
              {loading ? 'Generating...' : 'Generate 3 Ad Variations'}
            </button>
          </form>

          {variations.length > 0 && (
            <div className="space-y-4">
              {variations.map((v, idx) => (
                <div key={idx} className="bg-white p-5 rounded-xl border border-gray-200">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Variation {idx + 1}</span>
                    <button
                      onClick={() => handleCopy(idx, `${v.headline}\n\n${v.body}\n\n${v.cta}`)}
                      className="text-xs px-3 py-1 border border-gray-200 rounded-lg hover:bg-gray-50"
                    >
                      {copied === idx ? '✓ Copied' : 'Copy all'}
                    </button>
                  </div>
                  <div className="space-y-2">
                    <div><span className="text-xs text-gray-400">Headline</span><p className="font-semibold">{v.headline}</p></div>
                    <div><span className="text-xs text-gray-400">Body</span><p className="text-sm text-gray-700">{v.body}</p></div>
                    <div><span className="text-xs text-gray-400">CTA</span><p className="text-sm font-medium text-blue-600">{v.cta}</p></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === 'history' && (
        <div className="space-y-3">
          {history.length === 0 && (
            <p className="text-gray-400 text-sm py-8 text-center">No generations yet.</p>
          )}
          {history.map((gen) => (
            <div key={gen.id} className="bg-white p-4 rounded-xl border border-gray-200">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm">{(gen.inputs as { service: string }).service}</span>
                <span className="text-xs text-gray-400">{gen.platform}</span>
              </div>
              <p className="text-xs text-gray-400">{new Date(gen.created_at).toLocaleDateString()}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add app/tools/adforge/
git commit -m "feat: add AdForge tool UI"
```

---

## Task 10: EstimateAI Tool UI

**Files:**
- Create: `app/tools/estimateai/page.tsx`
- Create: `app/tools/estimateai/client.tsx`

- [ ] **Step 1: Create `app/tools/estimateai/page.tsx`**

```tsx
// app/tools/estimateai/page.tsx
import { headers } from 'next/headers'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { Nav } from '@/components/nav'
import { EstimateAIClient } from './client'
import type { Estimate } from '@/types'

export default async function EstimateAIPage() {
  const h = await headers()
  const tier = await getTier(h)
  const userId = await getWhopUserId(h)

  const { data: business } = await supabase
    .from('businesses')
    .select('name')
    .eq('whop_user_id', userId)
    .single()

  const { data: estimates } = await supabase
    .from('estimates')
    .select('*')
    .eq('whop_user_id', userId)
    .order('created_at', { ascending: false })
    .limit(20)

  return (
    <div className="min-h-screen bg-gray-50">
      <Nav tier={tier} />
      <div className="max-w-3xl mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold mb-1">📋 EstimateAI</h1>
          <p className="text-gray-500">Create professional estimates and email them instantly.</p>
        </div>
        <EstimateAIClient
          businessName={business?.name ?? ''}
          tier={tier}
          estimates={(estimates as Estimate[]) ?? []}
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `app/tools/estimateai/client.tsx`**

```tsx
// app/tools/estimateai/client.tsx
'use client'

import { useState } from 'react'
import type { Estimate, Tier } from '@/types'

interface Props {
  businessName: string
  tier: Tier
  estimates: Estimate[]
}

export function EstimateAIClient({ businessName, tier, estimates: initialEstimates }: Props) {
  const [tab, setTab] = useState<'create' | 'history'>('create')
  const [form, setForm] = useState({
    customerName: '',
    customerEmail: '',
    jobType: '',
    scope: '',
    materials: '',
    laborHours: 4,
    markupPercent: 20,
  })
  const [preview, setPreview] = useState<{ estimate: any; estimateId: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const [sent, setSent] = useState(false)
  const [error, setError] = useState('')
  const [estimates, setEstimates] = useState(initialEstimates)

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError('')
    setPreview(null)

    const res = await fetch('/api/ai/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, businessName }),
    })

    const data = await res.json()
    if (!res.ok) {
      setError(data.error ?? 'Failed to generate estimate')
    } else {
      setPreview({ estimate: data.estimate, estimateId: data.estimateId })
    }
    setLoading(false)
  }

  async function handleSend() {
    if (!preview) return
    setSending(true)
    setError('')

    const res = await fetch('/api/email/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ estimateId: preview.estimateId, businessName }),
    })

    const data = await res.json()
    if (!res.ok) {
      setError(data.error ?? 'Failed to send estimate')
    } else {
      setSent(true)
      setTimeout(() => setSent(false), 3000)
    }
    setSending(false)
  }

  async function handleStatusUpdate(estimateId: string, status: 'accepted' | 'declined') {
    await fetch('/api/estimates/status', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ estimateId, status }),
    })
    setEstimates((prev) =>
      prev.map((e) => (e.id === estimateId ? { ...e, status } : e))
    )
  }

  return (
    <div>
      <div className="flex gap-4 mb-6 border-b border-gray-200">
        {(['create', 'history'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`pb-3 text-sm font-medium capitalize border-b-2 transition-colors ${
              tab === t ? 'border-black text-black' : 'border-transparent text-gray-400'
            }`}
          >
            {t === 'history' ? `Estimates (${estimates.length})` : 'New Estimate'}
          </button>
        ))}
      </div>

      {tab === 'create' && (
        <div className="space-y-6">
          <form onSubmit={handleGenerate} className="bg-white p-6 rounded-xl border border-gray-200 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Customer name</label>
                <input
                  required
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.customerName}
                  onChange={(e) => setForm({ ...form, customerName: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Customer email</label>
                <input
                  required
                  type="email"
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.customerEmail}
                  onChange={(e) => setForm({ ...form, customerEmail: e.target.value })}
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Job type</label>
              <input
                required
                placeholder="e.g. Furnace replacement, pipe repair"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.jobType}
                onChange={(e) => setForm({ ...form, jobType: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Scope of work</label>
              <textarea
                required
                rows={3}
                placeholder="Describe what needs to be done..."
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.scope}
                onChange={(e) => setForm({ ...form, scope: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Materials needed</label>
              <input
                required
                placeholder="e.g. 96% efficiency furnace, copper fittings, PVC pipe"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={form.materials}
                onChange={(e) => setForm({ ...form, materials: e.target.value })}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Labor hours</label>
                <input
                  required
                  type="number"
                  min={0.5}
                  step={0.5}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.laborHours}
                  onChange={(e) => setForm({ ...form, laborHours: Number(e.target.value) })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Markup %</label>
                <input
                  required
                  type="number"
                  min={0}
                  max={100}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                  value={form.markupPercent}
                  onChange={(e) => setForm({ ...form, markupPercent: Number(e.target.value) })}
                />
              </div>
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white py-2.5 rounded-lg text-sm font-medium disabled:opacity-50"
            >
              {loading ? 'Generating estimate...' : 'Generate Estimate'}
            </button>
          </form>

          {preview && (
            <div className="bg-white p-6 rounded-xl border border-gray-200">
              <h3 className="font-semibold mb-4">Estimate Preview</h3>
              <p className="text-sm text-gray-600 mb-4">{preview.estimate.intro}</p>
              <table className="w-full text-sm mb-4">
                <thead>
                  <tr className="border-b border-gray-100">
                    <th className="text-left py-2 font-medium text-gray-500">Description</th>
                    <th className="text-right py-2 font-medium text-gray-500">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.estimate.line_items.map((item: any, i: number) => (
                    <tr key={i} className="border-b border-gray-50">
                      <td className="py-2">{item.description}</td>
                      <td className="py-2 text-right">${item.amount.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="font-semibold">
                    <td className="py-3">Total</td>
                    <td className="py-3 text-right">${preview.estimate.total.toFixed(2)}</td>
                  </tr>
                </tfoot>
              </table>
              <p className="text-xs text-gray-400 mb-4">{preview.estimate.terms}</p>
              <button
                onClick={handleSend}
                disabled={sending || sent}
                className="w-full bg-green-600 text-white py-2.5 rounded-lg text-sm font-medium disabled:opacity-50"
              >
                {sent ? '✓ Estimate Sent!' : sending ? 'Sending...' : `Email to ${form.customerEmail}`}
              </button>
            </div>
          )}
        </div>
      )}

      {tab === 'history' && (
        <div className="space-y-3">
          {estimates.length === 0 && (
            <p className="text-gray-400 text-sm py-8 text-center">No estimates yet.</p>
          )}
          {estimates.map((est) => (
            <div key={est.id} className="bg-white p-4 rounded-xl border border-gray-200">
              <div className="flex items-center justify-between mb-1">
                <div>
                  <span className="font-medium text-sm">{est.customer_name}</span>
                  <span className="text-gray-400 text-sm ml-2">— {est.job_type}</span>
                </div>
                <span className="font-semibold text-sm">${Number(est.total).toFixed(2)}</span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  est.status === 'accepted' ? 'bg-green-100 text-green-700' :
                  est.status === 'declined' ? 'bg-red-100 text-red-700' :
                  est.status === 'sent' ? 'bg-blue-100 text-blue-700' :
                  'bg-gray-100 text-gray-500'
                }`}>{est.status}</span>
                {est.status === 'sent' && (
                  <>
                    <button
                      onClick={() => handleStatusUpdate(est.id, 'accepted')}
                      className="text-xs text-green-600 underline"
                    >Mark accepted</button>
                    <button
                      onClick={() => handleStatusUpdate(est.id, 'declined')}
                      className="text-xs text-red-500 underline"
                    >Mark declined</button>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Create `app/api/estimates/status/route.ts`**

```ts
// app/api/estimates/status/route.ts
import { NextRequest } from 'next/server'
import { getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  estimateId: z.string().uuid(),
  status: z.enum(['accepted', 'declined']),
})

export async function PATCH(request: NextRequest) {
  const userId = await getWhopUserId(request.headers)
  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input' }, { status: 400 })
  }

  await supabase
    .from('estimates')
    .update({ status: parsed.data.status })
    .eq('id', parsed.data.estimateId)
    .eq('whop_user_id', userId)

  return Response.json({ success: true })
}
```

- [ ] **Step 4: Commit**

```bash
git add app/tools/estimateai/ app/api/estimates/
git commit -m "feat: add EstimateAI tool UI"
```

---

## Task 11: Settings Page

**Files:**
- Create: `app/settings/page.tsx`

- [ ] **Step 1: Create `app/settings/page.tsx`**

```tsx
// app/settings/page.tsx
import { headers } from 'next/headers'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { Nav } from '@/components/nav'
import { SettingsClient } from './client'

export default async function SettingsPage() {
  const h = await headers()
  const tier = await getTier(h)
  const userId = await getWhopUserId(h)

  const { data: business } = await supabase
    .from('businesses')
    .select('*')
    .eq('whop_user_id', userId)
    .single()

  const { data: locations } = await supabase
    .from('locations')
    .select('*')
    .eq('whop_user_id', userId)
    .order('created_at', { ascending: true })

  return (
    <div className="min-h-screen bg-gray-50">
      <Nav tier={tier} />
      <div className="max-w-2xl mx-auto p-8">
        <h1 className="text-2xl font-bold mb-8">Settings</h1>
        <SettingsClient business={business} locations={locations ?? []} tier={tier} />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `app/settings/client.tsx`**

```tsx
// app/settings/client.tsx
'use client'

import { useState } from 'react'
import type { Business, Location, Tier } from '@/types'

interface Props {
  business: Business | null
  locations: Location[]
  tier: Tier
}

export function SettingsClient({ business, locations: initialLocations, tier }: Props) {
  const [form, setForm] = useState({
    name: business?.name ?? '',
    type: business?.type ?? 'hvac',
    phone: business?.phone ?? '',
    google_review_url: business?.google_review_url ?? '',
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [locations, setLocations] = useState(initialLocations)
  const [newLocation, setNewLocation] = useState({ name: '', phone: '', google_review_url: '' })
  const [addingLocation, setAddingLocation] = useState(false)

  async function handleSaveProfile(e: React.FormEvent) {
    e.preventDefault()
    setSaving(true)
    await fetch('/api/onboarding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...form, starter_tool: business?.starter_tool ?? null }),
    })
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
    setSaving(false)
  }

  async function handleAddLocation(e: React.FormEvent) {
    e.preventDefault()
    setAddingLocation(true)
    const res = await fetch('/api/locations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newLocation),
    })
    const data = await res.json()
    if (res.ok) {
      setLocations((prev) => [...prev, data.location])
      setNewLocation({ name: '', phone: '', google_review_url: '' })
    }
    setAddingLocation(false)
  }

  return (
    <div className="space-y-8">
      {/* Business Profile */}
      <section className="bg-white p-6 rounded-xl border border-gray-200">
        <h2 className="font-semibold mb-4">Business Profile</h2>
        <form onSubmit={handleSaveProfile} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Business name</label>
            <input
              required
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Business phone</label>
            <input
              required
              type="tel"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
              value={form.phone}
              onChange={(e) => setForm({ ...form, phone: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Google review link</label>
            <input
              required
              type="url"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
              value={form.google_review_url}
              onChange={(e) => setForm({ ...form, google_review_url: e.target.value })}
            />
          </div>
          <button
            type="submit"
            disabled={saving}
            className="bg-black text-white px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
          >
            {saved ? '✓ Saved' : saving ? 'Saving...' : 'Save Changes'}
          </button>
        </form>
      </section>

      {/* Locations — Agency only */}
      {tier === 'agency' && (
        <section className="bg-white p-6 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold">Locations ({locations.length}/5)</h2>
          </div>
          <div className="space-y-2 mb-6">
            {locations.map((loc) => (
              <div key={loc.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium">{loc.name}</p>
                  <p className="text-xs text-gray-400">{loc.phone}</p>
                </div>
              </div>
            ))}
          </div>
          {locations.length < 5 && (
            <form onSubmit={handleAddLocation} className="space-y-3 border-t border-gray-100 pt-4">
              <p className="text-sm font-medium text-gray-600">Add location</p>
              <input
                required
                placeholder="Location name"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={newLocation.name}
                onChange={(e) => setNewLocation({ ...newLocation, name: e.target.value })}
              />
              <input
                required
                type="tel"
                placeholder="Phone"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={newLocation.phone}
                onChange={(e) => setNewLocation({ ...newLocation, phone: e.target.value })}
              />
              <input
                required
                type="url"
                placeholder="Google review URL"
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm"
                value={newLocation.google_review_url}
                onChange={(e) => setNewLocation({ ...newLocation, google_review_url: e.target.value })}
              />
              <button
                type="submit"
                disabled={addingLocation}
                className="bg-black text-white px-5 py-2 rounded-lg text-sm font-medium disabled:opacity-50"
              >
                {addingLocation ? 'Adding...' : 'Add Location'}
              </button>
            </form>
          )}
        </section>
      )}

      {/* Plan */}
      <section className="bg-white p-6 rounded-xl border border-gray-200">
        <h2 className="font-semibold mb-2">Plan</h2>
        <p className="text-sm text-gray-500 mb-4">
          Manage your subscription on Whop — upgrades, downgrades, and cancellations.
        </p>
        <a
          href="https://whop.com/tradestack"
          target="_top"
          className="inline-block bg-black text-white px-5 py-2 rounded-lg text-sm font-medium"
        >
          Manage on Whop
        </a>
      </section>
    </div>
  )
}
```

- [ ] **Step 3: Create `app/api/locations/route.ts`**

```ts
// app/api/locations/route.ts
import { NextRequest } from 'next/server'
import { getTier, getWhopUserId } from '@/lib/whop'
import { supabase } from '@/lib/supabase'
import { z } from 'zod'

const schema = z.object({
  name: z.string().min(1),
  phone: z.string().min(10),
  google_review_url: z.string().url(),
})

export async function POST(request: NextRequest) {
  const tier = await getTier(request.headers)
  if (tier !== 'agency') {
    return Response.json({ error: 'Multiple locations require Agency plan' }, { status: 403 })
  }

  const userId = await getWhopUserId(request.headers)

  const { count } = await supabase
    .from('locations')
    .select('*', { count: 'exact', head: true })
    .eq('whop_user_id', userId)

  if ((count ?? 0) >= 5) {
    return Response.json({ error: 'Maximum 5 locations reached' }, { status: 400 })
  }

  const body = await request.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    return Response.json({ error: 'Invalid input' }, { status: 400 })
  }

  const { data: location } = await supabase
    .from('locations')
    .insert({ whop_user_id: userId, ...parsed.data })
    .select('*')
    .single()

  return Response.json({ location })
}
```

- [ ] **Step 4: Commit**

```bash
git add app/settings/ app/api/locations/
git commit -m "feat: add settings page with business profile and Agency locations"
```

---

## Task 12: Deploy to Vercel + Register Whop App

**Files:**
- None (configuration steps)

- [ ] **Step 1: Push to GitHub**

```bash
git remote add origin https://github.com/<your-username>/tradestack.git
git push -u origin main
```

- [ ] **Step 2: Deploy to Vercel**

1. Go to vercel.com → New Project → Import from GitHub → select `tradestack`
2. Add all environment variables from `.env.local.example`
3. Deploy
4. Note your production URL (e.g. `https://tradestack.vercel.app`)

- [ ] **Step 3: Register TradeStack on Whop**

1. Go to whop.com/developer → Create App
2. App name: `TradeStack`
3. App URL: `https://tradestack.vercel.app`
4. Copy the `WHOP_API_KEY` → add to Vercel env vars

- [ ] **Step 4: Create 3 Whop Products**

In Whop dashboard, create 3 products:
- **TradeStack Starter** — $79/mo — Experience URL: `https://tradestack.vercel.app`
- **TradeStack Pro** — $197/mo — Experience URL: `https://tradestack.vercel.app`
- **TradeStack Agency** — $397/mo — Experience URL: `https://tradestack.vercel.app`

Copy each Experience ID → add to Vercel:
- `WHOP_EXPERIENCE_STARTER`
- `WHOP_EXPERIENCE_PRO`
- `WHOP_EXPERIENCE_AGENCY`

- [ ] **Step 5: Redeploy with Experience IDs**

```bash
# In Vercel dashboard → Deployments → Redeploy (or push any commit)
git commit --allow-empty -m "chore: trigger redeploy with Whop experience IDs"
git push
```

- [ ] **Step 6: Test end-to-end in Whop**

1. Buy a Starter pass (use Stripe test mode)
2. Open the experience in Whop
3. Confirm onboarding loads inside the iframe
4. Complete onboarding → verify dashboard loads with correct tier badge
5. Test AdForge (Starter tool)
6. Confirm Review Rocket shows upgrade CTA on Starter

---

## Task 13: Playwright E2E Tests

**Files:**
- Create: `e2e/review-rocket.spec.ts`
- Create: `e2e/adforge.spec.ts`
- Create: `e2e/estimateai.spec.ts`
- Create: `playwright.config.ts`

- [ ] **Step 1: Create `playwright.config.ts`**

```ts
// playwright.config.ts
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  use: {
    baseURL: 'http://localhost:3000',
    extraHTTPHeaders: {
      // Mock Whop Pro token for tests — set up a test helper that generates this
      'x-whop-user-token': process.env.TEST_WHOP_TOKEN ?? 'test-token',
    },
  },
})
```

- [ ] **Step 2: Create `e2e/adforge.spec.ts`**

```ts
// e2e/adforge.spec.ts
import { test, expect } from '@playwright/test'

test('AdForge generates 3 ad variations', async ({ page }) => {
  await page.goto('/tools/adforge')
  await page.fill('input[placeholder*="Service"]', 'AC tune-up')
  await page.fill('input[placeholder*="hook"]', '$49 special')
  await page.click('button:has-text("Generate 3 Ad Variations")')
  await expect(page.locator('text=Variation 1')).toBeVisible({ timeout: 15000 })
  await expect(page.locator('text=Variation 2')).toBeVisible()
  await expect(page.locator('text=Variation 3')).toBeVisible()
})
```

- [ ] **Step 3: Create `e2e/review-rocket.spec.ts`**

```ts
// e2e/review-rocket.spec.ts
import { test, expect } from '@playwright/test'

test('Review Rocket generates SMS preview', async ({ page }) => {
  await page.goto('/tools/review-rocket')
  await page.fill('input[placeholder*=""]', 'John') // customer name
  // Fill all required fields
  await page.locator('input').nth(0).fill('John')
  await page.locator('input').nth(1).fill('+15141234567')
  await page.locator('input').nth(2).fill('furnace repair')
  await page.click('button:has-text("Generate SMS")')
  await expect(page.locator('textarea')).toBeVisible({ timeout: 15000 })
})
```

- [ ] **Step 4: Create `e2e/estimateai.spec.ts`**

```ts
// e2e/estimateai.spec.ts
import { test, expect } from '@playwright/test'

test('EstimateAI generates an estimate preview', async ({ page }) => {
  await page.goto('/tools/estimateai')
  await page.locator('input').nth(0).fill('Jane Smith')
  await page.locator('input').nth(1).fill('jane@example.com')
  await page.locator('input').nth(2).fill('Furnace replacement')
  await page.locator('textarea').fill('Remove old furnace, install new 96% efficiency unit')
  await page.locator('input').nth(3).fill('Carrier furnace, flue pipe')
  await page.click('button:has-text("Generate Estimate")')
  await expect(page.locator('text=Estimate Preview')).toBeVisible({ timeout: 20000 })
  await expect(page.locator('text=Total')).toBeVisible()
})
```

- [ ] **Step 5: Run E2E tests**

```bash
pnpm exec playwright test
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add e2e/ playwright.config.ts
git commit -m "test: add Playwright E2E tests for all 3 tools"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| Next.js App Router + Vercel | Task 1 |
| Whop SDK auth + tier gating | Task 3 |
| Supabase schema (5 tables) | Task 2 |
| Claude API (all 3 prompts) | Task 4, 5 |
| Review Rocket: generate + send SMS | Task 5, 8 |
| AdForge: generate + history | Task 5, 9 |
| EstimateAI: generate + email | Task 5, 10 |
| Estimate status tracking | Task 10 |
| Onboarding flow | Task 7 |
| Dashboard with tier-gated cards | Task 7 |
| ToolGate component | Task 6 |
| Settings: profile + locations | Task 11 |
| Agency locations (max 5) | Task 11 |
| SMS limits (200 Pro, 1000 Agency) | Task 5 |
| Generation limits (100 Starter) | Task 5 |
| Twilio failure logging | Task 5 |
| Resend retry-once | Task 5 |
| Vercel deploy + Whop registration | Task 12 |
| Playwright E2E (3 flows) | Task 13 |
| Vitest prompt builder tests | Task 4 |
| iframe headers (X-Frame-Options) | Task 1 |
| White-label estimates (Agency) | Task 5 |
