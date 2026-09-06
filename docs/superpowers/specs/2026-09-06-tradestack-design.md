# TradeStack — Platform Design Spec
**Date:** 2026-09-06
**Status:** Approved (revised for Whop)

---

## Overview

TradeStack is a Next.js SaaS app offering 3 AI-powered tools for home service businesses (HVAC, plumbing, auto shops, roofing): Review Rocket, AdForge, and EstimateAI. Distributed and sold exclusively on Whop.com — Whop handles all billing, subscriptions, and user identity. The app runs inside a Whop iframe. Built on: Next.js App Router, Supabase (app data only), Claude API, Twilio, Resend.

---

## Architecture

### Stack
| Layer | Technology |
|---|---|
| Frontend + API | Next.js App Router (Vercel) |
| Distribution + Billing + Auth | Whop.com + `@whop/sdk` |
| App Data (no auth) | Supabase (postgres) |
| AI | Claude claude-sonnet-4-6 (Anthropic) |
| SMS | Twilio |
| Transactional Email | Resend |

### How Whop Works
- TradeStack is registered as a Whop App with one Experience per tier
- Users buy a pass on Whop marketplace → Whop loads the app in an iframe
- Every request includes an `x-whop-user-token` header (short-lived JWT)
- `verifyUserToken()` from `@whop/sdk` validates the JWT → returns `userId`
- `checkAccess({ to: experienceId, headers })` verifies the user has an active pass for a given tier
- No custom login, no Stripe — Whop owns the payment and identity layer entirely

### Whop Products (create in Whop dashboard)
| Product | Experience ID env var | Price |
|---|---|---|
| TradeStack Starter (1 tool) | `WHOP_EXPERIENCE_STARTER` | $79/mo |
| TradeStack Pro (all 3 tools) | `WHOP_EXPERIENCE_PRO` | $197/mo |
| TradeStack Agency (5 locations) | `WHOP_EXPERIENCE_AGENCY` | $397/mo |

Each experience points to the same app URL — the app reads which experience the user has access to and unlocks features accordingly.

### Route Structure
```
/ — app entry point (Whop iframe loads this)
/onboarding — business profile setup (first visit only)
/dashboard — tool hub
/tools/review-rocket — Review Rocket
/tools/adforge — AdForge
/tools/estimateai — EstimateAI
/settings — business profile, locations (Agency)
/api/ai/review — Claude: generate review SMS
/api/ai/ad — Claude: generate ad copy
/api/ai/estimate — Claude: generate estimate
/api/sms/send — Twilio: send review SMS
/api/email/send — Resend: send estimate email
```

### Database Schema (Supabase — app data only, no auth tables)

**`businesses`**
- `id`, `whop_user_id` (text, unique), `name`, `type` (hvac/plumbing/auto/roofing/other)
- `phone`, `google_review_url`, `onboarded_at`
- `created_at`

**`locations`** (Agency tier — up to 5 per account)
- `id`, `whop_user_id`, `name`, `phone`, `google_review_url`
- `created_at`

**`sms_log`**
- `id`, `whop_user_id`, `customer_name`, `customer_phone`
- `message`, `twilio_sid`, `status` (queued/sent/delivered/failed)
- `location_id` (nullable — Agency multi-location)
- `sent_at`

**`estimates`**
- `id`, `whop_user_id`, `customer_name`, `customer_email`
- `job_type`, `scope` (text), `line_items` (JSONB), `total` (numeric), `terms` (text)
- `status` (draft/sent/accepted/declined)
- `location_id` (nullable)
- `sent_at`, `created_at`

**`ad_generations`**
- `id`, `whop_user_id`, `inputs` (JSONB), `variations` (JSONB — 3 ad copies)
- `platform` (facebook/google), `created_at`

---

## Tier Enforcement

Tier is determined at request time by calling `checkAccess` against each experience ID in order (Agency → Pro → Starter). The first passing check determines the user's tier. This is done in a shared `getTier(headers)` server utility used by every page and API route.

```
agency  → all tools, 5 locations, 1000 SMS/mo, white-label estimates
pro     → all tools, 1 location, 200 SMS/mo, unlimited generations
starter → 1 tool (user picks at onboarding), 1 location, 100 generations/mo
```

Locked tools render an upgrade prompt with a Whop checkout link.

---

## The 3 Tools

### Review Rocket
**Purpose:** Send AI-personalized SMS to customers after a job → drive Google reviews.

**Flow:**
1. Owner enters: customer name, phone number, job type, quick note
2. Claude generates a personalized, natural-sounding SMS
3. Owner previews, edits if needed, hits Send
4. Twilio delivers it — SMS includes direct Google Business review link
5. History tab shows send log with delivery status badges

**Limits:** Starter: locked. Pro: 200 SMS/mo. Agency: 1,000 SMS/mo.

### AdForge
**Purpose:** Generate 3 variations of ad copy for Facebook or Google.

**Flow:**
1. Owner selects: business type, service, target audience, offer/hook, tone, platform
2. Claude generates 3 ad variations — headline, body, CTA each
3. One-click copy to clipboard per variation
4. Generation history saved for reuse

**Limits:** Starter: 100 generations/mo (if chosen as their tool). Pro/Agency: unlimited.

### EstimateAI
**Purpose:** Generate professional customer-facing estimates — sent directly via email.

**Flow:**
1. Owner enters: customer name + email, job type, scope, materials, labor hours, markup %
2. Claude formats a clean estimate: line items, subtotal, markup, total, terms, intro paragraph
3. Owner previews, edits if needed
4. One-click sends to customer via Resend
5. PDF download via `@react-pdf/renderer` (server-side)
6. Owner marks accepted/declined — close rate shown in dashboard

**White-label:** Agency tier — no TradeStack branding on customer-facing estimate emails/PDFs.

---

## Onboarding

First visit detection: check if `businesses` row exists for `whop_user_id`. If not → redirect to `/onboarding`.

**Steps:**
1. Business name, type, phone, Google Business review URL
2. Starter only: pick which 1 tool to activate
3. Saved to `businesses` table → redirect to `/dashboard`

No plan selection UI — Whop handles that before the user ever reaches the app.

---

## Error Handling

| Failure | Behavior |
|---|---|
| Whop token invalid/expired | 401 — Whop re-authenticates automatically via iframe |
| Claude API error | Inline error + retry button, no silent failures |
| Twilio SMS failure | Log to `sms_log` with failed status, red badge in history |
| Resend email failure | Auto-retry once, then show error in dashboard |
| Generation limit reached | Hard block with Whop upgrade link, never silently drop |

---

## Testing

- **Vitest unit tests** — AI prompt builders, tier enforcement logic, `getTier()` utility
- **Whop token mock** — test helper that generates a mock `x-whop-user-token` header for each tier
- **Playwright E2E** — 3 core flows: send review SMS, generate ad copy, send estimate

---

## Environment Variables

```
# Whop
WHOP_API_KEY=
WHOP_EXPERIENCE_STARTER=
WHOP_EXPERIENCE_PRO=
WHOP_EXPERIENCE_AGENCY=

# Supabase
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# Anthropic
ANTHROPIC_API_KEY=

# Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

# Resend
RESEND_API_KEY=
RESEND_FROM_EMAIL=

# App
NEXT_PUBLIC_APP_URL=
```

---

## Out of Scope (v1)

- Mobile app
- Zapier / Make integrations
- AI-generated images for ads
- CRM / contact database
- Multi-user team accounts within one business
