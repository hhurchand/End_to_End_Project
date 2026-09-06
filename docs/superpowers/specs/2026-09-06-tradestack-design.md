# TradeStack — Platform Design Spec
**Date:** 2026-09-06
**Status:** Approved

---

## Overview

TradeStack is a Next.js SaaS platform offering 3 AI-powered tools for home service businesses (HVAC, plumbing, auto shops, roofing). One login, one subscription, three tools: Review Rocket, AdForge, and EstimateAI. Built on the existing stack: Next.js App Router, Supabase, Stripe, and Claude API.

---

## Architecture

### Stack
| Layer | Technology |
|---|---|
| Frontend + API | Next.js App Router (Vercel) |
| Auth + Database + Storage | Supabase |
| Billing | Stripe Checkout + Customer Portal |
| AI | Claude claude-sonnet-4-6 (Anthropic) |
| SMS | Twilio |
| Transactional Email | Resend |

### Route Structure
```
/ — marketing landing page
/login — email/password login + magic link
/signup — new account creation
/dashboard — tool hub (all 3 tools)
/tools/review-rocket — Review Rocket tool
/tools/adforge — AdForge tool
/tools/estimateai — EstimateAI tool
/settings — billing, business profile, locations, team
/api/ai/review — Claude: generate review SMS
/api/ai/ad — Claude: generate ad copy
/api/ai/estimate — Claude: generate estimate
/api/sms/send — Twilio: send review SMS
/api/email/send — Resend: send estimate email
/api/stripe/webhook — Stripe event handler
/api/stripe/checkout — create Stripe Checkout session
```

### Database Schema (Supabase)

**`businesses`**
- `id`, `user_id`, `name`, `type` (hvac/plumbing/auto/roofing/other)
- `phone`, `google_review_url`
- `created_at`

**`subscriptions`**
- `id`, `user_id`, `stripe_customer_id`, `stripe_subscription_id`
- `tier` (starter/pro/agency), `status` (trialing/active/canceled/past_due)
- `trial_ends_at`, `current_period_end`

**`locations`** (Agency tier — up to 5 per account)
- `id`, `user_id`, `business_id`, `name`, `phone`, `google_review_url`

**`sms_log`**
- `id`, `business_id`, `customer_name`, `customer_phone`
- `message`, `twilio_sid`, `status` (queued/sent/delivered/failed)
- `sent_at`

**`estimates`**
- `id`, `business_id`, `customer_name`, `customer_email`
- `job_type`, `scope`, `line_items` (JSONB), `total`, `terms`
- `status` (draft/sent/accepted/declined)
- `sent_at`, `created_at`

**`ad_generations`**
- `id`, `business_id`, `inputs` (JSONB), `variations` (JSONB — 3 ad copies)
- `platform` (facebook/google), `created_at`

---

## The 3 Tools

### Review Rocket
**Purpose:** Send AI-personalized SMS to customers after a job → drive Google reviews.

**Flow:**
1. Owner enters: customer name, phone number, job type, quick note about the job
2. Claude generates a personalized, natural-sounding SMS
3. Owner previews the SMS, edits if needed, hits Send
4. Twilio delivers it — SMS includes direct Google Business review link
5. Dashboard shows send history with delivery status badges and review count trend

**Limits:** 200 SMS/mo (Pro), 1,000 SMS/mo (Agency), none on Starter (tool locked)

### AdForge
**Purpose:** Generate 3 variations of ad copy for Facebook or Google in seconds.

**Flow:**
1. Owner selects: business type, service to promote, target audience, offer/hook, tone (professional/casual/urgent), platform (Facebook/Google)
2. Claude generates 3 ad variations — each with headline, body copy, and CTA
3. One-click copy to clipboard per variation
4. Generation history saved — owner can reuse or regenerate winners

**Limits:** 100 generations/mo (Starter), unlimited (Pro/Agency)

### EstimateAI
**Purpose:** Generate professional customer-facing estimates from job notes — sent directly to customer via email.

**Flow:**
1. Owner fills in: customer name + email, job type, scope (bullet points or free text), materials list, labor hours, markup %
2. Claude formats a clean estimate: line items, subtotal, markup, total, payment terms, personalized intro paragraph
3. Owner previews, edits if needed
4. One-click sends to customer via Resend (branded with business name)
5. PDF download also available (generated server-side via `@react-pdf/renderer`)
6. Owner marks estimates as accepted/declined — close rate tracked in dashboard

**White-label:** Agency tier — customer sees business branding only, no TradeStack branding

---

## Pricing & Billing

| Plan | Price | Tools | Locations | SMS/mo | Generations |
|---|---|---|---|---|---|
| Starter | $79/mo | 1 tool | 1 | — | 100 |
| Pro | $197/mo | All 3 | 1 | 200 | Unlimited |
| Agency | $397/mo | All 3 | 5 | 1,000 | Unlimited |

**Trial:** 14-day free trial, no credit card required. Stripe Checkout triggered at trial end.

**Mechanics:**
- Stripe Checkout for initial subscription
- Stripe Customer Portal for self-serve upgrades, downgrades, cancellations
- Stripe webhooks sync subscription status to `subscriptions` table in real time
- Downgrade locks tool access immediately; upgrade unlocks instantly
- Locked tools on lower tiers show upgrade prompt inline

---

## Auth & Onboarding

**Auth:** Supabase email/password with magic link fallback. No OAuth.

**Onboarding flow (post-signup):**
1. Business profile — name, type, phone, Google Business review URL
2. Plan selection — Starter users pick which 1 tool to activate; Pro/Agency get all 3. Free trial starts immediately, no card required.
3. Dashboard — all tools visible; locked tools show upgrade CTA

**Agency locations:** Owner adds up to 4 additional locations at `/settings/locations`. Each location has its own business profile, review link, and SMS sender name.

**Trial activation:** At trial end, dashboard banner prompts "Activate your plan" → Stripe Checkout session → card collected → subscription activates.

---

## Error Handling

| Failure | Behavior |
|---|---|
| Claude API error | Inline error message, retry button — no silent failures |
| Twilio SMS failure | Log to `sms_log` with failed status, red badge in history |
| Resend email failure | Auto-retry once, then notify owner in dashboard |
| Stripe webhook failure | Idempotent handlers, Stripe retries automatically |
| Generation limit reached | Hard block with upgrade prompt, never silently drop |

---

## Testing

- **Vitest unit tests** — AI prompt builders (validate output format before sending)
- **Stripe webhook tests** — mock payloads for all subscription events
- **Playwright E2E** — 3 core flows: send review SMS, generate ad copy, send estimate
- **Seed script** — fake business, fake subscription, sample generations for local dev

---

## Out of Scope (v1)

- Mobile app
- Native Zapier / Make integrations
- AI-generated images for ads
- CRM / contact database
- Multi-user team accounts within a business
