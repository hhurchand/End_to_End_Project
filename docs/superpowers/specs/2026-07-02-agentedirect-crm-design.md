# AgenteDirect CRM — Design Spec
**Date:** 2026-07-02
**Project:** `agentedirect-crm`
**Domain:** `crm.agentedirect.ca`
**Deployment:** Vercel
**Author:** Greg Spunt

---

## 1. Overview

A branded, bilingual (FR/EN) CRM for AgenteDirect — a Montreal software agency offering flat-rate pricing and fast delivery. Single-user, desktop-first, mobile-responsive web app. Covers the full client lifecycle: inbound lead → pipeline → closed deal → project delivery → invoicing.

---

## 2. Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Database + Auth | Supabase (Postgres + magic link) |
| Styling | Tailwind CSS + shadcn/ui |
| i18n | next-intl (FR/EN) |
| Invoice PDF | @react-pdf/renderer (server-side) |
| SMS / Call logging | Twilio SDK |
| Drag-and-drop | dnd-kit |
| Deployment | Vercel |
| Cron | Vercel Cron Jobs |

---

## 3. Brand & Design System

### Colors
```css
--purple:        #8b5cf6;
--purple-deep:   #5b21b6;
--purple-light:  #a78bfa;
--purple-lighter:#c4b5fd;
--blue:          #3b82f6;
--blue-light:    #60a5fa;
--gradient:      linear-gradient(135deg, #8b5cf6, #3b82f6);
--surface-dark:  #0a0a0f;   /* near-black app background */
--surface-card:  #111827;   /* card backgrounds */
--surface-muted: #1f2937;   /* hover / muted surfaces */
--border:        #374151;
--text-primary:  #f9fafb;
--text-muted:    #9ca3af;
```

### Typography
- **Display / Headings:** Oswald (Google Fonts)
- **Body / UI:** Inter (Google Fonts)

### Effects
- Card shadow: `0 16px 64px rgba(0,0,0,0.6)`
- Purple glow on active/hover cards: `0 0 28px rgba(139,92,246,0.3)`
- Gradient used on: sidebar logo, invoice header bar, primary buttons, stage badges, Kanban column headers

### Component Conventions
- All buttons: rounded-lg, gradient primary or ghost outline
- Badges: pill-shaped, color-coded by stage/status
- Cards: dark surface, subtle border, purple glow on hover
- Sidebar: dark navy, gradient logo mark, active nav item gets left purple border

---

## 4. Project Structure

```
agentedirect-crm/
├── app/
│   ├── (dashboard)/
│   │   ├── layout.tsx            ← sidebar + topbar shell
│   │   ├── pipeline/
│   │   │   └── page.tsx          ← Kanban board
│   │   ├── leads/
│   │   │   └── [id]/page.tsx     ← Lead detail
│   │   ├── clients/
│   │   │   ├── page.tsx          ← Client list
│   │   │   └── [id]/page.tsx     ← Client profile
│   │   ├── projects/
│   │   │   ├── page.tsx          ← Project tracker table
│   │   │   └── [id]/page.tsx     ← Project detail
│   │   └── invoices/
│   │       ├── page.tsx          ← Invoice list
│   │       ├── new/page.tsx      ← Invoice generator
│   │       └── [id]/page.tsx     ← Invoice view
│   ├── api/
│   │   ├── webhook/lead/route.ts ← Chatbot → new lead
│   │   ├── invoices/[id]/pdf/route.ts
│   │   ├── twilio/sms/route.ts
│   │   ├── twilio/call-log/route.ts
│   │   └── cron/follow-ups/route.ts
│   ├── auth/
│   │   └── page.tsx              ← Magic link login
│   └── layout.tsx                ← Root layout, next-intl, fonts
├── components/
│   ├── pipeline/
│   │   ├── KanbanBoard.tsx
│   │   ├── KanbanColumn.tsx
│   │   ├── LeadCard.tsx          ← includes Won/Lost toggle on closed
│   │   └── LeadSlideOver.tsx
│   ├── leads/
│   │   ├── LeadForm.tsx
│   │   ├── ActivityTimeline.tsx
│   │   └── FollowUpScheduler.tsx
│   ├── invoices/
│   │   ├── InvoiceForm.tsx
│   │   ├── InvoicePreview.tsx
│   │   └── InvoicePDF.tsx        ← react-pdf document component
│   └── ui/                       ← shadcn/ui overridden with brand styles
├── lib/
│   ├── supabase/
│   │   ├── client.ts             ← browser client
│   │   └── server.ts             ← server component client
│   ├── twilio.ts
│   └── utils.ts
├── messages/
│   ├── en.json
│   └── fr.json
├── supabase/
│   └── migrations/               ← SQL migration files
└── vercel.json                   ← cron job config
```

---

## 5. Data Model

All tables have RLS enabled in Supabase, scoped to the authenticated `user_id`.

### `leads`
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL
name            text NOT NULL
email           text
phone           text
company         text
business_type   text
stage           text NOT NULL DEFAULT 'new'
                  -- new | contacted | discovery | proposal | closed_won | closed_lost
source          text DEFAULT 'manual'
                  -- manual | website_widget | referral
booking_link    text          -- per-lead Google Calendar override
lang_pref       text DEFAULT 'en'  -- en | fr
notes           text
created_at      timestamptz DEFAULT now()
updated_at      timestamptz DEFAULT now()
```

### `clients`
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL
lead_id         uuid REFERENCES leads(id)
name            text NOT NULL
email           text
phone           text
company         text
business_type   text
notes           text
created_at      timestamptz DEFAULT now()
```

### `projects`
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL
client_id       uuid REFERENCES clients(id) NOT NULL
title           text NOT NULL
status          text DEFAULT 'active'   -- active | on_hold | complete
deadline        date
deliverables    jsonb DEFAULT '[]'      -- [{ label: string, done: boolean }]
flat_rate_price numeric(10,2)
created_at      timestamptz DEFAULT now()
```

### `invoices`
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL
project_id      uuid REFERENCES projects(id)
client_id       uuid REFERENCES clients(id) NOT NULL
invoice_number  text UNIQUE NOT NULL    -- auto-generated server-side: AD-YYYY-NNN (sequential per year)
line_items      jsonb NOT NULL          -- [{ description, amount }]
subtotal        numeric(10,2)
tax_rate        numeric(5,2) DEFAULT 0
total           numeric(10,2)
lang            text DEFAULT 'en'       -- en | fr (PDF render language)
status          text DEFAULT 'draft'    -- draft | sent | paid
issued_at       date DEFAULT CURRENT_DATE
due_at          date
```

### `follow_ups`
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL
lead_id         uuid REFERENCES leads(id)
client_id       uuid REFERENCES clients(id)
type            text NOT NULL           -- sms | manual_call  (email reserved for v2)
message         text
scheduled_at    timestamptz NOT NULL
sent_at         timestamptz
status          text DEFAULT 'pending'  -- pending | sent | skipped
```

### `activity_log`
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL
lead_id         uuid REFERENCES leads(id)
client_id       uuid REFERENCES clients(id)
type            text NOT NULL
                  -- stage_change | call_logged | sms_sent | note_added | invoice_created | lead_created
body            text
created_at      timestamptz DEFAULT now()
```

### `settings` (single row per user)
```sql
id              uuid PRIMARY KEY DEFAULT gen_random_uuid()
user_id         uuid REFERENCES auth.users NOT NULL UNIQUE
default_calendar_link  text
default_lang    text DEFAULT 'en'
twilio_from     text
```

---

## 6. Pages & UI

### Global Shell
- **Left sidebar** (240px, dark surface): AgenteDirect gradient logo mark + wordmark (Oswald), nav items (Pipeline, Clients, Projects, Invoices), FR/EN toggle at bottom, user avatar
- **Top bar**: "Quick Add Lead" button (gradient), notification bell showing count of pending follow-ups due today
- **Language toggle**: switches `next-intl` locale, persisted to `localStorage`

### `/pipeline` — Kanban Board
- 5 columns with gradient column headers: New · Contacted · Discovery Call · Proposal Sent · Closed
- Closed column header shows: **$X,XXX won** · **X lost** (live totals from closed cards)
- **Lead card** (dark surface, purple glow on hover):
  - Name, company, stage age ("3 days in stage")
  - Source badge (widget icon if from chatbot)
  - Quick actions: 📅 Book, 💬 SMS, ✏️ Edit
  - **Closed cards only:** `[✓ Won]` / `[✗ Lost]` toggle chip — green/red, updates `stage` between `closed_won` / `closed_lost` in one click
- Drag-and-drop columns via dnd-kit; drops update `leads.stage` + write to `activity_log`
- Supabase real-time subscription: new leads from chatbot webhook appear instantly without page refresh

### `/leads/[id]` — Lead Detail (accessible as slide-over from Kanban OR standalone route)
- Contact info block (name, email, phone, company) — inline editable
- Stage selector dropdown + stage history timeline
- "Book a Call" button → opens `booking_link ?? settings.default_calendar_link` in new tab
- Follow-up scheduler: date/time picker + SMS message textarea + Send / Schedule buttons
- Notes (autosave on blur)
- Activity timeline (all `activity_log` entries for this lead, newest first)
- "Convert to Client" button — visible when `stage = closed_won`; creates `clients` row, preserves `lead_id`

### `/clients` — Client List
- Table: Name, Company, Active Projects, Last Invoice (amount + status), Actions
- Search bar, filter by business type

### `/clients/[id]` — Client Profile
- Contact info, projects list (cards with status + deadline), invoice history table, activity log

### `/projects` — Project Tracker
- Table: Title, Client, Status badge, Deadline, Flat-rate Price, Deliverables progress (e.g. "3/5")
- Inline status toggle (click badge cycles active → on_hold → complete)
- "Generate Invoice" shortcut on each row

### `/projects/[id]` — Project Detail
- Title + client link
- Deliverables checklist (checkboxes, JSONB array, optimistic UI update)
- Deadline with countdown ("due in 4 days" / "overdue by 2 days" in red)
- Flat-rate price display
- "Generate Invoice" button → navigates to `/invoices/new?project=[id]`

### `/invoices` — Invoice List
- Table: Invoice #, Client, Amount, Status badge (Draft/Sent/Paid), Issued, Due
- One-click status change (Draft → Sent → Paid)
- Download PDF button per row

### `/invoices/new` + `/invoices/[id]`
- Client selector (pre-filled if coming from project)
- Line items: description + amount rows, add/remove
- Tax toggle (QC 14.975% preset option)
- Language selector (EN/FR for PDF output)
- **Live preview panel** (right side): rendered invoice preview updates as you type
- Download PDF → hits `/api/invoices/[id]/pdf`

### `/auth` — Login
- Supabase magic link: enter email → check inbox → click link → authenticated
- AgenteDirect branded: gradient logo, dark background, Oswald heading


### `/settings`
- Default Google Calendar booking URL (used when no per-lead override)
- Default language (EN/FR)
- Twilio from-number display (read-only, set via env var)
- Magic link re-send (for testing auth)

---

## 7. Integrations

### Supabase
- Browser client (`createBrowserClient`) for real-time subscriptions and client components
- Server client (`createServerClient`) for server components and API routes
- RLS policies: all tables require `auth.uid() = user_id`
- Real-time channel: `supabase.channel('leads').on('postgres_changes', ...)` in Kanban component

### Google Calendar
- No OAuth. User stores their personal booking URL once in Settings.
- Per-lead override stored in `leads.booking_link`.
- "Book a Call" button opens: `lead.booking_link ?? settings.default_calendar_link`

### Twilio
**Outbound SMS** (`POST /api/twilio/sms`):
- Body: `{ lead_id, message, phone }`
- Sends via `twilio.messages.create()`
- On success: writes to `follow_ups` (status: sent) + `activity_log` (type: sms_sent)

**Inbound call logging** (`POST /api/twilio/call-log`):
- Twilio webhook configured in Twilio console to hit this endpoint
- Matches `From` number to a `leads.phone` or `clients.phone`
- Writes to `activity_log` (type: call_logged, body includes duration + caller)

**Env vars (Vercel):**
```
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_FROM_NUMBER
```

### Chatbot Webhook (`POST /api/webhook/lead`)
- Protected by `X-Webhook-Secret` header matched against `WEBHOOK_SECRET` env var
- Accepts: `{ name, email, phone, company, message, lang }`
- Creates `leads` row: `stage: "new"`, `source: "website_widget"`, `lang_pref: lang ?? "en"`
- Writes to `activity_log`: `type: lead_created`
- Returns `201` with created lead `id`
- Supabase real-time fires → Kanban updates instantly

### Invoice PDF (`GET /api/invoices/[id]/pdf`)
- Server-side `@react-pdf/renderer` renders `InvoicePDF` component
- Returns `application/pdf` response with `Content-Disposition: attachment`
- PDF contents:
  - Header: gradient bar (135deg purple→blue), AgenteDirect logo, "INVOICE / FACTURE"
  - Invoice #, issued date, due date
  - Bill To: client name, company, email
  - Line items table (description, amount)
  - Subtotal, tax line (if applicable), **Total**
  - Footer: AgenteDirect · Montreal, QC · agentedirect.ca
  - Language: all labels in EN or FR based on `invoice.lang`

### Vercel Cron (`vercel.json`)
```json
{
  "crons": [{
    "path": "/api/cron/follow-ups",
    "schedule": "0 * * * *"
  }]
}
```
- Runs hourly; queries `follow_ups` where `scheduled_at <= now() AND status = 'pending'`
- Fires Twilio SMS for each, marks `status: sent`, logs to `activity_log`

---

## 8. Bilingual (FR/EN)

- `messages/en.json` and `messages/fr.json` cover all UI labels, stage names, status badges, button text, invoice labels, error messages
- Toggle in sidebar bottom; locale stored in `localStorage` and passed to `next-intl` provider
- Invoice PDF renders in the language set on the invoice (`invoice.lang`), independent of UI locale
- Chatbot-created leads inherit `lang_pref` from webhook payload

---

## 9. Auth

- Supabase magic link (email OTP) — no password
- Single user — no roles, no invite system
- Middleware (`middleware.ts`): redirect unauthenticated requests to `/auth`
- Session managed by Supabase SSR cookies

---

## 10. Environment Variables

```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=      # server-only (PDF API, cron)

# Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=

# Webhook
WEBHOOK_SECRET=                 # shared secret for chatbot webhook

# App
NEXT_PUBLIC_APP_URL=https://crm.agentedirect.ca
```

---

## 11. Deployment

- **Vercel project** connected to `agentedirect-crm` GitHub repo
- **Custom domain:** `crm.agentedirect.ca` → add CNAME in DNS to `cname.vercel-dns.com`
- **Production branch:** `main`
- All env vars set in Vercel dashboard → Production environment
- Vercel Cron enabled (Hobby plan supports cron)

---

## 12. Out of Scope (v1)

- Team roles / multi-user
- Email sending (only SMS via Twilio for now)
- Google Calendar OAuth / two-way sync
- Stripe / payment processing
- Mobile native app
- Reporting / analytics dashboard

---

## 13. Success Criteria

- Lead appears on Kanban within 2 seconds of chatbot POST
- Drag-and-drop stage change persists on reload
- Invoice PDF downloads correctly in both EN and FR
- FR/EN toggle switches all UI labels instantly
- SMS follow-up fires within 1 hour of scheduled time (cron)
- App loads at `crm.agentedirect.ca` with valid SSL
- All pages mobile-responsive (usable on iPhone)
