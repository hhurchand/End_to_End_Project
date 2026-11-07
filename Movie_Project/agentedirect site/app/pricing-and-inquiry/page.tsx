'use client';

import { useState, useRef, useCallback, type ReactNode } from 'react';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import Link from 'next/link';

const ease = [0.16, 1, 0.3, 1] as const;

// ── Configure recipient here ──────────────────────────────────────────────────
const BUSINESS_EMAIL = 'support@agentedirect.ca';

const PLANS = [
  {
    id: 'basic' as const,
    name: 'Custom Website',
    price: '$199 CAD',
    period: '/month',
    description: 'A professional, custom-designed website built around your business.',
    features: [
      'Built exactly to your request',
      'No limit on number of pages',
      'Responsive, mobile-friendly layout',
      'Clear calls to action',
      'SEO-friendly structure',
    ],
    formLabel: 'Website (No Chatbot)',
  },
  {
    id: 'chatbot' as const,
    name: 'Website + AI Chatbot',
    price: '$249 CAD',
    period: '/month',
    description: 'Everything in the Custom Website plan, with an AI chatbot to help visitors get answers.',
    features: [
      'Everything in Custom Website',
      'AI chatbot integration',
      'Answers to common customer questions',
      'Helps guide visitors toward contacting your business',
    ],
    formLabel: 'Website + AI Chatbot',
  },
] as const;

type PlanId = (typeof PLANS)[number]['id'];

interface FormState {
  name: string;
  businessName: string;
  email: string;
  currentUrl: string;
  businessDescription: string;
  plan: PlanId;
}

interface FormErrors {
  name?: string;
  businessName?: string;
  email?: string;
  businessDescription?: string;
}

// ── FAQ ───────────────────────────────────────────────────────────────────────
const FAQ: Array<{ q: string; a: string }> = [
  {
    q: 'Is there a setup fee?',
    a: 'No setup fee. Your first month covers everything — we get started as soon as you sign up.',
  },
  {
    q: 'Is hosting included?',
    a: 'Yes. Your monthly plan covers hosting, maintenance, and technical support — no extra fees.',
  },
  {
    q: 'Do I own my domain?',
    a: 'Always. If you have a domain, it stays in your name. If you need one, we help register it and ownership is yours.',
  },
  {
    q: 'Can I cancel?',
    a: 'Yes, month-to-month. Cancel anytime with 30 days\' notice — no long-term contracts. Your site stays live through the end of your paid period.',
  },
  {
    q: 'What is the AI chatbot?',
    a: 'A custom assistant trained on your business info. It answers common questions — services, hours, pricing — directly on your site and guides visitors toward reaching out.',
  },
];

const PLANS_FR = [
  {
    id: 'basic' as const,
    name: 'Site web sur mesure',
    price: '199 $ CA',
    period: '/mois',
    description: 'Un site web professionnel sur mesure conçu autour de votre entreprise.',
    features: [
      'Conçu exactement selon vos demandes',
      'Aucune limite sur le nombre de pages',
      'Mise en page responsive et mobile',
      'Appels à l\'action clairs',
      'Structure optimisée pour le SEO',
    ],
    formLabel: 'Site web (sans chatbot)',
  },
  {
    id: 'chatbot' as const,
    name: 'Site web + Chatbot IA',
    price: '249 $ CA',
    period: '/mois',
    description: 'Tout ce qui est inclus dans le forfait Site web sur mesure, plus un chatbot IA pour aider les visiteurs.',
    features: [
      'Tout ce qui est inclus dans Site web sur mesure',
      'Intégration d\'un chatbot IA',
      'Répond aux questions fréquentes des clients',
      'Guide les visiteurs vers votre entreprise',
    ],
    formLabel: 'Site web + Chatbot IA',
  },
];

const FAQ_FR: Array<{ q: string; a: string }> = [
  { q: 'Y a-t-il des frais d\'installation ?', a: 'Aucuns frais d\'installation. Votre premier mois couvre tout — on commence dès que vous vous inscrivez.' },
  { q: 'L\'hébergement est-il inclus ?', a: 'Oui. Votre forfait mensuel couvre l\'hébergement, la maintenance et le support technique — sans frais supplémentaires.' },
  { q: 'Est-ce que je suis propriétaire de mon domaine ?', a: 'Toujours. Si vous avez déjà un domaine, il reste à votre nom. Si vous en avez besoin, on vous aide à l\'enregistrer et il vous appartient.' },
  { q: 'Puis-je annuler ?', a: 'Oui, mois par mois. Annulez en tout temps avec un préavis de 30 jours — sans contrat à long terme. Votre site reste en ligne jusqu\'à la fin de la période payée.' },
  { q: 'C\'est quoi le chatbot IA ?', a: 'Un assistant personnalisé formé sur les informations de votre entreprise. Il répond aux questions courantes — services, horaires, tarifs — directement sur votre site et guide les visiteurs à vous contacter.' },
];

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function validate(f: FormState): FormErrors {
  const e: FormErrors = {};
  if (!f.name.trim())                e.name = 'Full name is required.';
  if (!f.businessName.trim())        e.businessName = 'Business name is required.';
  if (!f.email.trim())               e.email = 'Email address is required.';
  else if (!EMAIL_RE.test(f.email))  e.email = 'Please enter a valid email address.';
  if (!f.businessDescription.trim()) e.businessDescription = 'Please tell us what your business does.';
  return e;
}

function buildMailto(f: FormState): string {
  const plan = PLANS.find(p => p.id === f.plan)!;
  const lines = [
    `Hi AgenteDirect,`,
    ``,
    `My name is ${f.name} and I'm reaching out about getting a website built for my business.`,
    ``,
    `--- MY DETAILS ---`,
    `Name: ${f.name}`,
    `Business: ${f.businessName}`,
    `Email: ${f.email}`,
    `Current Website: ${f.currentUrl || 'None'}`,
    ``,
    `--- ABOUT MY BUSINESS ---`,
    f.businessDescription,
    ``,
    `--- SELECTED PLAN ---`,
    plan.formLabel,
    ``,
    `Looking forward to hearing from you.`,
    ``,
    f.name,
  ];
  return `mailto:${BUSINESS_EMAIL}?subject=${encodeURIComponent(`Website Inquiry — ${f.businessName}`)}&body=${encodeURIComponent(lines.join('\n'))}`;
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <circle cx="7" cy="7" r="6.5" stroke="rgba(139,92,246,0.35)" />
      <path d="M4 7l2 2 4-4" stroke="#A78BFA" strokeWidth="1.4"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FieldGroup({ label, required, hint, error, children }: {
  label: string; required?: boolean; hint?: string; error?: string; children: ReactNode;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <label style={{
        fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
        fontSize: 12.5, fontWeight: 600,
        color: 'rgba(220,210,245,0.65)', letterSpacing: '0.01em',
        display: 'flex', alignItems: 'center', gap: 5,
      }}>
        {label}
        {required && <span style={{ color: 'rgba(139,92,246,0.70)' }}>*</span>}
        {hint && <span style={{ color: 'rgba(220,210,245,0.30)', fontWeight: 400, fontSize: 12 }}>— {hint}</span>}
      </label>
      {children}
      {error && (
        <p style={{ fontSize: 12, color: 'rgba(239,68,68,0.82)', margin: 0, lineHeight: 1.4 }}>{error}</p>
      )}
    </div>
  );
}

export default function PricingPage({ lang = 'en', formOnly = false }: { lang?: 'en' | 'fr'; formOnly?: boolean } = {}) {
  const fr = lang === 'fr';
  const reduced = useReducedMotion();

  const [form, setForm] = useState<FormState>({
    name: '', businessName: '', email: '',
    currentUrl: '', businessDescription: '', plan: 'basic',
  });
  const [errors, setErrors] = useState<FormErrors>({});

  const formRef = useRef<HTMLElement>(null);
  const ctaRef  = useRef<HTMLElement>(null);

  const formInView = useInView(formRef, { once: true, margin: '-60px' });
  const ctaInView  = useInView(ctaRef,  { once: true, margin: '-60px' });

  const setField = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
    if (key in errors) setErrors(prev => { const n = { ...prev }; delete n[key as keyof FormErrors]; return n; });
  }, [errors]);

  const handleSubmit = useCallback(() => {
    const errs = validate(form);
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      document.getElementById(`field-${Object.keys(errs)[0]}`)?.focus();
      return;
    }
    window.location.href = buildMailto(form);
  }, [form]);

  const inputBase = (hasError: boolean): React.CSSProperties => ({
    width: '100%',
    background: 'rgba(255,255,255,0.03)',
    border: `1px solid ${hasError ? 'rgba(239,68,68,0.55)' : 'rgba(255,255,255,0.08)'}`,
    borderRadius: 10, padding: '12px 16px',
    fontSize: 14, color: '#F5F0FC',
    outline: 'none', transition: 'border-color 0.2s',
    fontFamily: "var(--font-inter), system-ui, sans-serif",
  });

  const onInputFocus = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.target.style.borderColor = 'rgba(139,92,246,0.52)';
  };
  const onInputBlur = (hasError: boolean) =>
    (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      e.target.style.borderColor = hasError ? 'rgba(239,68,68,0.55)' : 'rgba(255,255,255,0.08)';
    };

  return (
    <main style={{ background: '#08041A', minHeight: '100dvh', ...(formOnly ? { overflow: 'hidden', height: '100dvh' } : {}) }}>

      {/* Film grain */}
      <svg aria-hidden style={{
        position: 'fixed', inset: 0, width: '100%', height: '100%',
        pointerEvents: 'none', opacity: 0.022, zIndex: 0,
      }}>
        <filter id="pi-grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.78" numOctaves="4" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#pi-grain)" />
      </svg>

      {/* ── NAV ── */}
      {!formOnly && <motion.nav
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.50, ease }}
        style={{
          position: 'relative', zIndex: 20,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 clamp(24px, 5vw, 56px)',
          height: 64, flexShrink: 0,
          borderBottom: '1px solid rgba(139,92,246,0.07)',
        }}
      >
        <Link href={fr ? '/fr/home' : '/home'} style={{ display: 'flex', alignItems: 'center', lineHeight: 0 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="AgenteDirect"
            style={{ height: 34, width: 'auto', filter: 'brightness(0) invert(1)', opacity: 0.88 }} />
        </Link>

        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          {/* FR / EN toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Link
              href="/pricing-and-inquiry"
              style={{
                fontSize: 12, fontWeight: 700, letterSpacing: '0.08em',
                color: fr ? 'rgba(220,210,245,0.38)' : '#A78BFA',
                textDecoration: 'none', padding: '4px 6px', borderRadius: 4,
                transition: 'color 0.2s',
              }}
            >
              EN
            </Link>
            <span style={{ color: 'rgba(139,92,246,0.25)', fontSize: 11 }}>|</span>
            <Link
              href="/fr/pricing-and-inquiry"
              style={{
                fontSize: 12, fontWeight: 700, letterSpacing: '0.08em',
                color: fr ? '#A78BFA' : 'rgba(220,210,245,0.38)',
                textDecoration: 'none', padding: '4px 6px', borderRadius: 4,
                transition: 'color 0.2s',
              }}
            >
              FR
            </Link>
          </div>

          <Link
            href={fr ? '/fr/home' : '/home'}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              fontSize: 13, fontWeight: 600, color: 'rgba(220,210,245,0.50)',
              textDecoration: 'none', letterSpacing: '0.01em', transition: 'color 0.2s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLAnchorElement).style.color = '#A78BFA'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(220,210,245,0.50)'; }}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
              <path d="M9 11.5L4.5 7L9 2.5" stroke="currentColor" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {fr ? 'Retour à l\'accueil' : 'Back to home'}
          </Link>
        </div>
      </motion.nav>}

      {/* ── FORM + FAQ — same section, two columns ── */}
      <section
        id="inquiry"
        ref={formRef}
        aria-label="Project inquiry"
        style={{
          position: 'relative',
          ...(formOnly ? { height: '100dvh', overflow: 'auto' } : { minHeight: 'calc(100dvh - 64px)' }),
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div aria-hidden style={{
          position: 'absolute', top: 0, left: '6%', right: '6%', height: 1,
          background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.18) 25%, rgba(139,92,246,0.18) 75%, transparent)',
        }} />
        <div aria-hidden style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse 60% 50% at 50% 50%, rgba(109,40,217,0.05) 0%, transparent 70%)',
        }} />

        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          position: 'relative', zIndex: 1,
          padding: 'clamp(48px, 6vh, 80px) clamp(24px, 5vw, 56px)',
        }}>
        <div
          className="pi-form-faq"
          style={{
            maxWidth: 1160,
            margin: '0 auto',
            width: '100%',
            display: 'grid',
            gridTemplateColumns: '3fr 2fr',
            gap: 'clamp(32px, 5vw, 72px)',
            alignItems: 'start',
          }}
        >

          {/* ── Left: Form ── */}
          <div className="pi-form-left" style={{ display: 'flex', flexDirection: 'column', paddingTop: 'clamp(64px, 12vh, 140px)' }}>
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={formInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.60, ease }}
              style={{ marginBottom: 'clamp(36px, 6vh, 60px)' }}
            >
              <p style={{
                fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                fontSize: 10, fontWeight: 700, letterSpacing: '0.24em',
                textTransform: 'uppercase', color: '#8B5CF6', margin: '0 0 10px',
              }}>
                {fr ? 'Demande' : 'Inquiry'}
              </p>
              <h2 style={{
                fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                fontSize: 'clamp(22px, 2.8vw, 36px)',
                fontWeight: 800, color: '#F5F0FC',
                letterSpacing: '-0.04em', lineHeight: 1.07, margin: 0,
              }}>
                {fr ? 'Parlez-nous de votre entreprise.' : 'Tell us about your business.'}
              </h2>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 22 }}
              animate={formInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.60, ease, delay: 0.10 }}
              style={{ display: 'flex', flexDirection: 'column', gap: 14 }}
            >
              <style>{`
                @media (max-width: 520px) {
                  .pi-row2 { grid-template-columns: 1fr !important; }
                }
                @media (max-width: 767px) {
                  .pi-form-faq { grid-template-columns: 1fr !important; }
                  .pi-form-left { padding-top: 16px !important; }
                }
              `}</style>

              {/* Row 1: Name + Business */}
              <div className="pi-row2" style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr',
                gap: 14,
              }}>
                <FieldGroup label={fr ? 'Nom complet' : 'Full Name'} required error={errors.name}>
                  <input id="field-name" type="text" autoComplete="name"
                    placeholder="Your name" value={form.name}
                    onChange={e => setField('name', e.target.value)}
                    style={inputBase(!!errors.name)}
                    onFocus={onInputFocus} onBlur={onInputBlur(!!errors.name)} />
                </FieldGroup>
                <FieldGroup label={fr ? 'Nom de l\'entreprise' : 'Business Name'} required error={errors.businessName}>
                  <input id="field-businessName" type="text" autoComplete="organization"
                    placeholder="Your business" value={form.businessName}
                    onChange={e => setField('businessName', e.target.value)}
                    style={inputBase(!!errors.businessName)}
                    onFocus={onInputFocus} onBlur={onInputBlur(!!errors.businessName)} />
                </FieldGroup>
              </div>

              {/* Row 2: Email + URL */}
              <div className="pi-row2" style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr',
                gap: 14,
              }}>
                <FieldGroup label={fr ? 'Adresse courriel' : 'Email Address'} required error={errors.email}>
                  <input id="field-email" type="email" autoComplete="email"
                    placeholder="you@example.com" value={form.email}
                    onChange={e => setField('email', e.target.value)}
                    style={inputBase(!!errors.email)}
                    onFocus={onInputFocus} onBlur={onInputBlur(!!errors.email)} />
                </FieldGroup>
                <FieldGroup label={fr ? 'Site web actuel' : 'Current Website'} hint={fr ? 'Facultatif' : 'Optional'}>
                  <input id="field-currentUrl" type="url" autoComplete="url"
                    placeholder="https://yourbusiness.com" value={form.currentUrl}
                    onChange={e => setField('currentUrl', e.target.value)}
                    style={inputBase(false)}
                    onFocus={onInputFocus} onBlur={onInputBlur(false)} />
                </FieldGroup>
              </div>

              {/* Row 3: Description + Plan */}
              <div className="pi-row2" style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr',
                gap: 14, alignItems: 'start',
              }}>
                <FieldGroup label={fr ? 'Que fait votre entreprise\u00a0?' : 'What does your business do?'} required error={errors.businessDescription}>
                  <textarea id="field-businessDescription" rows={3}
                    placeholder="e.g. We're a plumbing company in Montréal specializing in residential repairs."
                    value={form.businessDescription}
                    onChange={e => setField('businessDescription', e.target.value)}
                    style={{ ...inputBase(!!errors.businessDescription), resize: 'none', lineHeight: 1.60 }}
                    onFocus={onInputFocus} onBlur={onInputBlur(!!errors.businessDescription)} />
                </FieldGroup>

                <div>
                  <p style={{
                    fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                    fontSize: 12.5, fontWeight: 600,
                    color: 'rgba(220,210,245,0.65)', margin: '0 0 7px', letterSpacing: '0.01em',
                  }}>
                    {fr ? 'Forfait souhaité' : 'Website Preference'} <span style={{ color: 'rgba(139,92,246,0.70)' }}>*</span>
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}
                    role="radiogroup" aria-label="Website preference">
                    {(fr ? PLANS_FR : PLANS).map(plan => {
                      const chosen = form.plan === plan.id;
                      return (
                        <div
                          key={plan.id}
                          role="radio" aria-checked={chosen} tabIndex={0}
                          onClick={() => setField('plan', plan.id)}
                          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setField('plan', plan.id); } }}
                          style={{
                            display: 'flex', alignItems: 'center', gap: 12,
                            padding: '11px 16px', borderRadius: 10,
                            border: `1px solid ${chosen ? 'rgba(139,92,246,0.50)' : 'rgba(255,255,255,0.07)'}`,
                            background: chosen ? 'rgba(109,40,217,0.07)' : 'rgba(255,255,255,0.02)',
                            cursor: 'pointer', outline: 'none',
                            transition: 'border-color 0.22s, background 0.22s',
                          }}
                        >
                          <div style={{
                            width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
                            border: `1.5px solid ${chosen ? '#8B5CF6' : 'rgba(255,255,255,0.18)'}`,
                            background: chosen ? '#7C3AED' : 'transparent',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            transition: 'border-color 0.22s, background 0.22s',
                          }}>
                            {chosen && <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#fff' }} />}
                          </div>
                          <span style={{
                            fontSize: 13, lineHeight: 1.4,
                            color: chosen ? '#F5F0FC' : 'rgba(220,210,245,0.52)',
                            fontWeight: chosen ? 600 : 400,
                            transition: 'color 0.22s',
                          }}>
                            {plan.formLabel}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Submit row */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={handleSubmit}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 8,
                    padding: '13px 28px', borderRadius: 50,
                    background: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
                    color: '#fff', fontSize: 14, fontWeight: 700,
                    border: 'none', cursor: 'pointer', letterSpacing: '0.01em',
                    boxShadow: '0 4px 24px rgba(109,40,217,0.44)',
                    transition: 'background 0.22s, box-shadow 0.22s, transform 0.18s',
                    flexShrink: 0,
                  }}
                  onMouseEnter={e => {
                    const el = e.currentTarget as HTMLButtonElement;
                    el.style.background = 'linear-gradient(135deg, #9D6FF5 0%, #7C3AED 100%)';
                    el.style.boxShadow = '0 8px 36px rgba(109,40,217,0.60)';
                    el.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={e => {
                    const el = e.currentTarget as HTMLButtonElement;
                    el.style.background = 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)';
                    el.style.boxShadow = '0 4px 24px rgba(109,40,217,0.44)';
                    el.style.transform = 'translateY(0)';
                  }}
                >
                  {fr ? 'Préparer ma demande' : 'Prepare My Inquiry'}
                  <svg width="12" height="12" viewBox="0 0 13 13" fill="none" aria-hidden>
                    <path d="M2 11L11 2M11 2H5.5M11 2V7.5"
                      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>

                <p style={{
                  fontSize: 12, color: 'rgba(220,210,245,0.26)',
                  lineHeight: 1.55, margin: 0,
                }}>
                  {fr ? 'Ouvre votre application courriel avec la demande préremplie.' : 'Opens your email app with the inquiry pre-filled.'}
                </p>
              </div>
            </motion.div>
          </div>

          {/* ── Right: FAQ ── */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={formInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.60, ease, delay: 0.05 }}
              style={{ marginBottom: 'clamp(12px, 1.5vh, 18px)' }}
            >
              <p style={{
                fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                fontSize: 10, fontWeight: 700, letterSpacing: '0.24em',
                textTransform: 'uppercase', color: '#8B5CF6', margin: '0 0 10px',
              }}>
                FAQ
              </p>
              <h2 style={{
                fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                fontSize: 'clamp(18px, 2vw, 26px)',
                fontWeight: 800, color: '#F5F0FC',
                letterSpacing: '-0.035em', lineHeight: 1.1, margin: 0,
              }}>
                {fr ? 'Questions fréquentes.' : 'Common questions.'}
              </h2>
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {(fr ? FAQ_FR : FAQ).map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 14 }}
                  animate={formInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.50, ease, delay: 0.12 + i * 0.06 }}
                  style={{
                    paddingTop: i === 0 ? 0 : 18,
                    paddingBottom: 18,
                    borderBottom: i < (fr ? FAQ_FR : FAQ).length - 1
                      ? '1px solid rgba(139,92,246,0.08)'
                      : 'none',
                  }}
                >
                  <p style={{
                    fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                    fontSize: 'clamp(13px, 1.1vw, 14.5px)',
                    fontWeight: 700, color: '#F5F0FC',
                    letterSpacing: '-0.01em', lineHeight: 1.3, margin: '0 0 7px',
                  }}>
                    {item.q}
                  </p>

                  <p style={{
                    fontSize: 'clamp(12px, 0.95vw, 13.5px)',
                    color: 'rgba(220,210,245,0.48)',
                    lineHeight: 1.70, margin: 0,
                  }}>
                    {item.a}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>

        </div>{/* end pi-form-faq grid */}
        </div>{/* end flex:1 centering wrapper */}
      </section>

      {/* ── BOTTOM CTA ── */}
      <section
        ref={ctaRef}
        aria-label="Get in touch"
        style={{
          position: 'relative',
          paddingTop: 'clamp(36px, 5vw, 60px)',
          paddingBottom: 'clamp(36px, 5vw, 60px)',
          textAlign: 'center',
        }}
      >
        <div aria-hidden style={{
          position: 'absolute', top: 0, left: '6%', right: '6%', height: 1,
          background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.10) 25%, rgba(139,92,246,0.10) 75%, transparent)',
        }} />

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={ctaInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.52, ease }}
          style={{ padding: '0 clamp(24px, 5vw, 56px)' }}
        >
          <p style={{
            fontSize: 'clamp(14px, 1.3vw, 16px)',
            color: 'rgba(220,210,245,0.36)', lineHeight: 1.75, margin: '0 0 10px',
          }}>
            {fr ? 'Une question avant de commencer\u00a0?' : 'Have a question before getting started?'}
          </p>
          <a
            href={`mailto:${BUSINESS_EMAIL}`}
            style={{
              fontSize: 'clamp(13.5px, 1.15vw, 15.5px)',
              color: '#A78BFA', fontWeight: 600,
              textDecoration: 'none', letterSpacing: '0.01em', transition: 'color 0.2s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLAnchorElement).style.color = '#C4B5FD'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLAnchorElement).style.color = '#A78BFA'; }}
          >
            {fr ? 'Contactez-nous —' : 'Get in touch —'} {BUSINESS_EMAIL}
          </a>
        </motion.div>
      </section>

      {/* ── FOOTER ── */}
      <footer style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '0 24px', height: 40,
        borderTop: '1px solid rgba(139,92,246,0.05)',
      }}>
        <p style={{ fontSize: 10, color: 'rgba(220,210,245,0.16)', letterSpacing: '0.05em', margin: 0 }}>
          © 2026 AgenteDirect · Montréal, Québec
        </p>
      </footer>

    </main>
  );
}
