'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState, useCallback, type ReactNode } from 'react';
import ChatWidget from '@/components/ChatWidget';
import PageNav from '@/components/PageNav';
import PageFooter from '@/components/PageFooter';

const ease = [0.16, 1, 0.3, 1] as const;
const BUSINESS_EMAIL = 'support@agentedirect.ca';
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const T = {
  en: {
    desktop: 'Desktop',
    beliefs: 'Our Beliefs',
    projects: 'Our Projects',
    contact: 'Contact',
    eyebrow: "Let's talk.",
    heading: 'Tell us about your business.',
    labelName: 'Full Name',
    labelBusiness: 'Business Name',
    labelEmail: 'Email Address',
    labelUrl: 'Current Website',
    hintUrl: 'Optional',
    labelDesc: "What's your vision for the site?",
    placeholderName: 'Your name',
    placeholderBusiness: 'Your business',
    placeholderDesc: 'e.g. Something clean and modern, easy to navigate, with a focus on trust and getting people to call us.',
    labelPlan: 'What are you interested in?',
    optLaunch: 'Website Launch',
    optGrowth: 'Growth System',
    optPro: 'Growth Pro',
    descLaunch: 'A professionally designed, custom-built website. Mobile-first, bilingual, SEO-ready and built around your brand. The foundation every business needs.',
    descGrowth: 'Everything in Website Launch, plus an AI chatbot trained on your business that qualifies leads, answers questions and captures opportunities around the clock.',
    descPro: 'Full digital growth strategy. Advanced local SEO, city pages, Google Business management, review generation and monthly reporting with a dedicated strategist.',
    submit: 'Send Inquiry',
    submitHint: 'Opens your email app with the inquiry pre-filled.',
    preferEmail: 'Prefer email?',
    errName: 'Full name is required.',
    errBusiness: 'Business name is required.',
    errEmail: 'Email address is required.',
    errEmailInvalid: 'Please enter a valid email address.',
    errDesc: 'Please tell us what your business does.',
    emailSubject: (biz: string) => `Website Inquiry: ${biz}`,
    emailGreeting: (name: string) => `Hi AgenteDirect,\n\nMy name is ${name} and I'm reaching out about getting a website built.`,
    emailSiteLabel: 'Current Website',
    emailNone: 'None',
    emailAbout: 'About my business:',
    emailInterested: 'Interested in:',
    emailClosing: 'Looking forward to hearing from you.',
    planLaunch: 'Website Launch',
    planGrowth: 'Growth System',
    planPro: 'Growth Pro',
  },
  fr: {
    desktop: 'Bureau',
    beliefs: 'Nos croyances',
    projects: 'Nos projets',
    contact: 'Contact',
    eyebrow: 'Parlons-en.',
    heading: 'Parlez-nous de votre entreprise.',
    labelName: 'Nom complet',
    labelBusiness: "Nom de l'entreprise",
    labelEmail: 'Adresse courriel',
    labelUrl: 'Site web actuel',
    hintUrl: 'Facultatif',
    labelDesc: 'Quelle est votre vision pour le site?',
    placeholderName: 'Votre nom',
    placeholderBusiness: 'Votre entreprise',
    placeholderDesc: 'Ex. Quelque chose de propre et moderne, facile à naviguer, axé sur la confiance et incitant les gens à nous appeler.',
    labelPlan: "Qu'est-ce qui vous intéresse?",
    optLaunch: 'Lancement Web',
    optGrowth: 'Système de croissance',
    optPro: 'Croissance Pro',
    descLaunch: 'Un site web conçu sur mesure. Mobile-first, bilingue, prêt pour Google et construit autour de votre marque. La fondation dont chaque entreprise a besoin.',
    descGrowth: 'Tout ce qui est inclus dans Lancement Web, plus un chatbot IA formé sur votre entreprise qui qualifie les prospects, répond aux questions et capte les opportunités en tout temps.',
    descPro: 'Stratégie de croissance numérique complète. SEO local avancé, pages par ville, gestion Google Business, génération d\'avis et rapports mensuels avec un stratège dédié.',
    submit: 'Envoyer la demande',
    submitHint: 'Ouvre votre application courriel avec la demande pré-remplie.',
    preferEmail: 'Préférez le courriel?',
    errName: 'Le nom complet est requis.',
    errBusiness: "Le nom de l'entreprise est requis.",
    errEmail: "L'adresse courriel est requise.",
    errEmailInvalid: 'Veuillez entrer une adresse courriel valide.',
    errDesc: 'Veuillez nous dire ce que fait votre entreprise.',
    emailSubject: (biz: string) => `Demande de site web: ${biz}`,
    emailGreeting: (name: string) => `Bonjour AgenteDirect,\n\nJe m'appelle ${name} et je vous contacte pour la création d'un site web.`,
    emailSiteLabel: 'Site web actuel',
    emailNone: 'Aucun',
    emailAbout: 'À propos de mon entreprise:',
    emailInterested: 'Intéressé par:',
    emailClosing: "Dans l'attente de votre réponse.",
    planLaunch: 'Lancement Web',
    planGrowth: 'Système de croissance',
    planPro: 'Croissance Pro',
  },
};

interface FormState {
  name: string;
  businessName: string;
  email: string;
  currentUrl: string;
  businessDescription: string;
  plan: 'launch' | 'growth' | 'pro';
}
interface FormErrors {
  name?: string;
  businessName?: string;
  email?: string;
  businessDescription?: string;
}

function validate(f: FormState, t: typeof T.en): FormErrors {
  const e: FormErrors = {};
  if (!f.name.trim())                e.name = t.errName;
  if (!f.businessName.trim())        e.businessName = t.errBusiness;
  if (!f.email.trim())               e.email = t.errEmail;
  else if (!EMAIL_RE.test(f.email))  e.email = t.errEmailInvalid;
  if (!f.businessDescription.trim()) e.businessDescription = t.errDesc;
  return e;
}


function FieldGroup({ label, required, hint, error, children }: {
  label: string; required?: boolean; hint?: string; error?: string; children: ReactNode;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <label style={{ fontSize: 12.5, fontWeight: 600, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.01em', display: 'flex', alignItems: 'center', gap: 5 }}>
        {label}
        {required && <span style={{ color: 'rgba(139,92,246,0.8)' }}>*</span>}
        {hint && <span style={{ color: 'rgba(255,255,255,0.25)', fontWeight: 400, fontSize: 12 }}>{hint}</span>}
      </label>
      {children}
      {error && <p style={{ fontSize: 12, color: 'rgba(239,68,68,0.82)', margin: 0 }}>{error}</p>}
    </div>
  );
}

export default function InquiryPage({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const t = lang === 'fr' ? T.fr : T.en;
  const fr = lang === 'fr';

  const [form, setForm] = useState<FormState>({
    name: '', businessName: '', email: '',
    currentUrl: '', businessDescription: '', plan: 'launch',
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [expandedPlan, setExpandedPlan] = useState<'launch' | 'growth' | 'pro' | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const setField = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
    if (key in errors) setErrors(prev => { const n = { ...prev }; delete n[key as keyof FormErrors]; return n; });
  }, [errors]);

  const handleSubmit = useCallback(async () => {
    const errs = validate(form, t);
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      document.getElementById(`field-${Object.keys(errs)[0]}`)?.focus();
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await fetch('/api/inquiry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, lang }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? 'Something went wrong.');
      setSubmitted(true);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Something went wrong. Please email us directly.');
    } finally {
      setSubmitting(false);
    }
  }, [form, t, lang]);

  const inputStyle = (hasError: boolean): React.CSSProperties => ({
    width: '100%',
    background: 'rgba(255,255,255,0.03)',
    border: `1px solid ${hasError ? 'rgba(239,68,68,0.55)' : 'rgba(255,255,255,0.08)'}`,
    borderRadius: 10, padding: '12px 16px',
    fontSize: 14, color: 'rgba(255,255,255,0.9)',
    outline: 'none', transition: 'border-color 0.2s',
    fontFamily: 'var(--font-space-grotesk), system-ui, sans-serif',
    boxSizing: 'border-box',
  });

  const onFocus = (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.target.style.borderColor = 'rgba(139,92,246,0.5)';
  };
  const onBlur = (hasError: boolean) => (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    e.target.style.borderColor = hasError ? 'rgba(239,68,68,0.55)' : 'rgba(255,255,255,0.08)';
  };

  return (
    <main
      className="min-h-screen flex flex-col"
      style={{ background: '#0F0C07', fontFamily: 'var(--font-space-grotesk)' }}
    >
      <PageNav lang={lang} active="contact" enHref="/inquiry" frHref="/fr/inquiry" />

      {/* Form */}
      <div className="flex-1 flex items-start justify-center px-6 pt-10 pb-16 md:pt-14 md:pb-24">
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease }}
          style={{ maxWidth: '600px', width: '100%' }}
        >
          {submitted ? (
            <div style={{ textAlign: 'center', paddingTop: 40 }}>
              <div style={{ fontSize: 40, marginBottom: 24 }}>
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none" style={{ margin: '0 auto', display: 'block' }}>
                  <circle cx="24" cy="24" r="23" stroke="rgba(139,92,246,0.5)" strokeWidth="1.5" />
                  <path d="M14 24l7 7 13-13" stroke="rgba(167,139,250,0.9)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <h2 style={{ fontSize: 'clamp(24px,3vw,32px)', fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: '0 0 16px', letterSpacing: '-0.02em' }}>
                {fr ? 'On a bien reçu votre demande.' : 'Got it. We\'ll be in touch soon.'}
              </h2>
              <p style={{ fontSize: 16, color: 'rgba(255,255,255,0.45)', lineHeight: 1.75, maxWidth: 400, margin: '0 auto' }}>
                {fr
                  ? 'Merci de nous avoir contactés. On vous reviendra rapidement pour discuter de votre projet.'
                  : 'Thanks for reaching out. We\'ll get back to you shortly to talk about your project.'}
              </p>
            </div>
          ) : (
          <>
          <p style={{ fontSize: '15px', lineHeight: 1.7, marginBottom: '10px', color: 'rgba(255,255,255,0.45)', fontStyle: 'italic' }}>
            {t.eyebrow}
          </p>
          <h1 style={{ fontSize: 'clamp(28px, 4vw, 42px)', fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: '0 0 36px', lineHeight: 1.1, letterSpacing: '-0.02em' }}>
            {t.heading}
          </h1>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Name + Business */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }} className="form-row">
              <FieldGroup label={t.labelName} required error={errors.name}>
                <input id="field-name" type="text" autoComplete="name" placeholder={t.placeholderName}
                  value={form.name} onChange={e => setField('name', e.target.value)}
                  style={inputStyle(!!errors.name)} onFocus={onFocus} onBlur={onBlur(!!errors.name)} />
              </FieldGroup>
              <FieldGroup label={t.labelBusiness} required error={errors.businessName}>
                <input id="field-businessName" type="text" autoComplete="organization" placeholder={t.placeholderBusiness}
                  value={form.businessName} onChange={e => setField('businessName', e.target.value)}
                  style={inputStyle(!!errors.businessName)} onFocus={onFocus} onBlur={onBlur(!!errors.businessName)} />
              </FieldGroup>
            </div>

            {/* Email + URL */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }} className="form-row">
              <FieldGroup label={t.labelEmail} required error={errors.email}>
                <input id="field-email" type="email" autoComplete="email" placeholder="you@example.com"
                  value={form.email} onChange={e => setField('email', e.target.value)}
                  style={inputStyle(!!errors.email)} onFocus={onFocus} onBlur={onBlur(!!errors.email)} />
              </FieldGroup>
              <FieldGroup label={t.labelUrl} hint={t.hintUrl}>
                <input id="field-currentUrl" type="url" autoComplete="url" placeholder="https://yourbusiness.com"
                  value={form.currentUrl} onChange={e => setField('currentUrl', e.target.value)}
                  style={inputStyle(false)} onFocus={onFocus} onBlur={onBlur(false)} />
              </FieldGroup>
            </div>

            {/* Description */}
            <FieldGroup label={t.labelDesc} required error={errors.businessDescription}>
              <textarea id="field-businessDescription" rows={4}
                placeholder={t.placeholderDesc}
                value={form.businessDescription}
                onChange={e => setField('businessDescription', e.target.value)}
                style={{ ...inputStyle(!!errors.businessDescription), resize: 'none', lineHeight: 1.6 }}
                onFocus={onFocus} onBlur={onBlur(!!errors.businessDescription)} />
            </FieldGroup>

            {/* Plan */}
            <FieldGroup label={t.labelPlan} required>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  { id: 'launch' as const, label: t.optLaunch, desc: t.descLaunch },
                  { id: 'growth' as const, label: t.optGrowth, desc: t.descGrowth },
                  { id: 'pro' as const, label: t.optPro, desc: t.descPro },
                ].map(opt => {
                  const chosen = form.plan === opt.id;
                  const expanded = expandedPlan === opt.id;
                  return (
                    <div
                      key={opt.id}
                      style={{
                        borderRadius: 10,
                        border: `1px solid ${chosen ? 'rgba(139,92,246,0.5)' : 'rgba(255,255,255,0.07)'}`,
                        background: chosen ? 'rgba(139,92,246,0.08)' : 'rgba(255,255,255,0.02)',
                        transition: 'border-color 0.2s, background 0.2s',
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        role="radio" aria-checked={chosen} tabIndex={0}
                        onClick={() => setField('plan', opt.id)}
                        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setField('plan', opt.id); } }}
                        style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '11px 16px', cursor: 'pointer', outline: 'none' }}
                      >
                        <div style={{
                          width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
                          border: `1.5px solid ${chosen ? '#8B5CF6' : 'rgba(255,255,255,0.18)'}`,
                          background: chosen ? '#8B5CF6' : 'transparent',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          transition: 'all 0.2s',
                        }}>
                          {chosen && <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#fff' }} />}
                        </div>
                        <span style={{ flex: 1, fontSize: 13, color: chosen ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.45)', fontWeight: chosen ? 600 : 400, transition: 'color 0.2s' }}>
                          {opt.label}
                        </span>
                        <button
                          type="button"
                          onClick={e => { e.stopPropagation(); setExpandedPlan(expanded ? null : opt.id); }}
                          aria-label={expanded ? 'Hide description' : 'Show description'}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px 4px', display: 'flex', alignItems: 'center', color: expanded ? 'rgba(139,92,246,0.7)' : 'rgba(255,255,255,0.25)', flexShrink: 0, transition: 'color 0.2s' }}
                        >
                          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>
                            <path d="M3 5l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </button>
                      </div>
                      {expanded && (
                        <div style={{ padding: '0 16px 13px 44px', fontSize: 12.5, color: 'rgba(220,210,245,0.48)', lineHeight: 1.65 }}>
                          {opt.desc}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </FieldGroup>

            {/* Submit */}
            <div style={{ paddingTop: 8 }}>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={submitting}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 8,
                  padding: '13px 28px', borderRadius: 50,
                  background: submitting ? 'rgba(139,92,246,0.08)' : 'rgba(139,92,246,0.15)',
                  color: submitting ? 'rgba(220,210,245,0.4)' : 'rgba(220,210,245,0.95)', fontSize: 14, fontWeight: 700,
                  border: '1px solid rgba(139,92,246,0.4)', cursor: submitting ? 'default' : 'pointer', letterSpacing: '0.01em',
                  transition: 'background 0.2s, border-color 0.2s, color 0.2s',
                }}
                onMouseEnter={e => { if (!submitting) { const el = e.currentTarget; el.style.background = 'rgba(139,92,246,0.28)'; el.style.borderColor = 'rgba(139,92,246,0.7)'; }}}
                onMouseLeave={e => { if (!submitting) { const el = e.currentTarget; el.style.background = 'rgba(139,92,246,0.15)'; el.style.borderColor = 'rgba(139,92,246,0.4)'; }}}
              >
                {submitting ? (fr ? 'Envoi...' : 'Sending...') : `${t.submit} \u2192`}
              </button>
              {submitError && (
                <p style={{ fontSize: 12, color: 'rgba(239,68,68,0.82)', marginTop: 10 }}>{submitError}</p>
              )}
            </div>

          </div>

          {/* Signature */}
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', marginTop: 48, paddingTop: 24 }}>
            <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.3)', margin: 0 }}>
              {t.preferEmail}{' '}
              <a href={`mailto:${BUSINESS_EMAIL}`} style={{ color: 'rgba(167,139,250,0.8)', textDecoration: 'none' }}>
                {BUSINESS_EMAIL}
              </a>
            </p>
          </div>
          </>
          )}
        </motion.div>
      </div>

      <ChatWidget lang={lang} />

      <PageFooter />
      <style>{`
        @media (max-width: 540px) {
          .form-row { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </main>
  );
}
