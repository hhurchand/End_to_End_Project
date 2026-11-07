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
    optBasic: 'Website',
    optChatbot: 'Website + AI Chatbot',
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
    planBasic: 'Website (No Chatbot)',
    planChatbot: 'Website + AI Chatbot',
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
    optBasic: 'Site web',
    optChatbot: 'Site web + Chatbot IA',
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
    planBasic: 'Site web (sans chatbot)',
    planChatbot: 'Site web + Chatbot IA',
  },
};

interface FormState {
  name: string;
  businessName: string;
  email: string;
  currentUrl: string;
  businessDescription: string;
  plan: 'basic' | 'chatbot';
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

function buildMailto(f: FormState, t: typeof T.en): string {
  const planLabel = f.plan === 'chatbot' ? t.planChatbot : t.planBasic;
  const lines = [
    t.emailGreeting(f.name),
    ``,
    `Name: ${f.name}`,
    `Business: ${f.businessName}`,
    `Email: ${f.email}`,
    `${t.emailSiteLabel}: ${f.currentUrl || t.emailNone}`,
    ``,
    t.emailAbout,
    f.businessDescription,
    ``,
    `${t.emailInterested} ${planLabel}`,
    ``,
    t.emailClosing,
    ``,
    f.name,
  ];
  return `mailto:${BUSINESS_EMAIL}?subject=${encodeURIComponent(t.emailSubject(f.businessName))}&body=${encodeURIComponent(lines.join('\n'))}`;
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
    currentUrl: '', businessDescription: '', plan: 'basic',
  });
  const [errors, setErrors] = useState<FormErrors>({});

  const setField = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
    if (key in errors) setErrors(prev => { const n = { ...prev }; delete n[key as keyof FormErrors]; return n; });
  }, [errors]);

  const handleSubmit = useCallback(() => {
    const errs = validate(form, t);
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      document.getElementById(`field-${Object.keys(errs)[0]}`)?.focus();
      return;
    }
    window.location.href = buildMailto(form, t);
  }, [form, t]);

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
                  { id: 'basic' as const, label: t.optBasic },
                  { id: 'chatbot' as const, label: t.optChatbot },
                ].map(opt => {
                  const chosen = form.plan === opt.id;
                  return (
                    <div
                      key={opt.id}
                      role="radio" aria-checked={chosen} tabIndex={0}
                      onClick={() => setField('plan', opt.id)}
                      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setField('plan', opt.id); } }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 12,
                        padding: '11px 16px', borderRadius: 10, cursor: 'pointer', outline: 'none',
                        border: `1px solid ${chosen ? 'rgba(139,92,246,0.5)' : 'rgba(255,255,255,0.07)'}`,
                        background: chosen ? 'rgba(109,40,217,0.08)' : 'rgba(255,255,255,0.02)',
                        transition: 'border-color 0.2s, background 0.2s',
                      }}
                    >
                      <div style={{
                        width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
                        border: `1.5px solid ${chosen ? '#8B5CF6' : 'rgba(255,255,255,0.18)'}`,
                        background: chosen ? '#7C3AED' : 'transparent',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        transition: 'all 0.2s',
                      }}>
                        {chosen && <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#fff' }} />}
                      </div>
                      <span style={{ fontSize: 13, color: chosen ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.45)', fontWeight: chosen ? 600 : 400, transition: 'color 0.2s' }}>
                        {opt.label}
                      </span>
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
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 8,
                  padding: '13px 28px', borderRadius: 50,
                  background: 'rgba(139,92,246,0.15)',
                  color: 'rgba(220,210,245,0.95)', fontSize: 14, fontWeight: 700,
                  border: '1px solid rgba(139,92,246,0.4)', cursor: 'pointer', letterSpacing: '0.01em',
                  transition: 'background 0.2s, border-color 0.2s',
                }}
                onMouseEnter={e => { const el = e.currentTarget; el.style.background = 'rgba(139,92,246,0.28)'; el.style.borderColor = 'rgba(139,92,246,0.7)'; }}
                onMouseLeave={e => { const el = e.currentTarget; el.style.background = 'rgba(139,92,246,0.15)'; el.style.borderColor = 'rgba(139,92,246,0.4)'; }}
              >
                {t.submit} →
              </button>
              <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.22)', marginTop: 10, lineHeight: 1.5 }}>
                {t.submitHint}
              </p>
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
