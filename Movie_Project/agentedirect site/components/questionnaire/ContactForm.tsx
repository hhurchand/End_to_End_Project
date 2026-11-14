'use client';

import { motion, AnimatePresence } from 'framer-motion';
import type { QuizAnswers } from '@/lib/types';

interface ContactFormProps {
  answers: QuizAnswers;
  onChange: (field: keyof QuizAnswers, value: string | boolean) => void;
  errors: Partial<Record<keyof QuizAnswers, string>>;
  direction: 1 | -1;
}

const variants = {
  enter: (dir: number) => ({ x: dir * 48, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (dir: number) => ({ x: dir * -48, opacity: 0 }),
};

function Field({
  label,
  required,
  error,
  children,
}: {
  label: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        className="block mb-1.5"
        style={{ fontSize: 12.5, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'rgba(245,240,232,0.66)' }}
      >
        {label}{required && <span style={{ color: '#C28DFF' }}> *</span>}
      </label>
      {children}
      {error && (
        <p style={{ fontSize: 13, color: '#EF8C8C', marginTop: 5 }}>{error}</p>
      )}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: 'rgba(245,240,232,0.07)',
  border: '1.5px solid rgba(245,240,232,0.14)',
  borderRadius: 10,
  padding: '11px 14px',
  fontSize: 15.5,
  color: '#F5F0E8',
  fontFamily: 'inherit',
  outline: 'none',
  transition: 'border-color 0.2s',
};

const inputFocusStyle: React.CSSProperties = {
  borderColor: 'rgba(162,75,255,0.55)',
};

export default function ContactForm({ answers, onChange, errors, direction }: ContactFormProps) {
  return (
    <AnimatePresence mode="wait" custom={direction}>
      <motion.div
        key="contact"
        custom={direction}
        variants={variants}
        initial="enter"
        animate="center"
        exit="exit"
        transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
        className="w-full"
      >
        <p
          className="mb-7"
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 'clamp(20px, 3vw, 26px)',
            fontWeight: 700,
            letterSpacing: '-0.03em',
            color: '#F5F0E8',
          }}
        >
          Tell us where to reach you.
        </p>

        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Your name" required error={errors.name}>
              <input
                type="text"
                placeholder="Greg Spunt"
                value={answers.name ?? ''}
                onChange={e => onChange('name', e.target.value)}
                style={inputStyle}
                onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                onBlur={e => Object.assign(e.target.style, { borderColor: 'rgba(245,240,232,0.14)' })}
              />
            </Field>
            <Field label="Business name" required error={errors.businessName}>
              <input
                type="text"
                placeholder="Metro Renovation Inc."
                value={answers.businessName ?? ''}
                onChange={e => onChange('businessName', e.target.value)}
                style={inputStyle}
                onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                onBlur={e => Object.assign(e.target.style, { borderColor: 'rgba(245,240,232,0.14)' })}
              />
            </Field>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Email address" required error={errors.email}>
              <input
                type="email"
                placeholder="you@yourbusiness.com"
                value={answers.email ?? ''}
                onChange={e => onChange('email', e.target.value)}
                style={inputStyle}
                onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                onBlur={e => Object.assign(e.target.style, { borderColor: 'rgba(245,240,232,0.14)' })}
              />
            </Field>
            <Field label="Phone (optional)" error={errors.phone}>
              <input
                type="tel"
                placeholder="(514) 555-0100"
                value={answers.phone ?? ''}
                onChange={e => onChange('phone', e.target.value)}
                style={inputStyle}
                onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
                onBlur={e => Object.assign(e.target.style, { borderColor: 'rgba(245,240,232,0.14)' })}
              />
            </Field>
          </div>

          <Field label="Current website (optional)">
            <input
              type="url"
              placeholder="https://yourbusiness.ca"
              value={answers.currentUrl ?? ''}
              onChange={e => onChange('currentUrl', e.target.value)}
              style={inputStyle}
              onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
              onBlur={e => Object.assign(e.target.style, { borderColor: 'rgba(245,240,232,0.14)' })}
            />
          </Field>

          <Field label="Anything else to share? (optional)">
            <textarea
              rows={3}
              placeholder="What's your main challenge with your current online presence?"
              value={answers.details ?? ''}
              onChange={e => onChange('details', e.target.value)}
              style={{ ...inputStyle, resize: 'vertical', minHeight: 80 }}
              onFocus={e => Object.assign(e.target.style, inputFocusStyle)}
              onBlur={e => Object.assign(e.target.style, { borderColor: 'rgba(245,240,232,0.14)' })}
            />
          </Field>

          {/* Consent */}
          <label
            className="flex items-start gap-3 cursor-pointer"
            style={{ marginTop: 4 }}
          >
            <div className="flex-shrink-0 mt-0.5">
              <input
                type="checkbox"
                checked={answers.consent ?? false}
                onChange={e => onChange('consent', e.target.checked)}
                className="sr-only"
              />
              <div
                className="flex items-center justify-center rounded"
                style={{
                  width: 20,
                  height: 20,
                  background: answers.consent ? '#A24BFF' : 'rgba(245,240,252,0.07)',
                  border: `1.5px solid ${answers.consent ? '#A24BFF' : 'rgba(245,240,252,0.2)'}`,
                  transition: 'all 0.2s',
                }}
              >
                {answers.consent && (
                  <svg width="11" height="9" viewBox="0 0 11 9" fill="none">
                    <path d="M1 4L4 7.5L10 1" stroke="#1A1610" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
              </div>
            </div>
            <span style={{ fontSize: 14, color: 'rgba(245,240,232,0.69)', lineHeight: 1.55 }}>
              I agree to be contacted about my inquiry. I understand this is not a marketing list.
            </span>
          </label>
          {errors.consent && (
            <p style={{ fontSize: 13, color: '#EF8C8C', marginTop: -8 }}>{errors.consent}</p>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
