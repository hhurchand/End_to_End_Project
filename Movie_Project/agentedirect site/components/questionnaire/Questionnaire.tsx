'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState, useCallback, useRef } from 'react';
import { QUESTIONS, TOTAL_STEPS } from '@/lib/questions';
import type { QuizAnswers } from '@/lib/types';
import ProgressDots from './ProgressDots';
import QuestionCard from './QuestionCard';
import ContactForm from './ContactForm';

const HOME_URL = '/home';

type FormErrors = Partial<Record<keyof QuizAnswers, string>>;

function validateEmail(v: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
}

interface QuestionnaireProps {
  onExit: () => void;
}

export default function Questionnaire({ onExit }: QuestionnaireProps) {
  const [step, setStep] = useState(1);            // 1-based, TOTAL_STEPS = 5
  const [direction, setDirection] = useState<1 | -1>(1);
  const [answers, setAnswers] = useState<QuizAnswers>({});
  const [otherText, setOtherText] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [status, setStatus] = useState<'idle' | 'submitting' | 'done' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const submittedRef = useRef(false);

  const isContactStep = step === TOTAL_STEPS;
  const question = QUESTIONS[step - 1];

  // ── Advance after selecting a multiple-choice answer ──
  const handleAnswer = useCallback((id: keyof QuizAnswers, value: string) => {
    setAnswers(prev => ({ ...prev, [id]: value }));
    // Auto-advance after short delay for MC questions
    if (!isContactStep) {
      setTimeout(() => {
        setDirection(1);
        setStep(s => Math.min(s + 1, TOTAL_STEPS));
      }, 280);
    }
  }, [isContactStep]);

  // ── Contact form field changes ──
  const handleContact = useCallback((field: keyof QuizAnswers, value: string | boolean) => {
    setAnswers(prev => ({ ...prev, [field]: value }));
    setErrors(prev => ({ ...prev, [field]: undefined }));
  }, []);

  // ── Back ──
  function goBack() {
    if (step <= 1) { onExit(); return; }
    setDirection(-1);
    setStep(s => s - 1);
  }

  // ── Validate contact step ──
  function validateContact(): boolean {
    const errs: FormErrors = {};
    if (!answers.name?.trim())       errs.name         = 'Please enter your name.';
    if (!answers.businessName?.trim()) errs.businessName = 'Please enter your business name.';
    if (!answers.email?.trim())      errs.email        = 'Please enter your email address.';
    else if (!validateEmail(answers.email)) errs.email  = 'Please enter a valid email address.';
    if (!answers.consent)            errs.consent      = 'Please confirm your consent to be contacted.';
    setErrors(errs);
    return Object.keys(errs).length === 0;
  }

  // ── Submit ──
  async function handleSubmit() {
    if (!validateContact()) return;
    if (submittedRef.current) return;
    submittedRef.current = true;
    setStatus('submitting');
    setErrorMessage('');

    const payload: QuizAnswers = {
      ...answers,
      businessTypeOther: answers.businessType === 'other' ? otherText : undefined,
    };

    try {
      const res = await fetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? 'Submission failed. Please try again.');
      }
      setStatus('done');
    } catch (err: unknown) {
      submittedRef.current = false;
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : 'Something went wrong. Please try again.');
    }
  }

  // ── Next button (contact step) ──
  function handleNext() {
    if (isContactStep) {
      handleSubmit();
    } else {
      // MC questions auto-advance, but allow manual advance if already answered
      if (answers[question.id]) {
        setDirection(1);
        setStep(s => s + 1);
      }
    }
  }

  // ── Success screen ──
  if (status === 'done') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col items-center text-center gap-6 max-w-md mx-auto"
      >
        {/* Animated checkmark */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, duration: 0.5, type: 'spring', stiffness: 200 }}
          className="flex items-center justify-center rounded-full"
          style={{ width: 72, height: 72, background: 'rgba(139,92,246,0.12)', border: '1.5px solid rgba(139,92,246,0.4)' }}
        >
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <motion.path
              d="M8 16L13.5 22L24 10"
              stroke="#8B5CF6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ delay: 0.5, duration: 0.5, ease: 'easeOut' }}
            />
          </svg>
        </motion.div>

        <div>
          <h2 style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 'clamp(26px,4vw,36px)', fontWeight: 800, letterSpacing: '-0.04em', color: '#F5F0E8', marginBottom: 12 }}>
            Invitation accepted.
          </h2>
          <p style={{ fontSize: 16, color: 'rgba(245,240,232,0.55)', lineHeight: 1.75, maxWidth: 380 }}>
            Thanks for telling us about your business. We&rsquo;ll review your answers and be in touch soon.
          </p>
        </div>

        <button
          onClick={onExit}
          className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full text-sm font-semibold"
          style={{
            background: 'rgba(139,92,246,0.12)',
            border: '1px solid rgba(139,92,246,0.4)',
            color: '#A78BFA',
            marginTop: 8,
            cursor: 'pointer',
          }}
        >
          Continue to the website →
        </button>
      </motion.div>
    );
  }

  return (
    <div className="w-full max-w-lg mx-auto flex flex-col">
      {/* Header row */}
      <div className="flex items-center justify-between mb-8">
        <ProgressDots current={step} />
        <button
          onClick={onExit}
          style={{ fontSize: 12, color: 'rgba(245,240,232,0.3)', letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}
        >
          Skip →
        </button>
      </div>

      {/* Step label */}
      <p style={{ fontSize: 11.5, color: 'rgba(245,240,232,0.35)', letterSpacing: '0.12em', textTransform: 'uppercase', fontWeight: 700, marginBottom: 20 }}>
        Question {step} of {TOTAL_STEPS}
      </p>

      {/* Question or contact form */}
      <div className="flex-1 min-h-0">
        {isContactStep
          ? <ContactForm answers={answers} onChange={handleContact} errors={errors} direction={direction} />
          : <QuestionCard question={question} answers={answers} onAnswer={handleAnswer} onOtherText={setOtherText} otherText={otherText} direction={direction} />
        }
      </div>

      {/* Error banner */}
      <AnimatePresence>
        {status === 'error' && (
          <motion.p
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 px-4 py-3 rounded-lg text-sm"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', color: '#EF8C8C' }}
          >
            {errorMessage}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Footer controls */}
      <div className="flex items-center justify-between mt-8 pt-6" style={{ borderTop: '1px solid rgba(245,240,232,0.08)' }}>
        <button
          onClick={goBack}
          className="flex items-center gap-2 text-sm font-semibold"
          style={{ color: 'rgba(245,240,232,0.35)', letterSpacing: '0.02em' }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 12L6 8L10 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back
        </button>

        {/* Show Next only on contact step (MC auto-advances) */}
        {isContactStep && (
          <motion.button
            onClick={handleNext}
            disabled={status === 'submitting'}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="px-7 py-3 rounded-full text-sm font-bold"
            style={{
              background: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
              color: '#FFFFFF',
              opacity: status === 'submitting' ? 0.7 : 1,
              cursor: status === 'submitting' ? 'wait' : 'pointer',
            }}
          >
            {status === 'submitting' ? 'Sending…' : 'Send my inquiry →'}
          </motion.button>
        )}
      </div>
    </div>
  );
}
