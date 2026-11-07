'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState, useRef, useEffect } from 'react';
import type { Question } from '@/lib/questions';
import type { QuizAnswers } from '@/lib/types';

interface QuestionCardProps {
  question: Question;
  answers: QuizAnswers;
  onAnswer: (id: keyof QuizAnswers, value: string) => void;
  onOtherText: (value: string) => void;
  otherText: string;
  direction: 1 | -1;
}

const variants = {
  enter: (dir: number) => ({ x: dir * 48, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (dir: number) => ({ x: dir * -48, opacity: 0 }),
};

export default function QuestionCard({ question, answers, onAnswer, onOtherText, otherText, direction }: QuestionCardProps) {
  const selected = answers[question.id] as string | undefined;
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (selected === 'other' && inputRef.current) {
      inputRef.current.focus();
    }
  }, [selected]);

  return (
    <AnimatePresence mode="wait" custom={direction}>
      <motion.div
        key={question.id}
        custom={direction}
        variants={variants}
        initial="enter"
        animate="center"
        exit="exit"
        transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
        className="w-full"
      >
        {/* Question text */}
        <p
          className="mb-8 leading-snug"
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 'clamp(20px, 3vw, 28px)',
            fontWeight: 700,
            letterSpacing: '-0.03em',
            color: '#F5F0E8',
          }}
        >
          {question.text}
        </p>

        {/* Options */}
        <div className="flex flex-col gap-3">
          {question.options.map(opt => {
            const active = selected === opt.id;
            return (
              <div key={opt.id}>
                <motion.button
                  onClick={() => onAnswer(question.id, opt.id)}
                  whileHover={{ x: 4 }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ duration: 0.18 }}
                  className="w-full text-left flex items-center gap-4 rounded-xl px-5 py-4 transition-colors"
                  style={{
                    background: active
                      ? 'rgba(139,92,246,0.14)'
                      : 'rgba(245,240,252,0.05)',
                    border: `1.5px solid ${active ? 'rgba(139,92,246,0.65)' : 'rgba(245,240,252,0.11)'}`,
                    cursor: 'pointer',
                  }}
                >
                  {/* Radio indicator */}
                  <div
                    className="flex-shrink-0 rounded-full flex items-center justify-center"
                    style={{
                      width: 22,
                      height: 22,
                      border: `2px solid ${active ? '#8B5CF6' : 'rgba(245,240,252,0.25)'}`,
                      background: active ? '#8B5CF6' : 'transparent',
                      transition: 'all 0.2s',
                    }}
                  >
                    {active && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="rounded-full"
                        style={{ width: 9, height: 9, background: '#1A1610' }}
                      />
                    )}
                  </div>
                  <span
                    style={{
                      fontSize: 15,
                      fontWeight: 500,
                      color: active ? '#F5F0E8' : 'rgba(245,240,232,0.7)',
                      lineHeight: 1.4,
                    }}
                  >
                    {opt.label}
                  </span>
                </motion.button>

                {/* Other text input */}
                {opt.hasInput && active && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                    className="overflow-hidden"
                  >
                    <input
                      ref={inputRef}
                      type="text"
                      placeholder="Please describe your business…"
                      value={otherText}
                      onChange={e => onOtherText(e.target.value)}
                      className="w-full mt-2 px-4 py-3 rounded-lg text-sm outline-none"
                      style={{
                        background: 'rgba(245,240,232,0.08)',
                        border: '1.5px solid rgba(245,240,232,0.2)',
                        color: '#F5F0E8',
                        fontSize: 14,
                      }}
                    />
                  </motion.div>
                )}
              </div>
            );
          })}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
