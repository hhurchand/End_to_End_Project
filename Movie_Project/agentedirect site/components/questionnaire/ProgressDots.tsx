'use client';

import { motion } from 'framer-motion';
import { TOTAL_STEPS } from '@/lib/questions';

interface ProgressDotsProps {
  current: number; // 1-based
}

export default function ProgressDots({ current }: ProgressDotsProps) {
  return (
    <div className="flex items-center gap-2" role="progressbar" aria-valuenow={current} aria-valuemax={TOTAL_STEPS}>
      {Array.from({ length: TOTAL_STEPS }).map((_, i) => {
        const done    = i + 1 < current;
        const active  = i + 1 === current;
        return (
          <motion.div
            key={i}
            aria-label={done ? 'Completed' : active ? 'Current step' : 'Upcoming step'}
            animate={{
              width:   active ? 24 : 8,
              opacity: done ? 0.5 : active ? 1 : 0.25,
              backgroundColor: done || active ? '#8B5CF6' : 'rgba(245,240,252,0.25)',
            }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            style={{ height: 8, borderRadius: 4 }}
          />
        );
      })}
    </div>
  );
}
