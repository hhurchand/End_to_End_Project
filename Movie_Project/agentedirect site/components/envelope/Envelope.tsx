'use client';

import { motion, useReducedMotion } from 'framer-motion';
import { useState } from 'react';
import Seal from './Seal';

interface EnvelopeProps {
  onOpen: () => void;
}

const W         = 572;
const H         = 377;
const CX        = W / 2;   // 286
const CY        = H / 2;   // 188.5

type EnvPhase = 'idle' | 'opening' | 'done';

export default function Envelope({ onOpen }: EnvelopeProps) {
  const shouldReduceMotion = useReducedMotion();
  const [phase, setPhase]  = useState<EnvPhase>('idle');

  function handleOpen() {
    if (phase !== 'idle') return;
    setPhase('opening');
    if (shouldReduceMotion) { onOpen(); return; }
    setTimeout(() => { setPhase('done'); onOpen(); }, 680);
  }

  const flapOpen = phase !== 'idle';

  return (
    <div
      style={{
        perspective:       '1100px',
        perspectiveOrigin: '50% 85%',
        width:             W,
        maxWidth:          '90vw',
      }}
    >
      <motion.div
        style={{
          width:        '100%',
          aspectRatio:  `${W} / ${H}`,
          position:     'relative',
          cursor:       phase === 'idle' ? 'pointer' : 'default',
          background:   '#8F2BFF',
          borderRadius: 8,
          boxShadow:    '0 4px 18px rgba(0,0,0,0.4), 0 12px 48px rgba(110,30,235,0.55)',
        }}
        animate={
          shouldReduceMotion || phase !== 'idle'
            ? {}
            : { y: [0, -8, 0], rotateX: [3, 5, 3], rotateZ: [-0.25, 0.25, -0.25] }
        }
        transition={
          shouldReduceMotion || phase !== 'idle'
            ? {}
            : { duration: 5.5, repeat: Infinity, ease: 'easeInOut' }
        }
        whileHover={phase === 'idle' ? { y: -8, scale: 1.018, rotateX: 8 } : {}}
        role="button"
        tabIndex={0}
        aria-label="Open the invitation"
        onClick={handleOpen}
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && handleOpen()}
        className="select-none outline-none"
      >

        {/* Purple drop shadow */}
        <div
          aria-hidden
          style={{
            position:     'absolute',
            bottom:       -47,
            left:         '7%',
            right:        '7%',
            height:       65,
            borderRadius: '50%',
            background:
              'radial-gradient(ellipse, rgba(90,40,200,0.65) 0%, rgba(30,8,80,0.5) 40%, transparent 70%)',
            filter:        'blur(20px)',
            transform:     flapOpen ? 'scaleX(0.62) translateY(-10px)' : 'scaleX(1)',
            transition:    'transform 0.75s ease',
            pointerEvents: 'none',
          }}
        />

        {/* ── SVG envelope body ── */}
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          aria-hidden
          style={{
            position:     'absolute',
            inset:        0,
            width:        '100%',
            height:       '100%',
            borderRadius: 7,
            overflow:     'hidden',
          }}
        >
          <defs>
            {/* Body — vivid brand purple */}
            <linearGradient id="g-body" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#8F2BFF" />
              <stop offset="100%" stopColor="#7A1DF0" />
            </linearGradient>

            {/* Interior pocket — dark, revealed once flap folds away */}
            <linearGradient id="g-inside" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#06021A" />
              <stop offset="100%" stopColor="#0C0530" />
            </linearGradient>

            {/* Left lower fold — slightly lighter (lit side) */}
            <linearGradient id="g-left" x1="0" y1="0.5" x2="1" y2="0.5">
              <stop offset="0%"   stopColor="#A24BFF" />
              <stop offset="100%" stopColor="#7A1DF0" />
            </linearGradient>

            {/* Right lower fold — slightly darker (shadow side) */}
            <linearGradient id="g-right" x1="1" y1="0.5" x2="0" y2="0.5">
              <stop offset="0%"   stopColor="#8F2BFF" />
              <stop offset="100%" stopColor="#5B21B6" />
            </linearGradient>

            {/* Bottom fold */}
            <linearGradient id="g-bottom" x1="0.5" y1="1" x2="0.5" y2="0">
              <stop offset="0%"   stopColor="#A24BFF" />
              <stop offset="100%" stopColor="#7A1DF0" />
            </linearGradient>

            {/* Flap — brightest purple at top */}
            <linearGradient id="g-flap" x1="0.3" y1="0" x2="0.7" y2="1">
              <stop offset="0%"   stopColor="#C28DFF" />
              <stop offset="100%" stopColor="#8F2BFF" />
            </linearGradient>

            {/* Noise */}
            <filter id="f-noise" x="0" y="0" width="100%" height="100%">
              <feTurbulence
                type="fractalNoise" baseFrequency="0.72"
                numOctaves="4" stitchTiles="stitch" result="noise"
              />
              <feColorMatrix in="noise" type="saturate" values="0" result="grey" />
              <feComponentTransfer in="grey" result="dimmed">
                <feFuncA type="linear" slope="0.04" />
              </feComponentTransfer>
              <feBlend in="SourceGraphic" in2="dimmed" mode="screen" />
            </filter>

          </defs>

          {/* Top V crease lines only — left and right from top corners to centre */}
          <line x1={0} y1={0} x2={CX} y2={CY} stroke="rgba(210,185,255,0.35)" strokeWidth="1"    />
          <line x1={W} y1={0} x2={CX} y2={CY} stroke="rgba(30,10,90,0.55)"    strokeWidth="0.75" />

          {/* Top catch-light */}
          <line x1="6" y1="0.75" x2={W - 6} y2="0.75"
            stroke="rgba(215,190,255,0.42)" strokeWidth="0.75" />

          {/* Outer rim */}
          <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="6.5"
            fill="none" stroke="rgba(170,140,255,0.28)" strokeWidth="1" />

          {/* Noise overlay — fill transparent so CSS background shows through */}
          <rect x="0" y="0" width={W} height={H} rx="7" fill="none" filter="url(#f-noise)" />
        </svg>

        {/* ── FLAP — pure 2D scaleY, NO rotateX, NO 3D ──
            clip-path on the SAME element as scaleY is fine because scaleY is
            a 2D transform — no browser 3D / clip-path split rendering bug.
            transformOrigin 'top center' makes the triangle fold upward.
        ── */}
        <motion.div
          aria-hidden
          style={{
            position:        'absolute',
            left:            0,
            right:           0,
            top:             0,
            height:          `${(CY / H) * 100}%`,
            clipPath:        'polygon(0% 0%, 100% 0%, 50% 100%)',
            transformOrigin: 'top center',
            background:      'linear-gradient(160deg, #C28DFF 0%, #8F2BFF 100%)',
            zIndex:          20,
            pointerEvents:   'none',
          }}
          animate={{
            scaleY:  flapOpen ? 0 : 1,
            opacity: flapOpen ? 0 : 1,
          }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
        />

        {/* Flap sheen */}
        <motion.div
          aria-hidden
          style={{
            position:        'absolute',
            left:            0,
            right:           0,
            top:             0,
            height:          `${(CY / H) * 100}%`,
            clipPath:        'polygon(0% 0%, 100% 0%, 50% 100%)',
            transformOrigin: 'top center',
            background:      'linear-gradient(148deg, rgba(215,190,255,0.15) 0%, transparent 55%)',
            zIndex:          21,
            pointerEvents:   'none',
          }}
          animate={{
            scaleY:  flapOpen ? 0 : 1,
            opacity: flapOpen ? 0 : 1,
          }}
          transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
        />

        {/* Seal — sits on the crease between flap and body */}
        <div
          style={{
            position:  'absolute',
            top:       `${(CY / H) * 100}%`,
            left:      '50%',
            transform: 'translate(-50%, -50%)',
            zIndex:    30,
          }}
        >
          <Seal opened={flapOpen} />
        </div>

        {/* Click hint */}
        {phase === 'idle' && (
          <motion.div
            style={{
              position:      'absolute',
              bottom:        '13px',
              left:          '50%',
              x:             '-50%',
              pointerEvents: 'none',
              whiteSpace:    'nowrap',
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0.65, 0] }}
            transition={{ duration: 2.8, repeat: Infinity, delay: 2.2 }}
          >
            <span
              style={{
                fontSize:      9,
                letterSpacing: '0.16em',
                color:         'rgba(185,160,255,0.75)',
                fontWeight:    600,
                textTransform: 'uppercase',
              }}
            >
              Click to open
            </span>
          </motion.div>
        )}

      </motion.div>
    </div>
  );
}
