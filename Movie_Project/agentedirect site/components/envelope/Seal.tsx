'use client';

import { motion, useReducedMotion } from 'framer-motion';

interface SealProps {
  opened: boolean;
}

const S = 62; // diameter
const R = S / 2;

export default function Seal({ opened }: SealProps) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <motion.div
      style={{ width: S, height: S, position: 'relative', flexShrink: 0 }}
      animate={
        shouldReduceMotion
          ? { opacity: opened ? 0 : 1 }
          : { scale: opened ? 0.45 : 1, opacity: opened ? 0 : 1 }
      }
      transition={
        opened
          ? { duration: 0.26, ease: [0.4, 0, 1, 1] }
          : { duration: 0.45, ease: [0.16, 1, 0.3, 1] }
      }
    >
      {/* Soft purple ambient halo */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          inset: -16,
          borderRadius: '50%',
          background:
            'radial-gradient(circle, rgba(139,92,246,0.28) 0%, rgba(109,40,217,0.1) 50%, transparent 75%)',
          pointerEvents: 'none',
        }}
      />

      <svg
        width={S}
        height={S}
        viewBox={`0 0 ${S} ${S}`}
        fill="none"
        style={{ display: 'block', overflow: 'visible' }}
        aria-hidden
      >
        <defs>
          {/* Silver-platinum main gradient — off-centre light at upper-left */}
          <radialGradient id="s-main" cx="32%" cy="26%" r="70%" gradientUnits="objectBoundingBox">
            <stop offset="0%"   stopColor="#EEEEF8" />
            <stop offset="20%"  stopColor="#C8C8E0" />
            <stop offset="52%"  stopColor="#8888B0" />
            <stop offset="78%"  stopColor="#505078" />
            <stop offset="100%" stopColor="#2C2C50" />
          </radialGradient>

          {/* Specular — tight bright bloom top-left */}
          <radialGradient id="s-spec" cx="28%" cy="22%" r="34%" gradientUnits="objectBoundingBox">
            <stop offset="0%"   stopColor="rgba(255,255,255,0.82)" />
            <stop offset="50%"  stopColor="rgba(255,255,255,0.12)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)" />
          </radialGradient>

          {/* Rim vignette — deepens the edge to sell emboss depth */}
          <radialGradient id="s-rim" cx="50%" cy="50%" r="50%" gradientUnits="objectBoundingBox">
            <stop offset="60%"  stopColor="rgba(0,0,0,0)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.32)" />
          </radialGradient>

          {/* Inner ring gradient */}
          <radialGradient id="s-ring" cx="32%" cy="26%" r="70%" gradientUnits="objectBoundingBox">
            <stop offset="0%"   stopColor="rgba(220,220,255,0.5)" />
            <stop offset="100%" stopColor="rgba(80,80,150,0.2)" />
          </radialGradient>

          {/* Text emboss shadow */}
          <filter id="s-emboss" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0.5" dy="1" stdDeviation="0.8" floodColor="rgba(0,0,0,0.55)" />
          </filter>

          {/* Outer ambient glow (purple-tinted) */}
          <filter id="s-glow" x="-45%" y="-45%" width="190%" height="190%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
            <feColorMatrix
              in="blur" type="matrix"
              values="0.3 0 0.8 0 0  0.1 0 0.5 0 0  0.5 0 1 0 0  0 0 0 0.55 0"
              result="colored"
            />
            <feMerge>
              <feMergeNode in="colored" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Outer shadow/glow ring */}
        <circle cx={R} cy={R} r={R - 1} fill="rgba(80,60,160,0.35)" filter="url(#s-glow)" />

        {/* Main silver disc */}
        <circle cx={R} cy={R} r={R - 3} fill="url(#s-main)" />

        {/* Embossed inner concentric ring */}
        <circle
          cx={R} cy={R} r={R - 9}
          fill="none"
          stroke="url(#s-ring)"
          strokeWidth="0.75"
        />

        {/* Rim shadow vignette */}
        <circle cx={R} cy={R} r={R - 3} fill="url(#s-rim)" />

        {/* Specular bloom */}
        <circle cx={R} cy={R} r={R - 3} fill="url(#s-spec)" />

        {/* AD monogram — dark purple on silver for contrast */}
        <text
          x={R}
          y={R + 5}
          textAnchor="middle"
          fontSize="13.5"
          fontWeight="700"
          fontFamily="'Space Grotesk', Georgia, serif"
          letterSpacing="2"
          fill="rgba(18,6,54,0.88)"
          filter="url(#s-emboss)"
          style={{ userSelect: 'none' }}
        >
          AD
        </text>
      </svg>

      {/* Rotating silver glare sweep */}
      {!shouldReduceMotion && !opened && (
        <motion.div
          aria-hidden
          style={{
            position: 'absolute',
            inset: 3,
            borderRadius: '50%',
            background:
              'conic-gradient(from 0deg, transparent 0%, rgba(240,240,255,0.22) 13%, rgba(240,240,255,0.06) 25%, transparent 38%)',
            pointerEvents: 'none',
          }}
          animate={{ rotate: [0, 360] }}
          transition={{ duration: 5.5, repeat: Infinity, ease: 'linear', repeatDelay: 3 }}
        />
      )}
    </motion.div>
  );
}
