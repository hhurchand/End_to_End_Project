'use client';

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

const ease = [0.16, 1, 0.3, 1] as const;

const SERVICES = [
  {
    id: 'design',
    title: 'Custom Website Design',
    body: 'Purpose-built websites that reflect your business, communicate your value, and help you stand out from competitors.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
        <rect x="1.5" y="2" width="17" height="11.5" rx="1.5" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M1.5 5.5h17" stroke="currentColor" strokeWidth="1.5"/>
        <circle cx="3.75" cy="3.75" r="0.7" fill="currentColor" opacity="0.55"/>
        <circle cx="6"    cy="3.75" r="0.7" fill="currentColor" opacity="0.55"/>
        <circle cx="8.25" cy="3.75" r="0.7" fill="currentColor" opacity="0.55"/>
        <path d="M7 18h6M10 13.5V18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M5.5 8.5h4M5.5 11h2.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" opacity="0.5"/>
        <path d="M12 11l1.5-1.5 1.5 1.5-1.5 1.5L12 11z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" opacity="0.7"/>
      </svg>
    ),
  },
  {
    id: 'leads',
    title: 'Lead Capture & Booking',
    body: 'Clear calls to action and straightforward inquiry flows that make it easy for potential customers to request a quote or book an estimate.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
        <rect x="2.5" y="1.5" width="11" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M5.5 5.5h5M5.5 8.5h5M5.5 11.5h3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" opacity="0.5"/>
        <circle cx="15.5" cy="14.5" r="3.5" fill="#6D28D9"/>
        <path d="M13.8 14.5l1.2 1.2 2.2-2.2" stroke="white" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    id: 'mobile',
    title: 'Mobile-First Experience',
    body: 'Fast, polished, responsive layouts designed to make a strong impression across phones, tablets, and desktops.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
        <rect x="5.5" y="1" width="9" height="17" rx="2" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M8.5 4h3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" opacity="0.35"/>
        <circle cx="10" cy="15.5" r="1" fill="currentColor" opacity="0.45"/>
        <path d="M3 8.5c1.8-2 4.2-3.2 7-3.2s5.2 1.2 7 3.2"
          stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" opacity="0.28"/>
        <path d="M5.5 11c1.2-1.3 2.7-2 4.5-2s3.3.7 4.5 2"
          stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" opacity="0.48"/>
      </svg>
    ),
  },
] as const;

const SERVICE_TEXT_FR = [
  { id: 'design', title: 'Conception de site sur mesure', body: 'Des sites conçus sur mesure qui reflètent votre entreprise, communiquent votre valeur et vous démarquent de la concurrence.' },
  { id: 'leads', title: 'Capture de prospects et réservation', body: 'Des appels à l\'action clairs et des formulaires de demande simples pour que vos clients potentiels puissent facilement demander un devis.' },
  { id: 'mobile', title: 'Expérience mobile d\'abord', body: 'Des mises en page rapides, soignées et responsives conçues pour faire forte impression sur téléphone, tablette et ordinateur.' },
];

export default function WhatWeDo({ lang = 'en' }: { lang?: 'en' | 'fr' } = {}) {
  const fr = lang === 'fr';
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <>
      <style>{`
        @media (min-width: 768px) {
          .wwd-grid { grid-template-columns: repeat(3, 1fr) !important; }
        }
        .wwd-card:hover {
          border-color: rgba(139,92,246,0.28) !important;
          background: rgba(139,92,246,0.05) !important;
          transform: translateY(-5px) !important;
          box-shadow: 0 20px 56px rgba(80,20,200,0.16) !important;
        }
        .wwd-card:hover .wwd-icon {
          background: rgba(139,92,246,0.18) !important;
          border-color: rgba(139,92,246,0.38) !important;
        }
      `}</style>

      <section
        ref={ref}
        aria-label="What we do"
        style={{
          position: 'relative',
          background: '#08041A',
          paddingTop: 'clamp(80px, 10vw, 120px)',
          paddingBottom: 'clamp(80px, 10vw, 120px)',
          overflow: 'hidden',
        }}
      >
        {/* Atmosphere */}
        <div aria-hidden style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse 60% 50% at 50% 60%, rgba(109,40,217,0.07) 0%, transparent 70%)',
        }} />

        {/* Top hairline */}
        <div aria-hidden style={{
          position: 'absolute', top: 0, left: '6%', right: '6%', height: 1,
          background: 'linear-gradient(90deg, transparent 0%, rgba(139,92,246,0.22) 25%, rgba(139,92,246,0.22) 75%, transparent 100%)',
        }} />

        <div className="max-w-7xl mx-auto px-6 md:px-14">

          {/* ── Header ── */}
          <motion.div
            initial={{ opacity: 0, y: 22 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.70, ease }}
            style={{ marginBottom: 'clamp(48px, 7vw, 72px)', maxWidth: '52ch' }}
          >
            <p style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 10, fontWeight: 700, letterSpacing: '0.24em',
              textTransform: 'uppercase', color: '#8B5CF6', margin: '0 0 20px',
            }}>
              {fr ? 'Ce qu\'on fait' : 'What We Do'}
            </p>

            <h2 style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 'clamp(28px, 4vw, 48px)',
              fontWeight: 800, color: '#F5F0FC',
              letterSpacing: '-0.04em', lineHeight: 1.07,
              margin: '0 0 18px',
            }}>
              {fr ? 'Plus qu\'un' : 'More than a'}{' '}
              <span style={{ color: 'rgba(220,210,245,0.28)' }}>{fr ? 'beau site web.' : 'good-looking website.'}</span>
            </h2>

            <p style={{
              fontSize: 'clamp(14px, 1.3vw, 16px)',
              color: 'rgba(220,210,245,0.50)',
              lineHeight: 1.78, margin: 0,
            }}>
              {fr
                ? 'Votre site devrait faire plus que paraître professionnel. Il doit aider vos clients à comprendre ce que vous offrez, renforcer leur confiance et les amener à passer à l\'action.'
                : 'Your website should do more than look professional. It should help customers understand what you offer, build confidence in your business, and take the next step.'
              }
            </p>
          </motion.div>

          {/* ── Cards ── */}
          <div
            className="wwd-grid"
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr',
              gap: 'clamp(14px, 1.6vw, 20px)',
            }}
          >
            {SERVICES.map((s, i) => {
              const text = fr ? SERVICE_TEXT_FR[i] : s;
              return (
              <motion.div
                key={s.id}
                initial={{ opacity: 0, y: 28 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.65, ease, delay: 0.12 + i * 0.10 }}
                className="wwd-card"
                style={{
                  position: 'relative',
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.07)',
                  borderRadius: 16,
                  padding: 'clamp(28px, 3vw, 36px)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 20,
                  transition: 'border-color 0.28s ease, background 0.28s ease, transform 0.28s ease, box-shadow 0.28s ease',
                  cursor: 'default',
                }}
              >
                {/* Top accent hairline */}
                <div aria-hidden style={{
                  position: 'absolute',
                  top: 0, left: 24, right: 24, height: 1,
                  background: 'linear-gradient(90deg, transparent, rgba(139,92,246,0.30), transparent)',
                  borderRadius: '0 0 2px 2px',
                }} />

                {/* Icon */}
                <div
                  className="wwd-icon"
                  style={{
                    width: 44, height: 44, borderRadius: 10, flexShrink: 0,
                    background: 'rgba(139,92,246,0.09)',
                    border: '1px solid rgba(139,92,246,0.18)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#A78BFA',
                    transition: 'background 0.28s ease, border-color 0.28s ease',
                  }}
                >
                  {s.icon}
                </div>

                {/* Text */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <h3 style={{
                    fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                    fontSize: 'clamp(16px, 1.35vw, 18px)',
                    fontWeight: 700, color: '#F5F0FC',
                    letterSpacing: '-0.02em', lineHeight: 1.2, margin: 0,
                  }}>
                    {text.title}
                  </h3>
                  <p style={{
                    fontSize: 'clamp(13.5px, 1.05vw, 15px)',
                    color: 'rgba(220,210,245,0.50)',
                    lineHeight: 1.75, margin: 0,
                  }}>
                    {text.body}
                  </p>
                </div>
              </motion.div>
              );
            })}
          </div>

        </div>
      </section>
    </>
  );
}
