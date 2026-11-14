'use client';

import { Fragment, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import Link from 'next/link';

const ease = [0.16, 1, 0.3, 1] as const;

const STEPS = [
  {
    id: 'tell',
    num: '01',
    title: 'Tell us about your business.',
    body: 'We learn about your services, customers, goals, and what you want your website to accomplish.',
  },
  {
    id: 'build',
    num: '02',
    title: 'We bring your vision to life.',
    body: 'We design and build a custom website that reflects your business and makes the next step clear for customers.',
  },
  {
    id: 'launch',
    num: '03',
    title: 'Review, refine, launch.',
    body: 'We work through your feedback, finalize the details, and get your website ready to go live.',
  },
] as const;

const STEPS_FR = [
  {
    id: 'tell',
    num: '01',
    title: 'Parlez-nous de votre entreprise.',
    body: 'On apprend à connaître vos services, vos clients, vos objectifs et ce que vous voulez que votre site accomplit.',
  },
  {
    id: 'build',
    num: '02',
    title: 'On concrétise votre vision.',
    body: 'On conçoit et développe un site sur mesure qui reflète votre entreprise et guide clairement vos clients vers la prochaine étape.',
  },
  {
    id: 'launch',
    num: '03',
    title: 'Révision, ajustements, lancement.',
    body: 'On intègre vos retours, finalisons les détails et préparons votre site pour la mise en ligne.',
  },
] as const;

export default function OurProcess({ lang = 'en', overlay = false }: { lang?: 'en' | 'fr'; overlay?: boolean } = {}) {
  const fr = lang === 'fr';
  const steps = fr ? STEPS_FR : STEPS;
  const ref = useRef<HTMLElement>(null);
  const inViewRaw = useInView(ref, { once: true, margin: '-80px' });
  const inView = overlay || inViewRaw;

  return (
    <>
      <style>{`
        /* Desktop: 3-column, centered steps, horizontal connector visible */
        @media (min-width: 768px) {
          .op-grid {
            grid-template-columns: repeat(3, 1fr) !important;
          }
          .op-step {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
            border-top: none !important;
            padding-top: 32px !important;
            padding-left: 28px !important;
            padding-right: 28px !important;
            padding-bottom: 0 !important;
          }
          .op-step-content { align-items: center !important; }
          .op-connector { display: block !important; }
          .op-vline { display: none !important; }
        }
      `}</style>

      <section
        ref={ref}
        aria-label="Our process"
        style={{
          position: 'relative',
          background: '#08041A',
          paddingTop: overlay ? 'clamp(32px, 5vw, 40px)' : 'clamp(80px, 10vw, 120px)',
          paddingBottom: overlay ? 'clamp(40px, 5vw, 48px)' : 'clamp(80px, 10vw, 120px)',
          overflow: 'hidden',
          ...(overlay ? { height: '100%', display: 'flex', flexDirection: 'column' as const, justifyContent: 'center' } : {}),
        }}
      >
        {/* Atmosphere */}
        <div aria-hidden style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse 55% 45% at 50% 50%, rgba(122,29,240,0.06) 0%, transparent 70%)',
        }} />

        {/* Top hairline */}
        <div aria-hidden style={{
          position: 'absolute', top: 0, left: '6%', right: '6%', height: 1,
          background: 'linear-gradient(90deg, transparent 0%, rgba(162,75,255,0.22) 25%, rgba(162,75,255,0.22) 75%, transparent 100%)',
        }} />

        <div className="max-w-7xl mx-auto px-6 md:px-14">

          {/* ── Header ── */}
          <motion.div
            initial={{ opacity: 0, y: 22 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.70, ease }}
            style={{ marginBottom: overlay ? 24 : 'clamp(48px, 7vw, 72px)', maxWidth: '44ch' }}
          >
            <p style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 12, fontWeight: 700, letterSpacing: '0.24em',
              textTransform: 'uppercase', color: '#A24BFF', margin: '0 0 20px',
            }}>
              {fr ? 'Notre processus' : 'Our Process'}
            </p>

            <h2 style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 'clamp(28px, 4vw, 48px)',
              fontWeight: 800, color: '#F5F0FC',
              letterSpacing: '-0.04em', lineHeight: 1.07,
              margin: '0 0 18px',
            }}>
              {fr ? 'De la première conversation' : 'From first conversation'}{' '}
              <span style={{ color: 'rgba(220,210,245,0.55)' }}>{fr ? 'au lancement.' : 'to launch.'}</span>
            </h2>

            <p style={{
              fontSize: 'clamp(15px, 1.3vw,17px)',
              color: 'rgba(220,210,245,0.69)',
              lineHeight: 1.78, margin: 0,
            }}>
              {fr ? 'Pas de processus compliqué. Juste un site conçu pour votre entreprise.' : 'No confusing process. Just a website built around your business.'}
            </p>
          </motion.div>

          {/* ── Steps ── */}
          <div style={{ position: 'relative' }}>

            {/*
              Desktop horizontal connector line.
              Sits at badge centre (paddingTop 32 + badge radius 18 = 50px).
              Spans from centre of col-1 to centre of col-3 (~1/6 → 5/6 of width).
            */}
            <div
              className="op-connector"
              aria-hidden
              style={{
                display: 'none',
                position: 'absolute',
                top: 50,
                left: 'calc(100% / 6)',
                right: 'calc(100% / 6)',
                height: 1,
                background: 'linear-gradient(90deg, transparent, rgba(162,75,255,0.18) 20%, rgba(162,75,255,0.18) 80%, transparent)',
                pointerEvents: 'none',
                zIndex: 0,
              }}
            />

            <div
              className="op-grid"
              style={{ display: 'grid', gridTemplateColumns: '1fr' }}
            >
              {steps.map((step, i) => (
                <Fragment key={step.id}>
                  <motion.div
                    className="op-step"
                    initial={{ opacity: 0, y: 24 }}
                    animate={inView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.65, ease, delay: 0.14 + i * 0.11 }}
                    style={{
                      display: 'flex',
                      flexDirection: 'row',    /* mobile: badge left, text right */
                      alignItems: 'flex-start',
                      gap: 20,
                      borderTop: i > 0 ? '1px solid rgba(162,75,255,0.08)' : 'none',
                      paddingTop: i > 0 ? 28 : 0,
                      paddingBottom: 0,
                    }}
                  >
                    {/* Badge + mobile vline column */}
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                      <div style={{
                        width: 36, height: 36, borderRadius: '50%',
                        background: '#0A041E',
                        border: '1.5px solid rgba(162,75,255,0.28)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        flexShrink: 0,
                        position: 'relative', zIndex: 1,
                      }}>
                        <span style={{
                          fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                          fontSize: 12, fontWeight: 700,
                          color: '#C28DFF', letterSpacing: '0.06em',
                        }}>
                          {step.num}
                        </span>
                      </div>

                      {/* Mobile: vertical line below badge (not last step) */}
                      {i < STEPS.length - 1 && (
                        <div
                          className="op-vline"
                          aria-hidden
                          style={{
                            width: 1,
                            flex: 1,
                            minHeight: 28,
                            marginTop: 8,
                            background: 'rgba(162,75,255,0.18)',
                          }}
                        />
                      )}
                    </div>

                    {/* Step text */}
                    <div
                      className="op-step-content"
                      style={{
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'flex-start',
                        gap: 10,
                        paddingBottom: i < STEPS.length - 1 ? 28 : 0,
                      }}
                    >
                      <h3 style={{
                        fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                        fontSize: 'clamp(16px, 1.4vw, 18px)',
                        fontWeight: 700, color: '#F5F0FC',
                        letterSpacing: '-0.02em', lineHeight: 1.2, margin: 0,
                      }}>
                        {step.title}
                      </h3>
                      <p style={{
                        fontSize: 'clamp(14.5px, 1.05vw,16px)',
                        color: 'rgba(220,210,245,0.69)',
                        lineHeight: 1.75, margin: 0,
                      }}>
                        {step.body}
                      </p>
                    </div>
                  </motion.div>
                </Fragment>
              ))}
            </div>
          </div>

          {/* ── CTA ── */}
          {!overlay && <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.60, ease, delay: 0.50 }}
            style={{
              marginTop: 'clamp(56px, 8vw, 88px)',
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', textAlign: 'center', gap: 20,
            }}
          >
            {/* Subtle vertical thread above CTA */}
            <div aria-hidden style={{
              width: 1, height: 44,
              background: 'linear-gradient(to bottom, transparent, rgba(162,75,255,0.35))',
            }} />

            <p style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 'clamp(20px, 2.2vw, 28px)',
              fontWeight: 700, color: '#F5F0FC',
              letterSpacing: '-0.03em', lineHeight: 1.2, margin: 0,
            }}>
              {fr ? 'Vous avez un projet en tête\u00a0?' : 'Have a project in mind?'}
            </p>

            <Link
              href={fr ? '/fr/inquiry' : '/inquiry'}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLAnchorElement;
                el.style.background = 'linear-gradient(135deg, #9D6FF5 0%, #8F2BFF 100%)';
                el.style.boxShadow = '0 8px 36px rgba(122,29,240,0.60)';
                el.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLAnchorElement;
                el.style.background = 'linear-gradient(135deg, #A24BFF 0%, #7A1DF0 100%)';
                el.style.boxShadow = '0 4px 24px rgba(122,29,240,0.44)';
                el.style.transform = 'translateY(0)';
              }}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '13px 28px', borderRadius: 50,
                background: 'linear-gradient(135deg, #A24BFF 0%, #7A1DF0 100%)',
                color: '#fff', fontSize: 15, fontWeight: 700,
                textDecoration: 'none', letterSpacing: '0.01em',
                boxShadow: '0 4px 24px rgba(122,29,240,0.44)',
                transition: 'background 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease',
              }}
            >
              {fr ? 'Commencer' : 'Get Started'}
              <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden>
                <path d="M1.5 9.5L9.5 1.5M9.5 1.5H4.5M9.5 1.5V6.5"
                  stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
          </motion.div>}

        </div>
      </section>
    </>
  );
}
