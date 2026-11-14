'use client';

import { useState, useRef } from 'react';
import { motion, useReducedMotion, useInView } from 'framer-motion';

const ease = [0.16, 1, 0.3, 1] as const;

const PROJECTS = [
  {
    id: 'carworks',
    num: '01',
    name: 'CarWorks Canada',
    category: 'Auto Body & Collision Repair',
    location: 'Saint-Laurent, Montréal',
    tagline: 'Collision Repair Done Right.',
    description:
      'A modern website for a full-service collision repair shop in Saint-Laurent. Covers insurance claims handling, frame repair, vinyl wrap, rim customization, and more, giving customers a clear path to a free estimate.',
    src: '/mockup-carworks.jpg',
    href: 'https://carworkscanada.com/',
    url: 'carworkscanada.com',
  },
  {
    id: 'jakoa',
    num: '02',
    name: 'Jakoa Productions',
    category: 'Event Entertainment',
    location: 'Montréal, QC',
    tagline: 'Your Event. Our Energy.',
    description:
      "A personality-driven website for a Montréal DJ and entertainment company specializing in weddings, bar and bat mitzvahs, and corporate events. Built to capture the brand's character and give prospective clients everything they need to start planning.",
    src: '/mockup-jakoa.jpg',
    href: 'https://jakoa.ca/',
    url: 'jakoa.ca',
  },
  {
    id: 'gagnonfils',
    num: '03',
    name: 'Pavage Gagnon & Fils',
    category: 'Paving & Exterior Works',
    location: 'Montréal · Laval · Rive-Sud',
    tagline: 'The driveway that sets the tone.',
    description:
      'A bilingual website for a paving contractor serving Montréal and surrounding areas. Presents asphalt, interlocking, decorative concrete, and sealing services in French and English, with a clear path to a free on-site estimate.',
    src: '/mockup-gagnonfils.jpg',
    href: 'https://gagnonfils.netlify.app/',
    url: 'gagnonfils.netlify.app',
  },
  {
    id: 'lacucina',
    num: '04',
    name: 'La Cucina',
    category: 'Italian Restaurant',
    location: 'Montréal, QC',
    tagline: 'The taste of Italy, fresh every day.',
    description:
      'A bilingual demo website for an Italian restaurant. Fresh pasta, wood-fired pizza, and antipasti — presented with cinematic imagery and a full digital menu to drive reservations and delivery orders.',
    src: 'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=1200&q=80&fit=crop&crop=top',
    href: 'https://lacucina.agentedirect.ca/',
    url: 'lacucina.agentedirect.ca',
  },
  {
    id: 'ciel',
    num: '05',
    name: 'Ciel Residences',
    category: 'Luxury Real Estate',
    location: 'Westmount, Montréal',
    tagline: 'Architect-designed homes above the city.',
    description:
      'A luxury real estate demo for a Montreal brand. A layered dusk hero with a cinematic wordmark, refractive glass navigation, and a room-by-room walkthrough of Villa Lumière in Westmount — shown one scroll at a time.',
    src: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1200&q=80&fit=crop&crop=top',
    href: 'https://cielresidence.netlify.app',
    url: 'cielresidence.netlify.app',
  },
] as const;

type Project = (typeof PROJECTS)[number];

const PROJECTS_FR_META: Record<string, { category: string; tagline: string }> = {
  carworks:   { category: 'Carrosserie & Réparation collision', tagline: 'La carrosserie bien faite.' },
  jakoa:      { category: 'Divertissement événementiel',        tagline: 'Votre événement. Notre énergie.' },
  gagnonfils: { category: 'Pavage & Travaux extérieurs',        tagline: 'L\'entrée qui donne le ton.' },
  lacucina:   { category: 'Restaurant italien',                  tagline: 'Le goût de l\'Italie, frais chaque jour.' },
  ciel:       { category: 'Immobilier de luxe',                 tagline: 'Des maisons d\'architecte au-dessus de la ville.' },
};

const DESCRIPTIONS_FR: Record<string, string> = {
  carworks: "Un site moderne pour un atelier de carrosserie complet à Saint-Laurent. Couvre les réclamations d'assurance, la réparation de châssis, l'enrobage vinyle, la personnalisation de jantes et plus — avec un chemin clair vers une estimation gratuite.",
  jakoa: "Un site à forte personnalité pour un DJ et une compagnie de divertissement montréalais spécialisés dans les mariages, bar et bat mitzvahs et événements corporatifs. Conçu pour capturer l'énergie de la marque et donner à chaque client potentiel tout ce qu'il faut pour commencer à planifier.",
  gagnonfils: "Un site bilingue pour un entrepreneur en pavage desservant Montréal et la région. Présente les services d'asphalte, de pavé uni, de béton décoratif et de scellant en français et en anglais, avec un accès facile à une estimation gratuite sur place.",
  lacucina: "Un site démo bilingue pour un restaurant italien. Pâtes fraîches, pizzas au four à bois et antipasti — présentés avec des images cinématographiques et un menu digital complet pour stimuler les réservations et les commandes en ligne.",
  ciel: "Un site démo immobilier de luxe pour une marque montréalaise. Un héros en couches au crépuscule avec un logotype cinématographique, une navigation en verre réfractif, et une visite pièce par pièce de la Villa Lumière à Westmount — un défilement à la fois.",
};

// ── Browser chrome ────────────────────────────────────────────────────────────

function BrowserFrame({ project, overlay = false }: { project: Project; overlay?: boolean }) {
  return (
    <div style={{
      borderRadius: 12,
      overflow: 'hidden',
      background: '#0E0828',
      border: '1px solid rgba(255,255,255,0.09)',
      boxShadow: '0 20px 70px rgba(0,0,0,0.60), 0 4px 20px rgba(110,30,235,0.22)',
      width: '100%',
      ...(overlay ? { display: 'flex', flexDirection: 'column' as const, height: '100%' } : {}),
    }}>
      {/* Chrome bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '11px 16px',
        background: '#160E34',
        borderBottom: '1px solid rgba(162,75,255,0.07)',
        flexShrink: 0,
      }}>
        {(['#FF5F57', '#FFBD2E', '#28C840'] as const).map(col => (
          <div key={col} style={{
            width: 10, height: 10, borderRadius: '50%',
            background: col, opacity: 0.60, flexShrink: 0,
          }} />
        ))}
        <div style={{
          flex: 1, marginLeft: 10,
          padding: '4px 14px', borderRadius: 6,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.05)',
          overflow: 'hidden', whiteSpace: 'nowrap',
        }}>
          <span style={{ fontSize: 12.5, color: 'rgba(220,210,245,0.60)', fontFamily: 'monospace' }}>
            {project.url}
          </span>
        </div>
      </div>
      {/* Screenshot */}
      <div style={overlay
        ? { flex: 1, overflow: 'hidden', minHeight: 0 }
        : { aspectRatio: '16 / 10', overflow: 'hidden' }
      }>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={project.src}
          alt={`Screenshot of the ${project.name} website`}
          style={{
            width: '100%', height: '100%',
            objectFit: 'cover', objectPosition: 'top center',
            display: 'block',
          }}
          loading="lazy"
        />
      </div>
    </div>
  );
}

// ── Main section ──────────────────────────────────────────────────────────────

export default function SelectedWork({ lang = 'en', overlay = false }: { lang?: 'en' | 'fr'; overlay?: boolean } = {}) {
  const fr = lang === 'fr';
  const reduced = useReducedMotion();
  const [active, setActive] = useState(0);
  const N = PROJECTS.length;
  const ref = useRef<HTMLElement>(null);
  const inViewRaw = useInView(ref, { once: true, margin: '-80px' });
  const inView = overlay || inViewRaw;
  const goPrev = () => setActive(i => (i - 1 + N) % N);
  const goNext = () => setActive(i => (i + 1) % N);

  return (
    <>
      <style>{`
        @media (max-width: 767px) {
          .sw-layout { flex-direction: column !important; }
          .sw-image  { flex: unset !important; width: 100% !important; }
          .sw-info   { flex: unset !important; width: 100% !important; }
          .sw-nav    { width: 100% !important; justify-content: center !important; margin-top: 8px; }
          .sw-nav-btn { width: 44px !important; height: 44px !important; }
          .sw-dot    { height: 9px !important; }
          .sw-dot-active { width: 26px !important; }

          /* Mobile overlay: fixed-height browser frame, full page visible */
          .sw-overlay-section .sw-image { height: 210px !important; overflow: hidden !important; }
          .sw-overlay-section .sw-info  { gap: 8px !important; }
          .sw-overlay-section .sw-nav   { width: 100% !important; justify-content: center !important; gap: 10px !important; }
          .sw-overlay-section .sw-nav-btn { width: 40px !important; height: 40px !important; display: flex !important; }
          .sw-overlay-section .sw-info p[style] { font-size: 12.5px !important; line-height: 1.5 !important; }
        }
      `}</style>

      <section
        ref={ref}
        id="selected-work"
        aria-label="Selected work"
        className={overlay ? 'sw-overlay-section' : ''}
        style={{
          position: 'relative',
          background: '#08041A',
          paddingTop: overlay ? 20 : 'clamp(80px, 10vw, 128px)',
          paddingBottom: overlay ? 20 : 'clamp(80px, 10vw, 120px)',
          overflow: 'hidden',
          ...(overlay ? { height: '100%', display: 'flex', flexDirection: 'column' as const, justifyContent: 'center' } : {}),
        }}
      >
        {/* Atmosphere */}
        <div aria-hidden style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse 65% 40% at 50% 25%, rgba(122,29,240,0.06) 0%, transparent 70%)',
        }} />

        {/* Top hairline */}
        <div aria-hidden style={{
          position: 'absolute', top: 0, left: '6%', right: '6%', height: 1,
          background: 'linear-gradient(90deg, transparent 0%, rgba(162,75,255,0.22) 25%, rgba(162,75,255,0.22) 75%, transparent 100%)',
        }} />

        <div className="max-w-7xl mx-auto px-6 md:px-14">

          {/* ── Header — hidden in overlay (user already knows the section) ── */}
          {!overlay && (
            <motion.div
              initial={{ opacity: 0, y: 22 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.70, ease }}
              style={{ marginBottom: 'clamp(40px, 6vw, 64px)' }}
            >
              <p style={{
                fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                fontSize: 12, fontWeight: 700, letterSpacing: '0.24em',
                textTransform: 'uppercase', color: '#A24BFF', margin: '0 0 20px',
              }}>
                {fr ? 'Travaux sélectionnés' : 'Selected Work'}
              </p>

              <h2 style={{
                fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                fontSize: 'clamp(28px, 4vw, 52px)',
                fontWeight: 800, color: '#F5F0FC',
                letterSpacing: '-0.04em', lineHeight: 1.07,
                margin: '0 0 16px', maxWidth: '20ch',
              }}>
                {fr ? 'Les bonnes entreprises méritent' : 'Good businesses deserve'}{' '}
                <span style={{ color: 'rgba(220,210,245,0.55)' }}>{fr ? 'd\'excellents sites web.' : 'great websites.'}</span>
              </h2>

              <p style={{
                fontSize: 'clamp(15px, 1.3vw,17px)',
                color: 'rgba(220,210,245,0.69)',
                lineHeight: 1.78, maxWidth: 400, margin: 0,
              }}>
                {fr ? 'Chaque entreprise est unique. Son site web aussi.' : 'Every business is different. Its website should be too.'}
              </p>
            </motion.div>
          )}

          {/* ── Projects — grid stack crossfade, controls outside so they never flicker ── */}
          <motion.div
            id="sw-panel"
            role="region"
            aria-label="Selected projects"
            initial={{ opacity: 0, y: 28 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.70, ease, delay: 0.14 }}
          >

            {/* Content stack */}
            <div style={{ display: 'grid', marginBottom: overlay ? 12 : 20 }}>
              {PROJECTS.map((p, i) => (
                <motion.div
                  key={p.id}
                  animate={{
                    opacity: i === active ? 1 : 0,
                    scale:   i === active ? 1 : 0.98,
                  }}
                  transition={{ duration: reduced ? 0 : 0.55, ease }}
                  className="sw-layout"
                  style={{
                    gridArea: '1 / 1',
                    pointerEvents: i === active ? 'auto' : 'none',
                    display: 'flex',
                    alignItems: overlay ? 'stretch' : 'center',
                    gap: overlay ? 'clamp(36px, 4vw, 56px)' : 'clamp(28px, 4vw, 56px)',
                  }}
                >
                  {/* Mockup */}
                  <div className="sw-image" style={{ flex: overlay ? '0 0 52%' : '0 0 58%', minWidth: 0 }}>
                    <BrowserFrame project={p} overlay={overlay} />
                  </div>

                  {/* Info */}
                  <div className="sw-info" style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: overlay ? 18 : 20, ...(overlay ? { justifyContent: 'space-between' } : {}) }}>
                    <p style={{
                      fontSize: 12, fontWeight: 700, letterSpacing: '0.20em',
                      textTransform: 'uppercase', color: 'rgba(220,210,245,0.62)',
                      margin: 0,
                    }}>
                      {fr ? (PROJECTS_FR_META[p.id]?.category ?? p.category) : p.category}
                    </p>

                    <div>
                      <h3 style={{
                        fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                        fontSize: 'clamp(22px, 2.2vw, 32px)',
                        fontWeight: 800, color: '#F5F0FC',
                        letterSpacing: '-0.035em', lineHeight: 1.1,
                        margin: '0 0 5px',
                      }}>
                        {p.name}
                      </h3>
                      <p style={{
                        fontSize: 12.5, fontWeight: 600,
                        color: 'rgba(220,210,245,0.55)',
                        letterSpacing: '0.08em', textTransform: 'uppercase', margin: 0,
                      }}>
                        {p.location}
                      </p>
                    </div>

                    {!overlay && <p style={{
                      fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                      fontSize: 'clamp(16px, 1.55vw, 21px)',
                      fontWeight: 600, fontStyle: 'italic',
                      color: 'rgba(194,141,255,0.89)',
                      letterSpacing: '-0.015em', lineHeight: 1.3, margin: 0,
                    }}>
                      &ldquo;{fr ? (PROJECTS_FR_META[p.id]?.tagline ?? p.tagline) : p.tagline}&rdquo;
                    </p>}

                    <p style={{
                      fontSize: 'clamp(14.5px, 1.1vw,16px)',
                      color: 'rgba(220,210,245,0.70)',
                      lineHeight: 1.80, margin: 0,
                    }}>
                      {fr ? (DESCRIPTIONS_FR[p.id] ?? p.description) : p.description}
                    </p>

                    {(p.id === 'gagnonfils' || p.id === 'lacucina' || p.id === 'ciel') && (
                      <p style={{
                        fontSize: 12, lineHeight: 1.6, margin: 0,
                        color: 'rgba(220,210,245,0.52)',
                        letterSpacing: '0.01em',
                      }}>
                        {fr
                          ? `* Conception uniquement. « ${p.name} » est une entreprise fictive créée à titre illustratif. Aucun client réel, donnée de performance ou résultat commercial n'est implicite ou garanti.`
                          : `* Concept design only. "${p.name}" is a fictional business created for illustration purposes. No real client, performance data, or business results are implied or guaranteed.`}
                      </p>
                    )}

                    <a
                      href={p.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      onMouseEnter={e => {
                        (e.currentTarget as HTMLAnchorElement).style.borderColor = 'rgba(162,75,255,0.60)';
                        (e.currentTarget as HTMLAnchorElement).style.background = 'rgba(162,75,255,0.10)';
                      }}
                      onMouseLeave={e => {
                        (e.currentTarget as HTMLAnchorElement).style.borderColor = 'rgba(162,75,255,0.30)';
                        (e.currentTarget as HTMLAnchorElement).style.background = 'transparent';
                      }}
                      style={{
                        display: 'inline-flex', alignItems: 'center', gap: 7,
                        padding: '10px 22px', borderRadius: 50,
                        border: '1px solid rgba(162,75,255,0.30)',
                        background: 'transparent',
                        color: '#C28DFF', fontSize: 14, fontWeight: 600,
                        textDecoration: 'none', letterSpacing: '0.01em',
                        transition: 'border-color 0.22s ease, background 0.22s ease',
                        alignSelf: 'flex-start', whiteSpace: 'nowrap',
                      }}
                    >
                      {fr ? 'Visiter le site' : 'Visit Live Website'}
                      <svg width="11" height="11" viewBox="0 0 11 11" fill="none" aria-hidden>
                        <path d="M1.5 9.5L9.5 1.5M9.5 1.5H4.5M9.5 1.5V6.5"
                          stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </a>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* ← • • • → — rendered once, never fades */}
            <div
              role="group"
              aria-label="Project navigation"
              className="sw-nav"
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                width: '58%',          /* aligns under the mockup column */
                justifyContent: 'center',
              }}
            >
              <button
                type="button" onClick={goPrev} aria-label="Previous project"
                className="sw-nav-btn"
                style={{
                  width: 30, height: 30, borderRadius: '50%',
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.09)',
                  color: 'rgba(220,210,245,0.72)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', flexShrink: 0,
                  transition: 'background 0.18s, border-color 0.18s',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLButtonElement).style.background = 'rgba(162,75,255,0.14)';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(162,75,255,0.32)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.09)';
                }}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
                  <path d="M7.5 10L3.5 6L7.5 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>

              {PROJECTS.map((_, j) => (
                <button
                  key={j} type="button" onClick={() => setActive(j)}
                  aria-label={`Project ${j + 1}: ${PROJECTS[j].name}`}
                  aria-current={j === active ? 'true' : undefined}
                  className={j === active ? 'sw-dot sw-dot-active' : 'sw-dot'}
                  style={{
                    width: j === active ? 20 : 7, height: 7, borderRadius: 4,
                    background: j === active ? '#8F2BFF' : 'rgba(162,75,255,0.22)',
                    border: 'none', cursor: 'pointer', padding: 0,
                    transition: 'all 0.28s ease', flexShrink: 0,
                  }}
                />
              ))}

              <button
                type="button" onClick={goNext} aria-label="Next project"
                className="sw-nav-btn"
                style={{
                  width: 30, height: 30, borderRadius: '50%',
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.09)',
                  color: 'rgba(220,210,245,0.72)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', flexShrink: 0,
                  transition: 'background 0.18s, border-color 0.18s',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLButtonElement).style.background = 'rgba(162,75,255,0.14)';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(162,75,255,0.32)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.09)';
                }}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
                  <path d="M4.5 2L8.5 6L4.5 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </motion.div>

        </div>
      </section>
    </>
  );
}
