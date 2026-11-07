'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useState } from 'react';

const ease = [0.16, 1, 0.3, 1] as const;

const PROJECTS = [
  {
    id: 'carworks',
    src: '/mockup-carworks.jpg',
    label: 'CarWorks Canada',
    desc: 'Auto body & collision repair out of Montréal. We built them a site that leads with trust, real photos, real reviews, and a clear path to booking.',
    descFr: 'Carrosserie et réparation collision à Montréal. On leur a construit un site qui inspire confiance, avec de vraies photos, de vrais avis, et un chemin clair vers la prise de rendez-vous.',
    href: 'https://carworkscanada.com/',
    demo: false,
  },
  {
    id: 'jakoa',
    src: '/mockup-jakoa.jpg',
    label: 'Jakoa Productions',
    desc: 'DJ & event entertainment based in Montréal. The site captures the energy of the brand, dark, cinematic, and built to convert inquiries.',
    descFr: 'DJ et divertissement événementiel basé à Montréal. Le site capture l\'énergie de la marque, sombre, cinématographique, et conçu pour convertir les demandes.',
    href: 'https://jakoa.ca/',
    demo: false,
  },
  {
    id: 'lacucina',
    src: '/mockup-lacucina.jpg',
    label: 'La Cucina',
    desc: 'Italian restaurant in Pointe-Claire. Warm, inviting design with online reservations and a menu that makes you hungry just scrolling.',
    descFr: 'Restaurant italien à Pointe-Claire. Un design chaleureux et accueillant avec réservations en ligne et un menu qui donne faim rien qu\'en scrollant.',
    href: 'https://lacucina.agentedirect.ca/',
    demo: true,
  },
  {
    id: 'gagnonfils',
    src: '/mockup-gagnonfils.jpg',
    label: 'Pavage Gagnon & Fils',
    desc: 'Asphalt, paving stone & concrete services serving greater Montréal. Clean, authoritative, and built to rank locally.',
    descFr: 'Services d\'asphalte, pavé uni et béton pour le grand Montréal. Propre, autoritaire, et conçu pour bien se positionner localement.',
    href: 'https://gagnonfils.netlify.app/',
    demo: true,
  },
];

function ProjectCard({ src, flipped, demo }: { src: string; href: string; flipped: boolean; demo?: boolean }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'block',
        borderRadius: 16,
        overflow: 'hidden',
        position: 'relative',
        width: '100%',
        aspectRatio: '16/10',
        transformStyle: 'preserve-3d',
        transform: hovered
          ? `perspective(900px) rotateY(0deg) rotateX(0deg) translateZ(28px)`
          : `perspective(900px) rotateY(${flipped ? '6deg' : '-6deg'}) rotateX(3deg) translateZ(0px)`,
        transition: 'transform 0.5s cubic-bezier(0.16,1,0.3,1), box-shadow 0.5s cubic-bezier(0.16,1,0.3,1)',
        boxShadow: hovered
          ? '0 32px 80px rgba(0,0,0,0.65), 0 0 0 1px rgba(139,92,246,0.3)'
          : `${flipped ? '-20px' : '20px'} 24px 60px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.06)`,
        flexShrink: 0,
      }}
    >
      <Image
        src={src}
        alt=""
        fill
        style={{ objectFit: 'cover', objectPosition: 'top' }}
        sizes="(max-width: 768px) 100vw, 50vw"
      />
      <div style={{
        position: 'absolute', inset: 0,
        background: hovered ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,0.12)',
        transition: 'background 0.5s ease',
      }} />
      {demo && (
        <div style={{
          position: 'absolute', top: 12, left: 12,
          background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: 6, padding: '3px 9px',
          fontSize: 10.5, fontWeight: 700, letterSpacing: '0.1em',
          textTransform: 'uppercase', color: 'rgba(255,255,255,0.55)',
        }}>
          Demo
        </div>
      )}
    </div>
  );
}

export default function OurWorkPage({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';
  const homeHref = fr ? '/fr/home' : '/home';
  const beliefsHref = fr ? '/fr/our-beliefs' : '/our-beliefs';
  const projectsHref = fr ? '/fr/our-work' : '/our-work';
  const contactHref = fr ? '/fr/inquiry' : '/inquiry';

  return (
    <main
      className="min-h-screen flex flex-col"
      style={{ background: '#0F0C07', fontFamily: 'var(--font-space-grotesk)' }}
    >
      {/* Nav */}
      <nav className="flex items-center justify-between px-6 py-5 md:px-12">
        <Link href={homeHref}>
          <Image src="/logo.png" alt="AgenteDirect" width={140} height={36} className="h-8 w-auto" />
        </Link>
        <div className="flex items-center gap-6 text-sm font-medium" style={{ color: 'rgba(255,255,255,0.45)' }}>
          <Link href={homeHref} className="hover:text-white transition-colors">{fr ? 'Bureau' : 'Desktop'}</Link>
          <Link href={beliefsHref} className="hover:text-white transition-colors">{fr ? 'Nos croyances' : 'Our Beliefs'}</Link>
          <Link href={projectsHref} style={{ color: 'rgba(255,255,255,0.9)' }}>{fr ? 'Nos projets' : 'Our Projects'}</Link>
          <Link href={contactHref} className="hover:text-white transition-colors">{fr ? 'Contact' : 'Contact'}</Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 4, paddingLeft: 16, borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
            <Link href="/our-work" style={{ fontSize: 12, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.9)' }} className="hover:text-white transition-colors">EN</Link>
            <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 11 }}>|</span>
            <Link href="/fr/our-work" style={{ fontSize: 12, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.3)' }} className="hover:text-white transition-colors">FR</Link>
          </div>
        </div>
      </nav>

      {/* Projects */}
      <div style={{ flex: 1, padding: '40px clamp(24px, 6vw, 80px) 80px' }}>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease }}
          style={{ marginBottom: 72 }}
        >
          <h1 style={{ fontSize: 'clamp(32px, 5vw, 52px)', fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: 0, lineHeight: 1.1, letterSpacing: '-0.02em' }}>
            {fr ? 'Nos projets' : 'Our Projects'}
          </h1>
        </motion.div>

        {/* Project rows */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'clamp(64px, 10vw, 100px)' }}>
          {PROJECTS.map((p, i) => {
            const flipped = i % 2 === 1;
            return (
              <motion.div
                key={p.id}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.8, ease }}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: 'clamp(32px, 5vw, 64px)',
                  alignItems: 'center',
                }}
                className="project-row"
              >
                {/* Card side */}
                <div style={{ order: flipped ? 2 : 1 }}>
                  <ProjectCard src={p.src} href={p.href} flipped={flipped} demo={p.demo} />
                </div>

                {/* Text side */}
                <div style={{ order: flipped ? 1 : 2 }}>
                  <h2 style={{ fontSize: 'clamp(26px, 3vw, 38px)', fontWeight: 700, color: 'rgba(255,255,255,0.92)', margin: '0 0 18px', lineHeight: 1.1, letterSpacing: '-0.02em' }}>
                    {p.label}
                  </h2>
                  <p style={{ fontSize: 17, lineHeight: 1.75, color: 'rgba(255,255,255,0.55)', margin: '0 0 16px', maxWidth: 420 }}>
                    {fr ? p.descFr : p.desc}
                  </p>
                  {p.demo && (
                    <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.28)', margin: '0 0 20px', letterSpacing: '0.01em' }}>
                      {fr ? 'Site démo — conçu pour montrer ce qu\'on peut faire pour ce type d\'entreprise.' : 'Demo site, built to show what we can do for a business like this.'}
                    </p>
                  )}
                  <a
                    href={p.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 8,
                      fontSize: 13, fontWeight: 600, color: 'rgba(220,210,245,0.95)',
                      textDecoration: 'none', letterSpacing: '0.02em',
                      background: 'rgba(139,92,246,0.18)',
                      border: '1px solid rgba(139,92,246,0.4)',
                      padding: '9px 18px', borderRadius: 7,
                      transition: 'background 0.2s, border-color 0.2s',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(139,92,246,0.3)'; (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.7)'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(139,92,246,0.18)'; (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.4)'; }}
                  >
                    {fr ? 'Voir le site' : 'View site'} →
                  </a>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Footer CTA */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 0.7, ease }}
          style={{ marginTop: 96, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 56, textAlign: 'center' }}
        >
          <p style={{ fontSize: 17, color: 'rgba(255,255,255,0.5)', marginBottom: 24 }}>
            {fr ? 'Envie de travailler avec nous?' : 'Want to work with us?'}
          </p>
          <Link
            href={contactHref}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: 'rgba(139,92,246,0.15)', border: '1px solid rgba(139,92,246,0.35)',
              color: 'rgba(220,210,245,0.9)', fontSize: 14, fontWeight: 600,
              padding: '12px 28px', borderRadius: 8, textDecoration: 'none',
              letterSpacing: '0.01em',
            }}
          >
            {fr ? 'Parlons-en' : 'Let\'s talk'} →
          </Link>
        </motion.div>
      </div>

      <style>{`
        @media (max-width: 767px) {
          .project-row {
            grid-template-columns: 1fr !important;
          }
          .project-row > div {
            order: unset !important;
          }
        }
      `}</style>
    </main>
  );
}
