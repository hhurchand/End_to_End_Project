'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';

interface PageNavProps {
  lang?: 'en' | 'fr';
  active: 'beliefs' | 'work' | 'contact' | 'strategy' | 'privacy';
  enHref: string;
  frHref: string;
}

export default function PageNav({ lang = 'en', active, enHref, frHref }: PageNavProps) {
  const fr = lang === 'fr';
  const [open, setOpen] = useState(false);

  const folderMap: Record<string, string> = {
    beliefs: 'our-beliefs',
    work: 'selected-work',
    contact: 'work-with-us',
    strategy: 'the-strategy',
  };
  const openParam = folderMap[active] ? `?open=${folderMap[active]}` : '';
  const homeHref = fr ? `/fr/home${openParam}` : `/home${openParam}`;
  const beliefsHref = fr ? '/fr/our-beliefs' : '/our-beliefs';
  const projectsHref = fr ? '/fr/our-work' : '/our-work';
  const contactHref = fr ? '/fr/inquiry' : '/inquiry';
  const planHref = fr ? '/fr/strategy' : '/strategy';

  const activeStyle = { color: 'rgba(255,255,255,0.9)' };
  const dimStyle = { color: 'rgba(255,255,255,0.66)' };

  return (
    <>
      <nav
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '20px 24px', fontFamily: 'var(--font-space-grotesk)',
        }}
      >
        {/* Logo */}
        <Link href={homeHref} style={{ flexShrink: 0 }}>
          <Image src="/logo.png" alt="AgenteDirect" width={140} height={36} style={{ height: 32, width: 'auto' }} />
        </Link>

        {/* Desktop nav */}
        <div
          className="page-nav-desktop"
          style={{ display: 'flex', alignItems: 'center', gap: 24, fontSize: 15, fontWeight: 500 }}
        >
          <Link href={homeHref} style={dimStyle} className="hover:text-white transition-colors">{fr ? 'Bureau' : 'Desktop'}</Link>
          <Link href={beliefsHref} style={active === 'beliefs' ? activeStyle : dimStyle} className="hover:text-white transition-colors">{fr ? 'Nos croyances' : 'Our Beliefs'}</Link>
          <Link href={projectsHref} style={active === 'work' ? activeStyle : dimStyle} className="hover:text-white transition-colors">{fr ? 'Nos projets' : 'Our Projects'}</Link>
          <Link href={planHref} style={active === 'strategy' ? activeStyle : dimStyle} className="hover:text-white transition-colors">{fr ? 'Services et stratégie' : 'Services & Strategy'}</Link>
          <Link href={contactHref} style={active === 'contact' ? activeStyle : dimStyle} className="hover:text-white transition-colors">{fr ? 'Contact' : 'Contact'}</Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, paddingLeft: 16, borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
            <Link href={enHref} style={{ fontSize: 13, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.57)' : 'rgba(255,255,255,0.9)' }} className="hover:text-white transition-colors">EN</Link>
            <span style={{ color: 'rgba(255,255,255,0.50)', fontSize: 12.5 }}>|</span>
            <Link href={frHref} style={{ fontSize: 13, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.57)' }} className="hover:text-white transition-colors">FR</Link>
          </div>
        </div>

        {/* Mobile right: lang toggle + hamburger */}
        <div className="page-nav-mobile" style={{ display: 'none', alignItems: 'center', gap: 12 }}>
          {/* EN/FR visible in top bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Link href={enHref} style={{ fontSize: 13, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.57)' : 'rgba(255,255,255,0.9)', textDecoration: 'none' }}>EN</Link>
            <span style={{ color: 'rgba(255,255,255,0.50)', fontSize: 12.5 }}>|</span>
            <Link href={frHref} style={{ fontSize: 13, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.57)', textDecoration: 'none' }}>FR</Link>
          </div>
          {/* Hamburger */}
          <button
            onClick={() => setOpen(o => !o)}
            aria-label="Menu"
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 6, display: 'flex', flexDirection: 'column', gap: 5, alignItems: 'center', justifyContent: 'center' }}
          >
            <span style={{ display: 'block', width: 22, height: 1.5, background: 'rgba(255,255,255,0.8)', transformOrigin: 'center', transform: open ? 'translateY(3.25px) rotate(45deg)' : 'none', transition: 'transform 0.22s ease' }} />
            <span style={{ display: 'block', width: 22, height: 1.5, background: 'rgba(255,255,255,0.8)', opacity: open ? 0 : 1, transition: 'opacity 0.15s ease' }} />
            <span style={{ display: 'block', width: 22, height: 1.5, background: 'rgba(255,255,255,0.8)', transformOrigin: 'center', transform: open ? 'translateY(-3.25px) rotate(-45deg)' : 'none', transition: 'transform 0.22s ease' }} />
          </button>
        </div>
      </nav>

      {/* Mobile full-screen menu overlay */}
      {open && (
        <div
          className="page-nav-mobile"
          style={{
            position: 'fixed', inset: 0, zIndex: 50,
            background: 'rgba(8,4,22,0.97)', backdropFilter: 'blur(20px)',
            display: 'none', flexDirection: 'column',
            padding: '20px 24px',
            fontFamily: 'var(--font-space-grotesk)',
          }}
        >
          {/* Top row */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 48 }}>
            <Link href={homeHref} onClick={() => setOpen(false)}>
              <Image src="/logo.png" alt="AgenteDirect" width={140} height={36} style={{ height: 32, width: 'auto' }} />
            </Link>
            <button onClick={() => setOpen(false)} aria-label="Close menu" style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 8 }}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <line x1="2" y1="2" x2="18" y2="18" stroke="rgba(255,255,255,0.7)" strokeWidth="1.5" strokeLinecap="round"/>
                <line x1="18" y1="2" x2="2" y2="18" stroke="rgba(255,255,255,0.7)" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          </div>

          {/* Nav links */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {[
              { href: homeHref, label: fr ? 'Bureau' : 'Desktop', isActive: false },
              { href: beliefsHref, label: fr ? 'Nos croyances' : 'Our Beliefs', isActive: active === 'beliefs' },
              { href: projectsHref, label: fr ? 'Nos projets' : 'Our Projects', isActive: active === 'work' },
              { href: planHref, label: fr ? 'Services et stratégie' : 'Services & Strategy', isActive: active === 'strategy' },
              { href: contactHref, label: fr ? 'Contact' : 'Contact', isActive: active === 'contact' },
            ].map(item => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                style={{
                  fontSize: 28, fontWeight: 700, letterSpacing: '-0.02em',
                  color: item.isActive ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.60)',
                  textDecoration: 'none', padding: '10px 0',
                  borderBottom: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @media (max-width: 767px) {
          .page-nav-desktop { display: none !important; }
          .page-nav-mobile { display: flex !important; }
        }
        @media (min-width: 768px) {
          .page-nav-mobile { display: none !important; }
          .page-nav-desktop { display: flex !important; }
        }
      `}</style>
    </>
  );
}
