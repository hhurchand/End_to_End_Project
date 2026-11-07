'use client';

import { motion, useReducedMotion, AnimatePresence } from 'framer-motion';
import { useState, useCallback, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import WhyItMatters from '@/components/sections/WhyItMatters';
import WhatWeDo from '@/components/sections/WhatWeDo';
import SelectedWork from '@/components/sections/SelectedWork';
import OurProcess from '@/components/sections/OurProcess';

const ease = [0.16, 1, 0.3, 1] as const;
const hidden = { opacity: 0, y: 20 };

/* Real client work — URLs verified */
const CARDS = [
  {
    id: 'carworks',
    src: '/mockup-carworks.jpg',
    label: 'CarWorks Canada',
    desc: 'Auto body & collision repair · Montréal, QC',
    url: 'carworkscanada.com',
    href: 'https://carworkscanada.com/',
  },
  {
    id: 'jakoa',
    src: '/mockup-jakoa.jpg',
    label: 'Jakoa Productions',
    desc: 'DJ & event entertainment · Montréal, QC',
    url: 'jakoa.ca',
    href: 'https://jakoa.ca/',
  },
  {
    id: 'gagnonfils',
    src: '/mockup-gagnonfils.jpg',
    label: 'Pavage Gagnon & Fils',
    desc: 'Asphalte, pavé uni & béton · Montréal, QC',
    url: 'gagnonfils.netlify.app',
    href: 'https://gagnonfils.netlify.app/',
  },
  {
    id: 'lacucina',
    src: '/mockup-lacucina.jpg',
    label: 'La Cucina',
    desc: 'Italian cuisine · Pointe-Claire, QC',
    url: 'lacucina.agentedirect.ca',
    href: 'https://lacucina.agentedirect.ca/',
  },
  {
    id: 'ciel',
    src: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=1200&q=80&fit=crop&crop=top',
    label: 'Ciel Residences',
    desc: 'Luxury real estate · Westmount, Montréal',
    url: 'cielresidence.netlify.app',
    href: 'https://cielresidence.netlify.app',
  },
];

const BENEFITS = ['Built Around Your Business', 'Mobile-First Design', 'Conversion-Focused'];
const BENEFITS_FR = ['Sur mesure pour votre entreprise', 'Design mobile d\'abord', 'Axé sur la conversion'];

const N = CARDS.length;
const CARD_W = 340;  // active card width
const SIDE_X = 370;  // px offset from center to adjacent card center
const TOP_PAD = 16;  // clearance so top of cards isn't clipped by overflow:hidden

function getOffset(idx: number, active: number): number {
  let o = idx - active;
  if (o > N / 2) o -= N;
  if (o < -N / 2) o += N;
  return o;
}

function MacFolder({ label, selected = false, size = 'md' }: { label: string; selected?: boolean; size?: 'sm' | 'md' | 'lg' }) {
  const w = size === 'sm' ? 46 : size === 'lg' ? 82 : 62;
  const h = size === 'sm' ? 40 : size === 'lg' ? 72 : 54;
  const body = selected ? '#6D28D9' : '#7C3AED';
  const tab  = selected ? '#5B21B6' : '#6D28D9';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
      <svg width={w} height={h} viewBox="0 0 62 54" fill="none" aria-hidden>
        <path d="M3 18 Q3 11 10 11 L26 11 Q32 11 34 18 L3 18Z" fill={tab}/>
        <rect x="1" y="17" width="60" height="36" rx="5" fill={body}/>
        <rect x="3" y="19" width="56" height="8" rx="3" fill="rgba(255,255,255,0.16)"/>
        <rect x="1" y="43" width="60" height="10" rx="5" fill="rgba(0,0,0,0.14)"/>
        {selected && <rect x="0.5" y="16.5" width="61" height="37" rx="5.5" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="1.2"/>}
      </svg>
      <span style={{
        fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif",
        fontSize: size === 'sm' ? 10 : size === 'lg' ? 13 : 11, fontWeight: 600, letterSpacing: '0.03em',
        color: 'rgba(255,255,255,0.92)', textShadow: '0 1px 8px rgba(0,0,0,0.9)',
        whiteSpace: 'nowrap', textAlign: 'center',
      }}>{label}</span>
    </div>
  );
}

function FinderWindow({ fr, onSectionClick, isTouch, playClickSound }: { fr: boolean; onSectionClick: (id: string) => void; isTouch: boolean; playClickSound: () => void }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const sections = [
    { label: fr ? 'Nos croyances' : 'Our Beliefs', id: 'our-beliefs' },
    { label: fr ? 'Nos projets' : 'Our Projects', id: 'selected-work' },
    { label: fr ? 'Travaillez avec nous' : 'Work With Us', id: 'work-with-us' },
  ];
  return (
    <>
      <style>{`
        @media (max-width: 767px) {
          /* FinderWindow sizing */
          .fw-outer { width: calc(100vw - 24px) !important; max-width: 420px !important; }
          .fw-sidebar { display: none !important; }
          .fw-body { height: 290px !important; }
          .fw-hint { font-size: 11px !important; white-space: normal !important; text-align: center; }

          /* Intro overlay — natural scroll, full layout */
          .intro-copy-section { height: auto !important; justify-content: flex-start !important; overflow: visible !important; padding: 32px 24px 48px !important; }
          .intro-grid { grid-template-columns: 1fr !important; height: auto !important; gap: 40px !important; }
          .intro-copy-col { gap: 20px !important; height: auto !important; min-height: unset !important; }
          .intro-copy-section h2 { font-size: 34px !important; line-height: 1.08 !important; }
          .intro-copy-section .intro-copy-desc { font-size: 18px !important; line-height: 1.6 !important; color: rgba(220,210,245,0.65) !important; }

          /* Overlay sections — natural height, scrollable */
          .overlay-content section { height: auto !important; overflow: visible !important; min-height: unset !important; }

          /* Hide native iOS video controls */
          video::-webkit-media-controls { display: none !important; }
          video::-webkit-media-controls-enclosure { display: none !important; }
          video::-webkit-media-controls-start-playback-button { display: none !important; }
        }

      `}</style>
      <div className="fw-outer" style={{ width: 640, borderRadius: 12, overflow: 'hidden', background: 'rgba(10,5,28,0.96)', backdropFilter: 'blur(28px)', WebkitBackdropFilter: 'blur(28px)', border: '1px solid rgba(139,92,246,0.22)', boxShadow: '0 28px 80px rgba(0,0,0,0.80)' }}>
        {/* Title bar */}
        <div style={{ height: 36, background: 'rgba(18,8,42,0.99)', display: 'flex', alignItems: 'center', padding: '0 14px', gap: 7, borderBottom: '1px solid rgba(139,92,246,0.12)', position: 'relative' }}>
          {(['#FF5F57','#FFBD2E','#28C840'] as const).map(c => <div key={c} style={{ width: 12, height: 12, borderRadius: '50%', background: c, flexShrink: 0 }} />)}
          <span style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 600, color: 'rgba(220,210,245,0.50)', pointerEvents: 'none', letterSpacing: '0.01em' }}>{fr ? 'Site Web' : 'Website'}</span>
        </div>
        {/* Body */}
        <div className="fw-body" style={{ display: 'flex', height: 320 }}>
          {/* Sidebar */}
          <div className="fw-sidebar" style={{ width: 128, borderRight: '1px solid rgba(139,92,246,0.09)', padding: '10px 0', background: 'rgba(8,4,22,0.7)', flexShrink: 0 }}>
            <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: 'rgba(139,92,246,0.45)', margin: '0 0 6px', padding: '0 12px' }}>{fr ? 'Favoris' : 'Favorites'}</p>
            <div style={{ padding: '5px 12px', fontSize: 11.5, color: 'rgba(220,210,245,0.38)', display: 'flex', alignItems: 'center', gap: 7 }}>
              <svg width="13" height="11" viewBox="0 0 13 11" fill="none"><rect x="0.5" y="2.5" width="12" height="8" rx="1.5" fill="rgba(124,58,237,0.45)"/><path d="M0.5 4.5h12" stroke="rgba(124,58,237,0.3)" strokeWidth="0.75"/></svg>
              AgenteDirect
            </div>
          </div>
          {/* Right panel — folders centered, hint pinned to bottom */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }} onClick={e => { if (e.target === e.currentTarget) setSelectedId(null); }}>
            {/* File grid — auto height, not flex:1 */}
            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '14px 8px', alignContent: 'center', padding: '0 16px' }}>
              {sections.map((s, i) => (
                <motion.button
                  key={s.id} type="button"
                  onClick={() => {
                    if (isTouch) { onSectionClick(s.id); return; }
                    setSelectedId(s.id);
                  }}
                  onDoubleClick={!isTouch ? () => { setSelectedId(null); onSectionClick(s.id); } : undefined}
                  initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.07, duration: 0.32, ease: [0.16,1,0.3,1] }}
                  style={{
                    background: selectedId === s.id ? 'rgba(124,58,237,0.22)' : 'none',
                    border: 'none', cursor: 'pointer', padding: '10px 6px',
                    borderRadius: 7, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    outline: 'none',
                  }}
                >
                  <MacFolder label={s.label} size="lg" selected={selectedId === s.id} />
                </motion.button>
              ))}
            </div>
            {/* Disclaimer pinned to bottom of finder */}
            <div style={{ padding: '0 16px 16px', textAlign: 'center' }}>
              <p className="fw-hint" style={{ margin: 0, fontSize: 12.5, fontWeight: 500, color: 'rgba(255,255,255,0.55)', whiteSpace: 'nowrap' }}>
                {fr ? 'Cliquez la section que vous souhaitez découvrir' : isTouch ? "Tap a section to explore" : "Click the section you'd like to discover"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default function HomePage({ lang = 'en' }: { lang?: 'en' | 'fr' } = {}) {
  const fr = lang === 'fr';
  const router = useRouter();
  const [active, setActive] = useState(0);

  const reduced = useReducedMotion();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showFolders, setShowFolders] = useState(false);
  const [showCursor, setShowCursor] = useState(false);
  const [cursorClick, setCursorClick] = useState(false);
  const [folderOpened, setFolderOpened] = useState(false);
  const [unlocked, setUnlocked] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const sequenceStarted = useRef(false);
  const [isTouch, setIsTouch] = useState(false);
  const isTouchRef = useRef(false);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const touch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    setIsTouch(touch);
    isTouchRef.current = touch;
  }, []);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = 2.2;
  }, []);

  // Pre-unlock AudioContext on first gesture — play a silent 1-sample buffer
  // to fully unblock the context before the double-click sound fires
  useEffect(() => {
    const unlock = () => {
      try {
        if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
          audioCtxRef.current = new AudioContext();
        }
        const ctx = audioCtxRef.current;
        ctx.resume().then(() => {
          const buf = ctx.createBuffer(1, 1, ctx.sampleRate);
          const src = ctx.createBufferSource();
          src.buffer = buf;
          src.connect(ctx.destination);
          src.start(0);
        }).catch(() => {});
      } catch { /* silent */ }
      document.removeEventListener('click', unlock, true);
      document.removeEventListener('touchstart', unlock, true);
    };
    document.addEventListener('click', unlock, true);
    document.addEventListener('touchstart', unlock, true);
    return () => {
      document.removeEventListener('click', unlock, true);
      document.removeEventListener('touchstart', unlock, true);
    };
  }, []);


  // Always lock page scroll — this is a desktop metaphor, sections open as overlays
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = ''; };
  }, []);


  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    const start = window.scrollY;
    const target = el.getBoundingClientRect().top + window.scrollY;
    const dur = 1100;
    const startTime = performance.now();
    const ease = (t: number) => t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2;
    const tick = (now: number) => {
      const p = Math.min((now - startTime) / dur, 1);
      window.scrollTo(0, start + (target - start) * ease(p));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const playClickSound = () => {
    try {
      const ctx = audioCtxRef.current && audioCtxRef.current.state !== 'closed'
        ? audioCtxRef.current
        : new AudioContext();
      audioCtxRef.current = ctx;
      const fire = () => {
        const makeClick = (time: number) => {
          const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * 0.035), ctx.sampleRate);
          const data = buf.getChannelData(0);
          for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 10);
          const src = ctx.createBufferSource();
          src.buffer = buf;
          const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 1400;
          const gain = ctx.createGain();
          gain.gain.setValueAtTime(1.5, time);
          gain.gain.exponentialRampToValueAtTime(0.001, time + 0.035);
          src.connect(hp).connect(gain).connect(ctx.destination);
          src.start(time); src.stop(time + 0.035);
          const osc = ctx.createOscillator(); osc.type = 'sine';
          osc.frequency.setValueAtTime(200, time);
          osc.frequency.exponentialRampToValueAtTime(65, time + 0.025);
          const og = ctx.createGain();
          og.gain.setValueAtTime(0.55, time);
          og.gain.exponentialRampToValueAtTime(0.001, time + 0.025);
          osc.connect(og).connect(ctx.destination);
          osc.start(time); osc.stop(time + 0.025);
        };
        const now = ctx.currentTime;
        makeClick(now);
        makeClick(now + 0.12);
      };
      if (ctx.state === 'suspended') {
        ctx.resume().then(fire).catch(() => {});
      } else {
        fire();
      }
    } catch { /* silent fail */ }
  };

  const handleSectionClick = (id: string) => {
    playClickSound();
    if (id === 'our-beliefs') { setTimeout(() => router.push(fr ? '/fr/our-beliefs' : '/our-beliefs'), 200); return; }
    if (id === 'selected-work') { setTimeout(() => router.push(fr ? '/fr/our-work' : '/our-work'), 200); return; }
    if (id === 'work-with-us') { setTimeout(() => router.push(fr ? '/fr/inquiry' : '/inquiry'), 200); return; }
    setUnlocked(true);
    setActiveSection(id);
  };

  // Desktop folder sequence — triggered when video finishes
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const onEnded = () => {
      if (sequenceStarted.current) return;
      sequenceStarted.current = true;
      setShowFolders(true);
      setTimeout(() => setShowCursor(true), 800);
      setTimeout(() => { setCursorClick(true); playClickSound(); }, 2000);
      setTimeout(() => { setFolderOpened(true); setUnlocked(true); }, 2500);
    };
    vid.addEventListener('ended', onEnded);
    return () => vid.removeEventListener('ended', onEnded);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);



  const prev = useCallback(() => setActive(i => (i - 1 + N) % N), []);
  const next = useCallback(() => setActive(i => (i + 1) % N), []);

  // Keyboard navigation for carousel
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'ArrowLeft')  prev();
      if (e.key === 'ArrowRight') next();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [prev, next]);

  return (
    <main
      className="relative"
      style={{ background: '#08041A', minHeight: '100dvh' }}
    >
      {/* ── Page entrance — curtain sweeps up from dark, matching splash exit ── */}
      {!reduced && (
        <>
          {/* Purple bloom burst as curtain lifts */}
          <motion.div
            aria-hidden
            style={{
              position: 'fixed', inset: 0, zIndex: 98, pointerEvents: 'none',
              background: 'radial-gradient(ellipse 55% 45% at 50% 50%, rgba(109,40,217,0.55) 0%, rgba(80,20,180,0.18) 55%, transparent 78%)',
            }}
            initial={{ opacity: 0, scale: 0.75 }}
            animate={{ opacity: [0, 0.85, 0], scale: [0.75, 1.3, 1.8] }}
            transition={{ delay: 0.1, duration: 0.85, ease: [0.16, 1, 0.3, 1] }}
          />
          {/* Dark curtain — starts covering full screen, sweeps upward */}
          <motion.div
            aria-hidden
            style={{
              position: 'fixed', inset: 0, zIndex: 99,
              background: '#08041A',
              transformOrigin: 'top center',
              pointerEvents: 'none',
            }}
            initial={{ y: 0 }}
            animate={{ y: '-100%' }}
            transition={{ delay: 0.06, duration: 0.80, ease: [0.76, 0, 0.24, 1] }}
          />
        </>
      )}

      {/* ── Atmospheric lighting ── */}
      <div aria-hidden style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 72% 62% at 50% 44%, rgba(109,40,217,0.13) 0%, transparent 70%)',
      }} />
      <div aria-hidden style={{
        position: 'absolute', top: 0, right: 0, width: '42%', height: '50%', pointerEvents: 'none',
        background: 'radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.07) 0%, transparent 60%)',
      }} />
      {/* Film grain */}
      <svg aria-hidden style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', opacity: 0.024 }}>
        <filter id="hg">
          <feTurbulence type="fractalNoise" baseFrequency="0.78" numOctaves="4" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#hg)" />
      </svg>

      {/* ── NAV ── */}
      <motion.nav
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.55, duration: 0.65, ease }}
        className="z-20 flex items-center justify-between px-6 md:px-14"
        style={{ position: 'fixed', top: 0, left: 0, right: 0, height: 64, background: 'transparent' }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/image.png"
          alt="AgenteDirect"
          style={{ height: 36, width: 'auto', filter: 'brightness(0) invert(1)', opacity: 0.90 }}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>

          {/* FR / EN toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Link href="/home" style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', color: fr ? 'rgba(220,210,245,0.38)' : '#A78BFA', textDecoration: 'none', padding: '4px 6px', borderRadius: 4, transition: 'color 0.2s' }}>EN</Link>
            <span style={{ color: 'rgba(139,92,246,0.25)', fontSize: 11 }}>|</span>
            <Link href="/fr/home" style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', color: fr ? '#A78BFA' : 'rgba(220,210,245,0.38)', textDecoration: 'none', padding: '4px 6px', borderRadius: 4, transition: 'color 0.2s' }}>FR</Link>
          </div>


        </div>
      </motion.nav>


      {/* ── HERO ── */}
      <section aria-label="Hero" style={{ position: 'relative', height: '100dvh', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <video
          ref={videoRef}
          autoPlay muted playsInline aria-hidden
          style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%) scale(1.22)', minWidth: '100%', minHeight: '100%', width: 'auto', height: 'auto', objectFit: 'cover' }}
        >
          <source src="/kling_timeline_4k.mp4" type="video/mp4" />
        </video>
        {/* ── DESKTOP FOLDER SYSTEM ── */}
        {/* ── DESKTOP FOLDER — visible as soon as video screen lights up ── */}
        {showFolders && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 10, pointerEvents: folderOpened ? 'auto' : 'none' }}>

            {/* "Website" folder — upper-left of screen */}
            {!folderOpened && (
              <motion.div
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="fw-single-folder"
                style={{ position: 'absolute', top: isTouch ? '30%' : '38%', left: isTouch ? '50%' : '16%', transform: 'translate(-50%, -50%)', pointerEvents: 'none' }}
              >
                <MacFolder label={fr ? 'Site Web' : 'Website'} selected={cursorClick} />
              </motion.div>
            )}

            {/* Mobile: purple atmosphere overlay covers the video when finder opens */}
            <AnimatePresence>
              {folderOpened && isTouch && (
                <motion.div
                  key="mobile-bg"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
                  aria-hidden
                  style={{
                    position: 'absolute', inset: 0, zIndex: 8,
                    background: '#08041A',
                    pointerEvents: 'none',
                  }}
                >
                  {/* Atmospheric glow */}
                  <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 80% 60% at 50% 44%, rgba(109,40,217,0.22) 0%, transparent 70%)' }} />
                  <div style={{ position: 'absolute', top: 0, right: 0, width: '55%', height: '55%', background: 'radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.10) 0%, transparent 60%)' }} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Finder window opens after double-click */}
            <AnimatePresence>
              {folderOpened && (
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 9 }}>
                  <motion.div
                    key="finder"
                    initial={{ opacity: 0, scale: 0.88, y: 16 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ duration: 0.42, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <FinderWindow fr={fr} onSectionClick={handleSectionClick} isTouch={isTouch} playClickSound={playClickSound} />
                  </motion.div>
                </div>
              )}
            </AnimatePresence>


            {/* Animated cursor glides in and double-clicks */}
            <AnimatePresence>
              {showCursor && !folderOpened && (
                <motion.div
                  key="cursor"
                  aria-hidden
                  style={{ position: 'absolute', zIndex: 20, pointerEvents: 'none' }}
                  initial={{ opacity: 0, top: isTouch ? '18%' : '20%', left: isTouch ? '54%' : '22%' }}
                  animate={{ opacity: 1, top: isTouch ? '33%' : '40%', left: isTouch ? '54%' : '17.5%' }}
                  transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                >
                  <motion.div
                    animate={cursorClick ? { scale: [1, 0.72, 1, 0.72, 1] } : {}}
                    transition={{ duration: 0.22, times: [0, 0.2, 0.5, 0.7, 1] }}
                  >
                    <svg width="22" height="26" viewBox="0 0 22 26" fill="none" aria-hidden>
                      <path d="M2 2l6 18 3-5.5 6 3L2 2z" fill="white" stroke="rgba(0,0,0,0.55)" strokeWidth="1.2" strokeLinejoin="round"/>
                    </svg>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Transparent tap interceptor — blocks native iOS play button */}
        <div aria-hidden style={{ position: 'absolute', inset: 0, zIndex: 1, background: 'rgba(8,4,26,0.28)', cursor: 'default' }} onClick={e => {
          e.preventDefault();
          if (!showFolders && videoRef.current?.paused) videoRef.current.play().catch(() => {});
        }} />
        <div aria-hidden style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '18%', zIndex: 2, background: 'linear-gradient(to top, #08041A 0%, transparent 100%)' }} />
        {/* Welcome text — centered */}
        <div style={{ position: 'relative', zIndex: 3, pointerEvents: 'none', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <h1
            style={{
              fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif",
              fontSize: 'clamp(16px, 2vw, 26px)', fontWeight: 600,
              color: 'rgba(248,244,255,0.70)', letterSpacing: '0.22em', textTransform: 'uppercase',
              margin: 0, lineHeight: 1.2, textAlign: 'center',
            }}
          >
            {fr ? 'Bienvenue chez AgenteDirect' : 'Welcome to AgenteDirect'}
          </h1>
          <p style={{
            fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif",
            fontSize: 'clamp(10px, 1.1vw, 13px)', fontWeight: 700,
            color: '#8B5CF6', letterSpacing: '0.20em', textTransform: 'uppercase',
            margin: 0, textAlign: 'center',
          }}>
            {fr ? 'Conception Web · Montréal' : 'Web Design · Montréal'}
          </p>
        </div>
      </section>

      {/* ── DESKTOP-ONLY CONTENT (hidden on mobile — overlays handle everything) ── */}
      <div className="hidden md:block">

      {/* ── COPY + CAROUSEL ── */}
      <section style={{ background: '#08041A', padding: 'clamp(56px,8vw,100px) clamp(24px,5vw,56px)' }}>
        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-10 md:gap-6 items-center max-w-7xl mx-auto">

          {/* LEFT — Copy */}
          <div className="flex flex-col" style={{ gap: 22 }}>
            <motion.p
              initial={hidden} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.72, ease }}
              style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 10.5, fontWeight: 700, letterSpacing: '0.24em', textTransform: 'uppercase', color: '#8B5CF6', margin: 0 }}
            >
              {fr ? 'Conception Web · Montréal' : 'Web Design · Montréal'}
            </motion.p>

            <motion.h2
              initial={hidden} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.18, duration: 0.85, ease }}
              style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 'clamp(30px,3.5vw,54px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.07, margin: 0 }}
            >
              {fr ? 'Vous travaillez dur pour votre entreprise.' : 'You work hard for your business.'}{' '}
              <span style={{ color: '#A78BFA' }}>{fr ? 'Votre site devrait le montrer.' : 'Your website should show it.'}</span>
            </motion.h2>

            <motion.p
              initial={hidden} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.26, duration: 0.78, ease }}
              style={{ fontSize: 13, fontWeight: 600, color: 'rgba(167,139,250,0.65)', letterSpacing: '0.01em', lineHeight: 1.6, margin: 0 }}
            >
              {fr ? 'Sans contrat à long terme · En ligne en 7 jours' : 'No long-term contract · Live in 7 days'}
            </motion.p>

            <motion.p
              initial={hidden} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.32, duration: 0.78, ease }}
              style={{ fontSize: 15.5, color: 'rgba(220,210,245,0.53)', lineHeight: 1.75, margin: 0, maxWidth: 420 }}
            >
              {fr ? 'On crée des sites web pour les entreprises de Montréal qui convertissent vraiment.' : 'We build websites for Montréal businesses that actually convert.'}
            </motion.p>

            <motion.div
              initial={hidden} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.38, duration: 0.78, ease }}
              className="flex flex-wrap items-center" style={{ gap: 10 }}
            >
              <Link
                href={fr ? '/fr/inquiry' : '/inquiry'}
                style={{ display: 'inline-flex', alignItems: 'center', padding: '13px 26px', borderRadius: 50, background: 'linear-gradient(135deg,#8B5CF6 0%,#6D28D9 100%)', color: '#fff', fontSize: 14, fontWeight: 700, textDecoration: 'none', letterSpacing: '0.01em', boxShadow: '0 4px 22px rgba(109,40,217,0.45)', transition: 'transform 0.2s,box-shadow 0.2s' }}
                onMouseEnter={e => { (e.currentTarget as HTMLAnchorElement).style.transform = 'translateY(-2px)'; (e.currentTarget as HTMLAnchorElement).style.boxShadow = '0 8px 30px rgba(109,40,217,0.60)'; }}
                onMouseLeave={e => { (e.currentTarget as HTMLAnchorElement).style.transform = 'translateY(0)'; (e.currentTarget as HTMLAnchorElement).style.boxShadow = '0 4px 22px rgba(109,40,217,0.45)'; }}
              >
                {fr ? 'Créons votre site →' : "Let's Build Your Website →"}
              </Link>
              <button
                type="button"
                onClick={() => document.getElementById('selected-work')?.scrollIntoView({ behavior: 'smooth' })}
                style={{ display: 'inline-flex', alignItems: 'center', padding: '12px 22px', borderRadius: 50, background: 'transparent', border: '1px solid rgba(139,92,246,0.30)', color: 'rgba(167,139,250,0.78)', fontSize: 14, fontWeight: 600, cursor: 'pointer', letterSpacing: '0.01em', transition: 'border-color 0.2s,color 0.2s' }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.60)'; (e.currentTarget as HTMLButtonElement).style.color = '#A78BFA'; }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.30)'; (e.currentTarget as HTMLButtonElement).style.color = 'rgba(167,139,250,0.78)'; }}
              >
                {fr ? 'Nos réalisations →' : 'Our Work →'}
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              transition={{ delay: 0.48, duration: 0.72, ease }}
              className="flex flex-wrap items-center" style={{ gap: '8px 0', paddingTop: 2 }}
            >
              {(fr ? BENEFITS_FR : BENEFITS).map((label, i) => (
                <div key={label} className="flex items-center">
                  {i > 0 && <span style={{ display: 'inline-block', width: 1, height: 12, background: 'rgba(139,92,246,0.25)', margin: '0 14px', flexShrink: 0 }} />}
                  <span style={{ fontSize: 11.5, color: 'rgba(220,210,245,0.48)', letterSpacing: '0.02em', whiteSpace: 'nowrap' }}>{label}</span>
                </div>
              ))}
            </motion.div>
          </div>

          {/* RIGHT — Carousel */}
          <motion.div
            initial={{ opacity: 0, x: 22 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.22, duration: 0.88, ease }}
            className="flex flex-col items-center" style={{ gap: 14, position: 'relative' }}
          >
            <div role="region" aria-label="Portfolio projects carousel" aria-live="polite"
              style={{ position: 'relative', width: '100%', height: 370 + TOP_PAD, overflow: 'hidden' }}
            >
              {CARDS.map((card, i) => {
                const offset = getOffset(i, active);
                const abs    = Math.abs(offset);
                const xPx    = -CARD_W / 2 + offset * SIDE_X;
                const scale  = abs === 0 ? 1 : 0.83;
                const opac   = abs === 0 ? 1 : abs === 1 ? 0.65 : 0;
                const floatAnim = abs === 0 && !reduced ? { x: xPx, scale, opacity: opac, y: [0, -10, 0] } : { x: xPx, scale, opacity: opac, y: 0 };
                const floatTrans = abs === 0 && !reduced
                  ? { x: { duration: 0.50, ease }, scale: { duration: 0.50, ease }, opacity: { duration: 0.50, ease }, y: { duration: 4.5, repeat: Infinity, ease: 'easeInOut', delay: 0.9 } }
                  : { duration: 0.50, ease };
                return (
                  <motion.div key={card.id} animate={floatAnim} transition={floatTrans}
                    onClick={() => abs === 1 && setActive(i)}
                    aria-label={abs === 0 ? `Featured: ${card.label}` : `Switch to ${card.label}`}
                    tabIndex={abs <= 1 ? 0 : -1}
                    onKeyDown={e => { if (abs === 1 && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); setActive(i); } }}
                    style={{ position: 'absolute', left: '50%', top: TOP_PAD, width: CARD_W, zIndex: abs === 0 ? 10 : 5, cursor: abs === 1 ? 'pointer' : 'default', pointerEvents: abs <= 1 ? 'auto' : 'none', transformOrigin: '50% 0%', outline: 'none' }}
                  >
                    <div style={{ borderRadius: 13, overflow: 'hidden', border: abs === 0 ? '1px solid rgba(139,92,246,0.42)' : '1px solid rgba(255,255,255,0.07)', boxShadow: abs === 0 ? '0 14px 64px rgba(80,20,200,0.58),0 2px 12px rgba(0,0,0,0.55)' : '0 6px 24px rgba(0,0,0,0.32)', background: '#130C32', transition: 'border-color 0.4s,box-shadow 0.4s' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '9px 12px', background: '#1C1242', borderBottom: '1px solid rgba(139,92,246,0.09)' }}>
                        {(['#FF5F57','#FFBD2E','#28C840'] as const).map(col => <div key={col} style={{ width: 9, height: 9, borderRadius: '50%', background: col, opacity: 0.60, flexShrink: 0 }} />)}
                        <div style={{ flex: 1, marginLeft: 8, padding: '3px 9px', borderRadius: 5, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                          <span style={{ fontSize: 10, color: 'rgba(220,210,245,0.38)', letterSpacing: '0.01em' }}>{card.url}</span>
                        </div>
                      </div>
                      <div style={{ width: '100%', aspectRatio: '16/9', overflow: 'hidden' }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={card.src} alt={`Screenshot of ${card.label}`} style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'top center', display: 'block' }} loading={abs === 0 ? 'eager' : 'lazy'} />
                      </div>
                    </div>
                    <div style={{ padding: '11px 2px 0' }}>
                      <p style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 14.5, fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.02em', margin: '0 0 3px' }}>{card.label}</p>
                      <p style={{ fontSize: 11.5, color: 'rgba(220,210,245,0.42)', margin: '0 0 10px', lineHeight: 1.45 }}>{card.desc}</p>
                      <a href={card.href} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()}
                        style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '5px 13px', borderRadius: 50, border: '1px solid rgba(139,92,246,0.35)', color: '#A78BFA', fontSize: 11, fontWeight: 600, textDecoration: 'none', letterSpacing: '0.02em', transition: 'background 0.18s,border-color 0.18s' }}
                        onMouseEnter={e => { (e.currentTarget as HTMLAnchorElement).style.background = 'rgba(139,92,246,0.10)'; (e.currentTarget as HTMLAnchorElement).style.borderColor = 'rgba(139,92,246,0.55)'; }}
                        onMouseLeave={e => { (e.currentTarget as HTMLAnchorElement).style.background = 'transparent'; (e.currentTarget as HTMLAnchorElement).style.borderColor = 'rgba(139,92,246,0.35)'; }}
                      >Visit Site ↗</a>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* Controls */}
            <div role="group" aria-label="Carousel controls" className="flex items-center justify-center" style={{ gap: 8 }}>
              {[{ fn: prev, label: 'Previous', path: 'M7.5 10L3.5 6L7.5 2' }, { fn: next, label: 'Next', path: 'M4.5 2L8.5 6L4.5 10' }].map(({ fn, label, path }, idx) => (
                idx === 0 ? (
                  <button key={label} type="button" onClick={fn} aria-label={`${label} project`}
                    style={{ width: 30, height: 30, borderRadius: '50%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.09)', color: 'rgba(220,210,245,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0, transition: 'background 0.18s,border-color 0.18s' }}
                    onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(139,92,246,0.14)'; (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.32)'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)'; (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.09)'; }}
                  ><svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden><path d={path} stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg></button>
                ) : null
              ))}
              {CARDS.map((_, i) => (
                <button key={i} type="button" onClick={() => setActive(i)} aria-label={`Project ${i + 1}: ${CARDS[i].label}`} aria-current={i === active ? 'true' : undefined}
                  style={{ width: i === active ? 20 : 7, height: 7, borderRadius: 4, background: i === active ? '#7C3AED' : 'rgba(139,92,246,0.22)', border: 'none', cursor: 'pointer', padding: 0, transition: 'all 0.28s ease', flexShrink: 0 }}
                />
              ))}
              <button type="button" onClick={next} aria-label="Next project"
                style={{ width: 30, height: 30, borderRadius: '50%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.09)', color: 'rgba(220,210,245,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0, transition: 'background 0.18s,border-color 0.18s' }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(139,92,246,0.14)'; (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(139,92,246,0.32)'; }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)'; (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.09)'; }}
              ><svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden><path d="M4.5 2L8.5 6L4.5 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg></button>
            </div>
          </motion.div>
        </div>
      </section>

      <div id="why"><WhyItMatters lang={lang} /></div>

      <WhatWeDo lang={lang} />

      <SelectedWork lang={lang} />

      <div id="process"><OurProcess lang={lang} /></div>

      {/* ── PRICING STRIP ── */}
      <section id="pricing" style={{ padding: 'clamp(64px,8vw,100px) 0', background: '#0A0520', borderTop: '1px solid rgba(139,92,246,0.08)' }}>
        <div className="max-w-7xl mx-auto px-6 md:px-14">
          <div style={{ textAlign: 'center', marginBottom: 'clamp(32px,4vw,52px)' }}>
            <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.24em', textTransform: 'uppercase' as const, color: '#8B5CF6', margin: '0 0 16px' }}>
              {fr ? 'Tarifs simples' : 'Simple Pricing'}
            </p>
            <h2 style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 'clamp(26px,3.5vw,44px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.08, margin: '0 0 14px' }}>
              {fr ? 'Transparent. Abordable. Sans surprise.' : 'Transparent. Affordable. No surprises.'}
            </h2>
            <p style={{ fontSize: 15, color: 'rgba(220,210,245,0.48)', lineHeight: 1.7, margin: 0 }}>
              {fr ? 'Pas de contrats à long terme. Annulez quand vous voulez.' : 'No long-term contracts. Cancel anytime.'}
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 16, maxWidth: 680, margin: '0 auto 40px' }}>
            {/* Starter */}
            <div style={{ background: 'rgba(139,92,246,0.05)', border: '1px solid rgba(139,92,246,0.18)', borderRadius: 12, padding: '32px 28px', display: 'flex', flexDirection: 'column' as const, gap: 16 }}>
              <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: '#8B5CF6', margin: 0 }}>{fr ? 'Départ' : 'Starter'}</p>
              <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column' as const, gap: 10 }}>
                {(fr
                  ? ['Conçu selon vos demandes', 'Pages illimitées', 'Design mobile-first', 'SEO de base', 'Hébergement inclus']
                  : ['Built to your request', 'Unlimited pages', 'Mobile-first design', 'Basic SEO', 'Hosting included']
                ).map(f => (
                  <li key={f} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: 'rgba(220,210,245,0.65)' }}>
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden><path d="M2 7l3.5 3.5L12 3.5" stroke="#8B5CF6" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    {f}
                  </li>
                ))}
              </ul>
              <Link href={fr ? '/fr/inquiry' : '/inquiry'} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '11px 24px', borderRadius: 50, border: '1px solid rgba(139,92,246,0.40)', color: '#A78BFA', fontSize: 13, fontWeight: 700, textDecoration: 'none', letterSpacing: '0.01em', marginTop: 'auto' }}>
                {fr ? 'Faire une demande →' : 'Inquire →'}
              </Link>
            </div>

            {/* Pro */}
            <div style={{ background: 'rgba(139,92,246,0.10)', border: '1px solid rgba(139,92,246,0.40)', borderRadius: 12, padding: '32px 28px', display: 'flex', flexDirection: 'column' as const, gap: 16, position: 'relative' as const }}>
              <div style={{ position: 'absolute' as const, top: -11, left: '50%', transform: 'translateX(-50%)', background: '#7C3AED', color: '#fff', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' as const, padding: '3px 12px', borderRadius: 50 }}>
                {fr ? 'Populaire' : 'Most Popular'}
              </div>
              <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: '#A78BFA', margin: 0 }}>Pro</p>
              <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column' as const, gap: 10 }}>
                {(fr
                  ? ['Tout dans Départ', 'Chatbot IA intégré', 'Bilingue EN/FR', 'Analytics avancés', 'Support prioritaire']
                  : ['Everything in Starter', 'AI chatbot integrated', 'Bilingual EN/FR', 'Advanced analytics', 'Priority support']
                ).map(f => (
                  <li key={f} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: 'rgba(220,210,245,0.65)' }}>
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden><path d="M2 7l3.5 3.5L12 3.5" stroke="#A78BFA" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    {f}
                  </li>
                ))}
              </ul>
              <Link href={fr ? '/fr/inquiry' : '/inquiry'} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '11px 24px', borderRadius: 50, background: 'linear-gradient(135deg,#8B5CF6 0%,#6D28D9 100%)', color: '#fff', fontSize: 13, fontWeight: 700, textDecoration: 'none', letterSpacing: '0.01em', boxShadow: '0 4px 16px rgba(109,40,217,0.40)', marginTop: 'auto' }}>
                {fr ? 'Faire une demande →' : 'Inquire →'}
              </Link>
            </div>
          </div>
        </div>
      </section>

      </div>{/* end hidden md:block — overlays must be outside */}

      {/* ── SECTION OVERLAYS ── */}
      <AnimatePresence>
        {activeSection && (
          <motion.div
            key="section-overlay"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
            style={{ position: 'fixed', inset: 0, zIndex: 60, background: '#08041A', overflow: 'hidden' }}
          >
            {/* Back to desktop button */}
            <motion.button
              type="button"
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.18, duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
              onClick={() => setActiveSection(null)}
              style={{
                position: 'fixed', top: 20, left: 20, zIndex: 61,
                display: 'flex', alignItems: 'center', gap: 7,
                background: 'rgba(8,4,26,0.88)',
                border: '1px solid rgba(139,92,246,0.32)',
                color: '#A78BFA', padding: '9px 18px', borderRadius: 50,
                fontSize: 13, fontWeight: 600, cursor: 'pointer',
                backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
                letterSpacing: '0.01em', boxShadow: '0 2px 16px rgba(0,0,0,0.40)',
              }}
            >
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden>
                <path d="M8.5 1L2 6.5l6.5 5.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              {fr ? 'Bureau' : 'Desktop'}
            </motion.button>

            {/* Section content — absolutely positioned; page stays locked, content scrolls if needed */}
            <div className="overlay-content" style={{ position: 'absolute', inset: 0, overflow: 'auto', paddingTop: 64 }}>
              {activeSection === 'intro' && (
                <section className="intro-copy-section" style={{ height: '100%', display: 'flex', flexDirection: 'column' as const, justifyContent: 'center', background: '#08041A', overflow: 'hidden', padding: '0 clamp(24px,5vw,56px)' }}>
                  <div className="intro-grid w-full grid grid-cols-1 md:grid-cols-2 gap-8 items-center max-w-7xl mx-auto">
                    {/* LEFT — Copy */}
                    <div className="intro-copy-col flex flex-col" style={{ gap: 16 }}>
                      <p style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 10, fontWeight: 700, letterSpacing: '0.24em', textTransform: 'uppercase' as const, color: '#8B5CF6', margin: 0 }}>
                        {fr ? 'Conception Web · Montréal' : 'Web Design · Montréal'}
                      </p>
                      <h2 style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 'clamp(24px,3vw,46px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.07, margin: 0 }}>
                        {fr ? 'Vous travaillez dur pour votre entreprise.' : 'You work hard for your business.'}{' '}
                        <span style={{ color: '#A78BFA' }}>{fr ? 'Votre site devrait le montrer.' : 'Your website should show it.'}</span>
                      </h2>
                      <p style={{ fontSize: 13, fontWeight: 600, color: 'rgba(167,139,250,0.65)', letterSpacing: '0.01em', lineHeight: 1.6, margin: 0 }}>
                        {fr ? 'Sans contrat à long terme · En ligne en 7 jours' : 'No long-term contract · Live in 7 days'}
                      </p>
                      <p className="intro-copy-desc" style={{ fontSize: 14, color: 'rgba(220,210,245,0.50)', lineHeight: 1.75, margin: 0, maxWidth: 400 }}>
                        {fr ? 'On crée des sites web pour les entreprises de Montréal qui convertissent vraiment.' : 'We build websites for Montréal businesses that actually convert.'}
                      </p>
                      <div className="flex flex-wrap items-center" style={{ gap: 10, paddingTop: 4 }}>
                        <Link href={fr ? '/fr/inquiry' : '/inquiry'} style={{ display: 'inline-flex', alignItems: 'center', padding: '12px 24px', borderRadius: 50, background: 'linear-gradient(135deg,#8B5CF6 0%,#6D28D9 100%)', color: '#fff', fontSize: 13, fontWeight: 700, textDecoration: 'none', letterSpacing: '0.01em', boxShadow: '0 4px 22px rgba(109,40,217,0.45)' }}>
                          {fr ? 'Créons votre site →' : "Let's Build Your Website →"}
                        </Link>
                      </div>
                      <div className="flex flex-wrap items-center" style={{ gap: '6px 0' }}>
                        {(fr ? BENEFITS_FR : BENEFITS).map((label, i) => (
                          <div key={label} className="flex items-center">
                            {i > 0 && <span style={{ display: 'inline-block', width: 1, height: 12, background: 'rgba(139,92,246,0.25)', margin: '0 12px', flexShrink: 0 }} />}
                            <span style={{ fontSize: 11, color: 'rgba(220,210,245,0.45)', letterSpacing: '0.02em', whiteSpace: 'nowrap' }}>{label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    {/* RIGHT — Carousel (hidden on mobile; copy alone fills the screen) */}
                    <div className="intro-carousel flex flex-col items-center" style={{ gap: 12, position: 'relative' }}>
                      <div role="region" aria-label="Portfolio projects carousel" aria-live="polite"
                        style={{ position: 'relative', width: '100%', height: 300 + TOP_PAD, overflow: 'hidden' }}
                      >
                        {CARDS.map((card, i) => {
                          const offset = getOffset(i, active);
                          const abs    = Math.abs(offset);
                          const xPx    = -CARD_W / 2 + offset * SIDE_X;
                          const scale  = abs === 0 ? 1 : 0.83;
                          const opac   = abs === 0 ? 1 : abs === 1 ? 0.65 : 0;
                          return (
                            <motion.div key={card.id} animate={{ x: xPx, scale, opacity: opac, y: 0 }} transition={{ duration: 0.50, ease }}
                              onClick={() => abs === 1 && setActive(i)}
                              style={{ position: 'absolute', left: '50%', top: TOP_PAD, width: CARD_W, zIndex: abs === 0 ? 10 : 5, cursor: abs === 1 ? 'pointer' : 'default', pointerEvents: abs <= 1 ? 'auto' : 'none', transformOrigin: '50% 0%', outline: 'none' }}
                            >
                              <div style={{ borderRadius: 13, overflow: 'hidden', border: abs === 0 ? '1px solid rgba(139,92,246,0.42)' : '1px solid rgba(255,255,255,0.07)', boxShadow: abs === 0 ? '0 14px 64px rgba(80,20,200,0.58),0 2px 12px rgba(0,0,0,0.55)' : '0 6px 24px rgba(0,0,0,0.32)', background: '#130C32' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '8px 12px', background: '#1C1242', borderBottom: '1px solid rgba(139,92,246,0.09)' }}>
                                  {(['#FF5F57','#FFBD2E','#28C840'] as const).map(col => <div key={col} style={{ width: 8, height: 8, borderRadius: '50%', background: col, opacity: 0.60 }} />)}
                                  <div style={{ flex: 1, marginLeft: 6, padding: '2px 8px', borderRadius: 4, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                                    <span style={{ fontSize: 9.5, color: 'rgba(220,210,245,0.38)' }}>{card.url}</span>
                                  </div>
                                </div>
                                <div style={{ width: '100%', aspectRatio: '16/9', overflow: 'hidden' }}>
                                  {/* eslint-disable-next-line @next/next/no-img-element */}
                                  <img src={card.src} alt={`Screenshot of ${card.label}`} style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'top center', display: 'block' }} loading="lazy" />
                                </div>
                              </div>
                              <div style={{ padding: '8px 2px 0' }}>
                                <p style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 13, fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.02em', margin: '0 0 2px' }}>{card.label}</p>
                                <p style={{ fontSize: 11, color: 'rgba(220,210,245,0.42)', margin: 0, lineHeight: 1.45 }}>{card.desc}</p>
                              </div>
                            </motion.div>
                          );
                        })}
                      </div>
                      {/* Controls */}
                      <div className="flex items-center justify-center" style={{ gap: 6 }}>
                        <button type="button" onClick={prev} aria-label="Previous project" style={{ width: 28, height: 28, borderRadius: '50%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.09)', color: 'rgba(220,210,245,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                          <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden><path d="M7.5 10L3.5 6L7.5 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                        </button>
                        {CARDS.map((_, i) => (
                          <button key={i} type="button" onClick={() => setActive(i)} aria-label={`Project ${i + 1}`} style={{ width: i === active ? 18 : 6, height: 6, borderRadius: 4, background: i === active ? '#7C3AED' : 'rgba(139,92,246,0.22)', border: 'none', cursor: 'pointer', padding: 0, transition: 'all 0.28s ease' }} />
                        ))}
                        <button type="button" onClick={next} aria-label="Next project" style={{ width: 28, height: 28, borderRadius: '50%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.09)', color: 'rgba(220,210,245,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                          <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden><path d="M4.5 2L8.5 6L4.5 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
                        </button>
                      </div>
                    </div>
                  </div>
                </section>
              )}
              {activeSection === 'selected-work' && <SelectedWork lang={lang} overlay />}
              {activeSection === 'why' && <WhyItMatters lang={lang} overlay />}
              {activeSection === 'process' && <OurProcess lang={lang} overlay />}
              {activeSection === 'signup' && (
                <iframe
                  src={fr ? '/fr/inquiry' : '/inquiry'}
                  style={{ width: '100%', height: '100%', border: 'none', display: 'block', background: '#08041A' }}
                  title={fr ? 'Inscription' : 'Sign Up'}
                />
              )}
              {activeSection === 'pricing' && (
                <section className="pricing-overlay" style={{ height: '100%', display: 'flex', flexDirection: 'column' as const, justifyContent: 'center', background: '#0A0520', overflow: 'hidden' }}>
                  <div style={{ maxWidth: 720, margin: '0 auto', padding: '0 24px', width: '100%' }}>
                    {/* Header */}
                    <div style={{ textAlign: 'center', marginBottom: 20 }}>
                      <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.24em', textTransform: 'uppercase' as const, color: '#8B5CF6', margin: '0 0 10px' }}>
                        {fr ? 'Tarifs simples' : 'Simple Pricing'}
                      </p>
                      <h2 style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 'clamp(22px,2.8vw,36px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.08, margin: '0 0 8px' }}>
                        {fr ? 'Transparent. Abordable. Sans surprise.' : 'Transparent. Affordable. No surprises.'}
                      </h2>
                      <p style={{ fontSize: 13, color: 'rgba(220,210,245,0.45)', lineHeight: 1.5, margin: 0 }}>
                        {fr ? 'Pas de contrats à long terme. Annulez quand vous voulez.' : 'No long-term contracts. Cancel anytime.'}
                      </p>
                    </div>
                    {/* Cards */}
                    <div className="pricing-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 18 }}>
                      {/* Starter */}
                      <div style={{ background: 'rgba(139,92,246,0.05)', border: '1px solid rgba(139,92,246,0.18)', borderRadius: 12, padding: '20px 22px', display: 'flex', flexDirection: 'column' as const, gap: 12 }}>
                        <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: '#8B5CF6', margin: 0 }}>{fr ? 'Départ' : 'Starter'}</p>
                        <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column' as const, gap: 7 }}>
                          {(fr ? ['Conçu selon vos demandes','Pages illimitées','Design mobile-first','SEO de base','Hébergement inclus'] : ['Built to your request','Unlimited pages','Mobile-first design','Basic SEO','Hosting included']).map(f => (
                            <li key={f} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, color: 'rgba(220,210,245,0.62)' }}>
                              <svg width="12" height="12" viewBox="0 0 14 14" fill="none" aria-hidden><path d="M2 7l3.5 3.5L12 3.5" stroke="#8B5CF6" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                              {f}
                            </li>
                          ))}
                        </ul>
                        <Link href={fr ? '/fr/inquiry' : '/inquiry'} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '9px 16px', borderRadius: 50, border: '1px solid rgba(139,92,246,0.40)', color: '#A78BFA', fontSize: 12, fontWeight: 700, textDecoration: 'none', marginTop: 'auto' }}>
                          {fr ? 'Faire une demande →' : 'Inquire →'}
                        </Link>
                      </div>
                      {/* Pro */}
                      <div style={{ background: 'rgba(139,92,246,0.10)', border: '1px solid rgba(139,92,246,0.40)', borderRadius: 12, padding: '20px 22px', display: 'flex', flexDirection: 'column' as const, gap: 12, position: 'relative' as const }}>
                        <div style={{ position: 'absolute' as const, top: -10, left: '50%', transform: 'translateX(-50%)', background: '#7C3AED', color: '#fff', fontSize: 9, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' as const, padding: '3px 10px', borderRadius: 50, whiteSpace: 'nowrap' as const }}>
                          {fr ? 'Populaire' : 'Most Popular'}
                        </div>
                        <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: '#A78BFA', margin: 0 }}>Pro</p>
                        <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column' as const, gap: 7 }}>
                          {(fr ? ['Tout dans Départ','Chatbot IA intégré','Bilingue EN/FR','Analytics avancés','Support prioritaire'] : ['Everything in Starter','AI chatbot integrated','Bilingual EN/FR','Advanced analytics','Priority support']).map(f => (
                            <li key={f} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, color: 'rgba(220,210,245,0.62)' }}>
                              <svg width="12" height="12" viewBox="0 0 14 14" fill="none" aria-hidden><path d="M2 7l3.5 3.5L12 3.5" stroke="#A78BFA" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                              {f}
                            </li>
                          ))}
                        </ul>
                        <Link href={fr ? '/fr/inquiry' : '/inquiry'} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '9px 16px', borderRadius: 50, background: 'linear-gradient(135deg,#8B5CF6 0%,#6D28D9 100%)', color: '#fff', fontSize: 12, fontWeight: 700, textDecoration: 'none', boxShadow: '0 4px 14px rgba(109,40,217,0.40)', marginTop: 'auto' }}>
                          {fr ? 'Faire une demande →' : 'Inquire →'}
                        </Link>
                      </div>
                    </div>
                  </div>
                </section>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── FOOTER ── */}
      <footer className="hidden md:block" style={{ background: '#07030F', borderTop: '1px solid rgba(139,92,246,0.07)', padding: 'clamp(48px,6vw,80px) 0 0' }}>
        <div className="max-w-7xl mx-auto px-6 md:px-14">

          {/* CTA block */}
          <div style={{ textAlign: 'center', marginBottom: 'clamp(40px,5vw,64px)', paddingBottom: 'clamp(40px,5vw,64px)', borderBottom: '1px solid rgba(139,92,246,0.08)' }}>
            <h2 style={{ fontFamily: "var(--font-space-grotesk),'Space Grotesk',sans-serif", fontSize: 'clamp(24px,3vw,40px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.1, margin: '0 0 14px' }}>
              {fr ? 'Prêt à vous démarquer ?' : 'Ready to stand out?'}
            </h2>
            <p style={{ fontSize: 15, color: 'rgba(220,210,245,0.48)', lineHeight: 1.7, margin: '0 0 28px' }}>
              {fr ? 'Votre site en ligne en 7 jours.' : 'Your website live in 7 days.'}
            </p>
            <Link
              href={fr ? '/fr/inquiry' : '/inquiry'}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '13px 30px', borderRadius: 50, background: 'linear-gradient(135deg,#8B5CF6 0%,#6D28D9 100%)', color: '#fff', fontSize: 14, fontWeight: 700, textDecoration: 'none', boxShadow: '0 4px 22px rgba(109,40,217,0.45)' }}
            >
              {fr ? 'Créons votre site →' : "Let's Build Your Website →"}
            </Link>
          </div>

          {/* Footer bottom row */}
          <div style={{ display: 'flex', flexWrap: 'wrap' as const, alignItems: 'center', justifyContent: 'space-between', gap: 16, paddingBottom: 28 }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/image.png" alt="AgenteDirect" style={{ height: 28, width: 'auto', filter: 'brightness(0) invert(1)', opacity: 0.55 }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' as const }}>
              <a href="mailto:support@agentedirect.ca" style={{ fontSize: 13, color: 'rgba(220,210,245,0.38)', textDecoration: 'none', transition: 'color 0.2s' }} onMouseEnter={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(220,210,245,0.75)'} onMouseLeave={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(220,210,245,0.38)'}>support@agentedirect.ca</a>
              <Link href={fr ? '/home' : '/fr/home'} style={{ fontSize: 13, color: 'rgba(220,210,245,0.38)', textDecoration: 'none', transition: 'color 0.2s' }} onMouseEnter={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(220,210,245,0.75)'} onMouseLeave={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(220,210,245,0.38)'}>{fr ? 'English' : 'Français'}</Link>
            </div>
            <p style={{ fontSize: 11, color: 'rgba(220,210,245,0.18)', letterSpacing: '0.04em', margin: 0 }}>
              © 2026 AgenteDirect · Montréal, Québec
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
