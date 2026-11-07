'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { motion, useReducedMotion, useInView } from 'framer-motion';

const ease = [0.16, 1, 0.3, 1] as const;

const POINTS = [
  {
    num: '01',
    heading: 'Your next client already Googled you.',
    body: 'Before anyone picks up the phone, they search. If your site doesn\'t make a strong impression in the first few seconds, they move on — to a competitor with a better-looking page.',
  },
  {
    num: '02',
    heading: 'A weak online presence is costing you real jobs.',
    body: 'Every week without a professional site is a week where leads land on your competitor\'s page instead of yours. In Montréal\'s market, you can\'t afford to be invisible.',
  },
  {
    num: '03',
    heading: 'One good site pays for itself in a single job.',
    body: 'A new client from your website covers months of service. We build sites that work around the clock — capturing leads while you focus on what you do best.',
  },
];

const POINTS_FR = [
  {
    num: '01',
    heading: 'Votre prochain client vous a déjà cherché en ligne.',
    body: 'Avant d\'appeler quiconque, il fait une recherche. Si votre site ne crée pas une forte impression en quelques secondes, il passe à un concurrent qui en a un meilleur.',
  },
  {
    num: '02',
    heading: 'Une présence web faible vous coûte des contrats.',
    body: 'Chaque semaine sans site professionnel, des clients potentiels atterrissent chez vos concurrents plutôt que chez vous. À Montréal, vous ne pouvez pas vous permettre d\'être invisible.',
  },
  {
    num: '03',
    heading: 'Un bon site se rembourse en un seul contrat.',
    body: 'Un nouveau client via votre site couvre des mois de service. On construit des sites qui travaillent 24h/24 — captant des leads pendant que vous faites ce que vous faites le mieux.',
  },
];

// ─────────────────────────────────────────────────────────────────────────────
//  BEFORE — fictional 2009-era Montréal plumbing site
// ─────────────────────────────────────────────────────────────────────────────
function BeforePanel({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';

  const navItems = fr ? [
    { label: 'ACCUEIL', bg: '#0055BB', color: '#FFFF00' },
    { label: 'SERVICES', bg: '#003A8C', color: '#AACCFF' },
    { label: 'PRIX $$$', bg: '#003A8C', color: '#AACCFF' },
    { label: 'À PROPOS', bg: '#003A8C', color: '#AACCFF' },
    { label: 'GALERIE', bg: '#003A8C', color: '#AACCFF' },
    { label: '⭐ AVIS', bg: '#006600', color: '#CCFFCC' },
    { label: 'CONTACT', bg: '#003A8C', color: '#AACCFF' },
    { label: '🚨 URGENCE 24/7', bg: '#990000', color: '#FF9999' },
    { label: 'COUPONS!!', bg: '#CC6600', color: '#FFFF00' },
    { label: 'LIENS', bg: '#550055', color: '#FFCCFF' },
    { label: 'SITEMAP', bg: '#003A8C', color: '#AACCFF' },
    { label: 'IN ENGLISH', bg: '#003A8C', color: '#AACCFF' },
  ] : [
    { label: 'HOME', bg: '#0055BB', color: '#FFFF00' },
    { label: 'SERVICES', bg: '#003A8C', color: '#AACCFF' },
    { label: 'PRICES $$$', bg: '#003A8C', color: '#AACCFF' },
    { label: 'ABOUT', bg: '#003A8C', color: '#AACCFF' },
    { label: 'GALLERY', bg: '#003A8C', color: '#AACCFF' },
    { label: '⭐ REVIEWS', bg: '#006600', color: '#CCFFCC' },
    { label: 'CONTACT', bg: '#003A8C', color: '#AACCFF' },
    { label: '🚨 24/7 EMERGENCY', bg: '#990000', color: '#FF9999' },
    { label: 'COUPONS!!', bg: '#CC6600', color: '#FFFF00' },
    { label: 'LINKS', bg: '#550055', color: '#FFCCFF' },
    { label: 'SITEMAP', bg: '#003A8C', color: '#AACCFF' },
    { label: 'EN FRANÇAIS', bg: '#003A8C', color: '#AACCFF' },
  ];

  const serviceItems = fr ? [
    { label: '>> Réparation de fuite d\'eau <<', color: '#CC0000', font: '"Comic Sans MS", cursive', size: 10 },
    { label: 'DÉBOUCHAGE + DÉBLOCAGE DE DRAIN', color: '#0000CC', font: 'Arial Black, sans-serif', size: 8.5 },
    { label: '• Remplacement tuyaux (cuivre/PVC/ABS)', color: '#006600', font: 'Times New Roman, serif', size: 9 },
    { label: '~~~ Installation chauffe-eau ~~~', color: '#990099', font: 'Courier New, monospace', size: 9 },
    { label: 'Rénovation salle de bain $$$', color: '#FF6600', font: '"Comic Sans MS", cursive', size: 10.5 },
    { label: 'PLOMBERIE D\'URGENCE 24/7 !!!!!', color: '#CC0000', font: 'Arial Black, sans-serif', size: 11 },
    { label: '> Instal./répar. pompe de puisard <', color: '#003A8C', font: 'Times New Roman, serif', size: 8 },
    { label: '* Robinets Toilettes Éviers *', color: '#996600', font: 'Courier New, monospace', size: 9.5 },
    { label: 'Drain français / Travaux d\'excavation', color: '#006633', font: 'Arial, sans-serif', size: 8.5 },
    { label: '🚽 CLAPET ANTI-RETOUR 🚽', color: '#9900CC', font: '"Comic Sans MS", cursive', size: 8 },
  ] : [
    { label: '>> Water Leak Repair <<', color: '#CC0000', font: '"Comic Sans MS", cursive', size: 10 },
    { label: 'DRAIN CLEARING + SNAKING', color: '#0000CC', font: 'Arial Black, sans-serif', size: 8.5 },
    { label: '• Pipe Replacement (copper/PVC/ABS)', color: '#006600', font: 'Times New Roman, serif', size: 9 },
    { label: '~~~ Water Heater Install ~~~', color: '#990099', font: 'Courier New, monospace', size: 9 },
    { label: 'Bathroom Renovation $$$', color: '#FF6600', font: '"Comic Sans MS", cursive', size: 10.5 },
    { label: 'EMERGENCY PLUMBING 24/7 !!!!!', color: '#CC0000', font: 'Arial Black, sans-serif', size: 11 },
    { label: '> Sump Pump Install/Repair <', color: '#003A8C', font: 'Times New Roman, serif', size: 8 },
    { label: '* Faucets Toilets Sinks *', color: '#996600', font: 'Courier New, monospace', size: 9.5 },
    { label: 'French Drain / Excavation Works', color: '#006633', font: 'Arial, sans-serif', size: 8.5 },
    { label: '🚽 BACKFLOW PREVENTION DEVICE 🚽', color: '#9900CC', font: '"Comic Sans MS", cursive', size: 8 },
  ];

  const heroBtns = fr ? [
    { label: 'SOUMISSION GRATUITE', bg: '#009900', color: '#FFFF00', border: '2px outset #00CC00', font: 'Arial Black, sans-serif' },
    { label: 'NOS SERVICES!!', bg: '#CC00CC', color: '#FFFFFF', border: '2px outset #FF00FF', font: 'Comic Sans MS, cursive' },
    { label: 'nous_contacter.html', bg: '#FF6600', color: '#003A8C', border: '2px inset #FF9933', font: 'Courier New, monospace' },
    { label: 'CLIQUEZ ICI!', bg: '#FFFF00', color: '#CC0000', border: '2px outset #CCCC00', font: 'Arial, sans-serif' },
  ] : [
    { label: 'GET FREE ESTIMATE', bg: '#009900', color: '#FFFF00', border: '2px outset #00CC00', font: 'Arial Black, sans-serif' },
    { label: 'OUR SERVICES!!', bg: '#CC00CC', color: '#FFFFFF', border: '2px outset #FF00FF', font: 'Comic Sans MS, cursive' },
    { label: 'contact_us.html', bg: '#FF6600', color: '#003A8C', border: '2px inset #FF9933', font: 'Courier New, monospace' },
    { label: 'CLICK HERE!', bg: '#FFFF00', color: '#CC0000', border: '2px outset #CCCC00', font: 'Arial, sans-serif' },
  ];

  const certItems = fr ? [
    { label: 'Licence RBQ', color: '#006600', decoration: 'none' },
    { label: 'Cote A+ BBB', color: '#003A8C', decoration: 'underline' },
    { label: 'ISO 9001:2008', color: '#CC6600', decoration: 'none' },
    { label: 'WIN$TON AWARD 2004', color: '#CC0000', decoration: 'underline' },
    { label: 'Membre WebRing', color: '#0000CC', decoration: 'underline' },
  ] : [
    { label: 'RBQ Licensed', color: '#006600', decoration: 'none' },
    { label: 'BBB A+ Rating', color: '#003A8C', decoration: 'underline' },
    { label: 'ISO 9001:2008', color: '#CC6600', decoration: 'none' },
    { label: 'WIN$TON AWARD 2004', color: '#CC0000', decoration: 'underline' },
    { label: 'Member WebRing', color: '#0000CC', decoration: 'underline' },
  ];

  const footerLinks = fr ? [
    { l: 'Accueil', c: '#FFFF00', bg: '#003A8C' },
    { l: 'Services', c: '#CCFFCC', bg: '#005500' },
    { l: 'Contact', c: '#FFCCCC', bg: '#550000' },
    { l: 'Anglais', c: '#AACCFF', bg: '#003A8C' },
    { l: 'Sitemap', c: '#FFCCFF', bg: '#550055' },
    { l: 'Webmaster', c: '#FFFF99', bg: '#444400' },
  ] : [
    { l: 'Home', c: '#FFFF00', bg: '#003A8C' },
    { l: 'Services', c: '#CCFFCC', bg: '#005500' },
    { l: 'Contact', c: '#FFCCCC', bg: '#550000' },
    { l: 'Francais', c: '#AACCFF', bg: '#003A8C' },
    { l: 'Sitemap', c: '#FFCCFF', bg: '#550055' },
    { l: 'Webmaster', c: '#FFFF99', bg: '#444400' },
  ];

  return (
    <div
      style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        background: 'repeating-linear-gradient(45deg, #FFD700 0px, #FFD700 8px, #1a1a1a 8px, #1a1a1a 16px)',
        fontFamily: '"Times New Roman", Times, serif',
        overflow: 'hidden',
      }}
    >
      {/* ── Flashing error bar ── */}
      <div style={{
        flexShrink: 0,
        background: '#CC0000',
        padding: '2px 8px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        borderBottom: '3px solid #FFFF00',
      }}>
        <span style={{ color: '#FFFF00', fontSize: 10.5, fontWeight: 'bold', fontFamily: 'Arial Black, sans-serif', letterSpacing: '0.04em' }}>
          {fr ? '*** APPELEZ MAINTENANT!!! ☎ (514) 555-1234 *** PLOMBERIE D\'URGENCE 24H ***' : '*** CALL NOW!!! ☎ (514) 555-1234 *** 24H EMERGENCY PLUMBING ***'}
        </span>
        <span style={{ fontSize: 9, color: '#fff', fontFamily: 'Courier New, monospace' }}>
          {fr ? '[JAVASCRIPT REQUIS]\u00a0|\u00a0[COOKIES OBLIGATOIRES]' : '[JAVASCRIPT REQUIRED]\u00a0|\u00a0[COOKIES MUST BE ENABLED]'}
        </span>
      </div>

      {/* ── Chaotic header — 3 misaligned rows ── */}
      <div style={{
        flexShrink: 0,
        background: 'repeating-linear-gradient(90deg, #FFFFFF 0px, #FFFFFF 60px, #EEF4FF 60px, #EEF4FF 62px)',
        borderBottom: '5px solid #FF6600',
        borderTop: '3px solid #009900',
        padding: '4px 10px',
      }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 4 }}>
          <div>
            <div style={{
              color: '#CC0000', fontSize: 22, fontWeight: 'bold',
              fontFamily: '"Comic Sans MS", cursive',
              textShadow: '2px 2px 0 #003A8C',
              lineHeight: 1,
            }}>
              {fr ? 'Tremblay Plomberie Inc.™' : 'Tremblay Plumbing Inc.™'}
            </div>
            <div style={{ color: '#009900', fontSize: 9, fontFamily: 'Arial, sans-serif', fontWeight: 'bold' }}>
              {fr ? '★★★ AU SERVICE DE MONTRÉAL DEPUIS 1998!!! ★★★ \u00a0|\u00a0 CERTIFIÉ RBQ \u00a0|\u00a0 ASSURÉ' : '★★★ SERVING MONTREAL & AREA SINCE 1998!!! ★★★ \u00a0|\u00a0 RBQ CERTIFIED \u00a0|\u00a0 INSURED'}
            </div>
            <div style={{ color: '#0000EE', fontSize: 8, fontFamily: 'Courier New, monospace', textDecoration: 'underline' }}>
              {fr ? '>> CLIQUEZ ICI POUR SOUMISSION GRATUITE!! << \u00a0 courriel: tremblay_plomberie_inc@hotmail.com' : '>> CLICK HERE FOR FREE ESTIMATE!! << \u00a0 email us: tremblay_plumbing_inc@hotmail.com'}
            </div>
            <div style={{ color: '#FF00FF', fontSize: 7.5, fontFamily: 'Arial, sans-serif', marginTop: 1 }}>
              {fr ? '⚠ JAVASCRIPT REQUIS POUR UTILISER LE SITE \u00a0|\u00a0 Dernière mise à jour: 14 août 2007' : '⚠ JAVASCRIPT IS REQUIRED FOR FULL SITE FUNCTIONALITY \u00a0|\u00a0 Last Updated: August 14, 2007'}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, flexShrink: 0, alignItems: 'flex-end' }}>
            <div style={{ background: '#FF6600', padding: '3px 8px', border: '3px outset #FF9933', textAlign: 'center' }}>
              <div style={{ color: '#FFFF00', fontSize: 11, fontWeight: 'bold', fontFamily: 'Arial Black, sans-serif' }}>{fr ? 'SOUMISSION GRATUITE!' : 'FREE ESTIMATE!'}</div>
              <div style={{ color: '#fff', fontSize: 9, fontFamily: 'Arial, sans-serif' }}>(514) 555-1234</div>
            </div>
            <div style={{ background: '#003A8C', padding: '2px 8px', border: '3px outset #0055CC', textAlign: 'center' }}>
              <div style={{ color: '#FF0000', fontSize: 8, fontWeight: 'bold', fontFamily: 'Arial, sans-serif' }}>{fr ? '★ MEILLEUR PLOMBIER 2003 ★' : '★ WINNER BEST PLUMBER 2003 ★'}</div>
            </div>
            {/* fake hit counter */}
            <div style={{ background: '#000', padding: '2px 6px', border: '1px inset #555', fontFamily: 'Courier New, monospace', fontSize: 9, color: '#00FF00', letterSpacing: '0.1em' }}>
              HITS: 0 0 4 2 1 0 8
            </div>
          </div>
        </div>
      </div>

      {/* ── Nav — way too many items, 3 different colours ── */}
      <div style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'stretch', flexWrap: 'nowrap',
        borderBottom: '3px solid #FFFF00',
      }}>
        {navItems.map(item => (
          <div key={item.label} style={{
            padding: '5px 7px', color: item.color, fontSize: 9,
            fontWeight: 'bold', fontFamily: 'Arial, sans-serif',
            borderRight: '1px solid rgba(255,255,255,0.12)',
            background: item.bg, whiteSpace: 'nowrap', cursor: 'default',
          }}>{item.label}</div>
        ))}
      </div>

      {/* ── IE + free iPod notice bar ── */}
      <div style={{
        flexShrink: 0,
        background: '#FFFFCC',
        borderBottom: '1px solid #CC9900',
        padding: '1px 8px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontSize: 7.5, color: '#333', fontFamily: 'Arial, sans-serif' }}>
          {fr ? '⚠ MIEUX VU DANS INTERNET EXPLORER 6.0 À 800×600 — DÉSACTIVEZ LES BLOQUEURS DE FENÊTRES' : '⚠ BEST VIEWED IN INTERNET EXPLORER 6.0 AT 800×600 — DISABLE POP-UP BLOCKERS FOR FULL EXPERIENCE'}
        </span>
        <span style={{ fontSize: 8, color: '#CC0000', fontWeight: 'bold', fontFamily: 'Arial Black, sans-serif', textDecoration: 'underline', cursor: 'default' }}>
          {fr ? '🎁 VOUS AVEZ PEUT-ÊTRE GAGNÉ UN iPOD GRATUIT!! CLIQUEZ ICI!!!' : '🎁 YOU MAY HAVE WON A FREE iPOD!! CLICK HERE!!!'}
        </span>
      </div>

      {/* ── Disaster hero ── */}
      <div style={{
        flexShrink: 0, height: 90, overflow: 'hidden', position: 'relative',
        background: 'linear-gradient(135deg, #8B0000 0%, #BB1100 20%, #FF3300 40%, #AA0088 60%, #8B0000 100%)',
        borderBottom: '4px double #00FF00',
        display: 'flex', alignItems: 'center', padding: '0 12px', gap: 8,
      }}>
        <div aria-hidden style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          backgroundImage: 'repeating-linear-gradient(0deg, rgba(0,0,0,0.10) 0px, rgba(0,0,0,0.10) 1px, transparent 1px, transparent 5px)',
        }} />
        <div style={{ flex: 1, position: 'relative', zIndex: 1 }}>
          {/* totally unreadable headline */}
          <div style={{
            fontFamily: '"Comic Sans MS", cursive', fontSize: 16, fontWeight: 'bold',
            color: '#00FF00', textShadow: '1px 0 #FF0000, -1px 0 #0000FF, 0 1px #000',
            lineHeight: 1.1, marginBottom: 3,
          }}>
            {fr ? 'LE #1 PLOMBIER DE MONTRÉAL - RAPIDE & PAS CHER & PRO!!!' : 'MONTREALS #1 PLUMBER - FAST & CHEAP & PROFESSIONAL!!!'}
          </div>
          <div style={{ fontSize: 8.5, marginBottom: 5, fontFamily: 'Times New Roman, serif', fontStyle: 'italic' }}>
            <span style={{ color: '#FFFF00' }}>{fr ? 'Réparations' : 'Repairs'}</span>
            <span style={{ color: '#FF99FF' }}>{fr ? ' • Installations ' : ' • Installations '}</span>
            <span style={{ color: '#00FFFF' }}>{fr ? '• Drains' : '• Drains'}</span>
            <span style={{ color: '#FF6600' }}>{fr ? ' • Urgences 24/7' : ' • Emergency 24/7'}</span>
            <span style={{ color: '#CCFFCC' }}>{fr ? ' • Commercial • Résidentiel • Industriel • Installations nucléaires' : ' • Commercial • Residential • Industrial • Nuclear Facilities'}</span>
          </div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {heroBtns.map(b => (
              <div key={b.label} style={{ background: b.bg, color: b.color, fontSize: 8.5, fontWeight: 'bold', padding: '2px 7px', border: b.border, fontFamily: b.font, whiteSpace: 'nowrap' }}>
                {b.label}
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 3, flexShrink: 0 }}>
          <div style={{ width: 55, height: 55, borderRadius: '50%', background: 'radial-gradient(#FFD700 0%, #FF6600 55%, #CC0000 100%)', border: '3px solid #00FF00', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', boxShadow: '2px 2px 0 #000' }}>
            <span style={{ color: '#fff', fontSize: 6.5, fontWeight: 'bold', fontFamily: 'Arial Black, sans-serif', lineHeight: 1.2 }}>25 YRS<br/>EXP!!!</span>
          </div>
          <div style={{ width: 52, height: 55, background: '#003A8C', border: '3px dashed #FF0000', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
            <span style={{ color: '#FFFF00', fontSize: 6, fontFamily: 'Comic Sans MS, cursive', lineHeight: 1.3 }}>🏆BEST<br/>PRICE!!!<br/>GUARANT-<br/>EED!!!</span>
          </div>
          {/* broken image */}
          <div style={{ width: 52, height: 55, background: '#CCCCCC', border: '2px inset #999', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', gap: 2 }}>
            <span style={{ fontSize: 14 }}>🖼</span>
            <span style={{ fontSize: 6, color: '#666', fontFamily: 'Arial, sans-serif' }}>IMAGE<br/>NOT FOUND</span>
          </div>
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0, position: 'relative' }}>

        {/* Left main */}
        <div style={{
          flex: '0 0 60%',
          padding: '5px 8px',
          borderRight: '5px double #FF00FF',
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          overflow: 'hidden',
          background: 'repeating-linear-gradient(180deg, #FFFACD 0px, #FFFACD 9px, #FFE4E1 9px, #FFE4E1 18px, #E0F0FF 18px, #E0F0FF 27px)',
        }}>
          {/* "Welcome" box — impossible to read */}
          <div style={{ background: '#003A8C', border: '4px outset #FF6600', flexShrink: 0, padding: '3px 6px', position: 'relative' }}>
            <div style={{
              color: '#FFFF00', fontSize: 9, fontWeight: 'bold',
              fontFamily: '"Comic Sans MS", cursive',
              textShadow: '1px 1px 0 #FF0000',
              borderBottom: '2px dotted #00FF00', paddingBottom: 2, marginBottom: 3,
            }}>
              {fr ? '>>> BIENVENUE!!! / WELCOME!!!! <<< CLIQUEZ LES LIENS CI-DESSUS POUR NAVIGUER!!!' : '>>> BIENVENUE!!! / WELCOME!!!! <<< CLICK LINKS ABOVE TO NAVIGATE SITE!!!'}
            </div>
            <p style={{ fontSize: 7.5, color: '#CCFFCC', margin: '0 0 2px', fontFamily: 'Times New Roman, serif', lineHeight: 1.3 }}>
              {fr ? 'Tremblay Plomberie Inc.' : 'Tremblay Plumbing Inc.'} <span style={{ color: '#FF9900', fontFamily: 'Courier New, monospace' }}>{fr ? 'FONDÉE PAR MICHEL TREMBLAY EN 1998' : 'FOUNDED BY MICHEL TREMBLAY IN 1998'}</span> {fr ? 'est une entreprise familiale' : 'is a family business'} <span style={{ color: '#FF00FF' }}>RBQ #1234-5678</span> {fr ? 'au service de' : 'serving'} <u>Montréal</u>, <u>Laval</u>, <u>Longueuil</u>, <u>Brossard</u>, <u>Saint-Lambert</u>, <u>Boucherville</u>, <u>Varennes</u> {fr ? 'et' : 'and'} <span style={{ color: '#FF0000', fontWeight: 'bold' }}>{fr ? 'PLUS!!!' : 'MORE!!!'}</span>
            </p>
            <p style={{ fontSize: 7, color: '#FF9999', margin: 0, fontFamily: 'Arial, sans-serif' }}>
              {fr
                ? 'NOTE: CETTE PAGE EST MIEUX VUE DANS INTERNET EXPLORER. LES UTILISATEURS FIREFOX PEUVENT AVOIR DES PROBLÈMES. SI LA PAGE NE CHARGE PAS APPUYEZ F5 POUR RAFRAÎCHIR!!!'
                : 'NOTE: THIS PAGE IS BEST VIEWED IN INTERNET EXPLORER. FIREFOX USERS MAY EXPERIENCE ISSUES. IF PAGE DOES NOT LOAD PLEASE PRESS F5 TO REFRESH!!!'} <span style={{ color: '#FFFF00' }}>{fr ? 'DÉSACTIVEZ VOTRE BLOQUEUR DE PUBS POUR MEILLEURE EXPÉRIENCE!!' : 'DISABLE YOUR AD BLOCKER FOR BEST EXPERIENCE!!'}</span>
            </p>
          </div>

          {/* Services — every item different color and font */}
          <div style={{ background: '#FFEECC', border: '3px inset #996600', flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div style={{
              background: 'repeating-linear-gradient(90deg, #CC0000 0px, #CC0000 50px, #FF6600 50px, #FF6600 100px)',
              color: '#FFFFFF', fontSize: 10, fontWeight: 'bold',
              padding: '2px 6px', fontFamily: 'Arial Black, sans-serif',
              flexShrink: 0, letterSpacing: '0.08em',
            }}>
              {fr ? '*** NOS SERVICES *** CLIQUEZ POUR PLUS D\'INFO ***' : '*** OUR SERVICES *** CLICK FOR MORE INFO ***'}
            </div>
            <div style={{ padding: '3px 6px', overflow: 'hidden', flex: 1 }}>
              {serviceItems.map(s => (
                <div key={s.label} style={{
                  fontSize: s.size, color: s.color,
                  fontFamily: s.font, fontWeight: 'bold',
                  borderBottom: '1px dashed #AAAAAA',
                  padding: '1px 0',
                  textDecoration: s.size > 10 ? 'underline' : 'none',
                }}>
                  {s.label}
                </div>
              ))}
            </div>
          </div>

          {/* UNDER CONSTRUCTION bar */}
          <div style={{
            flexShrink: 0,
            background: 'repeating-linear-gradient(90deg, #000000 0px, #000000 12px, #FFCC00 12px, #FFCC00 24px)',
            padding: '3px 8px',
            display: 'flex', alignItems: 'center', gap: 6,
            border: '2px solid #FF0000',
          }}>
            <span style={{ fontSize: 9, color: '#FFFFFF', fontWeight: 'bold', fontFamily: 'Arial Black, sans-serif', background: '#CC0000', padding: '1px 5px' }}>
              {fr ? '🚧 EN CONSTRUCTION 🚧' : '🚧 UNDER CONSTRUCTION 🚧'}
            </span>
            <span style={{ fontSize: 7.5, color: '#FFFF00', fontFamily: 'Courier New, monospace', fontWeight: 'bold' }}>
              {fr ? 'PAGE GALERIE BIENTÔT DISPONIBLE!! \u00a0|\u00a0 PAGE DE PRIX EN CONSTRUCTION \u00a0|\u00a0 REVENEZ BIENTÔT!!!' : 'GALLERY PAGE COMING SOON!! \u00a0|\u00a0 NEW PRICES PAGE UNDER CONSTRUCTION \u00a0|\u00a0 CHECK BACK SOON!!!'}
            </span>
          </div>
        </div>

        {/* Right sidebar */}
        <div style={{
          flex: '0 0 40%',
          padding: '5px 7px',
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          overflow: 'hidden',
          background: 'repeating-linear-gradient(135deg, #EEFFEE 0px, #EEFFEE 10px, #FFEEFF 10px, #FFEEFF 20px)',
        }}>
          {/* Emergency — even more garish */}
          <div style={{
            background: 'repeating-linear-gradient(45deg, #FF0000 0px, #FF0000 4px, #CC0000 4px, #CC0000 8px)',
            border: '4px outset #FF9900',
            padding: '5px',
            textAlign: 'center',
            flexShrink: 0,
          }}>
            <div style={{ color: '#FFFF00', fontSize: 13, fontWeight: 'bold', fontFamily: 'Arial Black, sans-serif', textShadow: '2px 2px 0 #000', lineHeight: 1 }}>
              {fr ? '⚠ URGENCE??? ⚠' : '⚠ EMERGENCY??? ⚠'}
            </div>
            <div style={{
              color: '#FFFFFF', fontSize: 17, fontWeight: 'bold',
              fontFamily: '"Comic Sans MS", cursive', margin: '2px 0',
              textShadow: '1px 1px 0 #000, -1px -1px 0 #FF6600',
            }}>
              (514) 555-1234
            </div>
            <div style={{ color: '#FFFF00', fontSize: 7.5, fontFamily: 'Courier New, monospace', marginBottom: 3 }}>
              {fr ? 'DISPONIBLE 24H / 7 JOURS / 365 JOURS!!' : 'AVAILABLE 24H / 7 DAYS / 365 DAYS!!'}
            </div>
            <div style={{
              background: '#FFFF00', color: '#CC0000', fontSize: 10.5,
              fontWeight: 'bold', padding: '3px', border: '3px outset #FFCC00',
              fontFamily: 'Arial Black, sans-serif', letterSpacing: '0.05em',
            }}>
              {fr ? 'APPELEZ MAINTENANT!!! SOUMISSION GRATUITE!!!' : 'CALL NOW!!! FREE QUOTE!!!'}
            </div>
          </div>

          {/* "Quote form" — completely broken layout */}
          <div style={{ background: '#CCDDFF', border: '3px inset #003A8C', padding: '5px', flexShrink: 0 }}>
            <div style={{
              color: '#FF0000', fontSize: 11, fontWeight: 'bold', fontFamily: '"Comic Sans MS", cursive',
              borderBottom: '3px dotted #009900', paddingBottom: 2, marginBottom: 4,
            }}>
              {fr ? 'Soumission GRATUITE!!! Remplissez le formulaire!!' : 'Get FREE Estimate!!! Fill Form Below!!'}
            </div>
            {[
              { label: 'Nom / Name:', bg: '#FFFFCC', color: '#000080' },
              { label: 'Tel / Phone:', bg: '#FFCCFF', color: '#006600' },
              { label: 'Courriel / Email:', bg: '#CCFFCC', color: '#CC0000' },
            ].map(f => (
              <div key={f.label} style={{ marginBottom: 3, display: 'flex', alignItems: 'center', gap: 3 }}>
                <div style={{ fontSize: 8, color: f.color, fontWeight: 'bold', fontFamily: 'Times New Roman, serif', flexShrink: 0, width: 80 }}>
                  {f.label}
                </div>
                <div style={{
                  background: f.bg, border: '2px inset #888',
                  padding: '1px 4px', fontSize: 8.5, color: '#AAAAAA',
                  fontFamily: 'Courier New, monospace', flex: 1,
                }}>
                  ___________
                </div>
              </div>
            ))}
            <div style={{
              background: '#009900', color: '#FFFF00', textAlign: 'center',
              fontSize: 9.5, fontWeight: 'bold', padding: '3px',
              border: '3px outset #00CC00', cursor: 'default',
              fontFamily: 'Arial Black, sans-serif', marginTop: 3,
            }}>
              &gt;&gt; ENVOYER / SEND &lt;&lt;
            </div>
          </div>

          {/* Certifications — with bizarre extras */}
          <div style={{ background: '#FFFFF0', border: '2px solid #AAAAAA', padding: '4px', flexShrink: 0 }}>
            <div style={{
              color: '#FFFFFF', fontSize: 9, fontWeight: 'bold', background: '#550055',
              padding: '1px 5px', marginBottom: 3,
              fontFamily: 'Arial Black, sans-serif',
            }}>
              {fr ? 'CERTIFIÉ & ACCRÉDITÉ!!!' : 'CERTIFIED & ACCREDITED!!!'}
            </div>
            {certItems.map(c => (
              <div key={c.label} style={{
                fontSize: 8.5, color: c.color, padding: '1px 0',
                display: 'flex', gap: 3, alignItems: 'flex-start',
                fontFamily: c.label.includes('Web') ? 'Courier New, monospace' : 'Arial, sans-serif',
                textDecoration: c.decoration,
              }}>
                <span style={{ color: '#FF6600', fontWeight: 'bold', flexShrink: 0, fontSize: 9 }}>►</span>
                {c.label}
              </div>
            ))}
          </div>

          {/* Visitor counter */}
          <div style={{
            flexShrink: 0, background: '#000011', border: '2px inset #333366',
            padding: '3px 6px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1,
          }}>
            <div style={{ fontSize: 7, color: '#6666FF', fontFamily: 'Courier New, monospace', letterSpacing: '0.08em' }}>
              {fr ? 'VOUS ÊTES LE VISITEUR #:' : 'YOU ARE VISITOR #:'}
            </div>
            <div style={{ display: 'flex', gap: 1 }}>
              {['0','0','4','2','1','0','8'].map((d, i) => (
                <div key={i} style={{
                  background: '#000', border: '1px solid #444',
                  color: '#00FF00', fontSize: 10, fontFamily: 'Courier New, monospace',
                  fontWeight: 'bold', width: 12, textAlign: 'center', lineHeight: '14px',
                }}>
                  {d}
                </div>
              ))}
            </div>
            <div style={{ fontSize: 6.5, color: '#555588', fontFamily: 'Courier New, monospace' }}>
              since Jan 1 2001
            </div>
          </div>
        </div>

        {/* Floating popup-ad over content */}
        <div style={{
          position: 'absolute', top: 8, left: '18%',
          background: '#FFFF99', border: '4px solid #FF0000',
          padding: '5px 8px', zIndex: 99,
          boxShadow: '3px 3px 0 #000000',
          maxWidth: 130,
        }}>
          <div style={{ fontSize: 8.5, fontWeight: 'bold', color: '#CC0000', fontFamily: 'Arial Black, sans-serif', marginBottom: 2, lineHeight: 1.2 }}>
            {fr ? '*** OFFRE SPÉCIALE ***' : '*** SPECIAL DEAL ***'}
          </div>
          <div style={{ fontSize: 7.5, color: '#000', fontFamily: 'Times New Roman, serif', marginBottom: 3, lineHeight: 1.3 }}>
            {fr
              ? <>10% DE RABAIS sur toutes réparations!! Cette semaine seulement!!! Appelez <strong>(514) 555-1234</strong> et mentionnez le code: <span style={{ color: '#CC0000' }}>INTERNET2007</span></>
              : <>10% OFF all repairs!! This week only!!! Call <strong>(514) 555-1234</strong> and mention code: <span style={{ color: '#CC0000' }}>INTERNET2007</span></>}
          </div>
          <div style={{ background: '#FF6600', color: '#fff', fontSize: 8, fontWeight: 'bold', textAlign: 'center', padding: '2px', border: '1px outset #FF9900', fontFamily: 'Arial, sans-serif' }}>
            {fr ? 'PROFITEZ DE L\'OFFRE!!' : 'CLAIM OFFER!!'}
          </div>
          <div style={{ fontSize: 6.5, color: '#999', marginTop: 2, fontFamily: 'Arial, sans-serif', textAlign: 'right', cursor: 'default' }}>
            [X] close
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <div style={{
        flexShrink: 0,
        background: 'repeating-linear-gradient(90deg, #000033 0px, #000033 40px, #001a4d 40px, #001a4d 42px)',
        padding: '3px 8px',
        borderTop: '4px solid #FF6600',
        borderBottom: '3px solid #00FF00',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'nowrap', gap: 4 }}>
          <span style={{ color: '#AABBFF', fontSize: 7.5, fontFamily: 'Courier New, monospace', flexShrink: 0 }}>
            {fr ? '© 2007 Tremblay Plomberie Inc. | Dernière mise à jour: août 2008' : '© 2007 Tremblay Plumbing Inc. | Last updated: August 2008'}
          </span>
          <div style={{ display: 'flex', gap: 0, flexShrink: 0 }}>
            {footerLinks.map(l => (
              <span key={l.l} style={{
                color: l.c, fontSize: 7.5, background: l.bg,
                padding: '1px 5px', cursor: 'default',
                fontFamily: 'Arial, sans-serif', borderRight: '1px solid rgba(255,255,255,0.15)',
                textDecoration: 'underline',
              }}>
                {l.l}
              </span>
            ))}
          </div>
          <span style={{ fontSize: 7, color: '#FF6666', fontFamily: '"Comic Sans MS", cursive', flexShrink: 0 }}>
            {fr ? '🌐 Optimal avec IE6 @ 800×600' : '🌐 Best viewed IE6 @ 800×600'}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  AFTER — premium redesign in AgenteDirect's design language
// ─────────────────────────────────────────────────────────────────────────────
function AfterPanel({ lang = 'en', isMobile = false }: { lang?: 'en' | 'fr'; isMobile?: boolean }) {
  const fr = lang === 'fr';
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column',
      background: '#001430',
      fontFamily: 'Arial, "Helvetica Neue", Helvetica, sans-serif',
      overflow: 'hidden',
      isolation: 'isolate',
    }}>
      {/* ── Nav ── */}
      <div style={{
        flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 22px',
        background: 'rgba(0,12,50,0.92)',
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
        borderBottom: '1px solid rgba(255,255,255,0.10)',
      }}>
        <span style={{
          fontFamily: 'var(--font-space-grotesk, sans-serif)',
          fontWeight: 700,
          fontSize: 15,
          letterSpacing: '0.02em',
          color: '#ffffff',
          opacity: 0.92,
        }}>
          {fr ? 'Tremblay Plomberie' : 'Tremblay Plumbing'}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          {(fr ? ['Services', 'Nos réalisations', 'À propos', 'Contact'] : ['Services', 'Our Work', 'About', 'Contact']).map(item => (
            <span key={item} style={{ color: 'rgba(210,228,255,0.80)', fontSize: 12, fontWeight: 500, whiteSpace: 'nowrap' }}>
              {item}
            </span>
          ))}
          <div style={{
            background: '#FF6600',
            color: '#fff', fontSize: 11.5, fontWeight: 700,
            padding: '7px 15px', borderRadius: 3,
            whiteSpace: 'nowrap',
          }}>
            {fr ? '☎ Urgence 24/7' : '☎ Emergency 24/7'}
          </div>
        </div>
      </div>

      {/* ── Hero — full-bleed video with centered content ── */}
      <div style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'hidden', background: 'linear-gradient(135deg, #001a3a 0%, #00102a 40%, #001f4a 70%, #000d24 100%)' }}>
        {/* Full-bleed background — video on desktop, static image on mobile */}
        {isMobile ? (
          <div aria-hidden style={{
            position: 'absolute', inset: 0,
            backgroundImage: 'url(/image%20copy.png)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }} />
        ) : (
          <video
            autoPlay muted loop playsInline aria-hidden
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              objectFit: 'cover',
            }}
          >
            <source src="https://assets.mixkit.co/videos/49024/49024-720.mp4" type="video/mp4" />
          </video>
        )}
        {/* Dark overlay so text is readable */}
        <div aria-hidden style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(to bottom, rgba(0,15,50,0.72) 0%, rgba(0,10,40,0.60) 50%, rgba(0,5,30,0.80) 100%)',
          pointerEvents: 'none',
        }} />

        {/* Centered hero content */}
        <div style={{
          position: 'relative', zIndex: 2,
          height: '100%',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          textAlign: 'center',
          padding: '16px 28px',
          gap: 10,
        }}>
          <div style={{
            fontSize: 9.5, fontWeight: 700, color: '#FF6600',
            letterSpacing: '0.22em', textTransform: 'uppercase',
          }}>
            {fr ? 'Plomberie résidentielle & commerciale · Montréal' : 'Residential & Commercial Plumbing · Montreal'}
          </div>

          <h1 style={{
            fontSize: 'clamp(19px, 2.4vw, 30px)',
            fontWeight: 900, color: '#FFFFFF',
            letterSpacing: '-0.03em', lineHeight: 1.08, margin: 0,
          }}>
            {fr ? <>Un plombier de confiance,{' '}<span style={{ color: '#FF9933' }}>quand vous en avez besoin.</span></> : <>A plumber you can trust,{' '}<span style={{ color: '#FF9933' }}>when you need one.</span></>}
          </h1>

          <p style={{
            fontSize: 12, color: 'rgba(220,235,255,0.80)',
            lineHeight: 1.68, margin: 0, maxWidth: 440,
          }}>
            {fr ? 'Expertise en plomberie à travers Montréal. Intervention d\'urgence rapide, disponible jour et nuit.' : 'Trusted plumbing expertise across Montreal. Fast emergency response, available day and night.'}
          </p>

          {/* CTAs */}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 2 }}>
            <div style={{
              background: '#FF6600',
              color: '#fff', fontSize: 12, fontWeight: 700,
              padding: '9px 20px', borderRadius: 4,
              whiteSpace: 'nowrap',
              boxShadow: '0 4px 18px rgba(255,102,0,0.45)',
            }}>
              {fr ? 'Obtenez une soumission gratuite →' : 'Get a free estimate →'}
            </div>
            <div style={{
              border: '2px solid rgba(255,255,255,0.55)',
              color: '#FFFFFF', fontSize: 12, fontWeight: 700,
              padding: '7px 16px', borderRadius: 4,
              whiteSpace: 'nowrap',
            }}>
              {fr ? 'Nos services' : 'Our services'}
            </div>
          </div>

          {/* Trust row — generic credentials only, no invented numbers */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginTop: 2 }}>
            {(fr ? ['✓ Certifié RBQ', '✓ Pleinement assuré', '✓ Service d\'urgence', '✓ Soumissions gratuites'] : ['✓ RBQ Certified', '✓ Fully insured', '✓ Emergency service', '✓ Free estimates']).map((t, i) => (
              <div key={t} style={{ display: 'flex', alignItems: 'center' }}>
                {i > 0 && <div style={{ width: 1, height: 10, background: 'rgba(255,255,255,0.25)', margin: '0 10px' }} />}
                <span style={{ fontSize: 10, color: 'rgba(220,235,255,0.75)', whiteSpace: 'nowrap' }}>{t}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Services strip ── */}
      <div style={{
        flexShrink: 0,
        padding: '9px 22px',
        background: 'rgba(0,10,40,0.88)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        borderTop: '1px solid rgba(255,255,255,0.08)',
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 7 }}>
          {(fr ? [
            { icon: '⚡', name: 'Urgences', sub: '24h / 7 jours', accent: '#CC0000' },
            { icon: '🔧', name: 'Réparations', sub: 'Fuites, robinets', accent: '#003A8C' },
            { icon: '🏠', name: 'Rénovations', sub: 'Salle de bain, cuisine', accent: '#003A8C' },
            { icon: '🔍', name: 'Inspections', sub: 'Caméra, diagnostics', accent: '#003A8C' },
          ] : [
            { icon: '⚡', name: 'Emergencies', sub: '24h / 7 days', accent: '#CC0000' },
            { icon: '🔧', name: 'Repairs', sub: 'Leaks, faucets', accent: '#003A8C' },
            { icon: '🏠', name: 'Renovations', sub: 'Bath, kitchen', accent: '#003A8C' },
            { icon: '🔍', name: 'Inspections', sub: 'Camera, diagnostics', accent: '#003A8C' },
          ]).map(s => (
            <div key={s.name} style={{
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderTop: `2px solid ${s.accent}`,
              borderRadius: 5,
              padding: '7px 9px',
            }}>
              <div style={{ fontSize: 14, marginBottom: 3 }}>{s.icon}</div>
              <div style={{ fontSize: 11, fontWeight: 700, color: 'rgba(220,235,255,0.90)', marginBottom: 1 }}>{s.name}</div>
              <div style={{ fontSize: 9, color: 'rgba(180,210,255,0.55)' }}>{s.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Footer ── */}
      <div style={{
        flexShrink: 0,
        background: 'rgba(0,8,32,0.92)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        borderTop: '1px solid rgba(255,255,255,0.07)',
        padding: '6px 22px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ color: 'rgba(255,255,255,0.60)', fontSize: 9.5 }}>
          {fr ? '© 2024 Tremblay Plomberie Inc. · Montréal, Québec' : '© 2024 Tremblay Plumbing Inc. · Montreal, Quebec'}
        </span>
        <div style={{ display: 'flex', gap: 14 }}>
          {(fr ? ['Confidentialité', 'Conditions', 'Plan du site'] : ['Privacy', 'Terms', 'Sitemap']).map(l => (
            <span key={l} style={{ color: 'rgba(255,255,255,0.40)', fontSize: 9, cursor: 'default' }}>{l}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  COMPARISON SLIDER
//  Before (old) is on the LEFT. After (modern) is on the RIGHT.
//  AfterPanel is the full-width base. BeforePanel is clipped on top,
//  revealing the left portion. Slider handle sits at the boundary.
// ─────────────────────────────────────────────────────────────────────────────
function ComparisonSlider({ reduced, lang = 'en' }: { reduced: boolean; lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';
  // 0 = fully left (only After visible), 100 = fully right (only Before visible)
  // Default 50 = even split
  const [pct, setPct] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [interacted, setInteracted] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 640);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  const updateFromX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const { left, width } = el.getBoundingClientRect();
    // Full 0–100 range so user can see either side completely
    setPct(Math.max(0, Math.min(100, ((clientX - left) / width) * 100)));
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
    setInteracted(true);
    updateFromX(e.clientX);
  }, [updateFromX]);

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    setDragging(true);
    setInteracted(true);
    updateFromX(e.touches[0].clientX);
  }, [updateFromX]);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => updateFromX(e.clientX);
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      updateFromX(e.touches[0].clientX);
    };
    const onStop = () => setDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onStop);
    window.addEventListener('touchmove', onTouchMove, { passive: false });
    window.addEventListener('touchend', onStop);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onStop);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onStop);
    };
  }, [dragging, updateFromX]);

  const onHandleKeyDown = useCallback((e: React.KeyboardEvent) => {
    setInteracted(true);
    if (e.key === 'ArrowLeft')  setPct(p => Math.max(0, p - 3));
    if (e.key === 'ArrowRight') setPct(p => Math.min(100, p + 3));
  }, []);

  return (
    <div
      ref={containerRef}
      onMouseDown={onMouseDown}
      onTouchStart={onTouchStart}
      role="group"
      aria-label="Website comparison — drag to reveal before and after"
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: isMobile ? '3 / 2' : '16 / 9',
        borderRadius: isMobile ? 10 : 16,
        overflow: 'hidden',
        cursor: dragging ? 'col-resize' : 'ew-resize',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        border: '1px solid rgba(139,92,246,0.22)',
        boxShadow: '0 32px 80px rgba(0,0,0,0.70), 0 8px 24px rgba(80,20,200,0.22)',
      }}
    >
      {/* ── AFTER panel: full-width base (right side) ── */}
      <div style={isMobile ? { position: 'absolute', inset: 0, transformOrigin: 'top left', zoom: 0.62 } : {}}>
        <AfterPanel lang={lang} isMobile={isMobile} />
      </div>

      {/* ── BEFORE panel: clipped to show on left (pct%) ── */}
      <div style={{
        position: 'absolute',
        inset: 0,
        // Clips right by (100-pct)%, leaving left pct% of BeforePanel visible
        clipPath: `inset(0 ${(100 - pct).toFixed(2)}% 0 0)`,
        ...(dragging ? {} : { transition: 'clip-path 0.04s linear' }),
      }}>
        <div style={isMobile ? { position: 'absolute', inset: 0, zoom: 0.62 } : {}}>
          <BeforePanel lang={lang} />
        </div>
      </div>

      {/* ── Divider line at boundary ── */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: 0, bottom: 0,
          left: `${pct}%`,
          width: 1,
          transform: 'translateX(-50%)',
          background: 'rgba(255,255,255,0.70)',
          boxShadow: '0 0 10px rgba(255,255,255,0.30)',
          pointerEvents: 'none',
          ...(dragging ? {} : { transition: 'left 0.04s linear' }),
        }}
      />

      {/* ── Drag handle ── */}
      <div
        role="slider"
        aria-label={fr ? 'Glisser à gauche et à droite pour comparer les versions' : 'Drag left and right to compare website versions'}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct)}
        tabIndex={0}
        onKeyDown={onHandleKeyDown}
        style={{
          position: 'absolute',
          top: '50%',
          left: `${pct}%`,
          transform: 'translate(-50%, -50%)',
          width: isMobile ? 46 : 38,
          height: isMobile ? 46 : 38,
          borderRadius: '50%',
          background: 'rgba(8,4,26,0.72)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          border: '1.5px solid rgba(255,255,255,0.70)',
          boxShadow: '0 2px 18px rgba(0,0,0,0.55)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'col-resize',
          zIndex: 20,
          outline: 'none',
          ...(dragging ? {} : { transition: 'left 0.04s linear' }),
        }}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
          <path d="M6 6L3 9L6 12" stroke="rgba(255,255,255,0.90)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M12 6L15 9L12 12" stroke="rgba(255,255,255,0.90)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>

      {/* ── BEFORE label (bottom-left) ── */}
      <div
        aria-label={fr ? 'Avant : ancien site' : 'Before: old site'}
        style={{
          position: 'absolute', bottom: 12, left: 12,
          pointerEvents: 'none',
          opacity: pct > 6 ? 1 : 0,
          transition: 'opacity 0.25s',
          zIndex: 15,
        }}
      >
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 10px 4px 8px',
          borderRadius: 20,
          background: 'rgba(10,0,0,0.78)',
          backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,80,80,0.40)',
          boxShadow: '0 2px 12px rgba(0,0,0,0.45)',
        }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#FF5555', flexShrink: 0 }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.90)', letterSpacing: '0.02em', whiteSpace: 'nowrap' }}>
            {fr ? 'Avant' : 'Before'}
          </span>
        </div>
      </div>

      {/* ── AFTER label (bottom-right) ── */}
      <div
        aria-label={fr ? 'Après : nouveau site' : 'After: new site'}
        style={{
          position: 'absolute', bottom: 12, right: 12,
          pointerEvents: 'none',
          opacity: pct < 94 ? 1 : 0,
          transition: 'opacity 0.25s',
          zIndex: 15,
        }}
      >
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 10px 4px 8px',
          borderRadius: 20,
          background: 'rgba(0,12,30,0.78)',
          backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid rgba(0,200,100,0.40)',
          boxShadow: '0 2px 12px rgba(0,0,0,0.45)',
        }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#00CC66', flexShrink: 0 }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.90)', letterSpacing: '0.02em', whiteSpace: 'nowrap' }}>
            {fr ? 'Après' : 'After'}
          </span>
        </div>
      </div>

      {/* ── First-use hint (appears below handle, fades after first drag) ── */}
      {!interacted && !reduced && (
        <div
          aria-hidden
          style={{
            position: 'absolute',
            top: '50%', left: '50%',
            transform: 'translate(-50%, calc(-50% + 30px))',
            display: 'flex', alignItems: 'center', gap: 5,
            fontSize: 9, fontWeight: 600,
            color: 'rgba(255,255,255,0.50)',
            letterSpacing: '0.16em', textTransform: 'uppercase',
            pointerEvents: 'none', whiteSpace: 'nowrap',
          }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 5H8M2 5L4 3M2 5L4 7" stroke="rgba(255,255,255,0.50)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          {fr ? 'Glisser pour comparer' : 'Drag to compare'}
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M8 5H2M8 5L6 3M8 5L6 7" stroke="rgba(255,255,255,0.50)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION
// ─────────────────────────────────────────────────────────────────────────────
export default function WhyItMatters({ lang = 'en', overlay = false }: { lang?: 'en' | 'fr'; overlay?: boolean } = {}) {
  const fr = lang === 'fr';
  const points = fr ? POINTS_FR : POINTS;
  const reduced = useReducedMotion();
  const sectionRef = useRef<HTMLElement>(null);
  const inViewRaw = useInView(sectionRef, { once: true, margin: '-80px 0px' });
  const inView = overlay || inViewRaw;

  const vis = (delay: number) => ({
    initial: { opacity: 0, y: reduced ? 0 : 22 },
    animate: inView ? { opacity: 1, y: 0 } : {},
    transition: { duration: 0.72, delay, ease },
  });

  return (
    <>
      <style>{`
        @media (max-width: 767px) {
          .wim-point {
            border-right: none !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
          }
          .wim-point + .wim-point {
            border-top: 1px solid rgba(139,92,246,0.12);
            padding-top: 28px !important;
          }
          .wim-num-bg { display: none; }
        }
      `}</style>

      <section
        ref={sectionRef}
        aria-label="Why your website matters"
        style={{
          position: 'relative',
          background: '#08041A',
          paddingTop: overlay ? 'clamp(32px, 5vw, 40px)' : 'clamp(80px, 10vw, 128px)',
          paddingBottom: overlay ? 'clamp(40px, 5vw, 48px)' : 'clamp(80px, 10vw, 120px)',
          overflow: 'hidden',
          ...(overlay ? { height: '100%', display: 'flex', flexDirection: 'column' as const, justifyContent: 'center' } : {}),
        }}
      >
        {/* Atmosphere */}
        <div aria-hidden style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse 75% 55% at 50% 20%, rgba(109,40,217,0.07) 0%, transparent 70%)',
        }} />

        {/* Top hairline */}
        <div aria-hidden style={{
          position: 'absolute', top: 0, left: '6%', right: '6%',
          height: 1,
          background: 'linear-gradient(90deg, transparent 0%, rgba(139,92,246,0.22) 25%, rgba(139,92,246,0.22) 75%, transparent 100%)',
        }} />

        <div className="max-w-7xl mx-auto px-6 md:px-14">

          {/* ══ PART 1 ══════════════════════════════════════════════════════ */}
          <div style={{ marginBottom: overlay ? 0 : 'clamp(72px, 10vw, 110px)' }}>

            <motion.p {...vis(0)} style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 10, fontWeight: 700, letterSpacing: '0.24em',
              textTransform: 'uppercase', color: '#8B5CF6', margin: overlay ? '0 0 10px' : '0 0 22px',
            }}>
              {fr ? 'L\'opportunité' : 'The Opportunity'}
            </motion.p>

            <motion.h2 {...vis(0.08)} style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 'clamp(28px, 4vw, 56px)',
              fontWeight: 800, color: '#F5F0FC',
              letterSpacing: '-0.04em', lineHeight: 1.07,
              margin: overlay ? '0 0 10px' : '0 0 18px', maxWidth: '16ch',
            }}>
              {fr ? 'Votre entreprise a grandi.' : 'Your business has grown.'}{' '}
              <span style={{ color: 'rgba(220,210,245,0.30)' }}>{fr ? 'Et votre site web\u00a0?' : 'Has your website?'}</span>
            </motion.h2>

            <motion.p {...vis(0.16)} style={{
              fontSize: 'clamp(14px, 1.3vw, 16.5px)',
              color: 'rgba(220,210,245,0.50)',
              lineHeight: 1.78, maxWidth: 500, margin: overlay ? '0 0 24px' : '0 0 68px',
            }}>
              {fr
                ? 'Vous avez travaillé dur pour bâtir quelque chose d\'excellent. Votre site devrait refléter la qualité, le professionnalisme et le soin que vous mettez dans tout ce que vous faites.'
                : <>You&rsquo;ve put real work into building something great. Your website should reflect the quality, professionalism, and care that goes into everything you do.</>
              }
            </motion.p>

            <div className="grid grid-cols-1 md:grid-cols-3">
              {points.map((pt, i) => (
                <motion.div
                  key={pt.num}
                  {...vis(0.22 + i * 0.10)}
                  className="wim-point"
                  style={{
                    position: 'relative',
                    paddingBottom: overlay ? 0 : 32,
                    paddingLeft: i > 0 ? (overlay ? 24 : 44) : 0,
                    paddingRight: i < 2 ? (overlay ? 24 : 44) : 0,
                    borderRight: i < 2 ? '1px solid rgba(139,92,246,0.11)' : 'none',
                  }}
                >
                  <div aria-hidden className="wim-num-bg" style={{
                    position: 'absolute', top: -8, left: i > 0 ? 32 : -6,
                    fontSize: 88, fontWeight: 900,
                    color: 'rgba(109,40,217,0.065)',
                    fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                    letterSpacing: '-0.06em', lineHeight: 1,
                    userSelect: 'none', pointerEvents: 'none',
                  }}>
                    {pt.num}
                  </div>
                  <div style={{
                    width: 26, height: 2,
                    background: 'linear-gradient(90deg, #8B5CF6, rgba(139,92,246,0.15))',
                    borderRadius: 1, marginBottom: overlay ? 10 : 18,
                  }} />
                  <p style={{
                    fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
                    fontSize: 'clamp(15px, 1.35vw, 19px)', fontWeight: 700,
                    color: '#F5F0FC', letterSpacing: '-0.025em',
                    lineHeight: 1.3, margin: overlay ? '0 0 6px' : '0 0 11px',
                  }}>
                    {pt.heading}
                  </p>
                  <p style={{
                    fontSize: 'clamp(13px, 1vw, 14.5px)',
                    color: 'rgba(220,210,245,0.44)',
                    lineHeight: 1.72, margin: 0,
                  }}>
                    {pt.body}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>

          {/* ══ PART 2 ══════════════════════════════════════════════════════ */}
          {!overlay && <div>
            <motion.p {...vis(0.52)} style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 10, fontWeight: 700, letterSpacing: '0.24em',
              textTransform: 'uppercase', color: '#8B5CF6', margin: '0 0 14px',
            }}>
              {fr ? 'Voyez la différence' : 'See the Difference'}
            </motion.p>

            <motion.h3 {...vis(0.60)} style={{
              fontFamily: "var(--font-space-grotesk), 'Space Grotesk', sans-serif",
              fontSize: 'clamp(20px, 2.4vw, 34px)',
              fontWeight: 800, color: '#F5F0FC',
              letterSpacing: '-0.035em', lineHeight: 1.15,
              margin: '0 0 10px', maxWidth: '24ch',
            }}>
              {fr ? 'Du démodé au distinctif.' : 'From dated to distinctive.'}{' '}
              <span style={{ color: 'rgba(220,210,245,0.28)' }}>{fr ? 'Glissez pour voir.' : 'Drag to see it live.'}</span>
            </motion.h3>

            <motion.p {...vis(0.66)} style={{
              fontSize: 14, color: 'rgba(220,210,245,0.36)',
              margin: '0 0 28px', lineHeight: 1.65,
            }}>
              {fr ? 'Même entreprise fictive. Glissez à fond à gauche ou à droite pour voir chaque version en entier.' : 'Same fictional business. Drag all the way left or right to see either version in full.'}
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: reduced ? 0 : 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.88, delay: 0.70, ease }}
            >
              <ComparisonSlider reduced={reduced ?? false} lang={lang} />
            </motion.div>

            <motion.p {...vis(0.90)} style={{
              fontSize: 11, color: 'rgba(220,210,245,0.42)',
              marginTop: 14, letterSpacing: '0.01em', lineHeight: 1.6,
            }}>
              {fr
                ? '* Design conceptuel uniquement. « Tremblay Plomberie » est une entreprise fictive créée à des fins d\'illustration. Aucun client réel, donnée de performance ou résultat commercial n\'est impliqué ou garanti.'
                : <>* Concept design only. &ldquo;Tremblay Plumbing&rdquo; is a fictional business created for illustration purposes. No real client, performance data, or business results are implied or guaranteed.</>
              }
            </motion.p>
          </div>}
        </div>

        {/* Transition fade */}
        <div aria-hidden style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          height: 130,
          background: 'linear-gradient(to bottom, transparent 0%, rgba(8,4,26,0.72) 100%)',
          pointerEvents: 'none',
        }} />

        {/* Scroll indicator */}
        {!overlay && <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          paddingBottom: 'clamp(18px, 3vw, 32px)',
          gap: 7, zIndex: 5, pointerEvents: 'none',
        }}>
          <span style={{
            fontSize: 9.5, fontWeight: 700, letterSpacing: '0.22em',
            textTransform: 'uppercase', color: 'rgba(139,92,246,0.40)',
          }}>
            {fr ? 'Nos réalisations' : 'Our Work'}
          </span>
          {!reduced && (
            <motion.div
              animate={{ y: [0, 5, 0] }}
              transition={{ duration: 1.9, repeat: Infinity, ease: 'easeInOut' }}
            >
              <svg width="14" height="9" viewBox="0 0 14 9" fill="none" aria-hidden>
                <path d="M1 1L7 7.5L13 1" stroke="rgba(139,92,246,0.40)"
                  strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </motion.div>
          )}
        </div>}
      </section>
    </>
  );
}
