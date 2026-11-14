'use client';

import { useEffect, useState } from 'react';

const KEY = 'agentedirect-analytics-consent';

type GtagWindow = Window & { dataLayer?: unknown[]; gtag?: (...args: unknown[]) => void };

function loadAnalytics(id: string) {
  if (document.getElementById('ga-script')) return;
  const w = window as GtagWindow;
  w.dataLayer = w.dataLayer || [];
  w.gtag = function () {
    // gtag requires the real `arguments` object, not a rest array.
    // eslint-disable-next-line prefer-rest-params
    w.dataLayer!.push(arguments);
  };
  w.gtag('js', new Date());
  w.gtag('config', id);
  const s = document.createElement('script');
  s.id = 'ga-script';
  s.async = true;
  s.src = `https://www.googletagmanager.com/gtag/js?id=${id}`;
  document.head.appendChild(s);
}

function readChoice(): string | null {
  try { return localStorage.getItem(KEY); } catch { return null; }
}

export default function ConsentBanner({ lang, gaId }: { lang: 'en' | 'fr'; gaId: string }) {
  const fr = lang === 'fr';
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const choice = readChoice();
    if (choice === 'granted') loadAnalytics(gaId);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- stored choice only exists in the browser
    else if (choice !== 'denied') setVisible(true);
  }, [gaId]);

  const choose = (granted: boolean) => {
    try { localStorage.setItem(KEY, granted ? 'granted' : 'denied'); } catch { /* private mode: just close */ }
    if (granted) loadAnalytics(gaId);
    setVisible(false);
  };

  if (!visible) return null;

  const btn: React.CSSProperties = {
    padding: '9px 20px', borderRadius: 50, fontSize: 13.5, fontWeight: 700,
    cursor: 'pointer', letterSpacing: '0.01em',
  };

  return (
    <div
      role="dialog"
      aria-label={fr ? 'Témoins et statistiques' : 'Cookies and analytics'}
      style={{
        position: 'fixed', left: 16, right: 16, bottom: 16, zIndex: 100,
        maxWidth: 520, marginRight: 'auto',
        background: '#120E2B', border: '1px solid rgba(162,75,255,0.32)', borderRadius: 14,
        padding: '18px 20px', boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        display: 'flex', flexDirection: 'column', gap: 14,
      }}
    >
      <p style={{ margin: 0, fontSize: 13.5, lineHeight: 1.6, color: 'rgba(220,210,245,0.82)' }}>
        {fr
          ? 'On utilise des témoins de statistiques pour comprendre comment le site est utilisé. Le site fonctionne pareil si vous refusez.'
          : 'We use analytics cookies to understand how the site is used. The site works the same if you decline.'}{' '}
        <a href={fr ? '/fr/privacy' : '/privacy'} style={{ color: '#C28DFF', textDecoration: 'underline', textUnderlineOffset: 3 }}>
          {fr ? 'Confidentialité' : 'Privacy policy'}
        </a>
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <button type="button" onClick={() => choose(true)} style={{ ...btn, border: 'none', color: '#fff', background: 'linear-gradient(135deg,#A24BFF 0%,#7A1DF0 100%)' }}>
          {fr ? 'Accepter' : 'Accept'}
        </button>
        <button type="button" onClick={() => choose(false)} style={{ ...btn, background: 'transparent', color: '#C28DFF', border: '1px solid rgba(162,75,255,0.40)' }}>
          {fr ? 'Refuser' : 'Decline'}
        </button>
      </div>
    </div>
  );
}
