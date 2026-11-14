'use client';

// Update PHONE, INSTAGRAM_URL, FACEBOOK_URL with your real info
const PHONE = '(514) 663-4210';
const EMAIL = 'support@agentedirect.ca';
const INSTAGRAM_URL = 'https://www.instagram.com/agentedirect';
const FACEBOOK_URL = 'https://www.facebook.com/share/19mHFnGPXf/?mibextid=wwXIfr';

export default function PageFooter({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';
  return (
    <footer style={{
      borderTop: '1px solid rgba(255,255,255,0.06)',
      padding: '32px 24px',
      display: 'flex',
      flexWrap: 'wrap',
      alignItems: 'center',
      gap: 20,
      fontFamily: 'var(--font-space-grotesk)',
    }}>
      <a href={`tel:${PHONE.replace(/\D/g,'')}`} style={{ fontSize: 14, color: 'rgba(255,255,255,0.63)', textDecoration: 'none', letterSpacing: '0.01em' }}>
        {PHONE}
      </a>
      <span style={{ color: 'rgba(255,255,255,0.44)', fontSize: 14 }}>·</span>
      <a href={`mailto:${EMAIL}`} style={{ fontSize: 14, color: 'rgba(194,141,255,0.78)', textDecoration: 'none', letterSpacing: '0.01em' }}>
        {EMAIL}
      </a>
      <span style={{ color: 'rgba(255,255,255,0.44)', fontSize: 14 }}>·</span>
      <a href={fr ? '/fr/privacy' : '/privacy'} style={{ fontSize: 14, color: 'rgba(255,255,255,0.63)', textDecoration: 'none', letterSpacing: '0.01em' }}>
        {fr ? 'Confidentialité' : 'Privacy'}
      </a>
      <span style={{ color: 'rgba(255,255,255,0.44)', fontSize: 14 }}>·</span>

      {/* Instagram */}
      <a href={INSTAGRAM_URL} target="_blank" rel="noopener noreferrer" aria-label="Instagram"
        style={{ color: 'rgba(255,255,255,0.57)', transition: 'color 0.2s', display: 'flex' }}
        onMouseEnter={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.84)'}
        onMouseLeave={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.57)'}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          <rect x="2" y="2" width="20" height="20" rx="5" ry="5"/>
          <circle cx="12" cy="12" r="4"/>
          <circle cx="17.5" cy="6.5" r="1" fill="currentColor" stroke="none"/>
        </svg>
      </a>

      {/* Facebook */}
      <a href={FACEBOOK_URL} target="_blank" rel="noopener noreferrer" aria-label="Facebook"
        style={{ color: 'rgba(255,255,255,0.57)', transition: 'color 0.2s', display: 'flex' }}
        onMouseEnter={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.84)'}
        onMouseLeave={e => (e.currentTarget as HTMLAnchorElement).style.color = 'rgba(255,255,255,0.57)'}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"/>
        </svg>
      </a>
    </footer>
  );
}
