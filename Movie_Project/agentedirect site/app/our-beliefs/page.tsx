'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';

const ease = [0.16, 1, 0.3, 1] as const;

export default function OurBeliefsPage({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
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
          <Link href={beliefsHref} style={{ color: 'rgba(255,255,255,0.9)' }}>{fr ? 'Nos croyances' : 'Our Beliefs'}</Link>
          <Link href={projectsHref} className="hover:text-white transition-colors">{fr ? 'Nos projets' : 'Our Projects'}</Link>
          <Link href={contactHref} className="hover:text-white transition-colors">{fr ? 'Contact' : 'Contact'}</Link>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 4, paddingLeft: 16, borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
            <Link href="/our-beliefs" style={{ fontSize: 12, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.9)' }} className="hover:text-white transition-colors">EN</Link>
            <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 11 }}>|</span>
            <Link href="/fr/our-beliefs" style={{ fontSize: 12, fontWeight: 700, color: fr ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.3)' }} className="hover:text-white transition-colors">FR</Link>
          </div>
        </div>
      </nav>

      {/* Letter */}
      <div className="flex-1 flex items-center justify-center px-6 py-16 md:py-24">
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease }}
          style={{ maxWidth: '600px', width: '100%' }}
        >
          {/* Greeting */}
          <p style={{ fontSize: '15px', lineHeight: 1.7, marginBottom: '28px', color: 'rgba(255,255,255,0.45)', fontStyle: 'italic' }}>
            {fr ? 'À chaque propriétaire d\'entreprise avec qui nous travaillons,' : 'To every business owner we work with,'}
          </p>

          {/* Body */}
          <div style={{ fontSize: '17px', lineHeight: 1.9, color: 'rgba(255,255,255,0.85)' }}>
            {fr ? (
              <>
                <p style={{ marginBottom: '22px' }}>
                  Nous avons fondé AgenteDirect parce qu&apos;on voyait toujours la même chose : des entreprises locales avec des sites web qui auraient pu appartenir à n&apos;importe qui. Des mises en page génériques. Des photos de banque d&apos;images. Du contenu écrit par personne, pour personne.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Ce n&apos;est pas ce qu&apos;on fait.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Quand on prend un projet, on prend le temps. On demande vos vraies photos, vos vrais moments, ceux qui montrent vraiment ce qui rend votre entreprise unique. On construit autour de <em>vous</em>, pas autour d&apos;un gabarit qu&apos;on a utilisé cinquante fois.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Et on ne disparaît pas après le lancement.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On reste à vos côtés. Quand une nouvelle saison arrive, on vous contacte. Quand vous voulez pousser une promotion ou mettre quelque chose à jour, on est à un message. On se voit moins comme une agence web et plus comme l&apos;équipe qui fait travailler votre présence en ligne pour vous, toute l&apos;année.
                </p>
                <p style={{ marginBottom: '36px' }}>
                  Il n&apos;y a pas de page de tarifs ici. Pas de forfaits à choisir. Parce que chaque entreprise est différente, et on préfère avoir une vraie conversation plutôt que de vous remettre un menu déroulant.
                </p>
              </>
            ) : (
              <>
                <p style={{ marginBottom: '22px' }}>
                  We started AgenteDirect because we kept seeing the same thing: local businesses with websites that could belong to anyone. Generic layouts. Stock photos. Copy written by no one, for no one.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  That&apos;s not what we&apos;re here to do.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  When we take on a project, we slow down. We ask for your real photos, your real moments, the ones that actually show what makes your business worth choosing. We build around <em>you</em>, not around a template we&apos;ve used fifty times before.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  And we don&apos;t disappear after launch.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We stay by your side. When a new season comes around, we reach out. When you want to push a promotion or update something, we&apos;re a message away. We think of ourselves less as a web agency and more as the team quietly keeping your online presence working for you all year long.
                </p>
                <p style={{ marginBottom: '36px' }}>
                  There&apos;s no pricing page here. No packages to pick from. Because every business is different, and we&apos;d rather have a real conversation than hand you a dropdown menu.
                </p>
              </>
            )}
          </div>

          {/* Closing */}
          <p style={{ fontSize: '17px', lineHeight: 1.7, marginBottom: '40px', color: 'rgba(255,255,255,0.85)' }}>
            {fr ? (
              <>Si ça ressemble à ce que vous cherchez,{' '}
                <Link href="/fr/inquiry" style={{ color: '#fff', fontWeight: 600, textDecoration: 'underline', textUnderlineOffset: '4px', textDecorationColor: 'rgba(255,255,255,0.4)' }}>parlons-en.</Link>
              </>
            ) : (
              <>If that sounds like what you&apos;ve been looking for,{' '}
                <Link href="/inquiry" style={{ color: '#fff', fontWeight: 600, textDecoration: 'underline', textUnderlineOffset: '4px', textDecorationColor: 'rgba(255,255,255,0.4)' }}>let&apos;s talk.</Link>
              </>
            )}
          </p>

          {/* Signature */}
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '28px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
            <div>
              <p style={{ fontSize: '13px', color: 'rgba(255,255,255,0.35)', marginBottom: '4px' }}>{fr ? 'Avec soin,' : 'With care,'}</p>
              <p style={{ fontSize: '16px', fontWeight: 600, color: 'rgba(255,255,255,0.7)', letterSpacing: '-0.01em' }}>
                {fr ? 'L\'équipe AgenteDirect' : 'The AgenteDirect Team'}
              </p>
            </div>
            <Image src="/logo.png" alt="AgenteDirect" width={120} height={32} className="h-7 w-auto" style={{ opacity: 0.5 }} />
          </div>
        </motion.div>
      </div>
    </main>
  );
}
