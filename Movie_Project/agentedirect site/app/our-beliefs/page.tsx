'use client';

import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import PageNav from '@/components/PageNav';
import PageFooter from '@/components/PageFooter';

const ease = [0.16, 1, 0.3, 1] as const;

export default function OurBeliefsPage({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';

  return (
    <main
      className="min-h-screen flex flex-col"
      style={{ background: '#0F0C07', fontFamily: 'var(--font-space-grotesk)' }}
    >
      <PageNav lang={lang} active="beliefs" enHref="/our-beliefs" frHref="/fr/our-beliefs" />

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
                  On a fondé AgenteDirect après avoir vu trop d&apos;entreprises locales avec des sites qui auraient pu appartenir à n&apos;importe qui. Les mêmes gabarits. Les mêmes photos de stock. Du texte que personne n&apos;a réellement écrit, pour des clients que personne ne connaît vraiment.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On comprend que certaines entreprises cherchent exactement ça. Et c&apos;est tout à fait correct.
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
                  We started AgenteDirect after seeing too many local businesses with sites that could&apos;ve belonged to anyone. Same layouts. Same stock photos. Words no one actually wrote for them.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We understand that some businesses are looking for exactly that, and that&apos;s completely valid.
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
      <PageFooter />
    </main>
  );
}
