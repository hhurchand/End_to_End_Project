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
                  On a fondé AgenteDirect parce qu&apos;on voyait trop d&apos;entreprises locales avec des sites qui ne les représentaient pas vraiment. Certains ressemblaient à tous les autres, d&apos;autres étaient dépassés ou n&apos;avaient tout simplement pas de site.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On sait que toutes les entreprises n&apos;ont pas besoin de quelque chose de compliqué. Parfois, un site simple et propre, c&apos;est exactement ce qu&apos;il faut, et il n&apos;y a rien de mal à ça.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Mais quand on prend un projet, on veut s&apos;assurer qu&apos;il reflète vraiment votre entreprise. On prend le temps de comprendre ce que vous faites, ce qui vous distingue et qui sont vos clients. On utilise vos photos, votre histoire et les détails qui font que votre entreprise est la vôtre. On ne prend pas un gabarit en changeant juste le logo.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Notre travail ne s&apos;arrête pas une fois votre site en ligne.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Les entreprises évoluent, les saisons passent, et il y a toujours quelque chose à mettre à jour ou à promouvoir. On est là pour ça. Que vous ayez besoin de faire un petit changement, de promouvoir une offre spéciale ou de rafraîchir votre site, vous pouvez nous contacter.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On veut bâtir des relations à long terme avec les entreprises avec qui on travaille, pas juste construire un site et passer à autre chose.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Vous remarquerez aussi qu&apos;on n&apos;a pas de page de tarifs ni de liste de forfaits. C&apos;est parce que chaque entreprise a des besoins différents. On préfère discuter, comprendre ce que vous cherchez et trouver ce qui fait le plus de sens.
                </p>
                <p style={{ marginBottom: '36px' }}>
                  Si c&apos;est le genre d&apos;expérience que vous recherchez,{' '}
                  <Link href="/fr/inquiry" style={{ color: '#fff', fontWeight: 600, textDecoration: 'underline', textUnderlineOffset: '4px', textDecorationColor: 'rgba(255,255,255,0.4)' }}>parlons-en.</Link>
                </p>
              </>
            ) : (
              <>
                <p style={{ marginBottom: '22px' }}>
                  We started AgenteDirect because we kept seeing local businesses with websites that didn&apos;t really represent them. Some looked just like every other website out there, while others were outdated or didn&apos;t have a website at all.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We know not every business needs something complicated. Sometimes a simple, clean website is exactly what you need, and there&apos;s nothing wrong with that.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  But when we take on a project, we want to make sure it actually reflects your business. We take the time to understand what you do, what makes you different and who your customers are. We use your photos, your story and the details that make your business yours. We don&apos;t just take the same template and change the logo.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Our work doesn&apos;t stop once your website goes live, either.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Businesses change, seasons come and go, and there&apos;s always something new to promote or update. We&apos;re here to help with that. Whether you need to make a quick change, promote a special offer or give your website a refresh, you can reach out to us.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We want to build long-term relationships with the businesses we work with, not just build a website and move on.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  You&apos;ll also notice we don&apos;t have a pricing page or a list of packages. That&apos;s because every business has different needs. We&apos;d rather talk, understand what you&apos;re looking for and figure out what makes the most sense.
                </p>
                <p style={{ marginBottom: '36px' }}>
                  If that sounds like the kind of experience you&apos;re looking for,{' '}
                  <Link href="/inquiry" style={{ color: '#fff', fontWeight: 600, textDecoration: 'underline', textUnderlineOffset: '4px', textDecorationColor: 'rgba(255,255,255,0.4)' }}>let&apos;s talk.</Link>
                </p>
              </>
            )}
          </div>

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
