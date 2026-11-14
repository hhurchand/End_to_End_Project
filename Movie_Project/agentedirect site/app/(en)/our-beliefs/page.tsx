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
          <p style={{ fontSize: '15px', lineHeight: 1.7, marginBottom: '28px', color: 'rgba(255,255,255,0.66)', fontStyle: 'italic' }}>
            {fr ? 'À chaque propriétaire d\'entreprise avec qui nous travaillons,' : 'To every business owner we work with,'}
          </p>

          {/* Body */}
          <div style={{ fontSize: '17px', lineHeight: 1.9, color: 'rgba(255,255,255,0.91)' }}>
            {fr ? (
              <>
                <p style={{ marginBottom: '22px' }}>
                  On a fondé AgenteDirect parce qu&apos;on voyait trop d&apos;entreprises locales qui méritaient une meilleure présence en ligne. Certaines avaient des sites dépassés qui ne reflétaient plus la qualité de leur travail. D&apos;autres avaient des sites qui ressemblaient à tous les autres. Et certaines n&apos;avaient tout simplement pas de site.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On croit que chaque entreprise mérite un site qui la représente vraiment. Que vous partiez de zéro et ayez besoin d&apos;un tout nouveau site, ou que vous en ayez déjà un qui a besoin d&apos;une refonte complète, on est là pour vous aider.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On comprend aussi que toutes les entreprises n&apos;ont pas besoin de quelque chose de compliqué. Parfois, un site simple et propre, c&apos;est exactement ce qu&apos;il faut, et il n&apos;y a rien de mal à ça.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Ce qui compte pour nous, c&apos;est que le résultat final ressemble à votre entreprise. On prend le temps de comprendre ce que vous faites, ce qui vous distingue et qui sont vos clients. On utilise vos photos, votre histoire et les détails qui font que votre entreprise est la vôtre. On ne prend pas un gabarit en changeant juste le logo.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Et notre travail ne s&apos;arrête pas une fois votre site en ligne.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Les entreprises évoluent. Les saisons changent. De nouveaux services voient le jour, des promotions arrivent et repartent, et parfois votre site a simplement besoin d&apos;un peu d&apos;attention. On est là quand ces choses arrivent, que ce soit pour une mise à jour rapide, une nouvelle page ou une refonte plus importante.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  On veut bâtir des relations à long terme avec les entreprises avec qui on travaille, pas juste construire un site et passer à autre chose.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Vous remarquerez aussi qu&apos;on n&apos;affiche pas de tarifs. C&apos;est parce que chaque entreprise a des besoins différents. On préfère apprendre à connaître votre entreprise, comprendre ce que vous cherchez et vous proposer quelque chose qui a du sens pour vous.
                </p>
                <p style={{ marginBottom: '36px' }}>
                  Si c&apos;est le genre d&apos;expérience que vous recherchez,{' '}
                  <Link href="/fr/inquiry" style={{ color: '#fff', fontWeight: 600, textDecoration: 'underline', textUnderlineOffset: '4px', textDecorationColor: 'rgba(255,255,255,0.4)' }}>parlons-en.</Link>
                </p>
              </>
            ) : (
              <>
                <p style={{ marginBottom: '22px' }}>
                  We started AgenteDirect because we kept seeing local businesses that deserved a better online presence. Some had websites that were outdated or no longer reflected the quality of their work. Others had websites that looked just like everyone else&apos;s. And some didn&apos;t have a website at all.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We believe every business deserves a website that represents it properly. Whether you&apos;re starting from scratch and need a brand-new website, or you already have one that needs a complete redesign, we&apos;re here to help.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We also understand that not every business needs something complicated. Sometimes a simple, clean website is exactly what you need, and there&apos;s nothing wrong with that.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  What matters to us is that the end result feels like your business. We take the time to understand what you do, what makes you different and who your customers are. We use your photos, your story and the details that make your business yours. We don&apos;t just take a template and change the logo.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  And our work doesn&apos;t stop once your website goes live.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  Businesses evolve. Seasons change. New services launch, promotions come and go, and sometimes your website simply needs a little attention. We&apos;re here when those things happen, whether it&apos;s a quick update, a new page or a bigger refresh down the road.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  We want to build long-term relationships with the businesses we work with, not just build a website and move on.
                </p>
                <p style={{ marginBottom: '22px' }}>
                  You&apos;ll also notice we don&apos;t have any pricing. That&apos;s because every business has different needs. We&apos;d rather get to know your business, understand what you&apos;re looking for and put together something that makes sense for you.
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
              <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.60)', marginBottom: '4px' }}>{fr ? 'Avec soin,' : 'With care,'}</p>
              <p style={{ fontSize: '16px', fontWeight: 600, color: 'rgba(255,255,255,0.81)', letterSpacing: '-0.01em' }}>
                {fr ? 'L\'équipe AgenteDirect' : 'The AgenteDirect Team'}
              </p>
            </div>
            <Image src="/logo.png" alt="AgenteDirect" width={220} height={58} style={{ height: 52, width: 'auto', filter: 'brightness(1.25) drop-shadow(0 0 18px rgba(162,75,255,0.45))' }} />
          </div>
        </motion.div>
      </div>
      <PageFooter lang={lang} />
    </main>
  );
}
