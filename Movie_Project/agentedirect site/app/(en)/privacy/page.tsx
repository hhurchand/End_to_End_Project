'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import PageNav from '@/components/PageNav';
import PageFooter from '@/components/PageFooter';

const ease = [0.16, 1, 0.3, 1] as const;
const EMAIL = 'support@agentedirect.ca';

type Section = { h: string; p: string[] };

const EN: { title: string; updated: string; intro: string; sections: Section[] } = {
  title: 'Privacy Policy',
  updated: 'Last updated: November 13, 2026',
  intro:
    'AgenteDirect is a web design agency in Montréal, Québec. This page explains what personal information we collect through this website, why, who else handles it, and how to reach us about it.',
  sections: [
    {
      h: 'What we collect',
      p: [
        'Inquiry form: your name, business name, email address, current website (if any), industry, the description you write, and the option you are interested in. We collect this only when you send it to us.',
        'Chat assistant: the messages you type into the chat window. Please do not share sensitive information there, such as payment or health details. We do not keep chat conversations on our own servers.',
        'Analytics: if you accept analytics cookies, Google Analytics collects information about how you use the site, such as pages visited, approximate location, device and browser. If you decline, no analytics script is loaded.',
        'Your browser also stores your cookie choice on your device so we do not ask you again.',
      ],
    },
    {
      h: 'Why we use it',
      p: [
        'We use inquiry details only to reply to you, prepare a quote and, if you become a client, deliver your project. We do not sell your information, and we do not use it for unrelated marketing without asking you first.',
        'We use chat messages only to generate an answer. We use analytics only to understand what visitors read so we can improve the site.',
      ],
    },
    {
      h: 'Who else handles it',
      p: [
        'We rely on service providers to run the site: our hosting provider, an email delivery service that sends us your inquiry (Resend), an AI provider that writes chat replies (Anthropic), and, if you accept cookies, Google Analytics.',
        'Some of these providers operate outside Québec and outside Canada, including in the United States, so your information may be processed there. We share only what each service needs to do its job.',
      ],
    },
    {
      h: 'How long we keep it',
      p: [
        'We keep inquiry emails for as long as we need them to reply, quote and work with you, and then delete them. Analytics data is kept by Google according to its own retention settings. You can ask us to delete your information at any time.',
      ],
    },
    {
      h: 'Your choices and rights',
      p: [
        'You can ask to see the personal information we hold about you, to correct it, or to have it deleted. You can withdraw your consent to analytics at any time by clearing this site’s data in your browser, which brings the cookie choice back.',
        `To exercise any of these rights, or to ask a privacy question, email us at ${EMAIL}. We aim to answer within 30 days. The AgenteDirect team is responsible for the protection of personal information on this site.`,
      ],
    },
    {
      h: 'Changes',
      p: ['If we change this policy, we will update the date at the top of this page.'],
    },
  ],
};

const FR: typeof EN = {
  title: 'Politique de confidentialité',
  updated: 'Dernière mise à jour : 13 novembre 2026',
  intro:
    'AgenteDirect est une agence de conception web à Montréal, au Québec. Cette page explique quels renseignements personnels on recueille sur ce site, pourquoi, qui d’autre les traite et comment nous joindre à ce sujet.',
  sections: [
    {
      h: 'Ce qu’on recueille',
      p: [
        'Formulaire de demande : votre nom, le nom de votre entreprise, votre courriel, votre site actuel (s’il y a lieu), votre secteur, la description que vous écrivez et l’option qui vous intéresse. On ne recueille ces renseignements que lorsque vous nous les envoyez.',
        'Assistant de clavardage : les messages que vous écrivez dans la fenêtre de discussion. Veuillez ne pas y partager de renseignements sensibles, comme des données de paiement ou de santé. On ne conserve pas les conversations sur nos propres serveurs.',
        'Statistiques : si vous acceptez les témoins de statistiques, Google Analytics recueille des renseignements sur votre utilisation du site, comme les pages consultées, la région approximative, l’appareil et le navigateur. Si vous refusez, aucun script de statistiques n’est chargé.',
        'Votre navigateur conserve aussi votre choix sur votre appareil pour qu’on ne vous le redemande pas.',
      ],
    },
    {
      h: 'Pourquoi on les utilise',
      p: [
        'On utilise les renseignements de votre demande uniquement pour vous répondre, préparer une soumission et, si vous devenez client, réaliser votre projet. On ne vend pas vos renseignements et on ne les utilise pas pour du marketing sans lien avec votre demande sans vous le demander d’abord.',
        'On utilise les messages de clavardage seulement pour générer une réponse. On utilise les statistiques seulement pour comprendre ce que les visiteurs consultent afin d’améliorer le site.',
      ],
    },
    {
      h: 'Qui d’autre les traite',
      p: [
        'On fait appel à des fournisseurs pour faire fonctionner le site : notre hébergeur, un service de courriel qui nous transmet votre demande (Resend), un fournisseur d’IA qui rédige les réponses du clavardage (Anthropic) et, si vous acceptez les témoins, Google Analytics.',
        'Certains de ces fournisseurs exercent leurs activités à l’extérieur du Québec et du Canada, notamment aux États-Unis, et vos renseignements peuvent donc y être traités. On ne partage que ce dont chaque service a besoin pour faire son travail.',
      ],
    },
    {
      h: 'Combien de temps on les garde',
      p: [
        'On garde les courriels de demande aussi longtemps que nécessaire pour vous répondre, vous soumissionner et travailler avec vous, puis on les supprime. Les données de statistiques sont conservées par Google selon ses propres paramètres. Vous pouvez nous demander de supprimer vos renseignements en tout temps.',
      ],
    },
    {
      h: 'Vos choix et vos droits',
      p: [
        'Vous pouvez demander à consulter les renseignements personnels qu’on détient sur vous, à les faire corriger ou à les faire supprimer. Vous pouvez retirer votre consentement aux statistiques en tout temps en effaçant les données de ce site dans votre navigateur, ce qui fait réapparaître le choix des témoins.',
        `Pour exercer ces droits ou poser une question sur la vie privée, écrivez-nous à ${EMAIL}. On vise à répondre dans un délai de 30 jours. L’équipe d’AgenteDirect est responsable de la protection des renseignements personnels sur ce site.`,
      ],
    },
    {
      h: 'Modifications',
      p: ['Si on modifie cette politique, on mettra à jour la date en haut de cette page.'],
    },
  ],
};

export default function PrivacyPage({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';
  const c = fr ? FR : EN;

  return (
    <main
      className="min-h-screen flex flex-col"
      style={{ background: '#0F0C07', fontFamily: 'var(--font-space-grotesk)' }}
    >
      <PageNav lang={lang} active="privacy" enHref="/privacy" frHref="/fr/privacy" />

      <div className="flex-1 flex justify-center px-6 py-16 md:py-24">
        <motion.article
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease }}
          style={{ maxWidth: 640, width: '100%' }}
        >
          <h1 style={{ fontSize: 'clamp(28px,4vw,40px)', fontWeight: 700, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 10px' }}>
            {c.title}
          </h1>
          <p style={{ fontSize: 13.5, color: 'rgba(255,255,255,0.50)', margin: '0 0 28px' }}>{c.updated}</p>
          <p style={{ fontSize: 15, lineHeight: 1.75, color: 'rgba(255,255,255,0.72)', margin: '0 0 36px' }}>{c.intro}</p>

          {c.sections.map(s => (
            <section key={s.h} style={{ marginBottom: 32 }}>
              <h2 style={{ fontSize: 18, fontWeight: 600, color: '#F5F0FC', letterSpacing: '-0.01em', margin: '0 0 12px' }}>{s.h}</h2>
              {s.p.map(t => (
                <p key={t} style={{ fontSize: 15, lineHeight: 1.75, color: 'rgba(255,255,255,0.68)', margin: '0 0 12px' }}>{t}</p>
              ))}
            </section>
          ))}

          <p style={{ fontSize: 15, lineHeight: 1.75, color: 'rgba(255,255,255,0.68)', margin: '8px 0 0' }}>
            {fr ? 'Une question ? ' : 'Questions? '}
            <Link href={fr ? '/fr/inquiry' : '/inquiry'} style={{ color: '#C28DFF', textDecoration: 'underline', textUnderlineOffset: 4 }}>
              {fr ? 'Écrivez-nous' : 'Get in touch'}
            </Link>
            {' · '}
            <a href={`mailto:${EMAIL}`} style={{ color: '#C28DFF', textDecoration: 'underline', textUnderlineOffset: 4 }}>{EMAIL}</a>
          </p>
        </motion.article>
      </div>
      <PageFooter lang={lang} />
    </main>
  );
}
