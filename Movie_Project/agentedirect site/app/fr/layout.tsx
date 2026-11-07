import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: {
    default: 'Agence Web Montréal — Création de Sites Web | AgenteDirect',
    template: '%s | AgenteDirect',
  },
  description:
    'AgenteDirect conçoit des sites web professionnels pour les entreprises locales à Montréal et au Québec. Design sur mesure, bilingue, adapté aux mobiles. À partir de 199$/mois.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/home',
    languages: {
      'en-CA': 'https://agentedirect.ca/home',
      'fr-CA': 'https://agentedirect.ca/fr/home',
    },
  },
  openGraph: {
    title: 'Agence Web Montréal — Création de Sites Web | AgenteDirect',
    description:
      'Sites web professionnels pour entreprises locales à Montréal. Bilingue, mobile, SEO. À partir de 199$/mois.',
    url: 'https://agentedirect.ca/fr/home',
    locale: 'fr_CA',
    alternateLocale: 'en_CA',
  },
};

export default function FrLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
