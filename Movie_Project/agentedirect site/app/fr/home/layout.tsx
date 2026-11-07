import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Agence Web Montréal | AgenteDirect',
  description:
    'AgenteDirect crée des sites web professionnels pour les entreprises locales à Montréal et au Québec. Design sur mesure, bilingue EN/FR, mobile, SEO. À partir de 199$/mois.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/home',
    languages: {
      'en-CA': 'https://agentedirect.ca/home',
      'fr-CA': 'https://agentedirect.ca/fr/home',
    },
  },
  openGraph: {
    title: 'Agence Web Montréal | AgenteDirect',
    description:
      'Sites web sur mesure pour entreprises locales à Montréal. Bilingue, mobile, SEO. À partir de 199$/mois.',
    url: 'https://agentedirect.ca/fr/home',
    locale: 'fr_CA',
  },
};

export default function FrHomeLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
