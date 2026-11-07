import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Prix & Forfaits — Sites Web | AgenteDirect Montréal',
  description:
    'Tarification transparente pour votre site web professionnel. À partir de 199$/mois. Soumission gratuite, livraison rapide, support bilingue.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/pricing-and-inquiry',
    languages: {
      'en-CA': 'https://agentedirect.ca/pricing-and-inquiry',
      'fr-CA': 'https://agentedirect.ca/fr/pricing-and-inquiry',
    },
  },
  openGraph: {
    title: 'Prix & Forfaits — Sites Web | AgenteDirect Montréal',
    description:
      'Site web professionnel à partir de 199$/mois. Soumission gratuite, livraison rapide.',
    url: 'https://agentedirect.ca/fr/pricing-and-inquiry',
    locale: 'fr_CA',
  },
};

export default function FrPricingLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
