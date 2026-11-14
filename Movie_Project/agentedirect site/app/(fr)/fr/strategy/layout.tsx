import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'La Stratégie',
  description: 'Un site web professionnel conçu autour de votre entreprise — design sur mesure, bilingue, avec support continu.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/strategy',
    languages: {
      'en-CA': 'https://agentedirect.ca/strategy',
      'fr-CA': 'https://agentedirect.ca/fr/strategy',
    },
  },
  openGraph: {
    title: 'La Stratégie',
    description: 'Un site web professionnel conçu autour de votre entreprise — design sur mesure, bilingue, avec support continu.',
    url: 'https://agentedirect.ca/fr/strategy',
    locale: 'fr_CA',
  },
};

export default function FrStrategyLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
