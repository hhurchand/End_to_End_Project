import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Nos projets',
  description: 'Des sites web conçus pour des entreprises montréalaises, pensés pour convertir.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/our-work',
    languages: {
      'en-CA': 'https://agentedirect.ca/our-work',
      'fr-CA': 'https://agentedirect.ca/fr/our-work',
    },
  },
  openGraph: {
    title: 'Nos projets',
    description: 'Des sites web conçus pour des entreprises montréalaises, pensés pour convertir.',
    url: 'https://agentedirect.ca/fr/our-work',
    locale: 'fr_CA',
  },
};

export default function FrOurWorkLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
