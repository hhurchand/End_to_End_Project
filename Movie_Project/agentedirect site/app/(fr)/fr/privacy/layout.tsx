import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Politique de confidentialité',
  description: 'Comment AgenteDirect recueille, utilise et protège les renseignements personnels sur ce site.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/privacy',
    languages: {
      'en-CA': 'https://agentedirect.ca/privacy',
      'fr-CA': 'https://agentedirect.ca/fr/privacy',
    },
  },
  openGraph: {
    title: 'Politique de confidentialité',
    description: 'Comment AgenteDirect recueille, utilise et protège les renseignements personnels sur ce site.',
    url: 'https://agentedirect.ca/fr/privacy',
    locale: 'fr_CA',
  },
};

export default function FrPrivacyLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
