import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Nos croyances',
  description: 'Pourquoi on fait ce qu\'on fait — et comment on le fait différemment.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/our-beliefs',
    languages: {
      'en-CA': 'https://agentedirect.ca/our-beliefs',
      'fr-CA': 'https://agentedirect.ca/fr/our-beliefs',
    },
  },
  openGraph: {
    title: 'Nos croyances',
    description: 'Pourquoi on fait ce qu\'on fait — et comment on le fait différemment.',
    url: 'https://agentedirect.ca/fr/our-beliefs',
    locale: 'fr_CA',
  },
};

export default function FrOurBeliefsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
