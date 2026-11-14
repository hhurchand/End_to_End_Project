import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Our Beliefs',
  description: 'Why we do what we do — and how we do it differently.',
  alternates: {
    canonical: 'https://agentedirect.ca/our-beliefs',
    languages: {
      'en-CA': 'https://agentedirect.ca/our-beliefs',
      'fr-CA': 'https://agentedirect.ca/fr/our-beliefs',
    },
  },
  openGraph: {
    title: 'Our Beliefs',
    description: 'Why we do what we do — and how we do it differently.',
    url: 'https://agentedirect.ca/our-beliefs',
    locale: 'en_CA',
  },
};

export default function OurBeliefsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
