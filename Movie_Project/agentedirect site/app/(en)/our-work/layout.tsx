import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Our Projects',
  description: 'Websites we\'ve built for Montréal businesses that actually convert.',
  alternates: {
    canonical: 'https://agentedirect.ca/our-work',
    languages: {
      'en-CA': 'https://agentedirect.ca/our-work',
      'fr-CA': 'https://agentedirect.ca/fr/our-work',
    },
  },
  openGraph: {
    title: 'Our Projects',
    description: 'Websites we\'ve built for Montréal businesses that actually convert.',
    url: 'https://agentedirect.ca/our-work',
    locale: 'en_CA',
  },
};

export default function OurWorkLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
