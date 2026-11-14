import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Services & Strategy',
  description: 'A professional website built around your business — custom design, bilingual, and always there when you need updates.',
  alternates: {
    canonical: 'https://agentedirect.ca/strategy',
    languages: {
      'en-CA': 'https://agentedirect.ca/strategy',
      'fr-CA': 'https://agentedirect.ca/fr/strategy',
    },
  },
  openGraph: {
    title: 'Services & Strategy',
    description: 'A professional website built around your business — custom design, bilingual, and always there when you need updates.',
    url: 'https://agentedirect.ca/strategy',
    locale: 'en_CA',
  },
};

export default function StrategyLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
