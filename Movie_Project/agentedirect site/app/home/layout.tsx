import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Web Design Agency Montreal | AgenteDirect',
  description:
    'AgenteDirect builds custom websites for local businesses in Montreal and Quebec. Professional design, bilingual EN/FR, mobile-first, SEO-ready. Starting at $199/month.',
  alternates: {
    canonical: 'https://agentedirect.ca/home',
    languages: {
      'en-CA': 'https://agentedirect.ca/home',
      'fr-CA': 'https://agentedirect.ca/fr/home',
    },
  },
  openGraph: {
    title: 'Web Design Agency Montreal | AgenteDirect',
    description:
      'Custom websites for local businesses in Montreal. Bilingual, mobile-first, SEO-optimized. Starting at $199/month.',
    url: 'https://agentedirect.ca/home',
    locale: 'en_CA',
  },
};

export default function HomeLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
