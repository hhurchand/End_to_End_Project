import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Privacy Policy',
  description: 'How AgenteDirect collects, uses and protects personal information on this website.',
  alternates: {
    canonical: 'https://agentedirect.ca/privacy',
    languages: {
      'en-CA': 'https://agentedirect.ca/privacy',
      'fr-CA': 'https://agentedirect.ca/fr/privacy',
    },
  },
  openGraph: {
    title: 'Privacy Policy',
    description: 'How AgenteDirect collects, uses and protects personal information on this website.',
    url: 'https://agentedirect.ca/privacy',
    locale: 'en_CA',
  },
};

export default function PrivacyLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
