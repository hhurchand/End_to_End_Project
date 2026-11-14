import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Contact',
  description: 'Parlez-nous de votre entreprise et on vous propose un site qui vous ressemble. Gratuit, sans engagement.',
  alternates: {
    canonical: 'https://agentedirect.ca/fr/inquiry',
    languages: {
      'en-CA': 'https://agentedirect.ca/inquiry',
      'fr-CA': 'https://agentedirect.ca/fr/inquiry',
    },
  },
  openGraph: {
    title: 'Contact',
    description: 'Parlez-nous de votre entreprise et on vous propose un site qui vous ressemble. Gratuit, sans engagement.',
    url: 'https://agentedirect.ca/fr/inquiry',
    locale: 'fr_CA',
  },
};

export default function FrInquiryLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
