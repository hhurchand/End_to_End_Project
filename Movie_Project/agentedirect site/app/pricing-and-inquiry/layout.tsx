import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Website Plans & Pricing | AgenteDirect Montreal',
  description:
    'Transparent pricing for professional websites. Custom design starting at $199/month. Request a free quote — no commitment, fast turnaround.',
  alternates: {
    canonical: 'https://agentedirect.ca/pricing-and-inquiry',
    languages: {
      'en-CA': 'https://agentedirect.ca/pricing-and-inquiry',
      'fr-CA': 'https://agentedirect.ca/fr/pricing-and-inquiry',
    },
  },
  openGraph: {
    title: 'Website Plans & Pricing | AgenteDirect Montreal',
    description:
      'Professional website design starting at $199/month. Free quote, fast delivery, bilingual support.',
    url: 'https://agentedirect.ca/pricing-and-inquiry',
    locale: 'en_CA',
  },
};

export default function PricingLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
