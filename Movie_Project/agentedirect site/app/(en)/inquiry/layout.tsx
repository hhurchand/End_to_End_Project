import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Contact',
  description: 'Tell us about your business and we will put together a website that fits. Free, no commitment.',
  alternates: {
    canonical: 'https://agentedirect.ca/inquiry',
    languages: {
      'en-CA': 'https://agentedirect.ca/inquiry',
      'fr-CA': 'https://agentedirect.ca/fr/inquiry',
    },
  },
  openGraph: {
    title: 'Contact',
    description: 'Tell us about your business and we will put together a website that fits. Free, no commitment.',
    url: 'https://agentedirect.ca/inquiry',
    locale: 'en_CA',
  },
};

export default function InquiryLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
