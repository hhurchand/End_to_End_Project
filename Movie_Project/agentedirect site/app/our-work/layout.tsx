import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Our Work | AgenteDirect',
  description: 'Websites we\'ve built for Montréal businesses that actually convert.',
};

export default function OurWorkLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
