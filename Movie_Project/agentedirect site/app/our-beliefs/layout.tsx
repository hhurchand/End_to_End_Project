import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Our Beliefs | AgenteDirect',
  description: 'Why we do what we do — and how we do it differently.',
};

export default function OurBeliefsLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
