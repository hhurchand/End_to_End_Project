import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'The Strategy | AgenteDirect',
  description: 'A professional website built around your business — custom design, bilingual, and always there when you need updates. Starting at $199/month.',
};

export default function PlanLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
