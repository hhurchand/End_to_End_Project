import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'La Stratégie | AgenteDirect',
  description: 'Un site web professionnel conçu autour de votre entreprise — design sur mesure, bilingue, avec support continu. À partir de 199 $ CA/mois.',
};

export default function FrPlanLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
