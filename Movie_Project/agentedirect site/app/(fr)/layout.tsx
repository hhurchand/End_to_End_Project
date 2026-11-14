import RootShell, { siteMetadata } from "@/components/RootShell";

export const metadata = siteMetadata;

export default function FrRootLayout({ children }: { children: React.ReactNode }) {
  return <RootShell lang="fr">{children}</RootShell>;
}
