import RootShell, { siteMetadata } from "@/components/RootShell";

export const metadata = siteMetadata;

export default function EnRootLayout({ children }: { children: React.ReactNode }) {
  return <RootShell lang="en">{children}</RootShell>;
}
