import type { Metadata } from "next";
import { Space_Grotesk, Inter } from "next/font/google";
import "@/app/globals.css";
import ConsentBanner from "@/components/ConsentBanner";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const siteMetadata: Metadata = {
  metadataBase: new URL("https://agentedirect.ca"),
  title: {
    default: "AgenteDirect — Web Design Agency Montreal",
    template: "%s | AgenteDirect",
  },
  description:
    "AgenteDirect builds professional websites for local businesses in Montreal and Quebec. Custom design, bilingual (EN/FR), mobile-first, SEO-ready.",
  keywords: [
    "web design Montreal",
    "website design agency Montreal",
    "build website Montreal",
    "web design for small business",
    "agence web Montréal",
    "création site web Montréal",
    "website design Quebec",
    "local business website Canada",
    "bilingual website design",
    "affordable web design Montreal",
  ],
  authors: [{ name: "AgenteDirect", url: "https://agentedirect.ca" }],
  creator: "AgenteDirect",
  publisher: "AgenteDirect",
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-snippet": -1, "max-image-preview": "large" },
  },
  alternates: {
    canonical: "https://agentedirect.ca/home",
    languages: {
      "en-CA": "https://agentedirect.ca/home",
      "fr-CA": "https://agentedirect.ca/fr/home",
    },
  },
  icons: {
    icon: "/image.png",
    shortcut: "/image.png",
    apple: "/image.png",
  },
  openGraph: {
    title: "AgenteDirect — Web Design Agency Montreal",
    description:
      "Professional websites for local businesses in Montreal and Quebec. Bilingual, mobile-first, SEO-ready.",
    url: "https://agentedirect.ca",
    siteName: "AgenteDirect",
    images: [{ url: "/image.png", width: 800, height: 800, alt: "AgenteDirect — Web Design Agency Montreal" }],
    type: "website",
    locale: "en_CA",
    alternateLocale: "fr_CA",
  },
  twitter: {
    card: "summary_large_image",
    title: "AgenteDirect — Web Design Agency Montreal",
    description: "Professional websites for local businesses in Montreal and Quebec.",
    images: ["/image.png"],
  },
};

const jsonLd = {
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'Organization',
      '@id': 'https://agentedirect.ca/#organization',
      name: 'AgenteDirect',
      url: 'https://agentedirect.ca',
      logo: { '@type': 'ImageObject', url: 'https://agentedirect.ca/image.png' },
      email: 'support@agentedirect.ca',
      areaServed: ['Montreal', 'Quebec', 'Canada'],
      availableLanguage: ['English', 'French'],
    },
    {
      '@type': 'LocalBusiness',
      '@id': 'https://agentedirect.ca/#localbusiness',
      name: 'AgenteDirect',
      description:
        'Web design agency building professional, bilingual websites for local businesses in Montreal and Quebec.',
      url: 'https://agentedirect.ca',
      image: 'https://agentedirect.ca/image.png',
      email: 'support@agentedirect.ca',
      address: {
        '@type': 'PostalAddress',
        addressLocality: 'Montreal',
        addressRegion: 'QC',
        addressCountry: 'CA',
      },
      hasOfferCatalog: {
        '@type': 'OfferCatalog',
        name: 'Web Design Services',
        itemListElement: [
          {
            '@type': 'Offer',
            itemOffered: {
              '@type': 'Service',
              name: 'Custom Website Design',
              description: 'Professional website design for local businesses, bilingual EN/FR.',
            },
          },
          {
            '@type': 'Offer',
            itemOffered: {
              '@type': 'Service',
              name: 'Website + AI Chatbot',
              description: 'Professional website with integrated AI chatbot for local businesses.',
            },
          },
        ],
      },
    },
    {
      '@type': 'WebSite',
      '@id': 'https://agentedirect.ca/#website',
      url: 'https://agentedirect.ca',
      name: 'AgenteDirect',
      publisher: { '@id': 'https://agentedirect.ca/#organization' },
      inLanguage: ['en-CA', 'fr-CA'],
    },
  ],
};

export default function RootShell({
  lang,
  children,
}: {
  lang: "en" | "fr";
  children: React.ReactNode;
}) {
  return (
    <html
      lang={lang}
      className={`${spaceGrotesk.variable} ${inter.variable} h-full antialiased`}
    >
      <head>
        <link rel="icon" type="image/png" href="/image.png" />
        <link rel="shortcut icon" type="image/png" href="/image.png" />
        <link rel="apple-touch-icon" href="/image.png" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className="min-h-full flex flex-col" style={{ background: "#0F0C07" }}>
        {children}
        <ConsentBanner lang={lang} gaId="G-LWS7BZQ22V" />
      </body>
    </html>
  );
}
