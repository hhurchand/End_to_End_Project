import { MetadataRoute } from 'next';

const base = 'https://agentedirect.ca';

const pages: { path: string; priority: number }[] = [
  { path: 'home', priority: 1.0 },
  { path: 'strategy', priority: 0.9 },
  { path: 'our-work', priority: 0.9 },
  { path: 'our-beliefs', priority: 0.7 },
  { path: 'inquiry', priority: 0.9 },
  { path: 'privacy', priority: 0.3 },
];

export default function sitemap(): MetadataRoute.Sitemap {
  return pages.flatMap(({ path, priority }) => [
    {
      url: `${base}/${path}`,
      lastModified: new Date(),
      changeFrequency: 'monthly' as const,
      priority,
      alternates: { languages: { 'en-CA': `${base}/${path}`, 'fr-CA': `${base}/fr/${path}` } },
    },
    {
      url: `${base}/fr/${path}`,
      lastModified: new Date(),
      changeFrequency: 'monthly' as const,
      priority: Math.max(priority - 0.1, 0.5),
      alternates: { languages: { 'en-CA': `${base}/${path}`, 'fr-CA': `${base}/fr/${path}` } },
    },
  ]);
}
