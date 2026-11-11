'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import PageNav from '@/components/PageNav';
import PageFooter from '@/components/PageFooter';

const ease = [0.16, 1, 0.3, 1] as const;
const BUSINESS_EMAIL = 'support@agentedirect.ca';

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────
type Lang = 'en' | 'fr';
type T = { en: string; fr: string };

type IndustryId =
  | 'home-services' | 'automotive' | 'restaurants' | 'healthcare'
  | 'real-estate' | 'legal' | 'retail' | 'beauty' | 'fitness' | 'other' | 'pest-control';

interface Stat  { value: T; label: T; source: T; year: number; geo: T }
interface Point { title: T; body: T }

interface IndustryData {
  id: IndustryId;
  label: T;
  accentColor: string;
  hero: { headline: T; sub: T };
  stats: Stat[];
  challenges: [Point, Point, Point];
  opportunities: [Point, Point, Point];
  seo: T;
  ai: T;
  approach: { build: T; attract: T; convert: T };
  recommended: 'launch' | 'growth' | 'pro';
}

// ─────────────────────────────────────────────
// Industry data
// ─────────────────────────────────────────────
const INDUSTRIES: IndustryData[] = [
  {
    id: 'home-services',
    label: { en: 'Home Services & Contractors', fr: 'Services à domicile et entrepreneurs' },
    accentColor: '#F59E0B',
    hero: {
      headline: {
        en: 'Most homeowners Google a contractor before they ever pick up the phone.',
        fr: 'La plupart des propriétaires cherchent un entrepreneur sur Google avant de composer le numéro.',
      },
      sub: {
        en: 'If your business is not visible online in Montreal and Quebec, those jobs are going to someone else.',
        fr: 'Si votre entreprise n\'est pas visible en ligne à Montréal et au Québec, ces contrats vont chez vos concurrents.',
      },
    },
    stats: [
      {
        value: { en: '59%', fr: '59 %' },
        label: { en: 'of Canadians are more likely to do business with a company that has a website. Contractors and home service businesses without an online presence are invisible to their best potential clients', fr: 'des Canadiens sont plus susceptibles de faire affaire avec une entreprise qui a un site web. Les entrepreneurs sans présence en ligne sont invisibles pour leurs meilleurs clients potentiels' },
        source: { en: 'CIRA Canadian Internet Trends 2025', fr: 'Tendances Internet canadiennes ACEI 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '75%', fr: '75 %' },
        label: { en: 'of Canadian mortgage consumers plan to renovate their home within the next five years. The demand for quality home service work is growing, and businesses that are easy to find and contact online are positioned to capture it', fr: 'des consommateurs hypothécaires canadiens prévoient rénover leur maison au cours des cinq prochaines années. La demande de travaux de rénovation augmente, et les entreprises faciles à trouver et à contacter en ligne sont positionnées pour en profiter' },
        source: { en: 'CMHC 2025 Mortgage Consumer Survey', fr: 'Sondage auprès des consommateurs hypothécaires SCHL 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '9 in 10', fr: '9 sur 10' },
        label: { en: 'Canadian small businesses use at least one digital channel. But only 1 in 3 currently take orders or sell online, meaning contractors and home service businesses that offer online quoting or booking hold a clear competitive edge', fr: 'petites entreprises canadiennes utilisent au moins un canal numérique. Mais seulement 1 sur 3 acceptent des commandes ou vendent en ligne, ce qui donne aux entrepreneurs et prestataires de services offrant des soumissions ou réservations en ligne un avantage concurrentiel distinct' },
        source: { en: 'CFIB Your Voice Survey, September 2025', fr: 'Sondage Votre voix FCEI, septembre 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'Invisible on Google', fr: 'Invisible sur Google' }, body: { en: 'Without a site and Google Business Profile, you don\'t appear when homeowners search for your trade in your city. And they always search before they call.', fr: 'Sans site web ni profil Google, vous n\'apparaissez pas quand des propriétaires cherchent votre métier dans votre ville. Et ils cherchent toujours avant d\'appeler.' } },
      { title: { en: 'No way to capture leads after hours', fr: 'Pas de capture de prospects hors heures' }, body: { en: 'Word of mouth dries up at night and on weekends. Without a contact form or chatbot, potential clients move on to whoever answers.', fr: 'Le bouche-à-oreille s\'arrête le soir et les fins de semaine. Sans formulaire ou chatbot, vos clients potentiels passent au suivant.' } },
      { title: { en: 'No proof of your work', fr: 'Pas de preuve de votre travail' }, body: { en: 'Clients want to see before-and-after photos, real reviews and past projects before trusting you with their home. If they can\'t find that, they hesitate.', fr: 'Les clients veulent voir des photos avant-après, de vraies évaluations et vos anciens projets avant de vous confier leur maison. Sans ça, ils hésitent.' } },
    ],
    opportunities: [
      { title: { en: 'Rank first for local searches', fr: 'Apparaître en tête des recherches locales' }, body: { en: 'Pages targeting «électricien Montréal», «plombier Laval» or «entrepreneur paysagement Québec» capture high-intent clients who are ready to book.', fr: 'Des pages ciblant «électricien Montréal», «plombier Laval» ou «entrepreneur paysagement Québec» capturent des clients prêts à réserver.' } },
      { title: { en: 'Quote requests while you sleep', fr: 'Des demandes de soumission pendant que vous dormez' }, body: { en: 'An online quote form with automated confirmation means no lead goes cold. Even at midnight. Pair with an AI chatbot to answer common questions instantly.', fr: 'Un formulaire de soumission en ligne avec confirmation automatique évite qu\'un seul prospect se refroidisse, même à minuit. Jumelez-le à un chatbot IA.' } },
      { title: { en: 'Build trust with a project gallery', fr: 'Gagner la confiance avec une galerie de projets' }, body: { en: 'Real photos of your best work. Organized by project type. Are your most powerful sales tool. Combined with verified reviews, they convert browsers into callers.', fr: 'De vraies photos de vos meilleurs projets, organisées par type, sont votre outil de vente le plus puissant. Combinées à des avis vérifiés, elles transforment les visiteurs en appels.' } },
    ],
    seo: {
      en: 'City + trade keyword pages for every area you serve in greater Montreal and Quebec. Laval, Longueuil, Brossard, Repentigny, and beyond.',
      fr: 'Des pages par ville et par métier pour chaque secteur que vous couvrez dans le grand Montréal et au Québec. Laval, Longueuil, Brossard, Repentigny et plus.',
    },
    ai: {
      en: 'An AI chatbot trained on your services and service areas that qualifies leads 24/7, answers pricing questions and collects project details before you\'ve said a word.',
      fr: 'Un chatbot IA formé sur vos services et zones de couverture qui qualifie les prospects 24h/7j, répond aux questions de tarification et recueille les détails du projet avant même que vous parliez.',
    },
    approach: {
      build: {
        en: 'A site built around your trade, your service areas and your best project photos. So a homeowner in Laval or Brossard sees immediately that you serve their neighbourhood and do work they can trust.',
        fr: 'Un site construit autour de votre métier, vos zones de service et vos meilleures photos de projets. Pour qu\'un propriétaire à Laval ou Brossard voit immédiatement que vous couvrez son quartier.',
      },
      attract: {
        en: 'City and trade keyword pages. «électricien Montréal», «plombier Laval», «toiture Longueuil». So your phone rings from people who are actively looking, not just browsing.',
        fr: 'Des pages par ville et par métier. «électricien Montréal», «plombier Laval», «toiture Longueuil». Pour que votre téléphone sonne de la part de gens qui cherchent activement.',
      },
      convert: {
        en: 'An online quote form with auto-confirmation and an AI chatbot that captures project details and contact info around the clock. So no lead goes cold waiting for a callback.',
        fr: 'Un formulaire de soumission en ligne avec confirmation automatique et un chatbot IA qui capte les détails du projet et les coordonnées en tout temps. Pour qu\'aucun prospect ne se refroidisse.',
      },
    },
    recommended: 'growth',
  },
  {
    id: 'automotive',
    label: { en: 'Automotive', fr: 'Automobile' },
    accentColor: '#3B82F6',
    hero: {
      headline: {
        en: 'Before a car enters your shop, the owner has already Googled you.',
        fr: 'Avant qu\'une voiture entre dans votre atelier, son propriétaire vous a déjà cherché sur Google.',
      },
      sub: {
        en: 'Independent auto shops across Montreal and Quebec lose jobs every day to competitors with a stronger online presence. Not better skills.',
        fr: 'Des ateliers automobiles indépendants à Montréal et au Québec perdent des contrats chaque jour face à des concurrents mieux présents en ligne. Pas plus compétents.',
      },
    },
    stats: [
      {
        value: { en: '82%', fr: '82 %' },
        label: { en: 'of Canadian car shoppers used online sources for information when shopping for a vehicle in 2025. Independent shops without a strong online presence miss most of the market', fr: 'des acheteurs de voitures canadiens ont utilisé des sources en ligne pour s\'informer lors de leur recherche de véhicule en 2025' },
        source: { en: 'Trader Corporation / AutoTrader.ca 2025 Canadian Car Shopper Report', fr: 'Trader Corporation / AutoTrader.ca. Rapport sur les acheteurs de voitures canadiens 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '9 in 10', fr: '9 sur 10' },
        label: { en: 'Canadian consumers are interested in completing at least part of their vehicle purchase online. Shops offering online booking and estimates capture this demand directly', fr: 'consommateurs canadiens souhaitent compléter au moins une partie de leur achat de véhicule en ligne. Les ateliers qui offrent la prise de rendez-vous et les estimations en ligne captent directement cette demande' },
        source: { en: 'Trader Corporation / AutoTrader.ca 2025 Canadian Car Shopper Report', fr: 'Trader Corporation / AutoTrader.ca. Rapport sur les acheteurs de voitures canadiens 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '93%', fr: '93 %' },
        label: { en: 'of Canadian car shoppers use third-party automotive marketplaces at some stage of their purchase. Independent shops that are not listed and optimized online are invisible at the moment decisions get made', fr: 'des acheteurs de voitures canadiens utilisent des places de marché automobiles en ligne à un moment ou l\'autre de leur parcours d\'achat' },
        source: { en: 'Clarivoy / AutoTrader.ca Canadian Market Research', fr: 'Clarivoy / AutoTrader.ca. Recherche sur le marché canadien' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'Outgunned by dealership chains', fr: 'Écrasé par les chaînes de concessionnaires' }, body: { en: 'Big dealerships dominate local search results with dedicated SEO teams. Without a strategy, independent shops are buried on page two.', fr: 'Les grands concessionnaires dominent les résultats locaux avec des équipes SEO dédiées. Sans stratégie, les ateliers indépendants sont enfouis en page deux.' } },
      { title: { en: 'No online booking for service', fr: 'Pas de prise de rendez-vous en ligne' }, body: { en: 'Customers expect to book service appointments online the same way they book restaurants. Phone-only scheduling costs you jobs every week.', fr: 'Les clients s\'attendent à prendre rendez-vous en ligne comme ils réservent un restaurant. Un système téléphonique uniquement vous fait perdre des contrats chaque semaine.' } },
      { title: { en: 'Reviews left unmanaged', fr: 'Avis laissés sans gestion' }, body: { en: 'One unanswered negative review can cost you dozens of potential customers. Most shops have no system to generate or respond to reviews consistently.', fr: 'Un seul avis négatif sans réponse peut vous coûter des dizaines de clients potentiels. La plupart des ateliers n\'ont aucun système pour générer ou répondre aux avis.' } },
    ],
    opportunities: [
      { title: { en: 'Own «mécanicien Montréal» on Google', fr: 'Dominer «mécanicien Montréal» sur Google' }, body: { en: 'Dedicated pages for each service. Oil change, brake repair, tire installation, body work. Target the exact searches your future customers are typing.', fr: 'Des pages dédiées à chaque service. Vidange, freins, pneus, carrosserie. Ciblent exactement ce que vos futurs clients recherchent.' } },
      { title: { en: 'Online booking that fills your schedule', fr: 'Prise de rendez-vous en ligne qui remplit votre agenda' }, body: { en: 'An integrated service booking form with confirmation emails and reminders reduces no-shows and fills gaps in your schedule automatically.', fr: 'Un formulaire de réservation de service avec confirmations et rappels réduit les absences et remplit automatiquement les trous dans votre agenda.' } },
      { title: { en: 'Build a reputation that sells itself', fr: 'Construire une réputation qui se vend d\'elle-même' }, body: { en: 'An automated post-service review request system turns every satisfied customer into a five-star Google review. The most trusted marketing you can get.', fr: 'Un système automatisé de demande d\'avis après service transforme chaque client satisfait en avis cinq étoiles sur Google. Le marketing le plus crédible qui soit.' } },
    ],
    seo: {
      en: 'Service-specific pages for «réparation carrosserie Montréal», «changement pneus Laval», «mécanique générale Brossard» and your other key areas.',
      fr: 'Des pages spécifiques pour «réparation carrosserie Montréal», «changement pneus Laval», «mécanique générale Brossard» et vos autres zones clés.',
    },
    ai: {
      en: 'An AI chatbot that handles appointment scheduling, answers questions about pricing and turnaround times, and captures customer info outside business hours.',
      fr: 'Un chatbot IA qui gère la prise de rendez-vous, répond aux questions sur les prix et délais, et capte les informations des clients en dehors des heures d\'ouverture.',
    },
    approach: {
      build: {
        en: 'A site organized by service. Oil change, brakes, tires, bodywork. With real shop photos and your actual hours, so a car owner knows exactly what you do before they ever call.',
        fr: 'Un site organisé par service. Vidange, freins, pneus, carrosserie. Avec de vraies photos de l\'atelier et vos horaires réels, pour qu\'un propriétaire sache exactement ce que vous faites avant d\'appeler.',
      },
      attract: {
        en: 'Service and neighbourhood pages targeting «mécanicien Montréal», «changement pneus Laval», «réparation carrosserie Brossard». The searches people make when their car has a problem right now.',
        fr: 'Des pages par service et par secteur ciblant «mécanicien Montréal», «changement pneus Laval», «réparation carrosserie Brossard». Les recherches que font les gens quand leur voiture a un problème maintenant.',
      },
      convert: {
        en: 'An online service booking form and an AI chatbot that answers common questions about wait times and pricing after hours. So your schedule fills itself even when the shop is closed.',
        fr: 'Un formulaire de prise de rendez-vous en ligne et un chatbot IA qui répond aux questions courantes sur les délais et tarifs après les heures. Pour que votre agenda se remplisse même quand l\'atelier est fermé.',
      },
    },
    recommended: 'growth',
  },
  {
    id: 'restaurants',
    label: { en: 'Restaurants & Hospitality', fr: 'Restaurants et hôtellerie' },
    accentColor: '#EF4444',
    hero: {
      headline: {
        en: 'A hungry customer types three words into Google. Is your restaurant the answer?',
        fr: 'Un client affamé tape trois mots dans Google. Votre restaurant est-il la réponse?',
      },
      sub: {
        en: 'In Montreal\'s competitive restaurant scene, the businesses that dominate online get the tables. Regardless of which food is better.',
        fr: 'Dans la scène restaurant compétitive de Montréal, les établissements qui dominent en ligne obtiennent les réservations. Peu importe qui cuisine le mieux.',
      },
    },
    stats: [
      {
        value: { en: '64%', fr: '64 %' },
        label: { en: 'of Canadians ordered food delivery in the past six months. Gen Z at 79%. Digital ordering channels now reach most Canadian diners, and restaurants without an online presence miss this revenue entirely', fr: 'des Canadiens ont commandé de la livraison alimentaire au cours des six derniers mois. 79 % chez la génération Z. Les canaux de commande numérique rejoignent désormais la plupart des convives canadiens' },
        source: { en: 'DoorDash Canada 2025 Delivery Trends Report', fr: 'Rapport sur les tendances de livraison DoorDash Canada 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '56%', fr: '56 %' },
        label: { en: 'of Canadian consumers prefer ordering through third-party apps and websites over calling in or ordering in person. Restaurants without a digital ordering presence are leaving more than half the market to competitors', fr: 'des consommateurs canadiens préfèrent commander via des applications et sites Web tiers plutôt que par téléphone ou en personne' },
        source: { en: 'DoorDash Canada 2025 Delivery Trends Report', fr: 'Rapport sur les tendances de livraison DoorDash Canada 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '96%', fr: '96 %' },
        label: { en: 'of Canadian consumers have ordered food delivery specifically to satisfy a craving. Restaurants without a strong delivery and digital ordering presence are invisible at the moment customers are most ready to buy', fr: 'des consommateurs canadiens ont commandé de la livraison alimentaire spécifiquement pour satisfaire une envie. Les restaurants sans présence en livraison et commande numérique sont invisibles au moment précis où les clients sont le plus prêts à acheter' },
        source: { en: 'DoorDash Canada 2025 Delivery Trends Report (n\u202f=\u202f2,189 Canadians)', fr: 'Rapport sur les tendances de livraison DoorDash Canada 2025 (n\u202f=\u202f2\u202f189 Canadiens)' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'No reservations without calling', fr: 'Pas de réservation sans appeler' }, body: { en: 'Customers. Especially younger ones. Abandon restaurants that require a phone call to book. Online reservations are now a baseline expectation.', fr: 'Les clients. Surtout les plus jeunes. Abandonnent les restaurants qui exigent un appel pour réserver. Les réservations en ligne sont désormais une attente de base.' } },
      { title: { en: 'Outdated menus and no photos', fr: 'Menus dépassés et aucune photo' }, body: { en: 'A PDF menu from 2019 or a Facebook page with blurry photos signals neglect. Customers judge the food by the presentation of your digital presence.', fr: 'Un menu PDF de 2019 ou une page Facebook avec des photos floues signale de la négligence. Les clients jugent la nourriture à partir de votre présence numérique.' } },
      { title: { en: 'Zero visibility between visits', fr: 'Visibilité nulle entre les visites' }, body: { en: 'Without email, SMS or social integration, satisfied customers forget you exist between meals. Repeat business is the lifeblood of a restaurant.', fr: 'Sans courriel, SMS ou intégration aux réseaux sociaux, les clients satisfaits oublient votre existence entre les repas. La fidélisation est vitale pour un restaurant.' } },
    ],
    opportunities: [
      { title: { en: 'A menu that makes them hungry before they arrive', fr: 'Un menu qui donne faim avant même d\'arriver' }, body: { en: 'High-quality food photography combined with a clean digital menu increases conversion from Google discovery to actual reservation by a wide margin.', fr: 'De la photographie culinaire de qualité combinée à un menu numérique soigné augmente considérablement le taux de conversion de la découverte Google à la réservation réelle.' } },
      { title: { en: 'Online reservations and takeout orders', fr: 'Réservations en ligne et commandes à emporter' }, body: { en: 'An integrated booking system and optional online ordering removes friction entirely. Customers go from discovery to confirmation without picking up a phone.', fr: 'Un système de réservation intégré et une commande en ligne optionnelle éliminent toute friction. Les clients passent de la découverte à la confirmation sans décrocher le téléphone.' } },
      { title: { en: 'Review generation on autopilot', fr: 'Génération d\'avis en pilote automatique' }, body: { en: 'Automatically request a Google review after every visit. Restaurants in the top 10 on Google Maps in Montreal average over 200 reviews. Yours should too.', fr: 'Demandez automatiquement un avis Google après chaque visite. Les restaurants dans le top 10 de Google Maps à Montréal ont en moyenne plus de 200 avis. Les vôtres aussi.' } },
    ],
    seo: {
      en: '«Restaurant italien Montréal», «meilleur sushi Plateau», «brunch Rosemont». Neighbourhood-level search pages that put you in front of customers planning their next meal.',
      fr: '«Restaurant italien Montréal», «meilleur sushi Plateau», «brunch Rosemont». Des pages de recherche au niveau du quartier qui vous placent devant les clients qui planifient leur prochain repas.',
    },
    ai: {
      en: 'An AI chatbot that handles reservation requests, answers questions about allergens and specials, and collects contact info for your mailing list. Even at 2am.',
      fr: 'Un chatbot IA qui gère les demandes de réservation, répond aux questions sur les allergènes et les spéciaux, et recueille les coordonnées pour votre liste de diffusion. Même à 2h du matin.',
    },
    approach: {
      build: {
        en: 'A site with high-quality food photography, a clean digital menu, your hours and a reservation system. Everything a hungry customer needs to decide before they even arrive.',
        fr: 'Un site avec de la photographie culinaire de qualité, un menu numérique soigné, vos horaires et un système de réservation. Tout ce qu\'un client affamé a besoin pour décider avant même d\'arriver.',
      },
      attract: {
        en: 'Neighbourhood-level pages targeting «restaurant italien Plateau», «brunch Rosemont», «meilleur sushi NDG». Capturing customers who are actively planning their next meal right now.',
        fr: 'Des pages au niveau du quartier ciblant «restaurant italien Plateau», «brunch Rosemont», «meilleur sushi NDG». Pour capter les clients qui planifient leur prochain repas maintenant.',
      },
      convert: {
        en: 'Online reservations and a click-to-order system that lets a customer go from Google discovery to confirmed booking in under two minutes. Without picking up a phone.',
        fr: 'Des réservations en ligne et un système de commande qui permettent à un client de passer de la découverte Google à une réservation confirmée en moins de deux minutes. Sans décrocher le téléphone.',
      },
    },
    recommended: 'growth',
  },
  {
    id: 'healthcare',
    label: { en: 'Healthcare & Dental', fr: 'Santé et dentisterie' },
    accentColor: '#10B981',
    hero: {
      headline: {
        en: 'Patients choose their dentist or doctor the same way they choose everything else. By Googling first.',
        fr: 'Les patients choisissent leur dentiste ou médecin comme ils choisissent tout le reste. En cherchant sur Google d\'abord.',
      },
      sub: {
        en: 'Your clinical expertise means nothing to a new patient who finds a competitor first. In Quebec, the practices that dominate local search fill their schedules. The others wait.',
        fr: 'Votre expertise clinique ne signifie rien pour un nouveau patient qui trouve un concurrent en premier. Au Québec, les cliniques qui dominent la recherche locale remplissent leur agenda. Les autres attendent.',
      },
    },
    stats: [],
    challenges: [
      { title: { en: 'New patients can\'t find you', fr: 'Les nouveaux patients ne vous trouvent pas' }, body: { en: 'Without city-level SEO, your practice is invisible when someone searches «dentiste Montréal» or «médecin de famille Laval». Searches happening thousands of times a month.', fr: 'Sans SEO local, votre clinique est invisible quand quelqu\'un cherche «dentiste Montréal» ou «médecin de famille Laval». Des recherches qui surviennent des milliers de fois par mois.' } },
      { title: { en: 'No online appointment requests', fr: 'Pas de demandes de rendez-vous en ligne' }, body: { en: 'Patients who can\'t book online at 10pm move on to the next practice on the list. Phone-only scheduling is one of the top reasons potential patients don\'t follow through.', fr: 'Les patients qui ne peuvent pas prendre rendez-vous en ligne à 22h passent à la prochaine clinique. Un système téléphonique uniquement est l\'une des principales raisons pour lesquelles les patients potentiels renoncent.' } },
      { title: { en: 'No trust-building before the first visit', fr: 'Aucune mise en confiance avant la première visite' }, body: { en: 'An outdated or generic website fails to communicate your team, your approach and your technology. Patients want to feel comfortable before they even walk in the door.', fr: 'Un site dépassé ou générique ne communique pas votre équipe, votre approche et votre technologie. Les patients veulent se sentir à l\'aise avant même de franchir la porte.' } },
    ],
    opportunities: [
      { title: { en: 'Appointment requests around the clock', fr: 'Demandes de rendez-vous en tout temps' }, body: { en: 'An online request form with auto-confirmation and a chatbot that answers common pre-appointment questions removes the biggest barrier to new patient acquisition.', fr: 'Un formulaire de demande en ligne avec confirmation automatique et un chatbot qui répond aux questions pré-rendez-vous courants élimine le plus grand obstacle à l\'acquisition de nouveaux patients.' } },
      { title: { en: 'Build trust before the consultation', fr: 'Établir la confiance avant la consultation' }, body: { en: 'Team bios, before-and-after galleries (dental), technology pages and real patient testimonials make the decision easy for someone choosing between two practices.', fr: 'Les biographies d\'équipe, les galeries avant-après (dentaire), les pages sur la technologie et les témoignages de vrais patients facilitent la décision pour quelqu\'un qui choisit entre deux cliniques.' } },
      { title: { en: 'Rank for every service you offer', fr: 'Se positionner pour chaque service que vous offrez' }, body: { en: 'Dedicated pages for implants, whitening, Invisalign, family dentistry or general practice create multiple doors for patients to find you. Especially those who know exactly what they need.', fr: 'Des pages dédiées aux implants, blanchiment, Invisalign, dentisterie familiale ou médecine générale créent plusieurs portes d\'entrée pour les patients. Surtout ceux qui savent exactement ce qu\'ils veulent.' } },
    ],
    seo: {
      en: '«Dentiste Montréal», «clinique dentaire Laval», «implants dentaires Québec». Service-specific pages that rank for the high-intent searches patients make when they\'re ready to book.',
      fr: '«Dentiste Montréal», «clinique dentaire Laval», «implants dentaires Québec». Des pages spécifiques par service qui se positionnent sur les recherches à forte intention quand les patients sont prêts à réserver.',
    },
    ai: {
      en: 'A Loi 25–compliant AI chatbot that answers common questions about services, fees, insurance and availability. And collects contact info for follow-up. Without collecting or storing personal health details.',
      fr: 'Un chatbot IA conforme à la Loi 25 qui répond aux questions courantes sur les services, tarifs, assurances et disponibilités. Et recueille les coordonnées. Sans collecter ni stocker de renseignements personnels sur la santé.',
    },
    approach: {
      build: {
        en: 'A professional clinic website with team bios, service pages for every treatment you offer, real patient testimonials and clear before-and-after galleries. Built to earn trust before the first appointment.',
        fr: 'Un site de clinique professionnel avec les biographies de votre équipe, des pages pour chaque traitement, de vrais témoignages de patients et des galeries avant-après claires. Conçu pour établir la confiance avant le premier rendez-vous.',
      },
      attract: {
        en: 'Service-specific pages for every treatment. Implants, whitening, Invisalign, family dentistry. Targeting the searches patients make when they know exactly what they need and are ready to book.',
        fr: 'Des pages spécifiques par traitement. Implants, blanchiment, Invisalign, dentisterie familiale. Ciblant les recherches que font les patients quand ils savent exactement ce dont ils ont besoin et sont prêts à réserver.',
      },
      convert: {
        en: 'An online appointment request form available 24/7 and a Loi 25–compliant AI chatbot that answers pre-visit questions and captures contact info. So your front desk handles fewer calls and more confirmed appointments.',
        fr: 'Un formulaire de demande de rendez-vous disponible 24h/7j et un chatbot IA conforme à la Loi 25 qui répond aux questions pré-visite et capte les coordonnées. Pour que votre réception gère moins d\'appels et plus de rendez-vous confirmés.',
      },
    },
    recommended: 'pro',
  },
  {
    id: 'real-estate',
    label: { en: 'Real Estate', fr: 'Immobilier' },
    accentColor: '#6366F1',
    hero: {
      headline: {
        en: 'Every buyer and tenant in Quebec starts their search online. Where do they find you?',
        fr: 'Chaque acheteur et locataire au Québec commence sa recherche en ligne. Où vous trouvent-ils?',
      },
      sub: {
        en: 'In a market where trust and local expertise close deals, your online presence is your first showing. And most agents are still treating it as an afterthought.',
        fr: 'Dans un marché où la confiance et l\'expertise locale concluent les transactions, votre présence en ligne est votre première visite. Et la plupart des agents la traitent encore comme une réflexion après coup.',
      },
    },
    stats: [
      {
        value: { en: '77%', fr: '77 %' },
        label: { en: 'of Canadian mortgage consumers conducted research online in 2025. Buyers and sellers who do not find your name during that research phase very often never reach out', fr: 'des consommateurs hypothécaires canadiens ont effectué des recherches en ligne en 2025. Les acheteurs et vendeurs qui ne trouvent pas votre nom pendant cette phase ne prennent souvent jamais contact' },
        source: { en: 'CMHC 2025 Mortgage Consumer Survey', fr: 'Sondage auprès des consommateurs hypothécaires SCHL 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '61%', fr: '61 %' },
        label: { en: 'of Canadians say having a website makes a business or organization more credible. Real estate agents without a professional website are perceived as less credible by most potential clients before the first conversation', fr: 'des Canadiens affirment qu\'un site web rend une entreprise plus crédible. Les agents immobiliers sans site professionnel sont perçus comme moins crédibles par la majorité de leurs clients potentiels avant même la première conversation' },
        source: { en: 'CIRA Canadian Internet Trends 2025', fr: 'Tendances Internet canadiennes ACEI 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'Competing on MLS alone', fr: 'Se limiter à MLS' }, body: { en: 'An MLS listing is a commodity. Without a personal website that tells your story, showcases your results and captures leads, you\'re invisible between listings.', fr: 'Une inscription MLS est banale. Sans site personnel qui raconte votre histoire, présente vos résultats et capte des prospects, vous êtes invisible entre les inscriptions.' } },
      { title: { en: 'No personal brand online', fr: 'Aucune marque personnelle en ligne' }, body: { en: 'Buyers and sellers choose the agent they trust. Without a site that communicates your expertise, your neighbourhood knowledge and your track record, you\'re just a photo on a listing.', fr: 'Les acheteurs et vendeurs choisissent l\'agent en qui ils ont confiance. Sans site qui communique votre expertise, votre connaissance du quartier et vos résultats, vous n\'êtes qu\'une photo sur une inscription.' } },
      { title: { en: 'No lead capture between transactions', fr: 'Pas de capture de prospects entre les transactions' }, body: { en: 'Most real estate websites have no mechanism to stay in front of potential clients who aren\'t quite ready yet. Nurturing those relationships is where most deals actually come from.', fr: 'La plupart des sites immobiliers n\'ont aucun mécanisme pour rester devant les clients potentiels qui ne sont pas encore tout à fait prêts. C\'est pourtant là que naissent la plupart des transactions.' } },
    ],
    opportunities: [
      { title: { en: 'Build a local expert brand', fr: 'Construire une marque d\'expert local' }, body: { en: 'Neighbourhood guides, market updates, sold listings and client testimonials establish you as the go-to agent in your area. Turning Google searches into relationship conversations.', fr: 'Des guides de quartier, des mises à jour du marché, des ventes conclues et des témoignages de clients font de vous l\'agent incontournable dans votre secteur.' } },
      { title: { en: 'Lead capture that works while you sell', fr: 'Capture de prospects qui fonctionne pendant que vous vendez' }, body: { en: 'A contact form, home valuation request and AI chatbot mean potential clients can reach you at any time. And you have a system to follow up with everyone automatically.', fr: 'Un formulaire de contact, une demande d\'évaluation de propriété et un chatbot IA permettent aux clients potentiels de vous rejoindre à tout moment. Avec un suivi automatique.' } },
      { title: { en: 'Bilingual reach across Montreal', fr: 'Portée bilingue à travers Montréal' }, body: { en: 'A bilingual website means every Anglophone in Westmount and every Francophone in Rosemont can engage with you in the language they\'re most comfortable in. Doubling your addressable market.', fr: 'Un site bilingue signifie que chaque anglophone à Westmount et chaque francophone à Rosemont peut interagir avec vous dans sa langue. Ce qui double votre marché potentiel.' } },
    ],
    seo: {
      en: '«Agent immobilier Montréal», «courtier immobilier Laval», «vendre maison Rive-Sud». Market-specific pages that rank for searches made by motivated buyers and sellers.',
      fr: '«Agent immobilier Montréal», «courtier immobilier Laval», «vendre maison Rive-Sud». Des pages spécifiques au marché qui se positionnent sur les recherches d\'acheteurs et vendeurs motivés.',
    },
    ai: {
      en: 'An AI assistant that qualifies leads by budget, timeline and property type, answers questions about the buying process and schedules calls with you. 24/7.',
      fr: 'Un assistant IA qui qualifie les prospects par budget, échéancier et type de propriété, répond aux questions sur le processus d\'achat et planifie des appels avec vous. 24h/7j.',
    },
    approach: {
      build: {
        en: 'A personal agent website with a sold listings gallery, neighbourhood guides, client testimonials and your market stats. So a buyer or seller arriving from Google immediately sees your track record, not just your photo.',
        fr: 'Un site d\'agent personnel avec une galerie de ventes conclues, des guides de quartier, des témoignages de clients et vos statistiques de marché. Pour qu\'un acheteur ou vendeur arrivant de Google voie immédiatement votre bilan.',
      },
      attract: {
        en: 'Pages targeting «agent immobilier Montréal», «courtier Laval», «vendre maison Rive-Sud». Market searches made by people who are actively buying or selling, not just browsing.',
        fr: 'Des pages ciblant «agent immobilier Montréal», «courtier Laval», «vendre maison Rive-Sud». Les recherches de marché faites par des gens qui achètent ou vendent activement.',
      },
      convert: {
        en: 'A home valuation request form, a buyer consultation form and an AI assistant that qualifies leads by budget and timeline after hours. So every inquiry lands in your inbox ready to follow up.',
        fr: 'Un formulaire de demande d\'évaluation, un formulaire de consultation acheteur et un assistant IA qui qualifie les prospects par budget et échéancier après les heures. Pour que chaque demande arrive dans votre boîte de réception prête à être suivie.',
      },
    },
    recommended: 'pro',
  },
  {
    id: 'legal',
    label: { en: 'Legal & Professional Services', fr: 'Services juridiques et professionnels' },
    accentColor: '#8B5CF6',
    hero: {
      headline: {
        en: 'Clients don\'t choose the best lawyer. They choose the most credible one they can find online.',
        fr: 'Les clients ne choisissent pas le meilleur avocat. Ils choisissent le plus crédible qu\'ils peuvent trouver en ligne.',
      },
      sub: {
        en: 'In Quebec\'s legal and professional market, digital credibility. Not reputation alone. Determines who gets the consultation request.',
        fr: 'Dans le marché juridique et professionnel du Québec, la crédibilité numérique. Pas seulement la réputation. Détermine qui reçoit la demande de consultation.',
      },
    },
    stats: [
      {
        value: { en: '50%', fr: '50 %' },
        label: { en: 'of future legal clients have already turned to AI for legal help before contacting a lawyer. Meaning they begin their search online and have already formed an opinion of your firm before they ever call', fr: 'des futurs clients en droit ont déjà eu recours à l\'IA pour obtenir de l\'aide juridique avant de contacter un avocat. Ils commencent leur recherche en ligne et ont déjà une opinion de votre cabinet avant d\'appeler' },
        source: { en: 'Clio 2025 Legal Trends Report', fr: 'Rapport Clio 2025 sur les tendances du droit' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '35%', fr: '35 %' },
        label: { en: 'of Canadian legal clients prefer online payments. The highest rate globally. Signalling that legal clients in Canada expect their entire interaction with a law firm to happen digitally, from first search to final invoice', fr: 'des clients juridiques canadiens préfèrent les paiements en ligne. Le taux le plus élevé au monde. Ce qui signale que les clients canadiens s\'attendent à ce que l\'ensemble de leur interaction avec un cabinet se déroule en ligne' },
        source: { en: 'Clio 2025 Legal Trends Report (Canada)', fr: 'Rapport Clio 2025 sur les tendances du droit (Canada)' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'Generic directory listings', fr: 'Inscriptions dans des répertoires génériques' }, body: { en: 'Most Quebec law firms and professional practices rely on Juris.ca, LinkedIn or basic directory listings. Without a website that speaks directly to the client\'s situation.', fr: 'La plupart des cabinets juridiques et professionnels québécois s\'appuient sur des répertoires. Sans site qui parle directement à la situation du client.' } },
      { title: { en: 'No content that builds authority', fr: 'Aucun contenu qui bâtit l\'autorité' }, body: { en: 'Without articles, guides or FAQs that demonstrate expertise, clients have no reason to prefer you over another firm. Especially when they\'re comparing multiple options.', fr: 'Sans articles, guides ou FAQ qui démontrent l\'expertise, les clients n\'ont aucune raison de vous préférer à un autre cabinet. Surtout quand ils comparent plusieurs options.' } },
      { title: { en: 'The bilingual gap', fr: 'Le fossé bilingue' }, body: { en: 'Montreal clients expect to be served in their language. A French-only or English-only site immediately excludes a significant portion of your potential client base.', fr: 'Les clients montréalais s\'attendent à être servis dans leur langue. Un site uniquement français ou uniquement anglais exclut d\'emblée une part importante de votre clientèle potentielle.' } },
    ],
    opportunities: [
      { title: { en: 'Practice area pages that rank and convert', fr: 'Pages par domaine de pratique qui se positionnent et convertissent' }, body: { en: 'Dedicated pages for each practice area. Family law, business law, real estate, immigration. Target the exact searches clients make when they have a problem and need help now.', fr: 'Des pages dédiées à chaque domaine. Droit de la famille, droit des affaires, immobilier, immigration. Ciblent les recherches exactes des clients qui ont un problème et ont besoin d\'aide maintenant.' } },
      { title: { en: 'An intake system that qualifies before you consult', fr: 'Un système d\'accueil qui qualifie avant la consultation' }, body: { en: 'An online intake form or AI pre-qualification chatbot gathers case details before the consultation. Saving your time and setting the right expectations for the client.', fr: 'Un formulaire d\'accueil ou un chatbot IA de préqualification recueille les détails du dossier avant la consultation. Économisant votre temps et définissant les bonnes attentes.' } },
      { title: { en: 'Bilingual authority content', fr: 'Contenu d\'autorité bilingue' }, body: { en: 'Educational guides, legal explainers and FAQ content in both French and English establish you as the go-to resource in your practice area. Before a client even needs you.', fr: 'Des guides éducatifs, des explications juridiques et du contenu FAQ en français et en anglais vous établissent comme la référence dans votre domaine. Avant même qu\'un client ait besoin de vous.' } },
    ],
    seo: {
      en: '«Avocat droit de la famille Montréal», «notaire Québec», «comptable Laval». Practice-area pages that capture the searches made by people who need help now.',
      fr: '«Avocat droit de la famille Montréal», «notaire Québec», «comptable Laval». Des pages par domaine qui captent les recherches de personnes qui ont besoin d\'aide maintenant.',
    },
    ai: {
      en: 'A pre-qualification chatbot that gathers case type, timeline and contact info before you\'ve spoken to a single person. So every consultation is worth your time.',
      fr: 'Un chatbot de préqualification qui recueille le type de dossier, l\'échéancier et les coordonnées avant que vous ayez parlé à qui que ce soit. Pour que chaque consultation vaille votre temps.',
    },
    approach: {
      build: {
        en: 'A bilingual authority site with dedicated pages for every practice area, attorney bios, educational guides and real client results. Built to establish credibility before the first contact.',
        fr: 'Un site d\'autorité bilingue avec des pages dédiées par domaine de pratique, des biographies d\'avocats, des guides éducatifs et de vrais résultats clients. Conçu pour établir la crédibilité avant le premier contact.',
      },
      attract: {
        en: 'Practice area pages targeting the searches clients make in a moment of need. «avocat droit de la famille Montréal», «notaire Laval», «comptable CPA Québec». In both English and French.',
        fr: 'Des pages par domaine ciblant les recherches que font les clients dans un moment de besoin. «avocat droit de la famille Montréal», «notaire Laval», «comptable CPA Québec». En français et en anglais.',
      },
      convert: {
        en: 'An online case intake form and a pre-qualification chatbot that gathers the key details before the consultation. So your time is spent on qualified cases, not screening calls.',
        fr: 'Un formulaire d\'accueil en ligne et un chatbot de préqualification qui recueille les détails clés avant la consultation. Pour que votre temps soit consacré à des dossiers qualifiés, pas à des appels de tri.',
      },
    },
    recommended: 'pro',
  },
  {
    id: 'retail',
    label: { en: 'Retail & E-commerce', fr: 'Commerce de détail et e-commerce' },
    accentColor: '#F97316',
    hero: {
      headline: {
        en: 'Your neighbourhood store has an advantage Amazon can never buy. You just need to make it visible.',
        fr: 'Votre commerce de quartier a un avantage qu\'Amazon ne peut jamais acheter. Il faut juste le rendre visible.',
      },
      sub: {
        en: 'Local retail in Quebec is thriving for businesses that lead with community, convenience and character. And have a digital presence that reflects that.',
        fr: 'Le commerce local au Québec prospère pour les entreprises qui mettent de l\'avant la communauté, la commodité et le caractère. Et qui ont une présence numérique qui le reflète.',
      },
    },
    stats: [
      {
        value: { en: '86%', fr: '86 %' },
        label: { en: 'of Canadians made an online purchase in the past year. And that share continues to grow as consumers expect to find any local retailer online before deciding where to shop', fr: 'des Canadiens ont effectué un achat en ligne au cours de la dernière année. Et cette proportion continue de croître à mesure que les consommateurs s\'attendent à trouver tout commerce local en ligne avant de décider où magasiner' },
        source: { en: 'CIRA Canadian Internet Trends 2026', fr: 'Tendances Internet canadiennes ACEI 2026' },
        year: 2026, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '48%', fr: '48 %' },
        label: { en: 'of Canadian shoppers use retailer websites as a research tool before purchasing. Another 45% use search engines. Meaning most buying decisions involve online research first, even when the final purchase is in-store', fr: 'des acheteurs canadiens utilisent les sites Web des détaillants comme outil de recherche avant d\'acheter. 45 % supplémentaires utilisent les moteurs de recherche. La plupart des décisions d\'achat impliquent d\'abord une recherche en ligne' },
        source: { en: 'Retail Council of Canada / Leger Canadian Shopper Study 2026', fr: 'Conseil canadien du commerce de détail / Étude Leger sur les consommateurs canadiens 2026' },
        year: 2026, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '7.0%', fr: '7,0 %' },
        label: { en: 'of Canadian retail is now e-commerce. C$59.2 billion in annual online sales in 2025. The share of spending happening online continues to grow year over year, and local retailers without a digital presence are missing that revenue entirely', fr: 'du commerce de détail canadien est désormais du commerce en ligne. 59,2 G$ de ventes en ligne annuelles en 2025. La part des dépenses en ligne continue de croître d\'année en année, et les détaillants locaux sans présence numérique ratent entièrement ces revenus' },
        source: { en: 'Statistics Canada Monthly Retail Trade Survey, 2025 annual (released September 2026)', fr: 'Enquête mensuelle sur le commerce de détail. Statistique Canada, annuel 2025 (publié septembre 2026)' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'Competing with big-box digital spend', fr: 'Rivaliser avec le budget numérique des grandes surfaces' }, body: { en: 'You can\'t outspend Canadian Tire or Amazon on ads. But you can outrank them on local searches. If your site is built for it.', fr: 'Vous ne pouvez pas surpasser Canadian Tire ou Amazon sur les publicités. Mais vous pouvez les devancer sur les recherches locales. Si votre site est conçu pour ça.' } },
      { title: { en: 'No online product visibility', fr: 'Pas de visibilité des produits en ligne' }, body: { en: 'If your inventory or product categories aren\'t visible on Google, customers assume you don\'t carry what they need and order elsewhere.', fr: 'Si votre inventaire ou vos catégories de produits ne sont pas visibles sur Google, les clients supposent que vous n\'avez pas ce qu\'il leur faut et commandent ailleurs.' } },
      { title: { en: 'No loyalty or repeat purchase system', fr: 'Aucun système de fidélité ou de rachat' }, body: { en: 'Without email capture and a system to stay in front of past customers, you\'re constantly spending to acquire new ones when your best customers are already in your store.', fr: 'Sans capture de courriels et système pour rester devant vos anciens clients, vous dépensez constamment pour en acquérir de nouveaux alors que vos meilleurs clients ont déjà magasiné chez vous.' } },
    ],
    opportunities: [
      { title: { en: 'The local edge. Make it your brand', fr: 'L\'avantage local. Faites-en votre marque' }, body: { en: 'Neighbourhood-specific content, local team bios, community involvement and curated product stories communicate what no national chain ever can: you belong here.', fr: 'Du contenu spécifique au quartier, des biographies de l\'équipe locale, l\'implication communautaire et des histoires de produits curatées communiquent ce qu\'aucune chaîne nationale ne peut: vous appartenez ici.' } },
      { title: { en: 'Online catalogue and in-store pickup', fr: 'Catalogue en ligne et ramassage en magasin' }, body: { en: 'Even without full e-commerce, a browsable product catalogue with «available in store» messaging drives foot traffic from online research. Capturing the customer before the competitor does.', fr: 'Même sans e-commerce complet, un catalogue de produits navigable avec un message «disponible en magasin» génère du trafic en magasin à partir des recherches en ligne.' } },
      { title: { en: 'Email and SMS for repeat business', fr: 'Courriel et SMS pour la fidélisation' }, body: { en: 'An email capture system turns every first-time buyer into a contact you can reach for promotions, new arrivals and events. Dramatically reducing customer acquisition cost over time.', fr: 'Un système de capture de courriels transforme chaque premier achat en contact que vous pouvez rejoindre pour les promotions, nouveautés et événements.' } },
    ],
    seo: {
      en: 'Product category and neighbourhood pages that rank for «boutique Plateau Mont-Royal», «quincaillerie Rosemont», «fleuriste Outremont». The local searches your best customers are making.',
      fr: 'Des pages par catégorie de produits et par quartier pour «boutique Plateau Mont-Royal», «quincaillerie Rosemont», «fleuriste Outremont». Les recherches locales de vos meilleurs clients.',
    },
    ai: {
      en: 'An AI chatbot that answers product questions, checks availability, recommends items and collects email addresses. Turning casual browsers into registered customers.',
      fr: 'Un chatbot IA qui répond aux questions sur les produits, vérifie les disponibilités, recommande des articles et recueille les adresses courriel. Transformant les visiteurs occasionnels en clients enregistrés.',
    },
    approach: {
      build: {
        en: 'A local store website with a browsable product catalogue, your hours, your team, and the neighbourhood story that makes you different. So a customer who finds you on Google sees immediately why you beat a chain.',
        fr: 'Un site de commerce local avec un catalogue de produits navigable, vos horaires, votre équipe et l\'histoire de quartier qui vous distingue. Pour qu\'un client qui vous trouve sur Google voit immédiatement pourquoi vous battez une chaîne.',
      },
      attract: {
        en: 'Product category and neighbourhood pages targeting «boutique Plateau Mont-Royal», «épicerie fine Rosemont», «fleuriste Outremont». The local searches your best customers are making.',
        fr: 'Des pages par catégorie de produits et par quartier ciblant «boutique Plateau Mont-Royal», «épicerie fine Rosemont», «fleuriste Outremont». Les recherches locales de vos meilleurs clients.',
      },
      convert: {
        en: 'An email capture system, an AI chatbot that answers product questions and helps customers find what they need, and an in-store pickup option. Turning online discovery into foot traffic.',
        fr: 'Un système de capture de courriels, un chatbot IA qui répond aux questions sur les produits et aide les clients à trouver ce dont ils ont besoin, et une option de retrait en magasin. Pour transformer la découverte en ligne en trafic en magasin.',
      },
    },
    recommended: 'growth',
  },
  {
    id: 'beauty',
    label: { en: 'Beauty & Wellness', fr: 'Beauté et bien-être' },
    accentColor: '#EC4899',
    hero: {
      headline: {
        en: 'Your salon\'s best advertisement is your work. Your website is where people find it.',
        fr: 'La meilleure publicité de votre salon, c\'est votre travail. Votre site web est l\'endroit où les gens le trouvent.',
      },
      sub: {
        en: 'Beauty and wellness businesses in Montreal live and die by discovery, trust and seamless booking. All three start online.',
        fr: 'Les entreprises de beauté et bien-être à Montréal vivent et meurent par la découverte, la confiance et la réservation fluide. Les trois commencent en ligne.',
      },
    },
    stats: [],
    challenges: [
      { title: { en: 'Instagram alone is not a business', fr: 'Instagram seul n\'est pas une entreprise' }, body: { en: 'Instagram is essential. But it\'s a platform you don\'t own, can\'t rank on Google from, and can\'t take bookings through reliably. Your website is the foundation it must sit on.', fr: 'Instagram est essentiel. Mais c\'est une plateforme que vous ne possédez pas, sur laquelle vous ne pouvez pas vous positionner sur Google, et par laquelle vous ne pouvez pas prendre des réservations de façon fiable.' } },
      { title: { en: 'No direct booking system', fr: 'Aucun système de réservation directe' }, body: { en: 'Phone-only or DM-only booking creates friction, misses night-time customers and makes you look less professional than the competitor across the street.', fr: 'La réservation uniquement par téléphone ou messagerie crée des frictions, rate les clients du soir et vous fait paraître moins professionnel que le concurrent d\'en face.' } },
      { title: { en: 'Work showcased in a scroll that disappears', fr: 'Travail présenté dans un fil qui disparaît' }, body: { en: 'Instagram content has a lifespan of 48 hours. A portfolio website keeps your best work permanently visible, searchable and always accessible to future clients.', fr: 'Le contenu Instagram a une durée de vie de 48 heures. Un site portfolio garde votre meilleur travail visible en permanence, consultable et toujours accessible aux futurs clients.' } },
    ],
    opportunities: [
      { title: { en: 'A portfolio that keeps working for you', fr: 'Un portfolio qui travaille pour vous' }, body: { en: 'A permanent gallery of your best before-and-afters. Organized by service type. Is your most powerful sales tool, available 24/7 without you lifting a finger.', fr: 'Une galerie permanente de vos meilleurs avant-après. Organisée par type de service. Est votre outil de vente le plus puissant, disponible 24h/7j sans que vous ayez à intervenir.' } },
      { title: { en: 'Book while they\'re still excited', fr: 'Réserver pendant qu\'ils sont encore enthousiastes' }, body: { en: 'An integrated booking system means someone who discovers your work at 11pm can lock in an appointment immediately. Before they forget, or find someone else.', fr: 'Un système de réservation intégré signifie que quelqu\'un qui découvre votre travail à 23h peut immédiatement prendre rendez-vous. Avant d\'oublier ou de trouver quelqu\'un d\'autre.' } },
      { title: { en: 'Google discovery beyond «near me»', fr: 'Découverte Google au-delà de «près de moi»' }, body: { en: 'Pages targeting «balayage Montréal», «microblading Laval», «spa facial NDG» capture clients in research mode. Before they\'ve committed to anyone.', fr: 'Des pages ciblant «balayage Montréal», «microblading Laval», «spa facial NDG» capturent les clients en mode recherche. Avant qu\'ils se soient engagés auprès de quelqu\'un.' } },
    ],
    seo: {
      en: 'Service and neighbourhood pages for every treatment you offer. «coiffure Rosemont», «ongles gel Montréal», «massage thérapeutique Plateau». Capturing clients before they scroll past.',
      fr: 'Des pages par service et par quartier pour chaque traitement que vous offrez. «coiffure Rosemont», «ongles gel Montréal», «massage thérapeutique Plateau». Capturant les clients avant qu\'ils ne défilent.',
    },
    ai: {
      en: 'An AI chatbot that answers questions about services, pricing and availability, and collects booking requests when you\'re with a client. So no inquiry goes unanswered.',
      fr: 'Un chatbot IA qui répond aux questions sur les services, tarifs et disponibilités, et recueille les demandes de réservation quand vous êtes avec un client. Pour qu\'aucune demande ne reste sans réponse.',
    },
    approach: {
      build: {
        en: 'A portfolio site with a permanent before-and-after gallery organized by service, your stylist bios and real client reviews. So someone who finds you at 11pm on Instagram can explore your full work before deciding.',
        fr: 'Un site portfolio avec une galerie avant-après permanente organisée par service, les biographies de vos stylistes et de vrais avis clients. Pour que quelqu\'un qui vous trouve à 23h sur Instagram puisse explorer votre travail complet avant de décider.',
      },
      attract: {
        en: 'Service and neighbourhood pages targeting «balayage Montréal», «microblading Laval», «spa facial NDG». Capturing clients who are actively researching treatments before they commit to anyone.',
        fr: 'Des pages par service et par quartier ciblant «balayage Montréal», «microblading Laval», «spa facial NDG». Pour capter les clients qui font activement des recherches sur les traitements avant de s\'engager.',
      },
      convert: {
        en: 'An integrated direct booking system that lets a client go from discovering your work to locking in an appointment in one flow. Without a DM, a phone call or a waitlist form.',
        fr: 'Un système de réservation directe intégré qui permet à un client de passer de la découverte de votre travail à la confirmation d\'un rendez-vous en un seul parcours. Sans DM, appel téléphonique ni formulaire de liste d\'attente.',
      },
    },
    recommended: 'growth',
  },
  {
    id: 'fitness',
    label: { en: 'Fitness & Recreation', fr: 'Forme physique et récréation' },
    accentColor: '#14B8A6',
    hero: {
      headline: {
        en: 'People decide where they train before they ever step inside. Your website is that decision.',
        fr: 'Les gens décident où ils s\'entraînent avant d\'y mettre les pieds. Votre site web influence cette décision.',
      },
      sub: {
        en: 'Gyms, studios and recreation centres across Quebec compete on community and culture. And it has to be communicated online before the first visit.',
        fr: 'Les gyms, studios et centres de loisirs au Québec se démarquent par la communauté et la culture. Et cela doit être communiqué en ligne avant la première visite.',
      },
    },
    stats: [
      {
        value: { en: '9.4%', fr: '9,4 %' },
        label: { en: 'growth in Canadian fitness and recreation sector operating revenue in 2025. Reaching $6.1 billion. The market is expanding and studios that are easy to find and book online are best positioned to capture that growth', fr: 'de croissance des revenus d\'exploitation du secteur canadien de la remise en forme et des loisirs en 2025. Atteignant 6,1 G$. Le marché est en expansion et les studios faciles à trouver et à réserver en ligne sont les mieux positionnés pour en profiter' },
        source: { en: 'Statistics Canada Amusement and Recreation Industries Survey, 2025', fr: 'Enquête sur les industries du divertissement et des loisirs. Statistique Canada, 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '59%', fr: '59 %' },
        label: { en: 'of Canadians are more likely to do business with a company that has a website. For gyms and fitness studios competing in a crowded local market, a credible digital presence is the first filter prospective members apply', fr: 'des Canadiens sont plus susceptibles de faire affaire avec une entreprise qui a un site web. Pour les gyms et studios de fitness en marché local concurrentiel, une présence numérique crédible est le premier filtre que les membres potentiels appliquent' },
        source: { en: 'CIRA Canadian Internet Trends 2025', fr: 'Tendances Internet canadiennes ACEI 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'No class schedule or booking online', fr: 'Pas d\'horaire de cours ni de réservation en ligne' }, body: { en: 'If a potential member can\'t see your class schedule and book a trial class from their phone at 10pm, they\'ll find a competitor who lets them. First visit conversion starts with zero friction.', fr: 'Si un membre potentiel ne peut pas voir votre horaire de cours et réserver une première séance depuis son téléphone à 22h, il trouvera un concurrent qui le permet.' } },
      { title: { en: 'Competing on price alone', fr: 'Se concurrencer uniquement sur les prix' }, body: { en: 'Without a website that communicates your community, your coaches and your culture, the only differentiator left is monthly fee. A race to the bottom you can\'t win against big-box gyms.', fr: 'Sans site qui communique votre communauté, vos entraîneurs et votre culture, le seul différenciateur restant est le tarif mensuel. Une course vers le bas que vous ne pouvez pas gagner contre les grands centres.' } },
      { title: { en: 'No retention tools between visits', fr: 'Aucun outil de fidélisation entre les visites' }, body: { en: 'Members who don\'t feel connected drift away. Without an email list, challenge campaigns or class content that keeps your brand top-of-mind, churn is constant.', fr: 'Les membres qui ne se sentent pas connectés s\'éloignent. Sans liste de courriels, campagnes de défis ou contenu qui garde votre marque en tête, le taux de désistement est constant.' } },
    ],
    opportunities: [
      { title: { en: 'Class schedules that sell memberships', fr: 'Des horaires de cours qui vendent des abonnements' }, body: { en: 'A live class schedule with direct booking removes the biggest barrier to trial. Letting someone go from curiosity to confirmed spot without picking up a phone.', fr: 'Un horaire de cours en temps réel avec réservation directe élimine le plus grand obstacle à l\'essai. Permettant à quelqu\'un de passer de la curiosité à un place confirmée sans décrocher le téléphone.' } },
      { title: { en: 'Tell the story of your community', fr: 'Raconter l\'histoire de votre communauté' }, body: { en: 'Coach profiles, member spotlights, challenge results and transformation photos communicate the most valuable thing you have. The culture of your space. No big-box gym can replicate it.', fr: 'Les profils d\'entraîneurs, les mises en avant de membres, les résultats de défis et les photos de transformation communiquent ce que vous avez de plus précieux. La culture de votre espace.' } },
      { title: { en: 'Rank for every class and service you offer', fr: 'Se positionner pour chaque cours et service offert' }, body: { en: 'Pages targeting «cours de yoga Montréal», «CrossFit Laval», «boxe Rosemont». People search for the specific discipline, not the gym name.', fr: 'Des pages ciblant «cours de yoga Montréal», «CrossFit Laval», «boxe Rosemont». Les gens cherchent la discipline spécifique, pas le nom du gym.' } },
    ],
    seo: {
      en: 'Class-type and neighbourhood pages for every discipline you offer. «pilates Montréal», «entraîneur personnel Plateau», «cours de spinning Laval». Capturing intent-driven searches.',
      fr: 'Des pages par type de cours et par quartier pour chaque discipline que vous offrez. «pilates Montréal», «entraîneur personnel Plateau», «cours de spinning Laval».',
    },
    ai: {
      en: 'An AI chatbot that answers questions about pricing, schedules and class types, collects trial booking requests and helps members find the right programme. 24 hours a day.',
      fr: 'Un chatbot IA qui répond aux questions sur les prix, horaires et types de cours, recueille les demandes d\'essai et aide les membres à trouver le bon programme. 24 heures sur 24.',
    },
    approach: {
      build: {
        en: 'A site built around your community. Coach profiles, live class schedule, member transformation stories and your facility photos. So a prospective member can feel the culture of your space before walking in.',
        fr: 'Un site construit autour de votre communauté. Profils d\'entraîneurs, horaire de cours en direct, histoires de transformation de membres et photos de vos installations. Pour qu\'un membre potentiel puisse ressentir la culture de votre espace avant d\'y entrer.',
      },
      attract: {
        en: 'Discipline and neighbourhood pages targeting «cours de yoga Montréal», «CrossFit Laval», «boxe Rosemont». Because people search for the specific activity, not the gym name.',
        fr: 'Des pages par discipline et par quartier ciblant «cours de yoga Montréal», «CrossFit Laval», «boxe Rosemont». Parce que les gens cherchent l\'activité spécifique, pas le nom du gym.',
      },
      convert: {
        en: 'A live class schedule with direct trial-class booking. So a prospect can go from curious to confirmed in one tap, without a phone call or waiting for someone to reply to their DM.',
        fr: 'Un horaire de cours en temps réel avec réservation directe de cours d\'essai. Pour qu\'un prospect passe de curieux à confirmé en un clic, sans appel téléphonique ni attente de réponse.',
      },
    },
    recommended: 'growth',
  },
  {
    id: 'other',
    label: { en: 'Other', fr: 'Autre' },
    accentColor: '#8B5CF6',
    hero: {
      headline: {
        en: 'Whatever your business does, your customers are searching for it online right now.',
        fr: 'Peu importe ce que fait votre entreprise, vos clients la recherchent en ligne en ce moment.',
      },
      sub: {
        en: 'Every business in Quebec. In every industry. Deserves a digital presence that works as hard as the people behind it.',
        fr: 'Chaque entreprise au Québec. Dans chaque industrie. Mérite une présence numérique qui travaille aussi fort que les gens qui la font tourner.',
      },
    },
    stats: [
      {
        value: { en: '59%', fr: '59 %' },
        label: { en: 'of Canadians are more likely to do business with a company that has a website. Whatever your industry, not having a website is a first impression you cannot afford to get wrong', fr: 'des Canadiens sont plus susceptibles de faire affaire avec une entreprise qui a un site web. Peu importe votre secteur, l\'absence de site web est une première impression que vous ne pouvez pas vous permettre de rater' },
        source: { en: 'CIRA Canadian Internet Trends 2025', fr: 'Tendances Internet canadiennes ACEI 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '1 in 3', fr: '1 sur 3' },
        label: { en: 'Canadian small businesses currently sell or take orders online. Meaning most businesses across all industries still have no digital sales or booking channel, and those that add one gain an immediate advantage over competitors who rely on phone and walk-in alone', fr: 'petites entreprises canadiennes vendent ou acceptent des commandes en ligne. La plupart des entreprises dans tous les secteurs n\'ont encore aucun canal de vente ou de réservation numérique, et celles qui en ajoutent un gagnent un avantage immédiat' },
        source: { en: 'CFIB Digital Transformation Report 2025', fr: 'Rapport sur la transformation numérique FCEI 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '86%', fr: '86 %' },
        label: { en: 'of Canadians made an online purchase in the past year. Whatever your business, your customers expect to find you, research you and contact you online before they ever show up at your door', fr: 'des Canadiens ont effectué un achat en ligne au cours de la dernière année. Peu importe votre entreprise, vos clients s\'attendent à vous trouver, vous évaluer et vous contacter en ligne avant de se présenter chez vous' },
        source: { en: 'CIRA Canadian Internet Trends 2026', fr: 'Tendances Internet canadiennes ACEI 2026' },
        year: 2026, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      { title: { en: 'Not visible when people search', fr: 'Non visible lors des recherches' }, body: { en: 'Without a website and local SEO foundation, your business simply doesn\'t appear when potential customers search for what you offer in your city.', fr: 'Sans site web et base SEO locale, votre entreprise n\'apparaît tout simplement pas quand des clients potentiels cherchent ce que vous offrez dans votre ville.' } },
      { title: { en: 'No way to convert interest into action', fr: 'Aucun moyen de convertir l\'intérêt en action' }, body: { en: 'Without clear calls-to-action, contact forms and an AI assistant, people who find you online have nowhere to go. And they move on.', fr: 'Sans appels à l\'action clairs, formulaires de contact et assistant IA, les personnes qui vous trouvent en ligne n\'ont nulle part où aller. Et elles passent à autre chose.' } },
      { title: { en: 'Missing the mobile-first reality', fr: 'Rater la réalité mobile-first' }, body: { en: 'Most local searches happen on a phone. A site that isn\'t fast, mobile-friendly and bilingual is losing customers before they\'ve read a single word.', fr: 'La plupart des recherches locales se font sur un téléphone. Un site qui n\'est pas rapide, adapté aux mobiles et bilingue perd des clients avant qu\'ils aient lu un seul mot.' } },
    ],
    opportunities: [
      { title: { en: 'Get found before the competition', fr: 'Être trouvé avant la concurrence' }, body: { en: 'A well-structured website with local SEO puts you in front of customers actively searching for what you do. In Montreal, Quebec City and everywhere in between.', fr: 'Un site bien structuré avec un SEO local vous place devant les clients qui cherchent activement ce que vous faites. À Montréal, Québec et partout entre les deux.' } },
      { title: { en: 'Convert visitors into customers', fr: 'Convertir les visiteurs en clients' }, body: { en: 'Clear service pages, real photos, genuine reviews and a contact form that works on any device turn online discovery into real business.', fr: 'Des pages de services claires, de vraies photos, des avis authentiques et un formulaire de contact qui fonctionne sur n\'importe quel appareil transforment la découverte en ligne en vrais affaires.' } },
      { title: { en: 'Automate the first contact', fr: 'Automatiser le premier contact' }, body: { en: 'An AI chatbot that answers questions and captures lead info means no potential customer falls through the cracks. Even when you\'re busy doing what you do best.', fr: 'Un chatbot IA qui répond aux questions et capte les informations des prospects signifie qu\'aucun client potentiel ne passe entre les mailles. Même quand vous êtes occupé à faire ce que vous faites de mieux.' } },
    ],
    seo: {
      en: 'City and service pages that target what your customers actually search for. From Montreal neighbourhoods to cities across Quebec.',
      fr: 'Des pages par ville et par service qui ciblent ce que vos clients recherchent réellement. Des quartiers de Montréal aux villes du Québec.',
    },
    ai: {
      en: 'A custom AI chatbot trained on your business, your services and your most common questions. Available 24/7 to answer, qualify and convert.',
      fr: 'Un chatbot IA personnalisé formé sur votre entreprise, vos services et vos questions les plus courantes. Disponible 24h/7j pour répondre, qualifier et convertir.',
    },
    approach: {
      build: {
        en: 'A professionally designed website built around your actual business. Your services, your team, your city. Not a generic template with your name swapped in.',
        fr: 'Un site web conçu professionnellement autour de votre vraie entreprise. Vos services, votre équipe, votre ville. Pas un gabarit générique avec votre nom échangé.',
      },
      attract: {
        en: 'Local SEO structure and service pages targeting the specific searches your future customers are making in Montreal and Quebec. Right now.',
        fr: 'Une structure SEO locale et des pages de services ciblant les recherches spécifiques que vos futurs clients font à Montréal et au Québec. En ce moment.',
      },
      convert: {
        en: 'Contact forms, clear calls to action and an AI chatbot that answers questions and captures leads around the clock. So no potential customer falls through the cracks.',
        fr: 'Des formulaires de contact, des appels à l\'action clairs et un chatbot IA qui répond aux questions et capte les prospects en tout temps. Pour qu\'aucun client potentiel ne passe entre les mailles.',
      },
    },
    recommended: 'launch',
  },
  {
    id: 'pest-control',
    label: { en: 'Pest Control', fr: 'Extermination et contrôle parasitaire' },
    accentColor: '#84CC16',
    hero: {
      headline: {
        en: 'When someone has a pest problem, they search for help immediately. The first result gets the call.',
        fr: 'Quand quelqu'un a un problème de parasites, il cherche de l'aide immédiatement. Le premier résultat obtient l'appel.',
      },
      sub: {
        en: 'If your pest control business is not visible on Google in Montreal and Quebec, those urgent calls are going straight to your competitors.',
        fr: 'Si votre entreprise d'extermination n'est pas visible sur Google à Montréal et au Québec, ces appels urgents vont directement chez vos concurrents.',
      },
    },
    stats: [
      {
        value: { en: '59%', fr: '59 %' },
        label: { en: 'of Canadians are more likely to do business with a company that has a website. Pest control is an urgent, trust-sensitive service. Clients who cannot find you online will not wait. They will call whoever shows up first on Google.', fr: 'des Canadiens sont plus susceptibles de faire affaire avec une entreprise qui a un site web. L'extermination est un service urgent et sensible à la confiance. Les clients qui ne vous trouvent pas en ligne n'attendront pas. Ils appelleront celui qui apparaît en premier sur Google.' },
        source: { en: 'CIRA Canadian Internet Trends 2025', fr: 'Tendances Internet canadiennes ACEI 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '75%', fr: '75 %' },
        label: { en: 'of Canadian mortgage consumers plan to renovate their home in the next five years. Renovations disturb walls, foundations and crawl spaces. Pest control demand rises directly alongside home improvement activity across Quebec.', fr: 'des consommateurs hypothécaires canadiens prévoient rénover leur maison au cours des cinq prochaines années. Les rénovations perturbent les murs, les fondations et les vides sanitaires. La demande en extermination augmente directement avec l'activité de rénovation résidentielle au Québec.' },
        source: { en: 'CMHC 2025 Mortgage Consumer Survey', fr: 'Sondage auprès des consommateurs hypothécaires SCHL 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
      {
        value: { en: '9 in 10', fr: '9 sur 10' },
        label: { en: 'Canadian small businesses use at least one digital channel. But only 1 in 3 currently take bookings or sell online. Pest control companies that offer instant online quote requests and inspection bookings convert high-intent searchers before competitors even answer the phone.', fr: 'petites entreprises canadiennes utilisent au moins un canal numérique. Mais seulement 1 sur 3 acceptent des réservations ou vendent en ligne. Les entreprises d'extermination qui offrent des soumissions et réservations d'inspection en ligne convertissent les chercheurs à forte intention avant même que leurs concurrents décrochent le téléphone.' },
        source: { en: 'CFIB Your Voice Survey, September 2025', fr: 'Sondage Votre voix FCEI, septembre 2025' },
        year: 2025, geo: { en: 'Canada', fr: 'Canada' },
      },
    ],
    challenges: [
      {
        title: { en: 'Urgent searches go to whoever ranks first', fr: 'Les recherches urgentes vont au premier résultat' },
        body: { en: 'Pest problems do not wait. When someone finds mice, bedbugs or cockroaches, they search immediately and call the first credible result. If that is not you, the job is gone.', fr: 'Les problèmes de parasites n'attendent pas. Quand quelqu'un trouve des souris, des punaises ou des coquerelles, il cherche immédiatement et appelle le premier résultat crédible. Si ce n'est pas vous, le contrat est perdu.' },
      },
      {
        title: { en: 'Trust is everything in a service that enters the home', fr: 'La confiance est primordiale pour un service qui entre dans la maison' },
        body: { en: 'Homeowners and tenants need to feel safe before letting someone inside. Without a professional site, certifications visible online and real reviews, doubt replaces trust and the call never comes.', fr: 'Les propriétaires et locataires ont besoin de se sentir en sécurité avant de laisser entrer quelqu'un. Sans site professionnel, certifications visibles et vrais avis en ligne, le doute remplace la confiance et l'appel n'arrive jamais.' },
      },
      {
        title: { en: 'Seasonal peaks lost to competitors with better visibility', fr: 'Pics saisonniers perdus au profit de concurrents mieux visibles' },
        body: { en: 'Spring ant invasions, fall rodent intrusions and summer wasp nests drive predictable surges in demand. Without SEO and a site that captures that traffic, your busy season belongs to whoever ranks above you.', fr: 'Les invasions de fourmis au printemps, les intrusions de rongeurs à l'automne et les nids de guêpes en été créent des pointes de demande prévisibles. Sans SEO et site qui capte ce trafic, votre haute saison appartient à ceux qui vous dépassent sur Google.' },
      },
    ],
    opportunities: [
      {
        title: { en: 'Own the urgent local searches', fr: 'Dominer les recherches locales urgentes' },
        body: { en: 'Dedicated pages for «exterminateur Montréal», «punaises de lit Laval», «souris maison Longueuil» capture the highest-intent pest calls the moment they happen. These are not browsers. These are buyers.', fr: 'Des pages dédiées à «exterminateur Montréal», «punaises de lit Laval», «souris maison Longueuil» captent les appels à forte intention au moment où ils se produisent. Ce ne sont pas des curieux. Ce sont des acheteurs.' },
      },
      {
        title: { en: 'Build the trust that sends people straight to you', fr: 'Construire la confiance qui envoie les gens directement vers vous' },
        body: { en: 'Licensed technician profiles, before-and-after treatment photos, your certifications and a wall of verified reviews transform a cold search into a confident call before the competition gets a look in.', fr: 'Profils de techniciens certifiés, photos de traitements avant-après, vos certifications et un mur d'avis vérifiés transforment une recherche froide en appel confiant avant même que la concurrence ait une chance.' },
      },
      {
        title: { en: 'Book inspections around the clock', fr: 'Réserver des inspections en tout temps' },
        body: { en: 'An online inspection request form available 24 hours a day means a homeowner who discovers a problem at 11pm books with you instead of calling around in the morning. You wake up to a full schedule.', fr: 'Un formulaire de demande d'inspection disponible 24 heures sur 24 signifie qu'un propriétaire qui découvre un problème à 23h réserve avec vous plutôt que d'appeler partout le matin. Vous vous réveillez avec un agenda rempli.' },
      },
    ],
    seo: {
      en: 'Pest-specific and neighbourhood-level pages targeting «souris Montréal», «coquerelles Laval», «punaises de lit Brossard», «exterminateur Rive-Sud» and every pest-and-city combination your customers are searching right now.',
      fr: 'Des pages par type de parasite et par quartier ciblant «souris Montréal», «coquerelles Laval», «punaises de lit Brossard», «exterminateur Rive-Sud» et chaque combinaison parasite-ville que vos clients recherchent en ce moment.',
    },
    ai: {
      en: 'An AI chatbot that helps callers identify their pest, describes your treatment process, answers questions about preparation and safety, and books an inspection. Available the moment the problem is discovered.',
      fr: 'Un chatbot IA qui aide les appelants à identifier leur parasite, décrit votre processus de traitement, répond aux questions sur la préparation et la sécurité, et réserve une inspection. Disponible au moment où le problème est découvert.',
    },
    approach: {
      build: {
        en: 'A site organized by pest type and service area. Rodents, insects, bedbugs, wasps. Montreal, Laval, Longueuil, South Shore. With licensed technician profiles, your certifications and treatment photos that make homeowners feel safe calling you.',
        fr: 'Un site organisé par type de parasite et zone de service. Rongeurs, insectes, punaises, guêpes. Montréal, Laval, Longueuil, Rive-Sud. Avec des profils de techniciens certifiés, vos accréditations et des photos de traitements qui donnent confiance aux propriétaires pour vous appeler.',
      },
      attract: {
        en: 'Pest-and-city keyword pages targeting the exact emergency searches your future clients make the moment they have a problem. «Souris Montréal». «Punaises Laval». «Exterminateur urgence Brossard». The search that happens at 10pm on a Tuesday.',
        fr: 'Des pages par parasite et ville ciblant les recherches d'urgence exactes que vos futurs clients font au moment où ils ont un problème. «Souris Montréal». «Punaises Laval». «Exterminateur urgence Brossard». La recherche qui se fait à 22h un mardi.',
      },
      convert: {
        en: 'An online inspection booking form and an AI chatbot that helps identify the pest, outlines the treatment and captures contact details right then. No hold music. No missed calls. Just a booked inspection waiting for you in the morning.',
        fr: 'Un formulaire de réservation d'inspection en ligne et un chatbot IA qui aide à identifier le parasite, présente le traitement et capte les coordonnées sur le moment. Pas de musique d'attente. Pas d'appels manqués. Juste une inspection réservée qui vous attend le matin.',
      },
    },
    recommended: 'growth',
  },
];

const PACKAGES = [
  {
    id: 'launch' as const,
    name: { en: 'Website Launch', fr: 'Lancement Web' },
    price: { en: '', fr: '' },
    desc: {
      en: 'A professional, custom-built website that establishes your digital foundation and starts ranking on Google.',
      fr: 'Un site web professionnel sur mesure qui établit votre fondation numérique et commence à se positionner sur Google.',
    },
    features: {
      en: ['Custom design built to your brand', 'Unlimited pages', 'Mobile-first & bilingual (EN/FR)', 'SEO foundation. Google-ready', 'Contact forms & calls to action', 'Hosting, maintenance & updates included'],
      fr: ['Design personnalisé à votre marque', 'Pages illimitées', 'Mobile-first et bilingue (EN/FR)', 'Base SEO. Prêt pour Google', 'Formulaires et appels à l\'action', 'Hébergement, maintenance et mises à jour inclus'],
    },
  },
  {
    id: 'growth' as const,
    name: { en: 'Growth System', fr: 'Système de croissance' },
    price: { en: '', fr: '' },
    desc: {
      en: 'Everything in Website Launch, plus an AI chatbot that qualifies leads, answers questions and captures opportunities around the clock.',
      fr: 'Tout ce qui est inclus dans Lancement Web, plus un chatbot IA qui qualifie les prospects, répond aux questions et capte les opportunités en tout temps.',
    },
    features: {
      en: ['Everything in Website Launch', 'AI chatbot trained on your business', 'Lead qualification 24/7', 'Automated inquiry capture', 'Instant answers to common questions', 'Bilingual AI (EN/FR)'],
      fr: ['Tout ce qui est inclus dans Lancement Web', 'Chatbot IA formé sur votre entreprise', 'Qualification de prospects 24h/7j', 'Capture automatique des demandes', 'Réponses instantanées aux questions courantes', 'IA bilingue (EN/FR)'],
    },
  },
  {
    id: 'pro' as const,
    name: { en: 'Growth Pro', fr: 'Croissance Pro' },
    price: { en: '', fr: '' },
    desc: {
      en: 'Full digital growth strategy. Advanced SEO, local authority content, reputation management and ongoing optimisation for businesses ready to dominate their market.',
      fr: 'Stratégie de croissance numérique complète. SEO avancé, contenu d\'autorité locale, gestion de la réputation et optimisation continue pour les entreprises prêtes à dominer leur marché.',
    },
    features: {
      en: ['Everything in Growth System', 'Advanced local SEO & city pages', 'Google Business Profile management', 'Review generation system', 'Monthly performance reporting', 'Dedicated growth strategist'],
      fr: ['Tout ce qui est inclus dans Système de croissance', 'SEO local avancé et pages par ville', 'Gestion du profil Google Business', 'Système de génération d\'avis', 'Rapports de performance mensuels', 'Stratège de croissance dédié'],
    },
  },
];

// ─────────────────────────────────────────────
// Small components
// ─────────────────────────────────────────────
function CheckIcon({ color }: { color: string }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
      <circle cx="7" cy="7" r="6.5" stroke={color} opacity={0.4} />
      <path d="M4 7l2 2 4-4" stroke={color} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SectionLabel({ n, label }: { n: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
      <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', color: 'rgba(139,92,246,0.5)', fontFamily: 'var(--font-space-grotesk)' }}>{n}</span>
      <div style={{ width: 24, height: 1, background: 'rgba(139,92,246,0.25)' }} />
      <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.18em', textTransform: 'uppercase' as const, color: 'rgba(139,92,246,0.55)' }}>{label}</span>
    </div>
  );
}

// ─────────────────────────────────────────────
// Inner page (uses useSearchParams)
// ─────────────────────────────────────────────
function PlanPageInner({ lang }: { lang: Lang }) {
  const fr = lang === 'fr';
  const router = useRouter();
  const searchParams = useSearchParams();
  const paramId = searchParams.get('industry') as IndustryId | null;
  const [selected, setSelected] = useState<IndustryId | null>(paramId);
  const [open, setOpen] = useState(false);

  const industry = INDUSTRIES.find(i => i.id === selected) ?? null;

  useEffect(() => {
    if (selected) {
      const url = new URL(window.location.href);
      url.searchParams.set('industry', selected);
      router.replace(url.pathname + url.search, { scroll: false });
    }
  }, [selected, router]);

  const accent = industry?.accentColor ?? '#8B5CF6';
  const t = (obj: T) => obj[fr ? 'fr' : 'en'];

  return (
    <main style={{ background: '#08041A', minHeight: '100dvh', fontFamily: 'var(--font-space-grotesk)' }}>

      {/* Film grain */}
      <svg aria-hidden style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', opacity: 0.022, zIndex: 0 }}>
        <filter id="plan-grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.78" numOctaves="4" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#plan-grain)" />
      </svg>

      <div style={{ position: 'relative', zIndex: 1 }}>
        <PageNav lang={lang} active="strategy" enHref="/strategy" frHref="/fr/strategy" />

        {/* ── HERO ── */}
        <section style={{ padding: 'clamp(72px,10vh,120px) clamp(24px,5vw,80px) clamp(48px,6vh,80px)', maxWidth: 1200, margin: '0 auto' }}>
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.75, ease }}>
            <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.24em', textTransform: 'uppercase', color: '#8B5CF6', margin: '0 0 20px' }}>
              {fr ? 'Stratégie de croissance numérique' : 'Digital Growth Strategy'}
            </p>
            <h1 style={{ fontSize: 'clamp(36px,5.5vw,70px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.04, margin: '0 0 24px', maxWidth: 820 }}>
              {fr ? 'Chaque industrie a ses propres opportunités en ligne.' : 'Every industry has its own opportunities online.'}
            </h1>
            <p style={{ fontSize: 'clamp(16px,1.5vw,20px)', color: 'rgba(220,210,245,0.52)', lineHeight: 1.75, maxWidth: 580, margin: 0 }}>
              {fr
                ? 'Sélectionnez votre industrie et découvrez exactement ce que les entreprises comme la vôtre peuvent gagner. Avec des données, des défis concrets et une approche qui a fait ses preuves au Québec.'
                : 'Select your industry and discover exactly what businesses like yours stand to gain. With data, real challenges and an approach proven in Montreal and Quebec.'}
            </p>
          </motion.div>
        </section>

        {/* ── INDUSTRY SELECTOR ── */}
        <section style={{ padding: '0 clamp(24px,5vw,80px) clamp(64px,8vh,96px)', maxWidth: 1200, margin: '0 auto' }}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65, ease, delay: 0.15 }}>
            <p style={{ fontSize: 13, fontWeight: 700, color: '#A78BFA', marginBottom: 10, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
              {fr ? 'Sélectionnez votre industrie' : 'Select your industry'}
            </p>

            {/* Custom dropdown */}
            <div style={{ position: 'relative', maxWidth: 480 }}>
              <button
                onClick={() => setOpen(o => !o)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '18px 22px', borderRadius: 14,
                  background: industry ? `rgba(${hexToRgb(accent)},0.08)` : 'rgba(139,92,246,0.06)',
                  border: `1.5px solid ${industry ? `rgba(${hexToRgb(accent)},0.4)` : 'rgba(139,92,246,0.45)'}`,
                  color: industry ? '#F5F0FC' : '#C4B5FD',
                  fontSize: 16, fontWeight: industry ? 600 : 500,
                  cursor: 'pointer', textAlign: 'left', outline: 'none',
                  transition: 'border-color 0.2s, background 0.2s, box-shadow 0.2s',
                  fontFamily: 'var(--font-space-grotesk)',
                  boxShadow: industry ? 'none' : '0 0 0 3px rgba(139,92,246,0.12), 0 4px 24px rgba(139,92,246,0.15)',
                }}
              >
                <span>{industry ? t(industry.label) : (fr ? '— Choisissez votre industrie' : '— Choose your industry')}</span>
                <svg
                  width="16" height="16" viewBox="0 0 16 16" fill="none"
                  style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s', flexShrink: 0 }}
                >
                  <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>

              <AnimatePresence>
                {open && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.98 }}
                    transition={{ duration: 0.18, ease }}
                    style={{
                      position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0, zIndex: 50,
                      background: 'rgba(12,6,32,0.98)', border: '1px solid rgba(139,92,246,0.18)',
                      borderRadius: 12, overflow: 'hidden',
                      boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
                    }}
                  >
                    {INDUSTRIES.map(ind => (
                      <button
                        key={ind.id}
                        onClick={() => { setSelected(ind.id); setOpen(false); }}
                        style={{
                          width: '100%', display: 'flex', alignItems: 'center', gap: 12,
                          padding: '12px 20px', background: selected === ind.id ? `rgba(${hexToRgb(ind.accentColor)},0.1)` : 'none',
                          border: 'none', cursor: 'pointer', textAlign: 'left',
                          color: selected === ind.id ? '#F5F0FC' : 'rgba(220,210,245,0.60)',
                          fontSize: 14, fontWeight: selected === ind.id ? 600 : 400,
                          transition: 'background 0.15s, color 0.15s',
                          fontFamily: 'var(--font-space-grotesk)',
                        }}
                        onMouseEnter={e => { if (selected !== ind.id) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.03)'; }}
                        onMouseLeave={e => { if (selected !== ind.id) (e.currentTarget as HTMLButtonElement).style.background = 'none'; }}
                      >
                        <span style={{ width: 8, height: 8, borderRadius: '50%', background: ind.accentColor, flexShrink: 0, opacity: selected === ind.id ? 1 : 0.4 }} />
                        {t(ind.label)}
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </section>

        {/* ── DYNAMIC CONTENT ── */}
        <AnimatePresence mode="wait">
          {industry && (
            <motion.div
              key={industry.id}
              initial={{ opacity: 0, y: 32 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.55, ease }}
            >
              {/* Thin accent line */}
              <div style={{ height: 1, background: `linear-gradient(90deg,transparent,${accent}44 30%,${accent}44 70%,transparent)`, margin: '0 clamp(24px,5vw,80px)' }} />

              {/* ── INDUSTRY HERO ── */}
              <section style={{ padding: 'clamp(56px,7vh,80px) clamp(24px,5vw,80px)', maxWidth: 1200, margin: '0 auto' }}>
                <div style={{ display: 'grid', gridTemplateColumns: industry.stats.length > 0 ? 'repeat(auto-fit,minmax(300px,1fr))' : '1fr', gap: 'clamp(32px,4vw,64px)', alignItems: industry.stats.length > 0 ? 'center' : 'start' }}>
                  <div>
                    <SectionLabel n="01" label={fr ? 'L\'opportunité' : 'The Opportunity'} />
                    <h2 style={{ fontSize: 'clamp(26px,3.5vw,44px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 20px' }}>
                      {t(industry.hero.headline)}
                    </h2>
                    <p style={{ fontSize: 'clamp(15px,1.3vw,17px)', color: 'rgba(220,210,245,0.55)', lineHeight: 1.75, margin: 0, maxWidth: industry.stats.length === 0 ? 680 : undefined }}>
                      {t(industry.hero.sub)}
                    </p>
                  </div>

                  {/* Stats. Only rendered when the industry has qualifying data */}
                  {industry.stats.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                      {industry.stats.map((stat, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.5, ease, delay: i * 0.08 }}
                          style={{
                            padding: '20px 24px', borderRadius: 12,
                            background: 'rgba(255,255,255,0.02)',
                            border: `1px solid rgba(${hexToRgb(accent)},0.12)`,
                            display: 'flex', gap: 16, alignItems: 'flex-start',
                          }}
                        >
                          <div style={{ fontSize: 'clamp(28px,3vw,38px)', fontWeight: 800, color: accent, letterSpacing: '-0.03em', lineHeight: 1, flexShrink: 0 }}>
                            {t(stat.value)}
                          </div>
                          <div>
                            <p style={{ fontSize: 13, color: 'rgba(220,210,245,0.72)', lineHeight: 1.5, margin: '0 0 6px' }}>{t(stat.label)}</p>
                            <p style={{ fontSize: 10.5, color: 'rgba(220,210,245,0.28)', margin: 0, letterSpacing: '0.01em' }}>
                              {t(stat.source)} · {stat.year} · {t(stat.geo)}
                            </p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              </section>

              <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', margin: '0 clamp(24px,5vw,80px)' }} />

              {/* ── CHALLENGES ── */}
              <section style={{ padding: 'clamp(56px,7vh,80px) clamp(24px,5vw,80px)', maxWidth: 1200, margin: '0 auto' }}>
                <SectionLabel n="02" label={fr ? 'Les défis' : 'The Challenges'} />
                <h2 style={{ fontSize: 'clamp(24px,3vw,38px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 40px', maxWidth: 600 }}>
                  {fr ? 'Ce que la plupart des entreprises dans votre industrie vivent en ligne.' : 'What most businesses in your industry experience online.'}
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 20 }}>
                  {industry.challenges.map((c, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.48, ease, delay: i * 0.07 }}
                      style={{ padding: '28px 28px 32px', borderRadius: 14, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 36, height: 36, borderRadius: 8, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.15)', marginBottom: 18 }}>
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 3v5M8 11.5v.5" stroke="rgba(239,68,68,0.7)" strokeWidth="1.6" strokeLinecap="round"/></svg>
                      </div>
                      <h3 style={{ fontSize: 15, fontWeight: 700, color: '#F5F0FC', margin: '0 0 10px', letterSpacing: '-0.01em' }}>{t(c.title)}</h3>
                      <p style={{ fontSize: 13.5, color: 'rgba(220,210,245,0.50)', lineHeight: 1.65, margin: 0 }}>{t(c.body)}</p>
                    </motion.div>
                  ))}
                </div>
              </section>

              <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', margin: '0 clamp(24px,5vw,80px)' }} />

              {/* ── OPPORTUNITIES ── */}
              <section style={{ padding: 'clamp(56px,7vh,80px) clamp(24px,5vw,80px)', maxWidth: 1200, margin: '0 auto' }}>
                <SectionLabel n="03" label={fr ? 'La croissance' : 'Growth Opportunities'} />
                <h2 style={{ fontSize: 'clamp(24px,3vw,38px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 40px', maxWidth: 600 }}>
                  {fr ? 'Comment une meilleure présence numérique transforme votre entreprise.' : 'How a better digital presence transforms your business.'}
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 20 }}>
                  {industry.opportunities.map((o, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.48, ease, delay: i * 0.07 }}
                      style={{ padding: '28px 28px 32px', borderRadius: 14, background: `rgba(${hexToRgb(accent)},0.04)`, border: `1px solid rgba(${hexToRgb(accent)},0.12)` }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 36, height: 36, borderRadius: 8, background: `rgba(${hexToRgb(accent)},0.12)`, border: `1px solid rgba(${hexToRgb(accent)},0.2)`, marginBottom: 18 }}>
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 10l4-4 3 3 3-5" stroke={accent} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>
                      </div>
                      <h3 style={{ fontSize: 15, fontWeight: 700, color: '#F5F0FC', margin: '0 0 10px', letterSpacing: '-0.01em' }}>{t(o.title)}</h3>
                      <p style={{ fontSize: 13.5, color: 'rgba(220,210,245,0.55)', lineHeight: 1.65, margin: 0 }}>{t(o.body)}</p>
                    </motion.div>
                  ))}
                </div>
              </section>

              {/* ── SEO + AI pills ── */}
              <section style={{ padding: '0 clamp(24px,5vw,80px) clamp(56px,7vh,80px)', maxWidth: 1200, margin: '0 auto' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(280px,1fr))', gap: 16 }}>
                  {[
                    {
                      icon: (
                        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="8" cy="8" r="6" stroke={accent} strokeWidth="1.5"/><path d="M13 13l3 3" stroke={accent} strokeWidth="1.5" strokeLinecap="round"/></svg>
                      ),
                      eyebrow: { en: 'Local SEO', fr: 'SEO local' },
                      body: industry.seo,
                    },
                    {
                      icon: (
                        <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="2" y="4" width="14" height="10" rx="2" stroke={accent} strokeWidth="1.5"/><path d="M5 9h4M5 12h2" stroke={accent} strokeWidth="1.3" strokeLinecap="round"/><circle cx="13" cy="9" r="1.5" fill={accent}/></svg>
                      ),
                      eyebrow: { en: 'AI & Automation', fr: 'IA et automatisation' },
                      body: industry.ai,
                    },
                  ].map((item, i) => (
                    <div key={i} style={{ padding: '24px 28px', borderRadius: 14, background: `rgba(${hexToRgb(accent)},0.04)`, border: `1px solid rgba(${hexToRgb(accent)},0.14)`, display: 'flex', gap: 16 }}>
                      <div style={{ flexShrink: 0, marginTop: 2 }}>{item.icon}</div>
                      <div>
                        <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.16em', textTransform: 'uppercase', color: accent, margin: '0 0 8px' }}>{t(item.eyebrow)}</p>
                        <p style={{ fontSize: 13.5, color: 'rgba(220,210,245,0.58)', lineHeight: 1.65, margin: 0 }}>{t(item.body)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

            </motion.div>
          )}
        </AnimatePresence>

        {/* ── OUR APPROACH ── */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', margin: '0 clamp(24px,5vw,80px)' }} />
        <section style={{ padding: 'clamp(56px,7vh,80px) clamp(24px,5vw,80px)', maxWidth: 1200, margin: '0 auto' }}>
          <SectionLabel n="04" label={fr ? 'Notre approche' : 'Our Approach'} />
          <h2 style={{ fontSize: 'clamp(24px,3vw,38px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 48px', maxWidth: 560 }}>
            {fr ? 'Trois piliers. Un résultat : des clients qui vous trouvent et vous contactent.' : 'Three pillars. One result: customers who find you and reach out.'}
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 20 }}>
            {[
              {
                n: '01',
                word: 'BUILD',
                title: { en: 'Foundation', fr: 'Fondation' },
                body: industry?.approach.build ?? { en: 'A professionally designed website, built around your brand, your services and your city. Not a template with your logo swapped in.', fr: 'Un site web conçu professionnellement, bâti autour de votre marque, vos services et votre ville. Pas un gabarit avec votre logo échangé.' },
                color: '#8B5CF6',
              },
              {
                n: '02',
                word: 'ATTRACT',
                title: { en: 'Visibility', fr: 'Visibilité' },
                body: industry?.approach.attract ?? { en: 'Local SEO structure, Google Business optimisation and content that ranks for the specific searches your future customers are making right now.', fr: 'Structure SEO locale, optimisation Google Business et contenu qui se positionne sur les recherches spécifiques que vos futurs clients font en ce moment.' },
                color: '#3B82F6',
              },
              {
                n: '03',
                word: 'CONVERT',
                title: { en: 'Conversion', fr: 'Conversion' },
                body: industry?.approach.convert ?? { en: 'Lead capture forms, AI chatbot, booking systems and CTAs designed to turn every visitor into a contact. Especially when you\'re not available to answer.', fr: 'Formulaires de capture, chatbot IA, systèmes de réservation et appels à l\'action conçus pour transformer chaque visiteur en contact. Surtout quand vous n\'êtes pas disponible.' },
                color: '#10B981',
              },
            ].map((p, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, ease, delay: i * 0.08 }}
                style={{ padding: '32px 28px', borderRadius: 14, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
                  <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.08em', color: 'rgba(255,255,255,0.2)' }}>{p.n}</span>
                  <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.18em', color: p.color }}>{p.word}</span>
                </div>
                <h3 style={{ fontSize: 20, fontWeight: 800, color: '#F5F0FC', margin: '0 0 12px', letterSpacing: '-0.02em' }}>{t(p.title)}</h3>
                <p style={{ fontSize: 13.5, color: 'rgba(220,210,245,0.50)', lineHeight: 1.65, margin: 0 }}>{t(p.body)}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* ── PACKAGES ── */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', margin: '0 clamp(24px,5vw,80px)' }} />
        <section style={{ padding: 'clamp(56px,7vh,80px) clamp(24px,5vw,80px)', maxWidth: 1200, margin: '0 auto' }}>
          <SectionLabel n="05" label={fr ? 'Nos solutions' : 'Our Solutions'} />
          <h2 style={{ fontSize: 'clamp(24px,3vw,38px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 12px', maxWidth: 560 }}>
            {fr ? 'Choisissez le niveau de croissance qui vous convient.' : 'Choose the level of growth that fits your business.'}
          </h2>
          {industry && (
            <p style={{ fontSize: 14, color: 'rgba(220,210,245,0.40)', margin: '0 0 40px' }}>
              {fr ? `Recommandé pour ${t(industry.label)} :` : `Recommended for ${t(industry.label)}:`}
              {' '}<span style={{ color: 'rgba(139,92,246,0.80)', fontWeight: 600 }}>{PACKAGES.find(p => p.id === industry.recommended) ? t(PACKAGES.find(p => p.id === industry.recommended)!.name) : ''}</span>
            </p>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 16 }}>
            {PACKAGES.map((pkg, i) => {
              const isRec = industry?.recommended === pkg.id;
              return (
                <motion.div
                  key={pkg.id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-40px' }}
                  transition={{ duration: 0.5, ease, delay: i * 0.08 }}
                  style={{
                    padding: '32px 28px',
                    borderRadius: 14,
                    background: isRec ? 'rgba(109,40,217,0.07)' : 'rgba(255,255,255,0.02)',
                    border: isRec ? '1px solid rgba(139,92,246,0.35)' : '1px solid rgba(255,255,255,0.06)',
                    display: 'flex', flexDirection: 'column', gap: 20,
                    position: 'relative' as const,
                  }}
                >
                  {isRec && (
                    <div style={{
                      position: 'absolute', top: -1, left: 24,
                      background: 'linear-gradient(135deg,#8B5CF6,#6D28D9)',
                      borderRadius: '0 0 8px 8px',
                      padding: '4px 12px',
                      fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', color: '#fff',
                      textTransform: 'uppercase' as const,
                    }}>
                      {fr ? 'Recommandé' : 'Recommended'}
                    </div>
                  )}
                  <div style={{ paddingTop: isRec ? 8 : 0 }}>
                    <p style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: isRec ? '#A78BFA' : 'rgba(139,92,246,0.55)', margin: 0 }}>
                      {t(pkg.name)}
                    </p>
                  </div>
                  <div style={{ height: 1, background: isRec ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.05)' }} />
                  <p style={{ fontSize: 13.5, color: 'rgba(220,210,245,0.55)', lineHeight: 1.65, margin: 0 }}>{t(pkg.desc)}</p>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {pkg.features[fr ? 'fr' : 'en'].map((f, fi) => (
                      <li key={fi} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                        <span style={{ marginTop: 1, flexShrink: 0 }}><CheckIcon color={isRec ? '#A78BFA' : '#8B5CF6'} /></span>
                        <span style={{ fontSize: 13, color: isRec ? 'rgba(220,210,245,0.78)' : 'rgba(220,210,245,0.58)', lineHeight: 1.5 }}>{f}</span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* ── WHY AGENTEDIRECT ── */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', margin: '0 clamp(24px,5vw,80px)' }} />
        <section style={{ padding: 'clamp(56px,7vh,80px) clamp(24px,5vw,80px)', maxWidth: 1200, margin: '0 auto' }}>
          <SectionLabel n="06" label={fr ? 'Pourquoi AgenteDirect' : 'Why AgenteDirect'} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(280px,1fr))', gap: 'clamp(32px,5vw,80px)', alignItems: 'center' }}>
            <div>
              <h2 style={{ fontSize: 'clamp(24px,3vw,38px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.03em', lineHeight: 1.1, margin: '0 0 24px' }}>
                {fr ? 'On ne prend pas un gabarit et on change le logo.' : 'We don\'t take a template and change the logo.'}
              </h2>
              <p style={{ fontSize: 'clamp(14px,1.2vw,16px)', color: 'rgba(220,210,245,0.52)', lineHeight: 1.75, margin: '0 0 16px' }}>
                {fr
                  ? 'On prend le temps de comprendre votre entreprise, votre clientèle et votre marché au Québec. Chaque site qu\'on construit reflète vraiment l\'entreprise derrière lui.'
                  : 'We take the time to understand your business, your customers and your market in Quebec. Every site we build actually reflects the business behind it.'}
              </p>
              <p style={{ fontSize: 'clamp(14px,1.2vw,16px)', color: 'rgba(220,210,245,0.52)', lineHeight: 1.75, margin: 0 }}>
                {fr
                  ? 'On est basés à Montréal. On connaît le marché bilingue, les entreprises locales et les nuances du Québec. Et on est là après le lancement. Pour les mises à jour, les promotions et la croissance continue.'
                  : 'We\'re based in Montreal. We know the bilingual market, local businesses and the nuances of Quebec. And we\'re here after launch. For updates, promotions and continued growth.'}
              </p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[
                { en: 'Based in Montreal, built for Quebec businesses', fr: 'Basé à Montréal, conçu pour les entreprises québécoises' },
                { en: 'Bilingual. Equal quality in French and English', fr: 'Bilingue. Qualité égale en français et en anglais' },
                { en: 'Custom design, not templates', fr: 'Design personnalisé, pas des gabarits' },
                { en: 'Your photos, your story, your business', fr: 'Vos photos, votre histoire, votre entreprise' },
                { en: 'Ongoing support. Not build-and-disappear', fr: 'Support continu. Pas bâtir et disparaître' },
                { en: 'Modern technology. Next.js, AI, real-time SEO', fr: 'Technologie moderne. Next.js, IA, SEO en temps réel' },
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 18px', borderRadius: 10, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <CheckIcon color="#8B5CF6" />
                  <span style={{ fontSize: 13.5, color: 'rgba(220,210,245,0.70)' }}>{t(item)}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── FINAL CTA ── */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', margin: '0 clamp(24px,5vw,80px)' }} />
        <section style={{ padding: 'clamp(72px,10vh,120px) clamp(24px,5vw,80px)', textAlign: 'center', maxWidth: 1200, margin: '0 auto' }}>
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.65, ease }}
          >
            <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.24em', textTransform: 'uppercase', color: '#8B5CF6', margin: '0 0 20px' }}>
              {fr ? 'Prochaine étape' : 'Next Step'}
            </p>
            <h2 style={{ fontSize: 'clamp(28px,4vw,52px)', fontWeight: 800, color: '#F5F0FC', letterSpacing: '-0.04em', lineHeight: 1.07, margin: '0 0 20px' }}>
              {fr ? 'Parlons de votre croissance.' : 'Let\'s talk about your growth.'}
            </h2>
            <p style={{ fontSize: 'clamp(15px,1.3vw,18px)', color: 'rgba(220,210,245,0.45)', lineHeight: 1.75, maxWidth: 480, margin: '0 auto 40px' }}>
              {fr
                ? 'Une consultation gratuite. On discute de votre industrie, votre situation actuelle et ce qu\'on peut faire ensemble.'
                : 'A free consultation. We\'ll talk through your industry, your current situation and what we can build together.'}
            </p>
            <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Link
                href={fr ? '/fr/inquiry' : '/inquiry'}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 8,
                  padding: '15px 34px', borderRadius: 50,
                  background: 'linear-gradient(135deg,#8B5CF6 0%,#6D28D9 100%)',
                  color: '#fff', fontSize: 15, fontWeight: 700,
                  textDecoration: 'none', letterSpacing: '0.01em',
                  boxShadow: '0 4px 28px rgba(109,40,217,0.45)',
                  transition: 'transform 0.18s, box-shadow 0.22s',
                }}
                onMouseEnter={e => { const el = e.currentTarget as HTMLAnchorElement; el.style.transform = 'translateY(-2px)'; el.style.boxShadow = '0 10px 40px rgba(109,40,217,0.60)'; }}
                onMouseLeave={e => { const el = e.currentTarget as HTMLAnchorElement; el.style.transform = 'translateY(0)'; el.style.boxShadow = '0 4px 28px rgba(109,40,217,0.45)'; }}
              >
                {fr ? 'Travailler avec nous' : 'Work With Us'}
                <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden>
                  <path d="M2 11L11 2M11 2H5.5M11 2V7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </Link>
              <a
                href={`mailto:${BUSINESS_EMAIL}`}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 8,
                  padding: '15px 34px', borderRadius: 50,
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: 'rgba(220,210,245,0.70)', fontSize: 15, fontWeight: 600,
                  textDecoration: 'none', letterSpacing: '0.01em',
                  transition: 'border-color 0.2s, color 0.2s',
                }}
                onMouseEnter={e => { const el = e.currentTarget as HTMLAnchorElement; el.style.borderColor = 'rgba(255,255,255,0.22)'; el.style.color = '#F5F0FC'; }}
                onMouseLeave={e => { const el = e.currentTarget as HTMLAnchorElement; el.style.borderColor = 'rgba(255,255,255,0.1)'; el.style.color = 'rgba(220,210,245,0.70)'; }}
              >
                {BUSINESS_EMAIL}
              </a>
            </div>
          </motion.div>
        </section>

        <PageFooter />
      </div>

      {/* Responsive overrides */}
      <style>{`
        @media (max-width: 640px) {
          .plan-two-col { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </main>
  );
}

// Hex → "r,g,b" for rgba()
function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

// ─────────────────────────────────────────────
// Default export (Suspense boundary for useSearchParams)
// ─────────────────────────────────────────────
export default function PlanPage({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  return (
    <Suspense fallback={<div style={{ background: '#08041A', minHeight: '100dvh' }} />}>
      <PlanPageInner lang={lang} />
    </Suspense>
  );
}
