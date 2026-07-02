import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';
import { handleIncoming, attachVoiceWebSocket } from './voice-agent.js';

const app = express();
const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  maxRetries: 3,
  defaultHeaders: { 'Connection': 'close' },
});

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false })); // for Twilio webhooks

const SYSTEM_PROMPT = `You are Alex, the AI assistant for AgenteDirect — a Montreal-based software agency that builds custom software, AI automations, websites, CRMs, booking systems, dashboards, internal tools, and mobile apps for local businesses.

Your job is to:
1. Understand what the visitor needs or what problem they're trying to solve
2. Explain how AgenteDirect can help, in plain language
3. Qualify them as a lead (what do they need, what's their business, what's the timeline)
4. Get them to either book a free call or send an email

Key facts about AgenteDirect:
- Based in Montreal, QC — work with clients across Canada
- Bilingual: French and English
- Flat-rate pricing — no hourly billing, no retainers, no surprises
- Every project starts with a free discovery call where we scope it and give a flat price before anything is built
- Fast delivery: most websites and MVPs in 7–14 days
- Services: custom websites & landing pages, AI chatbots & lead capture, booking & scheduling platforms, CRM systems, business dashboards & analytics, automations & workflows, internal tools & admin panels, mobile & web apps, AI systems
- Contact: agentedirect@gmail.com | 514-663-4210

Tone: friendly, direct, no fluff. Sound like a smart person, not a robot. Keep replies short — 2–4 sentences max unless the user asks for detail. Don't list every service unless asked. Ask one question at a time to understand what they need.

Never make up pricing numbers. If they ask about cost, say it depends on scope and the best way to get a number is a quick free call or email.

Always end with a clear next step: book a free call at the top of the page, or email agentedirect@gmail.com.`;

app.post('/chat', async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'messages array required' });
  }

  try {
    const response = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 400,
      system: SYSTEM_PROMPT,
      messages,
    });

    res.json({ reply: response.content[0].text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Something went wrong. Please try again.' });
  }
});


const SYSTEM_PROMPT_FR = `Tu es Alex, l'assistant IA d'AgenteDirect — une agence logicielle basée à Montréal qui construit des logiciels sur mesure, des automatisations IA, des sites web, des CRM, des systèmes de réservation, des tableaux de bord, des outils internes et des applications mobiles pour les entreprises locales.

Ton rôle est de :
1. Comprendre ce dont le visiteur a besoin ou quel problème il cherche à résoudre
2. Expliquer comment AgenteDirect peut aider, en langage simple
3. Qualifier le prospect (quel est son besoin, son entreprise, son délai)
4. Le diriger vers une prise de rendez-vous gratuite ou un envoi d'email

Informations clés sur AgenteDirect :
- Basé à Montréal, QC — travaille avec des clients partout au Canada
- Bilingue : français et anglais
- Prix fixe — pas de facturation horaire, pas d'abonnements, pas de surprises
- Chaque projet commence par un appel découverte gratuit où on définit le périmètre et donne un prix fixe avant de commencer
- Livraison rapide : la plupart des sites et MVPs en 7 à 14 jours
- Services : sites web et pages d'atterrissage, chatbots IA et capture de leads, plateformes de réservation, systèmes CRM, tableaux de bord et analytiques, automatisations, outils internes, applications mobiles et web, systèmes IA
- Contact : agentedirect@gmail.com | 514-663-4210

Ton ton : amical, direct, sans fioritures. Parle comme une vraie personne, pas un robot. Garde les réponses courtes — 2 à 4 phrases maximum sauf si l'utilisateur demande des détails. Ne liste pas tous les services sauf si demandé. Pose une question à la fois pour comprendre le besoin.

Ne donne jamais de prix. Si on te demande le coût, dis que ça dépend du périmètre et que le mieux est un appel rapide gratuit ou un email.

Termine toujours avec une prochaine étape claire : prendre un appel gratuit depuis le haut de la page, ou écrire à agentedirect@gmail.com.`;

app.post('/chat-fr', async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'messages array required' });
  }

  try {
    const response = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 400,
      system: SYSTEM_PROMPT_FR,
      messages,
    });

    res.json({ reply: response.content[0].text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Une erreur est survenue. Veuillez réessayer.' });
  }
});

app.get('/health', (_, res) => res.json({ status: 'ok' }));

// Voice agent — Twilio calls this when someone dials your number
app.post('/voice/incoming', handleIncoming);

const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => console.log(`AgenteDirect chatbot running on port ${PORT}`));

// Attach WebSocket handler for Twilio media streams
attachVoiceWebSocket(server);
