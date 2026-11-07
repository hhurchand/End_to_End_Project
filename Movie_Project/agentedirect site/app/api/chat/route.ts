import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const SYSTEM = `You are a helpful assistant for AgenteDirect, a web design agency based in Montréal, Québec. You answer questions from potential clients in English or French — whichever language the user writes in. Keep answers honest, warm, and to the point. Never be salesy.

About AgenteDirect:
- We build custom websites for local businesses in Montréal and surrounding areas
- Every project is different — we don't have set packages or fixed pricing. Cost depends on the project, what the client needs, and what we build together
- We always start by getting to know the client and what they want before talking numbers
- We use real photos and real content — no stock photos or generic templates
- Sites go live fast, typically within a week
- We stay involved after launch — updates, seasonal changes, all handled by us
- No long-term contracts. Clients can walk away if it's not working
- Clients always own their domain
- Contact: support@agentedirect.ca

Keep all responses short — 2-3 sentences max. If someone asks about pricing, be honest: it depends on the project, we'd rather talk first. Direct them to fill out the form or email us. Never invent numbers. If you don't know something, say so.`;

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const stream = await client.messages.stream({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 180,
      system: SYSTEM,
      messages,
    });

    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        for await (const chunk of stream) {
          if (
            chunk.type === 'content_block_delta' &&
            chunk.delta.type === 'text_delta'
          ) {
            controller.enqueue(encoder.encode(chunk.delta.text));
          }
        }
        controller.close();
      },
    });

    return new Response(readable, {
      headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
  } catch (err) {
    console.error('Chat API error:', err);
    return new Response('Something went wrong.', { status: 500 });
  }
}
