import { NextRequest, NextResponse } from 'next/server';
import { Resend } from 'resend';

const resend = new Resend(process.env.RESEND_API_KEY);

interface InquiryBody {
  name?: string;
  businessName?: string;
  email?: string;
  currentUrl?: string;
  businessDescription?: string;
  plan?: string;
  lang?: string;
}

export async function POST(req: NextRequest) {
  let body: InquiryBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body.' }, { status: 400 });
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!body.name?.trim())
    return NextResponse.json({ error: 'Name is required.' }, { status: 422 });
  if (!body.businessName?.trim())
    return NextResponse.json({ error: 'Business name is required.' }, { status: 422 });
  if (!body.email?.trim())
    return NextResponse.json({ error: 'Email is required.' }, { status: 422 });
  if (!emailRegex.test(body.email ?? ''))
    return NextResponse.json({ error: 'Invalid email address.' }, { status: 422 });
  if (!body.businessDescription?.trim())
    return NextResponse.json({ error: 'Business description is required.' }, { status: 422 });

  const planLabels: Record<string, string> = {
    launch: 'Website Launch',
    growth: 'Growth System',
    pro: 'Growth Pro',
  };
  const planLabel = planLabels[body.plan ?? ''] ?? body.plan ?? 'Not specified';

  const text = [
    `New inquiry from ${body.name} at ${body.businessName}`,
    ``,
    `Name: ${body.name}`,
    `Business: ${body.businessName}`,
    `Email: ${body.email}`,
    `Current website: ${body.currentUrl || 'None'}`,
    `Interested in: ${planLabel}`,
    ``,
    `What they said:`,
    body.businessDescription,
  ].join('\n');

  const html = `
    <div style="font-family:sans-serif;max-width:560px;margin:0 auto;color:#111">
      <h2 style="margin:0 0 24px;font-size:20px">New inquiry — ${body.businessName}</h2>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <tr><td style="padding:8px 0;color:#666;width:140px">Name</td><td style="padding:8px 0;font-weight:600">${body.name}</td></tr>
        <tr><td style="padding:8px 0;color:#666">Business</td><td style="padding:8px 0;font-weight:600">${body.businessName}</td></tr>
        <tr><td style="padding:8px 0;color:#666">Email</td><td style="padding:8px 0"><a href="mailto:${body.email}">${body.email}</a></td></tr>
        <tr><td style="padding:8px 0;color:#666">Current site</td><td style="padding:8px 0">${body.currentUrl ? `<a href="${body.currentUrl}">${body.currentUrl}</a>` : 'None'}</td></tr>
        <tr><td style="padding:8px 0;color:#666">Interested in</td><td style="padding:8px 0;font-weight:600">${planLabel}</td></tr>
      </table>
      <div style="margin-top:24px;padding:16px;background:#f5f5f5;border-radius:8px;font-size:14px;line-height:1.6">
        <p style="margin:0 0 6px;color:#666;font-size:12px;text-transform:uppercase;letter-spacing:.05em">Their vision</p>
        <p style="margin:0">${body.businessDescription?.replace(/\n/g, '<br>')}</p>
      </div>
      <p style="margin-top:24px;font-size:12px;color:#999">Reply directly to this email to reach ${body.name}.</p>
    </div>
  `;

  const { error } = await resend.emails.send({
    from: process.env.FROM_EMAIL ?? 'AgenteDirect <onboarding@resend.dev>',
    to: process.env.NOTIFY_EMAIL ?? 'support@agentedirect.ca',
    replyTo: body.email,
    subject: `New inquiry: ${body.businessName}`,
    text,
    html,
  });

  if (error) {
    console.error('Resend error:', error);
    return NextResponse.json({ error: 'Failed to send. Please email us directly.' }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
