import { NextRequest, NextResponse } from 'next/server';

interface InquiryBody {
  name?: string;
  businessName?: string;
  email?: string;
  currentUrl?: string;
  businessDescription?: string;
  plan?: string;
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

  return NextResponse.json({ ok: true });
}
