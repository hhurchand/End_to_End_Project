import { NextRequest, NextResponse } from 'next/server';
import type { QuizAnswers } from '@/lib/types';


export async function POST(req: NextRequest) {
  // Parse body
  let body: QuizAnswers;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body.' }, { status: 400 });
  }

  // Server-side validation
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!body.name?.trim())          return NextResponse.json({ error: 'Name is required.'          }, { status: 422 });
  if (!body.businessName?.trim())  return NextResponse.json({ error: 'Business name is required.' }, { status: 422 });
  if (!body.email?.trim())         return NextResponse.json({ error: 'Email is required.'         }, { status: 422 });
  if (!emailRegex.test(body.email)) return NextResponse.json({ error: 'Invalid email address.'    }, { status: 422 });

  return NextResponse.json({ ok: true });
}
