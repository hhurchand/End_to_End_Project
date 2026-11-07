'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatWidget({ lang = 'en' }: { lang?: 'en' | 'fr' }) {
  const fr = lang === 'fr';
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 120);
      if (messages.length === 0) {
        setMessages([{
          role: 'assistant',
          content: fr
            ? 'Bonjour! Posez-moi n\'importe quelle question sur AgenteDirect — comment on travaille, les délais, ce à quoi vous attendre. En français ou en anglais, à vous de choisir 👋'
            : 'Hi! Ask me anything about AgenteDirect — how we work, what to expect, turnaround time, anything at all. English or French, your call 👋',
        }]);
      }
    }
  }, [open]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');

    const next: Message[] = [...messages, { role: 'user', content: text }];
    setMessages(next);
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: next }),
      });

      if (!res.body) throw new Error('No body');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let reply = '';

      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        reply += decoder.decode(value, { stream: true });
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: reply };
          return updated;
        });
      }
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, something went wrong. Email us at support@agentedirect.ca.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Chat panel */}
      {open && (
        <div style={{
          position: 'fixed', bottom: 88, right: 24, zIndex: 1000,
          width: 340, maxHeight: 480,
          background: '#120e1f',
          border: '1px solid rgba(139,92,246,0.25)',
          borderRadius: 16,
          boxShadow: '0 24px 64px rgba(0,0,0,0.7)',
          display: 'flex', flexDirection: 'column',
          fontFamily: 'var(--font-space-grotesk), system-ui, sans-serif',
          overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            padding: '14px 16px', borderBottom: '1px solid rgba(139,92,246,0.12)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: '#0f0b1e',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#7C3AED', boxShadow: '0 0 8px rgba(124,58,237,0.8)' }} />
              <span style={{ fontSize: 13, fontWeight: 600, color: 'rgba(255,255,255,0.85)' }}>AgenteDirect</span>
            </div>
            <button onClick={() => setOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.35)', fontSize: 18, lineHeight: 1, padding: 2 }}>×</button>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px 8px' }}>
            {messages.map((m, i) => (
              <div key={i} style={{
                marginBottom: 10,
                display: 'flex',
                justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
              }}>
                <div style={{
                  maxWidth: '82%',
                  padding: '9px 13px',
                  borderRadius: m.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                  background: m.role === 'user' ? 'rgba(124,58,237,0.7)' : 'rgba(255,255,255,0.06)',
                  color: m.role === 'user' ? '#fff' : 'rgba(255,255,255,0.82)',
                  fontSize: 13.5, lineHeight: 1.55,
                  border: m.role === 'assistant' ? '1px solid rgba(255,255,255,0.06)' : 'none',
                }}>
                  {m.content || <span style={{ opacity: 0.4 }}>▍</span>}
                </div>
              </div>
            ))}
            {loading && messages[messages.length - 1]?.role !== 'assistant' && (
              <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 10 }}>
                <div style={{ padding: '9px 14px', borderRadius: '14px 14px 14px 4px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: 13 }}>▍</span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={{ padding: '10px 12px 12px', borderTop: '1px solid rgba(139,92,246,0.1)' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder={fr ? 'Posez une question...' : 'Ask anything...'}
                className="chat-input"
                style={{
                  flex: 1, background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 10, padding: '9px 13px',
                  fontSize: 13.5, color: 'rgba(255,255,255,0.88)',
                  outline: 'none', fontFamily: 'inherit',
                }}
              />
              <button
                onClick={send}
                disabled={!input.trim() || loading}
                style={{
                  width: 36, height: 36, borderRadius: 10, flexShrink: 0,
                  background: input.trim() && !loading ? '#7C3AED' : 'rgba(124,58,237,0.2)',
                  border: 'none', cursor: input.trim() && !loading ? 'pointer' : 'default',
                  color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'background 0.2s',
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Label */}
      {!open && (
        <div
          onClick={() => setOpen(true)}
          style={{
            position: 'fixed', bottom: 32, right: 84, zIndex: 1001,
            cursor: 'pointer', textAlign: 'right',
            fontFamily: 'var(--font-space-grotesk), system-ui, sans-serif',
          }}
        >
          <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'rgba(255,255,255,0.85)', lineHeight: 1.3 }}>
            {fr ? 'Une question?' : 'Have a question?'}
          </p>
          <p style={{ margin: 0, fontSize: 11.5, color: 'rgba(167,139,250,0.75)', lineHeight: 1.3 }}>
            {fr ? 'Parlez à notre IA' : 'Chat with our AI'}
          </p>
        </div>
      )}

      {/* Bubble button */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          position: 'fixed', bottom: 24, right: 24, zIndex: 1001,
          width: 52, height: 52, borderRadius: '50%',
          background: 'rgba(139,92,246,0.15)',
          border: '1px solid rgba(139,92,246,0.4)', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: 'background 0.2s, border-color 0.2s, transform 0.2s',
        }}
        onMouseEnter={e => { const el = e.currentTarget; el.style.background = 'rgba(139,92,246,0.28)'; el.style.borderColor = 'rgba(139,92,246,0.7)'; el.style.transform = 'scale(1.08)'; }}
        onMouseLeave={e => { const el = e.currentTarget; el.style.background = 'rgba(139,92,246,0.15)'; el.style.borderColor = 'rgba(139,92,246,0.4)'; el.style.transform = 'scale(1)'; }}
        aria-label="Open chat"
      >
        {open ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6l12 12" stroke="white" strokeWidth="2.2" strokeLinecap="round"/>
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </button>
      <style>{`@media (max-width: 767px) { .chat-input { font-size: 16px !important; } }`}</style>
    </>
  );
}
