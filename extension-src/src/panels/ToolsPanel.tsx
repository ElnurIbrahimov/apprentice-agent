import React from 'react';
import { FileText, List, HelpCircle, MessageSquare, Globe, Edit, PenLine } from 'lucide-react';
import { useStore } from '../store';
import { sendMsg } from '../ext';

interface Tool {
  action: string;
  icon: React.ReactNode;
  label: string;
  desc: string;
}

const TOOLS: Tool[] = [
  { action: 'summarize-page', icon: <FileText size={18} />, label: 'Summarize Page', desc: 'Get a concise summary' },
  { action: 'key-points', icon: <List size={18} />, label: 'Key Points', desc: 'Extract 5 main points' },
  { action: 'explain-page', icon: <HelpCircle size={18} />, label: 'Explain Simply', desc: 'In plain language' },
  { action: 'questions', icon: <MessageSquare size={18} />, label: 'Generate Questions', desc: '5 insightful questions' },
  { action: 'deep-research', icon: <Globe size={18} />, label: 'Deep Research', desc: 'Multi-source research' },
  { action: 'fact-check', icon: <HelpCircle size={18} />, label: 'Fact Check', desc: 'Verify claims' },
  { action: 'write-essay', icon: <PenLine size={18} />, label: 'Write Essay', desc: 'Start writing' },
  { action: 'improve-writing', icon: <Edit size={18} />, label: 'Improve Writing', desc: 'Enhance your text' },
];

export default function ToolsPanel() {
  const { setPanel, setPendingCtx } = useStore();

  const handle = async (action: string) => {
    switch (action) {
      case 'summarize-page':
      case 'key-points':
      case 'explain-page':
      case 'questions': {
        const resp = await sendMsg({ type: 'GET_PAGE_CONTENT' });
        if (resp?.ok && resp.text) {
          setPendingCtx({ text: resp.text, title: resp.title, url: resp.url, action: 'ask' });
          setPanel('chat');
          const prompts: Record<string, string> = {
            'summarize-page': 'Please summarize this page concisely.',
            'key-points': 'Extract the 5 most important key points from this page.',
            'explain-page': 'Explain this page content in simple terms for a general audience.',
            'questions': 'Generate 5 insightful questions based on this page content.',
          };
          window.dispatchEvent(new CustomEvent('aura-send', { detail: { text: prompts[action] } }));
        }
        break;
      }
      case 'deep-research':
      case 'fact-check':
        setPanel('search');
        break;
      case 'write-essay':
        setPanel('chat');
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('aura-send', {
            detail: { text: 'Help me start writing an essay. Ask me for the topic and tone, then draft an opening paragraph.' },
          }));
        }, 80);
        break;
      case 'improve-writing':
        setPanel('chat');
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('aura-send', {
            detail: { text: 'I want to improve some writing. Paste the text in your next message and tell me the style or tone you want.' },
          }));
        }, 80);
        break;
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden p-3">
      <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 12 }}>
        Quick Tools
      </div>
      <div className="grid gap-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
        {TOOLS.map(tool => (
          <button
            key={tool.action}
            onClick={() => handle(tool.action)}
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-lg)',
              padding: '14px 12px',
              cursor: 'pointer',
              textAlign: 'left',
              fontFamily: 'inherit',
              transition: 'all 0.15s',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLButtonElement).style.background = 'var(--s3)';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--b2)';
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLButtonElement).style.background = 'var(--s2)';
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--b1)';
            }}
          >
            <span style={{ color: 'var(--pl)' }}>{tool.icon}</span>
            <div>
              <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)' }}>{tool.label}</div>
              <div style={{ fontSize: '11px', color: 'var(--mu)', marginTop: 2 }}>{tool.desc}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
