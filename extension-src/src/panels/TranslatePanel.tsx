import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Copy, Check, ArrowLeftRight, Globe, Languages, X } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { sendMsg } from '../ext';
import ext from '../ext';

const LANGS = [
  'auto',
  'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
  'Russian', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Japanese', 'Korean',
  'Arabic', 'Hindi', 'Turkish', 'Vietnamese', 'Thai', 'Indonesian',
  'Polish', 'Dutch', 'Swedish', 'Norwegian', 'Danish', 'Finnish',
  'Greek', 'Czech', 'Romanian', 'Hungarian', 'Ukrainian', 'Hebrew', 'Bengali',
  'Azerbaijani', 'Persian', 'Malay', 'Filipino', 'Swahili',
];

const STORAGE_KEY = 'aura_translate_langs';

function wordCount(text: string): number {
  if (!text.trim()) return 0;
  return text.trim().split(/\s+/).length;
}

export default function TranslatePanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [from, setFrom] = useState('auto');
  const [to, setTo] = useState('English');
  const [inputText, setInputText] = useState('');
  const [output, setOutput] = useState('');
  const [copied, setCopied] = useState(false);
  const [pageTranslating, setPageTranslating] = useState(false);
  const [pageTranslateActive, setPageTranslateActive] = useState(false);
  const [pageTranslateMode, setPageTranslateMode] = useState<'bilingual' | 'translated'>('bilingual');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load saved language pair on mount
  useEffect(() => {
    ext?.storage?.local?.get([STORAGE_KEY], (d: any) => {
      const saved = d?.[STORAGE_KEY];
      if (saved) {
        if (saved.from && LANGS.includes(saved.from)) setFrom(saved.from);
        if (saved.to && LANGS.includes(saved.to)) setTo(saved.to);
      }
    });
  }, []);

  // Save language pair whenever it changes
  useEffect(() => {
    ext?.storage?.local?.set({ [STORAGE_KEY]: { from, to } });
  }, [from, to]);

  const swapLangs = () => {
    if (from === 'auto') return; // can't swap auto-detect
    const oldFrom = from;
    const oldTo = to;
    setFrom(oldTo);
    setTo(oldFrom);
    // Also swap text if there's output
    if (output) {
      setInputText(output);
      setOutput(inputText);
    }
  };

  const doTranslate = useCallback((text?: string) => {
    const toTranslate = text ?? inputText.trim();
    if (!toTranslate) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) {
      setOutput('AURA is offline.');
      return;
    }
    if (activeStream) return;

    setOutput('Translating...');
    const fromLang = from === 'auto' ? 'the detected language' : from;
    const prompt = `Translate from ${fromLang} to ${to}. Output only the translation, no explanation:\n\n${toTranslate}`;

    setActiveStream({
      type: 'translate',
      rawText: '',
      onFirstChunk: () => setOutput(''),
      onDone: (rawText) => setOutput(rawText),
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('translate'),
      conversation_id: null,
    }));
  }, [inputText, from, to, wsReady, ws, activeStream, setActiveStream, getModel]);

  // Full page translate — sends PAGE_TRANSLATE to content script via background
  const translateFullPage = async () => {
    setPageTranslating(true);
    setPageTranslateActive(false);
    setOutput('');

    try {
      // Get active tab and send message to content script
      const tabResp = await sendMsg({ type: 'GET_CURRENT_TAB' });
      if (!tabResp?.ok) {
        setOutput('No active tab found.');
        setPageTranslating(false);
        return;
      }

      // Send PAGE_TRANSLATE to the content script of the active tab
      const resp = await new Promise<any>((resolve) => {
        ext.tabs.query({ active: true, currentWindow: true }, (tabs: any[]) => {
          const tab = tabs?.[0];
          if (!tab?.id) {
            resolve({ ok: false, error: 'No active tab' });
            return;
          }
          ext.tabs.sendMessage(tab.id, { type: 'PAGE_TRANSLATE', targetLang: to }, (r: any) => {
            if (ext.runtime.lastError) {
              resolve({ ok: false, error: ext.runtime.lastError.message });
            } else {
              resolve(r || { ok: false, error: 'No response' });
            }
          });
        });
      });

      if (resp?.ok) {
        setPageTranslateActive(true);
        setPageTranslateMode('bilingual');
      } else {
        setOutput(resp?.error || 'Page translation failed. Try on a regular website.');
      }
    } catch {
      setOutput('Failed to start page translation.');
    }
    setPageTranslating(false);
  };

  const togglePageMode = async () => {
    const newMode = pageTranslateMode === 'bilingual' ? 'translated' : 'bilingual';
    setPageTranslateMode(newMode);

    ext.tabs.query({ active: true, currentWindow: true }, (tabs: any[]) => {
      const tab = tabs?.[0];
      if (tab?.id) {
        ext.tabs.sendMessage(tab.id, { type: 'TRANSLATE_TOGGLE_MODE', mode: newMode }, () => {});
      }
    });
  };

  const removePageTranslation = async () => {
    setPageTranslateActive(false);
    setPageTranslateMode('bilingual');

    ext.tabs.query({ active: true, currentWindow: true }, (tabs: any[]) => {
      const tab = tabs?.[0];
      if (tab?.id) {
        ext.tabs.sendMessage(tab.id, { type: 'TRANSLATE_REMOVE' }, () => {});
      }
    });
  };

  // Legacy "Page" button for in-panel page translation
  const translatePage = async () => {
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) {
      setOutput('AURA is offline.');
      return;
    }
    if (activeStream) return;

    setPageTranslating(true);
    setOutput('Loading page content...');

    try {
      const resp = await sendMsg({ type: 'GET_PAGE_CONTENT' });
      if (!resp?.ok || !resp.text) {
        setOutput('Could not read page. Try on a regular website.');
        setPageTranslating(false);
        return;
      }

      const pageText = resp.text.slice(0, 15000);
      setInputText(pageText.slice(0, 2000) + (pageText.length > 2000 ? '...' : ''));
      setPageTranslating(false);
      doTranslate(pageText);
    } catch {
      setOutput('Failed to load page content.');
      setPageTranslating(false);
    }
  };

  const copyOutput = async () => {
    if (!output) return;
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard may fail in some contexts
    }
  };

  // Track streaming output
  const stream = useStore(s => s.activeStream);
  const streamText = (stream && stream !== true && stream.type === 'translate') ? stream.rawText : null;
  const displayText = streamText !== null ? streamText : output;

  const inputWords = wordCount(inputText);
  const outputWords = wordCount(displayText || '');

  const selectStyle: React.CSSProperties = {
    flex: 1,
    background: 'var(--s2)',
    border: '1px solid var(--b1)',
    borderRadius: 'var(--r-md)',
    color: 'var(--tx)',
    fontSize: '12px',
    padding: '6px 8px',
    fontFamily: 'inherit',
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Lang selectors */}
      <div className="flex items-center gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <select value={from} onChange={e => setFrom(e.target.value)} style={selectStyle}>
          {LANGS.map(l => (
            <option key={l} value={l}>{l === 'auto' ? 'Auto-detect' : l}</option>
          ))}
        </select>

        <button
          onClick={swapLangs}
          disabled={from === 'auto'}
          title="Swap languages"
          style={{
            background: 'none',
            border: 'none',
            cursor: from === 'auto' ? 'not-allowed' : 'pointer',
            color: from === 'auto' ? 'var(--di)' : 'var(--pl)',
            padding: '4px',
            display: 'flex',
            alignItems: 'center',
            flexShrink: 0,
          }}
        >
          <ArrowLeftRight size={16} />
        </button>

        <select value={to} onChange={e => setTo(e.target.value)} style={selectStyle}>
          {LANGS.filter(l => l !== 'auto').map(l => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>
      </div>

      {/* Full Page Translate Section */}
      <div className="flex-shrink-0 p-3" style={{ borderBottom: '1px solid var(--b1)' }}>
        {!pageTranslateActive ? (
          <button
            onClick={translateFullPage}
            disabled={pageTranslating}
            style={{
              width: '100%',
              background: pageTranslating ? 'var(--s3)' : 'linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(124, 58, 237, 0.08))',
              border: '1px solid rgba(124, 58, 237, 0.3)',
              borderRadius: 'var(--r-md)',
              color: pageTranslating ? 'var(--mu)' : 'var(--tx)',
              padding: '10px 14px',
              cursor: pageTranslating ? 'not-allowed' : 'pointer',
              fontSize: '12.5px',
              fontFamily: 'inherit',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              transition: 'all 0.15s ease',
            }}
          >
            <Languages size={16} style={{ color: 'var(--p, #7c3aed)' }} />
            {pageTranslating ? 'Translating page...' : `Translate This Page to ${to}`}
          </button>
        ) : (
          <div style={{
            background: 'rgba(124, 58, 237, 0.08)',
            border: '1px solid rgba(124, 58, 237, 0.25)',
            borderRadius: 'var(--r-md)',
            padding: '8px 12px',
          }}>
            <div className="flex items-center justify-between" style={{ marginBottom: 6 }}>
              <div className="flex items-center gap-2">
                <span style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: '#7c3aed', display: 'inline-block',
                }} />
                <span style={{ fontSize: '11px', color: 'var(--mu)' }}>
                  Page translated to <strong style={{ color: 'var(--p, #7c3aed)' }}>{to}</strong>
                </span>
              </div>
              <button
                onClick={removePageTranslation}
                title="Remove translation"
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--mu)',
                  cursor: 'pointer',
                  padding: 2,
                  display: 'flex',
                  alignItems: 'center',
                  transition: 'color 0.15s',
                }}
                onMouseEnter={e => (e.currentTarget.style.color = 'var(--err, #ef4444)')}
                onMouseLeave={e => (e.currentTarget.style.color = 'var(--mu)')}
              >
                <X size={14} />
              </button>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={togglePageMode}
                style={{
                  flex: 1,
                  background: pageTranslateMode === 'bilingual'
                    ? 'rgba(124, 58, 237, 0.2)'
                    : 'transparent',
                  border: '1px solid rgba(124, 58, 237, 0.2)',
                  borderRadius: 'var(--r-sm)',
                  color: pageTranslateMode === 'bilingual' ? 'var(--tx)' : 'var(--mu)',
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontFamily: 'inherit',
                  transition: 'all 0.15s',
                }}
              >
                Bilingual
              </button>
              <button
                onClick={togglePageMode}
                style={{
                  flex: 1,
                  background: pageTranslateMode === 'translated'
                    ? 'rgba(124, 58, 237, 0.2)'
                    : 'transparent',
                  border: '1px solid rgba(124, 58, 237, 0.2)',
                  borderRadius: 'var(--r-sm)',
                  color: pageTranslateMode === 'translated' ? 'var(--tx)' : 'var(--mu)',
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontFamily: 'inherit',
                  transition: 'all 0.15s',
                }}
              >
                Translated only
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex-1 flex flex-col gap-3 p-3 overflow-hidden">
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <textarea
            ref={inputRef}
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            placeholder="Enter text to translate..."
            autoFocus
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                doTranslate();
              }
            }}
            style={{
              flex: 1,
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--tx)',
              fontSize: '12.5px',
              padding: '10px',
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
            }}
          />
          {inputText && (
            <div style={{ fontSize: '10px', color: 'var(--mu)', textAlign: 'right', marginTop: 2 }}>
              {inputWords} word{inputWords !== 1 ? 's' : ''}
            </div>
          )}
        </div>

        {/* Output */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
            <div
              style={{
                height: '100%',
                background: 'var(--s1)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: '10px',
                paddingRight: '36px',
                fontSize: '12.5px',
                color: displayText ? 'var(--tx)' : 'var(--di)',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {displayText || 'Translation will appear here...'}
            </div>

            {/* Copy button */}
            {displayText && displayText !== 'Translating...' && (
              <button
                onClick={copyOutput}
                title="Copy translation"
                style={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-sm)',
                  color: copied ? 'var(--green, #22c55e)' : 'var(--mu)',
                  cursor: 'pointer',
                  padding: '4px',
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                {copied ? <Check size={13} /> : <Copy size={13} />}
              </button>
            )}
          </div>
          {displayText && displayText !== 'Translating...' && (
            <div style={{ fontSize: '10px', color: 'var(--mu)', textAlign: 'right', marginTop: 2 }}>
              {outputWords} word{outputWords !== 1 ? 's' : ''}
            </div>
          )}
        </div>

        {/* Actions row */}
        <div className="flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <ModelPill featureKey="translate" />
            <button
              onClick={translatePage}
              disabled={!!activeStream || pageTranslating}
              title="Translate page content in panel"
              style={{
                background: 'none',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--mu)',
                padding: '5px 10px',
                cursor: activeStream ? 'not-allowed' : 'pointer',
                fontSize: '11px',
                fontFamily: 'inherit',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <Globe size={12} />
              Page
            </button>
          </div>
          <button
            onClick={() => doTranslate()}
            disabled={!!activeStream || !inputText.trim()}
            style={{
              background: (activeStream || !inputText.trim()) ? 'var(--s3)' : 'var(--p)',
              border: 'none',
              borderRadius: 'var(--r-md)',
              color: 'white',
              padding: '7px 18px',
              cursor: (activeStream || !inputText.trim()) ? 'not-allowed' : 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
            }}
          >
            Translate
          </button>
        </div>
      </div>
    </div>
  );
}
