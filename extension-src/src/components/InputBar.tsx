import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { Brain, Globe, FileText, X, Paperclip, Mic, Hash, Zap, Image as ImageIcon, FileCode, RefreshCw, BookOpen } from 'lucide-react';
import { useStore } from '../store';
import { getPageContentCached, clearPageContentCache } from '../ext';
import ModelPill from './ModelPill';
import { processFile } from './DropZone';
import type { ThinkingLevel, FileAttachment } from '../types';

// Web Speech API types
interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition?: new () => SpeechRecognitionInstance;
  }
}

const THINKING_LABELS: Record<ThinkingLevel, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
};

const THINKING_TOOLTIPS: Record<ThinkingLevel, string> = {
  low: 'Quick reasoning check',
  medium: 'Step-by-step analysis',
  high: 'Deep multi-perspective reasoning',
};

const THINKING_CYCLE: ThinkingLevel[] = ['low', 'medium', 'high'];

/* ── Slash command presets (imported from data module) ── */
import { SLASH_COMMANDS } from '../data/slash-commands';

/* ── Autocomplete item type ── */
interface AutocompleteItem {
  id: string;
  label: string;
  sublabel?: string;
  icon?: string;
  type: 'model' | 'slash';
  value: string; // model name or template text
}

interface Props {
  onSend: (text: string, overrideModel?: string) => void;
  featureKey?: string;
  placeholder?: string;
  disabled?: boolean;
  fileAttachments?: FileAttachment[];
  onRemoveAttachment?: (id: string) => void;
  onFilesAdded?: (files: FileAttachment[]) => void;
}

const PLACEHOLDER_SUGGESTIONS = [
  'Ask anything...',
  'Summarize this page...',
  'Translate to...',
  'Explain this code...',
  'Find key points...',
];

const MAX_VISIBLE_LINES = 6;
const LINE_HEIGHT_PX = 20.25; // 13.5px * 1.5
const MAX_TEXTAREA_HEIGHT = Math.ceil(MAX_VISIBLE_LINES * LINE_HEIGHT_PX) + 20; // + padding
const MAX_POPUP_ITEMS = 6;

export default function InputBar({ onSend, featureKey = 'chat', placeholder, disabled, fileAttachments = [], onRemoveAttachment, onFilesAdded }: Props) {
  const { thinkingMode, setThinkingMode, thinkingLevel, setThinkingLevel, deepResearch, setDeepResearch, activeStream, setPendingCtx, pendingCtx, pageContextEnabled, setPageContextEnabled, pageContext, setPageContext, agentRunning, stopAgentTask } = useStore();
  const [showThinkTooltip, setShowThinkTooltip] = useState(false);
  const thinkLongPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Global hotkey focus event (fired by background.ts focus-chat command)
  useEffect(() => {
    const handler = () => {
      const el = textareaRef.current;
      if (el) {
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
      }
    };
    window.addEventListener('aura-focus-input', handler);
    return () => window.removeEventListener('aura-focus-input', handler);
  }, []);

  // Voice dictate hotkey event (fired by background.ts dictate command)
  useEffect(() => {
    const handler = () => {
      const el = textareaRef.current;
      if (el) el.focus();
      // Ref-stable: toggleRecording is defined below; use a slight delay so the
      // effect re-reads the current function via a window event.
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('aura-inputbar-toggle-mic'));
      }, 50);
    };
    window.addEventListener('aura-dictate-start', handler);
    return () => window.removeEventListener('aura-dictate-start', handler);
  }, []);
  const [pageLoading, setPageLoading] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [hasText, setHasText] = useState(false);
  const [charCount, setCharCount] = useState(0);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [placeholderFade, setPlaceholderFade] = useState(true);

  // Autocomplete state
  const [acOpen, setAcOpen] = useState(false);
  const [acType, setAcType] = useState<'model' | 'slash' | null>(null);
  const [acQuery, setAcQuery] = useState('');
  const [acIndex, setAcIndex] = useState(0);
  const [acTriggerPos, setAcTriggerPos] = useState(0); // char index where @ or / was typed
  const [mentionedModel, setMentionedModel] = useState<string | null>(null);
  const popupRef = useRef<HTMLDivElement>(null);

  const isStreaming = !!activeStream;

  // Build model list from store for @ mentions
  const { mdlCloudList, mdlLocalList, mdlChatgptList, loadModels } = useStore();

  const allModelItems = useMemo<AutocompleteItem[]>(() => {
    const items: AutocompleteItem[] = [];
    for (const m of mdlChatgptList) {
      items.push({
        id: m,
        label: m.replace(/^chatgpt:/, ''),
        sublabel: 'chatgpt',
        type: 'model',
        value: m,
      });
    }
    for (const m of mdlCloudList) {
      items.push({
        id: m,
        label: m.replace(/:cloud$/, ''),
        sublabel: 'cloud',
        type: 'model',
        value: m,
      });
    }
    for (const m of mdlLocalList) {
      items.push({
        id: m,
        label: m,
        sublabel: 'local',
        type: 'model',
        value: m,
      });
    }
    return items;
  }, [mdlCloudList, mdlLocalList, mdlChatgptList]);

  // Build slash command items
  const slashItems = useMemo<AutocompleteItem[]>(() => {
    return SLASH_COMMANDS.map(s => ({
      id: `slash-${s.cmd}`,
      label: s.cmd,
      sublabel: s.template.slice(0, 40),
      type: 'slash' as const,
      value: s.template,
    }));
  }, []);

  // Filtered items based on query
  const acItems = useMemo<AutocompleteItem[]>(() => {
    if (!acOpen || !acType) return [];
    const q = acQuery.toLowerCase();
    let source = acType === 'model' ? allModelItems : slashItems;
    if (q) {
      source = source.filter(item =>
        item.label.toLowerCase().includes(q) ||
        (item.sublabel && item.sublabel.toLowerCase().includes(q))
      );
    }
    return source.slice(0, MAX_POPUP_ITEMS);
  }, [acOpen, acType, acQuery, allModelItems, slashItems]);

  // Keep acIndex in bounds
  useEffect(() => {
    if (acIndex >= acItems.length) {
      setAcIndex(Math.max(0, acItems.length - 1));
    }
  }, [acItems.length, acIndex]);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const next = Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT);
    el.style.height = next + 'px';
    el.style.overflowY = el.scrollHeight > MAX_TEXTAREA_HEIGHT ? 'auto' : 'hidden';
  }, []);

  // Detect @ and / triggers in textarea value
  const detectAutocomplete = useCallback((val: string, cursorPos: number) => {
    // Check for @ mention trigger
    // Look backwards from cursor for an unfinished @mention
    const beforeCursor = val.slice(0, cursorPos);

    // @ trigger: find last @ that is at start or preceded by whitespace, with no space after it
    const atMatch = beforeCursor.match(/(^|[\s])@([^\s]*)$/);
    if (atMatch) {
      const query = atMatch[2]; // text after @
      const triggerPos = beforeCursor.length - query.length - 1; // position of @
      // Ensure models are loaded
      if (!allModelItems.length) loadModels();
      setAcType('model');
      setAcQuery(query);
      setAcTriggerPos(triggerPos);
      setAcOpen(true);
      setAcIndex(0);
      return;
    }

    // / trigger: only at the very start of input or after a newline
    const slashMatch = beforeCursor.match(/(^|\n)\/([^\s]*)$/);
    if (slashMatch) {
      const query = slashMatch[2];
      const triggerPos = beforeCursor.length - query.length - 1;
      setAcType('slash');
      setAcQuery(query);
      setAcTriggerPos(triggerPos);
      setAcOpen(true);
      setAcIndex(0);
      return;
    }

    // No trigger found — close popup
    if (acOpen) {
      setAcOpen(false);
      setAcType(null);
      setAcQuery('');
    }
  }, [acOpen, allModelItems.length, loadModels]);

  const handleInput = useCallback(() => {
    autoResize();
    const el = textareaRef.current;
    if (!el) return;
    const val = el.value;
    setHasText(val.trim().length > 0);
    setCharCount(val.length);
    detectAutocomplete(val, el.selectionStart ?? val.length);
  }, [autoResize, detectAutocomplete]);

  // Select an autocomplete item
  const selectAcItem = useCallback((item: AutocompleteItem) => {
    const el = textareaRef.current;
    if (!el) return;

    const val = el.value;
    const before = val.slice(0, acTriggerPos); // text before the @ or /
    const after = val.slice(el.selectionStart ?? val.length); // text after cursor

    if (item.type === 'model') {
      // Replace @query with @displayName, store model for send
      const displayName = item.label;
      el.value = before + '@' + displayName + ' ' + after;
      setMentionedModel(item.value);
      // Move cursor after the inserted text
      const newPos = before.length + 1 + displayName.length + 1;
      el.setSelectionRange(newPos, newPos);
    } else {
      // Slash: replace /command with the template
      el.value = before + item.value + after;
      const newPos = before.length + item.value.length;
      el.setSelectionRange(newPos, newPos);
    }

    setAcOpen(false);
    setAcType(null);
    setAcQuery('');
    setHasText(el.value.trim().length > 0);
    setCharCount(el.value.length);
    autoResize();
    el.focus();
  }, [acTriggerPos, autoResize]);

  const dismissAc = useCallback(() => {
    setAcOpen(false);
    setAcType(null);
    setAcQuery('');
  }, []);

  // --- Clipboard paste handler for images ---
  const handlePaste = useCallback(async (e: React.ClipboardEvent) => {
    const items = Array.from(e.clipboardData.items);
    const imageItems = items.filter(item => item.type.startsWith('image/'));

    if (imageItems.length === 0 || !onFilesAdded) return;

    e.preventDefault();
    const files: FileAttachment[] = [];
    for (const item of imageItems) {
      const file = item.getAsFile();
      if (!file) continue;
      const attachment = await processFile(file);
      if (attachment) files.push(attachment);
    }
    if (files.length > 0) onFilesAdded(files);
  }, [onFilesAdded]);

  // --- Speech recognition ---
  const [isRecording, setIsRecording] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const [showMicTooltip, setShowMicTooltip] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
    }
  }, []);

  const toggleRecording = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      setShowMicTooltip(true);
      setTimeout(() => setShowMicTooltip(false), 2500);
      return;
    }

    if (isRecording && recognitionRef.current) {
      recognitionRef.current.stop();
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognitionRef.current = recognition;

    // Track the baseline length so we only append new speech
    const baseLength = textareaRef.current?.value.length ?? 0;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      if (textareaRef.current) {
        const base = textareaRef.current.value.slice(0, baseLength);
        const sep = base.length > 0 && !base.endsWith(' ') ? ' ' : '';
        textareaRef.current.value = base + sep + transcript;
        handleInput();
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error !== 'aborted') {
        console.warn('Speech recognition error:', event.error);
      }
      setIsRecording(false);
      recognitionRef.current = null;
    };

    recognition.onend = () => {
      setIsRecording(false);
      recognitionRef.current = null;
    };

    recognition.start();
    setIsRecording(true);
  }, [isRecording, handleInput]);

  // Bridge the two useEffects above — toggleRecording is defined here, and
  // the dictate hotkey useEffect fires `aura-inputbar-toggle-mic` instead of
  // calling it directly to avoid stale-closure issues.
  useEffect(() => {
    const h = () => toggleRecording();
    window.addEventListener('aura-inputbar-toggle-mic', h);
    return () => window.removeEventListener('aura-inputbar-toggle-mic', h);
  }, [toggleRecording]);

  // Stop recording on Escape key
  useEffect(() => {
    if (!isRecording) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  // Cycle placeholder suggestions with crossfade
  useEffect(() => {
    if (placeholder) return;
    const interval = setInterval(() => {
      setPlaceholderFade(false);
      setTimeout(() => {
        setPlaceholderIdx(prev => (prev + 1) % PLACEHOLDER_SUGGESTIONS.length);
        setPlaceholderFade(true);
      }, 250);
    }, 3500);
    return () => clearInterval(interval);
  }, [placeholder]);

  const currentPlaceholder = placeholder || PLACEHOLDER_SUGGESTIONS[placeholderIdx];

  const addPageCtx = async () => {
    setPageLoading(true);
    try {
      const resp = await getPageContentCached();
      if (resp?.ok && resp.text) {
        setPendingCtx({ text: resp.text.slice(0, 20000), title: resp.title, url: resp.url, action: 'ask' });
      }
    } finally {
      setPageLoading(false);
    }
  };

  const refreshPageCtx = async () => {
    setPageLoading(true);
    try {
      clearPageContentCache();
      const resp = await getPageContentCached();
      if (resp?.ok && resp.text) {
        setPendingCtx({ text: resp.text.slice(0, 20000), title: resp.title, url: resp.url, action: 'ask' });
      }
    } finally {
      setPageLoading(false);
    }
  };

  const togglePageContext = useCallback(async () => {
    const next = !pageContextEnabled;
    setPageContextEnabled(next);
    if (next) {
      setPageLoading(true);
      try {
        clearPageContentCache();
        const resp = await getPageContentCached();
        if (resp?.ok && resp.text) {
          setPageContext({ text: resp.text.slice(0, 20000), title: resp.title, url: resp.url });
        } else {
          // Failed to get page — turn off
          setPageContextEnabled(false);
          setPageContext(null);
        }
      } finally {
        setPageLoading(false);
      }
    } else {
      setPageContext(null);
    }
  }, [pageContextEnabled, setPageContextEnabled, setPageContext]);

  // Drag-and-drop on input area
  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounter = useRef(0);
  const handleInputDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.types.includes('Files')) { dragCounter.current++; setIsDragOver(true); }
  };
  const handleInputDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current <= 0) { dragCounter.current = 0; setIsDragOver(false); }
  };
  const handleInputDragOver = (e: React.DragEvent) => { e.preventDefault(); };
  const handleInputDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    for (const file of files) {
      const attachment = await processFile(file);
      if (attachment && onFilesAdded) onFilesAdded([attachment]);
    }
  };

  const handleSend = () => {
    const raw = textareaRef.current?.value.trim();
    if (!raw || isStreaming || disabled) return;

    // Strip @mention from the sent text (clean user message)
    let text = raw;
    let modelOverride = mentionedModel;

    // Find and remove @modelName patterns from the text
    if (modelOverride) {
      // Remove the @displayName from text (find the mention and strip it)
      const matchingItem = allModelItems.find(m => m.value === modelOverride);
      if (matchingItem) {
        const pattern = '@' + matchingItem.label;
        text = text.replace(pattern, '').replace(/\s{2,}/g, ' ').trim();
      }
    }

    onSend(text, modelOverride || undefined);

    if (textareaRef.current) {
      textareaRef.current.value = '';
      setHasText(false);
      setCharCount(0);
      setMentionedModel(null);
      autoResize();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // If autocomplete is open, intercept keyboard navigation
    if (acOpen && acItems.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setAcIndex(prev => (prev + 1) % acItems.length);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setAcIndex(prev => (prev - 1 + acItems.length) % acItems.length);
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        selectAcItem(acItems[acIndex]);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        dismissAc();
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearMentionedModel = () => {
    setMentionedModel(null);
    // Also remove @mention text from textarea if present
    const el = textareaRef.current;
    if (el) {
      const matchingItem = allModelItems.find(m => m.value === mentionedModel);
      if (matchingItem) {
        const pattern = '@' + matchingItem.label;
        el.value = el.value.replace(pattern, '').replace(/\s{2,}/g, ' ').trim();
        setHasText(el.value.trim().length > 0);
        setCharCount(el.value.length);
        autoResize();
      }
    }
  };

  const canSend = hasText && !isStreaming && !disabled;

  // Derive favicon URL from pending context
  const faviconUrl = pendingCtx?.url
    ? (() => {
        try {
          const u = new URL(pendingCtx.url);
          return `https://www.google.com/s2/favicons?domain=${u.hostname}&sz=32`;
        } catch {
          return null;
        }
      })()
    : null;

  return (
    <div className="input-bar-root">
      {/* File attachments area */}
      {fileAttachments.length > 0 && (
        <div className="input-attachments">
          {fileAttachments.map(att => (
            <div key={att.id} className="input-attachment-pill">
              {att.type === 'image' ? (
                att.data ? (
                  <img
                    src={`data:${att.mimeType};base64,${att.data}`}
                    alt={att.name}
                    className="input-attachment-thumb"
                  />
                ) : (
                  <ImageIcon size={10} />
                )
              ) : att.type === 'pdf' ? (
                <FileText size={10} />
              ) : (
                <FileCode size={10} />
              )}
              <span className="input-attachment-name">{att.name}</span>
              <span className="input-attachment-size">
                {att.size < 1024 ? `${att.size}B` : `${(att.size / 1024).toFixed(0)}KB`}
              </span>
              {onRemoveAttachment && (
                <button
                  className="input-attachment-remove"
                  onClick={() => onRemoveAttachment(att.id)}
                  aria-label="Remove attachment"
                >
                  <X size={10} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Mentioned model indicator */}
      {mentionedModel && (
        <div className="input-mention-bar">
          <Zap size={11} />
          <span className="input-mention-label">
            Using <strong>{mentionedModel.replace(/:cloud$/, '').replace(/^chatgpt:/, '')}</strong> for this message
          </span>
          <button
            onClick={clearMentionedModel}
            className="input-mention-remove"
            aria-label="Remove model override"
          >
            <X size={11} />
          </button>
        </div>
      )}

      {/* Context bar */}
      {pendingCtx && (
        <div className="input-ctx-bar">
          {faviconUrl && (
            <img
              src={faviconUrl}
              alt=""
              width={14}
              height={14}
              className="input-ctx-favicon"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          )}
          <FileText size={12} className="input-ctx-icon" />
          <span className="input-ctx-title">
            {pendingCtx.title || pendingCtx.url || 'Page context'}
          </span>
          <button
            onClick={() => setPendingCtx(null)}
            className="input-ctx-remove"
            aria-label="Remove context"
          >
            <X size={12} />
          </button>
        </div>
      )}

      {/* Page context active indicator */}
      {pageContextEnabled && pageContext && (
        <div className="input-page-ctx-bar">
          <BookOpen size={11} className="input-page-ctx-icon" />
          <span className="input-page-ctx-label">
            Chatting with: <strong>{pageContext.title ? (pageContext.title.length > 40 ? pageContext.title.slice(0, 40) + '…' : pageContext.title) : (pageContext.url || 'this page')}</strong>
          </span>
          <button
            onClick={() => { setPageContextEnabled(false); setPageContext(null); }}
            className="input-ctx-remove"
            aria-label="Disable page context"
          >
            <X size={11} />
          </button>
        </div>
      )}

      {/* Glass input wrapper */}
      <div
        className={`input-glass-wrap ${isFocused ? 'input-focused' : ''} ${hasText ? 'has-text' : ''} ${isRecording ? 'input-recording' : ''}`}
        onDragEnter={handleInputDragEnter}
        onDragLeave={handleInputDragLeave}
        onDragOver={handleInputDragOver}
        onDrop={handleInputDrop}
        style={isDragOver ? { borderColor: 'var(--p)', borderStyle: 'dashed' } : undefined}
      >

        {/* Autocomplete popup — positioned above textarea */}
        {acOpen && acItems.length > 0 && (
          <div ref={popupRef} className="ac-popup">
            {acItems.map((item, i) => (
              <div
                key={item.id}
                className={`ac-popup-item ${i === acIndex ? 'ac-popup-item-active' : ''}`}
                onMouseEnter={() => setAcIndex(i)}
                onMouseDown={(e) => {
                  e.preventDefault(); // prevent textarea blur
                  selectAcItem(item);
                }}
              >
                {item.type === 'model' ? (
                  <>
                    <Hash size={11} className="ac-popup-icon" />
                    <span className="ac-popup-label">{item.label}</span>
                    {item.sublabel && (
                      <span className={`ac-popup-badge ac-badge-${item.sublabel}`}>
                        {item.sublabel}
                      </span>
                    )}
                  </>
                ) : (
                  <>
                    <span className="ac-popup-emoji">{item.icon}</span>
                    <span className="ac-popup-label">{item.label}</span>
                    {item.sublabel && (
                      <span className="ac-popup-sublabel">{item.sublabel}</span>
                    )}
                  </>
                )}
              </div>
            ))}
            <div className="ac-popup-hint">
              {acType === 'model' ? (
                <><kbd>↑↓</kbd> navigate <kbd>↵</kbd> select <kbd>esc</kbd> dismiss</>
              ) : (
                <><kbd>↑↓</kbd> navigate <kbd>↵</kbd> insert <kbd>esc</kbd> dismiss</>
              )}
            </div>
          </div>
        )}

        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={currentPlaceholder}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled || isStreaming}
          className={`input-textarea ${placeholderFade ? 'placeholder-visible' : 'placeholder-hidden'}`}
          onFocus={() => setIsFocused(true)}
          onBlur={() => {
            setIsFocused(false);
            // Delay dismissing AC so click events on popup items fire first
            setTimeout(() => {
              if (!popupRef.current?.contains(document.activeElement)) {
                dismissAc();
              }
            }, 150);
          }}
        />

        {/* Action row */}
        <div className="input-action-row">
          {/* Left side: pills */}
          <div className="input-action-left">
            <button
              onClick={() => {
                if (!thinkingMode) {
                  setThinkingMode(true);
                } else {
                  setThinkingMode(false);
                }
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                if (thinkingMode) {
                  // Cycle level on right-click
                  const idx = THINKING_CYCLE.indexOf(thinkingLevel);
                  setThinkingLevel(THINKING_CYCLE[(idx + 1) % THINKING_CYCLE.length]);
                } else {
                  setThinkingMode(true);
                }
              }}
              onMouseDown={() => {
                // Long-press to cycle level
                thinkLongPressTimer.current = setTimeout(() => {
                  thinkLongPressTimer.current = null;
                  if (thinkingMode) {
                    const idx = THINKING_CYCLE.indexOf(thinkingLevel);
                    setThinkingLevel(THINKING_CYCLE[(idx + 1) % THINKING_CYCLE.length]);
                  } else {
                    setThinkingMode(true);
                  }
                }, 500);
              }}
              onMouseUp={() => {
                if (thinkLongPressTimer.current) {
                  clearTimeout(thinkLongPressTimer.current);
                  thinkLongPressTimer.current = null;
                }
              }}
              onMouseLeave={() => {
                if (thinkLongPressTimer.current) {
                  clearTimeout(thinkLongPressTimer.current);
                  thinkLongPressTimer.current = null;
                }
                setShowThinkTooltip(false);
              }}
              onMouseEnter={() => setShowThinkTooltip(true)}
              className={`input-pill ${thinkingMode ? 'input-pill-think-active' : ''}`}
              title=""
            >
              <Brain size={11} />
              <span>{thinkingMode ? `Think: ${THINKING_LABELS[thinkingLevel]}` : 'Think'}</span>
              {showThinkTooltip && (
                <div className="think-tooltip">
                  {thinkingMode
                    ? `${THINKING_TOOLTIPS[thinkingLevel]} — right-click to cycle level`
                    : 'Enable chain-of-thought reasoning'}
                </div>
              )}
            </button>
            <button
              onClick={() => setDeepResearch(!deepResearch)}
              className={`input-pill ${deepResearch ? 'input-pill-active' : ''}`}
              title="Deep research mode"
            >
              <Globe size={11} />
              <span>Research</span>
            </button>
            <button
              onClick={addPageCtx}
              disabled={pageLoading}
              className={`input-pill ${pendingCtx ? 'input-pill-active' : ''}`}
              title="Add page content as context"
              style={{ opacity: pageLoading ? 0.5 : 1 }}
            >
              <FileText size={11} />
              <span>{pageLoading ? '...' : 'Page'}</span>
            </button>
            {pendingCtx && (
              <button
                onClick={refreshPageCtx}
                disabled={pageLoading}
                className="input-pill"
                title="Refresh page context"
                style={{ opacity: pageLoading ? 0.5 : 1, padding: '2px 5px' }}
              >
                <RefreshCw size={10} />
              </button>
            )}
            <button
              onClick={togglePageContext}
              disabled={pageLoading}
              className={`input-pill ${pageContextEnabled ? 'input-pill-page-ctx-active' : ''}`}
              title={pageContextEnabled ? 'Disable: Chat with this page' : 'Enable: Chat with this page'}
              style={{ opacity: pageLoading ? 0.5 : 1 }}
            >
              <BookOpen size={11} />
              <span>This page</span>
            </button>
          </div>

          {/* Right side: char count, kbd hint, model, send */}
          <div className="input-action-right">
            {charCount > 500 && (
              <span className={`input-char-count ${charCount > 4000 ? 'input-char-warn' : ''}`}>
                {charCount.toLocaleString()}
              </span>
            )}
            <ModelPill featureKey={featureKey} />
            {!hasText && (
              <kbd className="input-kbd-hint">
                <span style={{ fontSize: '10px' }}>&#8984;</span>K
              </kbd>
            )}
            <div className="input-mic-wrap">
              <button
                onClick={toggleRecording}
                className={`input-mic-btn ${isRecording ? 'input-mic-recording' : ''}`}
                aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
                title={!speechSupported ? 'Voice not available in this browser' : isRecording ? 'Stop recording (Esc)' : 'Voice input'}
              >
                <Mic size={16} />
              </button>
              {showMicTooltip && !speechSupported && (
                <div className="mic-tooltip">Voice not available in this browser</div>
              )}
            </div>
            {agentRunning ? (
              <button
                onClick={stopAgentTask}
                className="input-send-btn input-send-ready"
                aria-label="Stop agent"
                title="Stop running agent"
                style={{ background: 'rgba(239,68,68,0.18)', color: 'var(--rd)', border: '1px solid var(--rd)' }}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                  <rect x="2" y="2" width="8" height="8" rx="1" />
                </svg>
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!canSend}
                className={`input-send-btn ${canSend ? 'input-send-ready' : ''}`}
                aria-label="Send message"
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M7 11.5V2.5M7 2.5L3 6.5M7 2.5L11 6.5"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Listening indicator */}
      {isRecording && (
        <div className="input-listening-indicator">
          <span className="input-listening-dot" />
          <span>Listening...</span>
        </div>
      )}
    </div>
  );
}
