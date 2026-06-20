import { useState, useCallback } from 'react';
import {
  EllipsisHorizontalIcon,
  DocumentTextIcon,
  CodeBracketIcon,
  GlobeAltIcon,
  ArrowDownTrayIcon,
  LinkIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { exportAsMarkdown, exportAsJSON, exportAsHTML, downloadExport } from '../utils/exportConversation';
import { copyText } from '../utils/clipboard';
import { toast } from './Toast';
import { apiFetch } from '../utils/apiFetch';
import type { Message } from '../types';

interface ChatToolbarProps {
  messages: Message[];
}

export default function ChatToolbar({ messages }: ChatToolbarProps) {
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState(false);

  const [shareModalOpen, setShareModalOpen] = useState(false);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareLoading, setShareLoading] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);

  const handleExportMarkdown = useCallback(() => {
    const md = exportAsMarkdown(messages);
    downloadExport(md, `aura-chat-${Date.now()}.md`, 'text/markdown');
    toast.success('Exported as Markdown');
    setExportMenuOpen(false);
  }, [messages]);

  const handleExportJSON = useCallback(() => {
    const json = exportAsJSON(messages);
    downloadExport(json, `aura-chat-${Date.now()}.json`, 'application/json');
    toast.success('Exported as JSON');
    setExportMenuOpen(false);
  }, [messages]);

  const handleExportHTML = useCallback(() => {
    const html = exportAsHTML(messages);
    downloadExport(html, `aura-chat-${Date.now()}.html`, 'text/html');
    toast.success('Exported as HTML');
    setExportMenuOpen(false);
  }, [messages]);

  const handleCopyConversation = useCallback(async () => {
    const md = exportAsMarkdown(messages);
    if (await copyText(md)) {
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 1500);
    }
    setExportMenuOpen(false);
  }, [messages]);

  const handleShareLink = useCallback(async () => {
    setExportMenuOpen(false);
    setShareUrl(null);
    setShareCopied(false);
    setShareModalOpen(true);
    setShareLoading(true);
    try {
      const html = exportAsHTML(messages);
      const res = await apiFetch('/api/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_name: 'Aura Conversation',
          files: { 'index.html': html },
          entry_point: 'index.html',
          expires_days: 7,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const url = data.url.startsWith('http') ? data.url : `${window.location.origin}${data.url}`;
      setShareUrl(url);
    } catch (e: any) {
      setShareModalOpen(false);
      toast.error('Share failed', e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setShareLoading(false);
    }
  }, [messages]);

  const handleCopyShareUrl = useCallback(async () => {
    if (!shareUrl) return;
    if (await copyText(shareUrl)) {
      setShareCopied(true);
      setTimeout(() => setShareCopied(false), 2000);
    }
  }, [shareUrl]);

  return (
    <>
      {messages.length > 0 && (
        <div className="flex justify-end px-4 py-1.5 relative z-20">
          <div className="relative">
            <button
              onClick={() => setExportMenuOpen(!exportMenuOpen)}
              className="p-1.5 rounded-lg text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.06] transition-colors"
              aria-label="Chat options"
            >
              <EllipsisHorizontalIcon className="w-5 h-5" />
            </button>
            {exportMenuOpen && (
              <>
                <div className="fixed inset-0 z-[40]" onClick={() => setExportMenuOpen(false)} />
                <div className="absolute right-0 top-full mt-1 z-[50] ctx-menu min-w-[190px]">
                  <button className="ctx-menu-item w-full" onClick={handleExportMarkdown}>
                    <DocumentTextIcon className="w-4 h-4" />
                    Export as Markdown
                  </button>
                  <button className="ctx-menu-item w-full" onClick={handleExportJSON}>
                    <CodeBracketIcon className="w-4 h-4" />
                    Export as JSON
                  </button>
                  <button className="ctx-menu-item w-full" onClick={handleExportHTML}>
                    <GlobeAltIcon className="w-4 h-4" />
                    Export as HTML
                  </button>
                  <button className="ctx-menu-item w-full" onClick={handleCopyConversation}>
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    {copyFeedback ? 'Copied!' : 'Copy to Clipboard'}
                  </button>
                  <div className="border-t border-white/[0.06] my-1" />
                  <button className="ctx-menu-item w-full" onClick={handleShareLink}>
                    <LinkIcon className="w-4 h-4" />
                    Share Link
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Share link modal */}
      {shareModalOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4">
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setShareModalOpen(false)} />
          <div
            className="relative w-full max-w-sm rounded-xl shadow-2xl animate-slide-up-fade"
            style={{ background: 'var(--surface-1)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
              <div className="flex items-center gap-2">
                <LinkIcon className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-chat-text">Share Conversation</span>
              </div>
              <button
                onClick={() => setShareModalOpen(false)}
                className="p-1 rounded-md text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.06] transition-colors"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
            {/* Body */}
            <div className="px-5 py-5">
              {shareLoading ? (
                <div className="flex flex-col items-center gap-3 py-4">
                  <div className="w-6 h-6 rounded-full border-2 border-purple-400/40 border-t-purple-400 animate-spin" />
                  <span className="text-sm text-chat-text-secondary">Generating share link…</span>
                </div>
              ) : shareUrl ? (
                <>
                  <p className="text-xs text-chat-text-secondary mb-3">
                    Anyone with this link can view the conversation for 7 days.
                  </p>
                  <div className="flex items-center gap-2 p-2.5 rounded-lg bg-surface-2 border border-white/[0.06]">
                    <span className="flex-1 text-xs text-chat-text truncate select-all">{shareUrl}</span>
                    <button
                      onClick={handleCopyShareUrl}
                      className="flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
                      style={{
                        background: shareCopied ? 'rgba(74,222,128,0.15)' : 'rgba(139,92,246,0.25)',
                        color: shareCopied ? '#4ade80' : '#c4b5fd',
                      }}
                    >
                      {shareCopied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <p className="text-[10px] text-chat-text-secondary/50 mt-2">
                    Expires in 7 days. No sign-in required to view.
                  </p>
                </>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
