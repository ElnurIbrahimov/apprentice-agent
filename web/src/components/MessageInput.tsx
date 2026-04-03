import { useState, useRef, useEffect, KeyboardEvent, FormEvent, DragEvent, ClipboardEvent } from 'react';
import {
  PlusIcon, MicrophoneIcon, ChevronDownIcon,
  PhotoIcon, MagnifyingGlassIcon, BeakerIcon, CpuChipIcon, UserGroupIcon, ScaleIcon, GlobeAltIcon,
  PaperAirplaneIcon,
} from '@heroicons/react/24/outline';
import { AttachmentList } from './AttachmentPreview';
import { useFileUpload, isSupported } from '../hooks/useFileUpload';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { haptic } from '../utils/haptics';
import { sounds } from '../utils/sounds';
import type { FileAttachment } from '../types';

// Action modes
type ActionMode = 'none' | 'search' | 'research' | 'deep_research' | 'swarm' | 'agent' | 'compare';

const MODE_LABELS: Record<ActionMode, string> = {
  none: '',
  search: 'Search',
  research: 'Research',
  deep_research: 'Deep Research',
  agent: 'Agent',
  swarm: 'Swarm',
  compare: 'Compare',
};

const MODE_COLORS: Record<ActionMode, string> = {
  none: '',
  search: 'rgba(59,130,246,0.25)',
  research: 'rgba(16,185,129,0.25)',
  deep_research: 'rgba(245,158,11,0.25)',
  agent: 'rgba(168,85,247,0.25)',
  swarm: 'rgba(6,182,212,0.25)',
  compare: 'rgba(249,115,22,0.25)',
};

const MODE_ICONS: Record<ActionMode, string> = {
  none: '',
  search: '\uD83D\uDD0D',
  research: '\uD83D\uDCDA',
  deep_research: '\uD83E\uDDD0',
  agent: '\u26A1',
  swarm: '\uD83D\uDC1D',
  compare: '\u2696\uFE0F',
};

interface MessageInputProps {
  onSend: (message: string, attachments?: FileAttachment[], actionMode?: string | null) => void;
  onStop?: () => void;
  onTypingStart?: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export function MessageInput({
  onSend,
  onStop,
  onTypingStart,
  disabled = false,
  isLoading = false,
  placeholder = 'Message AURA...',
}: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [actionMode, setActionMode] = useState<ActionMode>('none');
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const plusMenuRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const [isListening, setIsListening] = useState(false);
  const [voiceError, setVoiceError] = useState(false);
  const recognitionRef = useRef<any>(null);
  const mountedRef = useRef(true);

  const { selectedModel, availableModels, setSelectedModel } = useChatStore();
  const { settings } = useSettingsStore();

  const {
    attachments,
    uploadFiles,
    removeAttachment,
    clearAttachments,
    isUploading,
  } = useFileUpload();

  const hasText = message.trim().length > 0;
  const hasAttachments = attachments.length > 0;
  const hasReadyAttachments = attachments.some(a => !a.uploading && !a.error);
  const canSend = (hasText || hasReadyAttachments) && !disabled && !isUploading;

  const modelLabel = selectedModel
    ? selectedModel.split('/').pop()?.replace(/-\d{8}$/, '') ?? selectedModel
    : 'Auto';

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

  // Focus on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // Keyboard shortcut: Ctrl+K focus input
  useEffect(() => {
    const handler = () => textareaRef.current?.focus();
    document.addEventListener('aura:focus-input', handler);
    return () => document.removeEventListener('aura:focus-input', handler);
  }, []);

  // Click outside to close dropdowns
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (plusMenuRef.current && !plusMenuRef.current.contains(e.target as Node)) {
        setShowPlusMenu(false);
      }
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Cleanup speech recognition on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      recognitionRef.current?.abort();
    };
  }, []);

  const handleVoiceToggle = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;

    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
      return;
    }

    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      if (mountedRef.current) {
        setMessage(prev => prev ? `${prev} ${transcript}` : transcript);
        setIsListening(false);
      }
    };
    recognition.onerror = () => {
      setIsListening(false);
      recognitionRef.current = null;
      // Flash the voice button red briefly on error
      setVoiceError(true);
      setTimeout(() => setVoiceError(false), 2000);
    };
    recognition.onend = () => { setIsListening(false); recognitionRef.current = null; };

    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!canSend) return;

    // Haptic feedback on send (Android only)
    haptic(50);
    if (settings.soundEnabled) sounds.send();

    // Get ready attachments (not uploading, no errors)
    const readyAttachments = attachments.filter(a => !a.uploading && !a.error);

    // Just use the message as-is - backend detects trigger words
    // Only add minimal hint if action mode is selected but no trigger word in message
    let finalMessage = message.trim();

    // Pass action mode for auto-model selection (backend will use appropriate cloud model)
    const modeForBackend = actionMode !== 'none' ? actionMode : null;
    onSend(finalMessage, readyAttachments.length > 0 ? readyAttachments : undefined, modeForBackend);
    setMessage('');
    setActionMode('none'); // Reset mode after sending
    clearAttachments();

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      uploadFiles(files);
    }
    // Reset input so same file can be selected again
    e.target.value = '';
  };

  // Drag and drop handlers
  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      // Filter to supported files
      const supportedFiles = Array.from(files).filter(f => isSupported(f.name));
      if (supportedFiles.length > 0) {
        uploadFiles(supportedFiles);
      }
    }
  };

  // Paste handler for images
  const handlePaste = (e: ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    const imageItems = Array.from(items).filter(
      item => item.type.startsWith('image/')
    );

    if (imageItems.length > 0) {
      e.preventDefault();
      const files = imageItems
        .map(item => item.getAsFile())
        .filter((f): f is File => f !== null);

      if (files.length > 0) {
        uploadFiles(files);
      }
    }
  };

  return (
    <div
      className="px-3 sm:px-4 pb-3 sm:pb-6 pt-2 sm:pt-3 transition-all duration-200 input-safe-area"
      style={{ background: isDragOver ? 'rgba(124,58,237,0.05)' : 'transparent' }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
        {/* Attachment previews */}
        {hasAttachments && (
          <AttachmentList attachments={attachments} onRemove={removeAttachment} />
        )}

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".png,.jpg,.jpeg,.gif,.webp,.bmp,.pdf,.txt,.md,.json,.py,.js,.ts,.tsx,.jsx,.html,.css,.java,.c,.cpp,.h,.go,.rs,.rb,.php,.sh,.yaml,.yml,.toml,.xml,.sql,.zip"
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Glass input wrapper */}
        <div
          style={{
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--bg-panel)',
            border: isFocused ? '1px solid var(--border-focus)' : '1px solid var(--border-subtle)',
            borderRadius: 24,
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            boxShadow: isFocused
              ? '0 12px 40px rgba(0,0,0,0.3), 0 0 0 1px var(--border-default), inset 0 1px 0 var(--border-subtle)'
              : '0 12px 40px rgba(0,0,0,0.3), inset 0 1px 0 var(--border-subtle)',
            transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
          }}
        >
          {/* Drag overlay */}
          {isDragOver && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-purple-600/20 border-2 border-dashed border-purple-500 rounded-3xl pointer-events-none">
              <span className="text-purple-300 font-medium">Drop files here</span>
            </div>
          )}

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => {
              const val = e.target.value;
              if (val.length > 0 && message.length === 0 && onTypingStart) {
                onTypingStart();
              }
              setMessage(val);
            }}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            onPaste={handlePaste}
            placeholder={
              actionMode === 'search' ? 'What do you want to search for?' :
              actionMode === 'research' ? 'What topic should I research?' :
              actionMode === 'agent' ? 'Describe the task for the agent...' :
              placeholder
            }
            disabled={disabled}
            rows={1}
            className="bg-transparent text-chat-text outline-none resize-none w-full input-textarea"
            style={{
              padding: '14px 16px 6px',
              fontSize: '16px',
              lineHeight: 1.6,
              minHeight: 48,
              maxHeight: 120,
            }}
          />

          {/* Bottom row */}
          <div className="flex items-center gap-2 px-3 pb-3 pt-1">
            {/* + button with dropdown */}
            <div className="relative" ref={plusMenuRef}>
              <button
                type="button"
                onClick={() => { setShowPlusMenu(p => !p); setShowModelMenu(false); }}
                disabled={disabled}
                aria-label="More options"
                style={{
                  width: 36, height: 36, borderRadius: 8,
                  background: showPlusMenu ? 'var(--surface-3)' : 'var(--surface-2)',
                  border: '1px solid var(--border-default)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: disabled ? 'not-allowed' : 'pointer',
                  color: 'var(--text-secondary)',
                  transition: 'all 0.15s',
                  flexShrink: 0,
                }}
              >
                <PlusIcon className="w-4 h-4" />
              </button>

              {/* + Dropdown */}
              {showPlusMenu && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 40,
                    left: 0,
                    minWidth: 200,
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    borderRadius: 12,
                    backdropFilter: 'blur(20px)',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                    padding: '6px',
                    zIndex: 50,
                  }}
                >
                  {/* Attach files */}
                  <button
                    type="button"
                    onClick={() => { handleFileSelect(); setShowPlusMenu(false); }}
                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-chat-text hover:bg-white/8 transition-colors text-left"
                  >
                    <PhotoIcon className="w-4 h-4 text-chat-text-secondary flex-shrink-0" />
                    Add photos & files
                  </button>
                  <div style={{ height: 1, background: 'var(--border-default)', margin: '4px 6px' }} />
                  {/* Mode options */}
                  {([
                    { mode: 'search' as ActionMode, icon: MagnifyingGlassIcon, label: 'Search', desc: 'Quick web search' },
                    { mode: 'research' as ActionMode, icon: GlobeAltIcon, label: 'Research', desc: 'Comprehensive research' },
                    { mode: 'deep_research' as ActionMode, icon: BeakerIcon, label: 'Deep Research', desc: '20+ sources' },
                    { mode: 'agent' as ActionMode, icon: CpuChipIcon, label: 'Agent', desc: 'Autonomous execution' },
                    { mode: 'swarm' as ActionMode, icon: UserGroupIcon, label: 'Swarm', desc: 'Multi-agent parallel' },
                    { mode: 'compare' as ActionMode, icon: ScaleIcon, label: 'Compare', desc: 'Compare 3 models' },
                  ] as const).map(({ mode, icon: Icon, label, desc }) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => { setActionMode(actionMode === mode ? 'none' : mode); setShowPlusMenu(false); }}
                      className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors text-left"
                      style={{
                        color: actionMode === mode ? 'var(--text-primary)' : 'var(--text-secondary)',
                        background: actionMode === mode ? MODE_COLORS[mode] : 'transparent',
                      }}
                      onMouseEnter={e => { if (actionMode !== mode) (e.currentTarget as HTMLElement).style.background = 'var(--bg-panel-hover)'; }}
                      onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = actionMode === mode ? MODE_COLORS[mode] : 'transparent'; }}
                    >
                      <Icon className="w-4 h-4 flex-shrink-0 opacity-70" />
                      <span>{label}</span>
                      <span className="ml-auto text-xs opacity-40">{desc}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Active mode chip */}
            {actionMode !== 'none' && (
              <button
                type="button"
                onClick={() => setActionMode('none')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '3px 10px',
                  borderRadius: 20,
                  background: MODE_COLORS[actionMode],
                  border: '1px solid var(--border-default)',
                  color: 'var(--text-primary)',
                  fontSize: 12,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                {MODE_ICONS[actionMode]} {MODE_LABELS[actionMode]}
                <span style={{ opacity: 0.5, marginLeft: 2 }}>×</span>
              </button>
            )}

            {/* Spacer */}
            <div className="flex-1" />

            {/* Model pill */}
            <div className="relative" ref={modelMenuRef}>
              <button
                type="button"
                onClick={() => { setShowModelMenu(p => !p); setShowPlusMenu(false); }}
                className="flex items-center gap-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                style={{ padding: '4px 8px', borderRadius: 8, background: 'var(--border-subtle)' }}
              >
                <span className="max-w-[160px] truncate">{modelLabel}</span>
                <ChevronDownIcon className="w-3 h-3 opacity-50" />
              </button>

              {showModelMenu && availableModels.length > 0 && (() => {
                const chatgpt = availableModels.filter(m => m.startsWith('chatgpt:'));
                const apiProviders = ['anthropic:', 'openai:', 'gemini:', 'grok:', 'perplexity:', 'deepseek:', 'minimax:', 'qwen:', 'kimi:', 'glm:'];
                const apiModels = availableModels.filter(m => !m.startsWith('chatgpt:') && apiProviders.some(p => m.startsWith(p)));
                const cloud = availableModels.filter(m => !m.startsWith('chatgpt:') && !apiProviders.some(p => m.startsWith(p)) && (m.includes(':cloud') || m.includes('-cloud')));
                const local = availableModels.filter(m => !m.startsWith('chatgpt:') && !apiProviders.some(p => m.startsWith(p)) && !m.includes(':cloud') && !m.includes('-cloud'));
                const renderItem = (model: string, label: string) => (
                  <button
                    key={model}
                    type="button"
                    onClick={() => { setSelectedModel(model); setShowModelMenu(false); }}
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                    style={{ color: selectedModel === model ? 'var(--text-primary)' : 'var(--text-secondary)', background: selectedModel === model ? 'var(--surface-3)' : 'transparent' }}
                  >
                    {label}
                  </button>
                );
                const sectionHeader = (text: string, color: string) => (
                  <div key={`h-${text}`} style={{ fontSize: 10, fontWeight: 600, color, padding: '6px 10px 2px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{text}</div>
                );
                return (
                  <div
                    style={{
                      position: 'absolute',
                      bottom: 36,
                      right: 0,
                      width: 240,
                      maxHeight: 320,
                      background: 'var(--surface-1)',
                      border: '1px solid var(--border-default)',
                      borderRadius: 12,
                      backdropFilter: 'blur(20px)',
                      boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                      padding: '6px',
                      zIndex: 50,
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors flex-shrink-0"
                      style={{ color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)', background: !selectedModel ? 'var(--surface-3)' : 'transparent' }}
                    >
                      🤖 Auto (recommended)
                    </button>
                    <div style={{ height: 1, background: 'var(--border-default)', margin: '4px 6px', flexShrink: 0 }} />
                    <div style={{ overflowY: 'auto', flex: 1 }}>
                      {chatgpt.length > 0 && sectionHeader('ChatGPT', 'rgba(74,222,128,0.7)')}
                      {chatgpt.map(m => renderItem(m, '🟢 ' + m.replace('chatgpt:', '')))}
                      {apiModels.length > 0 && sectionHeader('API Providers', 'rgba(236,72,153,0.7)')}
                      {apiModels.map(m => renderItem(m, '🔑 ' + m))}
                      {cloud.length > 0 && sectionHeader('Cloud', 'rgba(96,165,250,0.7)')}
                      {cloud.map(m => renderItem(m, '☁️ ' + m.replace(/:cloud$/, '')))}
                      {local.length > 0 && sectionHeader('Local', 'rgba(251,146,60,0.7)')}
                      {local.map(m => renderItem(m, '💻 ' + m))}
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Voice button */}
            <button
              type="button"
              onClick={handleVoiceToggle}
              disabled={disabled}
              aria-label={isListening ? 'Stop listening' : 'Voice input'}
              style={{
                width: 36, height: 36, borderRadius: 8,
                background: isListening ? 'rgba(239,68,68,0.2)' : voiceError ? 'rgba(239,68,68,0.15)' : 'transparent',
                border: voiceError ? '1px solid rgba(239,68,68,0.4)' : 'none',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: disabled ? 'not-allowed' : 'pointer',
                color: isListening || voiceError ? '#f87171' : 'var(--text-secondary)',
                transition: 'all 0.15s',
                flexShrink: 0,
              }}
              title={voiceError ? 'Microphone access denied or unavailable' : isListening ? 'Stop listening' : 'Voice input'}
            >
              <MicrophoneIcon className={`w-5 h-5 ${isListening ? 'animate-pulse' : ''}`} />
            </button>

            {/* Send / Stop */}
            {isLoading && onStop ? (
              <button
                type="button"
                onClick={onStop}
                aria-label="Stop generation"
                style={{
                  width: 36, height: 36, borderRadius: 8,
                  background: 'var(--surface-2)',
                  border: '1px solid var(--border-default)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                <svg className="w-3.5 h-3.5 text-white" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="1" />
                </svg>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!canSend}
                aria-label="Send message"
                style={{
                  width: 36, height: 36, borderRadius: 8,
                  background: canSend ? 'var(--text-primary)' : 'var(--surface-2)',
                  border: 'none',
                  color: canSend ? 'var(--bg-base)' : 'var(--text-tertiary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: canSend ? 'pointer' : 'not-allowed',
                  boxShadow: canSend ? '0 2px 8px var(--border-strong)' : 'none',
                  transition: 'all 0.2s',
                  flexShrink: 0,
                }}
              >
                <PaperAirplaneIcon className="w-4 h-4" style={{ transform: 'rotate(-45deg)' }} />
              </button>
            )}
          </div>
        </div>

        {/* Footer hint */}
        <div className="mt-2 text-xs text-chat-text-secondary text-center font-light opacity-40 hidden sm:block">
          {isUploading ? 'Uploading...' : 'Enter to send · Shift+Enter for newline'}
        </div>
      </form>
    </div>
  );
}
