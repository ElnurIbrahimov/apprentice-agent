import { useState, useRef, useEffect, useMemo, useCallback, KeyboardEvent, FormEvent, DragEvent, ClipboardEvent } from 'react';
import {
  PlusIcon, MicrophoneIcon, ChevronDownIcon,
  PhotoIcon, MagnifyingGlassIcon, BeakerIcon, CpuChipIcon, UserGroupIcon, ScaleIcon, GlobeAltIcon,
  PaperAirplaneIcon, StopCircleIcon, RocketLaunchIcon,
} from '@heroicons/react/24/outline';
import { estimateTokens, usageBand, DEFAULT_CONTEXT_WINDOW, METER_VISIBILITY_THRESHOLD } from '../utils/tokenEstimate';
import { AttachmentList } from './AttachmentPreview';
import { ActionSheet } from './BottomSheet';
import { useFileUpload, isSupported } from '../hooks/useFileUpload';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { haptic } from '../utils/haptics';
import { sounds } from '../utils/sounds';
import { toast } from './Toast';
import type { FileAttachment } from '../types';
import { apiFetch } from '../utils/apiFetch';

// Mobile detection hook
function useIsMobile() {
  const [mobile, setMobile] = useState(false);
  useEffect(() => {
    const check = () => setMobile(window.innerWidth < 1024);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);
  return mobile;
}

// Action modes
type ActionMode = 'none' | 'search' | 'research' | 'deep_research' | 'swarm' | 'agent' | 'compare' | 'delegate';

const MODE_LABELS: Record<ActionMode, string> = {
  none: '',
  search: 'Search',
  research: 'Research',
  deep_research: 'Deep Research',
  agent: 'Agent',
  swarm: 'Swarm',
  compare: 'Compare',
  delegate: 'Delegate',
};

const MODE_COLORS: Record<ActionMode, string> = {
  none: '',
  search: 'rgba(59,130,246,0.25)',
  research: 'rgba(16,185,129,0.25)',
  deep_research: 'rgba(245,158,11,0.25)',
  agent: 'rgba(168,85,247,0.25)',
  swarm: 'rgba(6,182,212,0.25)',
  compare: 'rgba(249,115,22,0.25)',
  delegate: 'rgba(124,58,237,0.28)',
};

const MODE_ICONS: Record<ActionMode, string> = {
  none: '',
  search: '\uD83D\uDD0D',
  research: '\uD83D\uDCDA',
  deep_research: '\uD83E\uDDD0',
  agent: '\u26A1',
  swarm: '\uD83D\uDC1D',
  compare: '\u2696\uFE0F',
  delegate: '\uD83C\uDFAF',
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
  const [slashIndex, setSlashIndex] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const plusMenuRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  const messages = useChatStore(s => s.messages);
  const selectedModel = useChatStore(s => s.selectedModel);
  const availableModels = useChatStore(s => s.availableModels);
  const setSelectedModel = useChatStore(s => s.setSelectedModel);
  const setAvailableModels = useChatStore(s => s.setAvailableModels);
  const settings = useSettingsStore(s => s.settings);
  const isMobile = useIsMobile();

  // Context budget meter — computes current estimated token usage to surface
  // when a conversation is getting close to the model's context window.
  const contextTokens = estimateTokens(messages);
  const contextRatio = contextTokens / DEFAULT_CONTEXT_WINDOW;
  const showMeter = contextRatio >= METER_VISIBILITY_THRESHOLD;
  const band = usageBand(contextRatio);
  const meterColor =
    band === 'danger' ? '#ef4444' :
    band === 'warn' ? '#f59e0b' :
    '#10b981';

  // Fetch available models on mount
  useEffect(() => {
    apiFetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, [setAvailableModels]);

  const {
    attachments,
    uploadFiles,
    removeAttachment,
    clearAttachments,
    isUploading,
  } = useFileUpload();

  // Slash command menu: triggered when message begins with "/" and has no space yet.
  const slashMatch = useMemo(() => {
    const m = /^\/([a-z_]*)$/i.exec(message);
    return m ? m[1].toLowerCase() : null;
  }, [message]);
  const slashCommands: { key: ActionMode; command: string; label: string; desc: string }[] = useMemo(() => [
    { key: 'research', command: 'research', label: 'Research', desc: 'Multi-source deep research with citations' },
    { key: 'deep_research', command: 'deep', label: 'Deep Research', desc: '20+ sources, longer synthesis' },
    { key: 'search', command: 'search', label: 'Search', desc: 'Quick web search' },
    { key: 'agent', command: 'agent', label: 'Agent', desc: 'Autonomous multi-step with tools' },
    { key: 'swarm', command: 'swarm', label: 'Swarm', desc: 'Multi-agent parallel' },
    { key: 'compare', command: 'compare', label: 'Compare', desc: 'Side-by-side model comparison' },
    { key: 'delegate', command: 'delegate', label: 'Delegate', desc: 'Async task for Aura' },
  ], []);
  const filteredSlash = useMemo(() => {
    if (slashMatch === null) return [];
    if (!slashMatch) return slashCommands;
    return slashCommands.filter((c) => c.command.startsWith(slashMatch));
  }, [slashMatch, slashCommands]);
  const showSlashMenu = slashMatch !== null && filteredSlash.length > 0;

  useEffect(() => {
    if (!showSlashMenu) setSlashIndex(0);
    else if (slashIndex >= filteredSlash.length) setSlashIndex(0);
  }, [showSlashMenu, filteredSlash.length, slashIndex]);

  const pickSlashCommand = useCallback((cmd: typeof slashCommands[number]) => {
    setActionMode(cmd.key);
    setMessage(''); // clear the slash, user continues typing the actual query
    textareaRef.current?.focus();
  }, []);

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
    document.addEventListener('touchstart', handler as EventListener);
    return () => {
      document.removeEventListener('mousedown', handler);
      document.removeEventListener('touchstart', handler as EventListener);
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      mediaRecorderRef.current?.stop();
    };
  }, []);

  const stopRecording = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    mediaRecorderRef.current?.stop();
  };

  const handleMicClick = async () => {
    if (isRecording) {
      stopRecording();
      return;
    }

    if (isTranscribing) return;

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      toast.error('Microphone access denied', 'Allow microphone access in your browser settings.');
      return;
    }

    audioChunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
    const recorder = new MediaRecorder(stream, { mimeType });

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      setIsRecording(false);
      setRecordingSeconds(0);

      if (!mountedRef.current) return;

      const blob = new Blob(audioChunksRef.current, { type: mimeType });
      if (blob.size < 1000) {
        toast.warning('No speech detected', 'Try speaking closer to your microphone.');
        return;
      }

      setIsTranscribing(true);
      try {
        const ext = mimeType === 'audio/webm' ? 'webm' : 'ogg';
        const formData = new FormData();
        formData.append('file', blob, `recording.${ext}`);

        const res = await fetch('/api/transcribe', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const transcript = (data.text || '').trim();

        if (!transcript) {
          toast.warning('No speech detected', 'Try again and speak clearly.');
          return;
        }

        if (mountedRef.current) {
          setMessage(prev => prev ? `${prev} ${transcript}` : transcript);
          textareaRef.current?.focus();
        }
      } catch (err: any) {
        toast.error('Transcription failed', err?.message || 'Please try again.');
      } finally {
        if (mountedRef.current) setIsTranscribing(false);
      }
    };

    mediaRecorderRef.current = recorder;
    recorder.start(250);
    setIsRecording(true);
    setRecordingSeconds(0);

    recordingTimerRef.current = setInterval(() => {
      setRecordingSeconds(s => s + 1);
    }, 1000);
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
    // Slash command menu navigation takes priority
    if (showSlashMenu) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setSlashIndex((i) => (i + 1) % filteredSlash.length); return; }
      if (e.key === 'ArrowUp')   { e.preventDefault(); setSlashIndex((i) => (i - 1 + filteredSlash.length) % filteredSlash.length); return; }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        const cmd = filteredSlash[slashIndex];
        if (cmd) pickSlashCommand(cmd);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setMessage('');
        return;
      }
    }
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
      className="px-2.5 sm:px-4 pb-2 sm:pb-6 pt-1.5 sm:pt-3 transition-all duration-200 input-safe-area"
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

        {/* Slash command menu — floats above the input while the user types /... */}
        {showSlashMenu && (
          <div
            role="listbox"
            aria-label="Slash commands"
            className="rounded-xl overflow-hidden mb-2"
            style={{
              background: 'var(--surface-1)',
              border: '1px solid var(--border-default)',
              boxShadow: '0 12px 36px rgba(0,0,0,0.35)',
            }}
          >
            <div className="px-3 py-1.5 text-[10px] uppercase tracking-wide text-chat-text-secondary border-b border-white/5">
              Slash commands · <kbd className="font-mono">↑</kbd> <kbd className="font-mono">↓</kbd> to nav · <kbd className="font-mono">Enter</kbd> to pick · <kbd className="font-mono">Esc</kbd> to cancel
            </div>
            {filteredSlash.map((c, i) => (
              <button
                key={c.command}
                type="button"
                role="option"
                aria-selected={i === slashIndex}
                onMouseEnter={() => setSlashIndex(i)}
                onClick={() => pickSlashCommand(c)}
                className="w-full flex items-center gap-3 px-3 py-2 text-left transition-colors"
                style={{
                  background: i === slashIndex ? 'var(--surface-3)' : 'transparent',
                  color: 'var(--text-primary)',
                }}
              >
                <span
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded flex-shrink-0"
                  style={{ background: MODE_COLORS[c.key], color: 'var(--text-primary)' }}
                >
                  /{c.command}
                </span>
                <span className="text-sm flex-shrink-0 font-medium">{c.label}</span>
                <span className="text-xs text-chat-text-secondary truncate">{c.desc}</span>
              </button>
            ))}
          </div>
        )}

        {/* Context budget meter — hidden until 50% usage, recolors at 75/90% */}
        {showMeter && (
          <button
            type="button"
            onClick={() => toast.info(
              `${(contextRatio * 100).toFixed(0)}% of context used`,
              'Conversation getting long. Fork from an earlier message to start fresh, or continue — older messages may drop.'
            )}
            aria-label={`Context usage ${(contextRatio * 100).toFixed(0)} percent — tap for options`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '4px 10px',
              marginBottom: 6,
              background: 'var(--surface-1)',
              border: '1px solid var(--border-default)',
              borderRadius: 10,
              fontSize: 10,
              fontVariantNumeric: 'tabular-nums',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              width: '100%',
              textAlign: 'left',
            }}
          >
            <span style={{ opacity: 0.7 }}>Context</span>
            <div
              style={{
                flex: 1,
                height: 3,
                background: 'var(--border-subtle)',
                borderRadius: 2,
                overflow: 'hidden',
                position: 'relative',
              }}
            >
              <div
                style={{
                  width: `${Math.min(100, contextRatio * 100)}%`,
                  height: '100%',
                  background: meterColor,
                  transition: 'width 0.3s ease, background 0.3s ease',
                }}
              />
            </div>
            <span style={{ color: meterColor, minWidth: 32, textAlign: 'right' }}>
              {(contextRatio * 100).toFixed(0)}%
            </span>
          </button>
        )}

        {/* Glass input wrapper */}
        <div
          style={{
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--bg-panel)',
            border: isFocused ? '1px solid var(--border-strong)' : '1px solid var(--border-subtle)',
            borderRadius: 24,
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            boxShadow: isFocused
              ? '0 8px 32px rgba(0,0,0,0.25), 0 0 0 1px var(--border-default), inset 0 1px 0 rgba(255,255,255,0.04)'
              : '0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.03)',
            transition: 'border-color var(--duration-normal) var(--ease-out), box-shadow var(--duration-normal) var(--ease-out)',
          }}
        >
          {/* Drag overlay */}
          {isDragOver && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-blue-600/20 border-2 border-dashed border-blue-500 rounded-3xl pointer-events-none">
              <span className="text-blue-300 font-medium">Drop files here</span>
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
              actionMode === 'delegate' ? "Delegate a task to Aura — e.g. 'research X and summarize by tomorrow'" :
              placeholder
            }
            disabled={disabled}
            rows={1}
            className="bg-transparent text-chat-text outline-none resize-none w-full input-textarea"
            style={{
              padding: '12px 14px 4px',
              fontSize: '16px',
              lineHeight: 1.5,
              minHeight: 44,
              maxHeight: 140,
              caretColor: 'var(--text-primary)',
            }}
          />

          {/* Bottom row */}
          <div className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 pb-2.5 sm:pb-3 pt-0.5">
            {/* + button — opens dropdown (desktop) or bottom sheet (mobile) */}
            <div className="relative" ref={plusMenuRef}>
              <button
                type="button"
                onClick={() => { setShowPlusMenu(p => !p); setShowModelMenu(false); }}
                disabled={disabled}
                aria-label="More options"
                style={{
                  width: 40, height: 40, borderRadius: 10,
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

              {/* Desktop dropdown */}
              {showPlusMenu && !isMobile && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 44,
                    left: 0,
                    minWidth: 220,
                    maxWidth: 'min(280px, calc(100vw - 2rem))',
                    maxHeight: 'calc(100vh - 160px)',
                    overflowY: 'auto',
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    borderRadius: 14,
                    backdropFilter: 'blur(24px)',
                    WebkitBackdropFilter: 'blur(24px)',
                    boxShadow: '0 12px 40px rgba(0,0,0,0.35)',
                    padding: '6px',
                    zIndex: 50,
                  }}
                >
                  <button
                    type="button"
                    onClick={() => { handleFileSelect(); setShowPlusMenu(false); }}
                    className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-chat-text hover:bg-white/8 transition-colors text-left"
                  >
                    <PhotoIcon className="w-4 h-4 text-chat-text-secondary flex-shrink-0" />
                    Add photos & files
                  </button>
                  <div style={{ height: 1, background: 'var(--border-default)', margin: '4px 6px' }} />
                  {([
                    { mode: 'search' as ActionMode, icon: MagnifyingGlassIcon, label: 'Search', desc: 'Quick web search' },
                    { mode: 'research' as ActionMode, icon: GlobeAltIcon, label: 'Research', desc: 'Comprehensive research' },
                    { mode: 'deep_research' as ActionMode, icon: BeakerIcon, label: 'Deep Research', desc: '20+ sources' },
                    { mode: 'agent' as ActionMode, icon: CpuChipIcon, label: 'Agent', desc: 'Autonomous execution' },
                    { mode: 'swarm' as ActionMode, icon: UserGroupIcon, label: 'Swarm', desc: 'Multi-agent parallel' },
                    { mode: 'compare' as ActionMode, icon: ScaleIcon, label: 'Compare', desc: 'Compare 3 models' },
                    { mode: 'delegate' as ActionMode, icon: RocketLaunchIcon, label: 'Delegate', desc: 'Hand off async to Aura' },
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

              {/* Mobile bottom sheet */}
              <ActionSheet
                open={showPlusMenu && isMobile}
                onClose={() => setShowPlusMenu(false)}
                title="Actions"
                items={[
                  {
                    icon: <PhotoIcon className="w-5 h-5" />,
                    label: 'Add photos & files',
                    sublabel: 'Images, PDFs, code files',
                    onPress: () => { handleFileSelect(); setShowPlusMenu(false); },
                  },
                  ...([
                    { mode: 'search' as ActionMode, icon: <MagnifyingGlassIcon className="w-5 h-5" />, label: 'Search', desc: 'Quick web search' },
                    { mode: 'research' as ActionMode, icon: <GlobeAltIcon className="w-5 h-5" />, label: 'Research', desc: 'Comprehensive research' },
                    { mode: 'deep_research' as ActionMode, icon: <BeakerIcon className="w-5 h-5" />, label: 'Deep Research', desc: '20+ sources' },
                    { mode: 'agent' as ActionMode, icon: <CpuChipIcon className="w-5 h-5" />, label: 'Agent', desc: 'Autonomous execution' },
                    { mode: 'swarm' as ActionMode, icon: <UserGroupIcon className="w-5 h-5" />, label: 'Swarm', desc: 'Multi-agent parallel' },
                    { mode: 'compare' as ActionMode, icon: <ScaleIcon className="w-5 h-5" />, label: 'Compare', desc: 'Compare 3 models' },
                    { mode: 'delegate' as ActionMode, icon: <RocketLaunchIcon className="w-5 h-5" />, label: 'Delegate', desc: 'Hand off async to Aura' },
                  ]).map(({ mode, icon, label, desc }) => ({
                    icon,
                    label,
                    sublabel: desc,
                    active: actionMode === mode,
                    activeColor: MODE_COLORS[mode],
                    onPress: () => { setActionMode(actionMode === mode ? 'none' : mode); setShowPlusMenu(false); },
                  })),
                ]}
              />
            </div>

            {/* Active mode chip */}
            {actionMode !== 'none' && (
              <button
                type="button"
                onClick={() => setActionMode('none')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '6px 12px',
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
                <span style={{ opacity: 0.5, marginLeft: 4, padding: '2px 4px' }}>×</span>
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
                      maxWidth: 'calc(100vw - 2rem)',
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
            <div className="relative flex items-center" style={{ flexShrink: 0 }}>
              {isRecording && (
                <span
                  style={{
                    position: 'absolute',
                    right: 44,
                    fontSize: 11,
                    fontVariantNumeric: 'tabular-nums',
                    color: '#f87171',
                    whiteSpace: 'nowrap',
                    pointerEvents: 'none',
                  }}
                >
                  {Math.floor(recordingSeconds / 60)}:{String(recordingSeconds % 60).padStart(2, '0')}
                </span>
              )}
              {isRecording && (
                <span
                  style={{
                    position: 'absolute',
                    inset: -4,
                    borderRadius: 14,
                    border: '2px solid rgba(239,68,68,0.5)',
                    animation: 'mic-pulse 1.2s ease-out infinite',
                    pointerEvents: 'none',
                  }}
                />
              )}
              <button
                type="button"
                onClick={handleMicClick}
                disabled={disabled}
                aria-label={isRecording ? 'Stop recording' : isTranscribing ? 'Transcribing…' : 'Voice input'}
                style={{
                  width: 40, height: 40, borderRadius: 10,
                  background: isRecording ? 'rgba(239,68,68,0.2)' : isTranscribing ? 'rgba(124,58,237,0.2)' : 'transparent',
                  border: 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: (disabled || isTranscribing) ? 'not-allowed' : 'pointer',
                  color: isRecording ? '#f87171' : isTranscribing ? '#a78bfa' : 'var(--text-secondary)',
                  transition: 'all 0.15s',
                  position: 'relative',
                }}
                title={isRecording ? 'Stop recording' : isTranscribing ? 'Transcribing…' : 'Voice input'}
              >
                {isRecording ? (
                  <StopCircleIcon className="w-5 h-5" />
                ) : isTranscribing ? (
                  <MicrophoneIcon className="w-5 h-5 animate-pulse" />
                ) : (
                  <MicrophoneIcon className="w-5 h-5" />
                )}
              </button>
            </div>

            {/* Send / Stop — premium morphing button */}
            {isLoading && onStop ? (
              <button
                type="button"
                onClick={() => { haptic(25); onStop(); }}
                aria-label="Stop generation"
                className="animate-spring-scale"
                style={{
                  width: 40, height: 40, borderRadius: 12,
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer',
                  flexShrink: 0,
                  transition: 'all 0.2s',
                }}
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="#f87171">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!canSend}
                aria-label="Send message"
                style={{
                  width: 40, height: 40,
                  borderRadius: canSend ? 20 : 12,
                  background: canSend ? 'var(--chat-accent)' : 'var(--surface-2)',
                  border: 'none',
                  color: canSend ? '#fff' : 'var(--text-tertiary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: canSend ? 'pointer' : 'not-allowed',
                  boxShadow: canSend ? '0 4px 16px rgba(124, 58, 237, 0.4)' : 'none',
                  transition: 'all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)',
                  flexShrink: 0,
                  transform: canSend ? 'scale(1)' : 'scale(0.9)',
                }}
              >
                <PaperAirplaneIcon className="w-4 h-4" style={{ transform: 'rotate(-45deg) translateX(1px)' }} />
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
