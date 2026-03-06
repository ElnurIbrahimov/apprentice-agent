import { useState, useRef, useEffect, KeyboardEvent, FormEvent, DragEvent, ClipboardEvent } from 'react';
import { PaperAirplaneIcon, PaperClipIcon, MicrophoneIcon } from '@heroicons/react/24/solid';
import { MagnifyingGlassIcon, BookOpenIcon, CpuChipIcon, BeakerIcon, UserGroupIcon } from '@heroicons/react/24/outline';
import { AttachmentList } from './AttachmentPreview';
import { useFileUpload, isSupported } from '../hooks/useFileUpload';
import type { FileAttachment } from '../types';

// Action modes for quick actions
type ActionMode = 'none' | 'search' | 'research' | 'deep_research' | 'swarm' | 'agent';

interface MessageInputProps {
  onSend: (message: string, attachments?: FileAttachment[], actionMode?: string | null) => void;
  onStop?: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export function MessageInput({
  onSend,
  onStop,
  disabled = false,
  isLoading = false,
  placeholder = 'Message AURA...',
}: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [actionMode, setActionMode] = useState<ActionMode>('none');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

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

  // Cleanup speech recognition on unmount
  useEffect(() => {
    return () => { recognitionRef.current?.abort(); };
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
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript;
      setMessage(prev => prev ? `${prev} ${transcript}` : transcript);
      setIsListening(false);
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!canSend) return;

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
      className={`
        border-t border-chat-border/50 bg-chat-bg/80 backdrop-blur-sm px-4 py-4
        transition-all duration-200
        ${isDragOver ? 'bg-aura-purple/10' : ''}
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <form
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto relative"
      >
        {/* Attachment previews */}
        {hasAttachments && (
          <AttachmentList
            attachments={attachments}
            onRemove={removeAttachment}
          />
        )}

        {/* Drag overlay */}
        {isDragOver && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-aura-purple/20 border-2 border-dashed border-aura-purple rounded-xl pointer-events-none">
            <span className="text-aura-purple font-medium">Drop files here</span>
          </div>
        )}

        {/* Action mode buttons */}
        <div className="flex items-center gap-2 mb-2">
          <button
            type="button"
            onClick={() => setActionMode(actionMode === 'search' ? 'none' : 'search')}
            disabled={disabled || isLoading}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
              transition-all duration-200 border
              ${actionMode === 'search'
                ? 'bg-blue-500/20 border-blue-500/50 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.3)]'
                : 'bg-chat-assistant/60 border-chat-border/30 text-chat-text-secondary hover:text-blue-400 hover:border-blue-500/30 hover:bg-blue-500/10'
              }
              ${(disabled || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            title="Quick web search"
          >
            <MagnifyingGlassIcon className="w-3.5 h-3.5" />
            Search
          </button>

          <button
            type="button"
            onClick={() => setActionMode(actionMode === 'research' ? 'none' : 'research')}
            disabled={disabled || isLoading}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
              transition-all duration-200 border
              ${actionMode === 'research'
                ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                : 'bg-chat-assistant/60 border-chat-border/30 text-chat-text-secondary hover:text-emerald-400 hover:border-emerald-500/30 hover:bg-emerald-500/10'
              }
              ${(disabled || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            title="Comprehensive research with analysis"
          >
            <BookOpenIcon className="w-3.5 h-3.5" />
            Research
          </button>

          <button
            type="button"
            onClick={() => setActionMode(actionMode === 'agent' ? 'none' : 'agent')}
            disabled={disabled || isLoading}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
              transition-all duration-200 border
              ${actionMode === 'agent'
                ? 'bg-purple-500/20 border-purple-500/50 text-purple-400 shadow-[0_0_10px_rgba(168,85,247,0.3)]'
                : 'bg-chat-assistant/60 border-chat-border/30 text-chat-text-secondary hover:text-purple-400 hover:border-purple-500/30 hover:bg-purple-500/10'
              }
              ${(disabled || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            title="Autonomous agent mode"
          >
            <CpuChipIcon className="w-3.5 h-3.5" />
            Agent
          </button>

          <button
            type="button"
            onClick={() => setActionMode(actionMode === 'deep_research' ? 'none' : 'deep_research')}
            disabled={disabled || isLoading}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
              transition-all duration-200 border
              ${actionMode === 'deep_research'
                ? 'bg-amber-500/20 border-amber-500/50 text-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.3)]'
                : 'bg-chat-assistant/60 border-chat-border/30 text-chat-text-secondary hover:text-amber-400 hover:border-amber-500/30 hover:bg-amber-500/10'
              }
              ${(disabled || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            title="Deep multi-source research"
          >
            <BeakerIcon className="w-3.5 h-3.5" />
            Deep
          </button>

          <button
            type="button"
            onClick={() => setActionMode(actionMode === 'swarm' ? 'none' : 'swarm')}
            disabled={disabled || isLoading}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
              transition-all duration-200 border
              ${actionMode === 'swarm'
                ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'bg-chat-assistant/60 border-chat-border/30 text-chat-text-secondary hover:text-cyan-400 hover:border-cyan-500/30 hover:bg-cyan-500/10'
              }
              ${(disabled || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            title="Multi-agent swarm collaboration"
          >
            <UserGroupIcon className="w-3.5 h-3.5" />
            Swarm
          </button>

          {actionMode !== 'none' && (
            <span className="text-xs text-chat-text-secondary ml-2 animate-fade-in">
              {actionMode === 'search' && 'Quick web search'}
              {actionMode === 'research' && 'Comprehensive research'}
              {actionMode === 'deep_research' && 'Multi-source deep research (20+ pages)'}
              {actionMode === 'swarm' && 'Multiple agents working in parallel'}
              {actionMode === 'agent' && 'Autonomous task execution'}
            </span>
          )}
        </div>

        <div
          className={`
            relative flex items-end bg-chat-assistant/80 rounded-xl border
            transition-all duration-300 ease-out
            ${isFocused
              ? 'border-aura-purple/60 shadow-[0_0_0_2px_rgba(139,92,246,0.15),0_0_20px_rgba(139,92,246,0.2)]'
              : 'border-chat-border/50 hover:border-chat-border'
            }
            ${actionMode === 'search' ? 'border-blue-500/40' : ''}
            ${actionMode === 'research' ? 'border-emerald-500/40' : ''}
            ${actionMode === 'agent' ? 'border-purple-500/40' : ''}
            ${actionMode === 'deep_research' ? 'border-amber-500/40' : ''}
            ${actionMode === 'swarm' ? 'border-cyan-500/40' : ''}
          `}
        >
          {/* Attachment button */}
          <button
            type="button"
            onClick={handleFileSelect}
            disabled={disabled}
            aria-label="Attach file"
            className={`
              p-3 rounded-lg ml-1
              transition-all duration-200
              ${disabled
                ? 'text-chat-text-secondary/50 cursor-not-allowed'
                : 'text-chat-text-secondary hover:text-aura-purple hover:bg-aura-purple/10'
              }
            `}
            title="Attach files (images, documents, code)"
          >
            <PaperClipIcon className="w-5 h-5" />
          </button>

          {/* Voice input */}
          <button
            type="button"
            onClick={handleVoiceToggle}
            disabled={disabled}
            aria-label="Voice input"
            className={`
              p-3 rounded-lg transition-all duration-200
              ${isListening
                ? 'text-red-400 bg-red-500/20 animate-pulse'
                : disabled
                  ? 'text-chat-text-secondary/50 cursor-not-allowed'
                  : 'text-chat-text-secondary hover:text-aura-purple hover:bg-aura-purple/10'
              }
            `}
            title={isListening ? 'Stop listening' : 'Voice input'}
          >
            <MicrophoneIcon className="w-5 h-5" />
          </button>

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.gif,.webp,.bmp,.pdf,.txt,.md,.json,.py,.js,.ts,.tsx,.jsx,.html,.css,.java,.c,.cpp,.h,.go,.rs,.rb,.php,.sh,.yaml,.yml,.toml,.xml,.sql,.zip"
            onChange={handleFileChange}
            className="hidden"
          />

          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
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
            className="input-textarea flex-1 bg-transparent text-chat-text placeholder-chat-text-secondary px-2 py-3 pr-14 outline-none resize-none"
          />

          {isLoading && onStop ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generation"
              className="absolute right-2 bottom-2 w-10 h-10 rounded-full border border-gray-600 bg-gray-800/90 flex items-center justify-center hover:bg-gray-700 hover:border-gray-500 transition-all duration-150"
              title="Stop generation"
            >
              <svg className="w-4 h-4 text-gray-200" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              aria-label="Send message"
              className={`
                absolute right-2 bottom-2 p-2.5 rounded-lg
                transition-all duration-300 ease-out
                ${!canSend
                  ? 'text-chat-text-secondary cursor-not-allowed scale-100'
                  : 'bg-gradient-to-r from-aura-purple to-aura-blue text-white scale-105 shadow-glow-purple hover:shadow-glow-purple-lg hover:scale-110'
                }
              `}
            >
              <PaperAirplaneIcon className={`w-5 h-5 transition-transform duration-300 ${canSend ? '-rotate-45' : ''}`} />
            </button>
          )}
        </div>

        <div className="mt-2 text-xs text-chat-text-secondary text-center font-light tracking-wide">
          {isUploading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-1.5 h-1.5 bg-aura-purple rounded-full animate-pulse" />
              Uploading files...
            </span>
          ) : (
            'Press Enter to send, Shift+Enter for new line. Drag & drop or paste images.'
          )}
        </div>
      </form>
    </div>
  );
}
