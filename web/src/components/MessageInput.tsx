import { useState, useRef, useEffect, KeyboardEvent, FormEvent } from 'react';
import { PaperAirplaneIcon } from '@heroicons/react/24/solid';

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function MessageInput({
  onSend,
  disabled = false,
  placeholder = 'Message AURA...',
}: MessageInputProps) {
  const [message, setMessage] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const hasText = message.trim().length > 0;

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

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-chat-border/50 bg-chat-bg/80 backdrop-blur-sm px-4 py-4">
      <form
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto relative"
      >
        <div
          className={`
            relative flex items-end bg-chat-assistant/80 rounded-xl border
            transition-all duration-300 ease-out
            ${isFocused
              ? 'border-aura-purple/60 shadow-[0_0_0_2px_rgba(139,92,246,0.15),0_0_20px_rgba(139,92,246,0.2)]'
              : 'border-chat-border/50 hover:border-chat-border'
            }
          `}
        >
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="input-textarea flex-1 bg-transparent text-chat-text placeholder-chat-text-secondary px-4 py-3 pr-14 outline-none resize-none"
          />

          <button
            type="submit"
            disabled={disabled || !hasText}
            className={`
              absolute right-2 bottom-2 p-2.5 rounded-lg
              transition-all duration-300 ease-out
              ${disabled || !hasText
                ? 'text-chat-text-secondary cursor-not-allowed scale-100'
                : 'bg-gradient-to-r from-aura-purple to-aura-blue text-white scale-105 shadow-glow-purple hover:shadow-glow-purple-lg hover:scale-110'
              }
            `}
          >
            <PaperAirplaneIcon className={`w-5 h-5 transition-transform duration-300 ${hasText && !disabled ? '-rotate-45' : ''}`} />
          </button>
        </div>

        <div className="mt-2 text-xs text-chat-text-secondary text-center font-light tracking-wide">
          Press Enter to send, Shift+Enter for new line
        </div>
      </form>
    </div>
  );
}
