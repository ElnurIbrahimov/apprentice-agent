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
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
    <div className="border-t border-chat-border bg-chat-bg px-4 py-4">
      <form
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto relative"
      >
        <div className="relative flex items-end bg-chat-assistant rounded-xl border border-chat-border focus-within:border-chat-accent transition-colors">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="input-textarea flex-1 bg-transparent text-chat-text placeholder-chat-text-secondary px-4 py-3 pr-12 outline-none resize-none"
          />

          <button
            type="submit"
            disabled={disabled || !message.trim()}
            className={`absolute right-2 bottom-2 p-2 rounded-lg transition-colors ${
              disabled || !message.trim()
                ? 'text-chat-text-secondary cursor-not-allowed'
                : 'text-chat-accent hover:bg-chat-accent hover:text-white'
            }`}
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="mt-2 text-xs text-chat-text-secondary text-center">
          Press Enter to send, Shift+Enter for new line
        </div>
      </form>
    </div>
  );
}
