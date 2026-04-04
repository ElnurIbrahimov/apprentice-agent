import { useState, useCallback } from 'react';
import { ClipboardDocumentIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline';

interface CopyButtonProps {
  text: string;
  className?: string;
  /** How long to show the "Copied!" state in ms. Default: 1500 */
  feedbackDuration?: number;
}

export function CopyButton({ text, className, feedbackDuration = 1500 }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), feedbackDuration);
    });
  }, [text, feedbackDuration]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      disabled={!text}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
      className={
        className ??
        'p-1.5 rounded-md text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors'
      }
    >
      {copied ? (
        <span className="flex items-center gap-1 text-[10px] text-green-400 font-medium whitespace-nowrap">
          <ClipboardDocumentCheckIcon className="w-4 h-4" />
          Copied!
        </span>
      ) : (
        <ClipboardDocumentIcon className="w-4 h-4" />
      )}
    </button>
  );
}
