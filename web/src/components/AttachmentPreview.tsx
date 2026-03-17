import { XMarkIcon, DocumentIcon, CodeBracketIcon, PhotoIcon, ArchiveBoxIcon } from '@heroicons/react/24/solid';
import type { FileAttachment } from '../types';
import { formatFileSize } from '../hooks/useFileUpload';

interface AttachmentPreviewProps {
  attachment: FileAttachment;
  onRemove?: (id: string) => void;
  compact?: boolean;  // For display in messages (no remove button)
}

function getFileIcon(type: string) {
  switch (type) {
    case 'image':
      return PhotoIcon;
    case 'code':
      return CodeBracketIcon;
    case 'archive':
      return ArchiveBoxIcon;
    default:
      return DocumentIcon;
  }
}

export function AttachmentPreview({ attachment, onRemove, compact = false }: AttachmentPreviewProps) {
  const { id, filename, size, type, preview, uploading, error } = attachment;
  const Icon = getFileIcon(type);

  const isImage = type === 'image' && preview;

  return (
    <div
      className={`
        relative group
        ${compact ? 'inline-flex' : 'flex'}
        items-center gap-2 px-3 py-2 rounded-lg
        bg-chat-assistant/80 border border-chat-border/50
        ${error ? 'border-red-500/50' : ''}
        ${uploading ? 'opacity-70' : ''}
        transition-all duration-200
        ${!compact && 'hover:border-aura-purple/40'}
      `}
    >
      {/* Image preview or icon */}
      {isImage ? (
        <div className="w-10 h-10 rounded overflow-hidden flex-shrink-0 bg-gray-800">
          <img
            src={preview}
            alt={filename}
            className="w-full h-full object-cover"
          />
        </div>
      ) : (
        <div className="w-10 h-10 rounded bg-gray-800/50 flex items-center justify-center flex-shrink-0">
          <Icon className="w-5 h-5 text-chat-text-secondary" />
        </div>
      )}

      {/* File info */}
      <div className="flex-1 min-w-0 pr-6">
        <div className="text-sm text-chat-text truncate" title={filename}>
          {filename}
        </div>
        <div className="text-xs text-chat-text-secondary">
          {error ? (
            <span className="text-red-400">{error}</span>
          ) : uploading ? (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-aura-purple rounded-full animate-pulse" />
              Uploading...
            </span>
          ) : (
            formatFileSize(size)
          )}
        </div>
      </div>

      {/* Loading spinner overlay */}
      {uploading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 rounded-lg">
          <div className="w-5 h-5 border-2 border-aura-purple border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* Remove button */}
      {onRemove && !uploading && (
        <button
          onClick={() => onRemove(id)}
          className="
            absolute -top-1.5 -right-1.5
            w-5 h-5 rounded-full
            bg-gray-700 hover:bg-red-600
            flex items-center justify-center
            opacity-0 group-hover:opacity-100
            transition-all duration-200
            shadow-lg
          "
          title="Remove"
        >
          <XMarkIcon className="w-3 h-3 text-white" />
        </button>
      )}
    </div>
  );
}

interface AttachmentListProps {
  attachments: FileAttachment[];
  onRemove?: (id: string) => void;
  compact?: boolean;
}

export function AttachmentList({ attachments, onRemove, compact = false }: AttachmentListProps) {
  if (attachments.length === 0) return null;

  return (
    <div className={`flex flex-wrap gap-2 ${compact ? 'mt-2' : 'mb-3'}`}>
      {attachments.map((attachment) => (
        <AttachmentPreview
          key={attachment.id}
          attachment={attachment}
          onRemove={onRemove}
          compact={compact}
        />
      ))}
    </div>
  );
}
