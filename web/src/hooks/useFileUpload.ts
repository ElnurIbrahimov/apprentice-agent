import { useState, useCallback } from 'react';
import type { FileAttachment, AttachmentType } from '../types';
import { useChatStore } from '../store/chatStore';

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Supported file extensions
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']);
const DOCUMENT_EXTENSIONS = new Set(['.pdf', '.txt', '.md', '.json']);
const CODE_EXTENSIONS = new Set([
  '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css',
  '.java', '.c', '.cpp', '.h', '.go', '.rs', '.rb',
  '.php', '.sh', '.yaml', '.yml', '.toml', '.xml', '.sql'
]);
const ARCHIVE_EXTENSIONS = new Set(['.zip']);
const MAX_ARCHIVE_SIZE = 50 * 1024 * 1024; // 50MB

function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf('.');
  return lastDot === -1 ? '' : filename.slice(lastDot).toLowerCase();
}

function getAttachmentType(filename: string): AttachmentType | null {
  const ext = getFileExtension(filename);
  if (IMAGE_EXTENSIONS.has(ext)) return 'image';
  if (DOCUMENT_EXTENSIONS.has(ext)) return 'document';
  if (CODE_EXTENSIONS.has(ext)) return 'code';
  if (ARCHIVE_EXTENSIONS.has(ext)) return 'archive';
  return null;
}

function isSupported(filename: string): boolean {
  return getAttachmentType(filename) !== null;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface UseFileUploadReturn {
  attachments: FileAttachment[];
  uploadFile: (file: File) => Promise<FileAttachment | null>;
  uploadFiles: (files: FileList | File[]) => Promise<void>;
  removeAttachment: (id: string) => void;
  clearAttachments: () => void;
  isUploading: boolean;
}

export function useFileUpload(): UseFileUploadReturn {
  const [attachments, setAttachments] = useState<FileAttachment[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const createPreview = useCallback((file: File): Promise<string | undefined> => {
    return new Promise((resolve) => {
      if (!file.type.startsWith('image/')) {
        resolve(undefined);
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target?.result as string);
      };
      reader.onerror = () => resolve(undefined);
      reader.readAsDataURL(file);
    });
  }, []);

  const uploadFile = useCallback(async (file: File): Promise<FileAttachment | null> => {
    // Validate file type
    if (!isSupported(file.name)) {
      const ext = getFileExtension(file.name) || file.name.split('.').pop() || 'unknown';
      useChatStore.getState().setError(`Unsupported file type: .${ext}`);
      return null;
    }

    // Validate file size (archives get a higher limit)
    const sizeLimit = ARCHIVE_EXTENSIONS.has(getFileExtension(file.name)) ? MAX_ARCHIVE_SIZE : MAX_FILE_SIZE;
    if (file.size > sizeLimit) {
      useChatStore.getState().setError(`File too large: ${file.name} (${formatFileSize(file.size)})`);
      return null;
    }

    const tempId = `temp-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const attachmentType = getAttachmentType(file.name)!;

    // Create local preview for images
    const preview = await createPreview(file);

    // Add placeholder attachment
    const placeholder: FileAttachment = {
      id: tempId,
      filename: file.name,
      mimeType: file.type || 'application/octet-stream',
      size: file.size,
      type: attachmentType,
      preview,
      uploading: true,
    };

    setAttachments((prev) => [...prev, placeholder]);

    try {
      // Upload to server
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
      }
      const result = await response.json();

      if (result.success && result.attachment) {
        // Update with server response
        const uploaded: FileAttachment = {
          id: result.attachment.id,
          filename: result.attachment.filename,
          mimeType: result.attachment.mime_type,
          size: result.attachment.size,
          type: result.attachment.type as AttachmentType,
          path: result.attachment.path,
          preview,
          uploading: false,
        };

        setAttachments((prev) =>
          prev.map((a) => (a.id === tempId ? uploaded : a))
        );

        return uploaded;
      } else {
        // Upload failed
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === tempId
              ? { ...a, uploading: false, error: result.error || 'Upload failed' }
              : a
          )
        );
        return null;
      }
    } catch (error) {
      console.error('Upload error:', error);
      setAttachments((prev) =>
        prev.map((a) =>
          a.id === tempId
            ? { ...a, uploading: false, error: error instanceof Error ? error.message : 'Upload failed' }
            : a
        )
      );
      return null;
    }
  }, [createPreview]);

  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    setIsUploading(true);
    const fileArray = Array.from(files);

    await Promise.all(fileArray.map((file) => uploadFile(file)));

    setIsUploading(false);
  }, [uploadFile]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const clearAttachments = useCallback(() => {
    setAttachments([]);
  }, []);

  return {
    attachments,
    uploadFile,
    uploadFiles,
    removeAttachment,
    clearAttachments,
    isUploading,
  };
}

export { formatFileSize, isSupported, getAttachmentType };
