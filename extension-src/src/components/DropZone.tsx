import React, { useState, useRef, useCallback } from 'react';
import { Upload } from 'lucide-react';
import type { FileAttachment } from '../types';

const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
const ACCEPTED_TEXT_TYPES = ['text/plain', 'text/csv', 'text/html', 'text/xml', 'application/xml'];
const ACCEPTED_PDF_TYPES = ['application/pdf'];

const CODE_EXTENSIONS = [
  '.js', '.ts', '.tsx', '.jsx', '.py', '.rb', '.go', '.rs', '.java', '.c', '.cpp', '.h',
  '.css', '.scss', '.less', '.json', '.yaml', '.yml', '.toml', '.xml', '.html', '.md',
  '.sh', '.bash', '.zsh', '.sql', '.lua', '.php', '.swift', '.kt', '.r', '.m',
];

const MAX_FILE_SIZE = 2 * 1024 * 1024; // 2MB per file
const MAX_TEXT_SIZE = 100 * 1024; // 100KB for text content extraction

function classifyFile(file: File): FileAttachment['type'] | null {
  if (ACCEPTED_IMAGE_TYPES.includes(file.type)) return 'image';
  if (ACCEPTED_PDF_TYPES.includes(file.type)) return 'pdf';
  if (ACCEPTED_TEXT_TYPES.includes(file.type)) return 'text';

  // Check extension for code files
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  if (CODE_EXTENSIONS.includes(ext)) return 'code';

  // Fallback: treat small files with no type as text
  if (!file.type && file.size < MAX_TEXT_SIZE) return 'text';

  return null;
}

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the data:...;base64, prefix
      resolve(result.split(',')[1] || result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

async function processFile(file: File): Promise<FileAttachment | null> {
  if (file.size > MAX_FILE_SIZE) return null;

  const fileType = classifyFile(file);
  if (!fileType) return null;

  const base: Omit<FileAttachment, 'data' | 'textContent'> = {
    id: crypto.randomUUID(),
    name: file.name,
    type: fileType,
    mimeType: file.type || 'application/octet-stream',
    size: file.size,
  };

  if (fileType === 'image') {
    const data = await readFileAsBase64(file);
    return { ...base, data };
  }

  if (fileType === 'text' || fileType === 'code') {
    if (file.size > MAX_TEXT_SIZE) {
      // Too big for text extraction — just reference
      return { ...base, textContent: `(File too large for inline preview: ${(file.size / 1024).toFixed(1)}KB)` };
    }
    const textContent = await readFileAsText(file);
    return { ...base, textContent };
  }

  // PDF — just attach metadata
  return { ...base };
}

interface Props {
  children: React.ReactNode;
  onFilesAdded: (files: FileAttachment[]) => void;
}

export default function DropZone({ children, onFilesAdded }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  // Counter to handle child element dragenter/dragleave flicker
  const dragCounter = useRef(0);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (dragCounter.current === 1) {
      // Check if it actually has files
      if (e.dataTransfer.types.includes('Files')) {
        setIsDragging(true);
      }
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    const processed = await Promise.all(files.map(processFile));
    const valid = processed.filter(Boolean) as FileAttachment[];
    if (valid.length > 0) {
      onFilesAdded(valid);
    }
  }, [onFilesAdded]);

  return (
    <div
      style={{ position: 'relative', display: 'contents' }}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {children}

      {/* Drop overlay */}
      {isDragging && (
        <div className="dropzone-overlay">
          <div className="dropzone-content">
            <Upload size={32} className="dropzone-icon" />
            <span className="dropzone-label">Drop file to send to AURA</span>
            <span className="dropzone-hint">Images, PDFs, text, or code files</span>
          </div>
        </div>
      )}
    </div>
  );
}

// Export processFile for clipboard paste usage in InputBar
export { processFile };
