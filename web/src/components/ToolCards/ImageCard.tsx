import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';

// Renders a generated image with the prompt caption + a download link.

interface ImageCardProps {
  result: Extract<ToolResult, { kind: 'image' }>;
}

export function ImageCard({ result }: ImageCardProps) {
  const src = result.imageUrl
    || (result.imageB64 ? `data:image/png;base64,${result.imageB64}` : '');
  const promptText = result.prompt || '';

  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F3A8}'}
      title="Image"
      subtitle={promptText.length > 48 ? `${promptText.slice(0, 48)}\u2026` : (promptText || undefined)}
      status={result.status}
    >
      {src && (
        <div className="tool-card-image">
          <img src={src} alt={promptText} loading="lazy" />
          <div className="tool-card-image-actions">
            <a href={src} download={`aura-image-${result.id}.png`} className="tool-card-btn">
              Download
            </a>
          </div>
        </div>
      )}
      {!src && result.status === 'running' && (
        <div className="tool-card-placeholder">Generating\u2026</div>
      )}
      {!src && result.status === 'error' && (
        <div className="tool-card-placeholder tool-card-placeholder--error">
          Image generation failed
        </div>
      )}
    </ToolCardShell>
  );
}
