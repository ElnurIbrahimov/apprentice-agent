import { toast } from '../components/Toast';

/**
 * Copy text to clipboard with consistent error handling.
 * Returns true on success, false on failure.
 * Optionally shows a toast notification.
 */
export async function copyText(text: string, showToast?: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    if (showToast) toast.success(showToast);
    return true;
  } catch {
    return false;
  }
}

/**
 * Copy an image blob to clipboard (for generated images).
 * Falls back to copying the alt text on failure.
 */
export async function copyImage(dataUrl: string, fallbackText?: string): Promise<boolean> {
  try {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    return true;
  } catch {
    if (fallbackText) return copyText(fallbackText);
    return false;
  }
}
