import { useRef, useEffect, useCallback, type ReactNode } from 'react';

interface BottomSheetProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
}

const HALF_HEIGHT = 50; // vh
const FULL_HEIGHT = 90; // vh
const DISMISS_THRESHOLD = 30; // vh — below this, dismiss

export function BottomSheet({ open, onClose, children, title }: BottomSheetProps) {
  const sheetRef = useRef<HTMLDivElement>(null);
  const startYRef = useRef(0);
  const currentTranslateRef = useRef(0);
  const snapHeightRef = useRef(HALF_HEIGHT);
  const isDraggingRef = useRef(false);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup dismiss timer on unmount
  useEffect(() => {
    return () => { if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current); };
  }, []);

  // Snap to a height (in vh)
  const snapTo = useCallback((vh: number) => {
    const sheet = sheetRef.current;
    if (!sheet) return;
    snapHeightRef.current = vh;
    const translate = 100 - vh; // translate in % of sheet height (100vh basis)
    sheet.style.transition = 'transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)';
    sheet.style.transform = `translateY(${translate}vh)`;
    currentTranslateRef.current = translate;
  }, []);

  // Open → snap to half
  useEffect(() => {
    if (open) {
      // Small delay so the transition plays from the closed position
      requestAnimationFrame(() => snapTo(HALF_HEIGHT));
    }
  }, [open, snapTo]);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const sheet = sheetRef.current;
    if (!sheet) return;
    isDraggingRef.current = true;
    startYRef.current = e.touches[0].clientY;
    sheet.style.transition = 'none';
  }, []);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!isDraggingRef.current) return;
    const sheet = sheetRef.current;
    if (!sheet) return;
    const deltaY = e.touches[0].clientY - startYRef.current;
    const deltVh = (deltaY / window.innerHeight) * 100;
    const baseTranslate = 100 - snapHeightRef.current;
    const newTranslate = Math.max(baseTranslate + deltVh, 10); // clamp — at least 10vh from top
    sheet.style.transform = `translateY(${newTranslate}vh)`;
    currentTranslateRef.current = newTranslate;
  }, []);

  const handleTouchEnd = useCallback(() => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    const currentHeight = 100 - currentTranslateRef.current; // in vh
    if (currentHeight < DISMISS_THRESHOLD) {
      // Dismiss
      const sheet = sheetRef.current;
      if (sheet) {
        sheet.style.transition = 'transform 0.3s ease-in';
        sheet.style.transform = 'translateY(100vh)';
      }
      dismissTimerRef.current = setTimeout(onClose, 300);
    } else if (currentHeight > (HALF_HEIGHT + FULL_HEIGHT) / 2) {
      snapTo(FULL_HEIGHT);
    } else {
      snapTo(HALF_HEIGHT);
    }
  }, [onClose, snapTo]);

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="bottom-sheet-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Sheet */}
      <div
        ref={sheetRef}
        className="bottom-sheet"
        style={{ transform: 'translateY(100vh)' }}
        role="dialog"
        aria-modal="true"
        aria-label={title || 'Bottom sheet'}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {/* Drag handle */}
        <div className="bottom-sheet-handle-area">
          <div className="bottom-sheet-handle" />
        </div>
        {/* Title */}
        {title && (
          <div className="bottom-sheet-title">{title}</div>
        )}
        {/* Content */}
        <div className="bottom-sheet-content">
          {children}
        </div>
      </div>
    </>
  );
}
