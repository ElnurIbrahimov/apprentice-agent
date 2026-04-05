import { useRef, useEffect, useCallback, useState, type ReactNode } from 'react';
import { haptics } from '../utils/haptics';

/* ─── Types ─── */
interface BottomSheetProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: string;
  /** Snap points as fractions of viewport height (0–1). Default: [0.5, 0.92] */
  snapPoints?: number[];
  /** Which snap point to open to (index into snapPoints). Default: 0 */
  initialSnap?: number;
  /** Whether the sheet can be dismissed by dragging down. Default: true */
  dismissible?: boolean;
  /** Whether to show the drag handle. Default: true */
  showHandle?: boolean;
  /** Additional class for the sheet body */
  className?: string;
}

/* ─── Constants ─── */
const DISMISS_VELOCITY = 800;  // px/s — fling down faster than this dismisses
const DISMISS_FRACTION = 0.25; // below 25% of first snap → dismiss
const OVERDRAG_RESISTANCE = 0.3; // rubber-band factor when dragging above max snap

/* ─── Spring utility ─── */
function springTransform(el: HTMLElement, toY: number, opts?: { velocity?: number }) {
  const v = opts?.velocity ?? 0;
  // Duration based on distance and velocity — feels iOS-native
  const dist = Math.abs(toY - parseFloat(el.dataset.currentY || '0'));
  const baseDuration = Math.max(250, Math.min(500, dist * 1.2));
  const velocityBoost = Math.min(150, Math.abs(v) * 0.15);
  const duration = baseDuration - velocityBoost;

  el.style.transition = `transform ${duration}ms cubic-bezier(0.34, 1.56, 0.64, 1)`;
  el.style.transform = `translateY(${toY}px)`;
  el.dataset.currentY = String(toY);
}

export function BottomSheet({
  open,
  onClose,
  children,
  title,
  snapPoints = [0.5, 0.92],
  initialSnap = 0,
  dismissible = true,
  showHandle = true,
  className,
}: BottomSheetProps) {
  const sheetRef = useRef<HTMLDivElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);
  const startYRef = useRef(0);
  const lastYRef = useRef(0);
  const lastTimeRef = useRef(0);
  const velocityRef = useRef(0);
  const isDraggingRef = useRef(false);
  const currentSnapRef = useRef(initialSnap);
  const [isVisible, setIsVisible] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const vh = typeof window !== 'undefined' ? window.innerHeight : 800;

  // Convert snap fractions to pixel Y positions (from top)
  const snapPositions = snapPoints.map(frac => vh * (1 - frac));

  // Get the Y position to dismiss to (off-screen)
  const dismissY = vh + 20;

  /* ─── Open / close lifecycle ─── */
  useEffect(() => {
    if (open) {
      setIsVisible(true);
      setIsClosing(false);
      // Start from off-screen, then animate to initial snap
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const sheet = sheetRef.current;
          if (sheet) {
            sheet.dataset.currentY = String(dismissY);
            springTransform(sheet, snapPositions[initialSnap]);
            currentSnapRef.current = initialSnap;
          }
          // Fade backdrop
          if (backdropRef.current) {
            backdropRef.current.style.opacity = '1';
          }
        });
      });
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const dismiss = useCallback(() => {
    setIsClosing(true);
    const sheet = sheetRef.current;
    if (sheet) {
      springTransform(sheet, dismissY);
    }
    if (backdropRef.current) {
      backdropRef.current.style.opacity = '0';
    }
    setTimeout(() => {
      setIsVisible(false);
      setIsClosing(false);
      onClose();
    }, 350);
  }, [onClose, dismissY]);

  /* ─── Touch handling ─── */
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const sheet = sheetRef.current;
    if (!sheet) return;
    isDraggingRef.current = true;
    startYRef.current = e.touches[0].clientY;
    lastYRef.current = e.touches[0].clientY;
    lastTimeRef.current = Date.now();
    velocityRef.current = 0;
    sheet.style.transition = 'none'; // remove spring during drag
  }, []);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!isDraggingRef.current) return;
    const sheet = sheetRef.current;
    if (!sheet) return;

    const touchY = e.touches[0].clientY;
    const now = Date.now();
    const dt = now - lastTimeRef.current;
    if (dt > 0) {
      velocityRef.current = (touchY - lastYRef.current) / dt * 1000; // px/s
    }
    lastYRef.current = touchY;
    lastTimeRef.current = now;

    const currentSnap = snapPositions[currentSnapRef.current];
    const delta = touchY - startYRef.current;
    let newY = currentSnap + delta;

    // Rubber-band when dragging above the highest snap point
    const minY = snapPositions[snapPositions.length - 1];
    if (newY < minY) {
      const overDrag = minY - newY;
      newY = minY - overDrag * OVERDRAG_RESISTANCE;
    }

    // Don't allow dragging below dismiss point
    newY = Math.min(newY, dismissY);

    sheet.style.transform = `translateY(${newY}px)`;
    sheet.dataset.currentY = String(newY);

    // Backdrop opacity tied to sheet position
    if (backdropRef.current) {
      const progress = Math.max(0, Math.min(1, 1 - (newY - snapPositions[0]) / (dismissY - snapPositions[0])));
      backdropRef.current.style.opacity = String(progress);
    }
  }, [snapPositions, dismissY]);

  const handleTouchEnd = useCallback(() => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;

    const sheet = sheetRef.current;
    if (!sheet) return;

    const currentY = parseFloat(sheet.dataset.currentY || '0');
    const velocity = velocityRef.current;

    // Fling dismiss — fast downward swipe
    if (dismissible && velocity > DISMISS_VELOCITY) {
      haptics.light();
      dismiss();
      return;
    }

    // Fling up — snap to higher point
    if (velocity < -DISMISS_VELOCITY && currentSnapRef.current < snapPoints.length - 1) {
      haptics.light();
      currentSnapRef.current++;
      springTransform(sheet, snapPositions[currentSnapRef.current], { velocity });
      if (backdropRef.current) {
        backdropRef.current.style.transition = 'opacity 0.3s ease';
        backdropRef.current.style.opacity = '1';
      }
      return;
    }

    // Position-based snap — find closest snap point
    const firstSnapY = snapPositions[0];
    if (dismissible && currentY > firstSnapY + (dismissY - firstSnapY) * (1 - DISMISS_FRACTION)) {
      haptics.light();
      dismiss();
      return;
    }

    // Find nearest snap
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < snapPositions.length; i++) {
      const d = Math.abs(currentY - snapPositions[i]);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    currentSnapRef.current = bestIdx;
    springTransform(sheet, snapPositions[bestIdx], { velocity });
    if (backdropRef.current) {
      backdropRef.current.style.transition = 'opacity 0.3s ease';
      backdropRef.current.style.opacity = '1';
    }
  }, [dismiss, dismissible, snapPoints.length, snapPositions, dismissY]);

  /* ─── Keyboard dismiss ─── */
  useEffect(() => {
    if (!isVisible) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dismissible) {
        e.preventDefault();
        dismiss();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isVisible, dismiss, dismissible]);

  if (!isVisible && !open) return null;

  return (
    <>
      {/* Backdrop — blurred, opacity animated */}
      <div
        ref={backdropRef}
        className="fixed inset-0 z-[900] lg:hidden"
        style={{
          background: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(4px)',
          WebkitBackdropFilter: 'blur(4px)',
          opacity: 0,
          transition: 'opacity 0.3s ease',
          pointerEvents: isClosing ? 'none' : 'auto',
        }}
        onClick={dismissible ? dismiss : undefined}
        aria-hidden="true"
      />

      {/* Sheet */}
      <div
        ref={sheetRef}
        className={`fixed inset-x-0 bottom-0 z-[901] lg:hidden ${className || ''}`}
        style={{
          height: `${snapPoints[snapPoints.length - 1] * 100 + 5}vh`,
          transform: `translateY(${dismissY}px)`,
          borderRadius: '20px 20px 0 0',
          background: 'var(--surface-1)',
          borderTop: '1px solid var(--border-default)',
          boxShadow: '0 -8px 40px rgba(0, 0, 0, 0.3)',
          willChange: 'transform',
          touchAction: 'none',
          overscrollBehavior: 'contain',
        }}
        role="dialog"
        aria-modal="true"
        aria-label={title || 'Bottom sheet'}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {/* Drag handle */}
        {showHandle && (
          <div className="flex justify-center pt-3 pb-2">
            <div
              className="w-9 h-1 rounded-full"
              style={{ background: 'var(--border-strong)' }}
            />
          </div>
        )}

        {/* Title */}
        {title && (
          <div className="px-5 pb-3 text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            {title}
          </div>
        )}

        {/* Content — scrollable within the sheet */}
        <div
          className="flex-1 overflow-y-auto overscroll-contain px-2"
          style={{
            maxHeight: `calc(${snapPoints[snapPoints.length - 1] * 100}vh - ${title ? 80 : 48}px)`,
            paddingBottom: 'calc(1rem + env(safe-area-inset-bottom, 0px))',
          }}
          onTouchStart={(e) => e.stopPropagation()} // let content scroll without triggering sheet drag
        >
          {children}
        </div>
      </div>
    </>
  );
}

/* ─── Convenience wrapper: BottomSheet for action menus ─── */
interface ActionSheetItem {
  icon?: ReactNode;
  label: string;
  sublabel?: string;
  active?: boolean;
  activeColor?: string;
  destructive?: boolean;
  onPress: () => void;
}

interface ActionSheetProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  items: ActionSheetItem[];
}

export function ActionSheet({ open, onClose, title, items }: ActionSheetProps) {
  return (
    <BottomSheet
      open={open}
      onClose={onClose}
      title={title}
      snapPoints={[Math.min(0.55, 0.1 + items.length * 0.065)]}
      showHandle={true}
    >
      <div className="flex flex-col gap-0.5 px-2 pb-2">
        {items.map((item, i) => (
          <button
            key={i}
            onClick={() => {
              haptics.light();
              item.onPress();
            }}
            className="flex items-center gap-3 px-4 py-3.5 rounded-2xl text-left transition-all active:scale-[0.97]"
            style={{
              background: item.active ? (item.activeColor || 'var(--chat-accent)') + '22' : 'transparent',
              color: item.destructive ? '#ef4444' : item.active ? 'var(--text-primary)' : 'var(--text-secondary)',
            }}
          >
            {item.icon && (
              <span className="w-5 h-5 flex items-center justify-center flex-shrink-0 opacity-70">
                {item.icon}
              </span>
            )}
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium">{item.label}</div>
              {item.sublabel && (
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
                  {item.sublabel}
                </div>
              )}
            </div>
            {item.active && (
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: item.activeColor || 'var(--chat-accent)' }} />
            )}
          </button>
        ))}
      </div>
    </BottomSheet>
  );
}
