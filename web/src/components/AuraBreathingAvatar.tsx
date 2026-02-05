import { useEffect, useState, useRef } from 'react';
import { SparklesIcon } from '@heroicons/react/24/solid';

interface AuraBreathingAvatarProps {
  isActive?: boolean;
  isThinking?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * AURA's breathing avatar with organic, randomized animations
 * Creates the illusion of life through visible, irregular breathing patterns
 */
export function AuraBreathingAvatar({
  isActive = true,
  isThinking = false,
  size = 'md'
}: AuraBreathingAvatarProps) {
  const [breathPhase, setBreathPhase] = useState(0);
  const [breathIntensity, setBreathIntensity] = useState(1);
  const animationRef = useRef<number>();

  // Organic breathing with randomized timing
  useEffect(() => {
    if (!isActive) return;

    let startTime = Date.now();
    let currentCycleDuration = 3000 + Math.random() * 1500; // 3-4.5s per breath

    const animate = () => {
      const now = Date.now();
      const elapsed = now - startTime;

      // Calculate breath phase (0-1) using easing for organic feel
      const rawPhase = (elapsed % currentCycleDuration) / currentCycleDuration;

      // Use sine-based easing for smooth breathing curve
      // Inhale is slightly faster than exhale (more natural)
      let phase: number;
      if (rawPhase < 0.4) {
        // Inhale phase (0-0.4 -> 0-1)
        phase = Math.sin((rawPhase / 0.4) * Math.PI * 0.5);
      } else {
        // Exhale phase (0.4-1 -> 1-0)
        phase = Math.cos(((rawPhase - 0.4) / 0.6) * Math.PI * 0.5);
      }

      setBreathPhase(phase);

      // At the end of each breath cycle, randomize the next one
      if (elapsed >= currentCycleDuration) {
        startTime = now;
        currentCycleDuration = 3000 + Math.random() * 1500;
        // Random intensity variation between breaths
        setBreathIntensity(0.7 + Math.random() * 0.6);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive]);

  // Size classes
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-10 h-10',
    lg: 'w-14 h-14'
  };

  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-7 h-7'
  };

  // Calculate dynamic styles based on breath phase - MORE DRAMATIC values
  const breathScale = 1 + (breathPhase * 0.08 * breathIntensity); // 1.0 to 1.08
  const glowSpread = 8 + (breathPhase * 20 * breathIntensity); // 8px to 28px glow spread
  const glowOpacity = 0.4 + (breathPhase * 0.5 * breathIntensity); // 0.4 to 0.9 opacity
  const ringScale = 1 + (breathPhase * 0.3 * breathIntensity); // 1.0 to 1.3 for outer ring
  const ringOpacity = 0.15 + (breathPhase * 0.25 * breathIntensity); // 0.15 to 0.4

  // Thinking state adds faster, more energetic animation
  const thinkingPulse = isThinking ? 'animate-aura-thinking' : '';

  return (
    <div className="relative" style={{ width: '44px', height: '44px' }}>
      {/* Outermost breathing ring - most visible effect */}
      <div
        className="absolute rounded-xl"
        style={{
          inset: '-4px',
          background: `radial-gradient(circle, rgba(139, 92, 246, ${ringOpacity}) 0%, rgba(99, 102, 241, ${ringOpacity * 0.5}) 50%, transparent 70%)`,
          transform: `scale(${ringScale})`,
          transition: 'none',
          opacity: isActive ? 1 : 0.3
        }}
      />

      {/* Pulsing glow halo */}
      <div
        className="absolute rounded-xl blur-md"
        style={{
          inset: '-2px',
          background: `rgba(139, 92, 246, ${glowOpacity * 0.6})`,
          transform: `scale(${1 + breathPhase * 0.15})`,
          opacity: isActive ? 1 : 0.3
        }}
      />

      {/* Secondary ambient glow - slower CSS animation offset */}
      <div
        className="absolute inset-0 rounded-xl blur-lg animate-aura-ambient"
        style={{
          background: 'radial-gradient(circle, rgba(168, 85, 247, 0.3) 0%, transparent 70%)',
          opacity: isActive ? 0.7 : 0.2
        }}
      />

      {/* Main avatar container */}
      <div
        className={`
          ${sizeClasses[size]} rounded-xl
          bg-gradient-to-br from-purple-600 via-purple-500 to-blue-600
          flex items-center justify-center
          relative z-10
          ${thinkingPulse}
        `}
        style={{
          transform: `scale(${breathScale})`,
          boxShadow: `
            0 0 ${glowSpread}px rgba(139, 92, 246, ${glowOpacity}),
            0 0 ${glowSpread * 2}px rgba(139, 92, 246, ${glowOpacity * 0.5}),
            inset 0 1px 2px rgba(255, 255, 255, 0.2)
          `,
          transition: 'none'
        }}
      >
        {/* Inner highlight shimmer */}
        <div
          className="absolute inset-0 rounded-xl overflow-hidden"
          style={{
            background: `linear-gradient(135deg, rgba(255,255,255,${0.15 + breathPhase * 0.15}) 0%, transparent 50%, rgba(255,255,255,0.05) 100%)`
          }}
        />

        {/* Icon with glow */}
        <SparklesIcon
          className={`${iconSizes[size]} text-white relative z-10`}
          style={{
            filter: `drop-shadow(0 0 ${3 + breathPhase * 6}px rgba(255, 255, 255, ${0.6 + breathPhase * 0.4}))`,
            transform: `scale(${1 + breathPhase * 0.08})`
          }}
        />
      </div>

      {/* Thinking indicator - expanding rings */}
      {isThinking && (
        <>
          <div className="absolute inset-0 rounded-xl border-2 border-purple-400/60 animate-aura-thinking-ring" />
          <div className="absolute inset-0 rounded-xl border border-purple-300/40 animate-aura-thinking-ring" style={{ animationDelay: '0.5s' }} />
        </>
      )}
    </div>
  );
}

/**
 * Ambient status line component - shows what AURA is "noticing"
 */
interface AuraStatusLineProps {
  status?: string | null;
  isVisible?: boolean;
}

export function AuraStatusLine({ status, isVisible = true }: AuraStatusLineProps) {
  const [displayStatus, setDisplayStatus] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (status && isVisible) {
      setIsAnimating(true);
      setDisplayStatus(status);
    } else {
      setIsAnimating(false);
      // Delay clearing to allow fade out
      const timer = setTimeout(() => setDisplayStatus(null), 300);
      return () => clearTimeout(timer);
    }
  }, [status, isVisible]);

  if (!displayStatus) return null;

  return (
    <div
      className={`
        flex items-center gap-2 text-xs text-chat-text-secondary/70
        transition-all duration-300 ease-out
        ${isAnimating ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-1'}
      `}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-purple-400/60 animate-pulse" />
      <span className="italic">{displayStatus}</span>
    </div>
  );
}

/**
 * "Decided not to speak" indicator
 */
interface AuraConsideringProps {
  isConsidering?: boolean;
  decidedAgainst?: boolean;
}

export function AuraConsideringIndicator({ isConsidering = false, decidedAgainst = false }: AuraConsideringProps) {
  if (!isConsidering && !decidedAgainst) return null;

  return (
    <div className="flex items-center gap-2 text-xs">
      {isConsidering && (
        <span className="text-purple-400/70 flex items-center gap-1.5 animate-pulse">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-400/50" />
          considering...
        </span>
      )}
      {decidedAgainst && (
        <span className="text-chat-text-secondary/50 flex items-center gap-1.5 animate-fade-out">
          <span className="w-1.5 h-1.5 rounded-full bg-gray-500/50" />
          <span className="line-through">decided against</span>
        </span>
      )}
    </div>
  );
}
