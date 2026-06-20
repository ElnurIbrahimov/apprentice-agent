import { useEffect, useState, useCallback, useRef } from 'react';
import { usePolling } from './usePolling';
import { apiFetch } from '../utils/apiFetch';

interface IdleBehavior {
  type: string;
  intensity: string;
  status_message: string;
  breath_rate: number;
  attention_drift: number;
  age_seconds: number;
  duration_hint: number;
}

interface IdleState {
  is_idle: boolean;
  idle_duration: number;
  current_behavior: IdleBehavior | null;
  micro_movement_seed: number;
  attention_focus: number;
  time_period: string;
}

interface AnimationParams {
  breath_rate_modifier: number;
  breath_depth_modifier: number;
  glow_intensity: number;
  attention_x: number;
  attention_y: number;
  micro_movement_x: number;
  micro_movement_y: number;
  pulse_variation: number;
}

/**
 * Hook to manage AURA's ambient idle behaviors
 * Tracks user activity and provides animation parameters for "alive" feeling
 */
export function useIdleBehaviors(enabled: boolean = true) {
  const [idleState, setIdleState] = useState<IdleState | null>(null);
  const [animationParams, setAnimationParams] = useState<AnimationParams>({
    breath_rate_modifier: 1.0,
    breath_depth_modifier: 1.0,
    glow_intensity: 0.5,
    attention_x: 0.0,
    attention_y: 0.0,
    micro_movement_x: 0.0,
    micro_movement_y: 0.0,
    pulse_variation: 0.0,
  });

  const lastActivityRef = useRef<number>(Date.now());
  const activityTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Record activity to backend
  const recordActivity = useCallback(async () => {
    if (!enabled) return;

    lastActivityRef.current = Date.now();

    // Debounce activity recording
    if (activityTimeoutRef.current) {
      clearTimeout(activityTimeoutRef.current);
    }

    activityTimeoutRef.current = setTimeout(async () => {
      try {
        await apiFetch('/api/idle/activity', { method: 'POST' });
      } catch (e: any) {
        // Silently ignore
      }
    }, 30000); // Only record after 30 seconds of no activity
  }, [enabled]);

  // Track user activity
  useEffect(() => {
    if (!enabled) return;

    let throttleTimer: ReturnType<typeof setTimeout> | null = null;

    const throttledRecordActivity = () => {
      if (throttleTimer) return;
      throttleTimer = setTimeout(() => {
        throttleTimer = null;
        recordActivity();
      }, 500);
    };

    const handleActivity = () => {
      recordActivity();
    };

    window.addEventListener('mousemove', throttledRecordActivity);
    window.addEventListener('keydown', handleActivity);
    window.addEventListener('click', handleActivity);
    window.addEventListener('scroll', handleActivity);
    window.addEventListener('touchstart', handleActivity);

    return () => {
      window.removeEventListener('mousemove', throttledRecordActivity);
      window.removeEventListener('keydown', handleActivity);
      window.removeEventListener('click', handleActivity);
      window.removeEventListener('scroll', handleActivity);
      window.removeEventListener('touchstart', handleActivity);

      if (activityTimeoutRef.current) {
        clearTimeout(activityTimeoutRef.current);
      }
      if (throttleTimer) {
        clearTimeout(throttleTimer);
      }
    };
  }, [enabled, recordActivity]);

  // Fetch idle state
  const fetchIdleState = useCallback(async () => {
    if (!enabled) return;

    try {
      const response = await apiFetch('/api/idle/state');
      if (response.ok) {
        const data = await response.json();
        setIdleState(data);
      }
    } catch (e: any) {
      // Silently ignore
    }
  }, [enabled]);

  // Fetch animation parameters
  const fetchAnimationParams = useCallback(async () => {
    if (!enabled) return;

    try {
      const response = await apiFetch('/api/idle/animation');
      if (response.ok) {
        const data = await response.json();
        setAnimationParams(data);
      }
    } catch (e: any) {
      // Silently ignore
    }
  }, [enabled]);

  // Poll for updates (slowed to 10s - idle behaviors are ambient/cosmetic)
  usePolling(fetchIdleState, 10000, { enabled });
  usePolling(fetchAnimationParams, 10000, { enabled });

  // Get current status message
  const getStatusMessage = useCallback((): string | null => {
    if (idleState?.current_behavior) {
      return idleState.current_behavior.status_message;
    }
    return null;
  }, [idleState]);

  // Check if in specific behavior type
  const isInBehavior = useCallback((behaviorType: string): boolean => {
    return idleState?.current_behavior?.type === behaviorType;
  }, [idleState]);

  // Get idle intensity level (for UI adjustments)
  const getIntensityLevel = useCallback((): 'none' | 'light' | 'medium' | 'deep' => {
    if (!idleState?.is_idle) return 'none';

    const duration = idleState.idle_duration;
    if (duration < 30) return 'light';
    if (duration < 120) return 'medium';
    return 'deep';
  }, [idleState]);

  return {
    idleState,
    animationParams,
    isIdle: idleState?.is_idle ?? false,
    idleDuration: idleState?.idle_duration ?? 0,
    currentBehavior: idleState?.current_behavior ?? null,
    timePeriod: idleState?.time_period ?? 'day',
    getStatusMessage,
    isInBehavior,
    getIntensityLevel,
    recordActivity,
  };
}
