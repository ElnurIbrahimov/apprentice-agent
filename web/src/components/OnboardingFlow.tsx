import { useState, useEffect, useRef, useCallback } from 'react';
import { AuraBreathingAvatar } from './AuraBreathingAvatar';
import { useSettingsStore, applyColorPreset, type ColorPreset } from '../store/settingsStore';
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/solid';
import { ArrowRightIcon } from '@heroicons/react/24/outline';

interface OnboardingFlowProps {
  onComplete: () => void;
}

type Step = 1 | 2 | 3;

type ConnectionState = 'idle' | 'testing' | 'ok' | 'error';

const COLOR_PRESETS: { id: ColorPreset; label: string; color: string }[] = [
  { id: 'aura',   label: 'Aura',    color: '#7c3aed' },
  { id: 'ocean',  label: 'Ocean',   color: '#0284c7' },
  { id: 'forest', label: 'Forest',  color: '#059669' },
  { id: 'sunset', label: 'Sunset',  color: '#ea580c' },
  { id: 'rose',   label: 'Rose',    color: '#e11d48' },
  { id: 'mono',   label: 'Mono',    color: '#71717a' },
];

function StepDots({ current, total }: { current: Step; total: number }) {
  return (
    <div className="flex items-center justify-center gap-2">
      {Array.from({ length: total }, (_, i) => (
        <div
          key={i}
          style={{
            width: current === i + 1 ? '20px' : '6px',
            height: '6px',
            borderRadius: '9999px',
            background: current === i + 1 ? 'var(--chat-accent)' : 'rgba(255,255,255,0.2)',
            transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
          }}
        />
      ))}
    </div>
  );
}

export function OnboardingFlow({ onComplete }: OnboardingFlowProps) {
  const { updateSettings } = useSettingsStore();
  const [step, setStep] = useState<Step>(1);
  const [prevStep, setPrevStep] = useState<Step>(1);
  const [animating, setAnimating] = useState(false);

  // Step 2 state
  const defaultUrl = `${window.location.protocol}//${window.location.host}`;
  const [backendUrl, setBackendUrl] = useState(defaultUrl);
  const [connState, setConnState] = useState<ConnectionState>('idle');
  const [connError, setConnError] = useState('');

  // Step 3 state
  const [userName, setUserName] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<ColorPreset>('aura');

  const nameInputRef = useRef<HTMLInputElement>(null);
  const urlInputRef = useRef<HTMLInputElement>(null);

  // Focus inputs when step changes
  useEffect(() => {
    if (step === 2) {
      setTimeout(() => urlInputRef.current?.focus(), 350);
    } else if (step === 3) {
      setTimeout(() => nameInputRef.current?.focus(), 350);
    }
  }, [step]);

  // Preview color preset live on step 3
  useEffect(() => {
    if (step === 3) {
      applyColorPreset(selectedPreset);
    }
  }, [selectedPreset, step]);

  const goToStep = useCallback((next: Step) => {
    if (animating) return;
    setPrevStep(step);
    setAnimating(true);
    setTimeout(() => {
      setStep(next);
      setAnimating(false);
    }, 260);
  }, [animating, step]);

  const testConnection = useCallback(async () => {
    setConnState('testing');
    setConnError('');
    const url = backendUrl.replace(/\/$/, '');
    try {
      const res = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        setConnState('ok');
      } else {
        setConnState('error');
        setConnError(`Server responded with ${res.status}`);
      }
    } catch (err) {
      setConnState('error');
      const msg = err instanceof Error ? err.message : 'Could not reach server';
      setConnError(msg.includes('aborted') ? 'Connection timed out' : msg);
    }
  }, [backendUrl]);

  const handleComplete = useCallback(() => {
    updateSettings({
      onboardingDone: true,
      userName: userName.trim(),
      colorPreset: selectedPreset,
      backendUrl: backendUrl.replace(/\/$/, ''),
    });
    onComplete();
  }, [updateSettings, userName, selectedPreset, backendUrl, onComplete]);

  // Keyboard handler
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const isInput = tag === 'INPUT' || tag === 'TEXTAREA';

      if (e.key === 'Escape') {
        // Skip to end on escape
        if (step === 2) { goToStep(3); }
      }

      if (e.key === 'Enter' && !e.shiftKey) {
        if (step === 1) { goToStep(2); }
        else if (step === 2 && !isInput) { goToStep(3); }
        else if (step === 3 && !isInput) { handleComplete(); }
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [step, goToStep, handleComplete]);

  const slideDir = step > prevStep ? 1 : -1;

  return (
    // Full-screen backdrop
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: 'var(--bg-base)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
    >
      {/* Ambient mesh blobs */}
      <div className="mesh-bg" aria-hidden>
        <div className="mesh-blob blob-1" />
        <div className="mesh-blob blob-2" />
        <div className="mesh-blob blob-3" />
      </div>
      <div className="grain" aria-hidden />

      {/* Card */}
      <div
        style={{
          position: 'relative',
          width: '100%',
          maxWidth: '440px',
          background: 'var(--surface-1)',
          border: '1px solid var(--border-default)',
          borderRadius: '24px',
          boxShadow: '0 32px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.04)',
          padding: '40px 36px 32px',
          overflow: 'hidden',
        }}
      >
        {/* Subtle top accent line */}
        <div
          aria-hidden
          style={{
            position: 'absolute',
            top: 0,
            left: '20%',
            right: '20%',
            height: '1px',
            background: 'linear-gradient(90deg, transparent, var(--chat-accent), transparent)',
            opacity: 0.6,
          }}
        />

        {/* Animated content wrapper */}
        <div
          style={{
            transform: animating ? `translateX(${slideDir * -30}px)` : 'translateX(0)',
            opacity: animating ? 0 : 1,
            transition: 'transform 0.26s cubic-bezier(0.16,1,0.3,1), opacity 0.26s ease',
          }}
        >
          {step === 1 && <StepWelcome onNext={() => goToStep(2)} />}
          {step === 2 && (
            <StepConnect
              backendUrl={backendUrl}
              setBackendUrl={setBackendUrl}
              connState={connState}
              connError={connError}
              onTest={testConnection}
              onNext={() => goToStep(3)}
              onSkip={() => { setBackendUrl(defaultUrl); goToStep(3); }}
              urlInputRef={urlInputRef}
            />
          )}
          {step === 3 && (
            <StepPersonalize
              userName={userName}
              setUserName={setUserName}
              selectedPreset={selectedPreset}
              setSelectedPreset={setSelectedPreset}
              onComplete={handleComplete}
              nameInputRef={nameInputRef}
            />
          )}
        </div>

        {/* Step indicator */}
        <div style={{ marginTop: '32px' }}>
          <StepDots current={step} total={3} />
        </div>
      </div>
    </div>
  );
}

/* ── Step 1: Welcome ─────────────────────────────────────────────── */

function StepWelcome({ onNext }: { onNext: () => void }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '28px' }}>
        <div style={{ transform: 'scale(2)', transformOrigin: 'center' }}>
          <AuraBreathingAvatar isActive size="lg" />
        </div>
      </div>

      <h1
        style={{
          fontSize: '26px',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: '10px',
          letterSpacing: '-0.5px',
        }}
      >
        Welcome to Aura
      </h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: '15px', marginBottom: '36px', lineHeight: 1.5 }}>
        Your personal AI assistant
      </p>

      <button
        onClick={onNext}
        style={{
          width: '100%',
          padding: '13px 24px',
          borderRadius: '12px',
          background: 'var(--chat-accent)',
          color: '#fff',
          fontSize: '15px',
          fontWeight: 600,
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          transition: 'background 0.15s ease, transform 0.1s ease',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--chat-accent-hover)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'var(--chat-accent)')}
        onMouseDown={e => (e.currentTarget.style.transform = 'scale(0.98)')}
        onMouseUp={e => (e.currentTarget.style.transform = 'scale(1)')}
      >
        Get Started
        <ArrowRightIcon style={{ width: '16px', height: '16px' }} />
      </button>
    </div>
  );
}

/* ── Step 2: Connect ─────────────────────────────────────────────── */

interface StepConnectProps {
  backendUrl: string;
  setBackendUrl: (v: string) => void;
  connState: ConnectionState;
  connError: string;
  onTest: () => void;
  onNext: () => void;
  onSkip: () => void;
  urlInputRef: React.RefObject<HTMLInputElement>;
}

function StepConnect({
  backendUrl, setBackendUrl, connState, connError,
  onTest, onNext, onSkip, urlInputRef,
}: StepConnectProps) {
  const canProceed = connState === 'ok';

  return (
    <div>
      <h2
        style={{
          fontSize: '20px',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: '6px',
        }}
      >
        Connect to backend
      </h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px' }}>
        Point Aura at your running server
      </p>

      {/* URL input */}
      <label style={{ display: 'block', marginBottom: '8px' }}>
        <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Backend URL
        </span>
        <div style={{ position: 'relative', marginTop: '6px' }}>
          <input
            ref={urlInputRef}
            type="url"
            value={backendUrl}
            onChange={e => { setBackendUrl(e.target.value); }}
            placeholder="http://localhost:8000"
            onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); onTest(); } }}
            style={{
              width: '100%',
              padding: '11px 42px 11px 14px',
              borderRadius: '10px',
              border: `1px solid ${connState === 'ok' ? 'rgba(16,185,129,0.5)' : connState === 'error' ? 'rgba(239,68,68,0.5)' : 'var(--border-default)'}`,
              background: 'var(--surface-2)',
              color: 'var(--text-primary)',
              fontSize: '14px',
              fontFamily: 'inherit',
              outline: 'none',
              transition: 'border-color 0.2s ease',
            }}
          />
          {/* Status icon inside input */}
          {connState !== 'idle' && connState !== 'testing' && (
            <div style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)' }}>
              {connState === 'ok'
                ? <CheckCircleIcon style={{ width: '18px', height: '18px', color: '#10b981' }} />
                : <XCircleIcon style={{ width: '18px', height: '18px', color: '#ef4444' }} />
              }
            </div>
          )}
          {connState === 'testing' && (
            <div style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)' }}>
              <div style={{
                width: '16px', height: '16px',
                border: '2px solid var(--border-default)',
                borderTopColor: 'var(--chat-accent)',
                borderRadius: '50%',
                animation: 'spin 0.7s linear infinite',
              }} />
            </div>
          )}
        </div>
      </label>

      {/* Error message */}
      {connState === 'error' && connError && (
        <p style={{ color: '#ef4444', fontSize: '13px', marginTop: '6px' }}>{connError}</p>
      )}
      {connState === 'ok' && (
        <p style={{ color: '#10b981', fontSize: '13px', marginTop: '6px' }}>Connected successfully</p>
      )}

      {/* Test button */}
      <button
        onClick={onTest}
        disabled={connState === 'testing' || !backendUrl.trim()}
        style={{
          width: '100%',
          padding: '11px',
          marginTop: '14px',
          borderRadius: '10px',
          border: '1px solid var(--border-default)',
          background: connState === 'ok' ? 'rgba(16,185,129,0.12)' : 'var(--surface-3)',
          color: connState === 'ok' ? '#10b981' : 'var(--text-primary)',
          fontSize: '14px',
          fontWeight: 500,
          cursor: connState === 'testing' ? 'not-allowed' : 'pointer',
          opacity: connState === 'testing' ? 0.7 : 1,
          transition: 'background 0.2s ease',
        }}
      >
        {connState === 'testing' ? 'Testing...' : connState === 'ok' ? 'Test again' : 'Test connection'}
      </button>

      {/* Continue / skip */}
      <div style={{ display: 'flex', gap: '10px', marginTop: '12px' }}>
        <button
          onClick={onSkip}
          style={{
            flex: 1,
            padding: '11px',
            borderRadius: '10px',
            border: '1px solid var(--border-default)',
            background: 'transparent',
            color: 'var(--text-secondary)',
            fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          Skip (use default)
        </button>
        <button
          onClick={onNext}
          disabled={!canProceed}
          style={{
            flex: 2,
            padding: '11px',
            borderRadius: '10px',
            border: 'none',
            background: canProceed ? 'var(--chat-accent)' : 'var(--surface-3)',
            color: canProceed ? '#fff' : 'var(--text-tertiary)',
            fontSize: '14px',
            fontWeight: 600,
            cursor: canProceed ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'background 0.2s ease',
          }}
        >
          Continue
          <ArrowRightIcon style={{ width: '14px', height: '14px' }} />
        </button>
      </div>
    </div>
  );
}

/* ── Step 3: Personalize ─────────────────────────────────────────── */

interface StepPersonalizeProps {
  userName: string;
  setUserName: (v: string) => void;
  selectedPreset: ColorPreset;
  setSelectedPreset: (v: ColorPreset) => void;
  onComplete: () => void;
  nameInputRef: React.RefObject<HTMLInputElement>;
}

function StepPersonalize({
  userName, setUserName, selectedPreset, setSelectedPreset,
  onComplete, nameInputRef,
}: StepPersonalizeProps) {
  return (
    <div>
      <h2
        style={{
          fontSize: '20px',
          fontWeight: 700,
          color: 'var(--text-primary)',
          marginBottom: '6px',
        }}
      >
        Personalize
      </h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '24px' }}>
        Make Aura yours
      </p>

      {/* Name input */}
      <label style={{ display: 'block', marginBottom: '22px' }}>
        <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          What should I call you?
        </span>
        <input
          ref={nameInputRef}
          type="text"
          value={userName}
          onChange={e => setUserName(e.target.value)}
          placeholder="Your name (optional)"
          maxLength={40}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); onComplete(); } }}
          style={{
            display: 'block',
            width: '100%',
            marginTop: '8px',
            padding: '11px 14px',
            borderRadius: '10px',
            border: '1px solid var(--border-default)',
            background: 'var(--surface-2)',
            color: 'var(--text-primary)',
            fontSize: '14px',
            fontFamily: 'inherit',
            outline: 'none',
          }}
        />
      </label>

      {/* Color preset */}
      <div style={{ marginBottom: '28px' }}>
        <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Accent color
        </span>
        <div style={{ display: 'flex', gap: '10px', marginTop: '10px', flexWrap: 'wrap' }}>
          {COLOR_PRESETS.map(preset => (
            <button
              key={preset.id}
              title={preset.label}
              onClick={() => setSelectedPreset(preset.id)}
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: preset.color,
                border: selectedPreset === preset.id
                  ? `2px solid #fff`
                  : '2px solid transparent',
                boxShadow: selectedPreset === preset.id
                  ? `0 0 0 3px ${preset.color}60, 0 0 16px ${preset.color}80`
                  : `0 2px 8px ${preset.color}40`,
                cursor: 'pointer',
                transform: selectedPreset === preset.id ? 'scale(1.15)' : 'scale(1)',
                transition: 'transform 0.2s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.2s ease, border-color 0.2s ease',
              }}
            />
          ))}
        </div>
      </div>

      {/* Start chatting */}
      <button
        onClick={onComplete}
        style={{
          width: '100%',
          padding: '13px 24px',
          borderRadius: '12px',
          background: 'var(--chat-accent)',
          color: '#fff',
          fontSize: '15px',
          fontWeight: 600,
          border: 'none',
          cursor: 'pointer',
          transition: 'background 0.15s ease, transform 0.1s ease',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--chat-accent-hover)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'var(--chat-accent)')}
        onMouseDown={e => (e.currentTarget.style.transform = 'scale(0.98)')}
        onMouseUp={e => (e.currentTarget.style.transform = 'scale(1)')}
      >
        Start chatting
      </button>
    </div>
  );
}
