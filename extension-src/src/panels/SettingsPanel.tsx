import React, { useState, useEffect, useRef } from 'react';
import { useStore } from '../store';
import { User, FileText, Sparkles, RotateCcw, Save, Check } from 'lucide-react';

const STYLE_PRESETS: { label: string; icon: string; instructions: string }[] = [
  {
    label: 'Concise',
    icon: '',
    instructions:
      'Be concise and to the point. Use short sentences. Avoid filler words, unnecessary explanations, and over-qualifying. Get straight to the answer.',
  },
  {
    label: 'Detailed',
    icon: '',
    instructions:
      'Provide thorough, detailed responses. Include context, examples, and edge cases. Explain the reasoning behind your answers. Be comprehensive but organized.',
  },
  {
    label: 'Technical',
    icon: '',
    instructions:
      'Respond in a precise, technical manner. Use proper terminology. Include code examples, specs, and technical details. Assume I have engineering knowledge.',
  },
  {
    label: 'Casual',
    icon: '',
    instructions:
      'Keep it casual and conversational. Use simple language, be friendly, and feel free to use humor. Explain things like you would to a friend.',
  },
  {
    label: 'Creative',
    icon: '',
    instructions:
      'Be creative and expressive. Use vivid language, analogies, and metaphors. Think outside the box. Offer unique perspectives and imaginative solutions.',
  },
];

const MAX_CHARS = 2000;

export default function SettingsPanel() {
  const { customInstructions, userName, setCustomInstructions, setUserName } = useStore();

  const [localName, setLocalName] = useState(userName);
  const [localInstructions, setLocalInstructions] = useState(customInstructions);
  const [saved, setSaved] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
    };
  }, []);

  // Sync from store when it loads from storage (async)
  useEffect(() => {
    setLocalName(userName);
  }, [userName]);

  useEffect(() => {
    setLocalInstructions(customInstructions);
  }, [customInstructions]);

  // Detect which preset matches (if any)
  useEffect(() => {
    const match = STYLE_PRESETS.find((p) => p.instructions === localInstructions);
    setActivePreset(match ? match.label : null);
  }, [localInstructions]);

  const handleSave = () => {
    setUserName(localName.trim());
    setCustomInstructions(localInstructions.trim());
    setSaved(true);
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
    savedTimerRef.current = setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setLocalName('');
    setLocalInstructions('');
    setUserName('');
    setCustomInstructions('');
    setActivePreset(null);
  };

  const handlePreset = (preset: (typeof STYLE_PRESETS)[number]) => {
    if (activePreset === preset.label) {
      // Toggle off
      setLocalInstructions('');
      setActivePreset(null);
    } else {
      setLocalInstructions(preset.instructions);
      setActivePreset(preset.label);
    }
  };

  const charCount = localInstructions.length;
  const isDirty = localName !== userName || localInstructions !== customInstructions;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 py-4 panel-scroll-root" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

        {/* Section 1: Your Profile */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <User size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              YOUR PROFILE
            </h3>
          </div>
          <label
            htmlFor="settings-name"
            style={{ display: 'block', fontSize: 11, color: 'var(--fg2)', marginBottom: 6 }}
          >
            Name
          </label>
          <input
            id="settings-name"
            type="text"
            value={localName}
            onChange={(e) => setLocalName(e.target.value)}
            placeholder="Your name"
            maxLength={100}
            style={{
              width: '100%',
              padding: '8px 12px',
              fontSize: 13,
              borderRadius: 8,
              border: '1px solid var(--b2)',
              background: 'var(--glass)',
              color: 'var(--fg)',
              outline: 'none',
              fontFamily: 'inherit',
              transition: 'border-color 0.2s',
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
            onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
          />
          <p style={{ margin: '6px 0 0', fontSize: 10, color: 'var(--fg3)' }}>
            AURA will use your name to personalize responses.
          </p>
        </section>

        {/* Section 2: Custom Instructions */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <FileText size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              CUSTOM INSTRUCTIONS
            </h3>
          </div>
          <div style={{ position: 'relative' }}>
            <textarea
              value={localInstructions}
              onChange={(e) => {
                if (e.target.value.length <= MAX_CHARS) {
                  setLocalInstructions(e.target.value);
                }
              }}
              placeholder="Tell AURA how to respond. Example: 'Always be concise. Use bullet points. Explain like I'm an expert.'"
              rows={6}
              style={{
                width: '100%',
                padding: '10px 12px',
                paddingBottom: 28,
                fontSize: 12.5,
                lineHeight: 1.5,
                borderRadius: 8,
                border: '1px solid var(--b2)',
                background: 'var(--glass)',
                color: 'var(--fg)',
                outline: 'none',
                resize: 'vertical',
                fontFamily: 'inherit',
                minHeight: 120,
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--pl)')}
              onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--b2)')}
            />
            <span
              style={{
                position: 'absolute',
                bottom: 8,
                right: 12,
                fontSize: 10,
                color: charCount > MAX_CHARS * 0.9 ? 'var(--err, #f87171)' : 'var(--fg3)',
                pointerEvents: 'none',
              }}
            >
              {charCount}/{MAX_CHARS}
            </span>
          </div>
          <p style={{ margin: '6px 0 0', fontSize: 10, color: 'var(--fg3)' }}>
            These instructions are prepended to every message you send.
          </p>
        </section>

        {/* Section 3: Response Style Presets */}
        <section>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <Sparkles size={16} style={{ color: 'var(--pl)' }} />
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600, color: 'var(--fg)', letterSpacing: '0.03em' }}>
              RESPONSE STYLE
            </h3>
          </div>
          <p style={{ margin: '0 0 10px', fontSize: 11, color: 'var(--fg2)' }}>
            Quick presets — click to fill the instructions above.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {STYLE_PRESETS.map((preset) => {
              const isActive = activePreset === preset.label;
              return (
                <button
                  key={preset.label}
                  onClick={() => handlePreset(preset)}
                  style={{
                    padding: '6px 14px',
                    fontSize: 12,
                    fontWeight: 500,
                    borderRadius: 20,
                    border: isActive ? '1px solid var(--pl)' : '1px solid var(--b2)',
                    background: isActive ? 'rgba(124, 58, 237, 0.15)' : 'var(--glass)',
                    color: isActive ? 'var(--pl)' : 'var(--fg)',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    transition: 'all 0.2s',
                    outline: 'none',
                  }}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </section>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 10, paddingTop: 4, paddingBottom: 16 }}>
          <button
            onClick={handleSave}
            disabled={!isDirty && !saved}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              padding: '10px 0',
              fontSize: 13,
              fontWeight: 600,
              borderRadius: 10,
              border: 'none',
              background: saved
                ? 'rgba(52, 211, 153, 0.2)'
                : isDirty
                  ? 'linear-gradient(135deg, var(--p), var(--pl))'
                  : 'var(--glass)',
              color: saved ? '#34d399' : isDirty ? 'white' : 'var(--fg3)',
              cursor: isDirty ? 'pointer' : 'default',
              fontFamily: 'inherit',
              transition: 'all 0.25s',
              opacity: !isDirty && !saved ? 0.5 : 1,
            }}
          >
            {saved ? <Check size={15} /> : <Save size={15} />}
            {saved ? 'Saved' : 'Save'}
          </button>
          <button
            onClick={handleReset}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              padding: '10px 18px',
              fontSize: 13,
              fontWeight: 500,
              borderRadius: 10,
              border: '1px solid var(--b2)',
              background: 'transparent',
              color: 'var(--fg2)',
              cursor: 'pointer',
              fontFamily: 'inherit',
              transition: 'all 0.2s',
            }}
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
