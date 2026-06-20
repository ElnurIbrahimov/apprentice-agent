import { useState, useEffect } from 'react';
import { apiFetch } from '../utils/apiFetch';
import { LoginPage } from './LoginPage';
import { OnboardingFlow } from './OnboardingFlow';
import { useSettingsStore } from '../store/settingsStore';

type AuthState =
  | { status: 'checking' }
  | { status: 'anonymous'; configured: boolean }
  | { status: 'authenticated'; username: string };

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [auth, setAuth] = useState<AuthState>({ status: 'checking' });
  const onboardingDone = useSettingsStore((s) => s.settings.onboardingDone);
  const updateSettings = useSettingsStore((s) => s.updateSettings);

  // Probe current session on mount. If the server doesn't have cookie-auth
  // configured (`configured === false`), treat as authenticated so behavior
  // matches the pre-login world — useful for local dev / older deployments.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiFetch('/api/auth/web/me', { credentials: 'same-origin' });
        if (!res.ok) throw new Error(`status=${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        if (data.authenticated) {
          setAuth({ status: 'authenticated', username: data.username || '' });
        } else if (data.configured === false) {
          // Server isn't set up for cookie auth — let the app through.
          setAuth({ status: 'authenticated', username: '' });
        } else {
          setAuth({ status: 'anonymous', configured: true });
        }
      } catch {
        // /me endpoint unreachable — show the login page rather than silently
        // authenticating. Old servers that don't run cookie auth return
        // `configured: false` above, so hitting this branch means the server
        // is actually down or returned a non-JSON error.
        if (!cancelled) setAuth({ status: 'anonymous', configured: true });
      }
    })();
    return () => { cancelled = true; };
  }, []);

  if (auth.status === 'checking') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-400">
        <div className="animate-pulse">Loading…</div>
      </div>
    );
  }

  if (auth.status === 'anonymous') {
    return <LoginPage onLoggedIn={(u) => setAuth({ status: 'authenticated', username: u })} />;
  }

  if (!onboardingDone) {
    return (
      <OnboardingFlow
        onComplete={() => updateSettings({ onboardingDone: true })}
      />
    );
  }

  return <>{children}</>;
}
