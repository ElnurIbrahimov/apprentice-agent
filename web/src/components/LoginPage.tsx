import { useState, FormEvent, useEffect } from 'react';
import { apiFetch } from '../utils/apiFetch';

interface LoginPageProps {
  onLoggedIn: (username: string) => void;
}

/**
 * Single-user login gate for the Aura web UI.
 *
 * Posts credentials to `/api/auth/web/login`. On 200, the server sets
 * an HTTP-only `aura_session` cookie; we simply flip the app's auth
 * state via `onLoggedIn`. No token to manage client-side.
 */
export function LoginPage({ onLoggedIn }: LoginPageProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Auto-focus the username field on mount.
  useEffect(() => {
    const el = document.getElementById('aura-login-username');
    if (el) (el as HTMLInputElement).focus();
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await apiFetch('/api/auth/web/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ username, password }),
      });
      if (res.ok) {
        const data = await res.json().catch(() => ({ username }));
        onLoggedIn(data.username ?? username);
        return;
      }
      if (res.status === 401) {
        setError('Invalid username or password.');
      } else if (res.status === 429) {
        setError('Too many login attempts. Try again in a few minutes.');
      } else if (res.status === 503) {
        setError('Login not configured on the server. Set AURA_WEB_USERNAME / PASSWORD_HASH / SESSION_SECRET in .env.');
      } else {
        setError(`Login failed (${res.status}).`);
      }
    } catch (err: any) {
      setError(err?.message || 'Network error — could not reach the server.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-950 via-slate-950 to-black p-4">
      <div className="w-full max-w-sm bg-slate-900/60 backdrop-blur border border-purple-800/40 rounded-2xl shadow-2xl p-8">
        <div className="flex items-center justify-center mb-6">
          <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white text-xl font-bold">
            A
          </div>
        </div>
        <h1 className="text-2xl font-semibold text-white text-center mb-1">AURA</h1>
        <p className="text-sm text-slate-400 text-center mb-6">Sign in to continue</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="aura-login-username" className="block text-xs font-medium text-slate-300 mb-1">
              Username
            </label>
            <input
              id="aura-login-username"
              type="text"
              autoComplete="username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800/80 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              placeholder="Elikos"
            />
          </div>
          <div>
            <label htmlFor="aura-login-password" className="block text-xs font-medium text-slate-300 mb-1">
              Password
            </label>
            <input
              id="aura-login-password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800/80 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>

          {error && (
            <div className="text-sm text-red-300 bg-red-900/30 border border-red-800/50 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !username || !password}
            className="w-full py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {submitting ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="text-xs text-slate-500 text-center mt-6">
          Personal build · your credentials stay on your own server.
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
