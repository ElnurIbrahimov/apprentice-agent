import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';
import { connectWS, fetchStatus, startProactivePoll } from './ws';
import { initBackendUrl, API_KEY } from './api';
import ext from './ext';
import { useStore, storeHydrated } from './store';
import { initShortcuts } from './shortcuts';

// Init — load saved backend URL + hydrate store prefs before connecting.
// Awaiting storeHydrated prevents the first-frame mismatch (default theme /
// default prefs / empty model list) that used to appear while chrome.storage
// callbacks resolved in parallel with the first fetchStatus() call.
async function init() {
  ext?.runtime?.sendMessage({ type: 'SIDEBAR_READY' });
  window.addEventListener('beforeunload', () => {
    ext?.runtime?.sendMessage({ type: 'SIDEBAR_CLOSED' });
  });
  await Promise.all([initBackendUrl(), storeHydrated]);
  if (!API_KEY) {
    useStore.getState().addProactiveMessage({
      id: 'first-run-api-key',
      text: 'Paste your Aura API key in Settings → Connection to start chatting.',
      timestamp: Date.now(),
    });
    useStore.getState().setPanel('settings');
  }
  fetchStatus();
  setInterval(fetchStatus, 30_000);
  connectWS();
  startProactivePoll();
  initShortcuts(useStore);
}
init();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
