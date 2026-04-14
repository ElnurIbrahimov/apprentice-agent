import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';
import { connectWS, fetchStatus, startProactivePoll } from './ws';
import { initBackendUrl, API_KEY } from './api';
import ext from './ext';
import { useStore } from './store';
import { initShortcuts } from './shortcuts';

// Init — load saved backend URL before connecting
async function init() {
  ext?.runtime?.sendMessage({ type: 'SIDEBAR_READY' });
  window.addEventListener('beforeunload', () => {
    ext?.runtime?.sendMessage({ type: 'SIDEBAR_CLOSED' });
  });
  await initBackendUrl();
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
