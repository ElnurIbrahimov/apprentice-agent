import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';
import { connectWS, fetchStatus, startProactivePoll } from './ws';
import { initBackendUrl } from './api';
import ext from './ext';
import { useStore } from './store';
import { initShortcuts } from './shortcuts';

// Init — load saved backend URL before connecting
async function init() {
  ext?.runtime?.sendMessage({ type: 'SIDEBAR_READY' });
  await initBackendUrl();
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
