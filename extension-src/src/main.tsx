import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';
import { connectWS, fetchStatus } from './ws';
import ext from './ext';
import { useStore } from './store';
import { initShortcuts } from './shortcuts';

// Init
ext?.runtime?.sendMessage({ type: 'SIDEBAR_READY' });
fetchStatus();
setInterval(fetchStatus, 30_000);
connectWS();
initShortcuts(useStore);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
