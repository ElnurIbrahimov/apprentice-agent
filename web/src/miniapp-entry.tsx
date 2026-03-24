import React from 'react';
import { createRoot } from 'react-dom/client';
import MiniApp from './MiniApp';
import './miniapp.css';

createRoot(document.getElementById('miniapp-root')!).render(
  <React.StrictMode>
    <MiniApp />
  </React.StrictMode>
);
