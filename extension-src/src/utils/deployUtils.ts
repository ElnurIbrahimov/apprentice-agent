/**
 * deployUtils — share, deploy, and export helpers for Aura projects.
 */

import JSZip from 'jszip';
import { HTTP, getAuthHeaders } from '../api';
import ext from '../ext';

// ── Types ──

export interface ShareResult {
  url: string;
  id: string;
  projectName: string;
  fileCount: number;
  expiresAt: number;
}

export interface ShareInfo {
  id: string;
  project_name: string;
  entry_point: string;
  file_count: number;
  total_bytes: number;
  created_at: number;
  expires_at: number;
  url: string;
}

export interface Deployment {
  id: string;
  projectName: string;
  platform: 'aura' | 'github-pages';
  url: string;
  status: 'live' | 'building' | 'failed' | 'expired';
  createdAt: number;
  expiresAt?: number;
}

const DEPLOYMENTS_KEY = 'aura_deployments';
const MAX_DEPLOYMENTS = 50;

// ── Share via Aura backend ──

export async function shareProject(
  projectName: string,
  files: Record<string, string>,
  entryPoint: string = 'index.html',
  expiresDays: number = 7,
): Promise<ShareResult> {
  const resp = await fetch(`${HTTP}/api/share`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({
      project_name: projectName,
      files,
      entry_point: entryPoint,
      expires_days: expiresDays,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
    throw new Error(err.detail || `Share failed: ${resp.status}`);
  }
  const data = await resp.json();
  return {
    url: data.url,
    id: data.id,
    projectName: data.project_name,
    fileCount: data.file_count,
    expiresAt: data.expires_at,
  };
}

export async function listShares(): Promise<ShareInfo[]> {
  const resp = await fetch(`${HTTP}/api/shares`, {
    headers: getAuthHeaders(),
  });
  if (!resp.ok) return [];
  return resp.json();
}

export async function deleteShare(shareId: string): Promise<void> {
  await fetch(`${HTTP}/api/shares/${shareId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
}

// ── ZIP export ──

export async function downloadAsZip(
  projectName: string,
  files: Record<string, string>,
): Promise<void> {
  const zip = new JSZip();
  for (const [path, content] of Object.entries(files)) {
    zip.file(path, content);
  }
  const blob = await zip.generateAsync({ type: 'blob' });
  downloadBlob(blob, `${sanitizeFilename(projectName)}.zip`);
}

// ── GitHub Pages deploy ──

export async function deployToGitHubPages(
  repoName: string,
  files: Record<string, string>,
  token: string,
  isPrivate: boolean = false,
): Promise<string> {
  const headers = {
    Authorization: `token ${token}`,
    Accept: 'application/vnd.github.v3+json',
    'Content-Type': 'application/json',
  };

  // 1. Get authenticated user
  const userResp = await fetch('https://api.github.com/user', { headers });
  if (!userResp.ok) throw new Error('Invalid GitHub token');
  const user = await userResp.json();
  const owner = user.login;

  // 2. Create or get repo
  const safeRepo = sanitizeFilename(repoName).toLowerCase();
  let repoExists = false;
  const checkResp = await fetch(`https://api.github.com/repos/${owner}/${safeRepo}`, { headers });
  if (checkResp.ok) {
    repoExists = true;
  } else {
    const createResp = await fetch('https://api.github.com/user/repos', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        name: safeRepo,
        private: isPrivate,
        auto_init: true,
        description: 'Deployed from Aura',
      }),
    });
    if (!createResp.ok) {
      const err = await createResp.json().catch(() => ({}));
      throw new Error(err.message || 'Failed to create repo');
    }
    // Wait for repo initialization
    await new Promise(r => setTimeout(r, 2000));
  }

  // 3. Push files using Contents API
  for (const [path, content] of Object.entries(files)) {
    const encoded = btoa(unescape(encodeURIComponent(content)));
    const fileUrl = `https://api.github.com/repos/${owner}/${safeRepo}/contents/${path}`;

    // Check if file exists (need SHA for update)
    let sha: string | undefined;
    if (repoExists) {
      const existResp = await fetch(fileUrl, { headers });
      if (existResp.ok) {
        const existData = await existResp.json();
        sha = existData.sha;
      }
    }

    const putResp = await fetch(fileUrl, {
      method: 'PUT',
      headers,
      body: JSON.stringify({
        message: `Deploy ${path} from Aura`,
        content: encoded,
        ...(sha ? { sha } : {}),
      }),
    });
    if (!putResp.ok) {
      const err = await putResp.json().catch(() => ({}));
      throw new Error(`Failed to push ${path}: ${err.message || putResp.status}`);
    }
  }

  // 4. Enable GitHub Pages
  await fetch(`https://api.github.com/repos/${owner}/${safeRepo}/pages`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      source: { branch: 'main', path: '/' },
    }),
  }); // Ignore errors — Pages may already be enabled

  return `https://${owner}.github.io/${safeRepo}/`;
}

// ── Deployment tracking (chrome.storage.local) ──

export async function saveDeployment(deployment: Deployment): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.get([DEPLOYMENTS_KEY], (data: any) => {
      const list: Deployment[] = data[DEPLOYMENTS_KEY] || [];
      list.unshift(deployment);
      const trimmed = list.slice(0, MAX_DEPLOYMENTS);
      ext.storage.local.set({ [DEPLOYMENTS_KEY]: trimmed }, resolve);
    });
  });
}

export async function loadDeployments(): Promise<Deployment[]> {
  if (!ext?.storage?.local) return [];
  return new Promise(resolve => {
    ext.storage.local.get([DEPLOYMENTS_KEY], (data: any) => {
      resolve(data[DEPLOYMENTS_KEY] || []);
    });
  });
}

export async function removeDeployment(id: string): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.get([DEPLOYMENTS_KEY], (data: any) => {
      const list: Deployment[] = (data[DEPLOYMENTS_KEY] || []).filter((d: Deployment) => d.id !== id);
      ext.storage.local.set({ [DEPLOYMENTS_KEY]: list }, resolve);
    });
  });
}

// ── GitHub token storage ──

const GH_TOKEN_KEY = 'aura_github_token';

export async function saveGitHubToken(token: string): Promise<void> {
  if (!ext?.storage?.local) return;
  return new Promise(resolve => {
    ext.storage.local.set({ [GH_TOKEN_KEY]: token }, resolve);
  });
}

export async function loadGitHubToken(): Promise<string> {
  if (!ext?.storage?.local) return '';
  return new Promise(resolve => {
    ext.storage.local.get([GH_TOKEN_KEY], (data: any) => {
      resolve(data[GH_TOKEN_KEY] || '');
    });
  });
}

// ── Helpers ──

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9_\-. ]/g, '').replace(/\s+/g, '-').slice(0, 100) || 'project';
}
