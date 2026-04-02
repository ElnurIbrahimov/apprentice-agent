# Phase 6: Deploy & Share

**Effort:** 2-3 days
**Impact:** Projects go from "local preview" to "live on the internet"
**Depends on:** Phase 3 (multi-file) or works with single-file too

---

## The Problem

Currently, export options are:
- Download as `.html` file
- Copy code to clipboard
- Copy as data URL
- Open in CodeSandbox (form submission)
- Open in StackBlitz (form submission)

None of these give the user a **live URL** they can share. Every competitor (v0, Bolt, Lovable, Replit) provides instant deployment.

---

## 6A. Shareable Links via Aura Backend (Simplest)

### How It Works
Upload project files to Aura's own backend → serve as static files → return a shareable URL.

### Backend Endpoint

**New file:** `api/routes/share.py`

```python
@router.post("/api/share")
async def share_project(request: ShareRequest) -> ShareResponse:
    """
    Upload project files and get a shareable URL.
    
    Request:
    {
      "project_name": "my-portfolio",
      "files": {
        "index.html": "<html>...</html>",
        "styles/main.css": "body { ... }",
        "scripts/app.js": "console.log('hello')"
      },
      "entry_point": "index.html"
    }
    
    Response:
    {
      "url": "https://aura-elnur.duckdns.org/shared/abc123/",
      "id": "abc123",
      "expires_at": "2026-04-09T00:00:00Z"  // 7 days default
    }
    """
    share_id = generate_short_id()  # 8-char alphanumeric
    share_dir = SHARED_DIR / share_id
    
    # Write files to disk
    for path, content in request.files.items():
        file_path = share_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    # Register in SQLite with expiry
    db.insert_share(share_id, request.project_name, expires_days=7)
    
    return ShareResponse(
        url=f"{BASE_URL}/shared/{share_id}/",
        id=share_id,
        expires_at=...
    )

@router.get("/shared/{share_id}/{path:path}")
async def serve_shared(share_id: str, path: str = "index.html"):
    """Serve shared project files as static content."""
    file_path = SHARED_DIR / share_id / path
    if not file_path.exists():
        raise HTTPException(404)
    return FileResponse(file_path)
```

### Storage Management
- Shared files stored in `data/shared/{id}/`
- SQLite table tracks: id, name, created_at, expires_at, file_count, total_size
- Cron job or startup cleanup: delete expired shares
- Limits: 5MB per project, 20 active shares max, 7-day default expiry (configurable)

### Extension UI
- "Share" button in toolbar (cloud + link icon)
- On click: uploads all files from VirtualFS → receives URL → copies to clipboard
- Shows the URL in a toast with "Copy" and "Open" buttons
- "Manage Shares" section in settings: list active shares, delete, extend expiry

---

## 6B. GitHub Pages Deploy

### How It Works
Create a GitHub repo → push project files → enable GitHub Pages → return the URL.

### Prerequisites
- User provides a GitHub token (stored encrypted in chrome.storage.local)
- Or: OAuth flow via GitHub App (more setup but better UX)

### Flow
```
1. User clicks "Deploy to GitHub Pages"
2. Prompt for repo name (default: project name)
3. POST to GitHub API:
   a. Create repo (or use existing)
   b. Create/update files via Contents API or Git Data API
   c. Enable GitHub Pages on the main branch
4. Return URL: https://{username}.github.io/{repo-name}/
```

### Backend Endpoint

```python
@router.post("/api/deploy/github-pages")
async def deploy_github_pages(request: GHPagesRequest):
    """
    Request:
    {
      "repo_name": "my-portfolio",
      "files": { ... },
      "github_token": "ghp_...",      // or from stored settings
      "private": false
    }
    """
    # Use PyGithub or httpx to:
    # 1. Create repo
    # 2. Push files
    # 3. Enable Pages
    # 4. Return URL
```

### Extension UI
- "Deploy" dropdown → "GitHub Pages"
- First time: prompt for GitHub token (with link to token creation page)
- Shows deployment progress: Creating repo... Pushing files... Enabling Pages...
- Returns live URL

---

## 6C. Vercel Deploy (Optional)

### How It Works
Use Vercel's deployment API to create a serverless deployment.

### Flow
```
POST https://api.vercel.com/v13/deployments
Authorization: Bearer {vercel_token}
Body: { files: [...], projectSettings: { framework: "..." } }
```

### Benefits over GitHub Pages
- Instant deploys (seconds vs. minutes)
- Server-side rendering support (Next.js, etc.)
- Preview URLs for each deploy
- Custom domains

### Prerequisites
- Vercel token (stored in settings)
- For Next.js/server projects: needs the build output, not just source files

### Extension UI
- "Deploy" dropdown → "Vercel"
- First time: prompt for Vercel token
- Shows: Deploying... → Live at https://my-project-abc.vercel.app

---

## 6D. Export Improvements

Upgrade existing export options:

### Download as ZIP (for multi-file projects)
```typescript
import JSZip from 'jszip';

async function downloadAsZip(project: VirtualProject) {
  const zip = new JSZip();
  for (const [path, file] of project.files) {
    zip.file(path, file.content);
  }
  // Add package.json if React/Node project
  const blob = await zip.generateAsync({ type: 'blob' });
  downloadBlob(blob, `${project.name}.zip`);
}
```

### Open in StackBlitz (multi-file)
```typescript
// StackBlitz SDK supports multi-file projects
import sdk from '@stackblitz/sdk';

sdk.openProject({
  title: project.name,
  files: Object.fromEntries(
    Array.from(project.files).map(([path, file]) => [path, file.content])
  ),
  template: 'node',  // or 'html'
});
```

### Open in CodeSandbox (multi-file)
```typescript
// CodeSandbox define API
const parameters = getParameters({
  files: Object.fromEntries(
    Array.from(project.files).map(([path, file]) => [
      path, { content: file.content }
    ])
  ),
});
```

---

## 6E. Deploy Status Dashboard

Track all deployments in one place.

**New component:** `extension-src/src/components/DeployDashboard.tsx`

```typescript
interface Deployment {
  id: string;
  projectName: string;
  platform: 'aura' | 'github-pages' | 'vercel';
  url: string;
  status: 'live' | 'building' | 'failed' | 'expired';
  createdAt: number;
  expiresAt?: number;
}
```

UI:
- List of all deployments with status badges
- Quick actions: Open URL, Copy URL, Redeploy, Delete
- Filter by platform
- Accessible from WebCreator toolbar and Settings panel

---

## Definition of Done — Phase 6
- [ ] "Share" button uploads project to Aura backend and returns a live URL
- [ ] Shared projects serve correctly (all file types, correct MIME types)
- [ ] Shares expire after 7 days with auto-cleanup
- [ ] URL is copyable with one click
- [ ] GitHub Pages deploy works: creates repo, pushes files, enables Pages
- [ ] GitHub token stored securely in extension storage
- [ ] Download as ZIP works for multi-file projects
- [ ] StackBlitz and CodeSandbox open multi-file projects correctly
- [ ] Deploy status dashboard lists all active deployments
- [ ] Vercel deploy (stretch goal) works for static and Next.js projects
