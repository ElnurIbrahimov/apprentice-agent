# Agent 3 Audit — Media Panels
Date: 2026-03-07
Files audited: sidebar.js, sidebar.html, background.js, content.js, api/routes/youtube.py, api/routes/pdf.py, api/routes/transcribe.py, api/routes/ocr.py, api/routes/image_gen.py

---

## YOUTUBE PANEL

### 1. loadYoutubeSummary() — PASS
Function exists at sidebar.js line 1852. Posts to `POST /api/youtube/summarize` with `{url}` body. 90-second timeout set via `AbortSignal.timeout(90000)`.

### 2. Event listeners — PASS
- `yt-summarize-btn`: click listener at line 1916, calls `loadYoutubeSummary($('yt-url-inp').value)`
- `yt-url-inp`: keydown Enter listener at line 1917
- `yt-auto-btn`: click listener at line 1918 — fills URL input and calls `loadYoutubeSummary(ytAutoUrl)`
- `yt-snippet-toggle`: click listener at line 1919

### 3. YT_TAB_DETECTED banner — PASS
Message listener at line 742-748 handles `YT_TAB_DETECTED`: sets `ytAutoUrl`, sets banner title text, sets `banner.style.display = 'block'`. When the youtube panel is active, also fills the URL input.

### 4. background.js sends YT_TAB_DETECTED — PASS
`ext.tabs.onUpdated` listener at line 65-71 checks `url.includes('youtube.com/watch')` and sends `YT_TAB_DETECTED` with url and title.

### 5. Result elements populated — PASS
`loadYoutubeSummary` populates: `yt-res-title`, `yt-res-channel`, `yt-res-duration`, `yt-res-summary`, `yt-res-points` (as `<li class="yt-kp">` elements), `yt-res-snippet`. All IDs exist in sidebar.html.

### 6. Transcript snippet toggle — PASS
`yt-snippet-toggle` click toggles `yt-res-snippet` display between `'none'` and `'block'`, and rotates `yt-snippet-chevron` 90deg. After load, snippet is hidden by default with chevron reset.

**YOUTUBE PANEL: ALL CHECKS PASS. No bugs found.**

---

## PDF PANEL

### 1. pdf-upload-btn triggers file input — PASS
Line 981: `$('pdf-upload-btn').addEventListener('click', () => $('pdf-file').click())`. `pdf-file` input is `display:none` in HTML with `accept=".pdf"`.

### 2. pdf-file uploads to POST /api/pdf/extract — PASS
`pdf-file` change listener at line 983 calls `uploadPdf(file)`. `uploadPdf` posts `FormData` to `${HTTP}/api/pdf/extract`. On success, sets `pdfCtx` with `{text, page_count, word_count}` and reveals `pdf-inp-area`.

### 3. PDF_TAB_DETECTED auto banner — PASS
Message listener at line 749-758 handles `PDF_TAB_DETECTED`: adds `.on` class to `pdf-auto` (which CSS shows as `display:block`), sets `pdf-auto-title` text, wires `pdf-auto-btn.onclick` to `loadPdfUrl(msg.url)`.

### 4. PDF context prepended to WS message — PASS
`sendPdfQuestion()` at line 1030 builds `contextPrefix = '[PDF Context — N pages]\n${pdfCtx.text.slice(0,35000)}\n\n---\nQuestion: '` and sends `contextPrefix + q` as the WS chat message.

### 5. Loading states — PASS
- Upload: `pdf-status` shows "Extracting text…", `pdf-inp-area` hidden during upload, re-shown on success.
- Question: answer bubble shows dots animation while streaming, cleared on first chunk.

**PDF PANEL: ALL CHECKS PASS. No bugs found.**

---

## VOICE NOTES PANEL

### 1. SpeechRecognition detection — PASS
Lines 1080-1087: checks `window.SpeechRecognition || window.webkitSpeechRecognition`. If absent, shows `rec-ff-note` and disables/dims `rec-controls`. If present, hides the Firefox note.

### 2. rec-btn start/stop recording — PASS
Click listener at line 1137-1140: if `recRecognition` is active, calls `stopRec()`; else calls `startRec()`. `startRec` creates and starts a `SpeechRecognitionAPI` instance with `continuous=true, interimResults=true`. `stopRec` calls `.stop()` and nulls the reference.

### 3. Timer updates every second — PASS
`startRec` sets `setInterval` at 1000ms, incrementing `recSeconds` and displaying as `M:SS` in `rec-timer`. `stopRec` calls `clearInterval(recTimerInterval)`.

### 4. "Summarize as Notes" streams via WebSocket — PASS
`rec-summarize` click handler sends WS message with type `'chat'` and a prompt wrapping `fullTranscript`. Results stream into `rec-notes` element via `activeStream` with type `'write'`.

**BUG FIXED (sidebar.js line 1203):** `submitBtn` was `null`, meaning `rec-summarize` button was never disabled during streaming (no visual loading state) and `finalizeStream` would not re-enable it after stream ended. Fixed by setting `submitBtn: $('rec-summarize')` and adding `$('rec-summarize').disabled = true` before the WS send.

### 5. "Save transcript" posts to /api/knowledge/save — PASS
`rec-save-wb` click handler at line 1215 posts `{text, title, source_type: 'voice_note'}` to `${HTTP}/api/knowledge/save`. Shows "✓ Saved to Wisebase" on success.

### 6. Whisper upload via POST /api/transcribe — PASS
`rec-whisper-btn` click handler at line 1186 reads selected audio file, posts FormData to `${HTTP}/api/transcribe`, populates `fullTranscript` and `rec-transcript` on success.

**VOICE NOTES PANEL: 1 BUG FIXED.**

---

## OCR PANEL

### 1. ocr-capture sends OCR_START to background — PASS
Line 1209: `$('ocr-capture').addEventListener('click', () => { ext.runtime.sendMessage({ type: 'OCR_START' }); ... })`. Sets `ocr-result` to hint text and hides `ocr-actions`.

### 2. Background OCR_START flow — PASS
`background.js` lines 208-250 handle `OCR_START`:
1. Calls `chrome.tabs.captureVisibleTab` to get PNG dataUrl
2. Sends `SHOW_OCR_OVERLAY` to content script with dataUrl
3. Content script shows fullscreen overlay with crosshair selection canvas
4. On selection mouseup, returns `{ok, x, y, w, h, dpr}` region
5. Background crops using `OffscreenCanvas` and `createImageBitmap`
6. Posts base64-encoded crop to `${BACKEND}/api/ocr` as `{image_b64}`
7. Sends `OCR_RESULT` message back to sidebar

`content.js` `showOcrOverlay` handles the drag-select overlay correctly, supports Escape to cancel.

### 3. OCR_RESULT populates ocr-result — PASS
Message listener at line 759-767: on `OCR_RESULT`, if `msg.text` exists, sets `lastOcrText = msg.text`, sets `ocr-result.textContent = msg.text`, shows `ocr-actions`. If error (non-empty and not "Cancelled"), shows error text in `ocr-result`.

### 4. Action buttons — PASS
- **Send to Chat**: sets `pendingCtx`, calls `showCtx`, switches to chat panel.
- **Translate**: fills `tr-inp` with OCR text, switches to translate panel.
- **Copy**: `navigator.clipboard.writeText(lastOcrText)`, shows "✓ Copied" feedback.

**OCR PANEL: ALL CHECKS PASS. No bugs found.**

---

## IMAGE GENERATOR PANEL

### 1. Style pills toggle imgStyle — PASS
Lines 1356-1362: `querySelectorAll('.img-style')` forEach adds click listener. On click, removes `.on` from all, adds `.on` to clicked, sets `imgStyle = this.dataset.style`. HTML has Default (`data-style=""`), Photo, Anime, Abstract. Default starts with `.on` class; `imgStyle` initializes to `''` matching the Default `data-style`.

### 2. Generate button POSTs to /api/image/generate — PASS
Lines 1364-1404: `img-gen` click handler builds `fullPrompt = imgStyle ? prompt + ', ' + imgStyle : prompt`, POSTs `{prompt: fullPrompt, negative_prompt, steps: 20}` to `${HTTP}/api/image/generate`.

### 3. 503 shows img-comfy-note — PASS
After `await res.json()`, if `res.status === 503`, clears status text and sets `img-comfy-note.style.display = ''`. Note: FastAPI serializes `HTTPException(503, {"error":..., "install":...})` as `{"detail": {...}}`. The sidebar correctly checks `res.status === 503` rather than inspecting the body, so this works regardless of detail format.

### 4. Success shows img-out and download button — PASS
If `data.image_b64` is present: sets `img.src = 'data:image/png;base64,' + data.image_b64`, `img.style.display = 'block'`, shows `img-acts`, sets status to "✓ Generated". Download button creates an `<a>` with `href=img.src, download='aura-image.png'` and clicks it.

**BUG FIXED (image_gen.py):** The route used the synchronous `requests` library inside an `async def` FastAPI handler. Every image generation blocked FastAPI's entire event loop for up to 120 seconds (the polling timeout). All concurrent API requests (chat, search, etc.) would stall during image generation.

Fixed by replacing all `requests` calls with `httpx.AsyncClient` async calls. The 120-iteration polling loop now uses `await c.get(...)` instead of blocking `req.get(...)`, keeping the event loop free.

**IMAGE GENERATOR PANEL: 1 BUG FIXED.**

---

## SUMMARY OF BUGS FOUND AND FIXED

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `api/routes/image_gen.py` | Synchronous `requests` library used inside `async def generate_image` — blocks FastAPI event loop for up to 120s during image generation polling, stalling all other API requests | Replaced all `req.get`/`req.post` calls with `httpx.AsyncClient` async equivalents |
| 2 | `extension/sidebar.js` | `rec-summarize` button had `submitBtn: null` in its `activeStream` config — button showed no disabled state during streaming and `finalizeStream` would not re-enable it after the stream ended | Changed `submitBtn` to `$('rec-summarize')` and added `$('rec-summarize').disabled = true` before the WS send |

## ALL PANELS STATUS

| Panel | Status |
|-------|--------|
| YouTube | PASS — all event listeners, auto-detect, result population, and snippet toggle verified |
| PDF Chat | PASS — upload, URL load, auto-detect banner, context-prepend, loading states all correct |
| Voice Notes | FIXED — 1 bug fixed (summarize button loading state); all other features pass |
| OCR | PASS — capture flow, background crop, result display, and action buttons all correct |
| Image Generator | FIXED — 1 bug fixed (blocking event loop); style pills, 503 handling, success display all correct |
