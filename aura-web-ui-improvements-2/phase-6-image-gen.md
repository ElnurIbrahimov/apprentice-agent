# Phase 6: Image Generation + Media Panel

**Effort:** 2 days
**Impact:** Unlocks the image/video AI providers already configured in Settings

---

## The Problem

Settings has 6 image providers (DALL-E, Stable Diffusion, Midjourney, Flux, etc.) and 5 video providers configured, but there's no UI to actually generate images or video. The API keys are collected but never used from the web UI.

---

## 6A. Image Generation Panel

### Layout
New "Create" tab (or sub-panel in Tools):

```
┌────────────────────────────────────────────────────┐
│ Image Generation                                     │
├──────────────────────┬─────────────────────────────┤
│ Prompt input         │ Generated image(s)           │
│ [Describe image...]  │                              │
│                      │ ┌──────────────────────┐    │
│ Settings:            │ │                      │    │
│ • Model: [dropdown]  │ │    Preview Image     │    │
│ • Size: [selector]   │ │                      │    │
│ • Style: [selector]  │ └──────────────────────┘    │
│ • Negative prompt    │                              │
│                      │ [Download] [Copy] [Variations]│
│ [Generate]           │                              │
│                      │ History (thumbnails grid)     │
│ Recent prompts       │                              │
└──────────────────────┴─────────────────────────────┘
```

### Backend Endpoint
Check what image generation endpoints exist:

The extension has `ImagePanel` — search for the API it uses. Likely:
- `POST /api/image/generate` with `{ prompt, model, size, style, negative_prompt }`
- Or routes through the provider system

### Component: `web/src/components/ImageGenPanel.tsx`

**Features:**
- Text prompt with "Generate" button
- Model picker: lists available image models from `/api/models?type=image` or provider list
- Size presets: 1024x1024, 1024x1792, 1792x1024, 512x512
- Style presets: Natural, Vivid, Anime, Photographic, Digital Art, etc. (provider-dependent)
- Negative prompt (optional, collapsible)
- Generation progress indicator
- Generated image display with:
  - Download (PNG/JPEG)
  - Copy to clipboard
  - "Use as reference" (sends to chat as attachment)
  - "Generate variations" (sends back with variation prompt)
  - "Edit" (inpainting if supported by model)
- Generation history: grid of thumbnails (stored in localStorage, last 50)
- Prompt history: dropdown of recent prompts

### Inline Image Generation in Chat
When user asks "generate an image of..." in chat:
- Aura detects image generation intent
- Returns the image inline in the chat message
- "Open in Image Panel" button on the image for further editing

---

## 6B. Image Gallery

### Component: `web/src/components/ImageGallery.tsx`

Grid of all generated images with:
- Thumbnail grid (masonry or fixed grid)
- Click to expand (lightbox with zoom)
- Filter by prompt text search
- Sort by date
- Bulk download
- Delete individual or all
- Show prompt + settings used for each image

Storage: `localStorage` with key `aura_image_history`, max 50 items, images stored as base64 or blob URLs.

---

## 6C. Voice/Audio Panel (stretch)

### Problem
Voice/TTS is partially set up (ToolsPanel shows status) but no dedicated UI for:
- Text-to-speech generation
- Audio file transcription
- Voice cloning (if supported by provider)

### What To Build (if audio providers are functional)
- TTS panel: paste or type text → generate audio → play/download
- Transcription: upload audio file → get text
- Use in chat: "Read this response aloud" button on messages

---

## Definition of Done — Phase 6
- [ ] Image generation panel with prompt input, model/size/style pickers
- [ ] Generated images display with download, copy, variation options
- [ ] Generation history with thumbnail grid
- [ ] Prompt history dropdown
- [ ] Inline image generation in chat ("generate an image of...")
- [ ] Image gallery with search, sort, delete
- [ ] Light theme compatible
- [ ] Mobile responsive (stacked layout on narrow screens)
- [ ] Voice/audio panel (stretch goal)
