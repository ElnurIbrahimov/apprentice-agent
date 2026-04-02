# Phase 7: Visual AI Feedback Loop

**Effort:** 2-3 days
**Impact:** Closes the biggest quality gap — AI can SEE what it built and self-correct
**Depends on:** Phase 1 (error recovery loop) + a vision-capable model

---

## The Problem

When the AI generates a website, it has no idea what the result actually looks like. It can only read the code it wrote. If:
- The layout is broken but technically valid HTML
- Colors look bad together
- Text is unreadable on the background
- Spacing is off
- The design doesn't match the user's intent

...the AI has no way to know. The user has to manually describe the visual problem.

v0.dev and Claude Artifacts are starting to use visual feedback. This is the next frontier.

---

## 7A. Screenshot Capture

### Capturing the iframe

**New utility:** `extension-src/src/utils/screenshotCapture.ts`

```typescript
import html2canvas from 'html2canvas';

async function captureIframeScreenshot(
  iframe: HTMLIFrameElement,
  options?: { maxWidth?: number; quality?: number }
): Promise<string> {
  const { maxWidth = 800, quality = 0.7 } = options || {};
  
  try {
    // Method 1: html2canvas on iframe's document
    const iframeDoc = iframe.contentDocument;
    if (!iframeDoc) throw new Error('Cannot access iframe document');
    
    const canvas = await html2canvas(iframeDoc.body, {
      width: iframe.clientWidth,
      height: Math.min(iframe.contentDocument.body.scrollHeight, 2000),
      scale: 1,
      useCORS: true,
      logging: false,
    });
    
    // Resize if too large
    const resizedCanvas = resizeCanvas(canvas, maxWidth);
    
    // Convert to base64 JPEG (smaller than PNG for screenshots)
    return resizedCanvas.toDataURL('image/jpeg', quality);
    
  } catch (e) {
    // Method 2: Fallback — use chrome.tabs.captureVisibleTab if in extension context
    // This captures the whole tab, need to crop to iframe bounds
    console.warn('html2canvas failed, using fallback:', e);
    return await captureViaExtensionAPI(iframe);
  }
}

// Compress to keep under 100KB for LLM context
function compressScreenshot(base64: string, targetBytes: number = 100_000): string {
  // Reduce quality iteratively until under target size
  // Or resize dimensions
}
```

### Installation
```bash
cd D:/Aura/extension-src
npm install html2canvas
```

### When to Capture
1. **After generation completes** — wait 500ms for rendering to settle, then capture
2. **After error recovery** — capture after each fix attempt to verify
3. **On user request** — "Does this look right?" button
4. **Before iteration** — capture current state before sending edit request

---

## 7B. Visual Feedback Prompt

### Sending Screenshot to Vision Model

```typescript
async function getVisualFeedback(
  screenshot: string,     // base64 image
  userPrompt: string,     // what the user originally asked for
  currentCode: string,    // the generated code
  model?: string          // vision-capable model
): Promise<VisualFeedback> {
  
  const response = await fetch('/api/generate/raw', {
    method: 'POST',
    body: JSON.stringify({
      message: `
        I generated this web page based on the user's request.
        
        USER REQUEST: "${userPrompt}"
        
        SCREENSHOT: [attached image]
        
        Analyze the screenshot and identify any visual issues:
        1. Does the result match what the user asked for?
        2. Are there layout problems (overlapping, misaligned, overflow)?
        3. Are colors/contrast/readability good?
        4. Is the design professional and polished?
        5. Any missing elements the user likely expected?
        
        Return JSON:
        {
          "matches_intent": true/false,
          "score": 1-10,
          "issues": [
            { "type": "layout|color|typography|missing|broken",
              "description": "...",
              "severity": "high|medium|low",
              "suggestion": "..." }
          ],
          "overall": "brief assessment"
        }
      `,
      model: model || 'llava',  // or any vision model available via Ollama
      images: [screenshot],     // base64 image
    })
  });
  
  return parseJSON(response);
}
```

### Backend Support

The `/api/generate/raw` endpoint needs to support image inputs for vision models:

**Modify:** `api/routes/generate.py`

Add `images` field to the request schema:
```python
class RawGenerateRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    history: Optional[List[dict]] = None
    images: Optional[List[str]] = None  # NEW: base64 images for vision models
```

Pass images to the Ollama API call:
```python
ollama.chat(
    model=model,
    messages=[...],
    images=request.images  # Ollama supports base64 images natively
)
```

---

## 7C. Auto-Refinement Loop

### Flow
```
1. User prompts: "Build a pricing page with 3 tiers"
2. AI generates HTML → renders in iframe
3. Wait 500ms → capture screenshot
4. Send screenshot + original prompt to vision model
5. Vision model returns feedback:
   { score: 6, issues: [
     { type: "layout", description: "Cards are not evenly spaced", severity: "high" },
     { type: "color", description: "CTA button blends into background", severity: "medium" }
   ]}
6. If score < 8 AND auto-refine is ON:
   a. Build refinement prompt from issues
   b. Send to generation model: "Fix these visual issues: ..."
   c. New code → re-render → re-capture → re-evaluate
   d. Max 2 visual refinement rounds
7. Show user the final result + visual feedback summary
```

### Refinement Prompt Builder
```typescript
function buildRefinementPrompt(feedback: VisualFeedback, currentCode: string): string {
  const issueList = feedback.issues
    .filter(i => i.severity === 'high' || i.severity === 'medium')
    .map(i => `- ${i.type.toUpperCase()}: ${i.description}. Suggestion: ${i.suggestion}`)
    .join('\n');
  
  return `
    The current page has these visual issues:
    ${issueList}
    
    Current code:
    \`\`\`html
    ${currentCode}
    \`\`\`
    
    Fix ALL the issues above. Return the complete corrected code.
    Focus on: proper spacing, color contrast, professional appearance.
  `;
}
```

---

## 7D. Visual Feedback UI

### Quality Score Badge
After generation, show a visual quality score in the toolbar:
- 8-10: Green badge "Great"
- 5-7: Yellow badge "Needs work" (expandable to see issues)
- 1-4: Red badge "Issues found" (auto-opens issue panel)

### Issue Panel
Expandable panel below the preview showing detected issues:
```
Visual Analysis (Score: 7/10)
├── HIGH: Cards are not evenly spaced → suggested: use CSS grid with equal columns
├── MEDIUM: CTA button blends into background → suggested: increase contrast
└── LOW: Footer text is small → suggested: increase to 14px

[Auto-Fix Issues] [Dismiss]
```

"Auto-Fix Issues" triggers the refinement loop.

### Before/After Toggle
After refinement, show a toggle:
- "Before" — original generation
- "After" — refined version
- Side-by-side comparison (if panel is wide enough)

---

## 7E. Model Requirements

### Vision Models via Ollama
Check what's available:
- `llava` — 7B, good quality, ~4GB VRAM (fits RTX 4060)
- `llava:13b` — better quality but may not fit in 8GB
- `bakllava` — BakLLaVA, fast and compact
- `moondream` — tiny (1.6B), very fast, less accurate

### Cloud Vision Models
If local vision models aren't available or accurate enough:
- Route through Ollama cloud models that support vision
- Or add a direct API call to a vision provider

### Recommendation
Start with `moondream` for speed (visual quality check is a quick task, doesn't need a large model). Fall back to `llava` if accuracy is too low. Make the model configurable in settings.

---

## 7F. User-Initiated Visual Compare

Beyond auto-refinement, let users leverage visual feedback manually:

1. **"How does this look?"** button — sends screenshot to vision model, shows feedback
2. **Reference image upload** — user uploads a design mockup/screenshot → AI compares the generated output against the reference
3. **"Match this design"** — user provides a reference URL or image → CapturePanel clones it → visual feedback loop refines until it matches

---

## Definition of Done — Phase 7
- [ ] Screenshot capture works reliably for iframe content
- [ ] Screenshots compressed to <100KB for efficient LLM context
- [ ] `/api/generate/raw` supports image inputs for vision models
- [ ] Visual feedback prompt returns structured JSON with scores and issues
- [ ] Auto-refinement loop runs up to 2 rounds when score < 8
- [ ] Quality score badge shows in toolbar after generation
- [ ] Issue panel lists detected problems with fix suggestions
- [ ] "Auto-Fix Issues" button triggers targeted refinement
- [ ] Before/After toggle shows improvement
- [ ] Vision model is configurable (default: moondream or llava)
- [ ] Works without vision model — gracefully disabled with a message
- [ ] Reference image comparison (stretch goal)
