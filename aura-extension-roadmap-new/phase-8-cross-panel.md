# Phase 8: Cross-Panel Workflows + Component Library

**Effort:** 4-5 days
**Impact:** Turns isolated panels into a cohesive creation platform
**Depends on:** Phase 3 (VirtualFS), Phase 4 (Agent Build Mode)

---

## The Problem

Today, each creation panel is an island:
- WebCreator builds HTML pages — can't use Python data from CodePanel
- CodePanel runs Python — can't render output in ArtifactsPanel
- ArtifactsPanel generates components — can't be used in WebCreator
- CapturePanel clones pages — can't feed into WebCreator for remixing

Users have to manually copy-paste between panels. There's no concept of a "project" that spans panels.

---

## 8A. Unified Project Context

### The Project Abstraction

**New file:** `extension-src/src/utils/projectContext.ts`

```typescript
interface AuraProject {
  id: string;
  name: string;
  description: string;
  
  // Multi-panel file system
  files: VirtualFS;
  
  // Panel-specific state
  webCreator: {
    conversationHistory: Message[];
    activeFile: string;
  };
  artifacts: {
    savedArtifacts: SavedArtifact[];
    activeArtifact: string;
  };
  code: {
    exchanges: Exchange[];
    sessionVariables: Record<string, any>;
  };
  
  // Shared context
  components: ComponentEntry[];     // reusable components (8B)
  dataOutputs: DataOutput[];        // outputs from CodePanel available to other panels
  designTokens: DesignTokens;       // shared colors, fonts, spacing
  
  createdAt: number;
  updatedAt: number;
}

// Zustand slice for project context
interface ProjectStore {
  activeProject: AuraProject | null;
  projects: AuraProject[];
  
  createProject(name: string, description: string): AuraProject;
  openProject(id: string): void;
  closeProject(): void;
  
  // Cross-panel data flow
  publishData(from: PanelType, key: string, data: any): void;
  consumeData(key: string): any;
}
```

### Cross-Panel Data Flow

**CodePanel → WebCreator/Artifacts:**
```
CodePanel runs analysis → outputs a DataFrame or chart
  → User clicks "Use in Web Creator" on the output
  → Data is serialized and published to project context
  → WebCreator receives it as a variable: {{data.chart}} or {{data.table}}
  → AI generates HTML that renders the data
```

**CapturePanel → WebCreator:**
```
CapturePanel clones a page → extracts HTML/CSS
  → User clicks "Edit in Web Creator"
  → HTML loads in WebCreator as starting point
  → User refines via chat
```

**ArtifactsPanel → WebCreator:**
```
ArtifactsPanel generates a component (e.g., navbar)
  → User clicks "Save to Components"
  → Component saved in project's component library
  → WebCreator AI references it: "Use the saved navbar component"
```

**WebCreator → CodePanel:**
```
WebCreator needs dynamic data (e.g., "fetch weather data")
  → Auto-switches to CodePanel with pre-filled prompt: "Generate weather data as JSON"
  → CodePanel runs Python → output saved as project data
  → Returns to WebCreator → AI uses the data in the page
```

### UI: Project Switcher
- Top of sidebar (above Rail): project name dropdown
- "New Project" / "Open Project" / "Close Project"
- When a project is open, all panels share the same VirtualFS and context
- When no project is open, panels work independently (backward compatible)

---

## 8B. Component Library

### Vision
Users build up reusable components over time. The AI references existing components when building new pages instead of generating everything from scratch. This is the VOYAGER concept from tool_builder.py applied to UI components.

### Component Model

```typescript
interface UIComponent {
  id: string;
  name: string;           // e.g. "PricingCard", "DarkNavbar", "ContactForm"
  description: string;    // what it does, when to use it
  category: string;       // "navigation" | "cards" | "forms" | "layout" | "hero" | "footer" | etc.
  html: string;           // the component HTML
  css: string;            // scoped CSS (or Tailwind classes)
  js?: string;            // optional JavaScript
  thumbnail?: string;     // base64 preview
  tags: string[];
  usageCount: number;     // how many times it's been used
  createdAt: number;
  source: 'generated' | 'captured' | 'manual';  // where it came from
}
```

### Storage

**File:** `extension-src/src/utils/componentLibrary.ts`

```typescript
const COMPONENTS_KEY = 'aura_component_library';
const MAX_COMPONENTS = 100;

class ComponentLibrary {
  async list(category?: string): Promise<UIComponent[]>;
  async get(id: string): Promise<UIComponent | null>;
  async save(component: Omit<UIComponent, 'id' | 'createdAt' | 'usageCount'>): Promise<string>;
  async delete(id: string): Promise<void>;
  async update(id: string, updates: Partial<UIComponent>): Promise<void>;
  async search(query: string): Promise<UIComponent[]>;   // fuzzy search by name/description/tags
  async incrementUsage(id: string): Promise<void>;
  
  // For AI context: generate a summary of available components
  async getSummaryForLLM(): Promise<string>;
  // Returns: "Available components:\n- PricingCard: A three-tier pricing card...\n- DarkNavbar: ..."
}
```

### Saving Components

**From ArtifactsPanel:**
- "Save as Component" button on any generated artifact
- Opens modal: name, category (auto-suggested), tags, description
- Thumbnail auto-captured from preview

**From WebCreator (element selection):**
- Click an element → "Save as Component"
- Extracts the element's HTML + relevant CSS
- Saves with context about how it was used

**From CapturePanel:**
- After capturing a page, "Save Component" on individual captured elements
- Particularly useful for copying designs from other sites

### Using Components in Generation

When generating new pages, the AI system prompt includes available components:

```
You have these reusable components available. Use them when appropriate instead of building from scratch:

[COMPONENTS]
- PricingCard: A responsive three-tier pricing card with hover effects. HTML: <div class="pricing-card">...
- DarkNavbar: A sticky dark navigation bar with mobile hamburger menu. HTML: <nav class="dark-nav">...
- ContactForm: A validated contact form with name/email/message fields. HTML: <form class="contact-form">...
[/COMPONENTS]

When using a component, include its HTML and CSS as-is, adapting only the content (text, colors) to match the current project.
```

### Component Gallery UI

**New component:** `extension-src/src/components/ComponentGallery.tsx`

- Grid view of saved components with thumbnails
- Filter by category and search
- Click to preview (renders in a small iframe)
- "Insert" button → copies component HTML to clipboard or inserts at cursor in code editor
- "Use in new page" → opens WebCreator with the component pre-loaded
- Sort by: recently used, most used, newest, alphabetical
- Drag-and-drop from gallery into code editor (stretch goal)

---

## 8C. Design Tokens (Shared Theme)

### Problem
Each generation uses different colors, fonts, and spacing. The theme editor in WebCreator is single-page only. Components from different generations look mismatched.

### Solution: Project-Level Design Tokens

```typescript
interface DesignTokens {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    error: string;
    success: string;
  };
  fonts: {
    heading: string;     // e.g. "Inter, sans-serif"
    body: string;        // e.g. "Inter, sans-serif"
    mono: string;        // e.g. "JetBrains Mono, monospace"
  };
  spacing: {
    unit: number;        // base unit in px (default 4)
    radius: string;      // border-radius (default "8px")
  };
  darkMode: boolean;
}
```

### Usage
- When a project has design tokens, inject them as CSS variables into every generated page:
  ```css
  :root {
    --color-primary: #6366f1;
    --color-secondary: #8b5cf6;
    /* ... */
  }
  ```
- AI system prompt includes the tokens: "Use these design tokens for all styling. Use CSS variables (var(--color-primary)) instead of hardcoded colors."
- Theme editor (WebCreator) updates project tokens, which propagate to all panels

### Token Presets
- Aura Dark (default purple/dark theme)
- Aura Light
- Minimal (black/white/gray)
- Ocean (blue tones)
- Forest (green tones)
- Custom (user picks)

---

## 8D. Panel Handoff Protocol

### How Panels Pass Work to Each Other

```typescript
interface PanelHandoff {
  from: PanelType;
  to: PanelType;
  action: string;           // "edit", "render", "execute", "capture"
  data: {
    code?: string;
    files?: Record<string, string>;
    context?: string;        // natural language context
    selectedElement?: { html: string; cssPath: string };
  };
}

// In the Zustand store:
interface PanelHandoffStore {
  pendingHandoff: PanelHandoff | null;
  
  handoff(params: PanelHandoff): void;
  // Sets pendingHandoff + switches activePanel to the target
  
  consumeHandoff(): PanelHandoff | null;
  // Target panel reads and clears the handoff on mount
}
```

### Built-In Handoffs

| From | To | Action | Trigger |
|------|-----|--------|---------|
| CapturePanel | WebCreatorPanel | edit | "Edit in Web Creator" button |
| CapturePanel | ArtifactsPanel | render | "Open as Artifact" button |
| ArtifactsPanel | WebCreatorPanel | edit | "Edit in Web Creator" button |
| WebCreatorPanel | CodePanel | execute | "Add data with Python" button |
| CodePanel | ArtifactsPanel | render | "Render as Artifact" on HTML/chart output |
| CodePanel | WebCreatorPanel | edit | "Use in Web Creator" on data output |
| Any Panel | SlidesPanel | present | "Create Slides from This" |

---

## 8E. Smart Panel Suggestions

When the AI detects a need that another panel can fulfill better:

```
User in WebCreator: "Add a chart showing monthly sales data"

AI Response:
"To add a chart with real data, I'd recommend:
1. [Switch to Code Panel] → generate the sales data with Python
2. [Return here] → I'll render it as a Chart.js visualization

Or I can use placeholder data: [Generate with sample data]"
```

The `[Switch to Code Panel]` text is a clickable action that triggers a handoff.

---

## Definition of Done — Phase 8
- [ ] Project abstraction wraps VirtualFS + panel state + shared context
- [ ] Project switcher in sidebar header (create, open, close)
- [ ] Cross-panel data flow: CodePanel outputs available in WebCreator/Artifacts
- [ ] CapturePanel → WebCreator handoff works with one click
- [ ] ArtifactsPanel → Component Library save works
- [ ] Component Library stores up to 100 components with thumbnails
- [ ] Component Gallery UI with search, filter, preview, insert
- [ ] AI references available components in generation prompts
- [ ] Design tokens system with CSS variable injection
- [ ] Token presets (5 built-in themes)
- [ ] Theme editor updates propagate to all panels via project context
- [ ] Panel handoff protocol works for all defined handoff routes
- [ ] Smart panel suggestions appear when cross-panel workflow would help
- [ ] All features backward compatible: panels still work independently without a project
