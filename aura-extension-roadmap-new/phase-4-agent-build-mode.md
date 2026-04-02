# Phase 4: Agent Build Mode — Aura's Differentiator

**Effort:** 3-4 days
**Impact:** This is what makes Aura unique. Nobody else has a personal agent building apps live.
**Depends on:** Phase 3 (multi-file support)

---

## The Vision

User says: "Build me a portfolio website with a dark theme, project gallery, and contact form."

Aura's agent loop kicks in. The user watches in real-time as files appear one by one:
1. `index.html` — structure appears in preview
2. `styles/main.css` — styling flows in, layout snaps into place
3. `scripts/gallery.js` — gallery becomes interactive
4. `scripts/contact.js` — form validation wires up

Each file broadcasts via the existing WebSocket artifact system. The preview updates live. The file tree populates as files land. The user can intervene at any step — "make the header sticky" — and the agent adjusts.

This is Bolt.new behavior but powered by YOUR agent, YOUR tool system, YOUR personality.

---

## 4A. Agent Build Pipeline

### Current Flow (Live Mode)
```
Agent loop (aura/core/agentic_loop.py)
  → write_file tool call
  → broadcast_artifact(filename, code)  [api/routes/artifacts.py]
  → WebSocket → ArtifactsPanel Live mode
  → Shows latest file in iframe
```

This already works. The gap: it's passive. The agent writes files during normal chat. There's no dedicated "build me a project" workflow.

### New Flow: Agent Build Mode
```
User: "Build me a portfolio website"
  → Extension sends: POST /api/agent/build
    { description: "portfolio website with dark theme...",
      framework: "static",         // or "react", auto-detected
      files: []                    // empty = new project
    }
  → Backend creates a build plan (list of files to create)
  → Agent loop executes plan, calling write_file for each
  → Each write_file → broadcast_artifact → WebSocket → Extension
  → Extension VirtualFS receives files, updates file tree + preview
  → After all files: agent sends { type: "build_complete" }
```

### Backend Endpoint

**New file or addition to:** `api/routes/build.py`

```python
@router.post("/api/agent/build")
async def start_build(request: BuildRequest):
    """
    Triggers the agent to build a multi-file project.
    Returns immediately. Progress streams via WebSocket.
    """
    # 1. Create a build plan prompt
    plan_prompt = f"""
    Build a complete web project based on this description:
    {request.description}
    
    Framework: {request.framework}
    Existing files: {request.files or 'None (new project)'}
    
    Create all necessary files. Use write_file for each file.
    Build in this order: structure (HTML) → styling (CSS) → logic (JS) → polish.
    """
    
    # 2. Queue the build task in the agent loop
    task_id = await agent_loop.queue_task(plan_prompt, tools=['write_file', 'edit_file'])
    
    return { "task_id": task_id, "status": "building" }
```

### Build Progress Protocol

Extend the WebSocket `/api/artifacts/stream` to include build status:

```json
// Build started
{ "type": "build_start", "task_id": "...", "plan": ["index.html", "styles/main.css", "scripts/app.js"] }

// File created (existing artifact_update)
{ "type": "artifact_update", "filename": "index.html", "code": "...", "artifact_type": "html" }

// Build step status
{ "type": "build_progress", "step": 2, "total": 4, "message": "Creating styles..." }

// Build complete
{ "type": "build_complete", "task_id": "...", "files_created": 4 }

// Build error
{ "type": "build_error", "task_id": "...", "error": "..." }
```

---

## 4B. Build Mode UI

### Extension Panel Changes

**In WebCreatorPanel or a new dedicated BuildPanel:**

1. **Build Prompt Input** — large textarea at top:
   "Describe what you want to build..."
   
2. **Framework Picker** — icon buttons:
   Static HTML | React | Vue | Next.js | Express
   (auto-detected from description if not selected)

3. **Build Progress Bar** — appears during build:
   ```
   Building portfolio website...
   [████████░░░░] 3/5 files
   ✓ index.html
   ✓ styles/main.css  
   ✓ scripts/gallery.js
   ◌ scripts/contact.js (building...)
   ◌ assets/README.md
   ```

4. **Live Preview** — iframe updates as each file lands

5. **File Tree** — populates as files are created (Phase 3 VirtualFS)

6. **Interrupt Button** — "Stop Building" cancels the agent task

7. **Iterate Button** — after build completes, chat input appears for refinements:
   "Make the header gradient instead of solid"
   → Agent edits specific files → WebSocket broadcasts → preview updates

### Build Mode vs Chat Mode
- **Chat Mode** (existing): conversational, one message at a time, AI responds with a single file
- **Build Mode** (new): task-oriented, AI creates an entire project autonomously, user watches and intervenes

Toggle between modes via a switch in the panel toolbar.

---

## 4C. Smart Build Planning

### Pre-Build Analysis

Before the agent starts writing files, it should plan:

```python
async def plan_build(description: str, framework: str) -> BuildPlan:
    """Ask the LLM to create a file plan before building."""
    plan_prompt = f"""
    Plan a web project for: {description}
    Framework: {framework}
    
    Return a JSON list of files to create, in order:
    [
      {{ "path": "index.html", "purpose": "Main page structure", "priority": 1 }},
      {{ "path": "styles/main.css", "purpose": "Core styling", "priority": 2 }},
      ...
    ]
    
    Rules:
    - Order by dependency (HTML before CSS before JS)
    - Max 10 files for static projects, 15 for React
    - Include only essential files, no boilerplate
    """
    return await llm.generate(plan_prompt, response_format="json")
```

Send the plan to the extension BEFORE building starts so the user can:
- See what's coming
- Remove files they don't want
- Add files they do want
- Reorder priorities
- Approve the plan before execution begins

---

## 4D. Iterative Refinement

After the initial build, the agent stays in context. The user can:

1. **Click an element** (element selection from WebCreator) → "Change this to a card layout"
   - Agent receives: element HTML + CSS path + user instruction
   - Agent identifies which file(s) to edit
   - Agent calls `edit_file` → broadcast → preview updates

2. **Select a file in the tree** → "Add form validation to this file"
   - Agent receives: file path + content + instruction
   - Agent edits that specific file

3. **General instruction** → "Add a dark mode toggle"
   - Agent decides which files need changes
   - May create new files (e.g., `scripts/darkmode.js`)
   - Updates existing files (e.g., add toggle button to `index.html`)

### Context Management
- The agent maintains the full project context (all file paths + contents)
- For large projects: only include the relevant files in the prompt, not all of them
- Use the VirtualFS to provide a project summary:
  ```
  Project: portfolio-website
  Files:
  - index.html (2.1KB) — main page
  - styles/main.css (1.4KB) — styling
  - scripts/gallery.js (0.8KB) — gallery logic
  - scripts/contact.js (0.6KB) — contact form
  
  [Full content of the file being edited]
  ```

---

## 4E. Integration with scaffold.py

The existing `scaffold.py` (backend tool) generates full project scaffolds on disk. Agent Build Mode should:

1. **Use scaffold templates as starting points** — when the user picks "React" framework, use scaffold.py's React-Vite template as the base file set
2. **Sync scaffold output to VirtualFS** — scaffold creates files on disk → read them → push to extension VirtualFS via WebSocket
3. **Two-way sync** (stretch goal) — changes in VirtualFS can be written back to disk via the agent

### Flow
```
User picks "React" + describes the app
  → Agent calls scaffold.py React-Vite template → creates files on disk
  → Files broadcast via WebSocket → Extension VirtualFS populates
  → Agent then customizes files based on user's description
  → Each edit broadcasts → preview updates live
```

---

## Definition of Done — Phase 4
- [ ] `POST /api/agent/build` endpoint triggers multi-file agent build
- [ ] Build plan is generated and shown to user before execution
- [ ] User can approve/modify the plan before building starts
- [ ] Files appear in real-time via WebSocket as agent creates them
- [ ] Progress bar shows build status (X/Y files, current file)
- [ ] Live preview updates as each file lands
- [ ] File tree populates during build
- [ ] User can interrupt build at any point
- [ ] After build: iterative refinement via chat works (agent edits specific files)
- [ ] Element selection works for targeted edits ("change this element")
- [ ] Agent maintains project context across refinement turns
- [ ] scaffold.py templates used as starting points for framework projects
- [ ] Build mode toggle distinct from chat mode in the UI
