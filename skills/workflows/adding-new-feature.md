# Workflow: Adding a New Feature to AURA

## Step-by-Step

### 1. Research Phase
- Check existing tools for overlap
- Review `research/tools/complete-tool-inventory.md`
- Identify dependencies needed
- Design data model

### 2. Implementation Phase
- Create tool file: `aura/tools/<name>.py`
- Follow pattern from `skills/patterns/tool-creation-pattern.md`
- Implement core methods
- Add `execute()` dispatcher

### 3. Registration Phase
- `tools/__init__.py` — add import + __all__
- `agent.py` — import, register (core or conditional), _lazy_tools, _ensure_tool
- `brain.py` — description, TOOL: normalization, fallback detection

### 4. API Phase (Optional)
- Add routes in `api/routes/tools_new.py`
- Register router in `api/main.py`

### 5. Verification Phase
```bash
# Syntax check
python -m py_compile aura/tools/<name>.py

# Import check
python -c "from aura.tools import MyTool"

# Functional test
python -c "
from aura.tools.<name> import MyTool
t = MyTool()
print(t.execute('test_action'))
"
```

### 6. Documentation Phase
- Save research to `research/tools/<name>.md`
- Update `research/tools/complete-tool-inventory.md`

### 7. Commit
```bash
git add <files>
git commit -m "Add <ToolName>: <description>"
git push origin main
```

## Parallel Implementation
Tools with no dependencies between them can be created in parallel (Steps 2 can run concurrently). Registration (Step 3) must happen after all tools are created.
