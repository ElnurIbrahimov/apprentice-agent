# Common Issues and Fixes

## Import Errors

### "Module not found"
- Check `tools/__init__.py` has the import
- Check the class name matches exactly
- Check file is in the correct directory

### Circular imports
- Use lazy imports inside functions: `from module import Class`
- Don't import at module level if it causes cycles

## Tool Not Working

### Tool not in agent.tools
- Check `agent.py` registration (core dict or conditional block)
- Check `_ensure_tool()` has the tool
- Check `_lazy_tools` list includes it

### LLM not selecting the tool
- Check `brain.py` description is clear and distinctive
- Check TOOL: normalization rules match common names
- Check fallback detection covers natural language variations
- Test with explicit: "use the X tool to..."

### Tool returns error
- Check data directory exists (`data/` subfolder)
- Check file permissions
- Check dependencies installed

## Startup Issues

### Slow startup
- Move heavy tools to conditional loading (`if not fast_init:`)
- Use lazy model loading pattern
- Check for tools doing network calls in `__init__`

### Crash on startup
- Check all conditional tool loads have try/except
- Check for missing dependencies
- Run `python -m py_compile` on all modified files

## API Issues

### 404 on endpoint
- Check router is registered in `api/main.py`
- Check prefix matches (`/api/...`)

### Blocking event loop
- Wrap sync calls in `run_in_executor`
- Never call `agent.tools[x].method()` directly in async handler

## Git Issues

### Accidentally committed wrong files
- `git reset HEAD~1` (undo last commit, keep changes)
- `git checkout -- <file>` (discard changes to specific file)

### Windows CRLF warnings
- Safe to ignore. Git handles conversion automatically.
