"""Replace silent except Exception: pass with logger.debug() calls.

Targets ~157 instances across 20 files. Skips __del__ methods and finally blocks.
Also fixes bare except: -> except Exception as e:.
"""

import re
import py_compile
import sys
from pathlib import Path

# Files to process with their log prefixes
TARGETS = {
    "aura/agent.py": "Agent",
    "aura/brain.py": "Brain",
    "aura/consciousness/idle_presence.py": "IdlePresence",
    "aura/consciousness/self_improvement.py": "SelfImprovement",
    "aura/consciousness/intrinsic_motivation.py": "IntrinsicMotivation",
    "aura/memory/unified_memory.py": "UnifiedMemory",
    "aura/memory/kg_contradiction.py": "KGContradiction",
    "aura/tools/amem.py": "AMem",
    "aura/tools/neurodream.py": "NeuroDream",
    "aura/tools/knowledge_graph.py": "KnowledgeGraph",
    "aura/emotion/alma_engine.py": "AlmaEngine",
    "aura/proactive/gateway_daemon.py": "GatewayDaemon",
    "aura/proactive/salience_filter.py": "SalienceFilter",
    "aura/consciousness/proactive_awareness.py": "ProactiveAwareness",
    "aura/core/agentic_loop.py": "AgenticLoop",
    "aura/core/mcp_client.py": "MCPClient",
    "aura/core/repo_map.py": "RepoMap",
    "aura/core/commands.py": "Commands",
    "aura/proactive/monitors/screen_monitor.py": "ScreenMonitor",
    "aura/tools/deep_research.py": "DeepResearch",
}

# Pattern: except Exception: followed by pass (with any indentation)
# Captures the indentation to preserve it
PATTERN_EXCEPTION_PASS = re.compile(
    r"^(\s*)except\s+Exception\s*:\s*\n(\s*)pass\s*$",
    re.MULTILINE,
)

# Pattern: bare except: followed by pass
PATTERN_BARE_EXCEPT_PASS = re.compile(
    r"^(\s*)except\s*:\s*\n(\s*)pass\s*$",
    re.MULTILINE,
)

# Pattern: except Exception as e: followed by pass
PATTERN_EXCEPTION_AS_PASS = re.compile(
    r"^(\s*)except\s+Exception\s+as\s+\w+\s*:\s*\n(\s*)pass\s*$",
    re.MULTILINE,
)


def is_in_del_or_finally(content: str, match_start: int) -> bool:
    """Check if the match is inside a __del__ method or finally block."""
    # Look backwards for def __del__ or finally:
    before = content[:match_start]
    lines_before = before.split("\n")

    # Check last 20 lines for __del__ or finally
    recent = lines_before[-20:] if len(lines_before) > 20 else lines_before
    for line in reversed(recent):
        stripped = line.strip()
        if stripped.startswith("def __del__"):
            return True
        if stripped == "finally:":
            return True
        # If we hit another def or class, stop looking
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
    return False


def ensure_logger_import(content: str) -> str:
    """Ensure file has logger = logging.getLogger(__name__)."""
    if "logger = logging.getLogger" in content:
        return content
    if "import logging" not in content:
        # Add both import and logger
        content = "import logging\n" + content
    # Add logger after the import block
    lines = content.split("\n")
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
        elif insert_at > 0 and line.strip() and not line.startswith("import ") and not line.startswith("from ") and not line.startswith("#"):
            break
    lines.insert(insert_at, "\nlogger = logging.getLogger(__name__)\n")
    return "\n".join(lines)


def process_file(filepath: Path, prefix: str) -> tuple[int, int]:
    """Process a single file. Returns (replaced_count, error_count)."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  SKIP {filepath}: {e}")
        return 0, 1

    original = content
    count = 0

    # Fix bare except: pass -> except Exception as e: logger.debug(...)
    def replace_bare(m):
        nonlocal count
        if is_in_del_or_finally(content, m.start()):
            return m.group(0)
        indent = m.group(1)
        count += 1
        return f'{indent}except Exception as e:\n{indent}    logger.debug(f"[{prefix}] non-critical: {{e}}")'

    content = PATTERN_BARE_EXCEPT_PASS.sub(replace_bare, content)

    # Fix except Exception: pass -> except Exception as e: logger.debug(...)
    def replace_exception(m):
        nonlocal count
        if is_in_del_or_finally(content, m.start()):
            return m.group(0)
        indent = m.group(1)
        count += 1
        return f'{indent}except Exception as e:\n{indent}    logger.debug(f"[{prefix}] non-critical: {{e}}")'

    content = PATTERN_EXCEPTION_PASS.sub(replace_exception, content)

    # Fix except Exception as e: pass -> except Exception as e: logger.debug(...)
    def replace_exception_as(m):
        nonlocal count
        if is_in_del_or_finally(content, m.start()):
            return m.group(0)
        indent = m.group(1)
        count += 1
        return f'{indent}except Exception as e:\n{indent}    logger.debug(f"[{prefix}] non-critical: {{e}}")'

    content = PATTERN_EXCEPTION_AS_PASS.sub(replace_exception_as, content)

    if count == 0:
        return 0, 0

    # Ensure logger exists
    content = ensure_logger_import(content)

    # Verify syntax
    filepath.write_text(content, encoding="utf-8")
    try:
        py_compile.compile(str(filepath), doraise=True)
        print(f"  OK {filepath.name}: {count} replacements")
        return count, 0
    except py_compile.PyCompileError as e:
        # Rollback
        filepath.write_text(original, encoding="utf-8")
        print(f"  FAIL {filepath.name}: syntax error after replacement, rolled back: {e}")
        return 0, 1


def main():
    root = Path(__file__).resolve().parent.parent
    total_replaced = 0
    total_errors = 0

    print(f"Processing {len(TARGETS)} files in {root}...\n")

    for rel_path, prefix in TARGETS.items():
        filepath = root / rel_path
        if not filepath.exists():
            print(f"  SKIP {rel_path}: file not found")
            total_errors += 1
            continue
        replaced, errors = process_file(filepath, prefix)
        total_replaced += replaced
        total_errors += errors

    print(f"\nDone: {total_replaced} replacements, {total_errors} errors")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
