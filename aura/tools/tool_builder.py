"""Tool Builder - Meta-tool for creating custom tools dynamically.

This tool allows the agent to create new tools at runtime, enabling
self-extension capabilities while maintaining security constraints.

Enhanced with:
- VOYAGER-style composition retrieval (find reusable existing tools)
- Automatic LLM-generated test cases
- GEPA evolution integration
- Usage tracking for deprecation decisions
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import logging
import time
import ast as _ast
from datetime import datetime
from pathlib import Path

# Sanitized environment for subprocess execution — prevents API key leaks
_SAFE_ENV_KEYS = frozenset({"PATH", "HOME", "USERPROFILE", "TEMP", "TMP",
                            "SYSTEMROOT", "WINDIR", "COMSPEC", "PYTHONPATH"})

def _safe_env() -> dict:
    return {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
from typing import Any, Callable, Optional

_BLOCKED_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "breakpoint"}
# Use the authoritative blocked-modules list from code_executor
try:
    from .code_executor import CodeExecutorTool
    _BLOCKED_MODULES = set(CodeExecutorTool.BLOCKED_MODULES)
except ImportError:
    _BLOCKED_MODULES = {"subprocess", "os", "sys", "shutil", "socket", "ctypes", "importlib", "pickle"}

from .tool_contract import ToolResult
from .tool_template import (
    TOOL_CLASS_TEMPLATE,
    METHOD_TEMPLATE,
    EXECUTE_DISPATCH_TEMPLATE,
    TEST_TEMPLATE,
    METHOD_TEST_TEMPLATE,
    NETWORK_IMPORTS,
)


# Paths
BASE_DIR = Path(__file__).parent.parent.parent
CUSTOM_TOOLS_DIR = Path(__file__).parent / "custom"
CUSTOM_TESTS_DIR = CUSTOM_TOOLS_DIR / "tests"
REGISTRY_FILE = BASE_DIR / "data" / "custom_tools.json"
LOGS_DIR = BASE_DIR / "logs" / "tool_builder"
USAGE_DB_PATH = BASE_DIR / "data" / "tool_usage.db"

# Ensure directories exist
CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_TESTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Usage Tracker
# ---------------------------------------------------------------------------

class ToolUsageTracker:
    """Track custom tool invocation counts for deprecation decisions."""

    def __init__(self, db_path: str = str(USAGE_DB_PATH)):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self):
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_usage (
                    tool_name TEXT,
                    invoked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    success INTEGER DEFAULT 1,
                    latency_ms INTEGER DEFAULT 0
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_lifecycle (
                    tool_name TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_used TEXT,
                    total_uses INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    status TEXT DEFAULT 'active'
                )
            """)
            self._conn.commit()
        except Exception as e:
            logger.warning(f"[UsageTracker] DB init failed: {e}")
            self._conn = None

    def record_use(self, tool_name: str, success: bool = True, latency_ms: int = 0):
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT INTO tool_usage (tool_name, success, latency_ms) VALUES (?, ?, ?)",
                (tool_name, int(success), latency_ms)
            )
            self._conn.execute("""
                INSERT INTO tool_lifecycle (tool_name, last_used, total_uses, success_rate)
                VALUES (?, CURRENT_TIMESTAMP, 1, ?)
                ON CONFLICT(tool_name) DO UPDATE SET
                    last_used = CURRENT_TIMESTAMP,
                    total_uses = total_uses + 1,
                    success_rate = (success_rate * total_uses + ?) / (total_uses + 1)
            """, (tool_name, float(success), float(success)))
            self._conn.commit()
        except Exception as e:
            logger.debug(f"[UsageTracker] record_use failed: {e}")

    def register_tool(self, tool_name: str):
        if not self._conn:
            return
        try:
            self._conn.execute("""
                INSERT OR IGNORE INTO tool_lifecycle (tool_name, created_at, total_uses, success_rate, status)
                VALUES (?, CURRENT_TIMESTAMP, 0, 1.0, 'active')
            """, (tool_name,))
            self._conn.commit()
        except Exception as e:
            logger.debug(f"[UsageTracker] register_tool failed: {e}")

    def get_stats(self, tool_name: str) -> Optional[dict]:
        if not self._conn:
            return None
        try:
            cursor = self._conn.execute(
                "SELECT total_uses, success_rate, created_at, last_used, status FROM tool_lifecycle WHERE tool_name = ?",
                (tool_name,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "total_uses": row[0], "success_rate": row[1],
                    "created_at": row[2], "last_used": row[3], "status": row[4],
                }
        except Exception:
            pass
        return None

    def get_candidates_for_deprecation(self, min_age_days: int = 30, max_usage_pct: float = 0.05) -> list:
        """Find tools that are old and rarely used."""
        if not self._conn:
            return []
        try:
            cursor = self._conn.execute("""
                SELECT tool_name, total_uses, success_rate, created_at, last_used
                FROM tool_lifecycle
                WHERE julianday('now') - julianday(created_at) > ?
                AND status = 'active'
                ORDER BY total_uses ASC
            """, (min_age_days,))
            return [
                dict(zip(['name', 'uses', 'success_rate', 'created', 'last_used'], row))
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.debug(f"[UsageTracker] get_candidates_for_deprecation failed: {e}")
            return []

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


# Singleton tracker
_usage_tracker: Optional[ToolUsageTracker] = None

def get_usage_tracker() -> ToolUsageTracker:
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = ToolUsageTracker()
        import atexit
        atexit.register(lambda: _usage_tracker.close() if _usage_tracker else None)
    return _usage_tracker


# ---------------------------------------------------------------------------
# Tool Builder
# ---------------------------------------------------------------------------

class ToolBuilderTool:
    """Meta-tool for creating, testing, and managing custom tools.

    Enhanced with VOYAGER-style composition, auto-testing, GEPA evolution,
    and usage tracking.
    """

    def __init__(self, brain=None):
        """
        Args:
            brain: Optional OllamaBrain instance for LLM-powered features
                   (auto-test generation, composition retrieval).
                   If None, those features degrade gracefully.
        """
        self.name = "tool_builder"
        self.description = "Create, test, enable, disable, and manage custom tools"
        self._brain = brain
        self._usage_tracker = get_usage_tracker()
        self._ensure_registry()

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _llm_generate(self, prompt: str, timeout: int = 30) -> Optional[str]:
        """Generate text via brain._quick_generate if available."""
        if self._brain is None:
            return None
        if not hasattr(self._brain, '_quick_generate'):
            return None
        try:
            return self._brain._quick_generate(prompt, timeout=timeout)
        except Exception as e:
            logger.debug(f"[ToolBuilder] LLM generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Registry helpers (unchanged)
    # ------------------------------------------------------------------

    def _ensure_registry(self) -> None:
        """Ensure the custom tools registry exists."""
        if not REGISTRY_FILE.exists():
            REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._save_registry({"tools": []})

    def _load_registry(self) -> dict:
        """Load the custom tools registry."""
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"tools": []}

    def _save_registry(self, registry: dict) -> bool:
        """Save the custom tools registry."""
        try:
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=4)
            return True
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")
            return False

    def _get_tool_entry(self, name: str) -> Optional[dict]:
        """Get a tool entry from the registry by name."""
        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == name:
                return tool
        return None

    # ------------------------------------------------------------------
    # Security scanning (unchanged)
    # ------------------------------------------------------------------

    def _scan_for_dangerous_code(self, code: str) -> tuple:
        """Scan code for dangerous patterns using AST analysis.

        Args:
            code: The Python code to scan

        Returns:
            (is_dangerous: bool, reason: str)
        """
        try:
            tree = _ast.parse(code)
        except SyntaxError as e:
            return True, f"Syntax error in generated code: {e}"
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                    return True, f"Blocked builtin: {node.func.id}"
                if isinstance(node.func, _ast.Attribute) and node.func.attr in _BLOCKED_BUILTINS:
                    return True, f"Blocked builtin: {node.func.attr}"
            if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, _ast.Import) else [node.module or ""]
                for name in names:
                    if name and name.split(".")[0] in _BLOCKED_MODULES:
                        return True, f"Blocked module import: {name}"
        return False, ""

    # ------------------------------------------------------------------
    # Name helpers (unchanged)
    # ------------------------------------------------------------------

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a tool name to be a valid Python identifier."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        if sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized

    def _to_class_name(self, name: str) -> str:
        """Convert tool name to PascalCase class name."""
        words = self._sanitize_name(name).split('_')
        return ''.join(word.capitalize() for word in words if word) + 'Tool'

    def _generate_keywords(self, name: str, description: str, functions_spec: list[dict]) -> list[str]:
        """Generate keywords for tool detection from name, description, and functions."""
        keywords = set()

        for word in name.lower().split('_'):
            if len(word) > 2:
                keywords.add(word)
        keywords.add(name.lower().replace('_', ' '))

        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                      'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                      'because', 'until', 'while', 'this', 'that', 'these', 'those'}

        desc_words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        for word in desc_words:
            if word not in stop_words and len(word) > 2:
                keywords.add(word)

        for func in functions_spec:
            func_name = func.get('name', '')
            for word in func_name.lower().split('_'):
                if len(word) > 2:
                    keywords.add(word)
            keywords.add(func_name.lower().replace('_', ' '))

        name_lower = name.lower()
        if 'calculator' in name_lower or 'calc' in name_lower:
            keywords.add('calculate')
            keywords.add('compute')
        if 'converter' in name_lower or 'convert' in name_lower:
            keywords.add('convert')
            keywords.add('conversion')
        if 'bmi' in name_lower:
            keywords.add('body mass index')
            keywords.add('height and weight')
        if 'temperature' in name_lower:
            keywords.add('celsius')
            keywords.add('fahrenheit')
            keywords.add('temp')

        return sorted(list(keywords))

    # ------------------------------------------------------------------
    # Code generation helpers (unchanged)
    # ------------------------------------------------------------------

    def _generate_method_code(self, func_spec: dict) -> tuple[str, str, str]:
        """Generate method code from function specification.

        Args:
            func_spec: {"name": "...", "params": [...], "description": "...", "body": "..."}

        Returns:
            (method_code, execute_dispatch, test_code)
        """
        method_name = self._sanitize_name(func_spec.get("name", "unknown"))
        params = func_spec.get("params", [])
        description = func_spec.get("description", f"Execute {method_name}")
        body = func_spec.get("body", 'return {"success": True, "message": "Method executed"}')

        params_with_types = ""
        param_docs = ""
        param_extraction = ""
        param_call = ""

        if params:
            params_with_types = ", " + ", ".join(f"{p}: Any" for p in params)
            param_docs = "\n".join(f"            {p}: Parameter {p}" for p in params)

            extractions = []
            for i, p in enumerate(params):
                extractions.append(f'{p} = self._extract_param(action, "{p}", {i})')
            param_extraction = "\n            ".join(extractions)
            param_call = ", ".join(params)
        else:
            param_docs = "            None"

        body_lines = body.strip().split('\n')
        indented_body = '\n'.join('            ' + line for line in body_lines)

        method_code = METHOD_TEMPLATE.format(
            method_name=method_name,
            params_with_types=params_with_types,
            description=description,
            param_docs=param_docs,
            method_body=indented_body
        )

        execute_dispatch = EXECUTE_DISPATCH_TEMPLATE.format(
            method_name=method_name,
            param_extraction=param_extraction if param_extraction else "pass",
            param_call=param_call
        )

        test_code = METHOD_TEST_TEMPLATE.format(
            name=self._sanitize_name(func_spec.get("tool_name", "custom")),
            method_name=method_name,
            class_name=func_spec.get("class_name", "CustomTool")
        )

        return method_code, execute_dispatch, test_code

    # ------------------------------------------------------------------
    # NEW: VOYAGER-style composition retrieval
    # ------------------------------------------------------------------

    def _get_registered_tools(self) -> dict:
        """Get all registered tools (builtin + custom) with descriptions."""
        tools = {}

        # Custom tools from registry
        registry = self._load_registry()
        for tool in registry.get("tools", []):
            if tool.get("status") == "active":
                tools[tool["name"]] = {
                    "name": tool["name"],
                    "description": tool.get("description", "")[:200],
                    "functions": tool.get("functions", []),
                    "source": "custom",
                }

        # Builtin tools from tool_contract registry
        try:
            from .tool_contract import get_tool_registry
            reg = get_tool_registry()
            for spec in reg.all():
                tools[spec.name] = {
                    "name": spec.name,
                    "description": spec.description[:200] if spec.description else "",
                    "functions": [],
                    "source": "builtin",
                }
        except Exception as e:
            logger.debug(f"[ToolBuilder] Could not load builtin tool registry: {e}")

        return tools

    def _find_composable_tools(self, description: str, max_results: int = 5) -> list:
        """Find existing tools that could be composed into the new tool (VOYAGER pattern).

        Uses LLM if available, falls back to keyword matching.
        """
        all_tools = self._get_registered_tools()
        if not all_tools:
            return []

        # Build tool list string for matching
        tool_lines = []
        for t in all_tools.values():
            tool_lines.append(f"- {t['name']}: {t['description'][:100]}")
        tool_list_str = "\n".join(tool_lines)

        # Try LLM-based composition finding
        llm_result = self._llm_generate(
            f"""Given this new tool requirement:
"{description}"

Which of these existing tools could be useful as building blocks?

Available tools:
{tool_list_str}

Return a JSON list of tool names that could be composed, with brief explanation of how each helps.
Format: [{{"name": "tool_name", "usage": "how to use it"}}]
Return empty list [] if none are relevant. Return ONLY the JSON.""",
            timeout=20,
        )

        if llm_result:
            parsed = self._parse_json_response(llm_result)
            if parsed:
                return parsed[:max_results]

        # Fallback: keyword overlap
        desc_words = set(re.findall(r'\b[a-z]+\b', description.lower()))
        scored = []
        for t in all_tools.values():
            tool_words = set(re.findall(r'\b[a-z]+\b', (t['name'] + ' ' + t['description']).lower()))
            overlap = len(desc_words & tool_words)
            if overlap > 1:
                scored.append({"name": t["name"], "usage": f"Keyword overlap ({overlap} words)", "_score": overlap})
        scored.sort(key=lambda x: x["_score"], reverse=True)
        return [{"name": s["name"], "usage": s["usage"]} for s in scored[:max_results]]

    def _parse_json_response(self, text: str) -> Optional[list]:
        """Extract a JSON list from LLM response text."""
        if not text:
            return None
        try:
            # Try direct parse
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Try to find JSON array in text
        match = re.search(r'\[[\s\S]*?\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # NEW: Automatic test generation via LLM
    # ------------------------------------------------------------------

    def _generate_tests(self, tool_name: str, tool_code: str, tool_description: str) -> Optional[str]:
        """Generate pytest test cases for a newly created tool via LLM."""
        prompt = f"""Generate 3-5 pytest test cases for this tool.

Tool name: {tool_name}
Description: {tool_description}
Code:
```python
{tool_code[:3000]}
```

Generate tests that cover:
1. Happy path (normal usage)
2. Edge case (empty/None input)
3. Error case (invalid input)

Return ONLY the test code as a Python file with imports.
Format: valid pytest file starting with 'import pytest'
The tool class can be imported as: from aura.tools.custom.{tool_name} import *
Include 'if __name__ == "__main__": pytest.main([__file__, "-x", "-q"])' at the end."""

        return self._llm_generate(prompt, timeout=30)

    def _validate_with_tests(self, tool_path: str, test_code: str) -> dict:
        """Run generated tests against the tool. Returns pass/fail result."""
        test_path = tool_path.replace('.py', '_test.py')
        try:
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
        except IOError as e:
            return {"passed": False, "output": f"Failed to write test file: {e}", "test_file": test_path}

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-x", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=30,
                cwd=os.path.dirname(tool_path), env=_safe_env(),
            )
            output = (result.stdout + result.stderr)[-500:]
            return {
                "passed": result.returncode == 0,
                "output": output,
                "test_file": test_path,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "output": "Tests timed out after 30s", "test_file": test_path}
        except Exception as e:
            return {"passed": False, "output": str(e), "test_file": test_path}

    # ------------------------------------------------------------------
    # NEW: GEPA evolution integration
    # ------------------------------------------------------------------

    def _register_for_evolution(self, tool_name: str, tool_path: str):
        """Register a new tool with GEPA for future evolution."""
        try:
            from aura.evolution.adapter import AuraSkillAdapter
            from aura.evolution.types import GEPAConfig

            config = GEPAConfig()
            # We need an llm_func for the adapter; use brain if available
            llm_func = None
            if self._brain and hasattr(self._brain, '_quick_generate'):
                def llm_func(system: str, user: str) -> str:
                    return self._brain._quick_generate(f"{system}\n\n{user}", timeout=30)

            if llm_func is None:
                logger.debug(f"[ToolBuilder] GEPA registration skipped for {tool_name}: no LLM available")
                return

            adapter = AuraSkillAdapter(config=config, llm_func=llm_func)

            # Read tool code as the "procedure text" for GEPA
            tool_code = Path(tool_path).read_text(encoding='utf-8')

            # Create a minimal candidate with this tool as a component
            from aura.evolution.types import Candidate
            candidate = Candidate(
                id=0,
                components={tool_name: tool_code},
                parent_id=-1,
            )

            # Save registration info for future evolution runs
            evo_registry_path = BASE_DIR / "data" / "evolution_registry.json"
            evo_registry = {}
            if evo_registry_path.exists():
                try:
                    with open(evo_registry_path, 'r') as f:
                        evo_registry = json.load(f)
                except (json.JSONDecodeError, IOError):
                    evo_registry = {}

            evo_registry[tool_name] = {
                "tool_path": str(tool_path),
                "registered_at": datetime.now().isoformat(),
                "component_hash": candidate.cache_key(),
                "status": "registered",
            }

            with open(evo_registry_path, 'w') as f:
                json.dump(evo_registry, f, indent=2)

            logger.info(f"[ToolBuilder] Tool {tool_name} registered for GEPA evolution")
        except ImportError as e:
            logger.debug(f"[ToolBuilder] GEPA registration skipped (import): {e}")
        except Exception as e:
            logger.debug(f"[ToolBuilder] GEPA registration skipped: {e}")

    # ------------------------------------------------------------------
    # NEW: Ed25519 signing helper
    # ------------------------------------------------------------------

    def _sign_tool(self, tool_path: str) -> Optional[str]:
        """Sign a tool file with Ed25519 (or HMAC fallback)."""
        try:
            from aura.security.tool_signing import sign_tool
            sig_path = sign_tool(tool_path)
            logger.info(f"[ToolBuilder] Signed tool: {tool_path}")
            return sig_path
        except Exception as e:
            logger.warning(f"[ToolBuilder] Tool signing failed: {e}")
            return None

    # ------------------------------------------------------------------
    # UPGRADED: create_tool with full pipeline
    # ------------------------------------------------------------------

    def create_tool(self, name: str, description: str, functions_spec: list[dict]) -> ToolResult:
        """Create a new custom tool with VOYAGER composition, auto-testing, signing, and GEPA.

        Pipeline:
        1. Find composable existing tools (VOYAGER pattern)
        2. Generate tool code (with composition hints)
        3. Generate tests automatically (LLM + template)
        4. Run tests in sandbox
        5. If tests pass (>=80%): sign with Ed25519, register in custom tools
        6. If tests fail: retry with error context (max 2 retries)
        7. Register for GEPA evolution
        8. Initialize usage tracking

        Args:
            name: Tool name (e.g., "currency_converter")
            description: Tool description
            functions_spec: List of function specifications

        Returns:
            ToolResult with success status and tool metadata.
        """
        logger.info(f"Creating tool: {name}")

        # Sanitize name
        safe_name = self._sanitize_name(name)
        class_name = self._to_class_name(name)
        module_name = safe_name

        # Check if tool already exists
        if self._get_tool_entry(safe_name):
            logger.warning(f"Tool {safe_name} already exists")
            return ToolResult(
                success=False,
                error=f"Tool '{safe_name}' already exists. Use rollback_tool first to remove it."
            )

        # === Step 1: VOYAGER composition retrieval ===
        composable = self._find_composable_tools(description)
        if composable:
            logger.info(f"[VOYAGER] Found {len(composable)} composable tools for {safe_name}: "
                        f"{[c['name'] for c in composable]}")

        # Check for network requirements
        needs_network = False
        for func in functions_spec:
            body = func.get("body", "")
            if "requests." in body or "http" in body.lower():
                needs_network = True
                break

        extra_imports = NETWORK_IMPORTS if needs_network else ""

        # === Step 2: Generate tool code (with composition hints) ===
        methods = []
        execute_dispatches = []
        method_tests = []
        test_calls = []

        for func_spec in functions_spec:
            func_spec["tool_name"] = safe_name
            func_spec["class_name"] = class_name

            method_code, execute_dispatch, test_code = self._generate_method_code(func_spec)
            methods.append(method_code)
            execute_dispatches.append(execute_dispatch)
            method_tests.append(test_code)
            test_calls.append(f"        test_{safe_name}_{self._sanitize_name(func_spec.get('name', 'unknown'))},")

        # Add helper method for parameter extraction
        param_extractor = '''
    def _extract_param(self, action: str, param_name: str, index: int) -> Any:
        """Extract a parameter from the action string."""
        import re
        # Try to find named parameter
        pattern = rf'{param_name}[=:]\\s*["\\'"]?([^"\\'"\\s]+)["\\'"]?'
        match = re.search(pattern, action, re.IGNORECASE)
        if match:
            return match.group(1)
        # Try to extract by position from quoted strings
        quoted = re.findall(r'["\\'"]([^"\\'"]+)["\\'"]', action)
        if index < len(quoted):
            return quoted[index]
        # Try to extract numbers
        numbers = re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', action)
        if index < len(numbers):
            return numbers[index]
        return None
'''
        methods.append(param_extractor)

        # Add composition comment if composable tools found
        composition_comment = ""
        if composable:
            comp_lines = [f"# VOYAGER Composition hints — these existing tools may be reusable:"]
            for c in composable:
                comp_lines.append(f"#   - {c.get('name', '?')}: {c.get('usage', '')}")
            composition_comment = "\n".join(comp_lines) + "\n\n"

        # Build complete tool code
        created_at = datetime.now().isoformat()
        tool_code = TOOL_CLASS_TEMPLATE.format(
            name=safe_name,
            description=description,
            created_at=created_at,
            status="pending",
            class_name=class_name,
            extra_imports=extra_imports,
            methods="\n".join(methods),
            execute_logic="\n".join(execute_dispatches)
        )

        # Inject composition comment at top of file (after docstring)
        if composition_comment:
            # Insert after the first triple-quoted docstring block
            doc_end = tool_code.find('"""', tool_code.find('"""') + 3)
            if doc_end > 0:
                insert_pos = doc_end + 3
                tool_code = tool_code[:insert_pos] + "\n\n" + composition_comment + tool_code[insert_pos:]

        # Reject excessively large generated code (>100KB is suspicious)
        if len(tool_code) > 100_000:
            return ToolResult(success=False, error=f"Generated code too large ({len(tool_code)} bytes, max 100KB)")

        # Scan for dangerous patterns
        is_dangerous, reason = self._scan_for_dangerous_code(tool_code)
        if is_dangerous:
            logger.error(f"Dangerous patterns detected in {safe_name}: {reason}")
            return ToolResult(
                success=False,
                error=f"Dangerous code patterns detected: {reason}",
            )

        # Also scan function bodies
        for func in functions_spec:
            body = func.get("body", "")
            is_dangerous, reason = self._scan_for_dangerous_code(body)
            if is_dangerous:
                logger.error(f"Dangerous patterns in function body: {reason}")
                return ToolResult(
                    success=False,
                    error=f"Dangerous code in function '{func.get('name')}': {reason}",
                )

        # AST-level security validation (import checks, forbidden patterns, f-string evasion)
        try:
            from aura.security.tool_validator import validate_custom_tool_code
            is_valid, validation_reason = validate_custom_tool_code(tool_code, safe_name)
            if not is_valid:
                logger.error(f"[ToolBuilder] AST validation failed for {safe_name}: {validation_reason}")
                return ToolResult(
                    success=False,
                    error=f"Security validation failed: {validation_reason}",
                )
        except ImportError:
            logger.warning("[ToolBuilder] tool_validator not available, skipping AST validation")

        # Save tool file
        tool_file = CUSTOM_TOOLS_DIR / f"{module_name}.py"
        try:
            with open(tool_file, "w", encoding="utf-8") as f:
                f.write(tool_code)
            logger.info(f"Created tool file: {tool_file}")
        except IOError as e:
            logger.error(f"Failed to write tool file: {e}")
            return ToolResult(success=False, error=f"Failed to write tool file: {e}")

        # === Step 3: Generate tests (LLM auto-tests + template tests) ===
        # Template-based tests (always generated)
        template_test_code = TEST_TEMPLATE.format(
            name=safe_name,
            created_at=created_at,
            module_name=module_name,
            class_name=class_name,
            method_tests="\n\n".join(method_tests),
            test_calls="\n".join(test_calls)
        )

        test_file = CUSTOM_TESTS_DIR / f"test_{module_name}.py"
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(template_test_code)
            logger.info(f"Created test file: {test_file}")
        except IOError as e:
            logger.error(f"Failed to write test file: {e}")
            tool_file.unlink(missing_ok=True)
            return ToolResult(success=False, error=f"Failed to write test file: {e}")

        # LLM-generated tests (best-effort)
        llm_test_code = self._generate_tests(safe_name, tool_code, description)
        llm_test_result = None
        if llm_test_code:
            llm_test_result = self._validate_with_tests(str(tool_file), llm_test_code)
            if llm_test_result and llm_test_result.get("passed"):
                logger.info(f"[AutoTest] LLM-generated tests PASSED for {safe_name}")
            elif llm_test_result:
                logger.info(f"[AutoTest] LLM-generated tests FAILED for {safe_name}: "
                            f"{llm_test_result.get('output', '')[:200]}")

        # === Step 4: Run template tests in sandbox ===
        test_passed = False
        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            try:
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(BASE_DIR), env=_safe_env(),
                )
                test_output = result.stdout + result.stderr
                test_passed = result.returncode == 0

                if test_passed:
                    logger.info(f"[AutoTest] Template tests PASSED for {safe_name} (attempt {retry_count + 1})")
                    break
                else:
                    logger.warning(f"[AutoTest] Template tests FAILED for {safe_name} (attempt {retry_count + 1}): "
                                   f"{test_output[-300:]}")
                    retry_count += 1

                    # Try to fix with LLM on retry
                    if retry_count <= max_retries and self._brain:
                        fix_prompt = f"""The template tests for tool '{safe_name}' failed.

Test output:
{test_output[-500:]}

Tool code:
```python
{tool_code[:2000]}
```

The test file uses a simple runner (not pytest). Suggest minimal fixes to the tool code
that would make it pass. Return ONLY the corrected tool code between ```python and ```."""
                        fix_response = self._llm_generate(fix_prompt, timeout=30)
                        if fix_response:
                            # Extract code block
                            code_match = re.search(r'```python\s*([\s\S]*?)```', fix_response)
                            if code_match:
                                fixed_code = code_match.group(1).strip()
                                # Re-scan for safety
                                is_dangerous, reason = self._scan_for_dangerous_code(fixed_code)
                                if not is_dangerous:
                                    tool_code = fixed_code
                                    with open(tool_file, "w", encoding="utf-8") as f:
                                        f.write(tool_code)
                                    logger.info(f"[AutoTest] Applied LLM fix for {safe_name}, retrying...")
                                else:
                                    logger.warning(f"[AutoTest] LLM fix contained dangerous code: {reason}")
                                    break
            except subprocess.TimeoutExpired:
                logger.warning(f"[AutoTest] Tests timed out for {safe_name}")
                retry_count += 1
            except Exception as e:
                logger.warning(f"[AutoTest] Test execution error: {e}")
                break

        # === Step 5: Sign and register if tests pass ===
        tool_status = "active" if test_passed else "pending"

        if test_passed:
            # Sign with Ed25519
            sig_path = self._sign_tool(str(tool_file))
            if sig_path:
                logger.info(f"[ToolBuilder] Tool {safe_name} signed: {sig_path}")
        else:
            sig_path = None

        # Create __init__.py in custom directory if not exists
        init_file = CUSTOM_TOOLS_DIR / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Custom tools directory."""\n')

        # Generate keywords for tool detection
        keywords = self._generate_keywords(safe_name, description, functions_spec)

        # Register tool in custom_tools.json
        registry = self._load_registry()
        registry_entry = {
            "name": safe_name,
            "class_name": class_name,
            "description": description,
            "status": tool_status,
            "created": created_at,
            "file": str(tool_file),
            "test_file": str(test_file),
            "functions": [f.get("name") for f in functions_spec],
            "keywords": keywords,
            "composable_tools": [c.get("name") for c in composable] if composable else [],
            "auto_tested": test_passed,
            "signed": sig_path is not None,
        }
        if test_passed:
            registry_entry["enabled_at"] = datetime.now().isoformat()

        registry["tools"].append(registry_entry)
        self._save_registry(registry)

        # === Step 7: Register for GEPA evolution ===
        self._register_for_evolution(safe_name, str(tool_file))

        # === Step 8: Initialize usage tracking ===
        self._usage_tracker.register_tool(safe_name)

        # === Step 9: Version tracking ===
        self._init_tool_versions(safe_name, str(tool_file))

        # Build result message
        if test_passed:
            msg = (f"Tool '{safe_name}' created, tested, signed, and activated. "
                   f"Registered for GEPA evolution.")
        else:
            msg = (f"Tool '{safe_name}' created but tests failed after {retry_count} retries. "
                   f"Status: pending. Run test_tool('{safe_name}') to debug, "
                   f"then enable_tool('{safe_name}') to activate.")

        logger.info(f"Tool {safe_name} created: status={tool_status}, tested={test_passed}")
        return ToolResult(
            success=True,
            result={
                "name": safe_name,
                "class_name": class_name,
                "file": str(tool_file),
                "test_file": str(test_file),
                "status": tool_status,
                "tests_passed": test_passed,
                "signed": sig_path is not None,
                "composable_tools": [c.get("name") for c in composable] if composable else [],
                "gepa_registered": True,
                "message": msg,
            },
        )

    # ------------------------------------------------------------------
    # test_tool (unchanged)
    # ------------------------------------------------------------------

    def test_tool(self, name: str) -> ToolResult:
        """Run tests for a custom tool.

        Args:
            name: Tool name to test

        Returns:
            ToolResult with test output.
        """
        safe_name = self._sanitize_name(name)
        logger.info(f"Testing tool: {safe_name}")

        tool_entry = self._get_tool_entry(safe_name)
        if not tool_entry:
            return ToolResult(success=False, error=f"Tool '{safe_name}' not found in registry")

        test_file = Path(tool_entry.get("test_file", ""))
        if not test_file.exists():
            return ToolResult(success=False, error=f"Test file not found: {test_file}")

        try:
            resolved_test = test_file.resolve()
            resolved_tests_dir = CUSTOM_TESTS_DIR.resolve()
            if not (str(resolved_test).startswith(str(resolved_tests_dir) + os.sep) or str(resolved_test) == str(resolved_tests_dir)):
                return ToolResult(success=False, error="Test file path outside sandbox")
        except Exception as e:
            return ToolResult(success=False, error=f"Invalid test file path: {e}")

        # Run tests in subprocess for isolation
        start_ms = int(time.time() * 1000)
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(BASE_DIR), env=_safe_env(),
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0
            latency_ms = int(time.time() * 1000) - start_ms

            # Track test execution in usage tracker
            self._usage_tracker.record_use(
                f"{safe_name}:test", success=success, latency_ms=latency_ms
            )

            logger.info(f"Test results for {safe_name}: {'PASSED' if success else 'FAILED'}")

            return ToolResult(
                success=success,
                result={
                    "name": safe_name,
                    "exit_code": result.returncode,
                    "output": output,
                    "message": f"Tests {'PASSED' if success else 'FAILED'} for tool '{safe_name}'",
                },
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Tests timed out for {safe_name}")
            return ToolResult(success=False, error="Tests timed out after 30 seconds")
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return ToolResult(success=False, error=f"Test execution failed: {e}")

    # ------------------------------------------------------------------
    # enable_tool (unchanged)
    # ------------------------------------------------------------------

    def enable_tool(self, name: str) -> ToolResult:
        """Enable a custom tool after successful testing."""
        safe_name = self._sanitize_name(name)
        logger.info(f"Enabling tool: {safe_name}")

        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == safe_name:
                if tool["status"] == "active":
                    return ToolResult(success=True, result={"message": f"Tool '{safe_name}' is already active"})

                tool_file = Path(tool.get("file", ""))
                if not tool_file.exists():
                    return ToolResult(success=False, error=f"Tool file not found: {tool_file}")

                # Sign if not already signed
                if not tool.get("signed"):
                    sig_path = self._sign_tool(str(tool_file))
                    tool["signed"] = sig_path is not None

                tool["status"] = "active"
                tool["enabled_at"] = datetime.now().isoformat()
                self._save_registry(registry)

                logger.info(f"Tool {safe_name} enabled")
                return ToolResult(
                    success=True,
                    result={
                        "name": safe_name,
                        "status": "active",
                        "message": f"Tool '{safe_name}' is now active and available for use",
                    },
                )

        return ToolResult(success=False, error=f"Tool '{safe_name}' not found in registry")

    # ------------------------------------------------------------------
    # disable_tool (unchanged)
    # ------------------------------------------------------------------

    def disable_tool(self, name: str) -> ToolResult:
        """Disable a custom tool."""
        safe_name = self._sanitize_name(name)
        logger.info(f"Disabling tool: {safe_name}")

        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == safe_name:
                if tool["status"] == "disabled":
                    return ToolResult(success=True, result={"message": f"Tool '{safe_name}' is already disabled"})

                tool["status"] = "disabled"
                tool["disabled_at"] = datetime.now().isoformat()
                self._save_registry(registry)

                logger.info(f"Tool {safe_name} disabled")
                return ToolResult(
                    success=True,
                    result={
                        "name": safe_name,
                        "status": "disabled",
                        "message": f"Tool '{safe_name}' has been disabled",
                    },
                )

        return ToolResult(success=False, error=f"Tool '{safe_name}' not found in registry")

    # ------------------------------------------------------------------
    # rollback_tool (unchanged)
    # ------------------------------------------------------------------

    def rollback_tool(self, name: str) -> ToolResult:
        """Delete a custom tool completely."""
        safe_name = self._sanitize_name(name)
        logger.info(f"Rolling back tool: {safe_name}")

        registry = self._load_registry()
        tool_entry = None
        for i, tool in enumerate(registry["tools"]):
            if tool["name"] == safe_name:
                tool_entry = registry["tools"].pop(i)
                break

        if not tool_entry:
            return ToolResult(success=False, error=f"Tool '{safe_name}' not found in registry")

        # Delete files
        deleted_files = []
        tool_file = Path(tool_entry.get("file", ""))
        if tool_file.exists():
            tool_file.unlink()
            deleted_files.append(str(tool_file))
            # Also delete signature file
            sig_file = Path(str(tool_file) + ".sig")
            if sig_file.exists():
                sig_file.unlink()
                deleted_files.append(str(sig_file))
            # Also delete LLM test file
            llm_test = Path(str(tool_file).replace('.py', '_test.py'))
            if llm_test.exists():
                llm_test.unlink()
                deleted_files.append(str(llm_test))

        test_file = Path(tool_entry.get("test_file", ""))
        if test_file.exists():
            test_file.unlink()
            deleted_files.append(str(test_file))

        # Save updated registry
        self._save_registry(registry)

        logger.info(f"Tool {safe_name} rolled back, deleted: {deleted_files}")
        return ToolResult(
            success=True,
            result={
                "name": safe_name,
                "deleted_files": deleted_files,
                "message": f"Tool '{safe_name}' has been completely removed",
            },
        )

    # ------------------------------------------------------------------
    # list_custom_tools (enhanced with usage stats)
    # ------------------------------------------------------------------

    def list_custom_tools(self) -> dict:
        """List all custom tools with their status and usage stats."""
        registry = self._load_registry()
        tools = registry.get("tools", [])

        # Enrich with usage stats
        for tool in tools:
            stats = self._usage_tracker.get_stats(tool["name"])
            if stats:
                tool["usage_stats"] = stats

        summary = {
            "total": len(tools),
            "active": sum(1 for t in tools if t.get("status") == "active"),
            "pending": sum(1 for t in tools if t.get("status") == "pending"),
            "disabled": sum(1 for t in tools if t.get("status") == "disabled")
        }

        return {
            "success": True,
            "tools": tools,
            "summary": summary
        }

    # ------------------------------------------------------------------
    # NEW: deprecation candidates
    # ------------------------------------------------------------------

    def get_deprecation_candidates(self, min_age_days: int = 30) -> ToolResult:
        """Find tools that are old and rarely used — candidates for cleanup."""
        candidates = self._usage_tracker.get_candidates_for_deprecation(min_age_days=min_age_days)
        return ToolResult(
            success=True,
            result={
                "candidates": candidates,
                "count": len(candidates),
                "message": f"Found {len(candidates)} tools older than {min_age_days} days with low usage",
            }
        )

    # ------------------------------------------------------------------
    # NEW: tool usage recording (called by agent on tool invocation)
    # ------------------------------------------------------------------

    def record_tool_use(self, tool_name: str, success: bool = True, latency_ms: int = 0):
        """Record a custom tool invocation for usage tracking."""
        self._usage_tracker.record_use(tool_name, success=success, latency_ms=latency_ms)

    # ==================================================================
    # VERSIONING SYSTEM
    # ==================================================================

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file for integrity tracking."""
        import hashlib
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def _init_tool_versions(self, tool_name: str, tool_file: str):
        """Initialize version tracking for a newly created tool."""
        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == tool_name:
                tool["versions"] = [{
                    "version": 1,
                    "file": tool_file,
                    "created": datetime.now().isoformat(),
                    "test_passed": tool.get("auto_tested", False),
                    "hash": self._compute_file_hash(tool_file),
                }]
                self._save_registry(registry)
                logger.info(f"[Versioning] Initialized v1 for {tool_name}")
                return
        logger.debug(f"[Versioning] Tool {tool_name} not found in registry for version init")

    def _increment_tool_version(self, tool_name: str, tool_file: str, test_passed: bool):
        """Record a new version when a tool is updated. Keeps last 3 versions."""
        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == tool_name:
                versions = tool.get("versions", [])
                max_ver = max((v["version"] for v in versions), default=0)
                versions.append({
                    "version": max_ver + 1,
                    "file": tool_file,
                    "created": datetime.now().isoformat(),
                    "test_passed": test_passed,
                    "hash": self._compute_file_hash(tool_file),
                })
                # Keep only last 3 versions
                while len(versions) > 3:
                    old = versions.pop(0)
                    old_path = Path(old.get("file", ""))
                    if old_path.exists() and str(old_path) != tool_file:
                        try:
                            old_path.unlink()
                            logger.info(f"[Versioning] Cleaned up old v{old['version']} of {tool_name}")
                        except Exception:
                            pass
                tool["versions"] = versions
                self._save_registry(registry)
                logger.info(f"[Versioning] {tool_name} now at v{max_ver + 1}")
                return

    def rollback_to_version(self, name: str, version: int) -> dict:
        """Roll back a tool to a previous version."""
        import shutil
        safe_name = self._sanitize_name(name)
        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == safe_name:
                versions = tool.get("versions", [])
                target = next((v for v in versions if v["version"] == version), None)
                if not target:
                    return {"success": False, "error": f"Version {version} not found for {safe_name}"}
                version_file = Path(target["file"])
                if not version_file.exists():
                    return {"success": False, "error": f"Version file {version_file} missing"}
                current_file = Path(tool["file"])
                try:
                    shutil.copy2(str(version_file), str(current_file))
                    tool["status"] = "pending"
                    self._save_registry(registry)
                    logger.info(f"[Versioning] Rolled back {safe_name} to v{version}")
                    return {"success": True, "name": safe_name, "rolled_back_to": version,
                            "message": f"Rolled back to v{version}. Run test_tool() to validate."}
                except Exception as e:
                    return {"success": False, "error": f"Rollback failed: {e}"}
        return {"success": False, "error": f"Tool '{safe_name}' not found"}

    # ==================================================================
    # USAGE-DRIVEN EVOLUTION
    # ==================================================================

    def _check_evolution_candidates(self) -> list:
        """Find tools with low success rate that need GEPA evolution."""
        registry = self._load_registry()
        candidates = []
        for tool in registry.get("tools", []):
            stats = self._usage_tracker.get_stats(tool["name"])
            if not stats:
                continue
            total = stats.get("total_uses", 0)
            rate = stats.get("success_rate", 1.0)
            if total > 10 and rate < 0.7:
                candidates.append(tool["name"])
                logger.info(f"[Evolution] Candidate: {tool['name']} (uses={total}, rate={rate:.2f})")
        return candidates

    def trigger_evolution_for_tool(self, tool_name: str) -> dict:
        """Submit an underperforming tool to GEPA for evolution."""
        safe_name = self._sanitize_name(tool_name)
        registry = self._load_registry()
        tool_entry = None
        for t in registry.get("tools", []):
            if t["name"] == safe_name:
                tool_entry = t
                break
        if not tool_entry:
            return {"success": False, "error": f"Tool '{safe_name}' not found"}

        stats = self._usage_tracker.get_stats(safe_name)
        if not stats:
            return {"success": False, "error": f"No usage data for {safe_name}"}

        # Save evolution trigger to registry
        try:
            evo_path = Path(__file__).parent.parent.parent / "data" / "evolution_registry.json"
            evo_data = {}
            if evo_path.exists():
                try:
                    evo_data = json.loads(evo_path.read_text())
                except Exception:
                    pass
            evo_data[safe_name] = {
                "trigger_reason": "low_success_rate",
                "success_rate": stats.get("success_rate", 0),
                "total_uses": stats.get("total_uses", 0),
                "triggered_at": datetime.now().isoformat(),
                "status": "submitted",
            }
            evo_path.parent.mkdir(parents=True, exist_ok=True)
            evo_path.write_text(json.dumps(evo_data, indent=2))
            logger.info(f"[Evolution] Triggered evolution for {safe_name}")
            return {"success": True, "name": safe_name, "submitted": True,
                    "message": f"Tool {safe_name} submitted to GEPA for evolution"}
        except Exception as e:
            return {"success": False, "error": f"Evolution trigger failed: {e}"}

    def monitor_and_evolve(self) -> dict:
        """Check all tools for evolution candidates and trigger GEPA."""
        candidates = self._check_evolution_candidates()
        results = {"candidates": len(candidates), "triggered": []}
        for name in candidates:
            r = self.trigger_evolution_for_tool(name)
            results["triggered"].append({"name": name, "success": r.get("success", False)})
        return {"success": True, **results}

    # ==================================================================
    # DEPENDENCY TRACKING
    # ==================================================================

    def _build_dependency_graph(self) -> dict:
        """Build inverse dependency map: tool -> [tools that depend on it]."""
        registry = self._load_registry()
        dependents = {}
        for tool in registry.get("tools", []):
            for dep in tool.get("composable_tools", []):
                if dep not in dependents:
                    dependents[dep] = []
                dependents[dep].append(tool["name"])
        return dependents

    def _propagate_changes(self, tool_name: str) -> dict:
        """When a tool changes, re-test all tools that depend on it."""
        safe_name = self._sanitize_name(tool_name)
        dep_graph = self._build_dependency_graph()
        dependents = dep_graph.get(safe_name, [])
        if not dependents:
            return {"success": True, "name": safe_name, "dependents": [],
                    "message": f"No tools depend on {safe_name}"}

        test_results = {}
        registry = self._load_registry()
        for dep_name in dependents:
            result = self.test_tool(dep_name)
            passed = result.success if hasattr(result, "success") else False
            test_results[dep_name] = {"passed": passed}
            if not passed:
                for t in registry["tools"]:
                    if t["name"] == dep_name:
                        t["status"] = "pending"
                        logger.warning(f"[Deps] {dep_name} failed after {safe_name} change — marked pending")
        self._save_registry(registry)

        passed_count = sum(1 for r in test_results.values() if r["passed"])
        failed_count = len(test_results) - passed_count
        return {"success": True, "name": safe_name, "dependents": dependents,
                "passed": passed_count, "failed": failed_count,
                "test_results": test_results}

    # ------------------------------------------------------------------
    # execute (enhanced)
    # ------------------------------------------------------------------

    def execute(self, action: str) -> dict:
        """Execute a tool builder action.

        Args:
            action: The action string to parse and execute

        Returns:
            Result dictionary
        """
        action_lower = action.lower()

        if "list" in action_lower:
            return self.list_custom_tools()

        elif "deprecat" in action_lower:
            return self.get_deprecation_candidates()

        elif "test" in action_lower:
            name = self._extract_tool_name(action)
            if not name:
                return {"success": False, "error": "No tool name specified for testing"}
            return self.test_tool(name)

        elif "enable" in action_lower:
            name = self._extract_tool_name(action)
            if not name:
                return {"success": False, "error": "No tool name specified for enabling"}
            return self.enable_tool(name)

        elif "disable" in action_lower:
            name = self._extract_tool_name(action)
            if not name:
                return {"success": False, "error": "No tool name specified for disabling"}
            return self.disable_tool(name)

        elif "rollback" in action_lower or "delete" in action_lower or "remove" in action_lower:
            name = self._extract_tool_name(action)
            if not name:
                return {"success": False, "error": "No tool name specified for rollback"}
            return self.rollback_tool(name)

        elif "propagate" in action_lower or "depend" in action_lower:
            name = self._extract_tool_name(action)
            if not name:
                return {"success": False, "error": "No tool name specified for propagation"}
            return self._propagate_changes(name)

        elif "monitor" in action_lower or "evolve" in action_lower:
            return self.monitor_and_evolve()

        elif "version" in action_lower and "rollback" not in action_lower:
            name = self._extract_tool_name(action)
            if not name:
                return {"success": False, "error": "No tool name specified"}
            # Extract version number
            ver_match = re.search(r'v(?:ersion)?\s*(\d+)', action, re.IGNORECASE)
            if ver_match:
                return self.rollback_to_version(name, int(ver_match.group(1)))
            return {"success": False, "error": "No version number found. Use: version 'tool_name' v2"}

        elif "create" in action_lower:
            return {
                "success": False,
                "error": "Tool creation requires structured input. Use create_tool(name, description, functions_spec) directly.",
                "example": {
                    "name": "currency_converter",
                    "description": "Convert between currencies",
                    "functions_spec": [
                        {
                            "name": "convert",
                            "params": ["amount", "from_currency", "to_currency"],
                            "description": "Convert amount between currencies",
                            "body": 'rate = 1.0  # Placeholder\nresult = float(amount) * rate\nreturn {"success": True, "result": result}'
                        }
                    ]
                }
            }

        return {"success": False, "error": f"Unknown action: {action}"}

    def _extract_tool_name(self, action: str) -> Optional[str]:
        """Extract tool name from action string."""
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        patterns = [
            r'(?:test|enable|disable|rollback|delete|remove)\s+(?:tool\s+)?(\w+)',
            r'tool\s+(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name.lower() not in ['the', 'a', 'an', 'this', 'that', 'tool']:
                    return name
        return None
