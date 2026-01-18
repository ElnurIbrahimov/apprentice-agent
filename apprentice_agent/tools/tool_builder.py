"""Tool Builder - Meta-tool for creating custom tools dynamically.

This tool allows the agent to create new tools at runtime, enabling
self-extension capabilities while maintaining security constraints.
"""

import json
import re
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .tool_template import (
    BLOCKED_PATTERNS,
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


class ToolBuilderTool:
    """Meta-tool for creating, testing, and managing custom tools."""

    def __init__(self):
        self.name = "tool_builder"
        self.description = "Create, test, enable, disable, and manage custom tools"
        self._ensure_registry()

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

    def _scan_for_dangerous_code(self, code: str) -> list[str]:
        """Scan code for dangerous patterns.

        Args:
            code: The Python code to scan

        Returns:
            List of detected dangerous patterns
        """
        dangerous = []
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                dangerous.append(pattern)
        return dangerous

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a tool name to be a valid Python identifier."""
        # Remove non-alphanumeric characters except underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        # Ensure it doesn't start with a number
        if sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized

    def _to_class_name(self, name: str) -> str:
        """Convert tool name to PascalCase class name."""
        words = self._sanitize_name(name).split('_')
        return ''.join(word.capitalize() for word in words if word) + 'Tool'

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

        # Build parameter string for method signature
        params_with_types = ""
        param_docs = ""
        param_extraction = ""
        param_call = ""

        if params:
            params_with_types = ", " + ", ".join(f"{p}: Any" for p in params)
            param_docs = "\n".join(f"            {p}: Parameter {p}" for p in params)

            # Generate parameter extraction from action string
            extractions = []
            for i, p in enumerate(params):
                extractions.append(f'{p} = self._extract_param(action, "{p}", {i})')
            param_extraction = "\n            ".join(extractions)
            param_call = ", ".join(params)
        else:
            param_docs = "            None"

        # Indent body properly
        body_lines = body.strip().split('\n')
        indented_body = '\n'.join('            ' + line for line in body_lines)

        # Generate method
        method_code = METHOD_TEMPLATE.format(
            method_name=method_name,
            params_with_types=params_with_types,
            description=description,
            param_docs=param_docs,
            method_body=indented_body
        )

        # Generate execute dispatch
        execute_dispatch = EXECUTE_DISPATCH_TEMPLATE.format(
            method_name=method_name,
            param_extraction=param_extraction if param_extraction else "pass",
            param_call=param_call
        )

        # Generate test
        test_code = METHOD_TEST_TEMPLATE.format(
            name=self._sanitize_name(func_spec.get("tool_name", "custom")),
            method_name=method_name,
            class_name=func_spec.get("class_name", "CustomTool")
        )

        return method_code, execute_dispatch, test_code

    def create_tool(self, name: str, description: str, functions_spec: list[dict]) -> dict:
        """Create a new custom tool.

        Args:
            name: Tool name (e.g., "currency_converter")
            description: Tool description
            functions_spec: List of function specifications
                [{"name": "convert", "params": ["amount", "from_currency", "to_currency"],
                  "description": "Convert currency", "body": "..."}]

        Returns:
            Result dictionary with success status
        """
        logger.info(f"Creating tool: {name}")

        # Sanitize name
        safe_name = self._sanitize_name(name)
        class_name = self._to_class_name(name)
        module_name = safe_name

        # Check if tool already exists
        if self._get_tool_entry(safe_name):
            logger.warning(f"Tool {safe_name} already exists")
            return {
                "success": False,
                "error": f"Tool '{safe_name}' already exists. Use rollback_tool first to remove it."
            }

        # Check for network requirements
        needs_network = False
        for func in functions_spec:
            body = func.get("body", "")
            if "requests." in body or "http" in body.lower():
                needs_network = True
                break

        extra_imports = NETWORK_IMPORTS if needs_network else ""

        # Generate methods
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

        # Scan for dangerous patterns
        dangerous = self._scan_for_dangerous_code(tool_code)
        if dangerous:
            logger.error(f"Dangerous patterns detected in {safe_name}: {dangerous}")
            return {
                "success": False,
                "error": f"Dangerous code patterns detected: {dangerous}",
                "blocked_patterns": dangerous
            }

        # Also scan function bodies
        for func in functions_spec:
            body = func.get("body", "")
            dangerous = self._scan_for_dangerous_code(body)
            if dangerous:
                logger.error(f"Dangerous patterns in function body: {dangerous}")
                return {
                    "success": False,
                    "error": f"Dangerous code in function '{func.get('name')}': {dangerous}",
                    "blocked_patterns": dangerous
                }

        # Save tool file
        tool_file = CUSTOM_TOOLS_DIR / f"{module_name}.py"
        try:
            with open(tool_file, "w", encoding="utf-8") as f:
                f.write(tool_code)
            logger.info(f"Created tool file: {tool_file}")
        except IOError as e:
            logger.error(f"Failed to write tool file: {e}")
            return {"success": False, "error": f"Failed to write tool file: {e}"}

        # Generate and save test file
        test_code = TEST_TEMPLATE.format(
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
                f.write(test_code)
            logger.info(f"Created test file: {test_file}")
        except IOError as e:
            logger.error(f"Failed to write test file: {e}")
            # Cleanup tool file
            tool_file.unlink(missing_ok=True)
            return {"success": False, "error": f"Failed to write test file: {e}"}

        # Create __init__.py in custom directory if not exists
        init_file = CUSTOM_TOOLS_DIR / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Custom tools directory."""\n')

        # Register tool
        registry = self._load_registry()
        registry["tools"].append({
            "name": safe_name,
            "class_name": class_name,
            "description": description,
            "status": "pending",
            "created": created_at,
            "file": str(tool_file),
            "test_file": str(test_file),
            "functions": [f.get("name") for f in functions_spec]
        })
        self._save_registry(registry)

        logger.info(f"Tool {safe_name} created successfully")
        return {
            "success": True,
            "name": safe_name,
            "class_name": class_name,
            "file": str(tool_file),
            "test_file": str(test_file),
            "status": "pending",
            "message": f"Tool '{safe_name}' created. Run test_tool('{safe_name}') to verify, then enable_tool('{safe_name}') to activate."
        }

    def test_tool(self, name: str) -> dict:
        """Run tests for a custom tool.

        Args:
            name: Tool name to test

        Returns:
            Test results dictionary
        """
        safe_name = self._sanitize_name(name)
        logger.info(f"Testing tool: {safe_name}")

        tool_entry = self._get_tool_entry(safe_name)
        if not tool_entry:
            return {"success": False, "error": f"Tool '{safe_name}' not found in registry"}

        test_file = Path(tool_entry.get("test_file", ""))
        if not test_file.exists():
            return {"success": False, "error": f"Test file not found: {test_file}"}

        # Run tests in subprocess for isolation
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(BASE_DIR)
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0

            logger.info(f"Test results for {safe_name}: {'PASSED' if success else 'FAILED'}")

            return {
                "success": success,
                "name": safe_name,
                "exit_code": result.returncode,
                "output": output,
                "message": f"Tests {'PASSED' if success else 'FAILED'} for tool '{safe_name}'"
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Tests timed out for {safe_name}")
            return {"success": False, "error": "Tests timed out after 30 seconds"}
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {"success": False, "error": f"Test execution failed: {e}"}

    def enable_tool(self, name: str) -> dict:
        """Enable a custom tool after successful testing.

        Args:
            name: Tool name to enable

        Returns:
            Result dictionary
        """
        safe_name = self._sanitize_name(name)
        logger.info(f"Enabling tool: {safe_name}")

        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == safe_name:
                if tool["status"] == "active":
                    return {"success": True, "message": f"Tool '{safe_name}' is already active"}

                # Verify tool file exists
                tool_file = Path(tool.get("file", ""))
                if not tool_file.exists():
                    return {"success": False, "error": f"Tool file not found: {tool_file}"}

                tool["status"] = "active"
                tool["enabled_at"] = datetime.now().isoformat()
                self._save_registry(registry)

                logger.info(f"Tool {safe_name} enabled")
                return {
                    "success": True,
                    "name": safe_name,
                    "status": "active",
                    "message": f"Tool '{safe_name}' is now active and available for use"
                }

        return {"success": False, "error": f"Tool '{safe_name}' not found in registry"}

    def disable_tool(self, name: str) -> dict:
        """Disable a custom tool.

        Args:
            name: Tool name to disable

        Returns:
            Result dictionary
        """
        safe_name = self._sanitize_name(name)
        logger.info(f"Disabling tool: {safe_name}")

        registry = self._load_registry()
        for tool in registry["tools"]:
            if tool["name"] == safe_name:
                if tool["status"] == "disabled":
                    return {"success": True, "message": f"Tool '{safe_name}' is already disabled"}

                tool["status"] = "disabled"
                tool["disabled_at"] = datetime.now().isoformat()
                self._save_registry(registry)

                logger.info(f"Tool {safe_name} disabled")
                return {
                    "success": True,
                    "name": safe_name,
                    "status": "disabled",
                    "message": f"Tool '{safe_name}' has been disabled"
                }

        return {"success": False, "error": f"Tool '{safe_name}' not found in registry"}

    def rollback_tool(self, name: str) -> dict:
        """Delete a custom tool completely.

        Args:
            name: Tool name to delete

        Returns:
            Result dictionary
        """
        safe_name = self._sanitize_name(name)
        logger.info(f"Rolling back tool: {safe_name}")

        registry = self._load_registry()
        tool_entry = None
        for i, tool in enumerate(registry["tools"]):
            if tool["name"] == safe_name:
                tool_entry = registry["tools"].pop(i)
                break

        if not tool_entry:
            return {"success": False, "error": f"Tool '{safe_name}' not found in registry"}

        # Delete files
        deleted_files = []
        tool_file = Path(tool_entry.get("file", ""))
        if tool_file.exists():
            tool_file.unlink()
            deleted_files.append(str(tool_file))

        test_file = Path(tool_entry.get("test_file", ""))
        if test_file.exists():
            test_file.unlink()
            deleted_files.append(str(test_file))

        # Save updated registry
        self._save_registry(registry)

        logger.info(f"Tool {safe_name} rolled back, deleted: {deleted_files}")
        return {
            "success": True,
            "name": safe_name,
            "deleted_files": deleted_files,
            "message": f"Tool '{safe_name}' has been completely removed"
        }

    def list_custom_tools(self) -> dict:
        """List all custom tools with their status.

        Returns:
            Dictionary with tool list
        """
        registry = self._load_registry()
        tools = registry.get("tools", [])

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

        elif "test" in action_lower:
            # Extract tool name
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
        import re
        # Look for quoted name
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Look for name after common keywords
        patterns = [
            r'(?:test|enable|disable|rollback|delete|remove)\s+(?:tool\s+)?(\w+)',
            r'tool\s+(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Exclude common words
                if name.lower() not in ['the', 'a', 'an', 'this', 'that', 'tool']:
                    return name
        return None
