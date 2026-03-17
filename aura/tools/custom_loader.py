"""Dynamically load custom and marketplace tools from aura/tools/custom/."""
import importlib
import inspect
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import of the full AST security validator to avoid circular imports.
_validator = None

def _get_validator():
    """Lazy-load validate_custom_tool_code from the agent module."""
    global _validator
    if _validator is None:
        try:
            from aura.agent import validate_custom_tool_code
            _validator = validate_custom_tool_code
        except ImportError:
            logger.warning("[CustomLoader] Could not import validate_custom_tool_code; custom tool validation unavailable")
            _validator = False  # sentinel: unavailable
    return _validator


def load_custom_tools(custom_dir: Path | None = None) -> dict:
    """Scan custom/ directory and return {tool_name: tool_instance} for valid tools.

    A valid tool class has: name (str attr), description (str attr), execute (method).
    Every file is AST-validated before import to prevent untrusted code execution.
    """
    if custom_dir is None:
        custom_dir = Path(__file__).parent / "custom"

    if not custom_dir.exists():
        return {}

    loaded = {}
    for tool_file in sorted(custom_dir.glob("*.py")):
        if tool_file.name.startswith("_"):
            continue  # Skip __init__.py, __pycache__ etc.

        # SECURITY: Validate tool code before dynamic import
        try:
            code = tool_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"[CustomLoader] Failed to read {tool_file.name}: {e}")
            continue

        validator = _get_validator()
        if validator is False:
            logger.error(f"[CustomLoader] REFUSED to load {tool_file.name}: security validator unavailable")
            continue
        if validator:
            is_valid, validation_msg = validator(code, str(tool_file))
            if not is_valid:
                logger.warning(f"[CustomLoader] BLOCKED {tool_file.name}: {validation_msg}")
                continue

        module_name = f"aura.tools.custom.{tool_file.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.warning(f"[CustomLoader] Failed to import {tool_file.name}: {e}")
            continue

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(module, attr_name)
            if not inspect.isclass(obj):
                continue
            if not (hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "execute")):
                continue

            try:
                instance = obj()
                tool_name = getattr(instance, "name", tool_file.stem)
                loaded[tool_name] = instance
                logger.info(f"[CustomLoader] Loaded custom tool: {tool_name} from {tool_file.name}")
            except Exception as e:
                logger.warning(f"[CustomLoader] Failed to instantiate {attr_name} from {tool_file.name}: {e}")

    return loaded
