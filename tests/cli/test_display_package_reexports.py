"""Ensure the new display/ package preserves the full public API of the old display.py.

After the #6 split, consumers should keep using `from aura.cli.display import X`.
This test fails loudly if any previously-public name is accidentally dropped.
"""

import importlib


_EXPECTED_PUBLIC_NAMES = [
    "console",
    "show_banner",
    "show_welcome_info",
    "show_status_bar",
    "show_thinking",
    "get_thinking_label",
    "show_tool_call",
    "show_tool_result_inline",
    "show_permission_prompt",
    "show_response",
    "show_response_attribution",
    "show_context_summary",
    "show_error",
    "show_info",
    "show_warning",
    "show_help",
    "StreamingResponse",
    "show_checkpoint_picker",
    "show_rewind_result",
]

_EXPECTED_PRIVATE_NAMES = [
    "_split_for_streaming",
    "_split_into_blocks",
]


def test_display_package_imports_cleanly():
    importlib.import_module("aura.cli.display")


def test_all_public_names_present():
    mod = importlib.import_module("aura.cli.display")
    missing = [n for n in _EXPECTED_PUBLIC_NAMES if not hasattr(mod, n)]
    assert not missing, f"Missing from aura.cli.display: {missing}"


def test_previously_exposed_private_helpers_still_importable():
    mod = importlib.import_module("aura.cli.display")
    missing = [n for n in _EXPECTED_PRIVATE_NAMES if not hasattr(mod, n)]
    assert not missing, f"Missing internal helpers: {missing}"


def test_submodules_are_importable_directly():
    for sub in ("help", "streaming", "checkpoint_picker"):
        importlib.import_module(f"aura.cli.display.{sub}")


def test_streaming_class_is_same_object_across_paths():
    """StreamingResponse re-exported from __init__ must be the same class object
    as the one defined in the submodule — not a copy."""
    from aura.cli.display import StreamingResponse as from_package
    from aura.cli.display.streaming import StreamingResponse as from_submodule
    assert from_package is from_submodule
