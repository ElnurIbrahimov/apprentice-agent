"""Auto-generated tests for custom tool: temperature_converter

Created: 2026-01-18T18:50:47.041375
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from apprentice_agent.tools.custom.temperature_converter import TemperatureConverterTool


def test_temperature_converter_instantiation():
    """Test that the tool can be instantiated."""
    tool = TemperatureConverterTool()
    assert tool.name == "temperature_converter"
    assert tool.description is not None
    print(f"[PASS] TemperatureConverterTool instantiation")


def test_temperature_converter_execute_unknown():
    """Test that unknown actions return error."""
    tool = TemperatureConverterTool()
    result = tool.execute("unknown_action_xyz")
    assert result.get("success") == False
    assert "error" in result
    print(f"[PASS] TemperatureConverterTool handles unknown action")


def test_temperature_converter_celsius_to_fahrenheit():
    """Test celsius_to_fahrenheit method."""
    tool = TemperatureConverterTool()
    # Basic test - method should exist and be callable
    assert hasattr(tool, "celsius_to_fahrenheit")
    assert callable(getattr(tool, "celsius_to_fahrenheit"))
    print(f"[PASS] TemperatureConverterTool.celsius_to_fahrenheit exists and is callable")


def test_temperature_converter_fahrenheit_to_celsius():
    """Test fahrenheit_to_celsius method."""
    tool = TemperatureConverterTool()
    # Basic test - method should exist and be callable
    assert hasattr(tool, "fahrenheit_to_celsius")
    assert callable(getattr(tool, "fahrenheit_to_celsius"))
    print(f"[PASS] TemperatureConverterTool.fahrenheit_to_celsius exists and is callable")



def run_all_tests():
    """Run all tests and return results."""
    tests = [
        test_temperature_converter_instantiation,
        test_temperature_converter_execute_unknown,
        test_temperature_converter_celsius_to_fahrenheit,
        test_temperature_converter_fahrenheit_to_celsius,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"{test.__name__}: AssertionError - {e}")
        except Exception as e:
            failed += 1
            errors.append(f"{test.__name__}: {type(e).__name__} - {e}")

    return {
        "passed": passed,
        "failed": failed,
        "total": len(tests),
        "errors": errors,
        "success": failed == 0
    }


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\nResults: {results['passed']}/{results['total']} passed")
    if results["errors"]:
        print("Errors:")
        for err in results["errors"]:
            print(f"  - {err}")
    sys.exit(0 if results["success"] else 1)
