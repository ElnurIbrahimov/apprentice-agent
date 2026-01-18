"""Auto-generated tests for custom tool: bmi_calculator

Created: 2026-01-18T18:34:03.961262
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from apprentice_agent.tools.custom.bmi_calculator import BmiCalculatorTool


def test_bmi_calculator_instantiation():
    """Test that the tool can be instantiated."""
    tool = BmiCalculatorTool()
    assert tool.name == "bmi_calculator"
    assert tool.description is not None
    print(f"[PASS] BmiCalculatorTool instantiation")


def test_bmi_calculator_execute_unknown():
    """Test that unknown actions return error."""
    tool = BmiCalculatorTool()
    result = tool.execute("unknown_action_xyz")
    assert result.get("success") == False
    assert "error" in result
    print(f"[PASS] BmiCalculatorTool handles unknown action")


def test_bmi_calculator_calculate_bmi():
    """Test calculate_bmi method."""
    tool = BmiCalculatorTool()
    # Basic test - method should exist and be callable
    assert hasattr(tool, "calculate_bmi")
    assert callable(getattr(tool, "calculate_bmi"))
    print(f"[PASS] BmiCalculatorTool.calculate_bmi exists and is callable")


def test_bmi_calculator_calculate_bmi_imperial():
    """Test calculate_bmi_imperial method."""
    tool = BmiCalculatorTool()
    # Basic test - method should exist and be callable
    assert hasattr(tool, "calculate_bmi_imperial")
    assert callable(getattr(tool, "calculate_bmi_imperial"))
    print(f"[PASS] BmiCalculatorTool.calculate_bmi_imperial exists and is callable")



def run_all_tests():
    """Run all tests and return results."""
    tests = [
        test_bmi_calculator_instantiation,
        test_bmi_calculator_execute_unknown,
        test_bmi_calculator_calculate_bmi,
        test_bmi_calculator_calculate_bmi_imperial,
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
