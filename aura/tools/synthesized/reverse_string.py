def execute(input_string: str) -> dict:
    """
    Reverse a given string

    Returns:
        dict with 'success' (bool) and 'result' (any) keys
    """
    try:
        result = input_string[::-1]

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Test
if __name__ == "__main__":
    test_result = execute(input_string="hello")
    print(f"Test result: {test_result}")
    assert test_result.get("success"), f"Test failed: {test_result}"