def execute(password: str) -> dict:
    """
    Check password strength and return score from 0 to 100

    Returns:
        dict with 'success' (bool) and 'result' (any) keys
    """
    try:

        import re
        score = 0
        if len(password) > 8:
            score += 20
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if re.search(r'[^a-zA-Z0-9]', password):
            score += 20
        result = min(100, score)

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Test
if __name__ == "__main__":
    test_result = execute(password="MyP@ssw0rd")
    print(f"Test result: {test_result}")
    assert test_result.get("success"), f"Test failed: {test_result}"