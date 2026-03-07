def test_shell_executor_blocks_chaining():
    from aura.tools.shell_executor import ShellExecutorTool
    executor = ShellExecutorTool()
    result = executor.execute("echo hello && del important.txt")
    assert result["success"] == False
    assert "disallowed" in result["error"].lower()


def test_database_tool_blocks_drop():
    from aura.tools.database_tool import DatabaseTool
    db = DatabaseTool()
    result = db.query("DROP TABLE users", "test_db")
    assert result["success"] == False


def test_run_math_blocks_import():
    from aura.tools.code_executor import CodeExecutorTool
    ex = CodeExecutorTool()
    result = ex.run_math("__import__('os').system('echo pwned')")
    assert result["success"] == False
