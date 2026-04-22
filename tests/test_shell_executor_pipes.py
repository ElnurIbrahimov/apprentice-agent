"""Tests for the updated shell executor: pipes, redirects, chaining, and injection blocking."""
from aura.tools.shell_executor import (
    ShellExecutorTool,
    _contains_shell_injection,
    _is_pipeline_or_chain,
    _extract_base_command,
)


def _tool():
    return ShellExecutorTool()


# ---- Injection detection ----

class TestInjectionDetection:
    def test_backtick_blocked(self):
        assert _contains_shell_injection("echo `whoami`") is True

    def test_dollar_paren_blocked(self):
        assert _contains_shell_injection("echo $(whoami)") is True

    def test_dollar_double_paren_blocked(self):
        assert _contains_shell_injection("echo $((1+1))") is True

    def test_dollar_brace_blocked(self):
        assert _contains_shell_injection("echo ${HOME}") is True

    def test_eval_blocked(self):
        assert _contains_shell_injection("eval echo hello") is True

    def test_exec_blocked(self):
        assert _contains_shell_injection("exec /bin/sh") is True

    def test_source_blocked(self):
        assert _contains_shell_injection("source ~/.bashrc") is True

    def test_semicolon_blocked(self):
        assert _contains_shell_injection("echo hello ; rm -rf /") is True

    def test_newline_blocked(self):
        assert _contains_shell_injection("echo hello\nrm -rf /") is True

    def test_clobber_blocked(self):
        assert _contains_shell_injection("echo test >| file") is True

    def test_bg_redirect_blocked(self):
        assert _contains_shell_injection("cmd &> /dev/null") is True

    def test_process_sub_blocked(self):
        assert _contains_shell_injection("diff <(ls) file") is True

    # These should NOT be detected as injection
    def test_pipe_allowed(self):
        assert _contains_shell_injection("grep foo | wc -l") is False

    def test_redirect_allowed(self):
        assert _contains_shell_injection("echo test > file.txt") is False

    def test_append_redirect_allowed(self):
        assert _contains_shell_injection("echo test >> file.txt") is False

    def test_and_chain_allowed(self):
        assert _contains_shell_injection("ls && echo done") is False

    def test_or_chain_allowed(self):
        assert _contains_shell_injection("grep foo || echo not found") is False

    def test_simple_command_allowed(self):
        assert _contains_shell_injection("echo hello world") is False


# ---- Pipeline detection ----

class TestPipelineDetection:
    def test_simple_pipe(self):
        assert _is_pipeline_or_chain("grep foo | wc -l") is True

    def test_double_pipe(self):
        assert _is_pipeline_or_chain("ls || echo fail") is True

    def test_and_chain(self):
        assert _is_pipeline_or_chain("ls && echo ok") is True

    def test_redirect(self):
        assert _is_pipeline_or_chain("echo test > file") is True

    def test_append_redirect(self):
        assert _is_pipeline_or_chain("echo test >> file") is True

    def test_simple_command(self):
        assert _is_pipeline_or_chain("echo hello") is False


# ---- Base command extraction ----

class TestExtractBaseCommand:
    def test_simple(self):
        assert _extract_base_command("grep foo bar.txt") == "grep"

    def test_with_path(self):
        assert _extract_base_command("/usr/bin/grep foo") == "grep"

    def test_with_env_var(self):
        assert _extract_base_command("FOO=bar echo hello") == "echo"

    def test_with_redirect(self):
        assert _extract_base_command("echo hello > file.txt") == "echo"

    def test_empty(self):
        assert _extract_base_command("") is None


# ---- Command validation ----

class TestValidateCommand:
    def test_simple_allowed(self):
        valid, reason = _tool()._validate_command("echo hello")
        assert valid is True
        assert reason == "OK"

    def test_pipe_both_allowed(self):
        valid, reason = _tool()._validate_command("grep foo | wc -l")
        assert valid is True
        assert reason == "OK"

    def test_chain_both_allowed(self):
        valid, reason = _tool()._validate_command("ls -la && echo done")
        assert valid is True
        assert reason == "OK"

    def test_or_chain_allowed(self):
        valid, reason = _tool()._validate_command("grep foo || echo not found")
        assert valid is True
        assert reason == "OK"

    def test_multi_pipe_allowed(self):
        valid, reason = _tool()._validate_command("cat file | sort | uniq")
        assert valid is True
        assert reason == "OK"

    def test_redirect_allowed(self):
        valid, reason = _tool()._validate_command("echo hello > output.txt")
        assert valid is True
        assert reason == "OK"

    def test_kill_by_pid_allowed(self):
        valid, reason = _tool()._validate_command("kill 12345")
        assert valid is True
        assert reason == "OK"

    def test_killall_blocked(self):
        valid, reason = _tool()._validate_command("killall node")
        assert valid is False

    def test_taskkill_blocked(self):
        valid, reason = _tool()._validate_command("taskkill /F /IM node.exe")
        assert valid is False

    def test_pkill_blocked(self):
        valid, reason = _tool()._validate_command("pkill python")
        assert valid is False

    def test_pipe_to_disallowed_cmd(self):
        valid, reason = _tool()._validate_command("echo hello | del file.txt")
        assert valid is False

    def test_chain_with_disallowed_cmd(self):
        valid, reason = _tool()._validate_command("echo hello && del important.txt")
        assert valid is False

    def test_sandbox_required(self):
        valid, reason = _tool()._validate_command("python script.py")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_pipe_with_sandbox_segment(self):
        valid, reason = _tool()._validate_command("cat data.txt | python process.py")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_sudo_blocked(self):
        valid, reason = _tool()._validate_command("sudo ls /tmp")
        assert valid is False
        assert "privilege" in reason.lower()

    def test_rm_rf_root_blocked(self):
        valid, reason = _tool()._validate_command("rm -rf /")
        assert valid is False

    def test_pipe_to_shell_blocked(self):
        """Pipe to shell variants should still be blocked by BLOCKED_PATTERNS."""
        valid, reason = _tool()._validate_command("echo evil | bash")
        assert valid is False


# ---- Compiler / build-tool sandboxing (C1 fix) ----

class TestCompilerSandbox:
    """Compilers run arbitrary code via build scripts. They must be routed
    through the sandbox (SANDBOX_REQUIRED), not direct-exec."""

    def test_cargo_build_requires_sandbox(self):
        valid, reason = _tool()._validate_command("cargo build")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_gcc_compile_requires_sandbox(self):
        valid, reason = _tool()._validate_command("gcc main.c -o main")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_make_requires_sandbox(self):
        valid, reason = _tool()._validate_command("make install")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_rustc_requires_sandbox(self):
        valid, reason = _tool()._validate_command("rustc main.rs")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_go_requires_sandbox(self):
        valid, reason = _tool()._validate_command("go build ./...")
        assert valid is True
        assert reason == "SANDBOX_REQUIRED"

    def test_ruff_still_direct_exec(self):
        """Linters are read-only and stay on the direct-exec allowlist."""
        valid, reason = _tool()._validate_command("ruff check .")
        assert valid is True
        assert reason == "OK"

    def test_tsc_still_direct_exec(self):
        valid, reason = _tool()._validate_command("tsc --noEmit")
        assert valid is True
        assert reason == "OK"
