"""Code Execution API — real sandboxed Python execution for the CodePanel."""

import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Depends

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/code", tags=["code"], dependencies=[Depends(require_api_key)])

# Modules the interactive code panel is allowed to use (relaxed vs agent blocklist)
DATA_SCIENCE_MODULES = {
    "matplotlib", "numpy", "pandas", "scipy", "seaborn", "sklearn",
    "plotly", "io", "base64", "hashlib", "json", "csv", "math",
    "statistics", "collections", "itertools", "functools", "re",
    "datetime", "random",
}

# Sentinel markers for structured output parsing
IMG_START = "__AURA_IMG__"
IMG_END = "__AURA_IMG_END__"
TABLE_START = "__AURA_TABLE__"
TABLE_END = "__AURA_TABLE_END__"
STATE_START = "__AURA_STATE__"
STATE_END = "__AURA_STATE_END__"

# Limits
MAX_CODE_LENGTH = 50_000
MAX_TEXT_OUTPUT = 50_000  # 50KB text
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB per image
MAX_IMAGES = 10
MAX_TIMEOUT = 120

# --- Preamble / epilogue injected around user code ---

CAPTURE_INIT = '''
# --- Capture infrastructure (always defined) ---
_aura_orig_print = print
_aura_images = []
'''

MATPLOTLIB_PREAMBLE = '''
try:
    import matplotlib as _aura_mpl; _aura_mpl.use("Agg")
    import matplotlib.pyplot as _aura_plt
    import io as _aura_io, base64 as _aura_b64
    _orig_show = _aura_plt.show
    def _aura_show(*a, **kw):
        buf = _aura_io.BytesIO()
        _aura_plt.gcf().savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        _aura_images.append(_aura_b64.b64encode(buf.read()).decode())
        _aura_plt.close("all")
    _aura_plt.show = _aura_show
    _aura_plt.savefig = lambda f, *a, **kw: _aura_show()
except ImportError:
    pass
'''

PANDAS_PREAMBLE = '''
try:
    import pandas as _aura_pd
    _aura_tables = []
    def _aura_smart_print(*args, **kwargs):
        parts = []
        for a in args:
            if isinstance(a, _aura_pd.DataFrame):
                html = a.to_html(max_rows=50, max_cols=20, classes="aura-df")
                _aura_tables.append(html)
                parts.append(f"__AURA_TABLE__{html}__AURA_TABLE_END__")
            else:
                parts.append(str(a))
        _aura_orig_print(*parts, **kwargs)
    import builtins as _aura_builtins
    _aura_builtins.print = _aura_smart_print
except ImportError:
    pass
'''

STATE_EPILOGUE = '''
# --- Capture state ---
import json as _aura_json
_aura_state = {}
for _aura_k, _aura_v in dict(locals()).items():
    if _aura_k.startswith("_aura_") or _aura_k.startswith("_") or _aura_k == "builtins":
        continue
    try:
        if hasattr(_aura_v, "to_csv"):  # DataFrame-like
            _aura_state[_aura_k] = {
                "value": _aura_v.to_csv(index=False),
                "type_name": type(_aura_v).__name__,
                "is_dataframe": True,
            }
        else:
            _aura_json.dumps(_aura_v)  # test serializable
            _aura_state[_aura_k] = {
                "value": _aura_v,
                "type_name": type(_aura_v).__name__,
                "is_dataframe": False,
            }
    except (TypeError, ValueError, OverflowError):
        pass
if _aura_state:
    _aura_orig_print(f"__AURA_STATE__{_aura_json.dumps(_aura_state)}__AURA_STATE_END__")

# --- Emit captured images ---
for _aura_img_b64 in _aura_images[:10]:
    _aura_orig_print(f"__AURA_IMG__{_aura_img_b64}__AURA_IMG_END__")
'''


def _build_full_code(user_code: str, session_preamble: str) -> str:
    """Wrap user code with capture preamble + epilogue."""
    parts = [
        CAPTURE_INIT,
        MATPLOTLIB_PREAMBLE,
        PANDAS_PREAMBLE,
        session_preamble,
        "# --- User code ---",
        user_code,
        "# --- End user code ---",
        STATE_EPILOGUE,
    ]
    return "\n".join(parts)


def _parse_outputs(stdout: str, stderr: str, success: bool) -> tuple[list[dict], list[dict], str | None]:
    """Parse sentinel markers from stdout into structured output blocks.

    Returns (outputs, variables, state_json).
    """
    outputs: list[dict] = []
    variables: list[dict] = []
    state_json: str | None = None

    if not success and stderr:
        # Parse error info
        ename = "Error"
        evalue = stderr
        # Try to extract exception class name
        lines = stderr.strip().split("\n")
        for line in reversed(lines):
            m = re.match(r"^(\w+Error|\w+Exception):\s*(.+)", line)
            if m:
                ename = m.group(1)
                evalue = m.group(2)
                break
        outputs.append({
            "type": "error",
            "ename": ename,
            "evalue": evalue,
            "traceback": stderr.strip(),
        })
        return outputs, variables, state_json

    if not stdout:
        return outputs, variables, state_json

    # Process stdout: split on sentinels while preserving order
    remaining = stdout
    while remaining:
        # Find earliest sentinel
        img_pos = remaining.find(IMG_START)
        table_pos = remaining.find(TABLE_START)
        state_pos = remaining.find(STATE_START)

        positions = []
        if img_pos >= 0:
            positions.append(("image", img_pos))
        if table_pos >= 0:
            positions.append(("table", table_pos))
        if state_pos >= 0:
            positions.append(("state", state_pos))

        if not positions:
            # No more sentinels — rest is plain text
            text = remaining.strip()
            if text:
                outputs.append({"type": "stdout", "text": text[:MAX_TEXT_OUTPUT]})
            break

        # Process the earliest sentinel
        positions.sort(key=lambda x: x[1])
        sentinel_type, pos = positions[0]

        # Text before the sentinel
        before = remaining[:pos].strip()
        if before:
            outputs.append({"type": "stdout", "text": before[:MAX_TEXT_OUTPUT]})

        if sentinel_type == "image":
            end_pos = remaining.find(IMG_END, pos)
            if end_pos < 0:
                remaining = remaining[pos + len(IMG_START):]
                continue
            data = remaining[pos + len(IMG_START):end_pos]
            if len(data) <= MAX_IMAGE_BYTES and len([o for o in outputs if o.get("type") == "image"]) < MAX_IMAGES:
                outputs.append({"type": "image", "mime": "image/png", "data": data})
            remaining = remaining[end_pos + len(IMG_END):]

        elif sentinel_type == "table":
            end_pos = remaining.find(TABLE_END, pos)
            if end_pos < 0:
                remaining = remaining[pos + len(TABLE_START):]
                continue
            html = remaining[pos + len(TABLE_START):end_pos]
            outputs.append({"type": "html", "content": html[:MAX_TEXT_OUTPUT]})
            remaining = remaining[end_pos + len(TABLE_END):]

        elif sentinel_type == "state":
            end_pos = remaining.find(STATE_END, pos)
            if end_pos < 0:
                remaining = remaining[pos + len(STATE_START):]
                continue
            state_json = remaining[pos + len(STATE_START):end_pos]
            # Parse variables for the response
            try:
                state_data = json.loads(state_json)
                for name, info in state_data.items():
                    if isinstance(info, dict):
                        variables.append({
                            "name": name,
                            "type_name": info.get("type_name", "unknown"),
                            "repr": str(info.get("value", ""))[:200],
                        })
            except (json.JSONDecodeError, TypeError):
                pass
            remaining = remaining[end_pos + len(STATE_END):]

    # If stderr has warnings but execution succeeded, note them
    if success and stderr and stderr.strip():
        outputs.append({"type": "stdout", "text": f"[stderr] {stderr.strip()[:2000]}"})

    return outputs, variables, state_json


def _get_executor(timeout: int = 30):
    """Create a CodeExecutorTool with relaxed module allowlist for data science."""
    from aura.tools.code_executor import CodeExecutorTool
    return CodeExecutorTool(
        timeout=timeout,
        max_output_length=MAX_TEXT_OUTPUT,
        allowed_modules=DATA_SCIENCE_MODULES,
    )


from api.utils import EndpointRateLimiter
_code_exec_limiter = EndpointRateLimiter(max_per_minute=20)


@router.post("/execute")
async def execute_code(body: dict):
    """Execute Python code in a sandboxed environment.

    Body: { code: str, session_id?: str, timeout?: int }
    Returns: { success, outputs, variables, session_id, execution_time, sandbox }
    """
    _code_exec_limiter.check()
    code = body.get("code", "").strip()
    if not code:
        raise HTTPException(400, "code is required")
    if len(code) > MAX_CODE_LENGTH:
        raise HTTPException(400, f"code exceeds maximum length of {MAX_CODE_LENGTH} characters")

    session_id = body.get("session_id") or f"code-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r'[a-zA-Z0-9\-_]{1,64}', session_id):
        raise HTTPException(400, "Invalid session_id format (alphanumeric/hyphens, max 64 chars)")
    timeout = body.get("timeout", 30)
    if not isinstance(timeout, int) or timeout < 1 or timeout > MAX_TIMEOUT:
        timeout = 30

    # Security: reject code that references internal sentinel variable names.
    # \b doesn't match before underscore (both are \w), so use negative lookbehind.
    if re.search(r'(?<!\w)_aura_\w+', code):
        raise HTTPException(400, "Code cannot reference internal _aura_ variables")

    # SECURITY: AST-validate user-submitted code before any execution tier.
    # This catches dangerous imports, blocked builtins, and sandbox-escape
    # patterns before the code reaches Monty, E2B, or the local subprocess.
    from aura.agent import validate_script_code
    is_valid, validation_msg = validate_script_code(code, "<api_code_execute>")
    if not is_valid:
        logger.warning("[CodeExec] Blocked unsafe code submission: %s", validation_msg)
        raise HTTPException(400, f"Code blocked for safety: {validation_msg}")

    # Get session state preamble
    from aura.tools.code_session_manager import get_session_manager
    session_mgr = get_session_manager()
    session_preamble = session_mgr.get_preamble(session_id)

    # Build the full code with capture infrastructure
    full_code = _build_full_code(code, session_preamble)

    # Execute via execute_raw: the user code was already AST-validated above,
    # and full_code includes preamble/epilogue that use builtins/hasattr/locals
    # which would be false-positived by the executor's own safety_check.
    executor = _get_executor(timeout=timeout)

    loop = asyncio.get_running_loop()
    start_time = time.time()
    result = await loop.run_in_executor(
        None, lambda: executor.execute_raw(full_code, user_code=code)
    )
    execution_time = round(time.time() - start_time, 3)

    # Parse outputs
    stdout = result.get("output", "") or ""
    stderr = result.get("errors", "") or result.get("error", "") or ""
    success = result.get("success", False)
    sandbox = result.get("sandbox", "subprocess")

    logger.info("[CodeExec] session=%s sandbox=%s success=%s time=%.3fs code_len=%d",
                session_id, sandbox, success, execution_time, len(code))

    outputs, variables, state_json = _parse_outputs(stdout, stderr, success)

    # Save session state if we got state data
    if state_json:
        try:
            session_mgr.save_state(session_id, state_json)
        except Exception as e:
            logger.warning("[CodeExec] Failed to save session state: %s", e)

    # If executor succeeded and there are no error output blocks, it's a success.
    # An executor failure with no stdout/stderr should NOT report as success.
    has_errors = any(o.get("type") == "error" for o in outputs)
    return {
        "success": success and not has_errors,
        "outputs": outputs,
        "variables": variables,
        "session_id": session_id,
        "execution_time": execution_time,
        "sandbox": sandbox,
    }


@router.post("/session/reset")
async def reset_session(body: dict):
    """Clear session state.

    Body: { session_id: str }
    """
    session_id = body.get("session_id", "").strip()
    if not session_id:
        raise HTTPException(400, "session_id is required")

    from aura.tools.code_session_manager import get_session_manager
    get_session_manager().reset(session_id)

    return {"success": True, "session_id": session_id}
