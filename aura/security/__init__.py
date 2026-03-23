"""Aura security subsystem — stolen from OpenFang's 16-layer defense-in-depth.

Lazy imports to avoid 160ms+ penalty when only one module is needed.
Use: `from aura.security.ssrf_guard import validate_url_safe`
Or:  `from aura.security import validate_url_safe` (triggers lazy load)
"""


def __getattr__(name):
    """Lazy import — only load modules when their symbols are accessed."""
    _map = {
        # ssrf_guard
        "validate_url_safe": "aura.security.ssrf_guard",
        "is_private_ip": "aura.security.ssrf_guard",
        "safe_request": "aura.security.ssrf_guard",
        # taint_tracker
        "TaintTracker": "aura.security.taint_tracker",
        "TaintLabel": "aura.security.taint_tracker",
        "scan_for_secrets": "aura.security.taint_tracker",
        "redact": "aura.security.taint_tracker",
        "get_tracker": "aura.security.taint_tracker",
        # audit_chain
        "AuditChain": "aura.security.audit_chain",
        "AuditEntry": "aura.security.audit_chain",
        "get_audit_chain": "aura.security.audit_chain",
        # tool_signing
        "sign_tool": "aura.security.tool_signing",
        "verify_tool": "aura.security.tool_signing",
        "is_tool_signed": "aura.security.tool_signing",
        # tool_validator
        "validate_custom_tool_code": "aura.security.tool_validator",
        "validate_script_code": "aura.security.tool_validator",
        "ALLOWED_TOOL_IMPORTS": "aura.security.tool_validator",
        "FORBIDDEN_PATTERNS": "aura.security.tool_validator",
    }
    if name in _map:
        import importlib
        mod = importlib.import_module(_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'aura.security' has no attribute {name!r}")
