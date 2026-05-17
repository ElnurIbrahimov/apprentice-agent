"""
████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗    ███████╗██████╗ ██╗███╗   ██╗███████╗
╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║    ██╔════╝██╔══██╗██║████╗  ██║██╔════╝
   ██║   ██████╔╝██║   ██║   ██║   ███████║    ███████╗██████╔╝██║██╔██╗ ██║█████╗
   ██║   ██╔══██╗██║   ██║   ██║   ██╔══██║    ╚════██║██╔═══╝ ██║██║╚██╗██║██╔══╝
   ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║    ███████║██║     ██║██║ ╚████║███████╗
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝    ╚══════╝╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝

TRUTH SPINE — The Non-Negotiable Verification Layer
====================================================

Core principle: "If you can't verify it with an artifact, it's SPECULATION"

Every action must follow this contract:
    ACTION → ARTIFACT → VERIFICATION → MEMORY TIER

Memory Tiers:
    FACT = verified with artifact (hash, return code, file exists)
    BELIEF = inferred but not proven
    SPECULATION = unverified claims (including LLM output)
"""

import ast
import hashlib
import hmac
import json
import logging
import math
import operator
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Optional embedding support
try:
    from aura.memory.embedding import get_embedding as _get_embedding
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False
    _get_embedding = None


# ============================================================================
#                    SECURITY: Safe Expression Evaluator
# ============================================================================

class SafeCalculator:
    """
    Safe mathematical expression evaluator WITHOUT eval().
    Uses AST parsing to safely evaluate only allowed operations.

    SECURITY: This replaces eval() to prevent code injection attacks.
    """

    # Allowed binary operators
    OPERATORS: ClassVar[dict] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Allowed safe functions
    FUNCTIONS: ClassVar[dict] = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'int': int,
        'float': float,
    }

    def evaluate(self, expression: str) -> float:
        """Safely evaluate mathematical expression without eval()."""
        if not expression or not expression.strip():
            raise ValueError("Empty expression")

        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax: {e}") from e
        except (TypeError, KeyError) as e:
            raise ValueError(f"Invalid expression: {e}") from e

    def _eval_node(self, node) -> float:
        """Recursively evaluate AST nodes safely."""
        # Handle constants (Python 3.8+)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Invalid constant type: {type(node.value).__name__}")

        # Handle numbers (Python 3.7 compatibility)
        elif isinstance(node, ast.Num):
            return float(node.n)

        # Handle binary operations (+, -, *, /, etc.)
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            # Prevent exponent attacks (2**9999999 = DoS)
            if op_type == ast.Pow and right > 1000:
                raise ValueError("Exponent too large (max 1000)")
            return self.OPERATORS[op_type](left, right)

        # Handle unary operations (-, +)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return self.OPERATORS[op_type](operand)

        # Handle function calls (abs, round, min, max, etc.)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in self.FUNCTIONS:
                    raise ValueError(f"Unknown function: {func_name}")
                args = [self._eval_node(arg) for arg in node.args]
                return float(self.FUNCTIONS[func_name](*args))
            raise ValueError("Only named function calls allowed")

        # Handle Expression wrapper
        elif isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """
    Safe replacement for eval() - only allows mathematical expressions.

    SECURITY: Never use eval() with user input. Use this instead.
    """
    calculator = SafeCalculator()
    return calculator.evaluate(expression)


# ============================================================================
#                    PART 1: ARTIFACTS — Physical Proof
# ============================================================================

class ArtifactType(Enum):
    """Types of physical proof"""
    FILE = "file"
    STDOUT = "stdout"
    JSON = "json"
    HASH = "hash"
    NONE = "none"


@dataclass
class Artifact:
    """
    Physical proof that something happened.

    An artifact is verifiable evidence - not "I did it" but actual proof:
    - File with hash
    - Command output with return code
    - Structured JSON result
    """
    artifact_id: str
    artifact_type: ArtifactType
    content_hash: str
    raw_data: Any
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Artifact":
        """Create artifact from file with SHA256 hash"""
        path = Path(path)
        if not path.exists():
            return cls(
                artifact_id=str(uuid.uuid4())[:12],
                artifact_type=ArtifactType.NONE,
                content_hash="",
                raw_data={"error": f"File not found: {path}"},
                metadata={"path": str(path), "exists": False}
            )

        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()

        return cls(
            artifact_id=str(uuid.uuid4())[:12],
            artifact_type=ArtifactType.FILE,
            content_hash=content_hash,
            raw_data=content.decode('utf-8', errors='replace')[:10000],  # Cap for memory
            metadata={
                "path": str(path.resolve()),
                "size": len(content),
                "exists": True,
                "modified": path.stat().st_mtime
            }
        )

    @classmethod
    def from_stdout(cls, stdout: str, stderr: str = "", returncode: int = 0) -> "Artifact":
        """Create artifact from command output"""
        combined = f"stdout:{stdout}\nstderr:{stderr}\nreturncode:{returncode}"
        content_hash = hashlib.sha256(combined.encode()).hexdigest()

        return cls(
            artifact_id=str(uuid.uuid4())[:12],
            artifact_type=ArtifactType.STDOUT,
            content_hash=content_hash,
            raw_data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
            metadata={
                "stdout_length": len(stdout),
                "stderr_length": len(stderr),
                "returncode": returncode,
                "success": returncode == 0
            }
        )

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Artifact":
        """Create artifact from structured JSON result"""
        json_str = json.dumps(data, sort_keys=True)
        content_hash = hashlib.sha256(json_str.encode()).hexdigest()

        return cls(
            artifact_id=str(uuid.uuid4())[:12],
            artifact_type=ArtifactType.JSON,
            content_hash=content_hash,
            raw_data=data,
            metadata={
                "keys": list(data.keys()) if isinstance(data, dict) else [],
                "type": type(data).__name__
            }
        )

    @classmethod
    def empty(cls, reason: str = "No artifact produced") -> "Artifact":
        """Create empty artifact for failed operations"""
        return cls(
            artifact_id=str(uuid.uuid4())[:12],
            artifact_type=ArtifactType.NONE,
            content_hash="",
            raw_data={"reason": reason},
            metadata={"empty": True, "reason": reason}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "created_at": self.created_at
        }

    @property
    def is_valid(self) -> bool:
        return self.artifact_type != ArtifactType.NONE and bool(self.content_hash)


# ============================================================================
#                    PART 2: VERIFICATION CHECKS
# ============================================================================

class VerificationCheck(ABC):
    """Base class for verification checks"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Check name for reporting"""
        pass

    @abstractmethod
    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Run the check.
        Returns: (passed: bool, reason: str)
        """
        pass


class FileExistsCheck(VerificationCheck):
    """Verify file exists at expected path"""

    @property
    def name(self) -> str:
        return "file_exists"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        expected_path = context.get("expected_path")
        if not expected_path:
            # Check from artifact metadata
            path = artifact.metadata.get("path")
            if path and Path(path).exists():
                return True, f"File exists: {path}"
            return False, "No path to verify"

        if Path(expected_path).exists():
            return True, f"File exists: {expected_path}"
        return False, f"File not found: {expected_path}"


class HashMatchCheck(VerificationCheck):
    """Verify content hash matches expected"""

    @property
    def name(self) -> str:
        return "hash_match"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        expected_hash = context.get("expected_hash")
        if not expected_hash:
            return False, "No expected hash provided"

        if artifact.content_hash == expected_hash:
            return True, f"Hash matches: {expected_hash[:16]}..."
        return False, f"Hash mismatch: expected {expected_hash[:16]}..., got {artifact.content_hash[:16]}..."


class ReturnCodeCheck(VerificationCheck):
    """Verify command return code is zero (success)"""

    @property
    def name(self) -> str:
        return "return_code_zero"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        if artifact.artifact_type != ArtifactType.STDOUT:
            return False, "Not a stdout artifact"

        returncode = artifact.metadata.get("returncode")
        expected = context.get("expected_code", 0)

        if returncode == expected:
            return True, f"Return code {returncode} matches expected {expected}"
        return False, f"Return code {returncode} != expected {expected}"


class NotEmptyCheck(VerificationCheck):
    """Verify artifact has non-empty content"""

    @property
    def name(self) -> str:
        return "not_empty"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        if not artifact.is_valid:
            return False, "Invalid artifact"

        raw = artifact.raw_data
        if isinstance(raw, dict):
            if raw.get("stdout"):
                return True, f"Has stdout ({len(raw['stdout'])} chars)"
            if raw.get("result"):
                return True, "Has result"
        elif isinstance(raw, str) and raw.strip():
            return True, f"Has content ({len(raw)} chars)"

        return False, "Empty content"


class JSONValidCheck(VerificationCheck):
    """Verify content is valid JSON"""

    @property
    def name(self) -> str:
        return "json_valid"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        if artifact.artifact_type == ArtifactType.JSON:
            return True, "Already parsed as JSON"

        try:
            if isinstance(artifact.raw_data, str):
                json.loads(artifact.raw_data)
                return True, "Valid JSON string"
            elif isinstance(artifact.raw_data, dict):
                return True, "Valid JSON object"
        except (json.JSONDecodeError, TypeError) as e:
            return False, f"Invalid JSON: {e}"

        return False, "Cannot verify as JSON"


class NoErrorCheck(VerificationCheck):
    """Verify no error in result"""

    @property
    def name(self) -> str:
        return "no_error"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        raw = artifact.raw_data

        if isinstance(raw, dict):
            if raw.get("error"):
                return False, f"Error present: {raw['error']}"
            if raw.get("stderr") and "error" in raw["stderr"].lower():
                return False, "Stderr contains error"
            if raw.get("success") is False:
                return False, "success=False"

        return True, "No errors detected"


class ContainsTextCheck(VerificationCheck):
    """Verify output contains expected text"""

    def __init__(self, expected_text: str | None = None):
        self._expected = expected_text

    @property
    def name(self) -> str:
        return "contains_text"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        expected = self._expected or context.get("expected_text")
        if not expected:
            return False, "No expected text provided"

        raw = artifact.raw_data
        text = ""

        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict):
            text = str(raw.get("stdout", "")) + str(raw.get("result", ""))

        suffix = '...' if len(expected) > 30 else ''
        if expected.lower() in text.lower():
            return True, f"Contains '{expected[:30]}{suffix}'"
        return False, f"Missing '{expected[:30]}{suffix}'"


class SandboxPathCheck(VerificationCheck):
    """Verify path is within sandbox"""

    def __init__(self, sandbox_dir: Path):
        self._sandbox = sandbox_dir.resolve()

    @property
    def name(self) -> str:
        return "sandbox_path"

    def check(self, artifact: Artifact, context: Dict[str, Any]) -> Tuple[bool, str]:
        path = context.get("path") or artifact.metadata.get("path")
        if not path:
            return True, "No path to verify"

        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self._sandbox)
            return True, f"Path within sandbox: {resolved}"
        except ValueError:
            return False, f"Path outside sandbox: {resolved}"


# ============================================================================
#                    PART 3: VERIFIER SPINE
# ============================================================================

@dataclass
class VerificationResult:
    """Result of verification process"""
    is_verified: bool
    artifact: Artifact
    checks_passed: List[str]
    checks_failed: List[str]
    reasoning: str
    verification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "is_verified": self.is_verified,
            "artifact": self.artifact.to_dict(),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp
        }


class VerifierSpine:
    """
    The enforcement layer - THE critical component.

    Every action must produce an artifact that passes verification checks.
    No exceptions. No bypasses.
    """

    # Default checks for different action types
    DEFAULT_CHECKS: ClassVar[dict] = {
        "file_write": ["file_exists", "not_empty", "sandbox_path"],
        "file_read": ["file_exists", "not_empty", "sandbox_path"],
        "command": ["return_code_zero", "no_error"],
        "calculate": ["no_error", "not_empty"],
        "search": ["not_empty", "json_valid"],
        "tool_test": ["return_code_zero", "no_error", "not_empty"],
        "default": ["no_error"]
    }

    def __init__(self, sandbox_dir: Path):
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Register built-in checks
        self.checks: Dict[str, VerificationCheck] = {
            "file_exists": FileExistsCheck(),
            "hash_match": HashMatchCheck(),
            "return_code_zero": ReturnCodeCheck(),
            "not_empty": NotEmptyCheck(),
            "json_valid": JSONValidCheck(),
            "no_error": NoErrorCheck(),
            "contains_text": ContainsTextCheck(),
            "sandbox_path": SandboxPathCheck(self.sandbox_dir)
        }

        # Statistics
        self.stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "by_action_type": {}
        }

    def register_check(self, check: VerificationCheck):
        """Register a custom verification check"""
        self.checks[check.name] = check

    def verify_action(
        self,
        action_type: str,
        raw_result: Dict[str, Any],
        expected_checks: List[str] | None = None,
        context: Dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        Verify an action's result.

        Args:
            action_type: Type of action (file_write, command, etc.)
            raw_result: The raw result from the action
            expected_checks: Override default checks for this action type
            context: Additional context for checks (expected_path, expected_hash, etc.)

        Returns:
            VerificationResult with artifact and check results
        """
        context = context or {}

        # Create artifact from result
        artifact = self._create_artifact(raw_result, context)

        # Determine which checks to run
        check_names = expected_checks or self.DEFAULT_CHECKS.get(
            action_type, self.DEFAULT_CHECKS["default"]
        )

        # Run checks
        passed = []
        failed = []

        for check_name in check_names:
            check = self.checks.get(check_name)
            if not check:
                failed.append(f"{check_name}: Check not found")
                continue

            try:
                success, reason = check.check(artifact, context)
                if success:
                    passed.append(f"{check_name}: {reason}")
                else:
                    failed.append(f"{check_name}: {reason}")
            except Exception as e:
                failed.append(f"{check_name}: Exception - {e}")

        # Determine overall result
        is_verified = len(failed) == 0 and artifact.is_valid

        # Build reasoning
        if is_verified:
            reasoning = f"All {len(passed)} checks passed for {action_type}"
        else:
            reasoning = f"Failed {len(failed)}/{len(passed)+len(failed)} checks: {'; '.join(failed)}"

        # Update stats (thread-safe)
        with self._lock:
            self.stats["total_verifications"] += 1
            if is_verified:
                self.stats["passed"] += 1
            else:
                self.stats["failed"] += 1

            if action_type not in self.stats["by_action_type"]:
                self.stats["by_action_type"][action_type] = {"passed": 0, "failed": 0}
            self.stats["by_action_type"][action_type]["passed" if is_verified else "failed"] += 1

        return VerificationResult(
            is_verified=is_verified,
            artifact=artifact,
            checks_passed=passed,
            checks_failed=failed,
            reasoning=reasoning
        )

    def _create_artifact(self, raw_result: Dict[str, Any], context: Dict[str, Any]) -> Artifact:
        """Create appropriate artifact from raw result"""

        # Check if result has a file path
        if "path" in raw_result and raw_result.get("success", True):
            path = Path(raw_result["path"])
            if path.exists():
                return Artifact.from_file(path)

        # Check if result has stdout (command output)
        if "stdout" in raw_result or "returncode" in raw_result:
            return Artifact.from_stdout(
                stdout=raw_result.get("stdout", ""),
                stderr=raw_result.get("stderr", ""),
                returncode=raw_result.get("returncode", 0)
            )

        # Check if it's a structured result
        if raw_result.get("success") is not None or raw_result.get("result") is not None:
            return Artifact.from_json(raw_result)

        # Fallback to empty artifact
        return Artifact.empty(reason="Could not create artifact from result")

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        total = self.stats["total_verifications"]
        return {
            **self.stats,
            "success_rate": self.stats["passed"] / total if total > 0 else 0.0
        }

    def _hash_file(self, path: str) -> str | None:
        """Compute SHA256 hash of a file. Returns None if file doesn't exist or can't be read."""
        try:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    h.update(chunk)
            return h.hexdigest()
        except (OSError, IOError):
            return None

    def verify_file_artifact(self, path: str, expected_hash: str | None = None) -> dict:
        """Verify a file artifact after an operation.

        Args:
            path: File path to verify
            expected_hash: If provided, verify the file matches this hash

        Returns:
            dict with keys: verified (bool), hash (str|None), tier (FACT|BELIEF|SPECULATION), reason (str)
        """
        current_hash = self._hash_file(path)
        if current_hash is None:
            return {
                "verified": False,
                "hash": None,
                "tier": "SPECULATION",
                "reason": "File does not exist or cannot be read"
            }
        if expected_hash and current_hash != expected_hash:
            return {
                "verified": False,
                "hash": current_hash,
                "tier": "SPECULATION",
                "reason": f"Hash mismatch: expected {expected_hash[:8]}..., got {current_hash[:8]}..."
            }
        return {
            "verified": True,
            "hash": current_hash,
            "tier": "FACT",
            "reason": "File exists and hash verified"
        }


# ============================================================================
#                    PART 4: MEMORY TIERS
# ============================================================================

class MemoryTier(Enum):
    """Memory tiers based on verification status"""
    FACT = "FACT"           # Verified with artifact
    BELIEF = "BELIEF"       # Inferred but not proven
    SPECULATION = "SPECULATION"  # Unverified claims


@dataclass
class VerifiedMemoryTrace:
    """A memory trace with verification status"""
    trace_id: str
    tier: MemoryTier
    content: str
    source: str
    verification: Optional[VerificationResult]
    reasoning: str  # Why it's in this tier
    importance: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    content_hash: str = ""  # Dedup hash for content
    embedding: Optional[List[float]] = None  # Cached embedding vector

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "tier": self.tier.value,
            "content": self.content,
            "source": self.source,
            "verification": self.verification.to_dict() if self.verification else None,
            "reasoning": self.reasoning,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "content_hash": self.content_hash
            # Note: embeddings are NOT persisted to keep JSON small
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifiedMemoryTrace":
        return cls(
            trace_id=data["trace_id"],
            tier=MemoryTier(data["tier"]),
            content=data["content"],
            source=data["source"],
            verification=None,  # Don't reconstruct full verification
            reasoning=data["reasoning"],
            importance=data["importance"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            access_count=data.get("access_count", 0),
            content_hash=data.get("content_hash", "")
        )


class VerifiedMemory:
    """
    Tiered memory system.

    CRITICAL: Only verified content goes to FACTS.
    LLM outputs, inferences, and unverified claims go to BELIEF or SPECULATION.
    """

    def __init__(self, data_dir: Path | None = None, max_traces: int = 500):
        self.data_dir = Path(data_dir) if data_dir else Path("data/memory")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_traces = max_traces
        self._lock = threading.Lock()

        self.traces: Dict[str, VerifiedMemoryTrace] = {}
        self._load()

    def _load(self):
        """Load memory from disk"""
        memory_file = self.data_dir / "verified_memory.json"
        if memory_file.exists():
            try:
                data = json.loads(memory_file.read_text(encoding="utf-8"))
                for trace_data in data.get("traces", []):
                    trace = VerifiedMemoryTrace.from_dict(trace_data)
                    self.traces[trace.trace_id] = trace
            except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
                logger.debug(f"[Memory] Failed to load: {e}")

    def _prune_if_needed(self):
        """Evict lowest-value traces when over max_traces limit."""
        if len(self.traces) <= self.max_traces:
            return
        now = time.time()
        scored = sorted(
            self.traces.items(),
            key=lambda kv: kv[1].importance / (1 + (now - kv[1].last_accessed) / 86400),
            reverse=True  # highest score = keep
        )
        keep_ids = {k for k, _ in scored[:self.max_traces]}
        removed = len(self.traces) - len(keep_ids)
        self.traces = {k: v for k, v in self.traces.items() if k in keep_ids}
        logger.debug(f"[TruthSpine] Pruned {removed} traces, {len(self.traces)} remain")

    def _save(self):
        """Persist memory to disk (atomic write to prevent data loss on crash)"""
        # pruning removed — called explicitly before insertion in store methods
        memory_file = self.data_dir / "verified_memory.json"
        data = {
            "traces": [t.to_dict() for t in self.traces.values()],
            "saved_at": time.time()
        }
        content = json.dumps(data, indent=2)
        fd, tmp_path = tempfile.mkstemp(dir=memory_file.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(tmp_path, str(memory_file))
        except (OSError, IOError) as e:
            logger.error(f"[TruthSpine] Failed to save memory: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def store_fact(
        self,
        content: str,
        verification: VerificationResult,
        source: str,
        importance: float = 0.7
    ) -> Optional[VerifiedMemoryTrace]:
        """
        Store a verified fact.

        CRITICAL: Only call this with a passing verification result!
        """
        with self._lock:
            if not verification.is_verified:
                # Refuse to store unverified content as fact
                logger.debug("[Memory] REFUSED: Cannot store unverified content as FACT")
                return None

            h = self._content_hash(content)
            for trace in self.traces.values():
                if trace.tier == MemoryTier.FACT and trace.content_hash == h:
                    trace.last_accessed = time.time()
                    trace.access_count = getattr(trace, 'access_count', 0) + 1
                    return trace
            self._prune_if_needed()

            trace = VerifiedMemoryTrace(
                trace_id=str(uuid.uuid4())[:12],
                tier=MemoryTier.FACT,
                content=content,
                source=source,
                verification=verification,
                reasoning=f"Verified: {verification.reasoning}",
                importance=importance,
                created_at=time.time(),
                last_accessed=time.time(),
                content_hash=h
            )
            self._cache_embedding(trace)

            self.traces[trace.trace_id] = trace
            self._save()
            return trace

    def _content_hash(self, content: str) -> str:
        """Return a short hash of content for deduplication."""
        return hashlib.md5(content.strip().lower().encode()).hexdigest()[:16]

    @staticmethod
    def _cache_embedding(trace: "VerifiedMemoryTrace") -> None:
        """Compute and cache embedding on a trace (best-effort, non-blocking)."""
        if _EMBEDDINGS_AVAILABLE and trace.embedding is None:
            try:
                trace.embedding = _get_embedding(trace.content, timeout=2.0)
            except Exception as e:
                logger.debug(f"[TruthSpine] Embedding computation failed for trace: {e}")

    def store_belief(
        self,
        content: str,
        source: str,
        reasoning: str,
        importance: float = 0.5
    ) -> VerifiedMemoryTrace:
        """
        Store a belief - inferred but not proven.

        Use for: logical inferences, pattern recognition, educated guesses.
        """
        with self._lock:
            h = self._content_hash(content)
            for trace in self.traces.values():
                if trace.tier == MemoryTier.BELIEF and trace.content_hash == h:
                    trace.last_accessed = time.time()
                    trace.access_count = getattr(trace, 'access_count', 0) + 1
                    return trace

            self._prune_if_needed()
            trace = VerifiedMemoryTrace(
                trace_id=str(uuid.uuid4())[:12],
                tier=MemoryTier.BELIEF,
                content=content,
                source=source,
                verification=None,
                reasoning=f"Belief: {reasoning}",
                importance=importance,
                created_at=time.time(),
                last_accessed=time.time(),
                content_hash=h
            )
            self._cache_embedding(trace)

            self.traces[trace.trace_id] = trace
            self._save()
            return trace

    def store_speculation(
        self,
        content: str,
        source: str,
        reason: str,
        importance: float = 0.3
    ) -> VerifiedMemoryTrace:
        """
        Store speculation - unverified claims.

        Use for: LLM outputs, user claims, anything not verified.
        """
        with self._lock:
            h = self._content_hash(content)
            for trace in self.traces.values():
                if trace.tier == MemoryTier.SPECULATION and trace.content_hash == h:
                    trace.last_accessed = time.time()
                    trace.access_count = getattr(trace, 'access_count', 0) + 1
                    return trace

            self._prune_if_needed()
            trace = VerifiedMemoryTrace(
                trace_id=str(uuid.uuid4())[:12],
                tier=MemoryTier.SPECULATION,
                content=content,
                source=source,
                verification=None,
                reasoning=f"Speculation: {reason}",
                importance=importance,
                created_at=time.time(),
                last_accessed=time.time(),
                content_hash=h
            )
            self._cache_embedding(trace)

            self.traces[trace.trace_id] = trace
            self._save()
            return trace

    # ---- Auto-classification heuristics ----

    _FACT_PATTERNS = re.compile(
        r'(?:'
        r'\d{4}[-/]\d{2}[-/]\d{2}'          # dates
        r'|https?://'                         # URLs
        r'|[A-Za-z]:[/\\]'                    # Windows file paths
        r'|/(?:home|usr|var|tmp|etc)/'        # Unix file paths
        r'|\breturn(?:ed)?\s+(?:code\s+)?\d'  # return codes
        r'|\b(?:verified|confirmed)\b'        # verification keywords
        r'|\bsha256[:=]'                       # hashes
        r'|\b\d+\.\d+\.\d+'                   # version numbers
        r')',
        re.IGNORECASE,
    )

    _SPECULATION_PATTERNS = re.compile(
        r'\b(?:might|could|possibly|I\s+think|maybe|probably|not\s+sure'
        r'|seems?\s+like|perhaps|unclear|uncertain|guess(?:ing)?)\b',
        re.IGNORECASE,
    )

    def auto_store(self, content: str, source: str = "", context: str = "") -> dict:
        """Classify content as FACT/BELIEF/SPECULATION and store automatically.

        Heuristics:
            FACT: contains verifiable data (numbers, dates, URLs, paths, confirmed/verified)
            SPECULATION: contains hedging language (might, could, maybe, probably, etc.)
            BELIEF: everything else (inferred from conversation but not verified)

        Args:
            content: The text to classify and store.
            source: Origin of the content (e.g. "user", "tool_output", "llm").
            context: Optional context string appended to reasoning.

        Returns:
            dict with keys: tier (str), trace_id (str), content (str)
        """
        text = content.strip()
        ctx_note = f" | ctx: {context}" if context else ""

        # Check speculation first — hedging overrides factual patterns
        if self._SPECULATION_PATTERNS.search(text):
            trace = self.store_speculation(
                content=text,
                source=source or "auto",
                reason=f"Hedging language detected{ctx_note}",
            )
            return {"tier": "SPECULATION", "trace_id": trace.trace_id, "content": text}

        # Check for fact-like patterns
        if self._FACT_PATTERNS.search(text):
            # Facts normally need VerificationResult, but auto-classified "facts"
            # are really strong beliefs (no artifact). Store as high-importance belief.
            trace = self.store_belief(
                content=text,
                source=source or "auto",
                reasoning=f"Auto-classified FACT-like (verifiable data pattern){ctx_note}",
                importance=0.7,
            )
            return {"tier": "BELIEF", "trace_id": trace.trace_id, "content": text,
                    "note": "Contains fact-like data but no artifact; stored as high-importance belief"}

        # Default: belief
        trace = self.store_belief(
            content=text,
            source=source or "auto",
            reasoning=f"Auto-classified belief (no hedging, no verifiable data){ctx_note}",
            importance=0.5,
        )
        return {"tier": "BELIEF", "trace_id": trace.trace_id, "content": text}

    def retrieve_facts(self, query: str, k: int = 5) -> List[VerifiedMemoryTrace]:
        """Retrieve only verified facts matching query"""
        with self._lock:
            facts = [t for t in self.traces.values() if t.tier == MemoryTier.FACT]
            return self._rank_by_relevance(facts, query, k)

    def retrieve_beliefs(self, query: str, k: int = 5) -> List[VerifiedMemoryTrace]:
        """Retrieve beliefs matching query"""
        with self._lock:
            beliefs = [t for t in self.traces.values() if t.tier == MemoryTier.BELIEF]
            return self._rank_by_relevance(beliefs, query, k)

    def retrieve_all(self, query: str, k: int = 10) -> List[VerifiedMemoryTrace]:
        """Retrieve from all tiers, marked by tier"""
        with self._lock:
            all_traces = list(self.traces.values())
            return self._rank_by_relevance(all_traces, query, k)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _rank_by_relevance(
        self,
        traces: List[VerifiedMemoryTrace],
        query: str,
        k: int
    ) -> List[VerifiedMemoryTrace]:
        """Rank traces by relevance to query.

        Scoring blend (when embeddings available):
            40% embedding cosine similarity
            40% keyword overlap
            20% importance weight

        Falls back to keyword-only + importance when embeddings are unavailable.
        """
        if not traces:
            return []

        query_words = set(query.lower().split())

        # Try to get query embedding (best-effort)
        query_emb = None
        if _EMBEDDINGS_AVAILABLE:
            try:
                query_emb = _get_embedding(query, timeout=2.0)
            except Exception as e:
                logger.debug(f"[TruthSpine] Query embedding fetch failed: {e}")

        use_embeddings = query_emb is not None

        scored = []
        for trace in traces:
            # --- keyword overlap score (0..1) ---
            content_words = set(trace.content.lower().split())
            overlap = len(query_words & content_words)
            keyword_score = overlap / max(len(query_words), 1)

            # --- embedding similarity score (0..1) ---
            emb_score = 0.0
            if use_embeddings:
                # Lazy-fill embedding if missing
                if trace.embedding is None:
                    self._cache_embedding(trace)
                if trace.embedding is not None:
                    raw_sim = self._cosine_similarity(query_emb, trace.embedding)
                    emb_score = max(0.0, raw_sim)  # clamp negatives

            # --- importance weight (0..1) ---
            imp = trace.importance

            # --- blended score ---
            if use_embeddings:
                score = 0.4 * emb_score + 0.4 * keyword_score + 0.2 * imp
            else:
                # Original behavior: keyword * importance
                score = keyword_score * imp

            if score > 0:
                trace.last_accessed = time.time()
                trace.access_count += 1
                scored.append((score, trace))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [t for _, t in scored[:k]]
        if scored:
            self._save()
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics by tier"""
        stats = {tier.value: 0 for tier in MemoryTier}
        for trace in self.traces.values():
            stats[trace.tier.value] += 1

        return {
            "total": len(self.traces),
            "by_tier": stats,
            "facts": stats[MemoryTier.FACT.value],
            "beliefs": stats[MemoryTier.BELIEF.value],
            "speculations": stats[MemoryTier.SPECULATION.value]
        }

    def promote_to_fact(
        self,
        trace_id: str,
        verification: VerificationResult
    ) -> Optional[VerifiedMemoryTrace]:
        """Promote a belief/speculation to fact with new verification"""
        with self._lock:
            if not verification.is_verified:
                return None

            trace = self.traces.get(trace_id)
            if not trace:
                return None

            trace.tier = MemoryTier.FACT
            trace.verification = verification
            trace.reasoning = f"Promoted to FACT: {verification.reasoning}"
            trace.importance = min(1.0, trace.importance + 0.2)

            self._save()
            return trace

    def demote_to_speculation(self, trace_id: str, reason: str) -> Optional[VerifiedMemoryTrace]:
        """Demote a fact/belief when verification fails"""
        with self._lock:
            trace = self.traces.get(trace_id)
            if not trace:
                return None

            trace.tier = MemoryTier.SPECULATION
            trace.reasoning = f"Demoted: {reason}"
            trace.importance = max(0.1, trace.importance - 0.3)

            self._save()
            return trace


# ============================================================================
#                    PART 5: SECURE TOOL EXECUTOR
# ============================================================================

@dataclass
class PendingConfirmation:
    """A tool execution waiting for user confirmation.

    SECURITY: Confirmations are now:
    - Bound to a specific session/user (can't be confirmed by different user)
    - Cryptographically secure IDs (not guessable)
    - Time-limited (5 min expiry)
    - Attempt-limited (3 max wrong attempts)
    """
    confirmation_id: str
    tool_name: str
    params: Dict[str, Any]
    reason: str
    session_id: str  # Binds to specific session/user
    signature: str   # HMAC signature for tamper detection
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 min expiry
    attempt_count: int = 0
    max_attempts: int = 3


class SecureToolExecutor:
    """
    Tool execution with REAL security.

    - Sandbox enforcement (not suggestions)
    - Confirmation requirement for dangerous operations
    - Session-bound confirmations (can't be hijacked)
    - All results must produce artifacts

    SECURITY IMPROVEMENTS (v5.1):
    - Cryptographically secure confirmation IDs using secrets module
    - Session binding prevents cross-user confirmation attacks
    - HMAC signatures prevent confirmation ID tampering
    - Attempt limiting prevents brute force
    """

    # Tools that require user confirmation
    DANGEROUS_TOOLS: ClassVar[set] = {"write_file", "delete_file", "execute_command", "modify_system"}

    # Tools restricted to sandbox
    SANDBOX_ONLY_TOOLS: ClassVar[set] = {"read_file", "write_file", "list_files", "delete_file"}

    def __init__(self, sandbox_dir: Path, verifier: VerifierSpine):
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.verifier = verifier

        self.pending_confirmations: Dict[str, PendingConfirmation] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}

        # Security: Secret key for HMAC signatures (regenerated each run)
        self._secret_key = secrets.token_hex(32)

        # Lock for thread-safe confirmation handling
        self._confirmation_lock = threading.Lock()

        # Track active sessions (bounded: evict oldest when exceeding cap)
        self._sessions: Dict[str, float] = {}  # session_id -> created_at
        self._MAX_SESSIONS = 500

        self._register_secure_tools()

    def _register_secure_tools(self):
        """Register built-in tools with security metadata"""

        self.tools["calculate"] = {
            "func": self._calculate,
            "requires_confirmation": False,
            "sandbox_only": False,
            "description": "Perform mathematical calculations"
        }

        self.tools["read_file"] = {
            "func": self._read_file,
            "requires_confirmation": False,
            "sandbox_only": True,
            "description": "Read file from sandbox"
        }

        self.tools["write_file"] = {
            "func": self._write_file,
            "requires_confirmation": True,
            "sandbox_only": True,
            "description": "Write file to sandbox"
        }

        self.tools["list_files"] = {
            "func": self._list_files,
            "requires_confirmation": False,
            "sandbox_only": True,
            "description": "List files in sandbox"
        }

        self.tools["delete_file"] = {
            "func": self._delete_file,
            "requires_confirmation": True,
            "sandbox_only": True,
            "description": "Delete file from sandbox"
        }

        self.tools["execute_python"] = {
            "func": self._execute_python,
            "requires_confirmation": True,
            "sandbox_only": True,
            "description": "Execute Python code in sandbox"
        }

    def _is_in_sandbox(self, path: Path) -> bool:
        """
        Check if path is within sandbox, with SYMLINK PROTECTION.

        SECURITY FIX: Resolves symlinks and checks that the final
        destination is still within sandbox. Prevents symlink attacks
        where attacker creates: sandbox/link -> /etc/passwd
        """
        try:
            resolved = path.resolve(strict=False)

            # Check for symlinks that point outside sandbox
            if path.exists() and path.is_symlink():
                try:
                    link_target = path.readlink()
                    # If symlink uses absolute path, check destination
                    if link_target.is_absolute():
                        target_resolved = link_target.resolve()
                    else:
                        target_resolved = (path.parent / link_target).resolve()

                    # Symlink target must be in sandbox
                    try:
                        target_resolved.relative_to(self.sandbox_dir)
                    except ValueError:
                        return False  # Symlink escapes sandbox!
                except (OSError, ValueError):
                    return False  # Can't read symlink = reject

            # Final check: resolved path must be in sandbox
            resolved.relative_to(self.sandbox_dir)
            return True
        except (ValueError, OSError):
            return False

    def _resolve_sandbox_path(self, path_str: str) -> Optional[Path]:
        """
        Resolve path within sandbox, return None if outside.

        SECURITY: Checks for directory traversal (../) and symlink attacks.
        """
        if not path_str:
            return None

        # Early reject obvious traversal attempts
        if ".." in path_str:
            return None

        requested = Path(path_str)
        if not requested.is_absolute():
            requested = self.sandbox_dir / requested

        # Resolve to absolute path (follows symlinks)
        try:
            resolved = requested.resolve(strict=False)
        except (OSError, ValueError):
            return None

        # Must be in sandbox
        if not self._is_in_sandbox(resolved):
            return None

        return resolved

    def create_session(self, user_id: str | None = None) -> str:
        """Create a new session for confirmation tracking.

        SECURITY: Sessions bind confirmations to specific users.
        """
        session_id = secrets.token_urlsafe(24)
        with self._confirmation_lock:
            # Evict oldest sessions if at capacity
            if len(self._sessions) >= self._MAX_SESSIONS:
                oldest = sorted(self._sessions, key=self._sessions.get)[:len(self._sessions) - self._MAX_SESSIONS + 1]
                for old_id in oldest:
                    del self._sessions[old_id]
            self._sessions[session_id] = time.time()
        return session_id

    def _generate_confirmation_id(self, session_id: str, tool_name: str) -> Tuple[str, str]:
        """Generate secure confirmation ID with HMAC signature.

        SECURITY: Uses cryptographically secure random + HMAC to prevent:
        - Guessing (256-bit entropy)
        - Tampering (HMAC signature)
        - Cross-session use (session binding)
        """
        # Cryptographically secure random ID
        confirmation_id = secrets.token_urlsafe(32)

        # HMAC signature binds ID to session and tool
        signature_data = f"{confirmation_id}:{session_id}:{tool_name}"
        signature = hmac.new(
            self._secret_key.encode(),
            signature_data.encode(),
            'sha256'
        ).hexdigest()[:32]  # 128-bit truncation for security

        return confirmation_id, signature

    def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        confirmed: bool = False,
        session_id: str | None = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with security checks.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            confirmed: Whether user has confirmed (for dangerous operations)
            session_id: Session ID for confirmation binding (required for dangerous tools)

        Returns:
            Result dict with success, result/error, and possibly needs_confirmation

        SECURITY (v5.1):
        - Confirmations are bound to sessions (can't be hijacked)
        - Confirmation IDs are cryptographically secure
        - HMAC signatures prevent tampering
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        # Check sandbox restriction
        if tool["sandbox_only"]:
            path = params.get("path") or params.get("file_path")
            if path:
                resolved = self._resolve_sandbox_path(path)
                if resolved is None:
                    return {
                        "success": False,
                        "error": "Sandbox violation",
                        "blocked_by": "sandbox_policy",
                        "requested_path": path,
                        "sandbox_dir": str(self.sandbox_dir)
                    }
                params["_resolved_path"] = resolved

        # Check confirmation requirement
        if tool["requires_confirmation"] and not confirmed:
            # Generate session if not provided
            if not session_id:
                session_id = self.create_session()

            # Generate secure confirmation ID with signature
            confirmation_id, signature = self._generate_confirmation_id(session_id, tool_name)

            with self._confirmation_lock:
                self.pending_confirmations[confirmation_id] = PendingConfirmation(
                    confirmation_id=confirmation_id,
                    tool_name=tool_name,
                    params=params,
                    reason=f"Tool '{tool_name}' requires confirmation before execution",
                    session_id=session_id,
                    signature=signature
                )

            return {
                "success": False,
                "needs_confirmation": True,
                "confirmation_id": confirmation_id,
                "session_id": session_id,  # Client must provide this to confirm
                "tool_name": tool_name,
                "reason": f"Tool '{tool_name}' requires user confirmation",
                "message": f"Please confirm execution of {tool_name} with params: {params}",
                "expires_in": 300
            }

        # Execute the tool
        try:
            result = tool["func"](params)
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "exception": type(e).__name__}

    def confirm(self, confirmation_id: str, session_id: str | None = None) -> Dict[str, Any]:
        """
        Confirm and execute a pending operation.

        SECURITY (v5.1):
        - Session ID must match the one used to create the confirmation
        - Signature is verified to prevent tampering
        - Attempt limiting prevents brute force attacks
        - Expired confirmations are rejected

        Args:
            confirmation_id: The confirmation ID returned by execute()
            session_id: The session ID (must match original request)

        Returns:
            Result of tool execution, or error if validation fails
        """
        with self._confirmation_lock:
            pending = self.pending_confirmations.get(confirmation_id)

            if not pending:
                return {"success": False, "error": f"No pending confirmation: {confirmation_id}"}

            # Check session binding — require matching session when pending has one
            if pending.session_id and session_id != pending.session_id:
                # Different user/session trying to confirm - possible attack
                pending.attempt_count += 1
                if pending.attempt_count >= pending.max_attempts:
                    # Too many wrong attempts - delete confirmation
                    del self.pending_confirmations[confirmation_id]
                    return {
                        "success": False,
                        "error": "Confirmation invalidated due to too many failed attempts",
                        "blocked_by": "security_policy"
                    }
                return {
                    "success": False,
                    "error": "Session mismatch - cannot confirm from different session",
                    "attempts_remaining": pending.max_attempts - pending.attempt_count
                }

            # Verify HMAC signature
            expected_signature_data = f"{confirmation_id}:{pending.session_id}:{pending.tool_name}"
            expected_signature = hmac.new(
                self._secret_key.encode(),
                expected_signature_data.encode(),
                'sha256'
            ).hexdigest()[:32]  # 128-bit truncation for security

            if not hmac.compare_digest(pending.signature, expected_signature):
                # Signature mismatch - possible tampering
                del self.pending_confirmations[confirmation_id]
                return {
                    "success": False,
                    "error": "Confirmation signature invalid",
                    "blocked_by": "security_policy"
                }

            # Check expiry
            if time.time() > pending.expires_at:
                del self.pending_confirmations[confirmation_id]
                return {"success": False, "error": "Confirmation expired"}

            # All checks passed - remove and execute
            del self.pending_confirmations[confirmation_id]

        # Execute with confirmed=True (outside lock to avoid deadlock)
        return self.execute(pending.tool_name, pending.params, confirmed=True)

    def get_pending_confirmations(self, session_id: str | None = None) -> List[Dict[str, Any]]:
        """Get list of pending confirmations.

        Args:
            session_id: If provided, only return confirmations for this session

        Returns:
            List of pending confirmation details
        """
        with self._confirmation_lock:
            # Clean expired
            now = time.time()
            expired = [cid for cid, p in self.pending_confirmations.items() if now > p.expires_at]
            for cid in expired:
                del self.pending_confirmations[cid]

            confirmations = self.pending_confirmations.values()

            # Filter by session if provided
            if session_id:
                confirmations = [p for p in confirmations if p.session_id == session_id]

            return [
                {
                    "confirmation_id": p.confirmation_id,
                    "tool_name": p.tool_name,
                    "params": {k: v for k, v in p.params.items() if not k.startswith("_")},
                    "reason": p.reason,
                    "expires_in": int(p.expires_at - now)
                }
                for p in confirmations
            ]

    # =========================================================================
    #                      TOOL IMPLEMENTATIONS
    # =========================================================================

    def _calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform calculation using safe AST-based evaluator (NO eval()).

        SECURITY FIX: Replaced eval() with SafeCalculator to prevent
        code injection attacks like: ().__class__.__bases__[0].__subclasses__()
        """
        expression = params.get("expression", "")

        if not expression:
            return {"success": False, "error": "No expression provided"}

        try:
            # Use safe AST-based calculator instead of eval()
            result = safe_eval(expression)
            return {
                "success": True,
                "result": result,
                "expression": expression,
                "stdout": str(result),
                "returncode": 0
            }
        except ValueError as e:
            return {"success": False, "error": str(e), "returncode": 1}
        except ZeroDivisionError:
            return {"success": False, "error": "Division by zero", "returncode": 1}
        except Exception as e:
            return {"success": False, "error": f"Calculation error: {e}", "returncode": 1}

    def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read file from sandbox"""
        path = params.get("_resolved_path") or self._resolve_sandbox_path(
            params.get("path", params.get("file_path", ""))
        )

        if not path:
            return {"success": False, "error": "Invalid path"}

        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return {
                "success": True,
                "content": content,
                "path": str(path),
                "size": len(content),
                "stdout": content,
                "returncode": 0
            }
        except (OSError, IOError) as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write file to sandbox"""
        path = params.get("_resolved_path") or self._resolve_sandbox_path(
            params.get("path", params.get("file_path", ""))
        )
        content = params.get("content", "")

        if not path:
            return {"success": False, "error": "Invalid path"}

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            return {
                "success": True,
                "path": str(path),
                "size": len(content),
                "hash": content_hash,
                "stdout": f"Written {len(content)} bytes to {path}",
                "returncode": 0
            }
        except (OSError, IOError) as e:
            return {"success": False, "error": str(e)}

    def _list_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List files in sandbox"""
        path = params.get("_resolved_path") or self._resolve_sandbox_path(
            params.get("path", ".")
        )

        if not path:
            path = self.sandbox_dir

        try:
            files = []
            for item in path.iterdir():
                # Skip items that resolve outside the sandbox (e.g. symlinks)
                if not self._is_in_sandbox(item.resolve()):
                    continue
                files.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0
                })

            return {
                "success": True,
                "files": files,
                "path": str(path),
                "count": len(files),
                "stdout": "\n".join(f["name"] for f in files),
                "returncode": 0
            }
        except (OSError, IOError) as e:
            return {"success": False, "error": str(e)}

    def _delete_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete file from sandbox"""
        path = params.get("_resolved_path") or self._resolve_sandbox_path(
            params.get("path", params.get("file_path", ""))
        )

        if not path:
            return {"success": False, "error": "Invalid path"}

        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            return {
                "success": True,
                "deleted": str(path),
                "stdout": f"Deleted: {path}",
                "returncode": 0
            }
        except (OSError, IOError) as e:
            return {"success": False, "error": str(e)}

    def _execute_python(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in sandbox"""
        code = params.get("code", "")
        timeout = params.get("timeout", 30)

        # Write code to temp file in sandbox
        code_file = self.sandbox_dir / f"_exec_{uuid.uuid4().hex[:8]}.py"

        try:
            code_file.write_text(code, encoding="utf-8")

            # Sanitize environment — do not leak API keys to executed code
            import sys as _sys
            _SAFE_ENV_KEYS = {"PATH", "HOME", "USERPROFILE", "TEMP", "TMP",
                              "SYSTEMROOT", "WINDIR", "COMSPEC", "PYTHONPATH"}
            safe_env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
            result = subprocess.run(
                [_sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.sandbox_dir),
                env=safe_env,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timeout after {timeout}s", "returncode": -1}
        except Exception as e:
            return {"success": False, "error": str(e), "returncode": -1}
        finally:
            if code_file.exists():
                code_file.unlink()


# ============================================================================
#                              TESTS
# ============================================================================

def test_truth_spine():
    """Test the truth spine components"""
    logger.debug("\n" + "="*60)
    logger.debug("TRUTH SPINE TESTS")
    logger.debug("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir) / "sandbox"
        sandbox.mkdir()

        # Test 1: Artifact creation
        logger.debug("\n[TEST 1] Artifact Creation")
        test_file = sandbox / "test.txt"
        test_file.write_text("Hello, World!")

        artifact = Artifact.from_file(test_file)
        logger.debug(f"  File artifact: {artifact.artifact_type.value}")
        logger.debug(f"  Hash: {artifact.content_hash[:16]}...")
        assert artifact.is_valid, "Artifact should be valid"
        logger.debug("  PASSED")

        # Test 2: Verification checks
        logger.debug("\n[TEST 2] Verification Checks")
        verifier = VerifierSpine(sandbox)

        result = verifier.verify_action(
            "file_write",
            {"path": str(test_file), "success": True},
            context={"expected_path": str(test_file)}
        )
        logger.debug(f"  Verified: {result.is_verified}")
        logger.debug(f"  Passed: {result.checks_passed}")
        logger.debug(f"  Failed: {result.checks_failed}")
        assert result.is_verified, "File write should verify"
        logger.debug("  PASSED")

        # Test 3: Memory tiers
        logger.debug("\n[TEST 3] Memory Tiers")
        memory = VerifiedMemory(sandbox / "memory")

        # Store fact (with verification)
        fact = memory.store_fact("File created at test.txt", result, "test")
        assert fact is not None, "Should store fact"
        assert fact.tier == MemoryTier.FACT
        logger.debug(f"  Stored FACT: {fact.trace_id}")

        # Store belief
        belief = memory.store_belief("User might want more files", "inference", "Pattern detected")
        assert belief.tier == MemoryTier.BELIEF
        logger.debug(f"  Stored BELIEF: {belief.trace_id}")

        # Store speculation
        spec = memory.store_speculation("LLM says hello", "llm", "Unverified LLM output")
        assert spec.tier == MemoryTier.SPECULATION
        logger.debug(f"  Stored SPECULATION: {spec.trace_id}")

        # Retrieve
        facts = memory.retrieve_facts("file", k=5)
        logger.debug(f"  Retrieved {len(facts)} facts")
        assert len(facts) == 1, "Should find 1 fact"
        logger.debug("  PASSED")

        # Test 4: Secure executor
        logger.debug("\n[TEST 4] Secure Tool Executor")
        executor = SecureToolExecutor(sandbox, verifier)

        # Calculate (no confirmation needed)
        result = executor.execute("calculate", {"expression": "2 + 2"})
        logger.debug(f"  Calculate: {result}")
        assert result["success"] and result["result"] == 4

        # Write file (needs confirmation)
        result = executor.execute("write_file", {"path": "new.txt", "content": "data"}, confirmed=False)
        logger.debug(f"  Write (unconfirmed): needs_confirmation={result.get('needs_confirmation')}")
        assert result.get("needs_confirmation"), "Should need confirmation"

        # Confirm it
        confirmed = executor.confirm(result["confirmation_id"])
        logger.debug(f"  Write (confirmed): success={confirmed.get('success')}")
        assert confirmed["success"], "Should succeed after confirmation"

        # Try to read outside sandbox
        result = executor.execute("read_file", {"path": "/etc/passwd"})
        logger.debug(f"  Read outside sandbox: blocked={result.get('blocked_by')}")
        assert result.get("blocked_by") == "sandbox_policy", "Should block"

        logger.debug("  PASSED")

        # Stats
        logger.debug("\n[STATS]")
        logger.debug(f"  Verifier: {verifier.get_stats()}")
        logger.debug(f"  Memory: {memory.get_stats()}")

    logger.debug("\n" + "="*60)
    logger.debug("ALL TESTS PASSED")
    logger.debug("="*60)


if __name__ == "__main__":
    test_truth_spine()
