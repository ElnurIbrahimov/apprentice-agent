"""
██████╗ ██████╗  ██████╗ ████████╗ ██████╗        █████╗  ██████╗ ██╗    ██╗   ██╗███████╗
██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗      ██╔══██╗██╔════╝ ██║    ██║   ██║██╔════╝
██████╔╝██████╔╝██║   ██║   ██║   ██║   ██║█████╗███████║██║  ███╗██║    ██║   ██║███████╗
██╔═══╝ ██╔══██╗██║   ██║   ██║   ██║   ██║╚════╝██╔══██║██║   ██║██║    ╚██╗ ██╔╝╚════██║
██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝      ██║  ██║╚██████╔╝██║     ╚████╔╝ ███████║
╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝       ╚═╝  ╚═╝ ╚═════╝ ╚═╝      ╚═══╝  ╚══════╝

PROTO-AGI CORE v5 — TRUTH SPINE INTEGRATION
============================================

Evolution:
- v1: Basic agent loop
- v2: Needs + governance
- v3: Evidence-based cognition (probes, observable facts)
- v4: (skipped)
- v5: TRUTH SPINE - Non-negotiable verification layer

Core Principle: "If you can't verify it with an artifact, it's SPECULATION"

The Truth Spine Contract:
    ACTION → ARTIFACT → VERIFICATION → MEMORY TIER

    FACT = verified with artifact (hash, return code, file exists)
    BELIEF = inferred but not proven
    SPECULATION = unverified claims (including LLM output)
"""

import asyncio
import json
import logging
import time
import threading
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Import Truth Spine components
from .truth_spine import (
    Artifact,
    ArtifactType,
    VerifierSpine,
    VerificationResult,
    VerificationCheck,
    VerifiedMemory,
    VerifiedMemoryTrace,
    MemoryTier,
    SecureToolExecutor,
    PendingConfirmation
)


# ============================================================================
#                    PART 1: ACTION TYPES & STRUCTURES
# ============================================================================

class ActionType(Enum):
    """Structured action types"""
    # Safe actions (always allowed)
    RECALL = auto()
    THINK = auto()
    RESPOND = auto()

    # Moderate actions (budget-controlled)
    SEARCH = auto()
    READ_FILE = auto()
    ANALYZE = auto()

    # Sensitive actions (require confirmation)
    WRITE_FILE = auto()
    EXECUTE_CODE = auto()
    SEND_MESSAGE = auto()
    API_CALL = auto()

    # Dangerous actions (always require explicit approval)
    DELETE = auto()
    SYSTEM_MODIFY = auto()
    SEND_EMAIL = auto()


@dataclass
class ActionRequest:
    """
    A request to perform an action.

    The Truth Spine will verify the result and determine memory tier.
    """
    action_type: ActionType
    intent: str
    params: Dict[str, Any]
    expected_checks: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0

    @property
    def id(self) -> str:
        return hashlib.md5(
            f"{self.action_type.name}{self.intent}{json.dumps(self.params, sort_keys=True)}".encode()
        ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type.name,
            "intent": self.intent,
            "params": self.params,
            "expected_checks": self.expected_checks,
            "timeout": self.timeout_seconds
        }


@dataclass
class ActionResult:
    """
    Result of an action with full verification.
    """
    request: ActionRequest
    raw_result: Dict[str, Any]
    verification: VerificationResult
    memory_tier: MemoryTier
    memory_trace_id: Optional[str]
    timestamp: float = field(default_factory=time.time)

    @property
    def is_verified(self) -> bool:
        return self.verification.is_verified

    @property
    def is_fact(self) -> bool:
        return self.memory_tier == MemoryTier.FACT

    def to_dict(self) -> dict:
        return {
            "request": self.request.to_dict(),
            "verification": self.verification.to_dict(),
            "memory_tier": self.memory_tier.value,
            "memory_trace_id": self.memory_trace_id,
            "is_verified": self.is_verified,
            "timestamp": self.timestamp
        }


# ============================================================================
#                    PART 2: GOVERNANCE (v3 compatible)
# ============================================================================

class OperationMode(Enum):
    IDLE = "idle"
    ASSIST = "assist"
    OPERATE = "operate"


@dataclass
class ActionBudget:
    """Limits on autonomous actions"""
    max_actions_per_hour: int = 10
    max_messages_per_hour: int = 3

    actions_used: int = 0
    messages_used: int = 0
    hour_started: float = field(default_factory=time.time)

    def _reset_if_needed(self):
        if time.time() - self.hour_started > 3600:
            self.actions_used = 0
            self.messages_used = 0
            self.hour_started = time.time()

    def can_act(self) -> bool:
        self._reset_if_needed()
        return self.actions_used < self.max_actions_per_hour

    def can_message(self) -> bool:
        self._reset_if_needed()
        return self.messages_used < self.max_messages_per_hour

    def consume_action(self):
        self.actions_used += 1

    def consume_message(self):
        self.messages_used += 1


class Governor:
    """Governance layer - controls what actions can be taken"""

    SAFE_ACTIONS = {ActionType.RECALL, ActionType.THINK, ActionType.RESPOND}
    MODERATE_ACTIONS = {ActionType.SEARCH, ActionType.READ_FILE, ActionType.ANALYZE}
    SENSITIVE_ACTIONS = {ActionType.WRITE_FILE, ActionType.EXECUTE_CODE,
                         ActionType.SEND_MESSAGE, ActionType.API_CALL}
    DANGEROUS_ACTIONS = {ActionType.DELETE, ActionType.SYSTEM_MODIFY, ActionType.SEND_EMAIL}

    def __init__(self):
        self.mode = OperationMode.ASSIST
        self.budget = ActionBudget()
        self.pending_approvals: List[ActionRequest] = []

    def set_mode(self, mode: OperationMode):
        self.mode = mode

    def can_act(self, request: ActionRequest, is_user_initiated: bool = False) -> Tuple[bool, str]:
        """Check if action CAN be permitted - DOES NOT consume budget"""
        action_type = request.action_type

        if is_user_initiated:
            if not self.budget.can_act():
                return False, "Budget exceeded"
            return True, "User initiated"

        if self.mode == OperationMode.IDLE:
            if action_type in self.SAFE_ACTIONS:
                return True, "Safe action in IDLE mode"
            return False, "Only internal actions in IDLE mode"

        if self.mode == OperationMode.ASSIST:
            if action_type in self.SAFE_ACTIONS:
                return True, "Safe action in ASSIST mode"
            return False, "Autonomous actions not permitted in ASSIST mode"

        if self.mode == OperationMode.OPERATE:
            if action_type in self.SAFE_ACTIONS:
                return True, "Safe action"

            if action_type in self.MODERATE_ACTIONS:
                if self.budget.can_act():
                    return True, "Moderate action within budget"
                return False, "Budget exceeded"

            if action_type in self.SENSITIVE_ACTIONS:
                if request not in self.pending_approvals:
                    self.pending_approvals.append(request)
                return False, "Sensitive action requires approval"

            if action_type in self.DANGEROUS_ACTIONS:
                if request not in self.pending_approvals:
                    self.pending_approvals.append(request)
                return False, "Dangerous action requires explicit approval"

        return False, "Unknown action type"

    def grant_permission(self, request: ActionRequest) -> bool:
        """Actually grant permission and consume budget"""
        if request.action_type not in self.SAFE_ACTIONS:
            self.budget.consume_action()
        return True

    def can_message(self) -> Tuple[bool, str]:
        if self.mode != OperationMode.OPERATE:
            return False, "Proactive messaging requires OPERATE mode"
        if not self.budget.can_message():
            return False, "Message budget exceeded"
        return True, "Message permitted"

    def grant_message(self):
        self.budget.consume_message()

    def approve(self, request_id: str) -> Optional[ActionRequest]:
        for request in self.pending_approvals:
            if request.id == request_id:
                self.pending_approvals.remove(request)
                return request
        return None


# ============================================================================
#                    PART 3: NEEDS SYSTEM (unchanged from v3)
# ============================================================================

class DriveType(Enum):
    SURVIVE = "survive"
    CONNECT = "connect"
    UNDERSTAND = "understand"
    EXPRESS = "express"
    IMPROVE = "improve"
    EXPLORE = "explore"


@dataclass
class Need:
    drive: DriveType
    level: float = 100.0
    decay_rate: float = 1.0
    last_satisfied: float = field(default_factory=time.time)

    @property
    def urgency(self) -> float:
        return max(0, min(1, 1 - (self.level / 100)))

    def decay(self, minutes: float):
        self.level = max(0, self.level - (self.decay_rate * minutes))

    def satisfy(self, amount: float = 30):
        self.level = min(100, self.level + amount)
        self.last_satisfied = time.time()


class NeedSystem:
    def __init__(self):
        self.needs = {
            DriveType.SURVIVE: Need(DriveType.SURVIVE, decay_rate=0.5),
            DriveType.CONNECT: Need(DriveType.CONNECT, decay_rate=2.0),
            DriveType.UNDERSTAND: Need(DriveType.UNDERSTAND, decay_rate=1.0),
            DriveType.EXPRESS: Need(DriveType.EXPRESS, decay_rate=1.5),
            DriveType.IMPROVE: Need(DriveType.IMPROVE, decay_rate=0.3),
            DriveType.EXPLORE: Need(DriveType.EXPLORE, decay_rate=1.2),
        }

    def decay_all(self, minutes: float):
        for need in self.needs.values():
            need.decay(minutes)

    def get_dominant_drive(self) -> DriveType:
        return max(self.needs.values(), key=lambda n: n.urgency).drive

    def satisfy(self, drive: DriveType, amount: float = 30):
        if drive in self.needs:
            self.needs[drive].satisfy(amount)

    def to_dict(self) -> dict:
        return {d.value: {"level": round(n.level, 1), "urgency": round(n.urgency, 2)}
                for d, n in self.needs.items()}


# ============================================================================
#                    PART 4: PROTO-AGI v5 - TRUTH SPINE INTEGRATED
# ============================================================================

class ProtoAGI:
    """
    PROTO-AGI v5 — TRUTH SPINE INTEGRATION

    Every action follows the non-negotiable verification contract:
        ACTION → ARTIFACT → VERIFICATION → MEMORY TIER

    Key properties:
    1. All actions produce artifacts that are verified
    2. Memory has 3 tiers: FACT (verified), BELIEF (inferred), SPECULATION (unverified)
    3. Dangerous tools require user confirmation
    4. Sandbox enforcement - not suggestions, enforcement
    5. LLM output is SPECULATION until verified
    """

    def __init__(self,
                 llm_func: Callable = None,
                 action_func: Callable = None,
                 output_func: Callable = None,
                 data_path: str = "data/proto_agi_v5/"):

        self.llm = llm_func
        self.action_func = action_func
        self.output_func = output_func
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Sandbox directory
        self.sandbox_dir = self.data_path / "sandbox"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Core Truth Spine components
        self.verifier = VerifierSpine(self.sandbox_dir)
        self.memory = VerifiedMemory(self.data_path / "memory")
        self.executor = SecureToolExecutor(self.sandbox_dir, self.verifier)

        # Governance
        self.governor = Governor()
        self.needs = NeedSystem()

        # State (protected by lock for thread safety)
        self._state_lock = threading.Lock()
        self._is_running = False
        self._cycle_count = 0
        self._last_cycle = time.time()
        self._last_chat_id = None

        self._load_state()

    @property
    def is_running(self) -> bool:
        """Thread-safe access to running state."""
        with self._state_lock:
            return self._is_running

    @is_running.setter
    def is_running(self, value: bool):
        """Thread-safe setting of running state."""
        with self._state_lock:
            self._is_running = value

    @property
    def cycle_count(self) -> int:
        """Thread-safe access to cycle count."""
        with self._state_lock:
            return self._cycle_count

    @cycle_count.setter
    def cycle_count(self, value: int):
        """Thread-safe setting of cycle count."""
        with self._state_lock:
            self._cycle_count = value

    @property
    def last_cycle(self) -> float:
        """Thread-safe access to last cycle time."""
        with self._state_lock:
            return self._last_cycle

    @last_cycle.setter
    def last_cycle(self, value: float):
        """Thread-safe setting of last cycle time."""
        with self._state_lock:
            self._last_cycle = value

    def _load_state(self):
        """Load state from disk. Handles errors gracefully."""
        state_file = self.data_path / "state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                with self._state_lock:
                    self._cycle_count = data.get("cycle_count", 0)
                self.governor.mode = OperationMode(data.get("mode", "assist"))
            except (json.JSONDecodeError, ValueError, IOError) as e:
                # Log error but continue with defaults
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        (self.data_path / "state.json").write_text(json.dumps({
            "cycle_count": self.cycle_count,
            "mode": self.governor.mode.value,
            "last_save": time.time()
        }, indent=2))

    # =========================================================================
    #                      TRUTH SPINE EXECUTION FLOW
    # =========================================================================

    def process(self, request: ActionRequest, is_user_initiated: bool = False) -> ActionResult:
        """
        Process an action request through the Truth Spine.

        The contract:
            ACTION → ARTIFACT → VERIFICATION → MEMORY TIER
        """

        # Step 1: Governance check
        can_act, reason = self.governor.can_act(request, is_user_initiated)

        if not can_act:
            # Store as speculation - we tried but weren't allowed
            trace = self.memory.store_speculation(
                f"Attempted {request.action_type.name}: {request.intent}. Blocked: {reason}",
                source="governance",
                reason=reason
            )

            return ActionResult(
                request=request,
                raw_result={"blocked": True, "reason": reason},
                verification=VerificationResult(
                    is_verified=False,
                    artifact=Artifact.empty(reason),
                    checks_passed=[],
                    checks_failed=["governance: " + reason],
                    reasoning=reason
                ),
                memory_tier=MemoryTier.SPECULATION,
                memory_trace_id=trace.trace_id
            )

        # Step 2: Grant permission (consumes budget)
        self.governor.grant_permission(request)

        # Step 3: Execute the action
        raw_result = self._execute_action(request)

        # Step 4: Determine action type for verification
        action_type_map = {
            ActionType.READ_FILE: "file_read",
            ActionType.WRITE_FILE: "file_write",
            ActionType.EXECUTE_CODE: "command",
            ActionType.SEARCH: "search",
            ActionType.ANALYZE: "calculate"
        }
        verification_type = action_type_map.get(request.action_type, "default")

        # Step 5: Verify through Truth Spine
        context = {**request.params}
        if "path" in request.params:
            context["expected_path"] = request.params["path"]

        verification = self.verifier.verify_action(
            verification_type,
            raw_result,
            expected_checks=request.expected_checks if request.expected_checks else None,
            context=context
        )

        # Step 6: Determine memory tier based on verification
        memory_tier = self._determine_memory_tier(request, verification)

        # Step 7: Store in appropriate memory tier
        trace_id = self._store_in_memory(request, verification, memory_tier)

        # Step 8: Satisfy needs if successful
        if verification.is_verified:
            self.needs.satisfy(DriveType.IMPROVE, 15)

        return ActionResult(
            request=request,
            raw_result=raw_result,
            verification=verification,
            memory_tier=memory_tier,
            memory_trace_id=trace_id
        )

    def _execute_action(self, request: ActionRequest) -> Dict[str, Any]:
        """Execute the action and return raw result"""

        # Map action types to tool execution
        if request.action_type == ActionType.EXECUTE_CODE:
            code = request.params.get("code", "")
            return self.executor.execute("execute_python", {"code": code}, confirmed=True)

        elif request.action_type == ActionType.READ_FILE:
            path = request.params.get("path", "")
            return self.executor.execute("read_file", {"path": path})

        elif request.action_type == ActionType.WRITE_FILE:
            path = request.params.get("path", "")
            content = request.params.get("content", "")
            # Write file requires confirmation via executor
            result = self.executor.execute("write_file", {"path": path, "content": content})

            if result.get("needs_confirmation"):
                # Auto-confirm for user-initiated requests
                return self.executor.confirm(result["confirmation_id"])
            return result

        elif request.action_type in [ActionType.SEARCH, ActionType.API_CALL]:
            # Use external action_func if available
            if self.action_func:
                try:
                    result = self.action_func(request.intent)
                    return {
                        "success": result.get("success", True),
                        "result": result,
                        "stdout": str(result.get("result", "")),
                        "returncode": 0 if result.get("success") else 1
                    }
                except Exception as e:
                    return {"success": False, "error": str(e), "returncode": 1}
            return {"success": False, "error": "No action function available"}

        elif request.action_type == ActionType.ANALYZE:
            expr = request.params.get("expression", "")
            return self.executor.execute("calculate", {"expression": expr})

        elif request.action_type in [ActionType.RECALL, ActionType.THINK, ActionType.RESPOND]:
            # Safe internal actions - always succeed
            return {
                "success": True,
                "result": request.intent,
                "stdout": request.intent,
                "returncode": 0
            }

        else:
            return {"success": False, "error": f"Unhandled action type: {request.action_type.name}"}

    def _determine_memory_tier(self, request: ActionRequest, verification: VerificationResult) -> MemoryTier:
        """
        Determine which memory tier based on verification.

        FACT = verified with artifact
        BELIEF = partially verified or inferred
        SPECULATION = not verified
        """
        if verification.is_verified:
            return MemoryTier.FACT

        # Partial verification = belief
        if verification.checks_passed and len(verification.checks_passed) > len(verification.checks_failed):
            return MemoryTier.BELIEF

        return MemoryTier.SPECULATION

    def _store_in_memory(self, request: ActionRequest, verification: VerificationResult, tier: MemoryTier) -> str:
        """Store result in appropriate memory tier"""
        content = f"{request.action_type.name}: {request.intent}"

        if tier == MemoryTier.FACT:
            trace = self.memory.store_fact(
                content=content,
                verification=verification,
                source=f"action:{request.id}"
            )
            return trace.trace_id if trace else None

        elif tier == MemoryTier.BELIEF:
            trace = self.memory.store_belief(
                content=content,
                source=f"action:{request.id}",
                reasoning=verification.reasoning
            )
            return trace.trace_id

        else:  # SPECULATION
            trace = self.memory.store_speculation(
                content=content,
                source=f"action:{request.id}",
                reason=verification.reasoning
            )
            return trace.trace_id

    # =========================================================================
    #                      CONFIRMATION WORKFLOW
    # =========================================================================

    def confirm(self, confirmation_id: str) -> Dict[str, Any]:
        """
        Confirm a pending dangerous operation.

        This is the ONLY way to execute confirmed dangerous tools.
        """
        return self.executor.confirm(confirmation_id)

    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """Get all pending tool confirmations"""
        return self.executor.get_pending_confirmations()

    # =========================================================================
    #                      USER INTERACTION
    # =========================================================================

    def process_input(self, user_input: str, chat_id: str = None) -> str:
        """
        Process user input with Truth Spine awareness.

        LLM responses are stored as SPECULATION until verified.
        """
        if chat_id:
            self._last_chat_id = chat_id

        self.needs.satisfy(DriveType.CONNECT, 40)

        # Store user input as fact (we can verify we received it)
        artifact = Artifact.from_json({"user_input": user_input, "chat_id": chat_id})
        verification = self.verifier.verify_action("default", {"success": True, "result": user_input})

        self.memory.store_fact(
            f"User said: {user_input[:100]}",
            verification,
            source=f"user:{chat_id or 'unknown'}"
        )

        # Build context from verified facts
        context_parts = self._build_grounded_context(user_input)

        prompt = f"""You are AURA with Truth Spine v5 - verification-first cognition.

GROUNDED CONTEXT (Facts are VERIFIED, Beliefs are INFERRED, Speculations are UNVERIFIED):
{context_parts}

RESPONSE RULES:
1. Label claims by verification status:
   - [FACT:id] for verified facts from memory
   - [BELIEF] for logical inferences
   - [SPECULATION] for unverified claims
   - [UNKNOWN] for things you need to verify
2. Suggest verification steps for uncertain claims
3. Don't claim certainty without artifact proof

USER INPUT: {user_input}

Respond helpfully while maintaining verification discipline."""

        if self.llm:
            response = self.llm(prompt)
        else:
            # Fallback without LLM
            facts = self.memory.retrieve_facts(user_input, k=3)
            if facts:
                fact_strs = [f"[FACT:{t.trace_id}] {t.content}" for t in facts]
                response = "Based on verified facts:\n" + "\n".join(fact_strs)
            else:
                response = "[SPECULATION] I don't have verified facts about this. Would you like me to investigate?"

        # Store response as SPECULATION (LLM output is unverified)
        self.memory.store_speculation(
            f"I responded: {response[:200]}",
            source="llm_response",
            reason="LLM output - not verified with artifact"
        )

        self.needs.satisfy(DriveType.EXPRESS, 20)

        return response

    def _build_grounded_context(self, query: str) -> str:
        """Build context string with memory tier labels"""
        facts = self.memory.retrieve_facts(query, k=5)
        beliefs = self.memory.retrieve_beliefs(query, k=3)
        all_traces = self.memory.retrieve_all(query, k=3)

        parts = []

        if facts:
            parts.append("VERIFIED FACTS:")
            for trace in facts:
                parts.append(f"  [{trace.trace_id}] {trace.content}")
        else:
            parts.append("VERIFIED FACTS: None relevant found")

        if beliefs:
            parts.append("\nBELIEFS (inferred):")
            for trace in beliefs:
                parts.append(f"  [{trace.trace_id}] {trace.content}")

        # Add speculations separately labeled
        speculations = [t for t in all_traces if t.tier == MemoryTier.SPECULATION]
        if speculations:
            parts.append("\nSPECULATIONS (unverified):")
            for trace in speculations[:2]:
                parts.append(f"  [{trace.trace_id}] {trace.content}")

        return "\n".join(parts)

    # =========================================================================
    #                      COGNITIVE CYCLE
    # =========================================================================

    def cycle(self) -> dict:
        """One cycle of Truth Spine cognition"""
        cycle_start = time.time()
        result = {
            "cycle": self.cycle_count,
            "mode": self.governor.mode.value,
            "phases": {}
        }

        try:
            # PERCEIVE
            elapsed = (cycle_start - self.last_cycle) / 60
            self.needs.decay_all(elapsed)
            result["phases"]["perceive"] = {"elapsed_min": round(elapsed, 2)}

            # WANT
            dominant = self.needs.get_dominant_drive()
            result["phases"]["want"] = {"dominant": dominant.value}

            # THINK
            result["phases"]["think"] = {
                "memory_stats": self.memory.get_stats(),
                "verifier_stats": self.verifier.get_stats()
            }

            # ACT (only in OPERATE mode)
            if self.governor.mode == OperationMode.OPERATE:
                action = self._generate_action_from_drive(dominant)
                if action:
                    can_proceed, reason = self.governor.can_act(action, is_user_initiated=False)
                    if not can_proceed:
                        result["phases"]["act"] = {"skipped": reason}
                        action = None
                if action:
                    action_result = self.process(action)
                    result["phases"]["act"] = {
                        "action": action.action_type.name,
                        "verified": action_result.is_verified,
                        "memory_tier": action_result.memory_tier.value
                    }

                    if action_result.is_verified:
                        self.needs.satisfy(dominant, 25)
            else:
                result["phases"]["act"] = {"skipped": "not in OPERATE mode"}

            # EXPRESS
            if self.needs.needs[DriveType.CONNECT].urgency > 0.7:
                can_msg, _ = self.governor.can_message()
                if can_msg:
                    message = self._generate_grounded_message()
                    if message:
                        self.governor.grant_message()
                        result["phases"]["express"] = {"message": message[:50]}

                        if self.output_func:
                            try:
                                self.output_func(message, self._last_chat_id)
                            except Exception as e:
                                result["phases"]["express"]["error"] = str(e)

                        self.needs.satisfy(DriveType.CONNECT, 30)

        except Exception as e:
            result["error"] = str(e)
            self.memory.store_speculation(f"Cycle error: {e}", "system", "Exception during cycle")

        finally:
            self.cycle_count += 1
            self.last_cycle = cycle_start
            self._save_state()

        return result

    def _generate_action_from_drive(self, drive: DriveType) -> Optional[ActionRequest]:
        """Generate action request from drive"""
        if drive == DriveType.UNDERSTAND:
            return ActionRequest(
                action_type=ActionType.ANALYZE,
                intent="Analyze system state",
                params={"expression": "1+1"},
                expected_checks=["no_error", "not_empty"]
            )
        elif drive == DriveType.EXPLORE:
            return ActionRequest(
                action_type=ActionType.READ_FILE,
                intent="Explore sandbox contents",
                params={"path": "."},
                expected_checks=["not_empty"]
            )
        return None

    def _generate_grounded_message(self) -> Optional[str]:
        """Generate proactive message grounded in verified facts"""
        facts = self.memory.retrieve_facts("user", k=2)

        if facts:
            trace = facts[0]
            return f"[FACT:{trace.trace_id}] I remember: {trace.content}. How's that going?"
        else:
            return "[SPECULATION] I'd like to learn more about you. What are you working on?"

    # =========================================================================
    #                      CONTROL INTERFACE
    # =========================================================================

    def set_mode(self, mode: str):
        mode_map = {"idle": OperationMode.IDLE, "assist": OperationMode.ASSIST,
                    "operate": OperationMode.OPERATE}
        if mode in mode_map:
            self.governor.set_mode(mode_map[mode])
            self._save_state()

    def start(self, cycle_interval: float = 60.0):
        self.is_running = True

        def loop():
            while self.is_running:
                try:
                    result = self.cycle()
                    logger.debug(f"[v5] Cycle {result['cycle']} | Mode: {result['mode']} | "
                          f"Facts: {self.memory.get_stats()['facts']}")
                except Exception as e:
                    logger.error(f"[v5] Error: {e}")
                time.sleep(cycle_interval)

        threading.Thread(target=loop, daemon=True).start()
        logger.debug(f"[Proto-AGI v5] Started with Truth Spine in {self.governor.mode.value} mode")

    def stop(self):
        self.is_running = False

    def get_status(self) -> dict:
        return {
            "version": "v5-truth-spine",
            "running": self.is_running,
            "mode": self.governor.mode.value,
            "cycle_count": self.cycle_count,
            "needs": self.needs.to_dict(),
            "memory": self.memory.get_stats(),
            "verifier": self.verifier.get_stats(),
            "governance": {
                "actions_remaining": self.governor.budget.max_actions_per_hour - self.governor.budget.actions_used,
                "messages_remaining": self.governor.budget.max_messages_per_hour - self.governor.budget.messages_used,
                "pending_approvals": len(self.governor.pending_approvals)
            },
            "pending_confirmations": len(self.executor.get_pending_confirmations())
        }


# ============================================================================
#                              MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              PROTO-AGI CORE v5 — TRUTH SPINE                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  The Non-Negotiable Verification Layer                        ║
    ║                                                               ║
    ║  CONTRACT: ACTION → ARTIFACT → VERIFICATION → MEMORY TIER     ║
    ║                                                               ║
    ║  Memory Tiers:                                                ║
    ║    FACT = verified with artifact (hash, return code, etc.)    ║
    ║    BELIEF = inferred but not proven                           ║
    ║    SPECULATION = unverified (including LLM output)            ║
    ║                                                               ║
    ║  "If you can't verify it with an artifact, it's SPECULATION"  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    agi = ProtoAGI()

    print("\n=== Testing Truth Spine Integration ===")

    # Test 1: Create an action request
    print("\n[TEST 1] Action with verification")
    request = ActionRequest(
        action_type=ActionType.ANALYZE,
        intent="Calculate 2 + 2",
        params={"expression": "2 + 2"},
        expected_checks=["no_error", "not_empty"]
    )

    agi.set_mode("operate")
    result = agi.process(request, is_user_initiated=True)

    print(f"  Request: {request.intent}")
    print(f"  Verified: {result.is_verified}")
    print(f"  Memory Tier: {result.memory_tier.value}")
    print(f"  Trace ID: {result.memory_trace_id}")

    # Test 2: File operation (sandbox enforced)
    print("\n[TEST 2] Write file to sandbox")
    write_request = ActionRequest(
        action_type=ActionType.WRITE_FILE,
        intent="Create test file",
        params={"path": "test.txt", "content": "Hello from Truth Spine!"},
        expected_checks=["file_exists", "not_empty"]
    )

    result = agi.process(write_request, is_user_initiated=True)
    print(f"  Verified: {result.is_verified}")
    print(f"  Memory Tier: {result.memory_tier.value}")

    # Test 3: Process user input (LLM output = speculation)
    print("\n[TEST 3] Process user input")
    response = agi.process_input("What is the meaning of life?", chat_id="test")
    print(f"  Response: {response[:100]}...")

    # Status
    print("\n=== Status ===")
    status = agi.get_status()
    print(json.dumps(status, indent=2))
