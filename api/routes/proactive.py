"""Proactive system API endpoints - Gateway Daemon control."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/proactive", tags=["proactive"])

# Global daemon instance (lazy loaded)
_daemon = None
_daemon_task = None


async def _get_daemon():
    """Get the global Gateway Daemon singleton (shared with main.py auto-start)."""
    global _daemon
    if _daemon is None:
        try:
            from apprentice_agent.proactive.gateway_daemon import get_gateway_daemon
            _daemon = get_gateway_daemon()
            logger.info("[Proactive API] Using shared Gateway Daemon singleton")
        except ImportError as e:
            logger.error(f"[Proactive API] Failed to import GatewayDaemon: {e}")
            raise HTTPException(status_code=503, detail="Proactive system not available")
    return _daemon


# ============================================================================
# Request/Response Models
# ============================================================================

class DaemonStatusResponse(BaseModel):
    """Gateway Daemon status response."""
    running: bool
    state: str
    stats: Dict[str, Any]
    beliefs: Optional[Dict[str, float]] = None
    pending_messages: int = 0


class ContextUpdate(BaseModel):
    """Context update request."""
    app: Optional[str] = None
    task: Optional[str] = None
    keywords: Optional[List[str]] = None
    do_not_disturb: Optional[bool] = None


class ProactiveMessageResponse(BaseModel):
    """Proactive message from the daemon."""
    action: str
    content: str
    priority: str
    timestamp: str
    metadata: Dict[str, Any] = {}


class BeliefUpdateRequest(BaseModel):
    """Manual belief update request (for testing)."""
    user_activity: Optional[float] = None
    interaction_recency: Optional[float] = None
    urgent_events: Optional[float] = None
    context_changes: Optional[float] = None
    observation_confidence: Optional[float] = None


# ============================================================================
# Daemon Control Endpoints
# ============================================================================

@router.get("/status", response_model=DaemonStatusResponse)
async def get_daemon_status():
    """Get Gateway Daemon status and statistics."""
    try:
        daemon = await _get_daemon()
        stats = daemon.get_stats()

        beliefs = None
        try:
            belief_state = daemon.inference_engine.get_beliefs()
            beliefs = {
                "user_busy": belief_state.user_busy,
                "user_receptive": belief_state.user_receptive,
                "task_urgent": belief_state.task_urgent,
                "context_stable": belief_state.context_stable,
                "uncertainty": belief_state.uncertainty,
            }
        except Exception:
            pass

        return DaemonStatusResponse(
            running=daemon.state.value == "running",
            state=daemon.state.value,
            stats=stats,
            beliefs=beliefs,
            pending_messages=len(daemon._pending_messages)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Proactive API] Status error: {e}")
        return DaemonStatusResponse(
            running=False,
            state="error",
            stats={"error": str(e)},
            pending_messages=0
        )


@router.post("/start")
async def start_daemon(background_tasks: BackgroundTasks):
    """Start the Gateway Daemon."""
    global _daemon_task

    daemon = await _get_daemon()

    if daemon.state.value == "running":
        return {"status": "already_running", "message": "Daemon is already running"}

    try:
        # Start daemon in background
        async def run_daemon():
            await daemon.start()

        _daemon_task = asyncio.create_task(run_daemon())

        return {
            "status": "started",
            "message": "Gateway Daemon started",
            "state": daemon.state.value
        }
    except Exception as e:
        logger.error(f"[Proactive API] Start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_daemon():
    """Stop the Gateway Daemon."""
    daemon = await _get_daemon()

    if daemon.state.value == "stopped":
        return {"status": "already_stopped", "message": "Daemon is already stopped"}

    try:
        await daemon.stop()
        return {
            "status": "stopped",
            "message": "Gateway Daemon stopped",
            "state": daemon.state.value
        }
    except Exception as e:
        logger.error(f"[Proactive API] Stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause")
async def pause_daemon():
    """Pause proactive actions (daemon still processes events)."""
    daemon = await _get_daemon()
    daemon.pause()
    return {"status": "paused", "state": daemon.state.value}


@router.post("/resume")
async def resume_daemon():
    """Resume proactive actions."""
    daemon = await _get_daemon()
    daemon.resume()
    return {"status": "resumed", "state": daemon.state.value}


# ============================================================================
# Context & Beliefs
# ============================================================================

@router.post("/context")
async def update_context(update: ContextUpdate):
    """Update user context for relevance filtering."""
    daemon = await _get_daemon()

    daemon.update_context(
        app=update.app,
        task=update.task,
        keywords=update.keywords,
        do_not_disturb=update.do_not_disturb
    )

    return {
        "status": "updated",
        "context": {
            "current_app": daemon.user_context.current_app,
            "current_task": daemon.user_context.current_task,
            "keywords": list(daemon.salience_filter.context_keywords),
            "do_not_disturb": daemon.user_context.do_not_disturb,
        }
    }


@router.get("/context")
async def get_context():
    """Get current user context."""
    daemon = await _get_daemon()

    return {
        "current_app": daemon.user_context.current_app,
        "current_task": daemon.user_context.current_task,
        "keywords": list(daemon.salience_filter.context_keywords),
        "do_not_disturb": daemon.user_context.do_not_disturb,
        "activity_level": daemon.user_context.activity_level,
        "idle_since": daemon.user_context.idle_since.isoformat() if daemon.user_context.idle_since else None,
        "last_interaction": daemon.user_context.last_interaction.isoformat() if daemon.user_context.last_interaction else None,
    }


@router.post("/beliefs")
async def update_beliefs(update: BeliefUpdateRequest):
    """Manually update beliefs (for testing)."""
    daemon = await _get_daemon()

    observations = {}
    if update.user_activity is not None:
        observations["user_activity"] = update.user_activity
    if update.interaction_recency is not None:
        observations["interaction_recency"] = update.interaction_recency
    if update.urgent_events is not None:
        observations["urgent_events"] = update.urgent_events
    if update.context_changes is not None:
        observations["context_changes"] = update.context_changes
    if update.observation_confidence is not None:
        observations["observation_confidence"] = update.observation_confidence

    if observations:
        daemon.inference_engine.update_beliefs(observations)

    beliefs = daemon.inference_engine.get_beliefs()
    return {
        "status": "updated",
        "beliefs": {
            "user_busy": beliefs.user_busy,
            "user_receptive": beliefs.user_receptive,
            "task_urgent": beliefs.task_urgent,
            "context_stable": beliefs.context_stable,
            "uncertainty": beliefs.uncertainty,
        }
    }


# ============================================================================
# Proactive Messages
# ============================================================================

@router.get("/messages")
async def get_pending_messages():
    """Get and clear pending proactive messages."""
    daemon = await _get_daemon()
    messages = daemon.get_pending_messages()

    return {
        "count": len(messages),
        "messages": [
            {
                "action": msg.action.value,
                "content": msg.content,
                "priority": msg.priority.name if hasattr(msg.priority, 'name') else str(msg.priority),
                "timestamp": msg.timestamp.isoformat(),
                "delivered": msg.delivered,
                "metadata": msg.metadata,
            }
            for msg in messages
        ]
    }


@router.post("/decide")
async def trigger_decision():
    """Manually trigger a proactive decision (for testing)."""
    daemon = await _get_daemon()

    decision = daemon.inference_engine.select_action()

    return {
        "action": decision.action.value,
        "confidence": decision.confidence,
        "expected_free_energy": decision.expected_free_energy,
        "reasoning": decision.reasoning,
        "metadata": decision.metadata,
    }


# ============================================================================
# Test Message (for demonstration)
# ============================================================================

class TestMessageRequest(BaseModel):
    """Request for creating a test proactive message."""
    action: str = "suggest"
    content: Optional[str] = None


@router.post("/test-message")
async def create_test_message(request: TestMessageRequest):
    """Create a test proactive message for demonstration."""
    daemon = await _get_daemon()

    try:
        from apprentice_agent.proactive import ProactiveMessage, ProactiveAction, EventPriority

        # Default messages based on action
        default_messages = {
            "notify": "I noticed you've been working for a while. Remember to take a break!",
            "suggest": "Based on what you're working on, you might find it helpful to check the documentation.",
            "remind": "Just a gentle reminder - you mentioned wanting to review the code changes earlier.",
            "ask": "I've been observing your workflow. Would you like me to help optimize anything?",
            "intervene": "I detected something that might need your attention.",
        }

        action_str = request.action.lower()
        content = request.content or default_messages.get(action_str, "Hello! I wanted to reach out proactively.")

        # Map string to ProactiveAction enum
        action_map = {
            "notify": ProactiveAction.NOTIFY,
            "suggest": ProactiveAction.SUGGEST,
            "remind": ProactiveAction.REMIND,
            "ask": ProactiveAction.ASK,
            "intervene": ProactiveAction.INTERVENE,
        }

        action = action_map.get(action_str, ProactiveAction.SUGGEST)

        # Create and queue the message
        message = ProactiveMessage(
            action=action,
            content=content,
            priority=EventPriority.MEDIUM,
            metadata={"test": True, "confidence": 0.85}
        )

        daemon._pending_messages.append(message)

        return {
            "status": "queued",
            "action": action.value,
            "content": content,
            "pending_count": len(daemon._pending_messages)
        }

    except Exception as e:
        logger.error(f"[Proactive API] Test message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Event Publishing (for testing)
# ============================================================================

@router.post("/event")
async def publish_event(
    source: str,
    event_type: str,
    priority: int = 3,
    payload: Dict[str, Any] = None
):
    """Publish a test event to the daemon."""
    daemon = await _get_daemon()

    try:
        from apprentice_agent.proactive import Event, EventPriority

        event = Event(
            source=source,
            event_type=event_type,
            priority=EventPriority(priority),
            payload=payload or {}
        )

        success = await daemon.publish_event(event)

        return {
            "status": "published" if success else "failed",
            "event_id": event.event_id,
            "source": source,
            "event_type": event_type,
        }
    except Exception as e:
        logger.error(f"[Proactive API] Event publish error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Interaction Recording
# ============================================================================

@router.post("/interaction")
async def record_interaction():
    """Record that user interacted with AURA."""
    daemon = await _get_daemon()
    daemon.record_interaction()
    return {
        "status": "recorded",
        "last_interaction": daemon.user_context.last_interaction.isoformat() if daemon.user_context.last_interaction else None,
        "activity_level": daemon.user_context.activity_level,
    }


@router.post("/dismiss")
async def dismiss_proactive_message():
    """Record that user dismissed a proactive notification.

    Feeds back into active inference to discourage similar actions.
    """
    daemon = await _get_daemon()
    daemon.record_user_response(engaged=False, response_type="dismissed")
    return {
        "status": "recorded",
        "effect": "cooldown_increased",
    }


@router.post("/idle")
async def record_idle():
    """Record that user appears idle."""
    daemon = await _get_daemon()
    daemon.record_idle()
    return {
        "status": "recorded",
        "idle_since": daemon.user_context.idle_since.isoformat() if daemon.user_context.idle_since else None,
        "activity_level": daemon.user_context.activity_level,
    }


# ============================================================================
# Screen Awareness (Phase 3D)
# ============================================================================

@router.get("/screen-context")
async def get_screen_context():
    """Get current screen awareness context from Screenpipe.

    Returns what AURA can see: current app, window, recent text, errors.
    """
    try:
        from apprentice_agent.tools.screenpipe import get_screenpipe_client
        client = get_screenpipe_client()

        if not client.is_available():
            return {
                "available": False,
                "message": "Screenpipe not running. Install from https://screenpi.pe/"
            }

        ctx = client.get_screen_context(minutes=2, max_chars=1000)
        return {
            "available": ctx.get("available", False),
            "current_app": ctx.get("current_app"),
            "current_window": ctx.get("current_window"),
            "apps_used": ctx.get("apps_used", []),
            "has_errors": ctx.get("has_errors", False),
            "text_preview": ctx.get("recent_text", "")[:300],
            "result_count": ctx.get("result_count", 0),
        }
    except ImportError:
        return {"available": False, "message": "Screenpipe client not installed"}
    except Exception as e:
        logger.error(f"[Proactive API] Screen context error: {e}")
        return {"available": False, "error": str(e)}


@router.get("/workflow")
async def get_workflow_state():
    """Get current workflow boundary detection state (Phase 5B).

    Returns focus state, interruptibility, and recent app switches.
    """
    try:
        from apprentice_agent.proactive.monitors.workflow_detector import get_workflow_detector
        wd = get_workflow_detector()
        return wd.get_focus_state()
    except ImportError:
        return {"error": "Workflow detector not available"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/suggestion")
async def get_proactive_suggestion():
    """Get a proactive suggestion based on current context (Phase 5C).

    Returns what AURA would suggest right now based on screen, memory, patterns.
    """
    try:
        daemon = await _get_daemon()
        from apprentice_agent.proactive.active_inference import ProactiveAction

        suggestion = daemon._generate_message_content(ProactiveAction.SUGGEST)
        beliefs = daemon.inference_engine.get_beliefs()

        return {
            "suggestion": suggestion,
            "has_suggestion": suggestion is not None,
            "beliefs": {
                "task_urgent": beliefs.task_urgent,
                "uncertainty": beliefs.uncertainty,
                "user_engaged": beliefs.user_engaged,
            },
            "context": {
                "current_app": daemon.user_context.current_app,
                "current_task": daemon.user_context.current_task,
                "do_not_disturb": daemon.user_context.do_not_disturb,
            },
        }
    except Exception as e:
        return {"suggestion": None, "error": str(e)}
