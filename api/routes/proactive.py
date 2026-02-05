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
    """Get or create the Gateway Daemon instance."""
    global _daemon
    if _daemon is None:
        try:
            from apprentice_agent.proactive import GatewayDaemon
            _daemon = GatewayDaemon(use_redis=False)
            logger.info("[Proactive API] Gateway Daemon created")
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
