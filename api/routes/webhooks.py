"""Webhook endpoints — external services push events to AURA.

GitHub CI failures, generic alerts, and simple notification forwarding.
All endpoints return 200 immediately and process async in background.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header, Request
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

# Prevent background tasks from being garbage collected before completion.
# _fire_and_forget() returns a Task that must be held by a strong reference.
_background_tasks: set = set()


def _fire_and_forget(coro):
    """Schedule a coroutine as a background task with GC protection."""
    task = _fire_and_forget(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

router = APIRouter(
    prefix="/api/webhooks",
    tags=["webhooks"],
    dependencies=[Depends(require_api_key)],
)

# In-memory registry of webhook subscriptions (survives until restart)
_webhook_registry: List[Dict[str, Any]] = []

# Log of recent webhook events (last 100)
_webhook_log: List[Dict[str, Any]] = []
_LOG_MAX = 100


def _log_event(source: str, event_type: str, summary: str, severity: str = "info"):
    """Append to the in-memory webhook event log."""
    entry = {
        "id": uuid.uuid4().hex[:8],
        "source": source,
        "event_type": event_type,
        "summary": summary,
        "severity": severity,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    _webhook_log.append(entry)
    if len(_webhook_log) > _LOG_MAX:
        _webhook_log.pop(0)
    return entry


# ============================================================================
# Helpers
# ============================================================================

def _verify_github_signature(payload_body: bytes, signature_header: str) -> bool:
    """Verify GitHub webhook signature using GITHUB_WEBHOOK_SECRET."""
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        logger.warning("[Webhooks] GITHUB_WEBHOOK_SECRET not set — rejecting unsigned request")
        return False  # Reject when no secret configured (set env var to enable)
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


async def _notify_surfaces(text: str, channel: str = "all"):
    """Send a notification to bound surfaces via ConversationManager + Telegram."""
    # Try Telegram direct push
    if channel in ("telegram", "all"):
        try:
            from aura.messaging.telegram_bot import TelegramBot
            # Get the singleton bot instance if running
            bot = TelegramBot._instance if hasattr(TelegramBot, "_instance") else None
            if bot and bot.is_running and bot.active_chats:
                from aura.messaging.base_platform import OutgoingMessage
                for chat_id in bot.active_chats:
                    msg = OutgoingMessage(chat_id=chat_id, text=text)
                    await bot.send_message(msg)
                    logger.info(f"[Webhooks] Notified Telegram chat {chat_id}")
        except Exception as e:
            logger.warning(f"[Webhooks] Telegram notify failed: {e}")

    # Also broadcast via proactive WebSocket channel
    if channel in ("web", "all"):
        try:
            from api.routes.chat import broadcast_proactive_message
            from aura.proactive import ProactiveMessage, ProactiveAction, EventPriority
            msg = ProactiveMessage(
                action=ProactiveAction.NOTIFY,
                content=text,
                priority=EventPriority.HIGH,
                metadata={"source": "webhook"},
            )
            await broadcast_proactive_message(msg)
        except Exception as e:
            logger.debug(f"[Webhooks] WebSocket broadcast failed: {e}")


async def _route_to_agent(task_description: str, context: Dict[str, Any] = None):
    """Queue a task for the AURA agent to investigate."""
    try:
        from api.services.agent_service import agent_service
        if agent_service.is_ready:
            # Use the agent's process method in a background thread
            loop = asyncio.get_running_loop()
            prompt = f"[Webhook Task] {task_description}"
            if context:
                prompt += f"\n\nContext: {json.dumps(context, indent=2, default=str)}"
            result = await loop.run_in_executor(
                None, agent_service.process_message, prompt
            )
            # Notify surfaces with the finding
            if result:
                summary = result if isinstance(result, str) else str(result)
                # Truncate if very long
                if len(summary) > 1500:
                    summary = summary[:1500] + "..."
                await _notify_surfaces(f"Investigation complete:\n{summary}")
            return result
    except Exception as e:
        logger.error(f"[Webhooks] Agent routing failed: {e}")
        await _notify_surfaces(f"Webhook task failed: {task_description}\nError: {e}")
    return None


# ============================================================================
# Request/Response Models
# ============================================================================

class AlertRequest(BaseModel):
    """Generic alert webhook payload."""
    type: str = Field(..., description="Alert type: cpu, disk, error, custom")
    message: str = Field(..., description="Alert message")
    severity: str = Field(default="medium", description="low, medium, high, critical")
    source: str = Field(default="unknown", description="Source system")
    metadata: Optional[Dict[str, Any]] = None


class NotifyRequest(BaseModel):
    """Simple notification push payload."""
    text: str = Field(..., description="Notification text")
    channel: str = Field(default="all", description="telegram, web, or all")


class WebhookSubscription(BaseModel):
    """A registered webhook subscription."""
    id: str
    source: str
    event_types: List[str]
    url: str
    created_at: str
    active: bool = True


# ============================================================================
# POST /api/webhooks/github — GitHub webhook handler
# ============================================================================

@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(default="", alias="X-Hub-Signature-256"),
    x_github_event: str = Header(default="", alias="X-GitHub-Event"),
):
    """Handle GitHub webhook payloads (push, PR, CI failure).

    Verifies signature via X-Hub-Signature-256 header.
    Returns 200 immediately, processes in background.
    """
    body = await request.body()

    # Verify signature
    if not _verify_github_signature(body, x_hub_signature_256):
        logger.warning("[Webhooks] GitHub signature verification failed")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = x_github_event or payload.get("action", "unknown")
    repo = payload.get("repository", {}).get("full_name", "unknown")

    logger.info(f"[Webhooks] GitHub event: {event_type} from {repo}")
    _log_event("github", event_type, f"{event_type} on {repo}")

    # Auto-register this repo as a subscription
    _ensure_subscription("github", repo, [event_type])

    # --- CI Failure (check_run or workflow_run with failure) ---
    if event_type == "check_run":
        check = payload.get("check_run", {})
        conclusion = check.get("conclusion", "")
        if conclusion == "failure":
            name = check.get("name", "unknown check")
            url = check.get("html_url", "")
            head_sha = check.get("head_sha", "")[:7]

            async def _investigate_ci():
                await _notify_surfaces(
                    f"CI Failure on {repo}\n"
                    f"Check: {name}\n"
                    f"Commit: {head_sha}\n"
                    f"URL: {url}\n\n"
                    f"Investigating..."
                )
                await _route_to_agent(
                    f"CI check '{name}' failed on {repo} (commit {head_sha}). "
                    f"Investigate the failure and suggest a fix.",
                    {"repo": repo, "check_name": name, "url": url, "sha": head_sha},
                )

            _fire_and_forget(_investigate_ci())

    elif event_type == "workflow_run":
        run = payload.get("workflow_run", {})
        conclusion = run.get("conclusion", "")
        if conclusion == "failure":
            name = run.get("name", "unknown workflow")
            url = run.get("html_url", "")
            branch = run.get("head_branch", "unknown")

            async def _investigate_workflow():
                await _notify_surfaces(
                    f"Workflow Failed on {repo}\n"
                    f"Workflow: {name}\n"
                    f"Branch: {branch}\n"
                    f"URL: {url}\n\n"
                    f"Investigating..."
                )
                await _route_to_agent(
                    f"GitHub Actions workflow '{name}' failed on {repo} (branch {branch}). "
                    f"Investigate the failure.",
                    {"repo": repo, "workflow": name, "url": url, "branch": branch},
                )

            _fire_and_forget(_investigate_workflow())

    # --- Pull Request ---
    elif event_type == "pull_request":
        action = payload.get("action", "")
        pr = payload.get("pull_request", {})
        if action in ("opened", "synchronize", "reopened"):
            title = pr.get("title", "")
            number = pr.get("number", 0)
            url = pr.get("html_url", "")
            user = pr.get("user", {}).get("login", "unknown")
            additions = pr.get("additions", 0)
            deletions = pr.get("deletions", 0)
            body_text = (pr.get("body") or "")[:500]

            async def _summarize_pr():
                summary = (
                    f"New PR on {repo}\n"
                    f"#{number}: {title}\n"
                    f"By: {user} | +{additions} -{deletions}\n"
                    f"URL: {url}"
                )
                if body_text:
                    summary += f"\n\nDescription:\n{body_text}"
                await _notify_surfaces(summary)

            _fire_and_forget(_summarize_pr())

    # --- Push ---
    elif event_type == "push":
        commits = payload.get("commits", [])
        ref = payload.get("ref", "")
        branch = ref.split("/")[-1] if "/" in ref else ref
        pusher = payload.get("pusher", {}).get("name", "unknown")
        if commits:
            commit_msgs = [c.get("message", "").split("\n")[0] for c in commits[:5]]
            summary = (
                f"Push to {repo}/{branch} by {pusher}\n"
                f"{len(commits)} commit(s):\n" +
                "\n".join(f"  - {m}" for m in commit_msgs)
            )
            _fire_and_forget(_notify_surfaces(summary))

    return {"status": "accepted", "event": event_type, "repo": repo}


# ============================================================================
# POST /api/webhooks/alert — Generic alert webhook
# ============================================================================

@router.post("/alert")
async def alert_webhook(alert: AlertRequest):
    """Handle generic alert webhooks (CPU, disk, error, custom).

    Routes to the agent for investigation and notifies bound surfaces.
    """
    logger.info(f"[Webhooks] Alert: type={alert.type} severity={alert.severity} source={alert.source}")
    _log_event("alert", alert.type, alert.message, alert.severity)

    # Build notification
    severity_emoji = {
        "low": "INFO",
        "medium": "WARN",
        "high": "ALERT",
        "critical": "CRITICAL",
    }
    label = severity_emoji.get(alert.severity, "ALERT")
    notification = (
        f"[{label}] {alert.type.upper()} alert from {alert.source}\n"
        f"{alert.message}"
    )

    # For high/critical: investigate via agent
    if alert.severity in ("high", "critical"):
        async def _handle_alert():
            await _notify_surfaces(notification + "\n\nInvestigating...")
            await _route_to_agent(
                f"{alert.type} alert from {alert.source}: {alert.message}",
                {"severity": alert.severity, "metadata": alert.metadata},
            )

        _fire_and_forget(_handle_alert())
    else:
        # Low/medium: just notify
        _fire_and_forget(_notify_surfaces(notification))

    return {
        "status": "accepted",
        "type": alert.type,
        "severity": alert.severity,
    }


# ============================================================================
# POST /api/webhooks/notify — Simple notification push
# ============================================================================

@router.post("/notify")
async def notify_webhook(req: NotifyRequest):
    """Forward a simple text notification to the specified channel(s)."""
    logger.info(f"[Webhooks] Notify: channel={req.channel} len={len(req.text)}")
    _log_event("notify", "message", req.text[:100], "info")

    _fire_and_forget(_notify_surfaces(req.text, channel=req.channel))

    return {"status": "sent", "channel": req.channel}


# ============================================================================
# GET /api/webhooks/list — List registered webhook subscriptions
# ============================================================================

def _ensure_subscription(source: str, name: str, event_types: List[str]):
    """Auto-register a subscription when a webhook fires (idempotent)."""
    for sub in _webhook_registry:
        if sub["source"] == source and sub["name"] == name:
            # Update event types
            for et in event_types:
                if et not in sub["event_types"]:
                    sub["event_types"].append(et)
            sub["last_seen"] = datetime.utcnow().isoformat() + "Z"
            return
    _webhook_registry.append({
        "id": uuid.uuid4().hex[:8],
        "source": source,
        "name": name,
        "event_types": event_types,
        "url": f"/api/webhooks/{source}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_seen": datetime.utcnow().isoformat() + "Z",
        "active": True,
    })


@router.get("/list")
async def list_webhooks():
    """List registered webhook subscriptions and recent events."""
    base_url = os.environ.get("AURA_BASE_URL", "http://localhost:8000")
    return {
        "subscriptions": _webhook_registry,
        "recent_events": _webhook_log[-20:],
        "endpoints": {
            "github": f"{base_url}/api/webhooks/github",
            "alert": f"{base_url}/api/webhooks/alert",
            "notify": f"{base_url}/api/webhooks/notify",
        },
    }
