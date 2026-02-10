"""API endpoints for new AURA tools: Calendar, Spaced Repetition, Email, Screen Reader, Shell."""

import asyncio
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Lazy import to avoid blocking event loop at module load
def _get_agent_service():
    """Get agent_service with lazy loading."""
    from api.services.agent_service import agent_service
    return agent_service


router = APIRouter(prefix="/api", tags=["tools"])


# ============================================================================
# CALENDAR
# ============================================================================

class AddEventRequest(BaseModel):
    title: str
    start: str
    end: Optional[str] = None
    description: str = ""
    location: str = ""
    recurrence: Optional[str] = None
    reminders: Optional[List[int]] = None


@router.get("/calendar/today")
async def calendar_today():
    """Get today's events."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _calendar_today_sync)
        return result
    except Exception as e:
        logger.error(f"[Calendar] Error: {e}")
        return {"success": False, "error": str(e)}


def _calendar_today_sync() -> dict:
    agent = _get_agent_service().agent
    if "calendar" in agent.tools:
        return agent.tools["calendar"].today()
    return {"success": False, "error": "Calendar tool not loaded"}


@router.get("/calendar/upcoming")
async def calendar_upcoming(days: int = 7):
    """Get upcoming events."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _calendar_upcoming_sync(days))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _calendar_upcoming_sync(days: int) -> dict:
    agent = _get_agent_service().agent
    if "calendar" in agent.tools:
        return agent.tools["calendar"].upcoming(days=days)
    return {"success": False, "error": "Calendar tool not loaded"}


@router.post("/calendar/add")
async def calendar_add(request: AddEventRequest):
    """Add a calendar event."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _calendar_add_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _calendar_add_sync(request: AddEventRequest) -> dict:
    agent = _get_agent_service().agent
    if "calendar" in agent.tools:
        return agent.tools["calendar"].add_event(
            title=request.title,
            start=request.start,
            end=request.end,
            description=request.description,
            location=request.location,
            recurrence=request.recurrence,
            reminders=request.reminders,
        )
    return {"success": False, "error": "Calendar tool not loaded"}


@router.delete("/calendar/{event_id}")
async def calendar_remove(event_id: str):
    """Remove a calendar event."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _calendar_remove_sync(event_id))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _calendar_remove_sync(event_id: str) -> dict:
    agent = _get_agent_service().agent
    if "calendar" in agent.tools:
        return agent.tools["calendar"].remove_event(event_id)
    return {"success": False, "error": "Calendar tool not loaded"}


# ============================================================================
# SPACED REPETITION / FLASHCARDS
# ============================================================================

class AddCardRequest(BaseModel):
    front: str
    back: str
    tags: List[str] = []
    deck: str = "default"


class AnswerRequest(BaseModel):
    card_id: str
    quality: int  # 0-5


@router.get("/flashcards/due")
async def flashcards_due():
    """Get due cards count and next card."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _flashcards_due_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _flashcards_due_sync() -> dict:
    agent = _get_agent_service().agent
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].review()
    return {"success": False, "error": "Spaced repetition tool not loaded"}


@router.post("/flashcards/answer")
async def flashcards_answer(request: AnswerRequest):
    """Submit answer quality for a flashcard."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _flashcards_answer_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _flashcards_answer_sync(request: AnswerRequest) -> dict:
    agent = _get_agent_service().agent
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].answer(request.card_id, request.quality)
    return {"success": False, "error": "Spaced repetition tool not loaded"}


@router.get("/flashcards/stats")
async def flashcards_stats():
    """Get deck statistics."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _flashcards_stats_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _flashcards_stats_sync() -> dict:
    agent = _get_agent_service().agent
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].list_decks()
    return {"success": False, "error": "Spaced repetition tool not loaded"}


@router.post("/flashcards/add")
async def flashcards_add(request: AddCardRequest):
    """Add a flashcard."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _flashcards_add_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _flashcards_add_sync(request: AddCardRequest) -> dict:
    agent = _get_agent_service().agent
    if "spaced_repetition" in agent.tools:
        return agent.tools["spaced_repetition"].add_card(
            front=request.front,
            back=request.back,
            tags=request.tags,
            deck=request.deck,
        )
    return {"success": False, "error": "Spaced repetition tool not loaded"}


# ============================================================================
# EMAIL
# ============================================================================

class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    cc: Optional[str] = None
    bcc: Optional[str] = None


@router.get("/email/status")
async def email_status():
    """Check email configuration status."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _email_status_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _email_status_sync() -> dict:
    agent = _get_agent_service().agent
    if "email" in agent.tools:
        return agent.tools["email"].get_config_status()
    return {"success": False, "error": "Email tool not loaded"}


@router.get("/email/inbox")
async def email_inbox(limit: int = 10):
    """Get recent emails."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _email_inbox_sync(limit))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _email_inbox_sync(limit: int) -> dict:
    agent = _get_agent_service().agent
    if "email" in agent.tools:
        return agent.tools["email"].fetch_emails(limit=limit)
    return {"success": False, "error": "Email tool not loaded"}


@router.post("/email/send")
async def email_send(request: SendEmailRequest):
    """Send an email."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _email_send_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _email_send_sync(request: SendEmailRequest) -> dict:
    agent = _get_agent_service().agent
    if "email" in agent.tools:
        return agent.tools["email"].send_email(
            to=request.to,
            subject=request.subject,
            body=request.body,
            cc=request.cc,
            bcc=request.bcc,
        )
    return {"success": False, "error": "Email tool not loaded"}


# ============================================================================
# SCREEN READER
# ============================================================================

@router.get("/screen/read")
async def screen_read():
    """Read current screen via OCR."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _screen_read_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _screen_read_sync() -> dict:
    agent = _get_agent_service().agent
    if "screen_reader" in agent.tools:
        return agent.tools["screen_reader"].read_screen()
    return {"success": False, "error": "Screen reader tool not loaded"}


@router.get("/screen/active-window")
async def screen_active_window():
    """Get active window info."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _screen_active_window_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _screen_active_window_sync() -> dict:
    agent = _get_agent_service().agent
    if "screen_reader" in agent.tools:
        return agent.tools["screen_reader"].get_active_window()
    return {"success": False, "error": "Screen reader tool not loaded"}


# ============================================================================
# SHELL EXECUTOR
# ============================================================================

class ShellRunRequest(BaseModel):
    command: str
    session_id: Optional[str] = None
    timeout: int = 60
    cwd: Optional[str] = None


@router.post("/shell/run")
async def shell_run(request: ShellRunRequest):
    """Execute a shell command."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _shell_run_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _shell_run_sync(request: ShellRunRequest) -> dict:
    agent = _get_agent_service().agent
    if "shell_executor" in agent.tools:
        return agent.tools["shell_executor"].run(
            command=request.command,
            session_id=request.session_id,
            timeout=request.timeout,
            cwd=request.cwd,
        )
    return {"success": False, "error": "Shell executor tool not loaded"}


@router.get("/shell/sessions")
async def shell_sessions():
    """List active shell sessions."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _shell_sessions_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _shell_sessions_sync() -> dict:
    agent = _get_agent_service().agent
    if "shell_executor" in agent.tools:
        return agent.tools["shell_executor"].list_sessions()
    return {"success": False, "error": "Shell executor tool not loaded"}
