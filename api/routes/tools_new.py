"""API endpoints for new AURA tools: Calendar, Spaced Repetition, Email, Screen Reader, Shell."""

import asyncio
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

# Lazy import to avoid blocking event loop at module load
def _get_agent_service():
    """Get agent_service with lazy loading."""
    from api.services.agent_service import agent_service
    return agent_service


router = APIRouter(prefix="/api", tags=["tools"], dependencies=[Depends(require_api_key)])


def _safe_error(e: Exception, context: str = "") -> str:
    """Log full error internally, return generic message to client."""
    logger.error(f"[API] {context}: {e}", exc_info=True)
    return "Internal error — check server logs"


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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _shell_run_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": _safe_error(e, "shell_run")}


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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _shell_sessions_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _shell_sessions_sync() -> dict:
    agent = _get_agent_service().agent
    if "shell_executor" in agent.tools:
        return agent.tools["shell_executor"].list_sessions()
    return {"success": False, "error": "Shell executor tool not loaded"}


# ============================================================================
# TASK MANAGER
# ============================================================================

class AddTaskRequest(BaseModel):
    title: str
    description: str = ""
    priority: str = "medium"
    project: Optional[str] = None
    due_date: Optional[str] = None
    tags: List[str] = []


class UpdateTaskRequest(BaseModel):
    task_id: str
    status: Optional[str] = None
    priority: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


@router.get("/tasks/list")
async def tasks_list(status: Optional[str] = None, project: Optional[str] = None):
    """List tasks, optionally filtered by status or project."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_list_sync(status, project))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tasks_list_sync(status: Optional[str], project: Optional[str]) -> dict:
    agent = _get_agent_service().agent
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].list_tasks(status=status, project=project)
    return {"success": False, "error": "Task manager tool not loaded"}


@router.get("/tasks/board")
async def tasks_board():
    """Get kanban board view."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _tasks_board_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tasks_board_sync() -> dict:
    agent = _get_agent_service().agent
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].board()
    return {"success": False, "error": "Task manager tool not loaded"}


@router.post("/tasks/add")
async def tasks_add(request: AddTaskRequest):
    """Add a new task."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_add_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tasks_add_sync(request: AddTaskRequest) -> dict:
    agent = _get_agent_service().agent
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].add_task(
            title=request.title,
            description=request.description,
            priority=request.priority,
            project=request.project,
            due_date=request.due_date,
            tags=request.tags,
        )
    return {"success": False, "error": "Task manager tool not loaded"}


@router.put("/tasks/update")
async def tasks_update(request: UpdateTaskRequest):
    """Update a task."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_update_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tasks_update_sync(request: UpdateTaskRequest) -> dict:
    agent = _get_agent_service().agent
    if "task_manager" in agent.tools:
        kwargs = {}
        if request.status:
            kwargs["status"] = request.status
        if request.priority:
            kwargs["priority"] = request.priority
        if request.title:
            kwargs["title"] = request.title
        if request.description:
            kwargs["description"] = request.description
        return agent.tools["task_manager"].update_task(request.task_id, **kwargs)
    return {"success": False, "error": "Task manager tool not loaded"}


@router.delete("/tasks/{task_id}")
async def tasks_remove(task_id: str):
    """Remove a task."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _tasks_remove_sync(task_id))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tasks_remove_sync(task_id: str) -> dict:
    agent = _get_agent_service().agent
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].remove_task(task_id)
    return {"success": False, "error": "Task manager tool not loaded"}


@router.get("/tasks/overdue")
async def tasks_overdue():
    """Get overdue tasks."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _tasks_overdue_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _tasks_overdue_sync() -> dict:
    agent = _get_agent_service().agent
    if "task_manager" in agent.tools:
        return agent.tools["task_manager"].overdue()
    return {"success": False, "error": "Task manager tool not loaded"}


# ============================================================================
# API TESTER
# ============================================================================

class APITestRequest(BaseModel):
    method: str = "GET"
    url: str
    headers: Optional[dict] = None
    body: Optional[str] = None
    timeout: int = 30


@router.post("/api-tester/run")
async def api_tester_run(request: APITestRequest):
    """Execute an HTTP request."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _api_tester_run_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _api_tester_run_sync(request: APITestRequest) -> dict:
    agent = _get_agent_service().agent
    if "api_tester" in agent.tools:
        return agent.tools["api_tester"].request(
            method=request.method,
            url=request.url,
            headers=request.headers,
            body=request.body,
            timeout=request.timeout,
        )
    return {"success": False, "error": "API tester tool not loaded"}


@router.get("/api-tester/history")
async def api_tester_history(limit: int = 20):
    """Get API request history."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _api_tester_history_sync(limit))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _api_tester_history_sync(limit: int) -> dict:
    agent = _get_agent_service().agent
    if "api_tester" in agent.tools:
        return agent.tools["api_tester"].history(limit=limit)
    return {"success": False, "error": "API tester tool not loaded"}


# ============================================================================
# DATABASE
# ============================================================================

class SQLQueryRequest(BaseModel):
    sql: str
    db: str = "default"


class CSVImportRequest(BaseModel):
    csv_path: str
    table: str
    db: str = "default"


@router.post("/database/query")
async def database_query(request: SQLQueryRequest):
    """Execute a SQL query."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _database_query_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _database_query_sync(request: SQLQueryRequest) -> dict:
    agent = _get_agent_service().agent
    if "database" in agent.tools:
        return agent.tools["database"].query(sql=request.sql, db=request.db)
    return {"success": False, "error": "Database tool not loaded"}


@router.get("/database/schema")
async def database_schema(db: str = "default", table: Optional[str] = None):
    """Get database schema."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _database_schema_sync(db, table))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _database_schema_sync(db: str, table: Optional[str]) -> dict:
    agent = _get_agent_service().agent
    if "database" in agent.tools:
        return agent.tools["database"].schema(db=db, table=table)
    return {"success": False, "error": "Database tool not loaded"}


@router.post("/database/import-csv")
async def database_import_csv(request: CSVImportRequest):
    """Import CSV into a database table."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _database_import_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _database_import_sync(request: CSVImportRequest) -> dict:
    agent = _get_agent_service().agent
    if "database" in agent.tools:
        return agent.tools["database"].import_csv(
            csv_path=request.csv_path, table=request.table, db=request.db
        )
    return {"success": False, "error": "Database tool not loaded"}


# ============================================================================
# AUDIO TRANSCRIBER
# ============================================================================

class TranscribeRequest(BaseModel):
    file_path: str
    language: Optional[str] = None
    model_size: Optional[str] = None


@router.post("/audio/transcribe")
async def audio_transcribe(request: TranscribeRequest):
    """Transcribe an audio/video file."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _audio_transcribe_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _audio_transcribe_sync(request: TranscribeRequest) -> dict:
    agent = _get_agent_service().agent
    if "audio_transcriber" in agent.tools:
        return agent.tools["audio_transcriber"].transcribe(
            file_path=request.file_path,
            language=request.language,
            model_size=request.model_size,
        )
    return {"success": False, "error": "Audio transcriber tool not loaded"}


@router.get("/audio/transcripts")
async def audio_transcripts():
    """List saved transcripts."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _audio_transcripts_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _audio_transcripts_sync() -> dict:
    agent = _get_agent_service().agent
    if "audio_transcriber" in agent.tools:
        return agent.tools["audio_transcriber"].list_transcripts()
    return {"success": False, "error": "Audio transcriber tool not loaded"}


@router.get("/audio/status")
async def audio_status():
    """Check Whisper availability."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _audio_status_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _audio_status_sync() -> dict:
    agent = _get_agent_service().agent
    if "audio_transcriber" in agent.tools:
        return agent.tools["audio_transcriber"].status()
    return {"success": False, "error": "Audio transcriber tool not loaded"}


# ============================================================================
# CLIPBOARD HISTORY
# ============================================================================

@router.get("/clipboard/history")
async def clipboard_history(limit: int = 20, category: Optional[str] = None):
    """Get clipboard history."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _clipboard_history_sync(limit, category))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _clipboard_history_sync(limit: int, category: Optional[str]) -> dict:
    agent = _get_agent_service().agent
    if "clipboard_history" in agent.tools:
        return agent.tools["clipboard_history"].list_history(limit=limit, category=category)
    return {"success": False, "error": "Clipboard history tool not loaded"}


@router.post("/clipboard/capture")
async def clipboard_capture():
    """Capture current clipboard content."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _clipboard_capture_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _clipboard_capture_sync() -> dict:
    agent = _get_agent_service().agent
    if "clipboard_history" in agent.tools:
        return agent.tools["clipboard_history"].capture()
    return {"success": False, "error": "Clipboard history tool not loaded"}


@router.get("/clipboard/search")
async def clipboard_search(query: str):
    """Search clipboard history."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _clipboard_search_sync(query))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _clipboard_search_sync(query: str) -> dict:
    agent = _get_agent_service().agent
    if "clipboard_history" in agent.tools:
        return agent.tools["clipboard_history"].search(query)
    return {"success": False, "error": "Clipboard history tool not loaded"}


@router.get("/clipboard/stats")
async def clipboard_stats():
    """Clipboard usage statistics."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _clipboard_stats_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _clipboard_stats_sync() -> dict:
    agent = _get_agent_service().agent
    if "clipboard_history" in agent.tools:
        return agent.tools["clipboard_history"].stats()
    return {"success": False, "error": "Clipboard history tool not loaded"}


# ============================================================================
# RESEARCH
# ============================================================================

class SaveResearchRequest(BaseModel):
    title: str
    content: str
    category: str = "tools"
    tags: Optional[List[str]] = None
    sources: Optional[List[str]] = None


@router.get("/research/list")
async def research_list(category: Optional[str] = None):
    """List research files."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _research_list_sync(category))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _research_list_sync(category: Optional[str]) -> dict:
    agent = _get_agent_service().agent
    if "research" in agent.tools:
        return agent.tools["research"].list_research(category=category)
    return {"success": False, "error": "Research tool not loaded"}


@router.post("/research/save")
async def research_save(request: SaveResearchRequest):
    """Save a research note."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _research_save_sync(request))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _research_save_sync(request: SaveResearchRequest) -> dict:
    agent = _get_agent_service().agent
    if "research" in agent.tools:
        return agent.tools["research"].save(
            title=request.title,
            content=request.content,
            category=request.category,
            tags=request.tags,
            sources=request.sources,
        )
    return {"success": False, "error": "Research tool not loaded"}


@router.get("/research/search")
async def research_search(query: str, category: Optional[str] = None):
    """Search research files."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: _research_search_sync(query, category))
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _research_search_sync(query: str, category: Optional[str]) -> dict:
    agent = _get_agent_service().agent
    if "research" in agent.tools:
        return agent.tools["research"].search(query, category=category)
    return {"success": False, "error": "Research tool not loaded"}


@router.get("/research/stats")
async def research_stats():
    """Research statistics."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _research_stats_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _research_stats_sync() -> dict:
    agent = _get_agent_service().agent
    if "research" in agent.tools:
        return agent.tools["research"].stats()
    return {"success": False, "error": "Research tool not loaded"}


@router.get("/research/skills")
async def research_skills():
    """List saved skills."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _research_skills_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _research_skills_sync() -> dict:
    agent = _get_agent_service().agent
    if "research" in agent.tools:
        return agent.tools["research"].list_skills()
    return {"success": False, "error": "Research tool not loaded"}
