"""Task Scheduler Tool — schedule AURA tools and prompts to run on a cron/interval.

Uses APScheduler with SQLite persistence so jobs survive restarts.
AURA can autonomously execute any tool at scheduled times without being prompted.

Examples:
    - Daily memory consolidation at 2am
    - Weekly GitHub summary every Friday at 9am
    - Hourly mood check-in
    - Every 30 minutes: check clipboard for important content

Job types:
    - cron: Standard cron expression (e.g. "0 9 * * 1" = every Monday 9am)
    - interval: Repeat every N seconds/minutes/hours
    - date: Run once at a specific datetime
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

logger = logging.getLogger(__name__)

SCHEDULER_DB = Path(__file__).parent.parent.parent / "data" / "scheduler.db"
JOBS_LOG = Path(__file__).parent.parent.parent / "data" / "scheduler_log.json"

# Global singleton scheduler (shared across tool instances)
_scheduler: Optional[Any] = None
_scheduler_lock = threading.Lock()

# Registry of callable tool executors (populated at startup)
_tool_registry: Dict[str, Callable] = {}


def register_tool_executor(tool_name: str, executor: Callable):
    """Register a tool's execute function so the scheduler can call it."""
    _tool_registry[tool_name] = executor


def _execute_job(tool_name: str, action: str, kwargs_json: str):
    """Called by APScheduler when a job fires."""
    try:
        kwargs = json.loads(kwargs_json) if kwargs_json else {}
        if tool_name in _tool_registry:
            result = _tool_registry[tool_name](action, **kwargs)
            logger.info(f"[Scheduler] Job executed: {tool_name}.{action} → {result.get('success', '?')}")
        else:
            logger.warning(f"[Scheduler] Tool '{tool_name}' not registered — job skipped")
        _log_execution(tool_name, action, True)
    except Exception as e:
        logger.error(f"[Scheduler] Job failed: {tool_name}.{action}: {e}")
        _log_execution(tool_name, action, False, str(e))


def _log_execution(tool_name: str, action: str, success: bool, error: Optional[str] = None):
    """Append an execution record to the log file."""
    try:
        JOBS_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = []
        if JOBS_LOG.exists():
            try:
                with open(JOBS_LOG, "r") as f:
                    log = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning("[Scheduler] Corrupt log file, resetting")
                log = []
        log.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "action": action,
            "success": success,
            "error": error,
        })
        log = log[-500:]  # keep last 500 entries
        with open(JOBS_LOG, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning(f"[Scheduler] Log write failed: {e}")


def _get_scheduler() -> Any:
    """Get or create the global BackgroundScheduler."""
    global _scheduler
    if not APSCHEDULER_AVAILABLE:
        return None
    with _scheduler_lock:
        if _scheduler is None:
            SCHEDULER_DB.parent.mkdir(parents=True, exist_ok=True)
            jobstores = {
                "default": SQLAlchemyJobStore(url=f"sqlite:///{SCHEDULER_DB}")
            }
            _scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")

            def _on_job_error(event):
                logger.error(f"[Scheduler] Job error: {event.job_id}: {event.exception}")

            _scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)
            _scheduler.start()
            logger.info(f"[Scheduler] APScheduler started (DB: {SCHEDULER_DB})")
        return _scheduler


class TaskSchedulerTool:
    """Schedule AURA tools to run automatically at cron intervals or set times."""

    name = "task_scheduler"
    description = "Schedule AURA tools to run automatically — cron jobs, intervals, one-time tasks. 'run GitHub summary every Monday'"

    def __init__(self):
        self._scheduler = None
        if APSCHEDULER_AVAILABLE:
            try:
                self._scheduler = _get_scheduler()
            except Exception as e:
                logger.warning(f"[Scheduler] Init failed: {e}")

    def _not_available(self) -> Dict:
        return {"success": False, "error": "apscheduler not installed. Run: pip install apscheduler"}

    # ------------------------------------------------------------------ #
    # Job Management
    # ------------------------------------------------------------------ #

    def add_cron_job(
        self,
        tool_name: str,
        action: str,
        cron_expression: str,
        job_id: Optional[str] = None,
        label: Optional[str] = None,
        kwargs: Optional[Dict] = None,
    ) -> Dict:
        """Schedule a tool to run on a cron schedule.

        Args:
            tool_name: Name of the AURA tool to run (e.g. 'github', 'clipboard')
            action: Action to call on the tool (e.g. 'weekly_summary', 'list_recent')
            cron_expression: Cron expression e.g. '0 9 * * 1' (Mon 9am), '0 2 * * *' (daily 2am)
            job_id: Optional unique ID (auto-generated if not provided)
            label: Human-readable description
            kwargs: Additional kwargs to pass to tool.execute()
        """
        if not self._scheduler:
            return self._not_available()
        try:
            parts = cron_expression.strip().split()
            if len(parts) != 5:
                return {"success": False, "error": "Cron expression must have 5 parts: minute hour day month weekday"}
            minute, hour, day, month, day_of_week = parts
            trigger = CronTrigger(
                minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week
            )
            jid = job_id or f"{tool_name}_{action}_{uuid.uuid4().hex[:6]}"
            kwargs_json = json.dumps(kwargs or {})
            self._scheduler.add_job(
                _execute_job,
                trigger=trigger,
                id=jid,
                name=label or f"{tool_name}.{action}",
                args=[tool_name, action, kwargs_json],
                replace_existing=True,
                misfire_grace_time=3600,
            )
            next_run = self._scheduler.get_job(jid)
            return {
                "success": True,
                "job_id": jid,
                "type": "cron",
                "cron": cron_expression,
                "tool": tool_name,
                "action": action,
                "label": label or f"{tool_name}.{action}",
                "next_run": str(next_run.next_run_time) if next_run and next_run.next_run_time else "unknown",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_interval_job(
        self,
        tool_name: str,
        action: str,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        job_id: Optional[str] = None,
        label: Optional[str] = None,
        kwargs: Optional[Dict] = None,
    ) -> Dict:
        """Schedule a tool to run every N hours/minutes/seconds.

        Args:
            tool_name: AURA tool name
            action: Tool action
            hours: Repeat every N hours
            minutes: Repeat every N minutes
            seconds: Repeat every N seconds
            job_id: Optional unique ID
            label: Human-readable description
            kwargs: Additional kwargs for tool
        """
        if not self._scheduler:
            return self._not_available()
        if not any([hours, minutes, seconds]):
            return {"success": False, "error": "Specify at least one of: hours, minutes, seconds"}
        try:
            trigger = IntervalTrigger(hours=hours, minutes=minutes, seconds=seconds)
            jid = job_id or f"{tool_name}_{action}_{uuid.uuid4().hex[:6]}"
            kwargs_json = json.dumps(kwargs or {})
            self._scheduler.add_job(
                _execute_job,
                trigger=trigger,
                id=jid,
                name=label or f"{tool_name}.{action}",
                args=[tool_name, action, kwargs_json],
                replace_existing=True,
                misfire_grace_time=3600,
            )
            interval_str = f"{hours}h {minutes}m {seconds}s".replace("0h ", "").replace("0m ", "").replace("0s", "").strip()
            next_run = self._scheduler.get_job(jid)
            return {
                "success": True,
                "job_id": jid,
                "type": "interval",
                "interval": interval_str,
                "tool": tool_name,
                "action": action,
                "label": label or f"{tool_name}.{action}",
                "next_run": str(next_run.next_run_time) if next_run and next_run.next_run_time else "unknown",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_once_job(
        self,
        tool_name: str,
        action: str,
        run_at: str,
        job_id: Optional[str] = None,
        label: Optional[str] = None,
        kwargs: Optional[Dict] = None,
    ) -> Dict:
        """Schedule a tool to run once at a specific time.

        Args:
            tool_name: AURA tool name
            action: Tool action
            run_at: ISO datetime string e.g. '2026-03-01T15:30:00'
            job_id: Optional unique ID
            label: Human-readable description
            kwargs: Additional kwargs
        """
        if not self._scheduler:
            return self._not_available()
        try:
            run_time = datetime.fromisoformat(run_at)
            trigger = DateTrigger(run_date=run_time)
            jid = job_id or f"{tool_name}_{action}_{uuid.uuid4().hex[:6]}"
            kwargs_json = json.dumps(kwargs or {})
            self._scheduler.add_job(
                _execute_job,
                trigger=trigger,
                id=jid,
                name=label or f"{tool_name}.{action}",
                args=[tool_name, action, kwargs_json],
                replace_existing=True,
            )
            return {
                "success": True,
                "job_id": jid,
                "type": "once",
                "run_at": run_at,
                "tool": tool_name,
                "action": action,
                "label": label or f"{tool_name}.{action}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_jobs(self) -> Dict:
        """List all scheduled jobs."""
        if not self._scheduler:
            return self._not_available()
        try:
            jobs = self._scheduler.get_jobs()
            return {
                "success": True,
                "count": len(jobs),
                "jobs": [
                    {
                        "id": job.id,
                        "name": job.name,
                        "trigger": str(job.trigger),
                        "next_run": str(job.next_run_time) if job.next_run_time else "paused",
                    }
                    for job in jobs
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pause_job(self, job_id: str) -> Dict:
        """Pause a scheduled job."""
        if not self._scheduler:
            return self._not_available()
        try:
            self._scheduler.pause_job(job_id)
            return {"success": True, "paused": job_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def resume_job(self, job_id: str) -> Dict:
        """Resume a paused job."""
        if not self._scheduler:
            return self._not_available()
        try:
            self._scheduler.resume_job(job_id)
            job = self._scheduler.get_job(job_id)
            return {
                "success": True,
                "resumed": job_id,
                "next_run": str(job.next_run_time) if job and job.next_run_time else "unknown",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_job(self, job_id: str) -> Dict:
        """Remove / delete a scheduled job."""
        if not self._scheduler:
            return self._not_available()
        try:
            self._scheduler.remove_job(job_id)
            return {"success": True, "removed": job_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_now(self, job_id: str) -> Dict:
        """Immediately trigger a scheduled job (without removing it)."""
        if not self._scheduler:
            return self._not_available()
        try:
            job = self._scheduler.get_job(job_id)
            if not job:
                return {"success": False, "error": f"Job '{job_id}' not found"}
            # Get args from job and call directly
            args = job.args if job.args else []
            if len(args) >= 3:
                tool_name, action, kwargs_json = args[0], args[1], args[2]
                _execute_job(tool_name, action, kwargs_json)
                return {"success": True, "ran": job_id, "tool": tool_name, "action": action}
            return {"success": False, "error": "Cannot determine job parameters"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_execution_log(self, limit: int = 20) -> Dict:
        """Get recent job execution history."""
        try:
            if not JOBS_LOG.exists():
                return {"success": True, "log": [], "count": 0}
            with open(JOBS_LOG, "r") as f:
                log = json.load(f)
            recent = list(reversed(log))[:limit]
            return {"success": True, "log": recent, "count": len(log)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Presets
    # ------------------------------------------------------------------ #

    def setup_defaults(self) -> Dict:
        """Set up sensible default scheduled jobs for AURA."""
        results = []
        # Daily GitHub summary at 9am UTC
        r = self.add_cron_job("github", "weekly_summary", "0 9 * * 1",
                               job_id="weekly_github", label="Weekly GitHub summary (Mon 9am)")
        results.append(r)
        # Hourly clipboard auto-save
        r = self.add_interval_job("clipboard", "list_recent", hours=1,
                                   job_id="hourly_clipboard", label="Hourly clipboard check")
        results.append(r)
        return {
            "success": True,
            "defaults_installed": len([r for r in results if r.get("success")]),
            "results": results,
        }

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a scheduler action."""
        a = action.lower().strip()
        job_id = kwargs.get("job_id") or kwargs.get("id") or ""

        if "list" in a or "show" in a:
            return self.list_jobs()
        if "log" in a or "history" in a:
            return self.get_execution_log(kwargs.get("limit", 20))
        if "pause" in a:
            return self.pause_job(job_id)
        if "resume" in a:
            return self.resume_job(job_id)
        if "remove" in a or "delete" in a or "cancel" in a:
            return self.remove_job(job_id)
        if "run_now" in a or "trigger" in a or "now" in a:
            return self.run_now(job_id)
        if "default" in a or "setup" in a:
            return self.setup_defaults()
        if "cron" in a or "schedule" in a:
            tool = kwargs.get("tool") or kwargs.get("tool_name") or ""
            act = kwargs.get("tool_action") or kwargs.get("action") or ""
            cron = kwargs.get("cron") or kwargs.get("cron_expression") or ""
            if cron:
                return self.add_cron_job(tool, act, cron, job_id or None, kwargs.get("label"), kwargs.get("kwargs"))
        if "interval" in a or "every" in a:
            tool = kwargs.get("tool") or ""
            act = kwargs.get("tool_action") or ""
            return self.add_interval_job(tool, act, kwargs.get("hours", 0), kwargs.get("minutes", 0), kwargs.get("seconds", 0), job_id or None, kwargs.get("label"), kwargs.get("kwargs"))
        if "once" in a or "at" in a:
            tool = kwargs.get("tool") or ""
            act = kwargs.get("tool_action") or ""
            return self.add_once_job(tool, act, kwargs.get("run_at") or "", job_id or None, kwargs.get("label"), kwargs.get("kwargs"))
        return self.list_jobs()
