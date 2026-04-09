"""
SchedulingMixin — /remind, /schedule, /tasks, /cancel and _fire_reminder
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta

try:
    from telegram import Update
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class SchedulingMixin:
    """Reminder and scheduled-task command handlers."""

    async def _handle_remind(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remind <time> <message> — one-shot reminder."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args
        if not args or len(args) < 2:
            await update.message.reply_text("Usage: /remind <time> <message>\n\nExamples:\n/remind in 2h Check deployment\n/remind at 17:00 Call team\n/remind tomorrow 9am Review PRs")
            return

        text = " ".join(args)

        now = datetime.now()
        run_at = None
        message = text

        # "in Xh", "in Xm", "in X hours", "in X minutes"
        m = re.match(r"in\s+(\d+)\s*h(?:ours?)?\s*(.*)", text, re.I)
        if m:
            run_at = now + timedelta(hours=int(m.group(1)))
            message = m.group(2).strip()
        if not run_at:
            m = re.match(r"in\s+(\d+)\s*m(?:in(?:utes?)?)?\s*(.*)", text, re.I)
            if m:
                run_at = now + timedelta(minutes=int(m.group(1)))
                message = m.group(2).strip()
        if not run_at:
            m = re.match(r"at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?\s*(.*)", text, re.I)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2) or 0)
                ampm = (m.group(3) or "").lower()
                if ampm == "pm" and hour < 12: hour += 12
                if ampm == "am" and hour == 12: hour = 0
                run_at = now.replace(hour=hour, minute=minute, second=0)
                if run_at <= now:
                    run_at += timedelta(days=1)
                message = m.group(4).strip()
        if not run_at:
            # Try "tomorrow" prefix
            m = re.match(r"tomorrow\s*(.*)", text, re.I)
            if m:
                run_at = now + timedelta(days=1)
                message = m.group(1).strip()

        if not run_at:
            await update.message.reply_text("Could not parse time. Try: /remind in 2h Check something")
            return
        if not message:
            message = "Reminder!"

        chat_id = str(update.effective_chat.id)
        job_id = f"tg_remind_{int(time.time())}_{update.effective_user.id}"

        try:
            from aura.tools.task_scheduler import TaskSchedulerTool
            scheduler = TaskSchedulerTool()
            if not scheduler._scheduler.running:
                scheduler._scheduler.start()

            scheduler._scheduler.add_job(
                self._fire_reminder, "date", run_date=run_at,
                id=job_id, args=[chat_id, message],
                replace_existing=True,
            )
            time_str = run_at.strftime("%H:%M on %b %d")
            await update.message.reply_text(f"Reminder set for {time_str}:\n{message}")
        except Exception as e:
            logger.error(f"Remind error: {e}")
            await update.message.reply_text(f"Failed to set reminder: {e}")

    def _fire_reminder(self, chat_id: str, message: str):
        """Callback for APScheduler — sends reminder to Telegram."""
        async def _send():
            try:
                await self.bot.send_message(chat_id=int(chat_id), text=f"Reminder:\n{message}")
            except Exception as e:
                logger.error(f"Failed to send reminder: {e}")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from aura.pools import fire_and_forget
                fire_and_forget(_send())
            else:
                loop.run_until_complete(_send())
        except RuntimeError:
            asyncio.run(_send())

    async def _handle_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /schedule <interval> <task> — recurring scheduled task."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args
        if not args or len(args) < 2:
            await update.message.reply_text("Usage: /schedule <interval> <task>\n\nExamples:\n/schedule every 2h Check CPU\n/schedule daily at 9am Summarize notifications")
            return
        await update.message.reply_text("Scheduled tasks coming soon! Use /remind for one-shot reminders.")

    async def _handle_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tasks — list active scheduled tasks and reminders."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        try:
            from aura.tools.task_scheduler import TaskSchedulerTool
            scheduler = TaskSchedulerTool()
            jobs = scheduler._scheduler.get_jobs() if scheduler._scheduler.running else []
            if not jobs:
                await update.message.reply_text("No active tasks or reminders.")
                return
            lines = ["Active Tasks:\n"]
            for job in jobs:
                next_run = job.next_run_time.strftime("%H:%M %b %d") if job.next_run_time else "paused"
                lines.append(f"  {job.id}\n  Next: {next_run}\n")
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            await update.message.reply_text(f"Error listing tasks: {e}")

    async def _handle_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cancel <id> — cancel a scheduled task."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /cancel <task_id>")
            return
        job_id = args[0]
        try:
            from aura.tools.task_scheduler import TaskSchedulerTool
            scheduler = TaskSchedulerTool()
            scheduler._scheduler.remove_job(job_id)
            await update.message.reply_text(f"Cancelled: {job_id}")
        except Exception as e:
            await update.message.reply_text(f"Failed to cancel: {e}")
