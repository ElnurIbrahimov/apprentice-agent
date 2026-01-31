"""
Daily Log Manager - Append-only conversation logs

Like Clawdbot's memory/YYYY-MM-DD.md system.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class DailyLogManager:
    """Manages daily conversation logs."""

    def __init__(self, base_dir: str = "aura/data/memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DailyLogManager initialized at {self.base_dir}")

    def _get_log_path(self, date: datetime = None) -> Path:
        """Get path for a specific date's log."""
        if date is None:
            date = datetime.now()
        return self.base_dir / f"{date.strftime('%Y-%m-%d')}.md"

    def append_interaction(
        self,
        user_message: str,
        aura_response: str,
        chat_id: str = None,
        metadata: dict = None
    ):
        """Append an interaction to today's log."""

        log_path = self._get_log_path()
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Create header if new file
        if not log_path.exists():
            header = f"# Daily Log - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            log_path.write_text(header, encoding='utf-8')

        # Format entry
        entry = f"\n## [{timestamp}]"
        if chat_id:
            entry += f" (chat: {chat_id})"
        entry += f"\n\n**User:** {user_message}\n\n**AURA:** {aura_response}\n"

        if metadata:
            entry += f"\n*Metadata: {metadata}*\n"

        entry += "\n---\n"

        # Append to file
        with open(log_path, "a", encoding='utf-8') as f:
            f.write(entry)

        logger.debug(f"Appended interaction to {log_path}")

    def get_today_log(self) -> str:
        """Get today's complete log."""
        log_path = self._get_log_path()
        if log_path.exists():
            return log_path.read_text(encoding='utf-8')
        return ""

    def get_yesterday_log(self) -> str:
        """Get yesterday's complete log."""
        yesterday = datetime.now() - timedelta(days=1)
        log_path = self._get_log_path(yesterday)
        if log_path.exists():
            return log_path.read_text(encoding='utf-8')
        return ""

    def get_recent_logs(self, days: int = 7) -> List[str]:
        """Get logs from recent days."""
        logs = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            log_path = self._get_log_path(date)
            if log_path.exists():
                content = log_path.read_text(encoding='utf-8')
                logs.append(f"=== {date.strftime('%Y-%m-%d')} ===\n{content}")
        return logs

    def search_logs(self, query: str, days: int = 30) -> List[str]:
        """Simple keyword search through logs."""
        results = []
        query_lower = query.lower()

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            log_path = self._get_log_path(date)

            if log_path.exists():
                content = log_path.read_text(encoding='utf-8')

                # Find matching sections
                sections = content.split("## [")
                for section in sections[1:]:  # Skip header
                    if query_lower in section.lower():
                        results.append(f"[{date.strftime('%Y-%m-%d')}] {section[:500]}...")

        return results[:10]  # Limit results

    def get_log_summary(self, days: int = 7) -> str:
        """Get a summary of recent activity."""
        logs = self.get_recent_logs(days)

        if not logs:
            return "No recent conversations."

        # Count interactions
        total_interactions = sum(log.count("**User:**") for log in logs)

        return f"Last {days} days: {total_interactions} interactions across {len(logs)} days."
