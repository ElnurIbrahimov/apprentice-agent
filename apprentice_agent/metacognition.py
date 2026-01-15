"""Metacognition logging system for tracking agent decision quality."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class MetacognitionLogger:
    """Logs agent actions, confidence scores, and outcomes for analysis."""

    def __init__(self, log_dir: str = "logs/metacognition"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_goal: Optional[str] = None
        self._iteration: int = 0
        self._retried: bool = False

    def start_goal(self, goal: str) -> None:
        """Mark the start of a new goal."""
        self._current_goal = goal
        self._iteration = 0
        self._retried = False

    def increment_iteration(self) -> None:
        """Increment iteration count and track retries."""
        self._iteration += 1
        if self._iteration > 1:
            self._retried = True

    def log_evaluation(
        self,
        tool: str,
        action: str,
        confidence: int,
        success: bool,
        progress: Optional[str] = None,
        next_step: Optional[str] = None,
        result_summary: Optional[str] = None,
        model_used: Optional[str] = None
    ) -> None:
        """Log an action evaluation to the JSONL file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "goal": self._current_goal,
            "iteration": self._iteration,
            "tool": tool,
            "action": action[:200] if action else None,  # Truncate long actions
            "confidence": confidence,
            "success": success,
            "retried": self._retried,
            "progress": progress,
            "next_step": next_step,
            "result_summary": result_summary[:500] if result_summary else None,
            "model_used": model_used
        }

        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _get_log_file(self) -> Path:
        """Get the log file path for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{today}.jsonl"

    def get_stats(self, date: Optional[str] = None) -> dict:
        """Get statistics from logs for a given date (default: today)."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        log_file = self.log_dir / f"{date}.jsonl"
        if not log_file.exists():
            return {"error": "No logs found for this date"}

        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if not entries:
            return {"error": "No entries in log file"}

        total = len(entries)
        successful = sum(1 for e in entries if e.get("success"))
        retried = sum(1 for e in entries if e.get("retried"))
        avg_confidence = sum(e.get("confidence", 0) for e in entries) / total

        # Tool usage breakdown
        tool_counts = {}
        for e in entries:
            tool = e.get("tool", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Model usage breakdown
        model_counts = {}
        for e in entries:
            model = e.get("model_used", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "date": date,
            "total_actions": total,
            "successful": successful,
            "success_rate": round(successful / total * 100, 1),
            "retried": retried,
            "retry_rate": round(retried / total * 100, 1),
            "avg_confidence": round(avg_confidence, 1),
            "tool_usage": tool_counts,
            "model_usage": model_counts
        }
