"""Dream Mode - Memory consolidation and pattern analysis.

Analyzes metacognition logs to extract insights about agent behavior,
tool effectiveness, and learning opportunities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .brain import OllamaBrain
from .memory import MemorySystem
from .metacognition import MetacognitionLogger


class DreamMode:
    """Consolidates memories and extracts insights from agent experiences."""

    def __init__(self):
        self.metacog = MetacognitionLogger()
        self.memory = MemorySystem(collection_name="dream_insights")
        self.brain = OllamaBrain()

    def dream(self, date: Optional[str] = None) -> dict:
        """Run dream mode to consolidate memories and generate insights.

        Args:
            date: Date to analyze (YYYY-MM-DD). Defaults to today.

        Returns:
            Dictionary with analysis results and generated insights.
        """
        print("\n" + "=" * 60)
        print("DREAM MODE - Memory Consolidation")
        print("=" * 60 + "\n")

        # Step 1: Load today's logs
        print("[1/4] Loading metacognition logs...")
        logs = self._load_logs(date)
        if not logs:
            print("No logs found for analysis.")
            return {"success": False, "error": "No logs found"}
        print(f"      Found {len(logs)} log entries")

        # Step 2: Analyze patterns
        print("\n[2/4] Analyzing patterns...")
        patterns = self._analyze_patterns(logs)
        self._print_patterns(patterns)

        # Step 3: Generate insights using LLM
        print("\n[3/4] Generating insights...")
        insights = self._generate_insights(patterns, logs)
        print(f"      Generated {len(insights)} insights")

        # Step 4: Store insights in long-term memory
        print("\n[4/4] Storing insights in memory...")
        stored_ids = self._store_insights(insights, date)
        print(f"      Stored {len(stored_ids)} insights")

        # Print insights
        print("\n" + "-" * 60)
        print("INSIGHTS GENERATED:")
        print("-" * 60)
        for i, insight in enumerate(insights, 1):
            print(f"\n{i}. {insight}")

        print("\n" + "=" * 60)
        print("DREAM MODE COMPLETE")
        print("=" * 60 + "\n")

        return {
            "success": True,
            "logs_analyzed": len(logs),
            "patterns": patterns,
            "insights": insights,
            "stored_ids": stored_ids
        }

    def _load_logs(self, date: Optional[str] = None) -> list[dict]:
        """Load metacognition logs for the given date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        log_file = Path("logs/metacognition") / f"{date}.jsonl"
        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return logs

    def _analyze_patterns(self, logs: list[dict]) -> dict:
        """Analyze patterns in the logs."""
        patterns = {
            "total_actions": len(logs),
            "tools": {},
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0},
            "retry_analysis": {"first_attempt_success": 0, "needed_retry": 0},
            "goals": {}
        }

        for log in logs:
            tool = log.get("tool", "unknown")
            confidence = log.get("confidence", 0)
            success = log.get("success", False)
            retried = log.get("retried", False)
            goal = log.get("goal", "unknown")

            # Tool stats
            if tool not in patterns["tools"]:
                patterns["tools"][tool] = {
                    "total": 0,
                    "success": 0,
                    "avg_confidence": [],
                    "actions": []
                }
            patterns["tools"][tool]["total"] += 1
            if success:
                patterns["tools"][tool]["success"] += 1
            patterns["tools"][tool]["avg_confidence"].append(confidence)
            patterns["tools"][tool]["actions"].append(log.get("action", "")[:100])

            # Confidence distribution
            if confidence < 50:
                patterns["confidence_distribution"]["low"] += 1
            elif confidence < 80:
                patterns["confidence_distribution"]["medium"] += 1
            else:
                patterns["confidence_distribution"]["high"] += 1

            # Retry analysis - only count successful non-retried actions
            if success and not retried:
                patterns["retry_analysis"]["first_attempt_success"] += 1
            elif retried:
                patterns["retry_analysis"]["needed_retry"] += 1

            # Goal tracking
            if goal not in patterns["goals"]:
                patterns["goals"][goal] = {"attempts": 0, "completed": False}
            patterns["goals"][goal]["attempts"] += 1
            if success and log.get("next_step") == "complete":
                patterns["goals"][goal]["completed"] = True

        # Calculate averages
        for tool, stats in patterns["tools"].items():
            if stats["avg_confidence"]:
                stats["avg_confidence"] = round(
                    sum(stats["avg_confidence"]) / len(stats["avg_confidence"]), 1
                )
            else:
                stats["avg_confidence"] = 0
            stats["success_rate"] = round(
                (stats["success"] / stats["total"]) * 100, 1
            ) if stats["total"] > 0 else 0

        return patterns

    def _print_patterns(self, patterns: dict) -> None:
        """Print pattern analysis results."""
        print(f"\n      Total actions: {patterns['total_actions']}")
        print(f"\n      Tool Performance:")
        for tool, stats in patterns["tools"].items():
            print(f"        - {tool}: {stats['success_rate']}% success, "
                  f"avg confidence {stats['avg_confidence']}%")

        print(f"\n      Confidence Distribution:")
        cd = patterns["confidence_distribution"]
        print(f"        - Low (<50%): {cd['low']}")
        print(f"        - Medium (50-80%): {cd['medium']}")
        print(f"        - High (>80%): {cd['high']}")

        print(f"\n      Retry Analysis:")
        ra = patterns["retry_analysis"]
        print(f"        - First attempt success: {ra['first_attempt_success']}")
        print(f"        - Needed retry: {ra['needed_retry']}")

    def _generate_insights(self, patterns: dict, logs: list[dict]) -> list[str]:
        """Generate insights using the LLM."""
        insights = []

        # Prepare summary for LLM
        tool_summary = "\n".join([
            f"- {tool}: {stats['total']} uses, {stats['success_rate']}% success rate, "
            f"avg confidence {stats['avg_confidence']}%"
            for tool, stats in patterns["tools"].items()
        ])

        # Sample actions for context
        sample_actions = []
        for log in logs[:5]:
            sample_actions.append(
                f"Goal: {log.get('goal', 'N/A')[:50]}, "
                f"Tool: {log.get('tool')}, "
                f"Success: {log.get('success')}, "
                f"Confidence: {log.get('confidence')}%"
            )

        prompt = f"""Analyze this agent's performance data and generate 3-5 actionable insights.

TOOL PERFORMANCE:
{tool_summary}

CONFIDENCE DISTRIBUTION:
- Low confidence (<50%): {patterns['confidence_distribution']['low']} actions
- Medium confidence (50-80%): {patterns['confidence_distribution']['medium']} actions
- High confidence (>80%): {patterns['confidence_distribution']['high']} actions

RETRY STATS:
- First attempt successes: {patterns['retry_analysis']['first_attempt_success']}
- Needed retry: {patterns['retry_analysis']['needed_retry']}

SAMPLE ACTIONS:
{chr(10).join(sample_actions)}

Generate 3-5 specific, actionable insights about:
1. Which tools work best and when to use them
2. What confidence levels indicate about success likelihood
3. How to reduce retries and improve first-attempt success
4. Patterns in successful vs unsuccessful actions

Format each insight as a single clear sentence starting with a verb (Use, Prefer, Avoid, etc.)."""

        response = self.brain.think(
            prompt,
            system_prompt="You analyze agent behavior patterns and generate concise, actionable insights. Be specific and practical.",
            use_history=False
        )

        # Parse insights from response
        for line in response.split("\n"):
            line = line.strip()
            # Skip empty lines and headers
            if not line or line.startswith("#") or line.startswith("*"):
                continue
            # Remove numbering
            if line[0].isdigit() and (line[1] == "." or line[1] == ")"):
                line = line[2:].strip()
            elif line[0].isdigit() and line[1].isdigit() and line[2] in ".)" :
                line = line[3:].strip()
            # Remove bullet points
            if line.startswith("- "):
                line = line[2:]
            if line.startswith("• "):
                line = line[2:]
            # Keep meaningful insights
            if len(line) > 20 and any(c.isalpha() for c in line):
                insights.append(line)

        return insights[:5]  # Limit to 5 insights

    def _store_insights(self, insights: list[str], date: Optional[str] = None) -> list[str]:
        """Store insights in long-term memory."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        stored_ids = []
        for insight in insights:
            memory_id = self.memory.remember(
                content=insight,
                memory_type="dream_insight",
                metadata={
                    "source": "dream_mode",
                    "date_analyzed": date,
                    "generated_at": datetime.now().isoformat()
                }
            )
            stored_ids.append(memory_id)

        return stored_ids

    def recall_insights(self, query: str, n_results: int = 5) -> list[dict]:
        """Recall relevant insights from memory."""
        return self.memory.recall(query, n_results=n_results)

    def get_all_insights(self) -> list[dict]:
        """Get all stored insights."""
        return [m for m in self.memory.memories if m.get("type") == "dream_insight"]


def run_dream_mode(date: Optional[str] = None) -> dict:
    """Entry point for running dream mode."""
    dreamer = DreamMode()
    return dreamer.dream(date)
