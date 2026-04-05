"""Morning Briefing Hand — daily digest of projects, findings, and drives."""

import asyncio
import logging
from typing import Any

from .base import Hand, HandManifest, HandResult

logger = logging.getLogger(__name__)


class MorningBriefingHand(Hand):
    """Generates a personalized daily morning briefing."""

    def get_manifest(self) -> HandManifest:
        return HandManifest(
            name="morning_briefing",
            description="Daily morning briefing with project status, research findings, and drive states",
            version="1.0",
            interval_minutes=1440,
            idle_only=False,
            max_tokens=15000,
            max_cost_usd=0.15,
            model_preference="fast",
            trigger_on_drive="social",
            trigger_drive_threshold=0.5,
        )

    def get_system_prompt(self) -> str:
        return "You are a personal assistant creating a concise, friendly morning briefing."

    async def execute(self, brain: Any, tools: dict, context: dict) -> HandResult:
        step_cb = context.get("step_callback")
        start_tokens = getattr(brain, '_session_input_tokens', 0) + getattr(brain, '_session_output_tokens', 0)

        sections = []

        # Step 1: World model summary
        if step_cb: await step_cb(1, "Gathering project status...")
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            world_summary = wm.get_context_summary(max_tokens=300) if wm.enabled else ""
            if world_summary:
                sections.append(f"## Projects & Goals\n{world_summary}")
        except Exception as e:
            logger.debug(f"[MorningBriefing] World model failed: {e}")

        # Step 2: Drive states
        if step_cb: await step_cb(2, "Checking motivation drives...")
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            drives = get_intrinsic_motivation().get_drives_summary()
            if drives:
                drive_text = ", ".join(f"{k}: {v:.0%}" for k, v in drives.items())
                sections.append(f"## Motivation Drives\n{drive_text}")
        except Exception as e:
            logger.debug(f"[MorningBriefing] Drives failed: {e}")

        # Step 3: Recent findings from memory
        if step_cb: await step_cb(3, "Reviewing recent findings...")
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            recent = um.query("recent research findings updates news", k=5)
            if recent:
                recent_text = "\n".join(f"- {m.content[:120]}" for m in recent)
                sections.append(f"## Recent Findings\n{recent_text}")
        except Exception as e:
            logger.debug(f"[MorningBriefing] Memory query failed: {e}")

        # Step 4: Dream consolidation insights
        if step_cb: await step_cb(4, "Checking dream insights...")
        try:
            from aura.dream import get_dream_consolidator
            dc = get_dream_consolidator()
            insights = getattr(dc, 'get_recent_insights', lambda: [])()
            if insights:
                insight_text = "\n".join(f"- {i}" for i in insights[:3])
                sections.append(f"## Dream Insights\n{insight_text}")
        except Exception as e:
            logger.debug(f"[MorningBriefing] Dream insights failed: {e}")

        # Step 5: Synthesize via LLM
        if step_cb: await step_cb(5, "Composing your morning briefing...")

        data_block = "\n\n".join(sections) if sections else "No data available yet. The system is still learning about your projects and interests."

        prompt = f"""Create a brief, personal morning briefing based on this data:

{data_block}

Format as a friendly, concise morning briefing (5-8 bullet points max).
Include actionable suggestions where relevant.
Start with a warm greeting."""

        try:
            briefing = await asyncio.to_thread(
                lambda: brain.think(prompt, system_prompt=self.get_system_prompt(), use_history=False)
            )
        except Exception as e:
            briefing = f"Good morning! I couldn't generate a full briefing today ({e}), but here's what I have:\n\n{data_block}"

        # Store briefing in memory
        try:
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            um.store(
                content=f"Morning briefing: {briefing[:500]}",
                metadata={"type": "morning_briefing", "hand": "morning_briefing"},
            )
        except Exception as e:
            logger.debug(f"[MorningBriefing] Memory store failed: {e}")

        end_tokens = getattr(brain, '_session_input_tokens', 0) + getattr(brain, '_session_output_tokens', 0)

        return HandResult(
            hand_name="morning_briefing",
            success=True,
            summary=briefing[:500],
            tokens_used=end_tokens - start_tokens,
            cost_usd=(end_tokens - start_tokens) * 0.000001,
        )
