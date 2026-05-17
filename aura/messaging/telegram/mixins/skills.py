"""
SkillsMixin — /learn, /skill, _skill_list/search/info/improve/create_start,
              _check_skill_pending, _find_skill_by_id_or_name,
              _get_skill_store, _get_skill_learner
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time as _time

from aura.paths import EVOLUTION_RUNS_DIR

try:
    from telegram import Update
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

try:
    from aura_skill_library import (
        Skill,
        SkillCategory,
        SkillExample,
        SkillLearner,
        SkillMetadata,
        SkillStore,
    )
    SKILL_LIBRARY_AVAILABLE = True
except ImportError:
    SKILL_LIBRARY_AVAILABLE = False

try:
    from aura.evolution import AuraSkillAdapter, Candidate, GEPAConfig, GEPAEngine, GEPAResult
    from aura.evolution.types import EvalExample
    GEPA_AVAILABLE = True
except (ImportError, Exception):
    GEPA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SkillsMixin:
    """Skill system handlers — /learn and /skill subcommands."""

    def _get_skill_store(self) -> "SkillStore":
        """Lazy-load the SkillStore singleton."""
        if self._skill_store is None:
            if not SKILL_LIBRARY_AVAILABLE:
                raise RuntimeError("Skill library not installed")
            self._skill_store = SkillStore(storage_path="./aura_skills")
        return self._skill_store

    def _get_skill_learner(self) -> "SkillLearner":
        """Lazy-load the SkillLearner singleton."""
        if self._skill_learner is None:
            if not SKILL_LIBRARY_AVAILABLE:
                raise RuntimeError("Skill library not installed")
            store = self._get_skill_store()
            # Try to get an LLM function from the aura engine
            llm_func = None
            try:
                brain = getattr(self.aura, 'brain', None)
                if brain is None:
                    brain = getattr(getattr(self.aura, 'agent', None), 'brain', None)
                if brain and hasattr(brain, 'think'):
                    def llm_func(prompt):
                        return brain.think(prompt)
            except Exception:
                pass
            self._skill_learner = SkillLearner(
                store=store,
                llm_func=llm_func,
                min_examples_to_learn=1,
            )
        return self._skill_learner

    async def _handle_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /learn -- extract a reusable skill from the last conversation exchange."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        user_id = update.effective_user.id

        if not SKILL_LIBRARY_AVAILABLE:
            await update.message.reply_text("Skill library is not available on this server.")
            return

        exchange = self.store.get_skill_state(str(user_id)).get("last_exchange") or None
        if not exchange:
            await update.message.reply_text(
                "No recent conversation to learn from.\n"
                "Chat with me first, then use /learn to extract a skill."
            )
            return

        if _time.time() - exchange.get("timestamp", 0) > 1800:
            await update.message.reply_text(
                "The last exchange is too old (> 30 min).\n"
                "Have a fresh conversation first, then /learn."
            )
            return

        await update.message.reply_text("Analyzing our last exchange to create a skill...")
        await self.send_typing_indicator(str(update.effective_chat.id))

        try:
            learner = self._get_skill_learner()

            if learner.llm_func is None:
                await update.message.reply_text(
                    "Cannot learn skills right now -- no LLM backend available."
                )
                return

            prompt = (
                "Analyze this conversation exchange and extract a reusable skill.\n\n"
                f"User said: {exchange['input'][:1000]}\n\n"
                f"AURA responded: {exchange['output'][:1000]}\n\n"
                "Create a skill definition with:\n"
                "1. A clear, concise name (2-4 words)\n"
                "2. A description of what this skill does\n"
                "3. 3-5 trigger phrases that would activate this skill\n"
                "4. A step-by-step procedure that generalizes from this exchange\n"
                "5. The best category: coding, writing, research, automation, analysis, communication, learning\n\n"
                'Respond in this exact JSON format:\n'
                '{"name": "Skill Name", "description": "What this skill does...", '
                '"trigger_patterns": ["phrase 1", "phrase 2", "phrase 3"], '
                '"procedure": "Step 1: ...\\nStep 2: ...\\nStep 3: ...", '
                '"category": "coding", "tags": ["tag1", "tag2"]}\n\n'
                "Respond ONLY with the JSON, no other text."
            )

            response = await asyncio.to_thread(learner.llm_func, prompt)

            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                await update.message.reply_text(
                    "Could not extract a skill from that exchange. Try a more structured interaction."
                )
                return

            skill_data = json.loads(json_match.group())

            required = ["name", "description", "trigger_patterns", "procedure"]
            for fld in required:
                if fld not in skill_data:
                    await update.message.reply_text(f"Skill extraction incomplete -- missing {fld}. Try again.")
                    return

            triggers_str = ", ".join(f'"{t}"' for t in skill_data["trigger_patterns"][:5])
            procedure_preview = skill_data["procedure"][:300]
            if len(skill_data["procedure"]) > 300:
                procedure_preview += "..."

            msg = (
                f'Skill: "{skill_data["name"]}"\n'
                f'Category: {skill_data.get("category", "custom")}\n'
                f'Triggers: {triggers_str}\n'
                f'Procedure: {procedure_preview}\n\n'
                f'Save this skill? (reply "yes" to confirm)'
            )
            await update.message.reply_text(msg)

            self._skill_pending[user_id] = {
                "action": "learn_confirm",
                "skill_data": skill_data,
                "exchange": exchange,
                "timestamp": _time.time(),
            }

        except json.JSONDecodeError:
            await update.message.reply_text("Failed to parse skill data. The LLM returned invalid JSON.")
        except Exception as e:
            logger.error(f"[Telegram] /learn error: {e}", exc_info=True)
            await update.message.reply_text(f"Failed to learn skill: {e}")

    async def _handle_skill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /skill command -- dispatch to subcommands."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        if not SKILL_LIBRARY_AVAILABLE:
            await update.message.reply_text("Skill library is not available on this server.")
            return

        args = context.args or []

        if not args:
            await update.message.reply_text(
                "Usage:\n"
                "/skill list [category] -- List all skills\n"
                "/skill search <query> -- Search skills\n"
                "/skill info <name> -- Detailed skill info\n"
                "/skill improve <name> -- Evolve with GEPA\n"
                "/skill create <name> -- Define a new skill"
            )
            return

        subcmd = args[0].lower()
        rest = args[1:]

        if subcmd == "list":
            await self._skill_list(update, " ".join(rest) if rest else None)
        elif subcmd == "search":
            if not rest:
                await update.message.reply_text("Usage: /skill search <query>")
                return
            await self._skill_search(update, " ".join(rest))
        elif subcmd == "info":
            if not rest:
                await update.message.reply_text("Usage: /skill info <id_or_name>")
                return
            await self._skill_info(update, " ".join(rest))
        elif subcmd == "improve":
            if not rest:
                await update.message.reply_text("Usage: /skill improve <id_or_name>")
                return
            await self._skill_improve(update, " ".join(rest))
        elif subcmd == "create":
            if not rest:
                await update.message.reply_text("Usage: /skill create <name>")
                return
            await self._skill_create_start(update, " ".join(rest))
        else:
            await update.message.reply_text(
                f"Unknown subcommand: {subcmd}\n"
                "Use /skill for available commands."
            )

    async def _skill_list(self, update: Update, category_str: str | None = None):
        """List all learned skills, optionally filtered by category."""
        try:
            store = self._get_skill_store()

            cat_filter = None
            if category_str:
                try:
                    cat_filter = SkillCategory(category_str.lower())
                except ValueError:
                    await update.message.reply_text(
                        f"Unknown category: {category_str}\n"
                        f"Valid: {', '.join(c.value for c in SkillCategory)}"
                    )
                    return

            skills = store.list_all(category=cat_filter, sort_by="updated")

            if not skills:
                msg = "No skills found."
                if category_str:
                    msg += f" (category: {category_str})"
                await update.message.reply_text(msg)
                return

            page = skills[:10]
            lines = ["Learned Skills\n"]
            for i, s in enumerate(page, 1):
                name = s.get("name", "Unnamed")
                success_rate = s.get("success_rate", 0)
                total_uses = s.get("total_uses", 0)
                updated = s.get("updated_at", "")
                if updated and len(updated) > 10:
                    updated = updated[:10]

                rate_pct = f"{success_rate * 100:.0f}%" if success_rate else "N/A"
                lines.append(
                    f"{i}. {name}\n"
                    f"   Rate: {rate_pct} | Uses: {total_uses} | Updated: {updated}"
                )

            if len(skills) > 10:
                lines.append(f"\n... and {len(skills) - 10} more skills")

            lines.append("\nUse /skill info <name> for details")
            await update.message.reply_text("\n".join(lines))

        except Exception as e:
            logger.error(f"[Telegram] /skill list error: {e}", exc_info=True)
            await update.message.reply_text(f"Failed to list skills: {e}")

    async def _skill_search(self, update: Update, query: str):
        """Search skills by description/name."""
        try:
            store = self._get_skill_store()
            results = store.search(query, limit=5)

            if not results:
                await update.message.reply_text(f"No skills found matching: {query}")
                return

            lines = [f'Search results for "{query}"\n']
            for skill_id, score in results:
                info = store.index.get(skill_id, {})
                name = info.get("name", skill_id)
                desc = info.get("description", "")[:80]
                lines.append(f"  {name} (score: {score:.2f})\n   {desc}")

            lines.append("\nUse /skill info <name> for details")
            await update.message.reply_text("\n".join(lines))

        except Exception as e:
            logger.error(f"[Telegram] /skill search error: {e}", exc_info=True)
            await update.message.reply_text(f"Search failed: {e}")

    async def _skill_info(self, update: Update, id_or_name: str):
        """Show detailed info about a skill."""
        try:
            store = self._get_skill_store()
            skill, _skill_id = self._find_skill_by_id_or_name(store, id_or_name)

            if not skill:
                await update.message.reply_text(f"Skill not found: {id_or_name}")
                return

            triggers = ", ".join(f'"{t}"' for t in skill.trigger_patterns[:5])
            rate_pct = f"{skill.metadata.success_rate * 100:.0f}%" if skill.metadata.total_uses > 0 else "N/A"
            last_used = skill.metadata.last_used.strftime("%Y-%m-%d %H:%M") if skill.metadata.last_used else "Never"

            lines = [
                f"Skill: {skill.name}",
                f"ID: {skill.id}",
                f"Version: {skill.metadata.version}",
                f"Category: {skill.category.value}",
                f"Description: {skill.description}",
                "",
                f"Triggers: {triggers}",
                "",
                "Procedure:",
                skill.procedure[:500],
                "",
                f"Success Rate: {rate_pct}",
                f"Total Uses: {skill.metadata.total_uses}",
                f"Last Used: {last_used}",
                f"Tags: {', '.join(skill.metadata.tags) if skill.metadata.tags else 'None'}",
            ]

            if skill.metadata.parent_skill_id:
                lines.append(f"Evolved from: {skill.metadata.parent_skill_id}")

            await update.message.reply_text("\n".join(lines))

        except Exception as e:
            logger.error(f"[Telegram] /skill info error: {e}", exc_info=True)
            await update.message.reply_text(f"Failed to get skill info: {e}")

    async def _skill_improve(self, update: Update, id_or_name: str):
        """Trigger GEPA evolution on a skill."""
        user_id = update.effective_user.id

        if not GEPA_AVAILABLE:
            await update.message.reply_text("GEPA evolution engine is not available on this server.")
            return

        try:
            store = self._get_skill_store()
            skill, skill_id = self._find_skill_by_id_or_name(store, id_or_name)

            if not skill:
                await update.message.reply_text(f"Skill not found: {id_or_name}")
                return

            await update.message.reply_text(
                f'Evolving "{skill.name}"...\nThis may take a minute.'
            )
            await self.send_typing_indicator(str(update.effective_chat.id))

            brain = getattr(self.aura, 'brain', None)
            if brain is None:
                brain = getattr(getattr(self.aura, 'agent', None), 'brain', None)

            if not brain or not hasattr(brain, 'think'):
                await update.message.reply_text("Cannot evolve -- no LLM backend available.")
                return

            def llm_func(system: str, user: str) -> str:
                return brain.think(f"{system}\n\n{user}")

            config = GEPAConfig(
                max_iterations=3,
                max_metric_calls=30,
                timeout_seconds=120,
                no_improvement_patience=2,
                run_dir=str(EVOLUTION_RUNS_DIR / f"telegram_{skill_id}"),
            )

            adapter = AuraSkillAdapter(config=config, llm_func=llm_func)

            seed = Candidate(
                id=0,
                components={skill_id: skill.procedure},
                parent_id=-1,
            )

            eval_examples = await asyncio.to_thread(
                adapter.generate_eval_dataset, seed, num_examples=6
            )

            if len(eval_examples) < 2:
                await update.message.reply_text(
                    "Could not generate enough evaluation examples. "
                    "The skill may be too simple to evolve."
                )
                return

            engine = GEPAEngine(config=config, adapter=adapter, llm_func=llm_func)

            result = await asyncio.to_thread(
                engine.optimize, seed, eval_examples
            )

            best = result.best_candidate
            improved_procedure = best.components.get(skill_id, skill.procedure)

            if result.improvement <= 0.01:
                await update.message.reply_text(
                    f'Evolution complete for "{skill.name}"\n\n'
                    f"Iterations: {result.iterations_run}\n"
                    f"Score: {best.avg_score:.2f}\n"
                    f"No significant improvement found. The skill is already good!"
                )
                return

            procedure_preview = improved_procedure[:400]
            if len(improved_procedure) > 400:
                procedure_preview += "..."

            msg = (
                f'Evolution complete for "{skill.name}"\n\n'
                f"Iterations: {result.iterations_run}\n"
                f"Improvement: +{result.improvement:.3f}\n"
                f"Best score: {best.avg_score:.2f}\n\n"
                f"Improved procedure:\n{procedure_preview}\n\n"
                f'Apply improvement? (reply "yes" to confirm)'
            )
            await update.message.reply_text(msg)

            self._skill_pending[user_id] = {
                "action": "improve_confirm",
                "skill_id": skill_id,
                "improved_procedure": improved_procedure,
                "improvement": result.improvement,
                "timestamp": _time.time(),
            }

        except Exception as e:
            logger.error(f"[Telegram] /skill improve error: {e}", exc_info=True)
            await update.message.reply_text(f"Evolution failed: {e}")

    async def _skill_create_start(self, update: Update, name: str):
        """Start the interactive skill creation flow."""
        user_id = update.effective_user.id

        self._skill_create_state[user_id] = {
            "step": "description",
            "name": name,
            "timestamp": _time.time(),
        }

        await update.message.reply_text(
            f'Creating skill: "{name}"\n\n'
            "Step 1/3: What does this skill do?\n"
            "(Send the description)"
        )

    async def _check_skill_pending(self, update: Update, user_id: int, text: str) -> bool:
        """Check and handle pending skill actions. Returns True if handled."""
        text_lower = text.strip().lower() if text else ""

        # Handle skill create multi-step flow
        if user_id in self._skill_create_state:
            state = self._skill_create_state[user_id]

            # Timeout after 10 minutes
            if _time.time() - state.get("timestamp", 0) > 600:
                del self._skill_create_state[user_id]
                return False

            step = state.get("step")

            if step == "description":
                state["description"] = text.strip()
                state["step"] = "triggers"
                state["timestamp"] = _time.time()
                await update.message.reply_text(
                    "Step 2/3: What phrases should trigger this skill?\n"
                    "(Send comma-separated trigger phrases)"
                )
                return True

            elif step == "triggers":
                triggers = [t.strip() for t in text.split(",") if t.strip()]
                if not triggers:
                    await update.message.reply_text(
                        "Please provide at least one trigger phrase, separated by commas."
                    )
                    return True
                state["triggers"] = triggers
                state["step"] = "procedure"
                state["timestamp"] = _time.time()
                await update.message.reply_text(
                    "Step 3/3: What is the step-by-step procedure?\n"
                    "(Send the procedure -- use numbered steps)"
                )
                return True

            elif step == "procedure":
                state["procedure"] = text.strip()

                try:
                    store = self._get_skill_store()
                    skill = Skill.create(
                        name=state["name"],
                        description=state["description"],
                        category=SkillCategory.CUSTOM,
                        trigger_patterns=state["triggers"],
                        procedure=state["procedure"],
                        tags=[],
                    )
                    skill_id = store.save(skill)
                    del self._skill_create_state[user_id]

                    await update.message.reply_text(
                        f"Skill created!\n\n"
                        f"Name: {state['name']}\n"
                        f"ID: {skill_id}\n"
                        f"Triggers: {', '.join(state['triggers'])}\n\n"
                        f"Use /skill info {state['name']} to see full details."
                    )
                except Exception as e:
                    logger.error(f"[Telegram] skill create error: {e}", exc_info=True)
                    await update.message.reply_text(f"Failed to create skill: {e}")
                    if user_id in self._skill_create_state:
                        del self._skill_create_state[user_id]

                return True

        # Handle pending confirmations (learn_confirm, improve_confirm)
        if user_id in self._skill_pending:
            pending = self._skill_pending[user_id]

            # Timeout after 5 minutes
            if _time.time() - pending.get("timestamp", 0) > 300:
                del self._skill_pending[user_id]
                return False

            if text_lower != "yes":
                del self._skill_pending[user_id]
                await update.message.reply_text("Cancelled.")
                return True

            action = pending.get("action")

            if action == "learn_confirm":
                try:
                    skill_data = pending["skill_data"]
                    store = self._get_skill_store()

                    category_str = skill_data.get("category", "custom").lower()
                    try:
                        category = SkillCategory(category_str)
                    except ValueError:
                        category = SkillCategory.CUSTOM

                    skill = Skill.create(
                        name=skill_data["name"],
                        description=skill_data["description"],
                        category=category,
                        trigger_patterns=skill_data["trigger_patterns"],
                        procedure=skill_data["procedure"],
                        tags=skill_data.get("tags", []),
                    )
                    skill.id = f"learned_{skill.id}"

                    exchange = pending.get("exchange", {})
                    if exchange:
                        example = SkillExample(
                            input_context=exchange.get("input", ""),
                            input_data=None,
                            output=exchange.get("output", ""),
                            success=True,
                        )
                        skill.add_example(example)

                    skill_id = store.save(skill)
                    del self._skill_pending[user_id]

                    await update.message.reply_text(
                        f"Skill saved!\n\n"
                        f"Name: {skill_data['name']}\n"
                        f"ID: {skill_id}\n\n"
                        "I'll use this skill in future similar conversations."
                    )
                except Exception as e:
                    logger.error(f"[Telegram] learn confirm error: {e}", exc_info=True)
                    await update.message.reply_text(f"Failed to save skill: {e}")
                    if user_id in self._skill_pending:
                        del self._skill_pending[user_id]
                return True

            elif action == "improve_confirm":
                try:
                    skill_id_pending = pending["skill_id"]
                    improved_procedure = pending["improved_procedure"]
                    store = self._get_skill_store()
                    skill = store.load(skill_id_pending)

                    if not skill:
                        await update.message.reply_text("Skill no longer exists.")
                        del self._skill_pending[user_id]
                        return True

                    skill.procedure = improved_procedure
                    try:
                        current_version = float(skill.metadata.version)
                        skill.metadata.version = f"{current_version + 0.1:.1f}"
                    except ValueError:
                        skill.metadata.version = "1.1"

                    from datetime import datetime, timezone
                    skill.metadata.last_modified = datetime.now(timezone.utc)
                    skill.updated_at = datetime.now(timezone.utc)
                    store.save(skill)

                    del self._skill_pending[user_id]
                    await update.message.reply_text(
                        f'Improvement applied to "{skill.name}"!\n'
                        f"Version: {skill.metadata.version}\n"
                        f"Improvement: +{pending.get('improvement', 0):.3f}"
                    )
                except Exception as e:
                    logger.error(f"[Telegram] improve confirm error: {e}", exc_info=True)
                    await update.message.reply_text(f"Failed to apply improvement: {e}")
                    if user_id in self._skill_pending:
                        del self._skill_pending[user_id]
                return True

        return False

    def _find_skill_by_id_or_name(self, store, id_or_name: str):
        """Find a skill by ID or name (case-insensitive partial match).

        Returns (Skill, skill_id) or (None, None).
        """
        # Try direct ID match
        skill = store.load(id_or_name)
        if skill:
            return skill, id_or_name

        # Try name match (case-insensitive)
        name_lower = id_or_name.lower()
        for sid, info in store.index.items():
            if info.get("name", "").lower() == name_lower:
                skill = store.load(sid)
                if skill:
                    return skill, sid

        # Try partial name match
        for sid, info in store.index.items():
            if name_lower in info.get("name", "").lower():
                skill = store.load(sid)
                if skill:
                    return skill, sid

        # Try slug match (name with hyphens)
        slug = name_lower.replace(" ", "-")
        for sid, info in store.index.items():
            stored_slug = info.get("name", "").lower().replace(" ", "-")
            if slug == stored_slug or slug in stored_slug:
                skill = store.load(sid)
                if skill:
                    return skill, sid

        return None, None
