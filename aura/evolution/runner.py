"""
GEPA Runner — CLI and API integration for running skill evolution.

Usage:
    python -m aura.evolution.runner                    # Evolve all skills
    python -m aura.evolution.runner --skill skill_abc  # Evolve one skill
    python -m aura.evolution.runner --dry-run           # Preview without running
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_llm_func(model: str, base_url: str | None = None):
    """Create an LLM function using Ollama's API."""
    import os as _os
    import urllib.error
    import urllib.request
    if base_url is None:
        base_url = _os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def llm_func(system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                return data.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama call failed ({model}): {e}") from e

    return llm_func


def run_evolution(
    skill_ids: Optional[list] = None,
    config_overrides: Optional[dict] = None,
    dry_run: bool = False,
) -> dict:
    """
    Run GEPA evolution on Aura's skill library.

    Args:
        skill_ids: Specific skills to evolve (None = all)
        config_overrides: Override default GEPAConfig values
        dry_run: If True, just show what would happen

    Returns:
        Result summary dict
    """
    from .adapter import AuraSkillAdapter
    from .engine import GEPAEngine
    from .types import GEPAConfig

    # Load config
    config = GEPAConfig()
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Timestamp the run
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.run_dir = str(_PROJECT_ROOT / "aura_data" / "evolution_runs" / run_name)

    # Create LLM functions
    reflect_llm = create_llm_func(config.reflection_model, config.ollama_base_url)
    eval_llm = create_llm_func(config.eval_model, config.ollama_base_url)

    # Load skill store
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from aura_skill_library.skill_store import SkillStore

    store = SkillStore(storage_path=str(_PROJECT_ROOT / "aura_data" / "skill_library"))

    if not store.index:
        logger.error("No skills found in skill library. Learn some skills first.")
        return {"error": "No skills to evolve"}

    # Create adapter
    adapter = AuraSkillAdapter(config, reflect_llm, eval_llm)

    # Load seed candidate
    seed = adapter.load_skills_as_candidate(store)

    # Filter to specific skills if requested
    if skill_ids:
        seed.components = {
            sid: text for sid, text in seed.components.items()
            if sid in skill_ids
        }

    if not seed.components:
        logger.error("No matching skills found")
        return {"error": "No matching skills"}

    logger.info(f"Evolving {len(seed.components)} skills: {list(seed.components.keys())}")

    if dry_run:
        return {
            "dry_run": True,
            "skills": list(seed.components.keys()),
            "config": {
                "max_iterations": config.max_iterations,
                "reflection_model": config.reflection_model,
                "eval_model": config.eval_model,
            },
        }

    # Generate evaluation dataset
    logger.info("Generating evaluation dataset...")
    eval_examples = adapter.generate_eval_dataset(seed, num_examples=20)

    if len(eval_examples) < 3:
        logger.error(f"Only generated {len(eval_examples)} examples, need at least 3")
        return {"error": "Insufficient eval examples"}

    # Run GEPA
    engine = GEPAEngine(config, adapter, reflect_llm)
    result = engine.optimize(seed, eval_examples)

    # Apply best candidate back to skill store
    if result.improvement > 0:
        logger.info(f"Applying improvements (+{result.improvement:.3f})...")
        applied = _apply_to_store(store, result.best_candidate)
        logger.info(f"Updated {applied} skills")
    else:
        logger.info("No improvement found. Skills unchanged.")
        applied = 0

    # Report real-vs-synthetic eval mix so we can see the echo chamber shrink
    # over time as real episodes accumulate.
    source_mix_report = []
    if hasattr(adapter, "source_mix") and adapter.source_mix:
        for m in adapter.source_mix:
            source_mix_report.append({
                "skill_id": m.skill_id,
                "real": m.real_count,
                "synthetic": m.synthetic_count,
                "source": m.source,
            })

    return {
        "success": True,
        "seed_score": result.best_candidate.avg_score - result.improvement,
        "best_score": result.best_candidate.avg_score,
        "improvement": result.improvement,
        "skills_updated": applied,
        "iterations": result.iterations_run,
        "total_evals": result.total_evals,
        "duration_seconds": result.duration_seconds,
        "stop_reason": result.stop_reason,
        "run_dir": config.run_dir,
        "source_mix": source_mix_report,
    }


def _snapshot_skill_version(skill) -> Optional[Path]:
    """Save a pre-mutation copy of the skill so GEPA can be rolled back.

    Returns the path written, or None if snapshotting failed.
    """
    try:
        from aura.paths import SKILL_LIBRARY_DIR
        versions_dir = SKILL_LIBRARY_DIR / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        snap_path = versions_dir / f"{skill.id}_v{skill.metadata.version}_{ts}.json"
        payload = {
            "id": skill.id,
            "name": skill.name,
            "version": skill.metadata.version,
            "procedure": skill.procedure,
            "snapshotted_at": datetime.now(timezone.utc).isoformat(),
        }
        snap_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return snap_path
    except Exception:
        logger.debug("skill_snapshot_failed", exc_info=True)
        return None


def _apply_to_store(store, candidate) -> int:
    """Apply evolved skill procedures back to the SkillStore.

    On save failure we restore the in-memory skill to its pre-mutation state
    so a later successful write (e.g. after transient disk issue) doesn't
    carry a half-applied mutation.
    """
    applied = 0

    for skill_id, new_procedure in candidate.components.items():
        skill = store.load(skill_id)
        if not skill:
            continue

        if skill.procedure == new_procedure:
            continue  # No change

        # Snapshot pre-mutation fields for in-memory rollback on save failure,
        # and write a versioned snapshot to disk for manual rollback later.
        old_version = skill.metadata.version
        old_procedure = skill.procedure
        old_last_modified = skill.metadata.last_modified
        old_parent_skill_id = skill.metadata.parent_skill_id
        old_updated_at = skill.updated_at
        _snapshot_skill_version(skill)

        # Bump version
        try:
            ver = float(skill.metadata.version)
            skill.metadata.version = f"{ver + 0.1:.1f}"
        except ValueError:
            skill.metadata.version = "1.1"

        skill.procedure = new_procedure
        skill.metadata.last_modified = datetime.now(timezone.utc)
        skill.metadata.parent_skill_id = skill.id  # Track lineage
        skill.updated_at = datetime.now(timezone.utc)

        try:
            store.save(skill)
        except Exception as e:
            # Restore in-memory skill so a retry starts from a clean slate.
            skill.metadata.version = old_version
            skill.procedure = old_procedure
            skill.metadata.last_modified = old_last_modified
            skill.metadata.parent_skill_id = old_parent_skill_id
            skill.updated_at = old_updated_at
            logger.error(
                "GEPA save failed for skill %s (v%s): %s — in-memory state rolled back",
                skill.name, old_version, e,
            )
            raise
        applied += 1
        logger.info(f"Updated skill: {skill.name} (v{skill.metadata.version})")

    return applied


def main():
    parser = argparse.ArgumentParser(description="GEPA Skill Evolution for Aura")
    parser.add_argument("--skill", nargs="*", help="Specific skill IDs to evolve")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument(
        "--reflect-model", default="qwen3.5:397b-cloud",
        help="Model for reflection/mutation (default: qwen3.5:397b-cloud)"
    )
    parser.add_argument(
        "--eval-model", default="nemotron-3-super:cloud",
        help="Model for evaluation (default: nemotron-3-super:cloud)"
    )
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    overrides = {
        "max_iterations": args.iterations,
        "reflection_model": args.reflect_model,
        "eval_model": args.eval_model,
        "timeout_seconds": args.timeout,
    }

    result = run_evolution(
        skill_ids=args.skill,
        config_overrides=overrides,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
