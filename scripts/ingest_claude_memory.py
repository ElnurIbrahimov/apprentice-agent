"""Ingest Claude's accumulated memory (~/.aura/CLAUDE_MEMORY.md) into Aura's
knowledge graph and memory store.

Idempotent: Entity IDs are hashed from (name, type), MemoryStore uses INSERT OR IGNORE
on a stable UUID derived from the section title. Safe to re-run after editing the
source markdown.

Usage:
    python scripts/ingest_claude_memory.py
"""
from __future__ import annotations

import hashlib
import logging
import sys
import uuid
from pathlib import Path

AURA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AURA_ROOT))

from aura_knowledge_graph.graph_database import AURAKnowledgeGraph, Entity, Relationship
from aura_knowledge_graph.schema import EntityType
from aura.memory.store import MemoryRecord, get_memory_store

CLAUDE_MEMORY_FILE = Path.home() / ".aura" / "CLAUDE_MEMORY.md"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest_claude_memory")


# ── Structured facts about Elnur ──────────────────────────────────────────

PERSON = {
    "name": "Elnur",
    "description": "Ideas person and researcher. Does not code — AI agents do all coding. Based in Azerbaijan. Owns the Aura project.",
    "properties": {
        "email": "elnuribrahimov83@gmail.com",
        "location": "Azerbaijan",
        "platform": "Windows 11 + bash",
        "hardware": "RTX 4060 (8GB VRAM)",
    },
}

PROJECTS = [
    ("BroadMind", "Custom AI architecture, v0.74-v0.80. Targets ARC-AGI-2. Three-wisdom design (WisdomTable + CausalWisdom + MetaWisdom).", 0.9),
    ("Causeway", "Causal inference/reasoning in language models. GPT-2 and Mistral experiments.", 0.8),
    ("FluxMind", "MetaLearning research framework.", 0.8),
    ("MoR", "Mixture of Reasoning architecture experiments.", 0.7),
    ("ARC-AGI-2 Experiment", "7B model targeting 95% on ARC-AGI-2 through architectural innovation over scale. Convergence of BroadMind, Causeway, FluxMind, MoR.", 0.95),
    ("Aura", "Personal dev agent. Private, not commercial. Deployed at 89.167.107.134 via systemd. Telegram bot @Aura828Bot. This project.", 1.0),
    ("RentEase", "Next.js 14 property management SaaS for US small landlords. Supabase + Stripe BYOK + Twilio + Resend.", 0.7),
    ("Nəbz", "Clinic management SaaS (pulse in Azerbaijani), nebz.az. Next.js 16 + Supabase + next-intl. Epoint.az payments.", 0.7),
    ("prompt-shield", "npm package for prompt injection detection. 27 source files, 0% false positives.", 0.6),
    ("AvtoSes.az", "Automotive platform development plan.", 0.5),
]

TECHNOLOGIES = [
    ("Python", "Primary language for AI/ML work and Aura."),
    ("TypeScript", "For web frontends."),
    ("Next.js", "App Router framework for RentEase and Nəbz."),
    ("React", "Frontend library."),
    ("Supabase", "Database + auth for RentEase and Nəbz."),
    ("FastAPI", "Backend framework for Aura API."),
    ("PyTorch", "ML framework for architecture research."),
    ("Ollama", "Local and cloud LLM serving via Ollama Pro."),
    ("Kuzu", "Embedded graph database used for this knowledge graph."),
]

# Hard rules — stored as Concepts with high importance
HARD_RULES = [
    ("Never say ship", "Tied to past incidents of pushing broken work. Use publish/release/deploy instead.", 1.0),
    ("SOTA means SOTA", "Quality directives are literal bars, not politeness. No fast/cheap defaults.", 1.0),
    ("Never remove API keys during audits", "Preserve .env values. Rotate only when explicitly told.", 1.0),
    ("Not a scientist, not here to publish", "Never frame ideas as publishable or paper-worthy. Elnur builds for curiosity.", 1.0),
    ("Only Lemon Squeezy and Payoneer in Azerbaijan", "Stripe, PayPal, Polar, Paddle, Gumroad all BLOCKED for Elnur as seller.", 1.0),
    ("Never pitch attack-authoring plans", "Filter kills the session. Do not plan work that requires Elnur to author attack vectors.", 1.0),
    ("Never run blanket Node kill", "taskkill //F //IM node.exe and pkill -f node kill all CLI sessions. Has caused data loss.", 1.0),
    ("Never default to transformers in novel architectures", "BroadMind, Causeway, FluxMind, MoR, ARC-AGI-2 are NOT standard transformers. Build what Elnur designed.", 1.0),
    ("Direct commands mean execute, not ask", "commit and push / deploy / run it = just do it with sensible defaults.", 0.8),
]


def ingest_kg() -> dict:
    """Add structured entities and relationships to the KG."""
    kg = AURAKnowledgeGraph()

    # Person
    person = Entity(
        name=PERSON["name"],
        entity_type=EntityType.PERSON,
        description=PERSON["description"],
        properties=PERSON["properties"],
        importance=1.0,
    )
    person_id = kg.add_entity(person)
    log.info(f"Person: {person.name} ({person_id})")

    # Projects
    project_ids = {}
    for name, desc, importance in PROJECTS:
        e = Entity(name=name, entity_type=EntityType.PROJECT, description=desc, importance=importance)
        project_ids[name] = kg.add_entity(e)
        kg.add_relationship(Relationship(
            source_id=person_id, target_id=project_ids[name],
            relationship_type="WORKS_ON", weight=importance,
            evidence="CLAUDE_MEMORY.md",
        ))
    log.info(f"Projects: {len(project_ids)} added + WORKS_ON edges")

    # Technologies
    tech_ids = {}
    for name, desc in TECHNOLOGIES:
        e = Entity(name=name, entity_type=EntityType.TECHNOLOGY, description=desc, importance=0.5)
        tech_ids[name] = kg.add_entity(e)
    log.info(f"Technologies: {len(tech_ids)} added")

    # Project USES Technology edges
    tech_map = {
        "BroadMind": ["Python", "PyTorch"],
        "Causeway": ["Python", "PyTorch"],
        "FluxMind": ["Python", "PyTorch"],
        "MoR": ["Python", "PyTorch"],
        "ARC-AGI-2 Experiment": ["Python", "PyTorch"],
        "Aura": ["Python", "FastAPI", "Ollama", "Kuzu"],
        "RentEase": ["Next.js", "React", "TypeScript", "Supabase"],
        "Nəbz": ["Next.js", "React", "TypeScript", "Supabase"],
        "prompt-shield": ["TypeScript"],
    }
    for proj, techs in tech_map.items():
        for tech in techs:
            if proj in project_ids and tech in tech_ids:
                kg.add_relationship(Relationship(
                    source_id=project_ids[proj], target_id=tech_ids[tech],
                    relationship_type="USES", weight=0.8,
                    evidence="CLAUDE_MEMORY.md",
                ))

    # Hard rules as Concepts that Elnur HAS_SKILL about (using HAS_SKILL as closest allowed rel)
    rule_ids = {}
    for name, desc, importance in HARD_RULES:
        e = Entity(name=name, entity_type=EntityType.CONCEPT, description=desc, importance=importance)
        rule_ids[name] = kg.add_entity(e)
        # Hard rules RELATES_TO Elnur (generic relationship allowed)
        kg.add_relationship(Relationship(
            source_id=rule_ids[name], target_id=person_id,
            relationship_type="RELATES_TO", weight=importance,
            evidence="HARD RULE from CLAUDE_MEMORY.md",
        ))
    log.info(f"Hard rules: {len(rule_ids)} added as Concepts")

    return kg.get_statistics()


def ingest_memory_store() -> int:
    """Insert each section of CLAUDE_MEMORY.md as a high-importance memory record."""
    if not CLAUDE_MEMORY_FILE.exists():
        log.warning(f"Not found: {CLAUDE_MEMORY_FILE}")
        return 0

    text = CLAUDE_MEMORY_FILE.read_text(encoding="utf-8")
    store = get_memory_store()

    # Split on top-level markdown headers (## ...)
    sections: list[tuple[str, str]] = []
    current_title, current_body = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_title:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_title:
        sections.append((current_title, "\n".join(current_body).strip()))

    inserted = 0
    for title, body in sections:
        if not body:
            continue
        # Stable UUID from title → idempotent re-runs
        mid = uuid.UUID(hashlib.md5(f"claude_memory::{title}".encode()).hexdigest()).hex
        record = MemoryRecord(
            id=mid,
            content=f"# {title}\n\n{body}",
            title=f"Claude Memory — {title}",
            source="claude_memory_import",
            memory_type="semantic",
            importance=0.9,
            tags="elnur,profile,claude_memory,imported",
            category="user_profile",
            user_id="default_user",
            lifecycle_state="stable",
        )
        store.insert(record)
        inserted += 1

    log.info(f"MemoryStore: inserted {inserted} sections")
    return inserted


if __name__ == "__main__":
    log.info(f"Source: {CLAUDE_MEMORY_FILE}")
    if not CLAUDE_MEMORY_FILE.exists():
        log.error("CLAUDE_MEMORY.md not found. Run Claude's wiring step first.")
        sys.exit(1)

    kg_stats = ingest_kg()
    mem_count = ingest_memory_store()

    log.info("=" * 50)
    log.info(f"KG entities: {kg_stats.get('total_entities', '?')}")
    log.info(f"KG relationships: {kg_stats.get('total_relationships', '?')}")
    log.info(f"Memory records inserted: {mem_count}")
    log.info("Done.")
