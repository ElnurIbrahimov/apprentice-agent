# Persistent World Model for AURA: Research & Architecture Design

## Comprehensive Research Document

**Date**: February 2026
**Scope**: Persistent world model architecture for proactive situational awareness in AURA
**Codebase**: `C:\Users\asus\apprentice-agent`

---

## Executive Summary

AURA currently possesses powerful but **reactive** cognitive subsystems: episodic memory (Qdrant), a bi-temporal knowledge graph (NetworkX), unified memory queries, Theory of Mind user modeling, active inference, and pattern recognition (157 learned patterns). What it lacks is a **persistent, structured world model** -- a continuously maintained representation of "here is everything I believe about the user's situation right now" that drives **proactive** insight generation.

This document designs a **WorldModel** and **ProactiveAwarenessEngine** that sit atop AURA's existing systems and transform scattered memory fragments into a coherent, always-current situational picture. The world model is not a new memory system -- it is a **synthesizer and reasoner** over all existing ones, inspired by Endsley's Situation Awareness framework (Perception, Comprehension, Projection) and the Belief-Desire-Intention architecture.

---

## 1. Academic Foundations

### 1.1 World Models in AI Agents

**Foundational Work:**

- **"World Models" -- Ha & Schmidhuber (2018)**
  Paper: https://arxiv.org/abs/1803.10122
  The seminal work introducing learned world models for RL agents. A VAE compresses visual observations into latent codes, an RNN predicts future codes, and a tiny controller policy operates entirely in the latent space. Key insight: agents can be trained entirely inside their own "dream" and transfer back to real environments. For AURA, the architectural pattern of a compressed state representation + prediction module + policy is directly applicable.

- **"Understanding World or Predicting Future? A Comprehensive Survey of World Models" -- ACM Computing Surveys (2025)**
  Paper: https://dl.acm.org/doi/10.1145/3746449 | GitHub: https://github.com/tsinghua-fib-lab/World-Model
  The most comprehensive survey, covering world models across video generation, embodied AI, and autonomous driving. Establishes taxonomy: world models either build understanding of environment structure or predict future trajectories.

- **"Reinforcement World Model Learning for LLM-based Agents" (Feb 2026)**
  Paper: https://arxiv.org/abs/2602.05842
  Proposes RWML: self-supervised learning of action-conditioned world models for text-state LLM agents using sim-to-real gap rewards. Outperforms direct task-success RL by 6.9 points on ALFWorld. Key relevance: demonstrates LLMs can maintain and update world state representations through text.

- **WorldLLM: Curiosity-Driven Theory-Making (2025)**
  Paper: https://arxiv.org/abs/2506.06725
  Uses Bayesian inference to generate natural language "theories" (hypotheses about transition dynamics) that improve LLM world model predictions without fine-tuning. Key insight for AURA: world model updates can be expressed as natural language belief revisions.

- **"Web Agents with World Models" (Oct 2024)**
  Paper: https://arxiv.org/abs/2410.13232
  Demonstrates world-model-augmented agents that simulate action outcomes before committing. Achieves SOTA on Mind2Web. Key finding: current LLMs lack built-in world models and need explicit state tracking.

- **StateAct: Self-prompting and State-tracking (Oct 2024)**
  Paper: https://arxiv.org/abs/2410.02810
  Introduces "chain-of-states" -- explicit state tracking alongside chain-of-thought. Outperforms ReAct by 10-30% across benchmarks. Key design lesson: agents that explicitly maintain state representations outperform those relying on implicit context.

- **LeCun's JEPA / V-JEPA 2 (2024-2025)**
  Paper: https://arxiv.org/abs/2301.08243 | V-JEPA: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
  Joint Embedding Predictive Architecture predicts abstract representations (not pixels). V-JEPA 2 proves effective as a world model for robotics planning. Architectural lesson: predict in representation space, not observation space.

### 1.2 Belief-Desire-Intention (BDI) Architectures

- **"The Belief-Desire-Intention Ontology" (Nov 2025)**
  Paper: https://arxiv.org/pdf/2511.17162
  Formal BDI Ontology as a modular Ontology Design Pattern capturing agent cognitive architecture through beliefs, desires, intentions, and their dynamic interrelations. Directly applicable as the schema foundation for AURA's world model.

- **"Integrating Machine Learning into BDI Agents" (Oct 2025)**
  Paper: https://arxiv.org/pdf/2510.20641
  PRISMA-based review of 90 studies (2018-2025) on ML+BDI integration. Key finding: structured BDI reasoning augmented with ML adaptability produces more robust agents than either approach alone.

- **"BDI Agents in Natural Language Environments" -- AAMAS 2024**
  Paper: https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p880.pdf
  Demonstrates BDI agents operating in natural language task environments. Key relevance: beliefs can be maintained as natural language propositions, not just formal logic.

- **"Integrating BDI agents with LLMs for Reliable Human-Robot Interaction" (2024)**
  Paper: https://www.sciencedirect.com/science/article/pii/S0952197624019304
  Combines BDI's verifiable decision-making with LLM language understanding. Safety properties maintained through BDI structure while LLM handles NL interface.

### 1.3 Situation Awareness

- **Endsley's SA Model (1995, foundational)**
  Paper: https://www.researchgate.net/publication/210198492_Endsley_MR_Toward_a_Theory_of_Situation_Awareness_in_Dynamic_Systems_Human_Factors_Journal_371_32-64
  The three-level model: **Level 1 (Perception)** of elements, **Level 2 (Comprehension)** through pattern recognition and interpretation, **Level 3 (Projection)** of future states. This is the exact architecture AURA's world model needs: perceive state changes from conversations, comprehend their meaning for user goals, and project implications.

- **"Human-AI Teaming: State-of-the-Art and Research Needs" -- National Academies (2021)**
  Report: https://nap.nationalacademies.org/read/26355/chapter/6
  Chapter 4 on Situation Awareness in Human-AI teams. Establishes that shared SA between human and AI requires the AI to maintain an explicit model of the human's situation.

### 1.4 Proactive AI Assistants

- **"Proactive Conversational AI: A Comprehensive Survey" -- ACM TOIS (2025)**
  Paper: https://dl.acm.org/doi/10.1145/3715097
  Systematic survey across open-domain, task-oriented, and information-seeking dialogues. Taxonomy of proactive behaviors: topic suggestion, clarification, goal prediction, and initiative-taking.

- **"Proactive Conversational Agents with Inner Thoughts" -- CHI 2025**
  Paper: https://arxiv.org/abs/2501.00383 | ACM: https://dl.acm.org/doi/10.1145/3706598.3713760
  Proposes that proactive agents maintain parallel "inner thoughts" (covert thought trains) that build motivation to initiate. Uses an intrinsic motivation threshold (imThreshold) to decide when to speak. AURA already implements this pattern via `inner_thoughts_engine.py` -- the world model provides the *content* for these thoughts.

- **"Need Help? Designing Proactive AI Assistants for Programming" -- CHI 2025**
  Paper: https://dl.acm.org/doi/10.1145/3706598.3714002
  Field study showing proactive AI must understand programmer context to time interventions. Key finding: proactivity without context awareness is perceived as disruptive.

- **"Developer Interaction Patterns with Proactive AI: A Five-Day Field Study" (Jan 2026)**
  Paper: https://arxiv.org/html/2601.10253
  Finds the fundamental challenge is determining *when* to intervene without disrupting flow. Directly motivates AURA's WorkflowDetector integration.

- **ProActLLM**
  Website: https://proactllm.github.io/
  Explores the shift from reactive Q&A to proactive information-seeking assistants. Investigates how AI systems can anticipate needs before explicit expression.

### 1.5 Agent Memory & Living Knowledge Bases

- **"Memory in the Age of AI Agents: A Survey" (Dec 2025, updated Jan 2026)**
  Paper: https://arxiv.org/abs/2512.13564 | GitHub: https://github.com/Shichun-Liu/Agent-Memory-Paper-List
  Comprehensive taxonomy: token-level, parametric, and latent memory forms; factual, experiential, and working memory functions; formation, evolution, and retrieval dynamics. Establishes that memory is the "cornerstone" of agent capability.

- **"Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (Jan 2025)**
  Paper: https://arxiv.org/abs/2501.13956 | GitHub: https://github.com/getzep/graphiti
  Bi-temporal knowledge graph (Graphiti) that tracks event time and ingestion time with explicit validity intervals on every edge. Outperforms MemGPT on DMR benchmark (94.8% vs 93.4%), 18.5% improvement on LongMemEval, 90% latency reduction. AURA's KG already implements bi-temporal edges (Phase 4A) -- this validates the approach.

- **Letta/MemGPT: OS-style Agent Memory (2024-2025)**
  Docs: https://docs.letta.com/concepts/memgpt/ | GitHub: https://github.com/letta-ai/letta
  Treats LLM context window as RAM and external stores as disk. Agent manages its own memory hierarchy. Letta V1 (2025) adds Conversations API for shared memory across parallel experiences. Key pattern: the world model should be part of "core memory" (always in context), not archival memory.

- **Graphiti / Self-Evolving Knowledge Graphs (2025)**
  GitHub: https://github.com/getzep/graphiti
  Open-source temporal KG engine for real-time knowledge graph construction from conversational data. Dynamically synthesizes both unstructured and structured data while maintaining temporal relationships.

### 1.6 Active Inference & Free Energy

- **"Active Inference: The Free Energy Principle in Mind, Brain, and Behavior" -- Parr, Pezzulo, Friston (MIT Press, 2022)**
  Book: https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind
  The definitive text on Active Inference. Agents minimize surprise by either updating beliefs (perception) or acting to change the world (action). AURA already uses this via `active_inference.py` -- the world model provides the generative model that Active Inference operates over.

---

## 2. Concrete Architecture for AURA

### 2.1 Design Principles

1. **Synthesizer, not duplicator**: The world model does not store raw memories. It synthesizes structured beliefs from existing systems (episodic memory, KG, Pattern Prophet, Theory of Mind, conversation history).

2. **Endsley SA mapping**: Level 1 (Perception) = extract state changes from each conversation. Level 2 (Comprehension) = update structured world state. Level 3 (Projection) = infer implications, generate proactive insights.

3. **BDI alignment**: The world model explicitly tracks Beliefs (what AURA thinks is true about user's world), Desires (user's stated and inferred goals), and Intentions (user's current action plans). These map to user projects, goals, and active tasks.

4. **Letta-style core memory**: The world model summary lives in the LLM's system prompt (core memory), not behind a retrieval barrier. It is always available for reasoning.

5. **Contradiction-first updates**: New information is checked against existing beliefs before being accepted. Contradictions are flagged, timestamped, and resolved via LLM reasoning or user confirmation.

### 2.2 Where It Fits in AURA's Architecture

```
                    ┌─────────────────────────────┐
                    │      GLOBAL WORKSPACE        │
                    │   (Conscious Processing)     │
                    └──────────┬──────────────────┘
                               │ Broadcast
          ┌────────────────────┼────────────────────┐
          │                    │                     │
    ┌─────▼─────┐      ┌──────▼──────┐      ┌──────▼──────┐
    │  EMOTIONAL │      │   MEMORY    │      │  REASONING  │
    │  SYSTEM    │      │   SYSTEM    │      │  SYSTEM     │
    │  (ALMA)    │      │ (Unified)   │      │ (Brain)     │
    └─────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                    │                     │
          └────────────┬───────┴─────────────────────┘
                       │
               ┌───────▼────────┐
               │  WORLD MODEL   │  <── NEW: Persistent structured state
               │                │
               │ Projects       │  Synthesized from all memory systems
               │ Goals          │  Updated after every conversation
               │ Environment    │  Drives proactive inference
               │ Relationships  │  Contradiction detection
               │ Beliefs        │  Always in LLM context
               └───────┬────────┘
                       │
               ┌───────▼────────┐
               │   PROACTIVE    │
               │   AWARENESS    │  <── NEW: Unprompted insight engine
               │                │
               │ Staleness Det. │  Uses world model to generate
               │ Goal Inference │  proactive suggestions
               │ Pattern Alerts │  Feeds Gateway Daemon
               │ Priority Shift │
               └────────────────┘
```

### 2.3 Integration Points with Existing Systems

| Existing System | Integration | Direction |
|----------------|-------------|-----------|
| `unified_memory.py` | World model queries unified memory during updates | WorldModel -> UnifiedMemory |
| `knowledge_graph.py` | Entity/relationship extraction feeds world model; world model queries KG for context | Bidirectional |
| `theory_of_mind.py` | User emotional state and communication style feed world model's user profile | ToM -> WorldModel |
| `pattern_prophet.py` | Recognized patterns feed world model's pattern store; world model provides context for pattern matching | Bidirectional |
| `active_inference.py` | World model state feeds Active Inference's observation model; AI decisions update world model | Bidirectional |
| `inner_thoughts_engine.py` | World model provides content for inner thoughts; inner thoughts may generate world model updates | Bidirectional |
| `intrinsic_motivation.py` | World model gaps feed curiosity drive; coherence drive triggers contradiction resolution | WorldModel -> IntrinsicMotivation |
| `gateway_daemon.py` | Proactive awareness engine feeds suggestions to Gateway Daemon for delivery | Awareness -> Daemon |
| `brain.py` | World model summary injected into system prompt for every LLM call | WorldModel -> Brain |
| `idle_presence.py` | Idle time triggers world model maintenance (staleness checks, projection updates) | IdlePresence -> WorldModel |
| NeuroDream | Sleep consolidation can update/refine world model beliefs | NeuroDream -> WorldModel |

---

## 3. Data Schemas

### 3.1 JSON World State Schema

```json
{
  "version": "1.0.0",
  "last_updated": "2026-02-10T14:30:00Z",
  "last_conversation_id": "conv_abc123",

  "user_profile": {
    "name": "string",
    "known_since": "2026-01-15T00:00:00Z",
    "communication_style": {
      "verbosity": 0.6,
      "formality": 0.3,
      "technical_depth": 0.8,
      "preferred_response_length": "detailed"
    },
    "time_zone": "Asia/Baku",
    "typical_active_hours": {"start": "09:00", "end": "02:00"},
    "expertise_domains": [
      {"domain": "python", "level": 0.85, "confidence": 0.9},
      {"domain": "AI_agents", "level": 0.75, "confidence": 0.8}
    ]
  },

  "projects": [
    {
      "id": "proj_001",
      "name": "AURA Consciousness System",
      "status": "active",
      "description": "Building consciousness-like architecture for AI agent",
      "created_at": "2026-01-20T00:00:00Z",
      "last_mentioned": "2026-02-10T14:00:00Z",
      "last_activity": "2026-02-10T14:00:00Z",
      "mention_count": 47,
      "blockers": [
        {
          "description": "VRAM limitations on RTX 4060 for concurrent models",
          "severity": "medium",
          "identified_at": "2026-02-01T00:00:00Z",
          "status": "ongoing"
        }
      ],
      "milestones": [
        {
          "name": "Phase 6 Complete",
          "status": "completed",
          "completed_at": "2026-02-09T00:00:00Z"
        },
        {
          "name": "Persistent World Model",
          "status": "in_progress",
          "target_date": null
        }
      ],
      "technologies": ["python", "ollama", "qdrant", "networkx", "chromadb"],
      "related_project_ids": [],
      "priority": 0.95,
      "health": "green"
    }
  ],

  "goals": {
    "short_term": [
      {
        "id": "goal_st_001",
        "description": "Implement persistent world model for AURA",
        "created_at": "2026-02-10T00:00:00Z",
        "target_date": null,
        "progress": 0.1,
        "related_project_ids": ["proj_001"],
        "status": "active"
      }
    ],
    "medium_term": [
      {
        "id": "goal_mt_001",
        "description": "Make AURA genuinely proactive with situational awareness",
        "created_at": "2026-02-06T00:00:00Z",
        "progress": 0.7,
        "related_project_ids": ["proj_001"],
        "status": "active"
      }
    ],
    "long_term": [
      {
        "id": "goal_lt_001",
        "description": "Build a commercially viable AI assistant with consciousness-like properties",
        "created_at": "2026-01-15T00:00:00Z",
        "progress": 0.4,
        "status": "active"
      }
    ]
  },

  "environment": {
    "hardware": {
      "gpu": "RTX 4060 8GB VRAM",
      "constraints": ["8GB VRAM limit", "concurrent model loading"]
    },
    "tools": {
      "primary_ide": null,
      "llm_provider": "Ollama (local + cloud)",
      "models_in_use": ["mistral:7b", "llama3:8b", "qwen2.5-coder:7b", "llava"],
      "cloud_models": ["deepseek-v3.1:671b-cloud", "devstral-2:123b-cloud"],
      "databases": ["ChromaDB", "Qdrant", "SQLite", "NetworkX"]
    },
    "habits": {
      "late_night_coding": {"frequency": "often", "last_observed": null},
      "research_then_build": {"frequency": "consistent", "description": "Researches papers thoroughly before implementing"}
    },
    "schedule_patterns": []
  },

  "relationships": [
    {
      "id": "rel_001",
      "name": "string",
      "role": "user/creator",
      "relationship_type": "primary_user",
      "last_mentioned": "2026-02-10T00:00:00Z",
      "mention_count": 100,
      "context_notes": [],
      "sentiment": "positive"
    }
  ],

  "active_beliefs": [
    {
      "id": "belief_001",
      "statement": "User is building AURA as a genuine consciousness-like AI, not just a chatbot",
      "confidence": 0.95,
      "evidence": ["roadmap document", "research sessions", "implementation patterns"],
      "first_formed": "2026-01-20T00:00:00Z",
      "last_reinforced": "2026-02-10T00:00:00Z",
      "contradicted_by": [],
      "category": "user_intent"
    },
    {
      "id": "belief_002",
      "statement": "VRAM is the primary hardware constraint for AURA development",
      "confidence": 0.9,
      "evidence": ["config.py GPU_VRAM_GB=8", "model selection code", "VRAM-aware loading"],
      "first_formed": "2026-01-25T00:00:00Z",
      "last_reinforced": "2026-02-09T00:00:00Z",
      "contradicted_by": [],
      "category": "technical_constraint"
    }
  ],

  "contradictions": [
    {
      "id": "contra_001",
      "belief_a": "belief_id_a",
      "belief_b": "belief_id_b",
      "description": "Old info says X, new conversation says Y",
      "detected_at": "2026-02-10T00:00:00Z",
      "resolution": null,
      "resolution_strategy": "ask_user"
    }
  ],

  "staleness_tracker": {
    "proj_001": {
      "last_activity": "2026-02-10T14:00:00Z",
      "staleness_days": 0,
      "alert_threshold_days": 7,
      "alert_sent": false
    }
  },

  "meta": {
    "total_conversations_processed": 0,
    "total_updates_applied": 0,
    "total_contradictions_detected": 0,
    "total_proactive_insights_generated": 0
  }
}
```

### 3.2 SQLite Schema (`data/world_model.db`)

```sql
-- ============================================================
-- WORLD MODEL PERSISTENT STORAGE
-- ============================================================

-- Projects the user is working on
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'active',  -- active, paused, completed, abandoned
    description TEXT,
    created_at TEXT NOT NULL,
    last_mentioned TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    priority REAL DEFAULT 0.5,     -- 0-1, computed from recency + frequency
    health TEXT DEFAULT 'green',   -- green, yellow, red
    technologies TEXT,             -- JSON array
    metadata TEXT                  -- JSON blob for extensibility
);

-- Project blockers
CREATE TABLE project_blockers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id),
    description TEXT NOT NULL,
    severity TEXT DEFAULT 'medium',  -- low, medium, high, critical
    identified_at TEXT NOT NULL,
    resolved_at TEXT,
    status TEXT DEFAULT 'ongoing',   -- ongoing, resolved, wontfix
    resolution TEXT
);

-- Project milestones
CREATE TABLE project_milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id),
    name TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed
    target_date TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
);

-- User goals at different time horizons
CREATE TABLE goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    horizon TEXT NOT NULL,          -- short_term, medium_term, long_term
    created_at TEXT NOT NULL,
    target_date TEXT,
    progress REAL DEFAULT 0.0,     -- 0-1
    status TEXT DEFAULT 'active',  -- active, achieved, abandoned, blocked
    related_project_ids TEXT,      -- JSON array
    evidence TEXT                  -- JSON array of supporting observations
);

-- Goal-blocker relationships (inferred)
CREATE TABLE goal_blockers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL REFERENCES goals(id),
    blocker_description TEXT NOT NULL,
    blocker_source TEXT,           -- project_blocker_id, belief_id, or free text
    inferred_at TEXT NOT NULL,
    confidence REAL DEFAULT 0.5
);

-- People mentioned in conversations
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT,                     -- colleague, friend, manager, client, etc.
    relationship_type TEXT,        -- primary_user, mentioned_person, collaborator
    first_mentioned TEXT NOT NULL,
    last_mentioned TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    context_notes TEXT,            -- JSON array of context strings
    sentiment TEXT DEFAULT 'neutral'  -- positive, neutral, negative, mixed
);

-- Structured beliefs about the user's world
CREATE TABLE beliefs (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    confidence REAL DEFAULT 0.7,   -- 0-1
    category TEXT NOT NULL,        -- user_intent, technical_constraint, preference,
                                   -- project_state, relationship, schedule, habit
    evidence TEXT NOT NULL,        -- JSON array of evidence strings
    first_formed TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    valid_from TEXT NOT NULL,      -- Bi-temporal: when belief became true
    valid_to TEXT,                 -- Bi-temporal: when belief stopped being true (NULL = current)
    superseded_by TEXT,            -- ID of belief that replaced this one
    source_conversation_ids TEXT   -- JSON array
);

-- Detected contradictions between beliefs
CREATE TABLE contradictions (
    id TEXT PRIMARY KEY,
    belief_a_id TEXT NOT NULL REFERENCES beliefs(id),
    belief_b_id TEXT NOT NULL REFERENCES beliefs(id),
    description TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    resolution TEXT,               -- resolved_a, resolved_b, merged, asked_user, unresolved
    resolution_details TEXT,
    resolved_at TEXT
);

-- Environment observations (tools, habits, schedule)
CREATE TABLE environment (
    key TEXT PRIMARY KEY,
    category TEXT NOT NULL,        -- hardware, tool, habit, schedule, preference
    value TEXT NOT NULL,           -- JSON value
    confidence REAL DEFAULT 0.8,
    first_observed TEXT NOT NULL,
    last_observed TEXT NOT NULL,
    observation_count INTEGER DEFAULT 1
);

-- State change log: every update to the world model
CREATE TABLE state_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    conversation_id TEXT,
    change_type TEXT NOT NULL,     -- project_update, goal_update, belief_formed,
                                   -- belief_revised, contradiction_detected,
                                   -- relationship_update, environment_update,
                                   -- blocker_added, blocker_resolved
    entity_type TEXT NOT NULL,     -- project, goal, belief, relationship, environment
    entity_id TEXT NOT NULL,
    old_value TEXT,                -- JSON
    new_value TEXT,                -- JSON
    reasoning TEXT                 -- Why this change was made
);

-- Proactive insights generated by the awareness engine
CREATE TABLE proactive_insights (
    id TEXT PRIMARY KEY,
    insight_type TEXT NOT NULL,    -- staleness_alert, blocker_inference, pattern_alert,
                                   -- priority_shift, goal_risk, relationship_gap,
                                   -- deadline_approaching, contradiction_alert
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    urgency REAL DEFAULT 0.5,     -- 0-1
    confidence REAL DEFAULT 0.5,
    generated_at TEXT NOT NULL,
    delivered_at TEXT,             -- When shown to user (NULL = not yet)
    dismissed_at TEXT,             -- When user dismissed
    acted_on_at TEXT,              -- When user engaged
    related_entity_type TEXT,
    related_entity_id TEXT,
    reasoning TEXT                 -- Chain of reasoning that produced this insight
);

-- Indexes for performance
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_last_activity ON projects(last_activity);
CREATE INDEX idx_beliefs_category ON beliefs(category);
CREATE INDEX idx_beliefs_valid ON beliefs(valid_to);  -- NULL = current beliefs
CREATE INDEX idx_state_changes_timestamp ON state_changes(timestamp);
CREATE INDEX idx_state_changes_entity ON state_changes(entity_type, entity_id);
CREATE INDEX idx_insights_type ON proactive_insights(insight_type);
CREATE INDEX idx_insights_urgency ON proactive_insights(urgency);
CREATE INDEX idx_insights_delivered ON proactive_insights(delivered_at);
```

---

## 4. Python Class Skeletons

### 4.1 WorldModel Class

```python
"""
Persistent World Model for AURA.

Maintains a continuously updated structured representation of the user's
projects, goals, environment, relationships, and active beliefs.

Implements Endsley's Situation Awareness framework:
  Level 1 (Perception): Extract state changes from conversations
  Level 2 (Comprehension): Update structured world state
  Level 3 (Projection): Infer implications, generate proactive insights

Integrates with:
  - UnifiedMemory: Queries all memory systems during updates
  - KnowledgeGraph: Bidirectional entity/relationship sync
  - TheoryOfMind: User state feeds world model user profile
  - PatternProphet: Pattern data enriches world model
  - ActiveInference: World state feeds observation model
  - InnerThoughts: World model provides content for thoughts
  - Brain: World model summary injected into system prompt
  - NeuroDream: Sleep consolidation can refine beliefs
  - GatewayDaemon: Proactive insights fed to daemon for delivery

Storage: SQLite (data/world_model.db) + JSON snapshot (data/world_state.json)
"""

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ProjectHealth(str, Enum):
    GREEN = "green"      # On track, recent activity
    YELLOW = "yellow"    # Slowing down or minor blockers
    RED = "red"          # Stale, major blockers, or at risk


class GoalHorizon(str, Enum):
    SHORT_TERM = "short_term"    # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"      # Months to years


class BeliefCategory(str, Enum):
    USER_INTENT = "user_intent"
    TECHNICAL_CONSTRAINT = "technical_constraint"
    PREFERENCE = "preference"
    PROJECT_STATE = "project_state"
    RELATIONSHIP = "relationship"
    SCHEDULE = "schedule"
    HABIT = "habit"
    ENVIRONMENT = "environment"


class ChangeType(str, Enum):
    PROJECT_UPDATE = "project_update"
    GOAL_UPDATE = "goal_update"
    BELIEF_FORMED = "belief_formed"
    BELIEF_REVISED = "belief_revised"
    CONTRADICTION_DETECTED = "contradiction_detected"
    RELATIONSHIP_UPDATE = "relationship_update"
    ENVIRONMENT_UPDATE = "environment_update"
    BLOCKER_ADDED = "blocker_added"
    BLOCKER_RESOLVED = "blocker_resolved"


@dataclass
class Project:
    """A user project tracked by the world model."""
    id: str
    name: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    description: str = ""
    created_at: str = ""
    last_mentioned: str = ""
    last_activity: str = ""
    mention_count: int = 1
    priority: float = 0.5
    health: ProjectHealth = ProjectHealth.GREEN
    technologies: List[str] = field(default_factory=list)
    blockers: List[Dict[str, Any]] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Goal:
    """A user goal at a specific time horizon."""
    id: str
    description: str
    horizon: GoalHorizon
    created_at: str = ""
    target_date: Optional[str] = None
    progress: float = 0.0
    status: str = "active"
    related_project_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class Belief:
    """A structured belief about the user's world."""
    id: str
    statement: str
    confidence: float = 0.7
    category: BeliefCategory = BeliefCategory.USER_INTENT
    evidence: List[str] = field(default_factory=list)
    first_formed: str = ""
    last_reinforced: str = ""
    valid_from: str = ""
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    source_conversation_ids: List[str] = field(default_factory=list)


@dataclass
class Contradiction:
    """A detected contradiction between beliefs."""
    id: str
    belief_a_id: str
    belief_b_id: str
    description: str
    detected_at: str
    resolution: Optional[str] = None
    resolution_details: Optional[str] = None
    resolved_at: Optional[str] = None


@dataclass
class Relationship:
    """A person mentioned in conversations."""
    id: str
    name: str
    role: str = ""
    relationship_type: str = "mentioned_person"
    first_mentioned: str = ""
    last_mentioned: str = ""
    mention_count: int = 1
    context_notes: List[str] = field(default_factory=list)
    sentiment: str = "neutral"


@dataclass
class StateChange:
    """A logged change to the world model."""
    timestamp: str
    conversation_id: Optional[str]
    change_type: ChangeType
    entity_type: str
    entity_id: str
    old_value: Optional[Dict] = None
    new_value: Optional[Dict] = None
    reasoning: str = ""


# ============================================================================
# WorldModel Core
# ============================================================================

class WorldModel:
    """
    Persistent World Model -- AURA's structured understanding of the user's world.

    Architecture:
    - SQLite for persistent storage with full audit trail
    - In-memory cache for fast access during conversations
    - JSON snapshot for LLM context injection
    - LLM-powered extraction pipeline for state updates

    Update Pipeline (after every conversation):
    1. Extract: LLM analyzes conversation for state-relevant information
    2. Diff: Compare extracted info against current world state
    3. Detect: Identify contradictions with existing beliefs
    4. Apply: Update state, log changes, resolve or flag contradictions
    5. Project: Run proactive awareness checks on updated state
    6. Summarize: Generate updated context summary for brain.py
    """

    # How often to run full staleness checks (seconds)
    STALENESS_CHECK_INTERVAL = 3600  # 1 hour

    # Default thresholds
    PROJECT_STALE_DAYS = 7        # Project goes yellow after 7 days silence
    PROJECT_CRITICAL_DAYS = 14    # Project goes red after 14 days silence
    BELIEF_DECAY_HALF_LIFE = 336  # Hours (2 weeks) for belief confidence decay
    GOAL_STALE_DAYS = 14          # Goal progress warning after 14 days

    def __init__(self, db_path: str = "data/world_model.db",
                 snapshot_path: str = "data/world_state.json"):
        self.db_path = Path(db_path)
        self.snapshot_path = Path(snapshot_path)
        self._lock = threading.RLock()

        # In-memory cache
        self._projects: Dict[str, Project] = {}
        self._goals: Dict[str, Goal] = {}
        self._beliefs: Dict[str, Belief] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._environment: Dict[str, Any] = {}
        self._contradictions: List[Contradiction] = []

        # Lazy LLM reference
        self._brain = None

        self._init_db()
        self._load_from_db()

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def _init_db(self):
        """Create SQLite tables if they don't exist."""
        pass

    def _load_from_db(self):
        """Load current state from SQLite into memory cache."""
        pass

    # ----------------------------------------------------------------
    # UPDATE PIPELINE -- Called after every conversation
    # ----------------------------------------------------------------

    def process_conversation(self, conversation_id: str,
                             messages: List[Dict[str, str]],
                             metadata: Optional[Dict] = None) -> List[StateChange]:
        """
        Main update pipeline. Called after every conversation turn.

        Implements Endsley SA Levels 1-2:
          Level 1: Extract state-relevant elements from conversation
          Level 2: Comprehend meaning, update world state

        Args:
            conversation_id: Unique ID of the conversation
            messages: List of {"role": "user"/"assistant", "content": "..."}
            metadata: Optional context (tool results, timestamps, etc.)

        Returns:
            List of StateChange objects describing what was updated
        """
        changes = []

        # Step 1: EXTRACT -- LLM analyzes conversation for state changes
        extraction = self._extract_state_changes(messages, metadata)

        # Step 2: DIFF -- Compare against current state
        diffs = self._compute_diffs(extraction)

        # Step 3: DETECT -- Check for contradictions
        contradictions = self._detect_contradictions(diffs)
        for c in contradictions:
            changes.append(self._log_contradiction(c))

        # Step 4: APPLY -- Update state
        for diff in diffs:
            change = self._apply_diff(diff, conversation_id)
            if change:
                changes.append(change)

        # Step 5: Update snapshot for LLM context
        self._update_snapshot()

        return changes

    def _extract_state_changes(self, messages: List[Dict[str, str]],
                                metadata: Optional[Dict] = None) -> Dict:
        """
        Use LLM to extract state-relevant information from conversation.

        Prompts the LLM with current world state summary + conversation,
        asks it to identify:
        - New or updated projects
        - Progress on goals
        - New blockers or resolved blockers
        - New people mentioned
        - Changed preferences or habits
        - Technical environment changes
        - Statements that contradict existing beliefs

        Returns structured extraction dict.
        """
        pass

    def _compute_diffs(self, extraction: Dict) -> List[Dict]:
        """
        Compare extracted information against current world state.

        Returns list of diffs: {type, entity, field, old_value, new_value, confidence}
        """
        pass

    def _detect_contradictions(self, diffs: List[Dict]) -> List[Contradiction]:
        """
        Check if any new information contradicts existing beliefs.

        Uses semantic similarity + LLM judgment to detect contradictions.
        Example: Old belief "User prefers React" + new statement "I've switched to Svelte"

        Returns list of detected contradictions.
        """
        pass

    def _apply_diff(self, diff: Dict, conversation_id: str) -> Optional[StateChange]:
        """
        Apply a single diff to the world state.

        Updates both in-memory cache and SQLite.
        Logs the change with full audit trail.
        For belief revisions: marks old belief with valid_to, creates new belief.
        """
        pass

    def _log_contradiction(self, contradiction: Contradiction) -> StateChange:
        """Log a detected contradiction and determine resolution strategy."""
        pass

    # ----------------------------------------------------------------
    # CONTRADICTION RESOLUTION
    # ----------------------------------------------------------------

    def resolve_contradiction(self, contradiction_id: str,
                              resolution: str,
                              details: str = "") -> bool:
        """
        Resolve a detected contradiction.

        Args:
            contradiction_id: ID of the contradiction
            resolution: One of 'resolved_a', 'resolved_b', 'merged', 'asked_user'
            details: Explanation of resolution

        Returns:
            True if successfully resolved
        """
        pass

    def auto_resolve_contradictions(self) -> List[str]:
        """
        Attempt to auto-resolve contradictions using LLM reasoning.

        Resolution strategies:
        - Temporal: newer information supersedes older (most common)
        - Specificity: more specific belief supersedes general
        - Confidence: higher-confidence belief wins if difference > 0.3
        - Ask user: if ambiguous, queue for user confirmation

        Returns list of resolved contradiction IDs.
        """
        pass

    # ----------------------------------------------------------------
    # QUERY INTERFACE
    # ----------------------------------------------------------------

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        pass

    def get_projects_by_status(self, status: ProjectStatus) -> List[Project]:
        """Get all projects with a given status."""
        pass

    def get_stale_projects(self, days: int = None) -> List[Tuple[Project, int]]:
        """Get projects with no activity for N days. Returns (project, days_stale)."""
        pass

    def get_active_goals(self, horizon: Optional[GoalHorizon] = None) -> List[Goal]:
        """Get active goals, optionally filtered by time horizon."""
        pass

    def get_current_beliefs(self, category: Optional[BeliefCategory] = None) -> List[Belief]:
        """Get all current (valid_to IS NULL) beliefs, optionally filtered."""
        pass

    def get_unresolved_contradictions(self) -> List[Contradiction]:
        """Get all unresolved contradictions."""
        pass

    def get_relationship(self, name: str) -> Optional[Relationship]:
        """Find a relationship by person name (fuzzy match)."""
        pass

    def get_recent_changes(self, limit: int = 20) -> List[StateChange]:
        """Get the most recent state changes for audit/display."""
        pass

    # ----------------------------------------------------------------
    # CONTEXT GENERATION -- For LLM system prompt injection
    # ----------------------------------------------------------------

    def get_context_summary(self, max_tokens: int = 500) -> str:
        """
        Generate a concise summary of current world state for LLM context.

        This is injected into every LLM call via brain.py as part of the
        system prompt. It must be concise but information-dense.

        Format:
        ```
        [WORLD STATE]
        Active projects: AURA (green, active today), WebApp (yellow, 5d stale)
        Current goals: Implement world model (short-term, 10%), Ship MVP (medium, 60%)
        Blockers: VRAM limits (medium), API rate limits (low)
        Key beliefs: User is building consciousness AI, prefers detailed responses
        Recent people: Alice (collaborator, last: 2d ago)
        Contradictions: 1 unresolved (project timeline)
        ```

        Returns formatted summary string.
        """
        pass

    def get_full_state_json(self) -> Dict:
        """Get complete world state as JSON dict (for snapshot/export)."""
        pass

    # ----------------------------------------------------------------
    # SNAPSHOT MANAGEMENT
    # ----------------------------------------------------------------

    def _update_snapshot(self):
        """Write current state to JSON snapshot file."""
        pass

    def _load_snapshot(self) -> Optional[Dict]:
        """Load state from JSON snapshot (for quick startup)."""
        pass

    # ----------------------------------------------------------------
    # MAINTENANCE
    # ----------------------------------------------------------------

    def decay_beliefs(self):
        """
        Apply temporal decay to belief confidence.

        Beliefs not reinforced decay using Ebbinghaus curve:
          confidence *= e^(-decay_rate * hours_since_reinforced)
          where decay_rate = ln(2) / BELIEF_DECAY_HALF_LIFE

        Beliefs below 0.2 confidence are marked for review.
        """
        pass

    def update_project_health(self):
        """
        Recompute project health status based on activity recency and blockers.

        Green: active in last 3 days, no critical blockers
        Yellow: active in last 7 days, or has medium blockers
        Red: stale > 7 days, or has critical blockers
        """
        pass

    def compute_project_priority(self, project: Project) -> float:
        """
        Compute project priority from multiple signals.

        Factors: mention_frequency, recency, blocker_severity,
                 goal_alignment, user_emphasis_signals
        """
        pass

    def run_maintenance(self):
        """
        Run all maintenance tasks. Called during idle time or on schedule.

        1. Decay belief confidence
        2. Update project health
        3. Recompute priorities
        4. Clean up old state changes (keep last 1000)
        5. Auto-resolve low-ambiguity contradictions
        """
        pass
```

### 4.2 ProactiveAwareness Class

```python
"""
Proactive Awareness Engine for AURA.

Implements Endsley SA Level 3 (Projection):
Analyzes the world model to generate unprompted insights
about things the user hasn't asked about yet.

Insight types:
- Staleness alerts: projects going silent
- Goal-blocker inference: connecting blockers to goals
- Pattern alerts: cross-conversation pattern recognition
- Priority shift detection: when implicit priorities change
- Deadline awareness: approaching deadlines or milestones
- Relationship gaps: people not mentioned in a while
- Contradiction alerts: unresolved contradictions needing attention
- Stress correlation: connecting stressed topics to related areas

Integrates with:
- WorldModel: reads current state, writes insights
- GatewayDaemon: delivers insights at appropriate moments
- WorkflowDetector: gates delivery by user focus state
- ActiveInference: feeds insight urgency into belief updates
- InnerThoughts: high-urgency insights become inner thoughts
- IntrinsicMotivation: unresolved insights feed coherence/curiosity drives
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    STALENESS_ALERT = "staleness_alert"
    BLOCKER_INFERENCE = "blocker_inference"
    PATTERN_ALERT = "pattern_alert"
    PRIORITY_SHIFT = "priority_shift"
    DEADLINE_APPROACHING = "deadline_approaching"
    RELATIONSHIP_GAP = "relationship_gap"
    CONTRADICTION_ALERT = "contradiction_alert"
    STRESS_CORRELATION = "stress_correlation"
    GOAL_RISK = "goal_risk"
    OPPORTUNITY = "opportunity"


@dataclass
class ProactiveInsight:
    """An unprompted insight generated by the awareness engine."""
    id: str
    insight_type: InsightType
    title: str
    description: str
    urgency: float = 0.5          # 0-1, affects delivery priority
    confidence: float = 0.5       # 0-1, how sure we are
    generated_at: str = ""
    delivered_at: Optional[str] = None
    dismissed_at: Optional[str] = None
    acted_on_at: Optional[str] = None
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[str] = None
    reasoning: str = ""           # Chain of reasoning

    @property
    def delivery_score(self) -> float:
        """Combined score for delivery prioritization."""
        return self.urgency * 0.6 + self.confidence * 0.4


class ProactiveAwarenessEngine:
    """
    Generates proactive insights from the world model.

    Runs periodically (during idle time, after conversations, on schedule)
    to analyze the world state and generate insights the user hasn't
    asked for but may benefit from.

    Design: Each check_* method is an independent analysis pass.
    All insights are scored and only those above threshold are delivered.
    """

    # Minimum confidence to generate an insight
    MIN_CONFIDENCE = 0.4

    # Minimum urgency to deliver (below this, store but don't show)
    DELIVERY_URGENCY_THRESHOLD = 0.5

    # Cooldown between similar insights (hours)
    SIMILAR_INSIGHT_COOLDOWN = 24

    def __init__(self, world_model: 'WorldModel'):
        self.world_model = world_model
        self._pending_insights: List[ProactiveInsight] = []
        self._brain = None  # Lazy LLM reference

    # ----------------------------------------------------------------
    # MAIN ANALYSIS LOOP
    # ----------------------------------------------------------------

    def run_full_analysis(self) -> List[ProactiveInsight]:
        """
        Run all awareness checks against current world state.

        Called:
        - After every conversation (lightweight checks only)
        - During idle time (full analysis)
        - On schedule (hourly full analysis)

        Returns list of new insights generated.
        """
        insights = []

        insights.extend(self.check_staleness())
        insights.extend(self.check_goal_blockers())
        insights.extend(self.check_patterns())
        insights.extend(self.check_priority_shifts())
        insights.extend(self.check_deadlines())
        insights.extend(self.check_relationship_gaps())
        insights.extend(self.check_contradictions())
        insights.extend(self.check_stress_correlations())

        # Filter by confidence and dedup
        insights = self._filter_and_dedup(insights)

        # Store insights
        for insight in insights:
            self._store_insight(insight)

        return insights

    def run_quick_analysis(self) -> List[ProactiveInsight]:
        """
        Lightweight analysis after a conversation.
        Only runs fast checks (no LLM calls).
        """
        insights = []
        insights.extend(self.check_staleness())
        insights.extend(self.check_deadlines())
        insights.extend(self.check_contradictions())
        return self._filter_and_dedup(insights)

    # ----------------------------------------------------------------
    # STALENESS DETECTION
    # ----------------------------------------------------------------

    def check_staleness(self) -> List[ProactiveInsight]:
        """
        Detect projects, goals, and relationships going silent.

        Logic:
        - For each active project: compute days since last_activity
        - If > PROJECT_STALE_DAYS and no alert sent recently: generate insight
        - Scale urgency by how important the project is (priority * staleness)

        Example insight:
          "You haven't mentioned the WebApp project in 12 days.
           Last activity was about the authentication module.
           Is this project still active, or should we deprioritize it?"
        """
        pass

    # ----------------------------------------------------------------
    # GOAL-BLOCKER INFERENCE
    # ----------------------------------------------------------------

    def check_goal_blockers(self) -> List[ProactiveInsight]:
        """
        Infer connections between blockers and goals.

        Logic:
        - For each active goal: check related projects for blockers
        - For each blocker: estimate impact on goal progress
        - If a blocker is blocking a high-priority goal: generate insight

        Example insight:
          "Your VRAM limitation blocker on AURA might be slowing
           your medium-term goal of 'shipping proactive features'.
           Have you considered model quantization or cloud offloading?"

        Uses LLM to reason about blocker-goal connections when
        the connection isn't explicit in the data.
        """
        pass

    # ----------------------------------------------------------------
    # PATTERN RECOGNITION ALERTS
    # ----------------------------------------------------------------

    def check_patterns(self) -> List[ProactiveInsight]:
        """
        Surface cross-conversation patterns from PatternProphet.

        Logic:
        - Query PatternProphet for high-confidence patterns
        - Check if any pattern predictions are currently relevant
        - Generate insights for actionable patterns

        Example insight:
          "I've noticed you often research a topic thoroughly
           before implementing. You've been researching world models
           for 2 sessions now -- are you ready to start building?"
        """
        pass

    # ----------------------------------------------------------------
    # PRIORITY SHIFT DETECTION
    # ----------------------------------------------------------------

    def check_priority_shifts(self) -> List[ProactiveInsight]:
        """
        Detect when user's implicit priorities are shifting.

        Logic:
        - Track mention_count trends over time for each project
        - If a previously high-priority project is declining while
          another is rising: generate insight
        - Compare stated goals against actual activity patterns

        Example insight:
          "Your focus seems to be shifting from the WebApp to AURA.
           You've mentioned AURA 15 times this week vs. WebApp 0 times.
           Should I update priority rankings?"
        """
        pass

    # ----------------------------------------------------------------
    # DEADLINE AWARENESS
    # ----------------------------------------------------------------

    def check_deadlines(self) -> List[ProactiveInsight]:
        """
        Alert on approaching deadlines and milestones.

        Logic:
        - For each goal/milestone with a target_date:
          compute days_remaining
        - If days_remaining < threshold and progress < expected: alert
        - Threshold varies by urgency: 30d for long-term, 7d for short-term

        Example insight:
          "Your milestone 'Ship MVP' has a target date of March 1st
           (19 days away) but progress is at 60%. At current velocity,
           you may need 25 more days. Consider reprioritizing."
        """
        pass

    # ----------------------------------------------------------------
    # RELATIONSHIP GAPS
    # ----------------------------------------------------------------

    def check_relationship_gaps(self) -> List[ProactiveInsight]:
        """
        Detect important people not mentioned recently.

        Logic:
        - For each relationship with role in (collaborator, client, manager):
          check last_mentioned
        - If gap > threshold and person was mentioned frequently before: alert

        Example insight:
          "You haven't mentioned Alice (collaborator on WebApp) in 3 weeks.
           Last context: she was working on the design system.
           Should I check if there are any pending items with her?"
        """
        pass

    # ----------------------------------------------------------------
    # CONTRADICTION ALERTS
    # ----------------------------------------------------------------

    def check_contradictions(self) -> List[ProactiveInsight]:
        """
        Surface unresolved contradictions that may need attention.

        Logic:
        - Get unresolved contradictions from world model
        - If age > 24 hours and not yet surfaced: generate alert
        - Higher urgency for contradictions affecting active projects

        Example insight:
          "I have conflicting information about the deployment target.
           Earlier you said 'deploy to Vercel' but last session you
           mentioned 'self-hosting on the home server'. Which is current?"
        """
        pass

    # ----------------------------------------------------------------
    # STRESS CORRELATION
    # ----------------------------------------------------------------

    def check_stress_correlations(self) -> List[ProactiveInsight]:
        """
        Connect user stress signals to related areas.

        Logic:
        - Query ALMA emotional state for stress/frustration signals
        - Query Theory of Mind for user emotional prediction
        - If stress detected: check world model for related projects/blockers
        - Generate supportive insight connecting the dots

        Example insight:
          "You seem stressed about the authentication module (mentioned
           3 frustration signals in last 2 conversations). This blocker
           also affects your shipping timeline. Want to brainstorm
           alternative approaches?"
        """
        pass

    # ----------------------------------------------------------------
    # FILTERING AND DELIVERY
    # ----------------------------------------------------------------

    def _filter_and_dedup(self, insights: List[ProactiveInsight]) -> List[ProactiveInsight]:
        """
        Filter insights by confidence/urgency and deduplicate.

        Rules:
        - Drop insights below MIN_CONFIDENCE
        - Deduplicate by (insight_type, related_entity_id) within cooldown window
        - Sort by delivery_score descending
        - Cap at 3 insights per analysis run (avoid overwhelming user)
        """
        pass

    def _store_insight(self, insight: ProactiveInsight):
        """Store insight in world model database."""
        pass

    def get_pending_insights(self, max_count: int = 3) -> List[ProactiveInsight]:
        """
        Get insights ready for delivery, sorted by urgency.

        Only returns insights that:
        - Have not been delivered
        - Are above delivery urgency threshold
        - Pass the WorkflowDetector interruption check
        """
        pass

    def record_insight_feedback(self, insight_id: str,
                                 feedback: str):
        """
        Record user response to an insight.

        Args:
            insight_id: The insight that was shown
            feedback: 'engaged', 'dismissed', 'deferred'

        Updates insight delivery statistics and feeds back into
        Active Inference outcome learning.
        """
        pass

    # ----------------------------------------------------------------
    # CONTEXT FOR OTHER SYSTEMS
    # ----------------------------------------------------------------

    def get_awareness_context(self) -> str:
        """
        Generate awareness context string for inner thoughts engine.

        Provides the inner thoughts engine with current concerns,
        pending insights, and world model gaps to think about.
        """
        pass

    def get_drive_signals(self) -> Dict[str, float]:
        """
        Generate signals for the intrinsic motivation system.

        Returns:
            {
                "curiosity": 0.7,   # From knowledge gaps in world model
                "coherence": 0.8,   # From unresolved contradictions
                "social": 0.3,      # From relationship gaps
                "competence": 0.5,  # From blocker patterns
            }
        """
        pass
```

### 4.3 Extraction Pipeline (LLM-powered)

```python
"""
State extraction pipeline for the World Model.

Uses LLM to analyze conversations and extract structured state changes.
Designed to run after every conversation turn.
"""

# Extraction prompt template
EXTRACTION_PROMPT = """You are the World Model Updater for AURA, an AI assistant.

Your job: analyze the conversation below and extract any changes to the user's world state.

## Current World State Summary:
{world_state_summary}

## Conversation to Analyze:
{conversation}

## Extract the following (JSON format):

{{
  "projects": [
    {{
      "name": "string",
      "action": "new|update|mention",
      "status_change": null | "active|paused|completed|abandoned",
      "new_blockers": ["string"],
      "resolved_blockers": ["string"],
      "progress_notes": "string",
      "technologies_mentioned": ["string"]
    }}
  ],
  "goals": [
    {{
      "description": "string",
      "action": "new|update|achieved",
      "horizon": "short_term|medium_term|long_term",
      "progress_delta": 0.0,
      "evidence": "string"
    }}
  ],
  "beliefs": [
    {{
      "statement": "string",
      "category": "user_intent|technical_constraint|preference|habit|schedule",
      "confidence": 0.0-1.0,
      "contradicts_existing": null | "description of contradiction"
    }}
  ],
  "people_mentioned": [
    {{
      "name": "string",
      "role": "string",
      "context": "string",
      "sentiment": "positive|neutral|negative"
    }}
  ],
  "environment_changes": [
    {{
      "key": "string",
      "category": "hardware|tool|habit|schedule|preference",
      "value": "string"
    }}
  ],
  "emotional_signals": {{
    "stress_level": 0.0-1.0,
    "topics_causing_stress": ["string"],
    "enthusiasm_topics": ["string"]
  }}
}}

Rules:
- Only extract information explicitly stated or strongly implied
- Set confidence based on how explicit the information is
- Flag contradictions with existing beliefs
- If nothing relevant, return empty arrays
- Do NOT hallucinate information not in the conversation
"""


class StateExtractor:
    """Extracts world state changes from conversations using LLM."""

    def __init__(self, brain: 'OllamaBrain'):
        self.brain = brain

    def extract(self, messages: List[Dict[str, str]],
                current_state_summary: str) -> Dict:
        """
        Run the extraction pipeline on a conversation.

        Args:
            messages: Conversation messages
            current_state_summary: WorldModel.get_context_summary()

        Returns:
            Structured extraction dict matching EXTRACTION_PROMPT schema
        """
        # Format conversation
        conversation_text = self._format_conversation(messages)

        # Build prompt
        prompt = EXTRACTION_PROMPT.format(
            world_state_summary=current_state_summary,
            conversation=conversation_text
        )

        # Call LLM (use fast model for extraction -- this runs every turn)
        response = self.brain.think(
            prompt,
            task_type="reasoning",
            max_tokens=1500
        )

        # Parse JSON response
        return self._parse_extraction(response)

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format messages into readable conversation string."""
        lines = []
        for msg in messages[-10:]:  # Last 10 messages max
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:500]  # Truncate long messages
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_extraction(self, response: str) -> Dict:
        """Parse LLM response into structured extraction dict."""
        # Try to extract JSON from response
        # Handle markdown code blocks, partial JSON, etc.
        # Fall back to empty extraction on parse failure
        pass
```

---

## 5. Proactive Awareness Engine: How It Generates Insights

### 5.1 Staleness Detection

The staleness detector maintains a timeline of project activity and generates escalating alerts:

```
Day 0-3:   No alert (normal gap between sessions)
Day 3-7:   Project health -> YELLOW, low-urgency background note
Day 7-14:  Generate staleness insight (urgency: 0.6)
Day 14+:   Project health -> RED, high-urgency alert (urgency: 0.8)
Day 30+:   Suggest archiving/deprioritizing (urgency: 0.9)
```

Staleness is weighted by project priority. A high-priority project triggers faster alerts.

### 5.2 Goal-Blocker Inference

The engine builds a causal graph connecting blockers to goals:

```
[Blocker: VRAM limit]
    |
    +-- affects --> [Project: AURA] -- contributes_to --> [Goal: Ship proactive features]
    |
    +-- affects --> [Project: AURA] -- contributes_to --> [Goal: Consciousness architecture]
```

When a blocker is persistent (>7 days) and affects a high-priority goal, the engine generates an insight suggesting resolution strategies. It uses the LLM to reason about potential solutions based on the technical context in the world model.

### 5.3 Pattern Recognition Across Conversations

The engine integrates with PatternProphet (157 learned patterns in AURA):

1. **Behavioral patterns**: "User researches before building" -- detect when research phase is complete.
2. **Temporal patterns**: "User works on crypto project on weekends" -- anticipate weekend topics.
3. **Frustration patterns**: "After 3 failed debugging attempts, user needs a different approach" -- suggest alternative after detecting pattern.
4. **Topic sequence patterns**: "After discussing architecture, user usually asks about implementation" -- prepare implementation context.

### 5.4 Priority Shift Detection

Tracks mention frequency trends over sliding windows:

```python
# Priority signal computation
def compute_priority_signal(project, window_days=7):
    recent_mentions = count_mentions(project, last_n_days=window_days)
    previous_mentions = count_mentions(project, days_ago=window_days, window=window_days)

    if previous_mentions == 0:
        return 0.0  # New project, no trend yet

    trend = (recent_mentions - previous_mentions) / max(previous_mentions, 1)
    # trend > 0: rising priority
    # trend < 0: declining priority
    return trend
```

When a project's priority signal diverges significantly from its stated priority, the engine generates a priority shift insight.

### 5.5 Timing-Based Triggers

| Trigger | Condition | Insight |
|---------|-----------|---------|
| **Morning briefing** | First conversation of the day | Summarize overnight insights, today's deadlines |
| **Context switch** | User switches topics from project A to B | Note that A has pending items |
| **End of day** | Last conversation pattern detected | Summarize day's progress, tomorrow's priorities |
| **Weekly review** | 7 days since last review prompt | Suggest reviewing project health and goals |
| **After deadline** | target_date passed, goal not achieved | Alert on missed deadline, suggest replanning |
| **After absence** | >3 days since last conversation | Welcome back, summarize what might have changed |

---

## 6. Implementation Priority

### Phase 1: Foundation (Week 1-2) -- HIGH IMPACT, BUILD FIRST

| Component | Description | Files |
|-----------|-------------|-------|
| SQLite schema | Create `world_model.db` with all tables | `data/world_model.db` |
| `WorldModel.__init__` | DB initialization, loading, snapshot management | `aura/consciousness/world_model.py` |
| `WorldModel.get_context_summary` | Generate LLM-injectable summary from state | Same |
| `brain.py` integration | Inject world state summary into system prompt | `aura/brain.py` |
| Manual state population | CLI or API to seed initial world state | API route |

**Why first**: The context summary alone makes AURA significantly more aware. Even without automatic extraction, having project/goal/belief data in the system prompt transforms response quality.

### Phase 2: Extraction Pipeline (Week 2-3) -- CRITICAL

| Component | Description | Files |
|-----------|-------------|-------|
| `StateExtractor` | LLM-powered conversation analysis | `aura/consciousness/state_extractor.py` |
| `WorldModel.process_conversation` | Full update pipeline | `world_model.py` |
| `chat.py` integration | Call `process_conversation` after every turn | `api/routes/chat.py` |
| Contradiction detection | Semantic comparison against existing beliefs | `world_model.py` |
| State change logging | Full audit trail in SQLite | `world_model.py` |

**Why second**: This is what makes the world model *living*. Without automatic extraction, it would decay into a static document.

### Phase 3: Proactive Awareness (Week 3-4) -- DIFFERENTIATOR

| Component | Description | Files |
|-----------|-------------|-------|
| `ProactiveAwarenessEngine` | Core analysis engine | `aura/consciousness/proactive_awareness.py` |
| Staleness detection | Project/goal/relationship staleness | Same |
| Contradiction alerts | Surface unresolved contradictions | Same |
| Deadline awareness | Approaching milestone/goal deadlines | Same |
| Gateway Daemon integration | Deliver insights via proactive system | `aura/proactive/gateway_daemon.py` |

**Why third**: This is the *visible payoff* -- where the user sees AURA proactively identifying things they haven't asked about.

### Phase 4: Advanced Inference (Week 4-6) -- DEPTH

| Component | Description | Files |
|-----------|-------------|-------|
| Goal-blocker inference | LLM-powered causal reasoning | `proactive_awareness.py` |
| Pattern integration | Connect PatternProphet to awareness engine | Same |
| Priority shift detection | Trend analysis on mention frequency | Same |
| Stress correlation | ALMA + ToM integration | Same |
| Active Inference integration | Feed world state into observation model | `active_inference.py` |
| Inner thoughts integration | World model concerns as thought content | `inner_thoughts_engine.py` |
| Intrinsic motivation integration | World model gaps as drive signals | `intrinsic_motivation.py` |

### Phase 5: Refinement (Ongoing)

| Component | Description |
|-----------|-------------|
| Belief decay tuning | Calibrate half-life based on real usage patterns |
| Extraction accuracy | Improve LLM extraction prompt with few-shot examples from real conversations |
| Insight quality feedback loop | Track which insights users engage vs. dismiss, tune thresholds |
| NeuroDream integration | Sleep consolidation updates world model beliefs |
| World model as GW specialist | Register world model as a Global Workspace codelet |

---

## 7. Key Design Decisions & Rationale

### 7.1 Why SQLite + JSON, not just the Knowledge Graph?

AURA's existing KG (NetworkX + JSONL) stores fine-grained entity-relationship triples. The world model stores higher-level *structured state* -- project objects with status fields, goals with progress percentages, beliefs with confidence scores. These are different abstractions:

- **KG**: "AURA" --uses--> "NetworkX", "User" --works_on--> "AURA" (entity-level)
- **World Model**: Project(name="AURA", status="active", health="green", blockers=[...]) (state-level)

The KG feeds into the world model (entity extraction), and the world model can write back to the KG (new relationships discovered during synthesis), but they serve different purposes.

### 7.2 Why LLM-powered extraction, not rule-based?

Rule-based extractors (regex, NER) miss nuanced state changes. "I'm getting frustrated with this caching bug" contains: (1) a project mention (implied), (2) a blocker (caching bug), (3) an emotional signal (frustrated). An LLM handles this naturally. The extraction LLM call uses the fast model (mistral:7b) and adds about 2-3 seconds per conversation turn -- acceptable latency.

### 7.3 Why bi-temporal beliefs?

Inspired by Zep/Graphiti's temporal model. When a user says "I've switched from React to Svelte," the old belief ("User prefers React") is not deleted -- it is marked with `valid_to = now` and the new belief is created with `valid_from = now`. This enables:
- Time-travel queries: "What did I believe about the user's stack in January?"
- Contradiction auditing: "This belief was superseded 3 times -- unstable area"
- Undo capability: "Actually I was wrong, revert to the previous belief"

### 7.4 Why cap proactive insights at 3 per run?

Based on research from CHI 2025 (Developer Interaction Patterns with Proactive AI): proactive suggestions are perceived as helpful up to a threshold, then become disruptive. Three insights per analysis cycle (which runs at most hourly during idle, or once after each conversation) balances helpfulness with respect for user attention.

---

## 8. References

### World Models
- [Ha & Schmidhuber, "World Models" (2018)](https://arxiv.org/abs/1803.10122)
- [ACM CSUR, "Understanding World or Predicting Future? A Comprehensive Survey of World Models" (2025)](https://dl.acm.org/doi/10.1145/3746449)
- [RWML, "Reinforcement World Model Learning for LLM-based Agents" (2026)](https://arxiv.org/abs/2602.05842)
- [WorldLLM, "Curiosity-Driven Theory-Making" (2025)](https://arxiv.org/abs/2506.06725)
- [WMA Web Agents, "Web Agents with World Models" (2024)](https://arxiv.org/abs/2410.13232)
- [StateAct, "Self-prompting and State-tracking" (2024)](https://arxiv.org/abs/2410.02810)
- [LeCun, I-JEPA (2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA, Meta AI (2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)

### BDI Architectures
- [BDI Ontology (2025)](https://arxiv.org/pdf/2511.17162)
- [Integrating ML into BDI Agents: Survey (2025)](https://arxiv.org/pdf/2510.20641)
- [BDI Agents in Natural Language Environments, AAMAS 2024](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p880.pdf)
- [BDI + LLMs for Human-Robot Interaction (2024)](https://www.sciencedirect.com/science/article/pii/S0952197624019304)

### Situation Awareness
- [Endsley, "Toward a Theory of Situation Awareness" (1995)](https://www.researchgate.net/publication/210198492_Endsley_MR_Toward_a_Theory_of_Situation_Awareness_in_Dynamic_Systems_Human_Factors_Journal_371_32-64)
- [National Academies, "Human-AI Teaming: Situation Awareness" (2021)](https://nap.nationalacademies.org/read/26355/chapter/6)

### Proactive AI
- [Proactive Conversational AI: Comprehensive Survey, ACM TOIS (2025)](https://dl.acm.org/doi/10.1145/3715097)
- [Proactive Agents with Inner Thoughts, CHI 2025](https://arxiv.org/abs/2501.00383)
- [Need Help? Proactive AI for Programming, CHI 2025](https://dl.acm.org/doi/10.1145/3706598.3714002)
- [Developer Interaction Patterns with Proactive AI (2026)](https://arxiv.org/html/2601.10253)
- [ProActLLM](https://proactllm.github.io/)

### Agent Memory
- [Memory in the Age of AI Agents: Survey (2025)](https://arxiv.org/abs/2512.13564)
- [Zep: Temporal Knowledge Graph for Agent Memory (2025)](https://arxiv.org/abs/2501.13956)
- [Graphiti, GitHub](https://github.com/getzep/graphiti)
- [Letta/MemGPT](https://docs.letta.com/concepts/memgpt/)

### Active Inference
- [Parr, Pezzulo, Friston, "Active Inference" (MIT Press, 2022)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
