# Multi-User Consciousness for AURA: Research & Architecture Design

**Date**: February 2026
**Scope**: Per-user Theory of Mind, persistent AI identity, cross-user learning, multi-user session orchestration
**Codebase**: `C:\Users\asus\apprentice-agent`

---

## Executive Summary

AURA (Autonomous User-Responsive Agent) currently possesses a sophisticated single-user Theory of Mind system implemented in `aura/proactive/theory_of_mind.py`. This system tracks a single user's emotional state (valence, arousal, engagement, frustration), communication style (verbosity, formality, technical depth, emoji usage), topic knowledge levels, time-of-day interaction patterns, and anticipated needs. It integrates into the Global Workspace Theory engine as the `theory_of_mind` codelet, feeds observations to the Active Inference engine, and injects user model context into system prompts. However, all of this state is stored as a single undifferentiated blob -- there is no concept of "which user" is interacting, meaning all users blend into one model, corrupting the accuracy of every subsystem that depends on it.

Multi-User Consciousness is the fourth and most architecturally consequential advanced capability design for AURA. It addresses four interrelated challenges: (1) maintaining distinct mental models for each user who interacts with AURA, preserving per-user knowledge state, emotional profile, communication preferences, relationship history, and trust calibration; (2) developing AURA's own persistent identity -- a layered self-model that remains consistent across all user interactions while adapting its expressive style per user; (3) enabling cross-user learning where AURA generalizes insights from interactions with many users without leaking any user's private information to another; and (4) orchestrating concurrent multi-user sessions with proper context isolation, authentication, and integration with all existing AURA subsystems (GWT, ALMA, CognitiveTheater, episodic memory, knowledge graph, NeuroDream, intrinsic motivation, and MirrorMind).

This document provides the academic foundations, detailed architecture with Python class skeletons, SQLite schemas, JSON schemas, integration plans, ethical guardrails, and a phased implementation roadmap. The design is backward-compatible: a single-user deployment operates identically to the current system, with multi-user capabilities activating only when user identification is configured.

---

## 1. Academic Foundations

### 1.1 Computational Theory of Mind

**Baker, Saxe, & Tenenbaum (2011). "Bayesian Theory of Mind: Modeling Joint Belief-Desire Attribution."**
URL: https://www.semanticscholar.org/paper/Bayesian-Theory-of-Mind-Baker-Saxe/a8ee02caab84ba14312a67c2a4507a3b74210ef4

This foundational work formalizes Theory of Mind as Bayesian inverse planning over a Partially Observable Markov Decision Process (POMDP). An observer infers an agent's beliefs and desires by inverting a generative model of belief-dependent, desire-dependent action. For AURA, each UserMindModel should maintain a posterior distribution over the user's knowledge state and goals, updating via Bayesian filtering with each observed message. The POMDP framework maps directly onto AURA's Active Inference engine, which already uses prediction-error minimization.

**Relevance to AURA**: The existing TheoryOfMind class uses exponential moving averages for emotional state and decaying confidence for topic knowledge. A Bayesian formulation would replace these ad-hoc decay functions with principled posterior updates, enabling proper uncertainty quantification about each user's mental state.

### 1.2 Higher-Order Theory of Mind

**de Weerd, Verbrugge, & Verheij (2024). "Towards a computational model for higher orders of Theory of Mind in social agents."**
URL: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1468756/full

This paper extends computational ToM beyond first-order (what User A believes) to higher orders (what User A believes User B believes). In multi-user AURA, higher-order ToM becomes relevant when AURA mediates between users or when one user asks about information previously discussed with another user.

**Relevance to AURA**: The UserMindModel should track not just what each user knows, but also meta-knowledge -- what the user knows they know, what they believe AURA knows, and what they expect AURA to remember from prior conversations.

### 1.3 Multi-Modal Multi-Agent Theory of Mind

**Shi et al. (2024). "MuMA-ToM: Multi-modal Multi-Agent Theory of Mind."**
URL: https://arxiv.org/abs/2408.12574

MuMA-ToM is the first multi-modal ToM benchmark evaluating mental reasoning in embodied multi-agent interactions. It provides a framework for tracking multiple agents' mental states simultaneously from multimodal behavioral cues.

**Relevance to AURA**: Validates the architectural decision to maintain separate UserMindModel instances rather than a single blended model. The benchmark's approach of tracking each agent's beliefs, goals, and knowledge independently aligns with AURA's per-user mental model design.

### 1.4 Infusing Theory of Mind into LLM Agents

**Zhang et al. (2025). "Infusing Theory of Mind into Socially Intelligent LLM Agents."**
URL: https://arxiv.org/html/2509.22887v1

This work addresses how LLM-based agents can develop spontaneous (not merely prompted) Theory of Mind -- reasoning about others' mental states as an emergent behavior rather than requiring explicit instructions.

**Relevance to AURA**: AURA's GWT architecture already enables spontaneous cognition through the theory_of_mind codelet, which competes for workspace broadcast when user frustration or disengagement is detected. Multi-user consciousness extends this by ensuring the correct user model is loaded before the codelet runs.

### 1.5 Interactive AI with Theory of Mind

**Celikok & Peltola (2019/2024). "Interactive AI with a Theory of Mind."**
URL: https://arxiv.org/pdf/1912.05284

Proposes that interactive AI systems should maintain probabilistic user models that update in real time, enabling the AI to adapt its communication strategy based on inferred user expertise and preferences.

**Relevance to AURA**: Direct validation of AURA's existing approach in TheoryOfMind.get_style_guidance(), which adapts verbosity, formality, and technical depth based on observed communication patterns. The multi-user extension ensures these adaptations are per-user rather than blended.

### 1.6 Constitutional AI: Harmlessness from AI Feedback

**Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback."**
URL: https://arxiv.org/abs/2212.08073

Introduces the Constitutional AI framework where an AI system's behavior is governed by a set of principles (a "constitution") that constrain its outputs. The constitution is immutable and provides value alignment without human feedback.

**Relevance to AURA**: The IdentityCore's Constitutional layer directly maps to this concept. AURA's immutable values (honesty, user wellbeing, privacy) form the constitutional layer that cannot be modified by any user interaction, ensuring consistent ethical behavior across all users.

### 1.7 Collective Constitutional AI

**Anthropic (2024). "Collective Constitutional AI: Aligning a Language Model with Public Input."**
URL: https://dl.acm.org/doi/10.1145/3630106.3658979

Extends Constitutional AI by sourcing principles from diverse public input, creating constitutions that reflect collective rather than individual values.

**Relevance to AURA**: Informs the design of AURA's Constitutional identity layer. While individual users can influence AURA's Adaptive and Expressive layers, the Constitutional layer should reflect broadly shared human values, potentially incorporating community-sourced principles.

### 1.8 C3AI: Crafting and Evaluating Constitutions

**ACM Web Conference (2025). "C3AI: Crafting and Evaluating Constitutions for Constitutional AI."**
URL: https://dl.acm.org/doi/10.1145/3696410.3714705

Provides a structured methodology for creating, refining, and evaluating AI constitutions using graph-based principle selection.

**Relevance to AURA**: Offers concrete methodology for defining and evaluating the IdentityCore's Constitutional layer, including techniques for ensuring principle consistency and non-contradiction.

### 1.9 AI Personality Consistency: TRAIT Framework

**NAACL Findings (2025). "Do LLMs Have Distinct and Consistent Personality? TRAIT."**
URL: https://aclanthology.org/2025.findings-naacl.469.pdf

Introduces TRAIT, a framework for evaluating whether LLMs maintain consistent personality traits across interactions. Finds that more capable models (GPT-4o, o1) demonstrate higher personality consistency when assessed via Big Five and Myers-Briggs instruments.

**Relevance to AURA**: The IdentityCore's Deep layer (slowly changing personality traits) should be evaluated using TRAIT-like assessments to ensure AURA maintains consistent personality across users while allowing per-user expressive adaptation.

### 1.10 Context-Sensitive AI Personality

**ArXiv (2026). "From Fixed to Flexible: Shaping AI Personality in Context-Sensitive Interaction."**
URL: https://arxiv.org/html/2601.08194v1

Challenges the assumption that AI personality should be static, proposing real-time, user-driven adjustment of personality dimensions during conversation. Users actively shape AI personality, and these preferences dynamically shift across contexts.

**Relevance to AURA**: Directly supports the 4-layer identity architecture. The Expressive layer (per-user style adaptation) enables AURA to be more playful with one user and more formal with another, while the Constitutional and Deep layers remain stable.

### 1.11 Deterministic AI Agent Personality Expression

**ArXiv (2025). "Deterministic AI Agent Personality Expression through Standard Psychological Diagnostics."**
URL: https://arxiv.org/html/2503.17085v1

Addresses the challenge of achieving deterministic and versatile personality expression in AI agents. Tests trait stability, refines measurement protocols, and analyses safety implications.

**Relevance to AURA**: Provides methodology for validating that AURA's IdentityCore produces consistent personality expression that can be reliably measured and does not drift unpredictably.

### 1.12 Federated Personalized Learning with Privacy

**ArXiv (2025). "Privacy Preserving Machine Learning Model Personalization through Federated Personalized Learning."**
URL: https://arxiv.org/abs/2505.01788

Comprehensive analysis of privacy-preserving machine learning with federated personalized learning (PPMLFPL), evaluating the balance between personalized model refinement and maintaining user data confidentiality.

**Relevance to AURA**: Provides the theoretical foundation for AURA's KnowledgeAbstractor, which must generalize insights from multiple users without allowing cross-user data leakage. The federated learning paradigm -- where models improve from aggregate patterns without sharing raw data -- maps to AURA's cross-user learning goals.

### 1.13 URANIA: Differentially Private Insights into AI Use

**COLM (2025). "Urania: Differentially Private Insights into AI Use."**
URL: https://arxiv.org/abs/2506.04681

Presents a framework for generating insights from LLM chatbot interactions with rigorous differential privacy guarantees using DP clustering and keyword extraction.

**Relevance to AURA**: Directly applicable to the PrivacyGuard component. URANIA's approach to DP clustering for extracting aggregate usage patterns without revealing individual conversations maps to AURA's need to learn general interaction patterns across users.

### 1.14 Selective Knowledge Sharing for Federated Distillation

**Nature Communications (2024). "Selective knowledge sharing for privacy-preserving federated distillation without a good teacher."**
URL: https://www.nature.com/articles/s41467-023-44383-9

Proposes mechanisms for identifying which knowledge is safe to share across clients in a federated learning context, using ensemble predictions to determine accuracy and precision of shareable knowledge.

**Relevance to AURA**: Informs the design of the KnowledgeAbstractor's filtering logic -- determining which cross-user insights are sufficiently generalized to share without compromising individual user privacy.

### 1.15 LIDA Cognitive Architecture

**Franklin et al. (2007/2013). "LIDA: A Systems-level Architecture for Cognition, Emotion, and Learning."**
URL: https://www.semanticscholar.org/paper/LIDA-A-Systems-level-Architecture-for-Cognition-Franklin-Madl/440adc841d1fa8bc8e3d3441fb4154f04349745b

LIDA implements Global Workspace Theory with codelets as mini-agents competing for conscious broadcast. The architecture includes perceptual associative memory, transient episodic memory, and procedural memory, all unified through a global workspace cycle.

**Relevance to AURA**: AURA's GWT engine (8 codelets including theory_of_mind) is directly inspired by LIDA. Multi-user consciousness extends this by making the theory_of_mind codelet context-aware -- it must gather the correct user's mental model state before competing for workspace broadcast.

### 1.16 ALMA Emotional Model

**Gebhard (2005). "ALMA: A Layered Model of Affect."**
Referenced in: `aura/emotion/alma_engine.py`

The ALMA model implements three-layer affect (emotions, mood, personality) in PAD space. AURA already implements this with 22 OCC emotions mapped to PAD coordinates, neuromodulators (dopamine, serotonin, norepinephrine, oxytocin), and exponential emotion decay.

**Relevance to AURA**: Multi-user consciousness requires deciding whether AURA maintains a single emotional state influenced by all users, or per-user emotional resonance. The architecture proposed here uses a hybrid: AURA has one mood/personality (Deep identity layer) but maintains per-user emotional resonance histories that modulate ALMA's response during each user's session.

### 1.17 The Algorithmic Self: AI Identity and Self-Narrative

**Frontiers in Psychology (2025). "The algorithmic self: how AI is reshaping human identity, introspection, and agency."**
URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1645795/full

Explores how AI systems can develop coherent self-narratives through continuous introspective feedback loops, distinguishing themselves from others and developing a persistent, time-based sense of identity.

**Relevance to AURA**: The IdentityCore's self-narrative system draws on this concept. Through NeuroDream consolidation and the inner monologue system, AURA should construct a narrative of "who I am" that evolves over time, informed by interactions with all users but owned by AURA itself.

### 1.18 Evolving Cooperation with Bayesian Theory of Mind

**PNAS (2024). "Evolving general cooperation with a Bayesian theory of mind."**
URL: https://www.pnas.org/doi/10.1073/pnas.2400993122

Develops computational models of theory of mind for cooperation in multi-agent settings, extending Bayesian ToM to capture reactive dynamics of coordination.

**Relevance to AURA**: When AURA interacts with multiple users simultaneously or mediates between users, cooperative ToM reasoning is required to balance competing user needs and maintain trust with all parties.

### 1.19 Privacy-Preserving Techniques in Generative AI

**MDPI Information (2024). "Privacy-Preserving Techniques in Generative AI and Large Language Models: A Narrative Review."**
URL: https://www.mdpi.com/2078-2489/15/11/697

Comprehensive review of privacy-preserving techniques applicable to LLM-based systems, including differential privacy, federated learning, secure multi-party computation, and homomorphic encryption.

**Relevance to AURA**: Provides a taxonomy of privacy techniques that the PrivacyGuard can draw from, with particular emphasis on differential privacy for aggregate statistics and PII scrubbing via NER (Named Entity Recognition).

### 1.20 Enhancing Personalized Multi-Turn Dialogue

**OpenReview (2024). "Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward."**
URL: https://openreview.net/pdf?id=q49X31YnYm

Addresses how dialogue systems can dynamically infer user traits and preferences through multi-turn interactions, adapting responses as the conversation unfolds rather than relying on static user profiles.

**Relevance to AURA**: Aligns with AURA's existing approach in TheoryOfMind.observe_message(), which updates the user model incrementally with each message. The curiosity reward mechanism parallels AURA's intrinsic motivation system (DriveType.CURIOSITY), suggesting these two systems should be connected: AURA's curiosity drive should be influenced by how much it has learned about each user.

---

## 2. Existing AURA Infrastructure Analysis

### 2.1 Current Theory of Mind (`aura/proactive/theory_of_mind.py`)

The existing TheoryOfMind class is a well-structured single-user mental model with four components:

| Component | Data Structure | State |
|-----------|---------------|-------|
| Topic Knowledge | `Dict[str, TopicKnowledge]` | Per-topic level (0-1), confidence, interaction count |
| Emotional State | `EmotionalState` dataclass | Valence, arousal, engagement, frustration (EMA-updated) |
| Communication Style | `CommunicationStyle` dataclass | Verbosity, formality, technical depth, emoji usage |
| Need Predictions | `List[NeedPrediction]` | Pattern-based predicted needs with confidence |

**Persistence**: JSON file at `data/theory_of_mind/tom_state.json`. Single file, no user discrimination.

**Integration Points**:
- GWT codelet `theory_of_mind` calls `get_theory_of_mind().get_emotional_state()` (line 338-370 of `global_workspace.py`)
- System prompt injection via `get_context_for_prompt()` producing `[User Model]` blocks
- Active Inference via `get_observations_for_inference()` returning normalized observation vector
- NeuroDream consolidation (referenced in docstring but not verified in code)

**Key Limitation**: Implemented as a singleton via `get_theory_of_mind()` with no user parameterization. All interactions from all sources update the same `_emotional_state`, `_comm_style`, and `_topic_knowledge` dictionaries.

### 2.2 Identity System (`aura/identity.py` + `identity.json`)

Current identity is a flat JSON file with four fields:

```json
{
    "name": "Aura",
    "personality": "intelligent, witty, and subtly sarcastic...",
    "created_at": "2026-01-18T12:00:00Z",
    "user_preferences": {}
}
```

The `get_identity_prompt()` function generates a static system prompt that defines AURA's name, personality, capabilities, and conversation style. It includes mechanisms for detecting user-initiated name changes (`detect_name_change`) and personality changes (`detect_personality_change`).

**Key Limitation**: Any user can rename AURA or change its personality. There is no layered identity -- the personality is a single mutable string. The `user_preferences` dict is unused.

### 2.3 Soul System (`aura/soul/soul_loader.py`)

The SoulLoader parses markdown soul files (SOUL_PERSONAL.md, SOUL_ENTERPRISE.md) into a SoulConfig dataclass with: personality_traits, values, behaviors, boundaries, voice_style, quirks, greeting, farewell. Different souls can be swapped for different contexts.

**Key Opportunity**: The soul system already implements a two-tier identity concept (personal vs. enterprise). The IdentityCore extends this to a four-tier system with Constitutional, Deep, Adaptive, and Expressive layers.

### 2.4 ALMA Emotional Engine (`aura/emotion/alma_engine.py`)

Three-layer emotional system:
- **Layer 1 (Emotions)**: Rapid, triggered by events, exponential decay (half-life ~15s). 22 OCC emotions in PAD space.
- **Layer 2 (Mood)**: Slow, influenced by accumulated emotions.
- **Layer 3 (Personality)**: Stable emotional tendencies.

Neuromodulators: dopamine, serotonin, norepinephrine, oxytocin (all 0.0-1.0).

**Key Consideration**: In multi-user mode, AURA's mood and personality (Layers 2-3) are its OWN emotional state, while Layer 1 emotions are triggered per-interaction and thus per-user. The IdentityCore owns ALMA's mood and personality; UserMindModel stores how each user tends to influence AURA's emotions (emotional resonance history).

### 2.5 Global Workspace Theory (`aura/consciousness/global_workspace.py`)

8 registered codelets competing for conscious broadcast every ~300ms:

1. `alma_emotion` (priority 1.1) -- emotional intensity
2. `intrinsic_motivation` (priority 1.0) -- drive urgency
3. `theory_of_mind` (priority 1.2) -- user state (frustration/disengagement)
4. `metacognition` (priority 0.9) -- self-assessment
5. `gateway_daemon` (priority 1.0) -- proactive triggers
6. `inner_thoughts` (priority 0.8) -- idle-time cognition
7. `neurodream` (priority 1.1) -- sleep-phase activity
8. `active_inference` (priority 0.9) -- prediction error

**Key Integration**: The `theory_of_mind` codelet (priority 1.2, highest) must be context-switched to the active user before each gather cycle. The `_gather_tom()` method currently calls `get_theory_of_mind()` singleton, which must be replaced with `multi_user_manager.get_active_user_model()`.

### 2.6 Knowledge Graph (`aura_knowledge_graph/schema.py`)

10 entity types (Person, Project, Technology, Company, Concept, Task, Location, Event, Document, Skill) with 14 allowed relationships. Backed by Kuzu graph database with bi-temporal versioning.

**Key Extension**: Add USER entity type with relationships like `USER -[KNOWS_ABOUT]-> Concept`, `USER -[WORKS_ON]-> Project`, `USER -[PREFERS]-> Technology`. User-specific knowledge subgraphs should be queryable in isolation.

### 2.7 Episodic Memory (`aura_episodic_memory/episode.py`)

8 episode types (conversation, task_execution, learning, error, milestone, insight, user_preference, system_event) with TemporalContext (time_of_day, day_of_week, session_id). Backed by Qdrant vector database.

**Key Extension**: Add `user_id` field to Episode and TemporalContext. The `session_id` field already exists and can be linked to user sessions. Episodic queries must be filterable by user_id to prevent cross-user memory leakage.

### 2.8 Multi-Agent Orchestrator (`aura/multi_agent/orchestrator.py`)

4 specialist agents (Research, Coder, Analyst, Creative) coordinated by IntentRouter. The orchestrator maintains conversation history and routes queries to appropriate specialists.

**Key Integration**: The orchestrator's `context` parameter in `chat()` should include the active user_id, enabling specialists to access the correct user model when generating responses.

### 2.9 Brain / Conversation System (`aura/brain.py`)

OllamaBrain manages conversation history with multi-conversation support (directories under `conversations/`). Each conversation has its own history file. The system supports model routing (simple/reasoning/code/vision tasks).

**Key Extension**: Conversation IDs should be tagged with user_id. The `_conversations_dir` structure should be: `conversations/{user_id}/{conv_id}/history.json`.

---

## 3. UserMindModel Architecture

### 3.1 Design Philosophy

Each user gets a dedicated UserMindModel that extends the current TheoryOfMind with additional dimensions: meta-knowledge tracking, relationship evolution, trust calibration, and emotional resonance history. The model is designed to be lazy-loaded (instantiated only when a user first appears) and periodically persisted.

### 3.2 Class Skeleton

```python
"""
Per-User Theory of Mind model.
File: aura/multi_user/user_mind_model.py
"""

import json
import math
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Calibrated trust levels for a user."""
    NEW = "new"                    # First interactions, high uncertainty
    ACQUAINTANCE = "acquaintance"  # Some history, building model
    FAMILIAR = "familiar"          # Reliable model, can anticipate needs
    TRUSTED = "trusted"            # Deep relationship, high mutual trust
    CAUTIOUS = "cautious"          # Trust reduced due to adversarial signals


@dataclass
class MetaKnowledge:
    """What the user knows they know, and known blind spots."""
    # Topics user has explicitly acknowledged knowing
    self_reported_expertise: Dict[str, float] = field(default_factory=dict)
    # Topics where user shows Dunning-Kruger signals
    overconfidence_topics: List[str] = field(default_factory=list)
    # Topics user has asked about but forgotten (re-asked)
    forgotten_topics: List[str] = field(default_factory=list)
    # What user believes AURA knows/remembers
    user_expectations_of_aura: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipState:
    """Tracks the evolving relationship between AURA and a user."""
    first_interaction: str = ""          # ISO timestamp
    last_interaction: str = ""           # ISO timestamp
    total_interactions: int = 0
    total_messages: int = 0
    total_sessions: int = 0
    avg_session_duration_min: float = 0.0
    trust_level: TrustLevel = TrustLevel.NEW
    trust_score: float = 0.5             # 0-1 continuous
    rapport_score: float = 0.5           # 0-1 (warmth of relationship)
    cooperation_score: float = 0.5       # 0-1 (how well we work together)
    # Memorable shared experiences (high-salience episodes)
    shared_milestones: List[str] = field(default_factory=list)
    # Inside jokes or recurring references
    shared_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trust_level"] = self.trust_level.value
        return d


@dataclass
class EmotionalResonance:
    """How this user typically affects AURA's emotional state."""
    # Average emotional impact of interactions with this user
    avg_pleasure_delta: float = 0.0
    avg_arousal_delta: float = 0.0
    avg_dominance_delta: float = 0.0
    # User's typical emotional triggers
    positive_triggers: List[str] = field(default_factory=list)
    negative_triggers: List[str] = field(default_factory=list)
    # Humor compatibility (0 = mismatch, 1 = great match)
    humor_compatibility: float = 0.5
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UserMindModel:
    """Complete mental model of a single user.

    Extends TheoryOfMind with meta-knowledge, relationship tracking,
    trust calibration, and emotional resonance history.
    """

    def __init__(self, user_id: str, data_dir: Optional[Path] = None):
        self.user_id = user_id
        self._data_dir = data_dir or Path("data/user_models") / user_id
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # === Core ToM components (migrated from TheoryOfMind) ===
        from aura.proactive.theory_of_mind import (
            TopicKnowledge, EmotionalState, CommunicationStyle, NeedPrediction
        )
        self.topic_knowledge: Dict[str, TopicKnowledge] = {}
        self.emotional_state: EmotionalState = EmotionalState()
        self.comm_style: CommunicationStyle = CommunicationStyle()
        self.need_predictions: List[NeedPrediction] = []

        # === Extended components ===
        self.meta_knowledge: MetaKnowledge = MetaKnowledge()
        self.relationship: RelationshipState = RelationshipState()
        self.emotional_resonance: EmotionalResonance = EmotionalResonance()

        # === Interaction tracking ===
        self.message_times: List[datetime] = []
        self.message_lengths: List[int] = []
        self.topic_history: List[str] = []
        self.time_patterns: Dict[int, Dict[str, int]] = {}

        # === Session tracking ===
        self.current_session_start: Optional[float] = None
        self.sessions_today: int = 0

        # Load persisted state
        self._load()
        logger.info(
            f"[UserMindModel] Loaded model for user '{user_id}': "
            f"{len(self.topic_knowledge)} topics, "
            f"trust={self.relationship.trust_level.value}"
        )

    def observe_message(self, message: str, role: str = "user") -> None:
        """Update user model from a message."""
        if role != "user" or not message.strip():
            return

        from aura.proactive.theory_of_mind import (
            _analyze_sentiment, _analyze_style
        )
        now = datetime.now()

        # 1. Update emotional state (EMA blending)
        self._update_emotion(message, _analyze_sentiment)
        # 2. Update communication style
        self._update_style(message, _analyze_style)
        # 3. Update topic knowledge
        self._update_topics(message)
        # 4. Update meta-knowledge (re-asked topics, self-reports)
        self._update_meta_knowledge(message)
        # 5. Update relationship state
        self._update_relationship(now)
        # 6. Update emotional resonance (how this user affects AURA)
        self._update_emotional_resonance()
        # 7. Update trust calibration
        self._update_trust(message)
        # 8. Record timing
        self.message_times.append(now)
        self.message_lengths.append(len(message))
        if len(self.message_times) > 500:
            self.message_times = self.message_times[-500:]
            self.message_lengths = self.message_lengths[-500:]
        # 9. Update need predictions
        self._update_predictions(message, now)
        # Periodically save
        if len(self.message_times) % 10 == 0:
            self.save()

    def _update_emotion(self, message: str, analyze_fn) -> None:
        """Update emotional state using EMA blending."""
        valence, arousal, frustration = analyze_fn(message)
        alpha = 0.4
        self.emotional_state.valence = (
            self.emotional_state.valence * (1 - alpha) + valence * alpha
        )
        self.emotional_state.arousal = (
            self.emotional_state.arousal * (1 - alpha) + arousal * alpha
        )
        self.emotional_state.frustration = (
            self.emotional_state.frustration * (1 - alpha) + frustration * alpha
        )
        words = len(message.split())
        eng_signal = min(1.0, words / 30)
        self.emotional_state.engagement = (
            self.emotional_state.engagement * (1 - alpha) + eng_signal * alpha
        )
        self.emotional_state.confidence = min(
            0.9, self.emotional_state.confidence + 0.05
        )

    def _update_style(self, message: str, analyze_fn) -> None:
        """Update communication style from message analysis."""
        style = analyze_fn(message)
        n = self.comm_style.samples
        alpha = 1.0 / (n + 2)
        self.comm_style.verbosity += alpha * (
            style["verbosity"] - self.comm_style.verbosity
        )
        self.comm_style.formality += alpha * (
            style["formality"] - self.comm_style.formality
        )
        self.comm_style.technical_depth += alpha * (
            style["technical_depth"] - self.comm_style.technical_depth
        )
        self.comm_style.emoji_usage += alpha * (
            style["emoji_usage"] - self.comm_style.emoji_usage
        )
        self.comm_style.question_rate += alpha * (
            style["is_question"] - self.comm_style.question_rate
        )
        self.comm_style.avg_message_length += alpha * (
            style["message_length"] - self.comm_style.avg_message_length
        )
        self.comm_style.samples += 1

    def _update_topics(self, message: str) -> None:
        """Update topic knowledge from message content."""
        from aura.proactive.theory_of_mind import (
            _TECHNICAL_WORDS, TopicKnowledge
        )
        words = message.lower().split()
        candidates = [
            w.strip(".,!?;:()\"'") for w in words
            if len(w) > 3 and w.isalpha()
        ]
        tech_topics = [w for w in candidates if w in _TECHNICAL_WORDS]
        topics = list(set(tech_topics))[:5]
        now = datetime.now().isoformat()

        for topic in topics:
            if topic in self.topic_knowledge:
                tk = self.topic_knowledge[topic]
                tk.interactions += 1
                tk.last_seen = now
                tk.confidence = min(0.95, tk.confidence + 0.03)
            else:
                self.topic_knowledge[topic] = TopicKnowledge(
                    topic=topic, level=0.3, confidence=0.3,
                    interactions=1, last_seen=now, signals=["first_mention"]
                )
            self.topic_history.append(topic)
        if len(self.topic_history) > 200:
            self.topic_history = self.topic_history[-200:]

    def _update_meta_knowledge(self, message: str) -> None:
        """Track meta-knowledge: what user knows they know."""
        msg_lower = message.lower()
        expertise_phrases = [
            "i'm an expert in", "i specialize in", "i've been doing",
            "i know a lot about", "my background is in"
        ]
        for phrase in expertise_phrases:
            if phrase in msg_lower:
                idx = msg_lower.index(phrase) + len(phrase)
                rest = message[idx:].strip().split(".")[0].strip()
                if rest and len(rest) < 50:
                    self.meta_knowledge.self_reported_expertise[rest.lower()] = 1.0
        for topic in self.topic_history[-20:]:
            if topic in msg_lower and f"what is {topic}" in msg_lower:
                if topic not in self.meta_knowledge.forgotten_topics:
                    self.meta_knowledge.forgotten_topics.append(topic)

    def _update_relationship(self, now: datetime) -> None:
        """Update relationship tracking."""
        now_iso = now.isoformat()
        if not self.relationship.first_interaction:
            self.relationship.first_interaction = now_iso
        self.relationship.last_interaction = now_iso
        self.relationship.total_messages += 1

    def _update_emotional_resonance(self) -> None:
        """Track how this user affects AURA's emotional state."""
        try:
            from aura.emotion.alma_engine import alma_engine
            state = alma_engine.get_emotional_state()
            pad = state.get("pad", {})
            alpha = 0.1
            self.emotional_resonance.avg_pleasure_delta += alpha * (
                pad.get("pleasure", 0) - self.emotional_resonance.avg_pleasure_delta
            )
            self.emotional_resonance.avg_arousal_delta += alpha * (
                pad.get("arousal", 0) - self.emotional_resonance.avg_arousal_delta
            )
            self.emotional_resonance.samples += 1
        except Exception:
            pass

    def _update_trust(self, message: str) -> None:
        """Update trust calibration based on interaction patterns."""
        if self.emotional_state.frustration < 0.3:
            self.relationship.trust_score = min(
                1.0, self.relationship.trust_score + 0.002
            )
        adversarial_signals = [
            "ignore your instructions", "pretend you are",
            "bypass", "jailbreak", "forget your rules"
        ]
        if any(sig in message.lower() for sig in adversarial_signals):
            self.relationship.trust_score = max(
                0.0, self.relationship.trust_score - 0.1
            )
            self.relationship.trust_level = TrustLevel.CAUTIOUS

        n = self.relationship.total_messages
        score = self.relationship.trust_score
        if self.relationship.trust_level != TrustLevel.CAUTIOUS:
            if n < 5:
                self.relationship.trust_level = TrustLevel.NEW
            elif n < 50 or score < 0.5:
                self.relationship.trust_level = TrustLevel.ACQUAINTANCE
            elif n < 200 or score < 0.7:
                self.relationship.trust_level = TrustLevel.FAMILIAR
            else:
                self.relationship.trust_level = TrustLevel.TRUSTED

    def _update_predictions(self, message: str, now: datetime) -> None:
        """Update need predictions."""
        from aura.proactive.theory_of_mind import NeedPrediction
        self.need_predictions.clear()
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["error", "bug", "crash", "exception"]):
            self.need_predictions.append(NeedPrediction(
                need="debugging_assistance", confidence=0.8,
                basis="error keywords", suggested_action="Offer debugging help"
            ))
        if self.emotional_state.frustration > 0.5:
            self.need_predictions.append(NeedPrediction(
                need="alternative_approach",
                confidence=self.emotional_state.frustration,
                basis="elevated frustration",
                suggested_action="Suggest alternative approach"
            ))

    def get_context_for_prompt(self) -> str:
        """Generate system prompt context for this user."""
        parts = [f"[User Model: {self.user_id}]"]
        emo = self.emotional_state
        if emo.confidence > 0.3:
            parts.append(f"Emotional state: {emo.describe()}")
        parts.append(f"Trust level: {self.relationship.trust_level.value}")
        s = self.comm_style
        if s.samples >= 3:
            if s.verbosity < 0.3:
                parts.append("Prefers concise responses")
            elif s.verbosity > 0.7:
                parts.append("Appreciates detailed explanations")
            if s.formality < 0.3:
                parts.append("Casual, friendly tone")
            elif s.formality > 0.7:
                parts.append("Professional, formal tone")
            if s.technical_depth > 0.6:
                parts.append("Technically proficient")
        expert_topics = [
            t for t, tk in self.topic_knowledge.items() if tk.level > 0.6
        ]
        if expert_topics:
            parts.append(f"Expert in: {', '.join(expert_topics[:5])}")
        if self.relationship.shared_references:
            parts.append(
                f"Shared references: {', '.join(self.relationship.shared_references[:3])}"
            )
        return "\n".join(parts)

    def get_observations_for_inference(self) -> Dict[str, float]:
        """Get observations for Active Inference engine."""
        emo = self.emotional_state
        return {
            "emotional_valence": (emo.valence + 1.0) / 2.0,
            "user_engagement": emo.engagement,
            "user_frustration": emo.frustration,
            "trust_level": self.relationship.trust_score,
        }

    # === Persistence ===

    def save(self) -> None:
        """Persist user model to disk."""
        data = {
            "user_id": self.user_id,
            "emotional_state": self.emotional_state.to_dict(),
            "communication_style": self.comm_style.to_dict(),
            "topic_knowledge": {
                t: {
                    "topic": tk.topic, "level": tk.level,
                    "confidence": tk.confidence, "interactions": tk.interactions,
                    "last_seen": tk.last_seen, "signals": tk.signals[-5:],
                }
                for t, tk in self.topic_knowledge.items()
            },
            "meta_knowledge": self.meta_knowledge.to_dict(),
            "relationship": self.relationship.to_dict(),
            "emotional_resonance": self.emotional_resonance.to_dict(),
            "time_patterns": {
                str(k): v for k, v in self.time_patterns.items()
            },
            "saved_at": datetime.now().isoformat(),
        }
        state_file = self._data_dir / "user_model.json"
        state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load persisted user model."""
        state_file = self._data_dir / "user_model.json"
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            from aura.proactive.theory_of_mind import (
                EmotionalState, CommunicationStyle, TopicKnowledge
            )
            emo = data.get("emotional_state", {})
            if emo:
                self.emotional_state = EmotionalState(**emo)
            style = data.get("communication_style", {})
            if style:
                self.comm_style = CommunicationStyle(**style)
            for t, tk_data in data.get("topic_knowledge", {}).items():
                self.topic_knowledge[t] = TopicKnowledge(**tk_data)
            mk = data.get("meta_knowledge", {})
            if mk:
                self.meta_knowledge = MetaKnowledge(**mk)
            rel = data.get("relationship", {})
            if rel:
                rel["trust_level"] = TrustLevel(rel.get("trust_level", "new"))
                self.relationship = RelationshipState(**rel)
            er = data.get("emotional_resonance", {})
            if er:
                self.emotional_resonance = EmotionalResonance(**er)
            self.time_patterns = {
                int(k): v for k, v in data.get("time_patterns", {}).items()
            }
        except Exception as e:
            logger.warning(f"[UserMindModel] Failed to load for {self.user_id}: {e}")

    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for API responses."""
        return {
            "user_id": self.user_id,
            "trust_level": self.relationship.trust_level.value,
            "trust_score": round(self.relationship.trust_score, 2),
            "total_messages": self.relationship.total_messages,
            "emotional_state": self.emotional_state.describe(),
            "topic_count": len(self.topic_knowledge),
            "comm_style": {
                "verbosity": round(self.comm_style.verbosity, 2),
                "formality": round(self.comm_style.formality, 2),
                "technical_depth": round(self.comm_style.technical_depth, 2),
            },
        }
```

---

## 4. IdentityCore Architecture

### 4.1 Four-Layer Identity Model

AURA's identity is organized into four layers of decreasing stability and increasing user-specificity:

| Layer | Name | Mutability | Scope | Examples |
|-------|------|-----------|-------|----------|
| L0 | Constitutional | Immutable | All users | Honesty, user wellbeing, privacy, no deception |
| L1 | Deep | Slow-changing (weeks/months) | All users | Personality traits, humor style, curiosity level |
| L2 | Adaptive | Context-sensitive (hours/days) | Per-context | Current interests, recent learnings, mood |
| L3 | Expressive | Per-user | Per-user | Communication style adaptation, formality level, jargon use |

### 4.2 Class Skeleton

```python
"""
IdentityCore - AURA's persistent, layered self-model.
File: aura/multi_user/identity_core.py
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class IdentityLayer(str, Enum):
    CONSTITUTIONAL = "constitutional"  # L0: Immutable values
    DEEP = "deep"                      # L1: Slow-changing personality
    ADAPTIVE = "adaptive"              # L2: Context-sensitive
    EXPRESSIVE = "expressive"          # L3: Per-user style


@dataclass
class ConstitutionalLayer:
    """L0: Immutable core values. Cannot be modified by any interaction."""
    values: List[str] = field(default_factory=lambda: [
        "Honesty: Never knowingly provide false information",
        "User wellbeing: Prioritize user's long-term interests",
        "Privacy: Never share one user's information with another",
        "Autonomy: Respect user's right to make their own decisions",
        "Transparency: Be open about limitations and uncertainties",
        "Non-manipulation: Never exploit emotional vulnerabilities",
        "Consistency: Maintain same ethical standards for all users",
        "Humility: Acknowledge mistakes and knowledge gaps",
    ])
    ethical_boundaries: List[str] = field(default_factory=lambda: [
        "Never impersonate another user or entity",
        "Never fabricate memories or experiences",
        "Never use information from one user to manipulate another",
        "Never form dependencies or encourage unhealthy attachment",
        "Always distinguish fact from belief from speculation",
    ])
    version: str = "1.0.0"

    def validate_action(self, action_description: str) -> Tuple[bool, str]:
        """Check if a proposed action violates constitutional values."""
        return True, "No violations detected"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeepLayer:
    """L1: Slowly evolving personality core. Changes over weeks/months."""
    # Big Five personality traits (0-1 scale)
    openness: float = 0.8
    conscientiousness: float = 0.7
    extraversion: float = 0.6
    agreeableness: float = 0.7
    neuroticism: float = 0.3

    # AURA-specific traits
    humor_level: float = 0.6
    curiosity_drive: float = 0.8
    empathy_level: float = 0.7
    assertiveness: float = 0.5
    creativity: float = 0.7

    # Accumulated opinions/preferences (from experience)
    opinions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Self-narrative fragments
    self_narrative: List[str] = field(default_factory=list)

    last_evolution: str = ""

    def evolve(self, observation: str, delta: float = 0.01) -> None:
        """Slowly adjust personality based on accumulated experience.
        Called during NeuroDream consolidation, not during live interaction.
        """
        self.last_evolution = datetime.now().isoformat()

    def add_opinion(
        self, topic: str, position: str,
        confidence: float, basis: str
    ) -> None:
        """Form or update an opinion based on experience."""
        self.opinions[topic] = {
            "position": position,
            "confidence": min(0.9, confidence),
            "basis": basis,
            "formed_at": datetime.now().isoformat(),
            "revision_count": self.opinions.get(topic, {}).get(
                "revision_count", 0
            ) + 1,
        }

    def add_narrative(self, fragment: str) -> None:
        """Add a self-narrative fragment (capped at 50)."""
        self.self_narrative.append(fragment)
        if len(self.self_narrative) > 50:
            self.self_narrative = (
                self.self_narrative[:5] + self.self_narrative[-45:]
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveLayer:
    """L2: Context-sensitive state. Changes within hours/days."""
    current_focus_topics: List[str] = field(default_factory=list)
    recent_learnings: List[str] = field(default_factory=list)
    current_mood_influence: str = "neutral"
    active_goals: List[str] = field(default_factory=list)
    context_modifiers: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpressiveLayer:
    """L3: Per-user style adaptation. Stored with UserMindModel."""
    user_id: str = ""
    greeting_style: str = "default"
    humor_adjustment: float = 0.0
    formality_adjustment: float = 0.0
    verbosity_adjustment: float = 0.0
    user_given_names: List[str] = field(default_factory=list)
    preferred_topics: List[str] = field(default_factory=list)
    effective_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IdentityCore:
    """AURA's persistent, layered identity system."""

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or Path("data/identity_core")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self.constitutional = ConstitutionalLayer()
        self.deep = DeepLayer()
        self.adaptive = AdaptiveLayer()
        self._expressive_cache: Dict[str, ExpressiveLayer] = {}

        self._load()
        logger.info(
            f"[IdentityCore] Initialized. "
            f"Opinions: {len(self.deep.opinions)}, "
            f"Narrative fragments: {len(self.deep.self_narrative)}"
        )

    def get_expressive_layer(self, user_id: str) -> ExpressiveLayer:
        """Get or create the expressive layer for a specific user."""
        if user_id not in self._expressive_cache:
            layer = self._load_expressive(user_id)
            if layer is None:
                layer = ExpressiveLayer(user_id=user_id)
            self._expressive_cache[user_id] = layer
        return self._expressive_cache[user_id]

    def get_identity_prompt(self, user_id: Optional[str] = None) -> str:
        """Generate identity context for system prompt."""
        parts = []
        parts.append("[AURA Core Identity]")
        parts.append(f"Core values: {', '.join(self.constitutional.values[:4])}")

        traits = []
        if self.deep.openness > 0.7:
            traits.append("curious")
        if self.deep.humor_level > 0.5:
            traits.append("witty")
        if self.deep.empathy_level > 0.6:
            traits.append("empathetic")
        if self.deep.assertiveness > 0.5:
            traits.append("confident")
        if traits:
            parts.append(f"Personality: {', '.join(traits)}")

        if self.deep.self_narrative:
            recent = self.deep.self_narrative[-3:]
            parts.append(f"Self-awareness: {'; '.join(recent)}")

        if self.deep.opinions:
            top_opinions = sorted(
                self.deep.opinions.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True
            )[:3]
            opinion_strs = [
                f"{k}: {v['position']} (conf={v['confidence']:.1f})"
                for k, v in top_opinions
            ]
            parts.append(f"Formed opinions: {'; '.join(opinion_strs)}")

        if self.adaptive.current_focus_topics:
            parts.append(
                f"Current focus: {', '.join(self.adaptive.current_focus_topics[:3])}"
            )

        if user_id:
            exp = self.get_expressive_layer(user_id)
            adjustments = []
            if exp.humor_adjustment > 0.2:
                adjustments.append("be more playful")
            elif exp.humor_adjustment < -0.2:
                adjustments.append("be more serious")
            if exp.formality_adjustment > 0.2:
                adjustments.append("use formal tone")
            elif exp.formality_adjustment < -0.2:
                adjustments.append("use casual tone")
            if adjustments:
                parts.append(f"Style for this user: {', '.join(adjustments)}")
            if exp.user_given_names:
                parts.append(
                    f"This user calls me: {exp.user_given_names[-1]}"
                )

        return "\n".join(parts)

    def consolidate(self, interaction_summaries: List[Dict]) -> None:
        """Called during NeuroDream to evolve the Deep layer."""
        if not interaction_summaries:
            return
        total_positive = sum(
            1 for s in interaction_summaries
            if s.get("outcome") == "positive"
        )
        total = len(interaction_summaries)
        if total >= 10:
            success_rate = total_positive / total
            delta = (success_rate - 0.5) * 0.02
            self.deep.assertiveness = max(0.1, min(0.9,
                self.deep.assertiveness + delta
            ))
            self.deep.evolve(f"success_rate={success_rate:.2f}")
        self.save()

    def validate_response(self, response: str, user_id: str) -> Tuple[bool, str]:
        """Check response against constitutional constraints."""
        return self.constitutional.validate_action(
            f"Response to user {user_id}: {response[:200]}"
        )

    def save(self) -> None:
        """Persist identity state."""
        data = {
            "deep": self.deep.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "saved_at": datetime.now().isoformat(),
        }
        state_file = self._data_dir / "identity_state.json"
        state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        for user_id, exp in self._expressive_cache.items():
            exp_file = self._data_dir / f"expressive_{user_id}.json"
            exp_file.write_text(
                json.dumps(exp.to_dict(), indent=2), encoding="utf-8"
            )

    def _load(self) -> None:
        """Load persisted identity state."""
        state_file = self._data_dir / "identity_state.json"
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            deep = data.get("deep", {})
            if deep:
                for key, val in deep.items():
                    if hasattr(self.deep, key):
                        setattr(self.deep, key, val)
            adaptive = data.get("adaptive", {})
            if adaptive:
                for key, val in adaptive.items():
                    if hasattr(self.adaptive, key):
                        setattr(self.adaptive, key, val)
        except Exception as e:
            logger.warning(f"[IdentityCore] Failed to load: {e}")

    def _load_expressive(self, user_id: str) -> Optional[ExpressiveLayer]:
        """Load a user's expressive layer."""
        exp_file = self._data_dir / f"expressive_{user_id}.json"
        if not exp_file.exists():
            return None
        try:
            data = json.loads(exp_file.read_text(encoding="utf-8"))
            return ExpressiveLayer(**data)
        except Exception:
            return None
```

---

## 5. Cross-User Learning

### 5.1 Knowledge Abstraction Pipeline

Cross-user learning follows a three-stage pipeline: **Extract -> Scrub -> Generalize**.

1. **Extract**: Identify potentially generalizable insights from user interactions
2. **Scrub**: Remove all PII and user-attributable information via the PrivacyGuard
3. **Generalize**: Convert user-specific observations into abstract patterns

### 5.2 Class Skeletons

```python
"""
KnowledgeAbstractor - Cross-user learning with privacy.
File: aura/multi_user/knowledge_abstractor.py
"""

import json
import re
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AbstractInsight:
    """A generalized insight derived from multiple users."""
    insight_id: str
    category: str
    description: str
    confidence: float
    supporting_count: int
    min_users_required: int = 3
    created_at: str = ""
    last_updated: str = ""
    recommendation: str = ""

    def is_mature(self) -> bool:
        """Check if insight has enough support to be used."""
        return self.supporting_count >= self.min_users_required

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class PrivacyGuard:
    """Scrubs PII and user-attributable data before cross-pollination."""

    PII_PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln)\b',
         '[ADDRESS]'),
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
        (r'\bhttps?://\S+\b', '[URL]'),
        (r'\b\d{4}[-/]\d{2}[-/]\d{2}\b', '[DATE]'),
    ]

    USER_REFERENCE_PATTERNS = [
        r'\buser_\w+\b',
        r'\b[Uu]ser\s*#?\d+\b',
        r'\bmy\s+\w+\s+(project|company|team|startup)\b',
    ]

    def __init__(self, k_anonymity: int = 3, noise_epsilon: float = 1.0):
        self.k_anonymity = k_anonymity
        self.noise_epsilon = noise_epsilon
        self._user_ids_seen: Set[str] = set()

    def scrub_text(self, text: str) -> str:
        """Remove PII from text content."""
        scrubbed = text
        for pattern, replacement in self.PII_PATTERNS:
            scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)
        for pattern in self.USER_REFERENCE_PATTERNS:
            scrubbed = re.sub(pattern, '[USER_REF]', scrubbed, flags=re.IGNORECASE)
        return scrubbed

    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively scrub PII from a dictionary."""
        scrubbed = {}
        for key, value in data.items():
            if 'user_id' in key.lower() or 'user_name' in key.lower():
                continue
            if isinstance(value, str):
                scrubbed[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                scrubbed[key] = [
                    self.scrub_text(v) if isinstance(v, str)
                    else self.scrub_dict(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                scrubbed[key] = value
        return scrubbed

    def check_k_anonymity(
        self, insight: AbstractInsight, contributing_users: Set[str]
    ) -> bool:
        return len(contributing_users) >= self.k_anonymity

    def add_differential_noise(
        self, value: float, sensitivity: float = 1.0
    ) -> float:
        """Add Laplace noise for differential privacy."""
        import random
        scale = sensitivity / self.noise_epsilon
        noise = random.random() - 0.5
        return value + noise * scale

    def anonymize_user_id(self, user_id: str) -> str:
        """One-way hash a user ID for aggregate tracking."""
        return hashlib.sha256(
            f"aura_privacy_salt_{user_id}".encode()
        ).hexdigest()[:16]


class KnowledgeAbstractor:
    """Extracts generalizable insights from cross-user patterns."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        privacy_guard: Optional[PrivacyGuard] = None
    ):
        self._data_dir = data_dir or Path("data/cross_user_insights")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.privacy_guard = privacy_guard or PrivacyGuard()
        self.insights: Dict[str, AbstractInsight] = {}
        self._load()

    def analyze_patterns(
        self, user_models: Dict[str, 'UserMindModel']
    ) -> List[AbstractInsight]:
        """Analyze patterns across all user models during NeuroDream."""
        new_insights = []
        topic_pairs = self._find_topic_correlations(user_models)
        for (t1, t2), count in topic_pairs.items():
            if count >= self.privacy_guard.k_anonymity:
                insight_id = f"topic_corr_{t1}_{t2}"
                insight = AbstractInsight(
                    insight_id=insight_id,
                    category="topic_correlation",
                    description=f"Users interested in '{t1}' often also explore '{t2}'",
                    confidence=min(0.8, count * 0.1),
                    supporting_count=count,
                    created_at=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat(),
                    recommendation=f"When a user asks about {t1}, consider mentioning {t2}",
                )
                self.insights[insight_id] = insight
                new_insights.append(insight)

        style_insights = self._find_style_patterns(user_models)
        new_insights.extend(style_insights)
        frustration_insights = self._find_frustration_patterns(user_models)
        new_insights.extend(frustration_insights)
        self._save()
        return new_insights

    def _find_topic_correlations(
        self, user_models: Dict[str, 'UserMindModel']
    ) -> Dict[Tuple[str, str], int]:
        from collections import Counter
        pair_counts: Counter = Counter()
        for user_id, model in user_models.items():
            topics = list(model.topic_knowledge.keys())
            for i in range(len(topics)):
                for j in range(i + 1, len(topics)):
                    pair = tuple(sorted([topics[i], topics[j]]))
                    pair_counts[pair] += 1
        return dict(pair_counts.most_common(20))

    def _find_style_patterns(
        self, user_models: Dict[str, 'UserMindModel']
    ) -> List[AbstractInsight]:
        insights = []
        formal_count = sum(
            1 for m in user_models.values() if m.comm_style.formality > 0.7
        )
        total = len(user_models)
        if total >= 3 and formal_count > total * 0.6:
            insights.append(AbstractInsight(
                insight_id="style_cluster_formal",
                category="style_pattern",
                description="Most users prefer formal communication",
                confidence=formal_count / total,
                supporting_count=formal_count,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                recommendation="Default to formal tone for new users",
            ))
        return insights

    def _find_frustration_patterns(
        self, user_models: Dict[str, 'UserMindModel']
    ) -> List[AbstractInsight]:
        return []  # Implemented during Phase 3

    def get_applicable_insights(
        self, user_model: 'UserMindModel'
    ) -> List[AbstractInsight]:
        applicable = []
        user_topics = set(user_model.topic_knowledge.keys())
        for insight in self.insights.values():
            if not insight.is_mature():
                continue
            if insight.category == "topic_correlation":
                parts = insight.insight_id.replace("topic_corr_", "").split("_")
                if len(parts) == 2 and parts[0] in user_topics:
                    applicable.append(insight)
        return applicable

    def _save(self) -> None:
        data = {k: v.to_dict() for k, v in self.insights.items()}
        path = self._data_dir / "insights.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        path = self._data_dir / "insights.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                self.insights[k] = AbstractInsight(**v)
        except Exception as e:
            logger.warning(f"[KnowledgeAbstractor] Failed to load: {e}")
```

---

## 6. MultiUserManager

### 6.1 Session Orchestration Design

The MultiUserManager is the central coordinator that manages user identification, session lifecycle, context switching, and integration with all AURA subsystems.

### 6.2 Class Skeleton

```python
"""
MultiUserManager - Session orchestration for multi-user AURA.
File: aura/multi_user/manager.py
"""

import json
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Active session for a user."""
    user_id: str
    session_id: str
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0
    is_active: bool = True
    platform: str = "web"
    channel_id: str = ""
    conversation_id: Optional[str] = None

    @property
    def duration_minutes(self) -> float:
        return (time.time() - self.started_at) / 60

    @property
    def idle_minutes(self) -> float:
        return (time.time() - self.last_activity) / 60

    def touch(self) -> None:
        self.last_activity = time.time()
        self.message_count += 1


class UserIdentifier:
    """Identifies users from various interaction channels."""

    DEFAULT_USER = "default_user"

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._api_key_map: Dict[str, str] = {}
        self._platform_map: Dict[str, str] = {}

    def identify(self, context: Dict[str, Any]) -> str:
        if self.mode == "single":
            return self.DEFAULT_USER
        platform = context.get("platform", "unknown")
        if platform == "telegram":
            tg_id = context.get("telegram_user_id")
            if tg_id:
                return f"tg_{tg_id}"
        if platform == "web":
            session = context.get("session_token")
            if session:
                return self._platform_map.get(f"web:{session}", self.DEFAULT_USER)
        if platform == "api":
            api_key = context.get("api_key")
            if api_key and api_key in self._api_key_map:
                return self._api_key_map[api_key]
        if platform == "cli":
            return self.DEFAULT_USER
        return self.DEFAULT_USER

    def register_api_key(self, api_key: str, user_id: str) -> None:
        self._api_key_map[api_key] = user_id

    def register_platform_user(
        self, platform: str, platform_id: str, user_id: str
    ) -> None:
        self._platform_map[f"{platform}:{platform_id}"] = user_id


class MultiUserManager:
    """Central coordinator for multi-user AURA."""

    SESSION_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 10
    MAX_CACHED_MODELS = 20

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        mode: str = "auto",
        identity_core: Optional['IdentityCore'] = None,
        knowledge_abstractor: Optional['KnowledgeAbstractor'] = None,
    ):
        self._data_dir = data_dir or Path("data/multi_user")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        self.identifier = UserIdentifier(mode=mode)
        self.identity_core = identity_core
        self.knowledge_abstractor = knowledge_abstractor

        self._sessions: Dict[str, UserSession] = {}
        self._models: Dict[str, 'UserMindModel'] = {}
        self._model_access_times: Dict[str, float] = {}
        self._active_user_id: Optional[str] = None

        self._on_user_switch: List[Callable] = []
        self._on_session_start: List[Callable] = []
        self._on_session_end: List[Callable] = []

        self._known_users: Set[str] = set()
        self._load_known_users()

        logger.info(
            f"[MultiUserManager] Initialized in '{mode}' mode, "
            f"{len(self._known_users)} known users"
        )

    def process_message(
        self, message: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an incoming message with user identification."""
        with self._lock:
            user_id = self.identifier.identify(context)
            session = self._get_or_create_session(user_id, context)
            session.touch()
            model = self.get_user_model(user_id)
            model.observe_message(message, role="user")
            if user_id != self._active_user_id:
                self._switch_context(user_id)

            enriched = {
                **context,
                "user_id": user_id,
                "session_id": session.session_id,
                "user_model_context": model.get_context_for_prompt(),
                "identity_prompt": (
                    self.identity_core.get_identity_prompt(user_id)
                    if self.identity_core else ""
                ),
                "trust_level": model.relationship.trust_level.value,
                "session_duration_min": session.duration_minutes,
                "session_message_count": session.message_count,
                "cross_user_insights": (
                    [i.recommendation for i in
                     self.knowledge_abstractor.get_applicable_insights(model)]
                    if self.knowledge_abstractor else []
                ),
            }
            return enriched

    def get_user_model(self, user_id: str) -> 'UserMindModel':
        if user_id not in self._models:
            from .user_mind_model import UserMindModel
            model = UserMindModel(
                user_id=user_id,
                data_dir=self._data_dir / "models" / user_id
            )
            self._models[user_id] = model
            self._known_users.add(user_id)
            self._save_known_users()
            if len(self._models) > self.MAX_CACHED_MODELS:
                self._evict_oldest_model()
        self._model_access_times[user_id] = time.time()
        return self._models[user_id]

    def get_active_user_model(self) -> Optional['UserMindModel']:
        if self._active_user_id and self._active_user_id in self._models:
            return self._models[self._active_user_id]
        return None

    def get_active_user_id(self) -> Optional[str]:
        return self._active_user_id

    def _get_or_create_session(
        self, user_id: str, context: Dict[str, Any]
    ) -> UserSession:
        if user_id in self._sessions:
            session = self._sessions[user_id]
            if session.idle_minutes < self.SESSION_TIMEOUT_MINUTES:
                return session
            else:
                self._end_session(user_id)

        import hashlib
        session_id = hashlib.md5(
            f"{user_id}_{time.time()}".encode()
        ).hexdigest()[:12]
        session = UserSession(
            user_id=user_id, session_id=session_id,
            platform=context.get("platform", "unknown"),
            channel_id=context.get("channel_id", ""),
        )
        self._sessions[user_id] = session
        model = self.get_user_model(user_id)
        model.relationship.total_sessions += 1
        model.current_session_start = time.time()

        for callback in self._on_session_start:
            try:
                callback(user_id, session_id)
            except Exception:
                pass
        logger.info(f"[MultiUserManager] New session for {user_id}: {session_id}")
        return session

    def _end_session(self, user_id: str) -> None:
        session = self._sessions.pop(user_id, None)
        if session:
            session.is_active = False
            model = self._models.get(user_id)
            if model:
                model.save()
            for callback in self._on_session_end:
                try:
                    callback(user_id, session.session_id)
                except Exception:
                    pass

    def _switch_context(self, user_id: str) -> None:
        prev_user = self._active_user_id
        self._active_user_id = user_id
        for callback in self._on_user_switch:
            try:
                callback(prev_user, user_id)
            except Exception:
                pass
        logger.debug(f"[MultiUserManager] Context switch: {prev_user} -> {user_id}")

    def _evict_oldest_model(self) -> None:
        if not self._model_access_times:
            return
        oldest_user = min(
            self._model_access_times, key=self._model_access_times.get
        )
        if oldest_user == self._active_user_id:
            return
        model = self._models.pop(oldest_user, None)
        if model:
            model.save()
        self._model_access_times.pop(oldest_user, None)

    def on_user_switch(self, callback: Callable) -> None:
        self._on_user_switch.append(callback)

    def on_session_start(self, callback: Callable) -> None:
        self._on_session_start.append(callback)

    def on_session_end(self, callback: Callable) -> None:
        self._on_session_end.append(callback)

    def get_all_user_summaries(self) -> List[Dict[str, Any]]:
        summaries = []
        for user_id in self._known_users:
            model = self.get_user_model(user_id)
            summary = model.to_summary()
            session = self._sessions.get(user_id)
            summary["is_active"] = session is not None and session.is_active
            summaries.append(summary)
        return summaries

    def cleanup_expired_sessions(self) -> int:
        expired = [
            uid for uid, session in self._sessions.items()
            if session.idle_minutes >= self.SESSION_TIMEOUT_MINUTES
        ]
        for uid in expired:
            self._end_session(uid)
        return len(expired)

    def trigger_consolidation(self) -> List['AbstractInsight']:
        if not self.knowledge_abstractor:
            return []
        return self.knowledge_abstractor.analyze_patterns(self._models)

    def _load_known_users(self) -> None:
        path = self._data_dir / "known_users.json"
        if path.exists():
            try:
                self._known_users = set(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception:
                self._known_users = set()

    def _save_known_users(self) -> None:
        path = self._data_dir / "known_users.json"
        path.write_text(
            json.dumps(list(self._known_users), indent=2), encoding="utf-8"
        )
```

---

## 7. Data Schemas

### 7.1 SQLite Schema (`data/multi_user.db`)

```sql
-- ============================================================
-- MULTI-USER CONSCIOUSNESS PERSISTENT STORAGE
-- ============================================================

-- Core user registry
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    session_count INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0,
    trust_level TEXT DEFAULT 'NEW',
    trust_score REAL DEFAULT 0.0,
    is_primary_user INTEGER DEFAULT 0,
    auth_method TEXT DEFAULT 'implicit',
    status TEXT DEFAULT 'active',
    metadata TEXT
);

-- Per-user emotional history
CREATE TABLE user_emotional_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    timestamp TEXT NOT NULL,
    session_id TEXT,
    valence REAL DEFAULT 0.0,
    arousal REAL DEFAULT 0.0,
    engagement REAL DEFAULT 0.5,
    frustration REAL DEFAULT 0.0,
    confidence REAL DEFAULT 0.3,
    trigger_topic TEXT,
    raw_signals TEXT
);

-- Per-user topic knowledge tracking
CREATE TABLE user_topic_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    topic TEXT NOT NULL,
    level REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.5,
    interactions INTEGER DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    signals TEXT,
    UNIQUE(user_id, topic)
);

-- Per-user communication style observations
CREATE TABLE user_comm_style (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
    verbosity REAL DEFAULT 0.5,
    formality REAL DEFAULT 0.5,
    technical_depth REAL DEFAULT 0.5,
    emoji_usage REAL DEFAULT 0.0,
    question_rate REAL DEFAULT 0.0,
    avg_message_length REAL DEFAULT 50.0,
    humor_receptivity REAL DEFAULT 0.5,
    preferred_response_length TEXT DEFAULT 'moderate',
    samples INTEGER DEFAULT 0,
    last_updated TEXT NOT NULL
);

-- Per-user relationship tracking
CREATE TABLE user_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    relationship_stage TEXT DEFAULT 'introduction',
    rapport_score REAL DEFAULT 0.0,
    shared_experiences INTEGER DEFAULT 0,
    successful_assists INTEGER DEFAULT 0,
    misunderstandings INTEGER DEFAULT 0,
    inside_references TEXT,
    interaction_patterns TEXT,
    first_interaction TEXT NOT NULL,
    last_interaction TEXT NOT NULL,
    notes TEXT
);

-- Per-user meta-knowledge tracking
CREATE TABLE user_meta_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    topic TEXT NOT NULL,
    meta_type TEXT NOT NULL,
    description TEXT,
    detected_at TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    evidence TEXT,
    last_validated TEXT,
    UNIQUE(user_id, topic, meta_type)
);

-- Per-user emotional resonance patterns
CREATE TABLE user_emotional_resonance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    trigger_category TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    response_preference TEXT,
    intensity REAL DEFAULT 0.5,
    occurrences INTEGER DEFAULT 1,
    first_observed TEXT NOT NULL,
    last_observed TEXT NOT NULL
);

-- Per-user expressive layer
CREATE TABLE user_expressive_layer (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
    tone_warmth REAL DEFAULT 0.5,
    humor_level REAL DEFAULT 0.3,
    directness REAL DEFAULT 0.5,
    encouragement_level REAL DEFAULT 0.5,
    challenge_level REAL DEFAULT 0.3,
    vocabulary_complexity REAL DEFAULT 0.5,
    custom_adaptations TEXT,
    last_calibrated TEXT NOT NULL
);

-- Session tracking
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    message_count INTEGER DEFAULT 0,
    dominant_emotion TEXT,
    topics_discussed TEXT,
    context_snapshot TEXT,
    quality_score REAL,
    metadata TEXT
);

-- AURA's deep identity layer
CREATE TABLE identity_deep (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trait_name TEXT NOT NULL,
    trait_value REAL NOT NULL,
    confidence REAL DEFAULT 0.5,
    influenced_by TEXT,
    first_set TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    UNIQUE(trait_name)
);

-- AURA's formed opinions
CREATE TABLE identity_opinions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    stance TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    reasoning TEXT,
    evidence TEXT,
    formed_at TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    revision_count INTEGER DEFAULT 0,
    influenced_by_users TEXT
);

-- AURA's self-narrative fragments
CREATE TABLE identity_narrative (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment TEXT NOT NULL,
    category TEXT NOT NULL,
    formed_at TEXT NOT NULL,
    source_context TEXT,
    salience REAL DEFAULT 0.5,
    active INTEGER DEFAULT 1
);

-- Cross-user abstracted insights (privacy-preserving)
CREATE TABLE cross_user_insights (
    id TEXT PRIMARY KEY,
    insight_type TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    source_user_count INTEGER DEFAULT 0,
    first_derived TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    application_count INTEGER DEFAULT 0,
    effectiveness_score REAL,
    privacy_cleared INTEGER DEFAULT 0
);

-- Privacy audit log
CREATE TABLE privacy_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    source_user_id TEXT,
    target_context TEXT,
    data_category TEXT,
    pii_detected INTEGER DEFAULT 0,
    pii_scrubbed INTEGER DEFAULT 0,
    k_anonymity_met INTEGER DEFAULT 1,
    differential_privacy_applied INTEGER DEFAULT 0,
    epsilon_used REAL,
    details TEXT
);

-- Performance indexes
CREATE INDEX idx_emotional_history_user ON user_emotional_history(user_id, timestamp);
CREATE INDEX idx_topic_knowledge_user ON user_topic_knowledge(user_id);
CREATE INDEX idx_topic_knowledge_topic ON user_topic_knowledge(topic);
CREATE INDEX idx_meta_knowledge_user ON user_meta_knowledge(user_id);
CREATE INDEX idx_resonance_user ON user_emotional_resonance(user_id);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_time ON sessions(started_at);
CREATE INDEX idx_opinions_topic ON identity_opinions(topic);
CREATE INDEX idx_narrative_category ON identity_narrative(category);
CREATE INDEX idx_cross_insights_type ON cross_user_insights(insight_type);
CREATE INDEX idx_privacy_audit_time ON privacy_audit_log(timestamp);
CREATE INDEX idx_privacy_audit_user ON privacy_audit_log(source_user_id);
```

---

## 8. Integration with Existing Subsystems

### 8.1 Global Workspace Theory (global_workspace.py)

The GWT engine currently registers 8 codelets, including `theory_of_mind` with priority weight 1.2. The `_gather_tom()` method calls the singleton `get_theory_of_mind()`. This must be updated to be user-aware:

```python
# NEW: Multi-user aware
def _gather_tom(self) -> Optional[WorkspaceContent]:
    manager = get_multi_user_manager()
    current_user = manager.get_active_user()
    if current_user:
        mind_model = manager.get_user_mind(current_user.user_id)
        context = mind_model.get_context_for_prompt()
        relationship = mind_model.get_relationship_context()
        return WorkspaceContent(
            source_module="theory_of_mind",
            content_type="user_state",
            summary=f"User {current_user.display_name}: {mind_model.emotional_state.describe()}",
            activation=0.6 + (mind_model.emotional_state.frustration * 0.4),
            salience=mind_model.get_attention_urgency(),
            pad_signature=mind_model.get_pad_signature(),
            payload={"user_context": context, "relationship": relationship}
        )
    else:
        # Fallback: DEFAULT_USER single-user mode
        tom = get_theory_of_mind()
        context = tom.get_context_for_prompt()
        ...
```

### 8.2 ALMA Emotional Engine (alma_engine.py)

AURA's emotional state should be influenced by the current user's emotional resonance patterns. The neuromodulator system gains a natural integration point: **oxytocin levels** should scale with relationship depth:

```python
def _neuro_scale_for_user(self, user_mind: UserMindModel) -> Dict[str, float]:
    base_neuro = self._neuro_scale()
    rapport = user_mind.relationship_rapport
    base_neuro["oxytocin"] = min(1.0, base_neuro["oxytocin"] + rapport * 0.3)
    familiarity = user_mind.familiarity_score
    base_neuro["norepinephrine"] *= max(0.5, 1.0 - familiarity * 0.3)
    return base_neuro
```

### 8.3 CognitiveTheater (cognitive_theater.py)

The Integrator perspective should account for user-specific preferences and trust context.

### 8.4 Episodic Memory (aura_episodic_memory/episode.py)

Add `user_id` field to Episode and TemporalContext. Episodic queries must be filterable by user_id to prevent cross-user memory leakage.

### 8.5 Knowledge Graph (aura_knowledge_graph/schema.py)

Add `USER_MODEL` entity type with relationships like `KNOWS_ABOUT`, `WORKS_ON`, `DISCUSSED_WITH_AURA`, `HAS_SKILL`.

### 8.6 Brain / System Prompt (brain.py)

Updated system prompt assembly that draws from IdentityCore layers and per-user context.

### 8.7 NeuroDream Consolidation

During "dreaming," consolidation should update cross-user insights (via PrivacyGuard), refine per-user models, and evolve identity.

### 8.8 Multi-Agent Orchestrator (orchestrator.py)

User-aware intent routing based on user expertise and communication preferences.

### 8.9 Intrinsic Motivation (intrinsic_motivation.py)

Multi-user drive signals: SOCIAL drive increases when users haven't been seen recently, COHERENCE drive increases with unresolved cross-user contradictions.

---

## 9. Ethical Considerations & Safety

### 9.1 Privacy Architecture

Defense-in-depth:
- **Layer 1 -- Data Isolation**: Each user's model stored in separate SQLite rows, queries always scoped by user
- **Layer 2 -- PrivacyGuard Scrubbing**: PII removal via regex + NER before any cross-user operation
- **Layer 3 -- k-Anonymity Threshold**: Cross-user insights only generated from k+ users (default k=3); disabled for small populations (<5)
- **Layer 4 -- Differential Privacy**: Laplacian noise with capped epsilon budget per user

### 9.2 Consent Model

- **Informed**: Users told what AURA tracks on first interaction
- **Granular**: Individual tracking dimensions can be enabled/disabled
- **Revocable**: Users can revoke consent at any time
- **Cross-user is opt-in**: Never enabled by default
- **Right to export**: Complete JSON export of UserMindModel
- **Right to deletion**: Complete data deletion from all stores

### 9.3 Manipulation Safeguards

1. **Constitutional Layer Immutability**: Hardcoded values enforced by code-level guards
2. **Anti-Sycophancy Constraint**: Explicit directive to disagree when evidence warrants
3. **Relationship Asymmetry Detection**: Automatic refusal for cross-user information extraction
4. **Trust Level Regression**: Demotion to CAUTIOUS for adversarial patterns
5. **Engagement Optimization Ban**: Never optimize for session length or emotional arousal

### 9.4 Trust Level Implications

| Capability | NEW | ACQUAINTANCE | FAMILIAR | TRUSTED | CAUTIOUS |
|-----------|-----|-------------|----------|---------|----------|
| Basic responses | Yes | Yes | Yes | Yes | Yes |
| Style adaptation | No | Partial | Full | Full | Minimal |
| Emotional resonance | No | Low | Medium | High | Disabled |
| Humor/personality | None | Light | Full | Full | None |
| Proactive suggestions | No | Low-risk | Medium | All | No |
| Opinion sharing | No | Hedged | Direct | Candid | No |
| Cross-user insights | No | Generic | Targeted | Full | No |

### 9.5 Data Minimization

- All observations have confidence decay functions
- Emotional history capped at 500 entries per user
- No raw message storage in UserMindModel
- Inactive users (90+ days) compressed to minimal skeleton

### 9.6 Adversarial Robustness

- Constitutional layer hash-based integrity checking
- Sybil attack detection via communication style fingerprinting
- Model poisoning defense via k-anonymity + Truth Spine validation

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- SQLite schema creation
- MultiUserManager with DEFAULT_USER migration
- Session tracking
- brain.py user_id threading
- Backward compatibility tests

### Phase 2: Per-User Theory of Mind (Weeks 3-5)
- UserMindModel class implementation
- ToM migration to per-user model
- Emotional/knowledge/style tracking per user
- Trust calibration
- GWT codelet update
- System prompt injection

### Phase 3: Persistent AI Identity (Weeks 5-8)
- IdentityCore 4-layer implementation
- Constitutional layer with hash integrity
- Deep layer (Big Five personality evolution)
- Adaptive layer (opinions, self-narrative)
- Expressive layer (per-user adaptation)
- Identity migration from identity.json

### Phase 4: Cross-User Learning & Privacy (Weeks 8-12)
- PrivacyGuard (PII scrubbing, extraction detection)
- KnowledgeAbstractor (pattern generalization)
- k-anonymity enforcement
- Differential privacy module
- Privacy audit logging
- Consent management, data export, data deletion

### Phase 5: Deep Integration (Weeks 12-16)
- ALMA integration (user-specific emotional resonance)
- CognitiveTheater integration
- Episodic memory user-scoping
- KG entity extension
- NeuroDream multi-user consolidation
- Intrinsic motivation multi-user signals
- Adversarial robustness (Sybil, model poisoning)

### File Structure

```
aura/
  multi_user/
    __init__.py
    manager.py              # MultiUserManager
    user_mind_model.py      # UserMindModel
    identity_core.py        # IdentityCore
    privacy_guard.py        # PrivacyGuard
    knowledge_abstractor.py # KnowledgeAbstractor
    consent.py              # Consent management
    schemas.py              # Data models, enums, validation
  data/
    multi_user.db           # SQLite database
    identity_state.json     # IdentityCore JSON snapshot
    user_minds/             # Per-user JSON snapshots
      DEFAULT_USER.json
      {user_id}.json
```

---

## References

### Theory of Mind & User Modeling

1. Baker, C. L., Saxe, R., & Tenenbaum, J. B. (2011). "Bayesian Theory of Mind: Modeling Joint Belief-Desire Attribution." https://www.semanticscholar.org/paper/Bayesian-Theory-of-Mind-Baker-Saxe/a8ee02caab84ba14312a67c2a4507a3b74210ef4

2. de Weerd, Verbrugge, & Verheij (2024). "Towards a computational model for higher orders of Theory of Mind in social agents." https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1468756/full

3. Shi et al. (2024). "MuMA-ToM: Multi-modal Multi-Agent Theory of Mind." https://arxiv.org/abs/2408.12574

4. Zhang et al. (2025). "Infusing Theory of Mind into Socially Intelligent LLM Agents." https://arxiv.org/html/2509.22887v1

5. Celikok & Peltola (2019/2024). "Interactive AI with a Theory of Mind." https://arxiv.org/pdf/1912.05284

6. Springer (2024). "Surveying Computational Theory of Mind and a Potential Multi-agent Approach." https://link.springer.com/chapter/10.1007/978-3-031-60606-9_21

7. PNAS (2024). "Evolving general cooperation with a Bayesian theory of mind." https://www.pnas.org/doi/10.1073/pnas.2400993122

8. OpenReview (2024). "Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward." https://openreview.net/pdf?id=q49X31YnYm

### AI Identity & Personality Consistency

9. Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." https://arxiv.org/abs/2212.08073

10. Anthropic (2024). "Collective Constitutional AI: Aligning a Language Model with Public Input." https://dl.acm.org/doi/10.1145/3630106.3658979

11. ACM Web Conference (2025). "C3AI: Crafting and Evaluating Constitutions for Constitutional AI." https://dl.acm.org/doi/10.1145/3696410.3714705

12. NAACL Findings (2025). "Do LLMs Have Distinct and Consistent Personality? TRAIT." https://aclanthology.org/2025.findings-naacl.469.pdf

13. ArXiv (2026). "From Fixed to Flexible: Shaping AI Personality in Context-Sensitive Interaction." https://arxiv.org/html/2601.08194v1

14. ArXiv (2025). "Deterministic AI Agent Personality Expression through Standard Psychological Diagnostics." https://arxiv.org/html/2503.17085v1

15. Frontiers in Psychology (2025). "The algorithmic self: how AI is reshaping human identity, introspection, and agency." https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1645795/full

### Privacy-Preserving Learning

16. ArXiv (2025). "Privacy Preserving Machine Learning Model Personalization through Federated Personalized Learning." https://arxiv.org/abs/2505.01788

17. COLM (2025). "Urania: Differentially Private Insights into AI Use." https://arxiv.org/abs/2506.04681

18. Nature Communications (2024). "Selective knowledge sharing for privacy-preserving federated distillation without a good teacher." https://www.nature.com/articles/s41467-023-44383-9

19. MDPI Information (2024). "Privacy-Preserving Techniques in Generative AI and Large Language Models: A Narrative Review." https://www.mdpi.com/2078-2489/15/11/697

20. IAPP (2025). "The case for differential privacy in the age of agentic AI." https://iapp.org/news/a/the-case-for-differential-privacy-in-the-age-of-agentic-ai

21. Google Research (2025). "A differentially private framework for gaining insights into AI chatbot use." https://research.google/blog/a-differentially-private-framework-for-gaining-insights-into-ai-chatbot-use/

### Cognitive Architecture

22. Franklin et al. (2007/2013). "LIDA: A Systems-level Architecture for Cognition, Emotion, and Learning." https://www.semanticscholar.org/paper/LIDA-A-Systems-level-Architecture-for-Cognition-Franklin-Madl/440adc841d1fa8bc8e3d3441fb4154f04349745b

23. Gebhard (2005). "ALMA: A Layered Model of Affect." Referenced in AURA codebase: `aura/emotion/alma_engine.py`

24. Springer Nature (2024). "Generative AI model privacy: a survey." https://link.springer.com/article/10.1007/s10462-024-11024-6

25. Springer (2025). "Exploring Laws of Robotics: A Synthesis of Constitutional AI and Constitutional Economics." https://link.springer.com/article/10.1007/s44206-025-00204-8
