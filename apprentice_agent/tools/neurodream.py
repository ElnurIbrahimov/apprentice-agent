"""NeuroDream - Sleep/Dream Memory Consolidation System for Aura.

Implements a biologically-inspired sleep system with phases:
- Light Sleep: Recent memory replay (last 24h)
- Deep Sleep: Pattern abstraction and compression
- REM Sleep: Creative synthesis and novel connections

Based on research showing 38% reduction in catastrophic forgetting
and 17.6% increase in zero-shot transfer through latent replay synthesis.
"""

import json
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import re


class SleepPhase(Enum):
    """Sleep cycle phases."""
    AWAKE = "awake"
    LIGHT = "light"      # Recent memory replay
    DEEP = "deep"        # Pattern abstraction
    REM = "rem"          # Creative synthesis
    WAKING = "waking"    # Transitioning to awake


class DreamTrigger(Enum):
    """What triggered the sleep cycle."""
    SCHEDULED = "scheduled"
    IDLE = "idle"
    MANUAL = "manual"
    LOW_RESOURCES = "low_resources"


@dataclass
class DreamInsight:
    """A novel insight generated during REM sleep."""
    id: str
    timestamp: str
    insight_type: str  # connection, pattern, hypothesis, prediction
    content: str
    confidence: float
    source_nodes: List[str]
    created_edges: List[Dict[str, str]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SleepSession:
    """Record of a complete sleep cycle."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    trigger: str
    phases_completed: List[str]

    # Light phase metrics
    memories_replayed: int = 0
    memories_strengthened: int = 0

    # Deep phase metrics
    patterns_found: int = 0
    edges_pruned: int = 0
    edges_strengthened: int = 0
    nodes_merged: int = 0

    # REM phase metrics
    insights_generated: int = 0
    novel_connections: int = 0
    creative_hypotheses: int = 0

    # Overall
    duration_seconds: float = 0
    interrupted: bool = False
    interrupt_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsolidatedPattern:
    """An abstracted pattern from deep sleep."""
    pattern_id: str
    timestamp: str
    pattern_type: str  # temporal, topical, emotional, behavioral
    description: str
    frequency: int
    confidence: float
    examples: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class NeuroDreamEngine:
    """Sleep/dream memory consolidation engine for Aura.

    Runs during idle periods to:
    - Replay and strengthen recent memories
    - Find and abstract patterns
    - Generate creative connections
    - Consolidate emotional experiences
    """

    def __init__(
        self,
        knowledge_graph=None,
        hybrid_memory=None,
        evoemo=None,
        inner_monologue=None,
        chromadb=None,
        data_dir: str = "data/neurodream",
        idle_threshold_minutes: int = 30,
        max_vram_gb: float = 4.0
    ):
        """Initialize NeuroDream engine.

        Args:
            knowledge_graph: KnowledgeGraphTool instance
            hybrid_memory: HybridMemory instance
            evoemo: EvoEmoTool instance
            inner_monologue: InnerMonologueTool instance
            chromadb: ChromaDB collection for memories
            data_dir: Directory for dream data
            idle_threshold_minutes: Minutes of inactivity before auto-sleep
            max_vram_gb: Maximum VRAM to use during sleep
        """
        self.kg = knowledge_graph
        self.memory = hybrid_memory
        self.evoemo = evoemo
        self.monologue = inner_monologue
        self.chromadb = chromadb

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "consolidated").mkdir(exist_ok=True)

        self.idle_threshold = timedelta(minutes=idle_threshold_minutes)
        self.max_vram_gb = max_vram_gb

        # State
        self.current_phase = SleepPhase.AWAKE
        self.current_session: Optional[SleepSession] = None
        self.last_activity_time = datetime.now()
        self.last_sleep_time: Optional[datetime] = None

        # Threading
        self._sleep_thread: Optional[threading.Thread] = None
        self._interrupt_flag = threading.Event()
        self._phase_lock = threading.Lock()

        # Callbacks
        self._on_phase_change: Optional[Callable[[SleepPhase], None]] = None
        self._on_insight: Optional[Callable[[DreamInsight], None]] = None

        # Load previous insights count
        self._total_insights = self._count_insights()
        self._total_sessions = self._count_sessions()

    def _count_insights(self) -> int:
        """Count total insights generated."""
        insights_file = self.data_dir / "insights.jsonl"
        if not insights_file.exists():
            return 0
        count = 0
        with open(insights_file, 'r') as f:
            for _ in f:
                count += 1
        return count

    def _count_sessions(self) -> int:
        """Count total sleep sessions."""
        journal_file = self.data_dir / "dream_journal.jsonl"
        if not journal_file.exists():
            return 0
        count = 0
        with open(journal_file, 'r') as f:
            for _ in f:
                count += 1
        return count

    def _generate_id(self, prefix: str = "dream") -> str:
        """Generate unique ID."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = random.randint(1000, 9999)
        return f"{prefix}_{ts}_{rand}"

    def record_activity(self):
        """Record user activity (resets idle timer)."""
        self.last_activity_time = datetime.now()

        # If sleeping, wake up
        if self.current_phase != SleepPhase.AWAKE:
            self.wake_up("user_activity")

    def check_idle_trigger(self) -> bool:
        """Check if idle threshold reached for auto-sleep."""
        if self.current_phase != SleepPhase.AWAKE:
            return False

        idle_duration = datetime.now() - self.last_activity_time
        return idle_duration >= self.idle_threshold

    def get_status(self) -> Dict[str, Any]:
        """Get current sleep status."""
        return {
            "phase": self.current_phase.value,
            "is_sleeping": self.current_phase != SleepPhase.AWAKE,
            "last_activity": self.last_activity_time.isoformat(),
            "last_sleep": self.last_sleep_time.isoformat() if self.last_sleep_time else None,
            "total_sessions": self._total_sessions,
            "total_insights": self._total_insights,
            "idle_minutes": (datetime.now() - self.last_activity_time).total_seconds() / 60,
            "current_session": self.current_session.to_dict() if self.current_session else None
        }

    def enter_sleep(self, trigger: str = "manual") -> Dict[str, Any]:
        """Enter sleep mode and begin dream cycle.

        Args:
            trigger: What triggered the sleep (manual, idle, scheduled)

        Returns:
            Status dict with session info
        """
        if self.current_phase != SleepPhase.AWAKE:
            return {
                "success": False,
                "error": f"Already in {self.current_phase.value} phase"
            }

        # Create new session
        self.current_session = SleepSession(
            session_id=self._generate_id("session"),
            start_time=datetime.now().isoformat(),
            end_time=None,
            trigger=trigger,
            phases_completed=[]
        )

        # Log to inner monologue if available
        if self.monologue:
            self.monologue.think(
                "reflect",
                f"Entering sleep mode (trigger: {trigger}). Beginning memory consolidation...",
                confidence=100
            )

        # Start sleep thread
        self._interrupt_flag.clear()
        self._sleep_thread = threading.Thread(target=self._run_sleep_cycle, daemon=True)
        self._sleep_thread.start()

        return {
            "success": True,
            "session_id": self.current_session.session_id,
            "message": f"Entering sleep mode (trigger: {trigger})"
        }

    def _run_sleep_cycle(self):
        """Run the complete sleep cycle in background thread."""
        try:
            # Phase 1: Light Sleep (recent replay)
            if not self._interrupt_flag.is_set():
                self._set_phase(SleepPhase.LIGHT)
                light_results = self.run_light_phase()
                if self.current_session:
                    self.current_session.memories_replayed = light_results.get("memories_replayed", 0)
                    self.current_session.memories_strengthened = light_results.get("memories_strengthened", 0)
                    self.current_session.phases_completed.append("light")

            # Phase 2: Deep Sleep (pattern abstraction)
            if not self._interrupt_flag.is_set():
                self._set_phase(SleepPhase.DEEP)
                deep_results = self.run_deep_phase()
                if self.current_session:
                    self.current_session.patterns_found = deep_results.get("patterns_found", 0)
                    self.current_session.edges_pruned = deep_results.get("edges_pruned", 0)
                    self.current_session.edges_strengthened = deep_results.get("edges_strengthened", 0)
                    self.current_session.nodes_merged = deep_results.get("nodes_merged", 0)
                    self.current_session.phases_completed.append("deep")

            # Phase 3: REM Sleep (creative synthesis)
            if not self._interrupt_flag.is_set():
                self._set_phase(SleepPhase.REM)
                rem_results = self.run_rem_phase()
                if self.current_session:
                    self.current_session.insights_generated = rem_results.get("insights_generated", 0)
                    self.current_session.novel_connections = rem_results.get("novel_connections", 0)
                    self.current_session.creative_hypotheses = rem_results.get("creative_hypotheses", 0)
                    self.current_session.phases_completed.append("rem")

            # Natural wake up
            if not self._interrupt_flag.is_set():
                self.wake_up("cycle_complete")

        except Exception as e:
            print(f"[NeuroDream] Error during sleep cycle: {e}")
            self.wake_up(f"error: {str(e)}")

    def _set_phase(self, phase: SleepPhase):
        """Set current phase with thread safety."""
        with self._phase_lock:
            self.current_phase = phase
            if self._on_phase_change:
                try:
                    self._on_phase_change(phase)
                except:
                    pass

    def run_light_phase(self) -> Dict[str, Any]:
        """Light Sleep: Replay recent memories (last 24h).

        Returns:
            Dict with phase results
        """
        results = {
            "memories_replayed": 0,
            "memories_strengthened": 0,
            "duration_seconds": 0
        }

        start_time = time.time()

        # Get recent memories from ChromaDB (with timeout protection)
        try:
            recent_memories = self._get_recent_memories(hours=24)
        except Exception as e:
            print(f"[NeuroDream] Error getting memories: {e}")
            recent_memories = []

        results["memories_replayed"] = len(recent_memories)

        # Quick return if no memories to process
        if not recent_memories:
            results["duration_seconds"] = time.time() - start_time
            self._log_dream("light", "Light sleep: No recent memories to consolidate.")
            return results

        # Replay and strengthen in Knowledge Graph (limit to 50 for speed)
        for memory in recent_memories[:50]:
            if self._interrupt_flag.is_set():
                break

            # Extract entities and strengthen their connections
            strengthened = self._strengthen_memory_connections(memory)
            results["memories_strengthened"] += strengthened

            # Minimal delay
            time.sleep(0.01)

        results["duration_seconds"] = time.time() - start_time

        # Log dream thought
        if self.monologue and not self._interrupt_flag.is_set():
            self.monologue.think(
                "recall",
                f"Light sleep complete. Replayed {results['memories_replayed']} memories, "
                f"strengthened {results['memories_strengthened']} connections.",
                confidence=85
            )

        return results

    def run_deep_phase(self) -> Dict[str, Any]:
        """Deep Sleep: Pattern abstraction and memory compression.

        Returns:
            Dict with phase results
        """
        results = {
            "patterns_found": 0,
            "edges_pruned": 0,
            "edges_strengthened": 0,
            "nodes_merged": 0,
            "duration_seconds": 0
        }

        start_time = time.time()
        temporal_patterns = []
        topical_patterns = []
        emotional_patterns = []

        # Find temporal patterns (when does user ask what)
        try:
            temporal_patterns = self._find_temporal_patterns()
            results["patterns_found"] += len(temporal_patterns)
        except Exception as e:
            print(f"[NeuroDream] Temporal patterns error: {e}")

        # Find topical patterns (recurring themes)
        if not self._interrupt_flag.is_set():
            try:
                topical_patterns = self._find_topical_patterns()
                results["patterns_found"] += len(topical_patterns)
            except Exception as e:
                print(f"[NeuroDream] Topical patterns error: {e}")

        # Find emotional patterns from EvoEmo
        if not self._interrupt_flag.is_set() and self.evoemo:
            try:
                emotional_patterns = self._find_emotional_patterns()
                results["patterns_found"] += len(emotional_patterns)
            except Exception as e:
                print(f"[NeuroDream] Emotional patterns error: {e}")

        # Prune weak edges in Knowledge Graph
        if not self._interrupt_flag.is_set() and self.kg:
            try:
                pruned = self._prune_weak_edges()
                results["edges_pruned"] = pruned
            except Exception as e:
                print(f"[NeuroDream] Prune edges error: {e}")

        # Strengthen frequently used edges
        if not self._interrupt_flag.is_set() and self.kg:
            try:
                strengthened = self._strengthen_frequent_edges()
                results["edges_strengthened"] = strengthened
            except Exception as e:
                print(f"[NeuroDream] Strengthen edges error: {e}")

        # Merge similar nodes
        if not self._interrupt_flag.is_set() and self.kg:
            try:
                merged = self._merge_similar_nodes()
                results["nodes_merged"] = merged
            except Exception as e:
                print(f"[NeuroDream] Merge nodes error: {e}")

        results["duration_seconds"] = time.time() - start_time

        # Save consolidated patterns
        all_patterns = temporal_patterns + topical_patterns + emotional_patterns
        self._save_consolidated_patterns(all_patterns)

        # Log dream thought
        if self.monologue and not self._interrupt_flag.is_set():
            self.monologue.think(
                "reason",
                f"Deep sleep complete. Found {results['patterns_found']} patterns, "
                f"pruned {results['edges_pruned']} weak edges, "
                f"merged {results['nodes_merged']} similar nodes.",
                confidence=80
            )

        return results

    def run_rem_phase(self) -> Dict[str, Any]:
        """REM Sleep: Creative synthesis and novel connections.

        Returns:
            Dict with phase results
        """
        results = {
            "insights_generated": 0,
            "novel_connections": 0,
            "creative_hypotheses": 0,
            "duration_seconds": 0
        }

        start_time = time.time()
        insights = []

        # Generate novel connections between distant concepts
        if self.kg and not self._interrupt_flag.is_set():
            novel_insights = self._generate_novel_connections()
            insights.extend(novel_insights)
            results["novel_connections"] = len(novel_insights)

        # Generate creative hypotheses
        if not self._interrupt_flag.is_set():
            hypotheses = self._generate_hypotheses()
            insights.extend(hypotheses)
            results["creative_hypotheses"] = len(hypotheses)

        # Save insights
        for insight in insights:
            if self._interrupt_flag.is_set():
                break
            self._save_insight(insight)
            results["insights_generated"] += 1

            # Callback if set
            if self._on_insight:
                try:
                    self._on_insight(insight)
                except:
                    pass

        self._total_insights += results["insights_generated"]
        results["duration_seconds"] = time.time() - start_time

        # Log dream thought
        if self.monologue and not self._interrupt_flag.is_set():
            self.monologue.think(
                "eureka",
                f"REM sleep complete. Generated {results['insights_generated']} insights, "
                f"including {results['novel_connections']} novel connections.",
                confidence=75
            )

        return results

    def wake_up(self, reason: str = "manual") -> Dict[str, Any]:
        """Wake up from sleep and return summary.

        Args:
            reason: Why waking up (cycle_complete, user_activity, manual, error)

        Returns:
            Sleep session summary
        """
        # Set interrupt flag to stop any running phases
        self._interrupt_flag.set()

        # Wait for sleep thread to finish
        if self._sleep_thread and self._sleep_thread.is_alive():
            self._sleep_thread.join(timeout=5)

        self._set_phase(SleepPhase.WAKING)

        # Finalize session
        summary = {}
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()

            start = datetime.fromisoformat(self.current_session.start_time)
            end = datetime.fromisoformat(self.current_session.end_time)
            self.current_session.duration_seconds = (end - start).total_seconds()

            if reason not in ["cycle_complete", "manual"]:
                self.current_session.interrupted = True
                self.current_session.interrupt_reason = reason

            # Save to journal
            self._save_session(self.current_session)
            self._total_sessions += 1

            summary = self.current_session.to_dict()
            self.current_session = None

        self.last_sleep_time = datetime.now()
        self._set_phase(SleepPhase.AWAKE)

        # Log wake up
        if self.monologue:
            phases = summary.get("phases_completed", [])
            self.monologue.think(
                "perceive",
                f"Waking up (reason: {reason}). Completed phases: {phases}. "
                f"Generated {summary.get('insights_generated', 0)} insights.",
                confidence=100
            )

        return {
            "success": True,
            "reason": reason,
            "summary": summary
        }

    # ==================== Memory Retrieval ====================

    def _get_recent_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get memories from the last N hours."""
        memories = []

        # From ChromaDB if available
        if self.chromadb:
            try:
                cutoff = datetime.now() - timedelta(hours=hours)
                # Query recent memories
                results = self.chromadb.query(
                    query_texts=["recent conversation memory"],
                    n_results=100,
                    where={"timestamp": {"$gte": cutoff.isoformat()}} if hasattr(self.chromadb, 'query') else None
                )
                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        memories.append({
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                            "id": results["ids"][0][i] if results.get("ids") else f"mem_{i}"
                        })
            except Exception as e:
                print(f"[NeuroDream] Error getting ChromaDB memories: {e}")

        # From hybrid memory if available
        if self.memory and hasattr(self.memory, 'recall'):
            try:
                recent = self.memory.recall("recent conversations", limit=50)
                for mem in recent:
                    memories.append({
                        "content": mem.get("content", str(mem)),
                        "metadata": mem.get("metadata", {}),
                        "id": mem.get("id", self._generate_id("mem"))
                    })
            except Exception as e:
                print(f"[NeuroDream] Error getting hybrid memories: {e}")

        # From inner monologue logs
        monologue_memories = self._get_monologue_memories(hours)
        memories.extend(monologue_memories)

        return memories[:200]  # Limit to prevent overload

    def _get_monologue_memories(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get memories from inner monologue session logs."""
        memories = []
        logs_dir = Path("logs/inner_monologue/sessions")

        if not logs_dir.exists():
            return memories

        cutoff = datetime.now() - timedelta(hours=hours)

        for log_file in logs_dir.glob("*.jsonl"):
            try:
                # Check file date from name
                date_str = log_file.stem.split("_session")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date.date() >= cutoff.date():
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                thought = json.loads(line.strip())
                                memories.append({
                                    "content": thought.get("content", ""),
                                    "metadata": {
                                        "type": thought.get("type", "thought"),
                                        "confidence": thought.get("confidence", 0),
                                        "timestamp": thought.get("timestamp", "")
                                    },
                                    "id": thought.get("id", self._generate_id("thought"))
                                })
                            except json.JSONDecodeError:
                                continue
            except Exception:
                continue

        return memories

    # ==================== Light Phase Helpers ====================

    def _strengthen_memory_connections(self, memory: Dict[str, Any]) -> int:
        """Strengthen KG connections related to a memory."""
        if not self.kg:
            return 0

        strengthened = 0
        content = memory.get("content", "")

        # Extract key terms (simple approach)
        words = set(re.findall(r'\b[A-Za-z]{4,}\b', content.lower()))
        important_words = [w for w in words if w not in self._get_stopwords()][:10]

        # Find related nodes and strengthen edges
        for word in important_words:
            try:
                nodes = self.kg.find_nodes(word, limit=3)
                for node in nodes:
                    # Strengthen edges to this node
                    if hasattr(self.kg, 'graph') and hasattr(self.kg.graph, 'edges'):
                        for edge in self.kg.graph.edges(node.id, data=True):
                            if 'weight' in edge[2]:
                                # Increase weight slightly
                                new_weight = min(1.0, edge[2]['weight'] + 0.05)
                                self.kg.graph.edges[edge[0], edge[1]]['weight'] = new_weight
                                strengthened += 1
            except Exception:
                continue

        return strengthened

    # ==================== Deep Phase Helpers ====================

    def _find_temporal_patterns(self) -> List[ConsolidatedPattern]:
        """Find patterns related to time (when user asks what)."""
        patterns = []

        # Analyze monologue logs for temporal patterns
        logs_dir = Path("logs/inner_monologue/sessions")
        if not logs_dir.exists():
            return patterns

        hour_topics = defaultdict(list)

        for log_file in logs_dir.glob("*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            thought = json.loads(line.strip())
                            ts = thought.get("timestamp", "")
                            if ts:
                                hour = datetime.fromisoformat(ts).hour
                                content = thought.get("content", "")
                                # Extract topic keywords
                                words = re.findall(r'\b[A-Za-z]{5,}\b', content.lower())
                                hour_topics[hour].extend(words[:5])
                        except:
                            continue
            except:
                continue

        # Find patterns in hour-topic relationships
        for hour, topics in hour_topics.items():
            if len(topics) < 5:
                continue

            topic_counts = Counter(topics)
            top_topics = topic_counts.most_common(3)

            if top_topics and top_topics[0][1] >= 3:
                pattern = ConsolidatedPattern(
                    pattern_id=self._generate_id("temporal"),
                    timestamp=datetime.now().isoformat(),
                    pattern_type="temporal",
                    description=f"At {hour}:00, user often discusses: {', '.join(t[0] for t in top_topics)}",
                    frequency=sum(t[1] for t in top_topics),
                    confidence=min(0.9, 0.3 + (top_topics[0][1] * 0.1)),
                    examples=[t[0] for t in top_topics],
                    metadata={"hour": hour, "topic_counts": dict(top_topics)}
                )
                patterns.append(pattern)

        return patterns[:10]  # Limit patterns

    def _find_topical_patterns(self) -> List[ConsolidatedPattern]:
        """Find recurring topical themes."""
        patterns = []

        if not self.kg:
            return patterns

        try:
            # Get most connected nodes as key topics
            stats = self.kg.get_stats()

            # Find clusters of related concepts
            if hasattr(self.kg, 'graph'):
                node_degrees = dict(self.kg.graph.degree())
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]

                for node_id, degree in top_nodes:
                    if degree < 3:
                        continue

                    # Get node info
                    node = self.kg.get_node(node_id) if hasattr(self.kg, 'get_node') else None
                    if node:
                        pattern = ConsolidatedPattern(
                            pattern_id=self._generate_id("topical"),
                            timestamp=datetime.now().isoformat(),
                            pattern_type="topical",
                            description=f"Recurring topic: {node.label} ({node.type}) with {degree} connections",
                            frequency=degree,
                            confidence=min(0.95, 0.5 + (degree * 0.05)),
                            examples=[],
                            metadata={"node_id": node_id, "node_type": node.type}
                        )
                        patterns.append(pattern)
        except Exception as e:
            print(f"[NeuroDream] Error finding topical patterns: {e}")

        return patterns

    def _find_emotional_patterns(self) -> List[ConsolidatedPattern]:
        """Find patterns in emotional history from EvoEmo."""
        patterns = []

        if not self.evoemo:
            return patterns

        try:
            # Get mood history
            history = self.evoemo.get_history(days=7) if hasattr(self.evoemo, 'get_history') else []

            if not history:
                return patterns

            # Analyze emotion transitions
            emotion_counts = Counter()
            hour_emotions = defaultdict(list)

            for entry in history:
                emotion = entry.get("emotion", "neutral")
                emotion_counts[emotion] += 1

                ts = entry.get("timestamp", "")
                if ts:
                    try:
                        hour = datetime.fromisoformat(ts).hour
                        hour_emotions[hour].append(emotion)
                    except:
                        pass

            # Dominant emotion pattern
            if emotion_counts:
                top_emotion, count = emotion_counts.most_common(1)[0]
                total = sum(emotion_counts.values())
                pattern = ConsolidatedPattern(
                    pattern_id=self._generate_id("emotional"),
                    timestamp=datetime.now().isoformat(),
                    pattern_type="emotional",
                    description=f"Dominant emotional state: {top_emotion} ({count}/{total} = {count/total:.0%})",
                    frequency=count,
                    confidence=count / total,
                    examples=list(emotion_counts.keys())[:5],
                    metadata={"emotion_distribution": dict(emotion_counts)}
                )
                patterns.append(pattern)

            # Time-based emotion patterns
            for hour, emotions in hour_emotions.items():
                if len(emotions) >= 3:
                    dominant = Counter(emotions).most_common(1)[0]
                    if dominant[1] >= 2:
                        pattern = ConsolidatedPattern(
                            pattern_id=self._generate_id("emotional_temporal"),
                            timestamp=datetime.now().isoformat(),
                            pattern_type="emotional",
                            description=f"At {hour}:00, user tends to feel {dominant[0]}",
                            frequency=dominant[1],
                            confidence=dominant[1] / len(emotions),
                            examples=emotions[:5],
                            metadata={"hour": hour, "emotion": dominant[0]}
                        )
                        patterns.append(pattern)
        except Exception as e:
            print(f"[NeuroDream] Error finding emotional patterns: {e}")

        return patterns[:10]

    def _prune_weak_edges(self, threshold: float = 0.1) -> int:
        """Remove weak edges from Knowledge Graph."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return 0

        pruned = 0
        edges_to_remove = []

        try:
            for u, v, data in self.kg.graph.edges(data=True):
                weight = data.get('weight', 0.5)
                if weight < threshold:
                    edges_to_remove.append((u, v))

            for u, v in edges_to_remove:
                self.kg.graph.remove_edge(u, v)
                pruned += 1

            # Save if method exists
            if hasattr(self.kg, 'save'):
                self.kg.save()
        except Exception as e:
            print(f"[NeuroDream] Error pruning edges: {e}")

        return pruned

    def _strengthen_frequent_edges(self) -> int:
        """Strengthen frequently accessed edges."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return 0

        strengthened = 0

        try:
            # Strengthen edges with high weight (frequently used)
            for u, v, data in self.kg.graph.edges(data=True):
                weight = data.get('weight', 0.5)
                if weight > 0.7:
                    # Boost slightly
                    new_weight = min(1.0, weight + 0.02)
                    self.kg.graph.edges[u, v]['weight'] = new_weight
                    strengthened += 1
        except Exception as e:
            print(f"[NeuroDream] Error strengthening edges: {e}")

        return strengthened

    def _merge_similar_nodes(self) -> int:
        """Merge nodes with very similar labels."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return 0

        # This is a simplified version - full implementation would use
        # embedding similarity
        merged = 0

        try:
            nodes = list(self.kg.graph.nodes(data=True))
            labels = {}

            for node_id, data in nodes:
                label = data.get('label', '').lower()
                if label in labels:
                    # Found duplicate - merge edges
                    original_id = labels[label]

                    # Transfer edges
                    for neighbor in list(self.kg.graph.neighbors(node_id)):
                        if not self.kg.graph.has_edge(original_id, neighbor):
                            edge_data = self.kg.graph.edges[node_id, neighbor]
                            self.kg.graph.add_edge(original_id, neighbor, **edge_data)

                    # Remove duplicate node
                    self.kg.graph.remove_node(node_id)
                    merged += 1
                else:
                    labels[label] = node_id

            if merged > 0 and hasattr(self.kg, 'save'):
                self.kg.save()
        except Exception as e:
            print(f"[NeuroDream] Error merging nodes: {e}")

        return merged

    # ==================== REM Phase Helpers ====================

    def _generate_novel_connections(self) -> List[DreamInsight]:
        """Generate novel connections between distant concepts."""
        insights = []

        if not self.kg or not hasattr(self.kg, 'graph'):
            return insights

        try:
            import networkx as nx

            nodes = list(self.kg.graph.nodes(data=True))
            if len(nodes) < 4:
                return insights

            # Find nodes that are far apart but might be related
            # (conceptually distant but potentially connectable)

            # Sample random pairs
            for _ in range(min(10, len(nodes) // 2)):
                if self._interrupt_flag.is_set():
                    break

                n1, n2 = random.sample(nodes, 2)
                node1_id, node1_data = n1
                node2_id, node2_data = n2

                # Check if not directly connected
                if self.kg.graph.has_edge(node1_id, node2_id):
                    continue

                # Check path distance
                try:
                    path_length = nx.shortest_path_length(
                        self.kg.graph.to_undirected(),
                        node1_id, node2_id
                    )
                except nx.NetworkXNoPath:
                    path_length = float('inf')

                # If far apart (3+ hops), consider creating connection
                if path_length >= 3:
                    label1 = node1_data.get('label', node1_id)
                    label2 = node2_data.get('label', node2_id)

                    # Generate insight about potential connection
                    insight = DreamInsight(
                        id=self._generate_id("insight"),
                        timestamp=datetime.now().isoformat(),
                        insight_type="connection",
                        content=f"Potential hidden connection: '{label1}' may relate to '{label2}' "
                               f"(currently {path_length} hops apart)",
                        confidence=0.3 + random.random() * 0.3,  # 0.3-0.6
                        source_nodes=[node1_id, node2_id],
                        created_edges=[]
                    )

                    # Optionally create the edge with low weight
                    if random.random() > 0.7:  # 30% chance to create edge
                        self.kg.graph.add_edge(
                            node1_id, node2_id,
                            type="dream_connection",
                            weight=0.2
                        )
                        insight.created_edges.append({
                            "source": node1_id,
                            "target": node2_id,
                            "type": "dream_connection"
                        })

                    insights.append(insight)
        except Exception as e:
            print(f"[NeuroDream] Error generating novel connections: {e}")

        return insights[:5]  # Limit insights

    def _generate_hypotheses(self) -> List[DreamInsight]:
        """Generate creative hypotheses based on patterns."""
        insights = []

        # Load recent patterns
        patterns_file = self.data_dir / "consolidated_patterns.jsonl"
        patterns = []

        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    for line in f:
                        patterns.append(json.loads(line.strip()))
            except:
                pass

        # Generate hypotheses from patterns
        for pattern in patterns[-10:]:  # Recent patterns
            if self._interrupt_flag.is_set():
                break

            pattern_type = pattern.get("pattern_type", "")
            description = pattern.get("description", "")

            if pattern_type == "temporal":
                # Hypothesis about user behavior
                insight = DreamInsight(
                    id=self._generate_id("hypothesis"),
                    timestamp=datetime.now().isoformat(),
                    insight_type="hypothesis",
                    content=f"Hypothesis: {description}. This might indicate a work/study pattern "
                           f"that could be used for proactive assistance.",
                    confidence=pattern.get("confidence", 0.5) * 0.8,
                    source_nodes=[],
                    created_edges=[]
                )
                insights.append(insight)

            elif pattern_type == "emotional":
                # Hypothesis about emotional wellbeing
                insight = DreamInsight(
                    id=self._generate_id("hypothesis"),
                    timestamp=datetime.now().isoformat(),
                    insight_type="prediction",
                    content=f"Prediction: {description}. Consider adjusting response tone "
                           f"during these times.",
                    confidence=pattern.get("confidence", 0.5) * 0.7,
                    source_nodes=[],
                    created_edges=[]
                )
                insights.append(insight)

        return insights[:3]  # Limit hypotheses

    # ==================== Storage ====================

    def _save_session(self, session: SleepSession):
        """Save sleep session to dream journal."""
        journal_file = self.data_dir / "dream_journal.jsonl"
        with open(journal_file, 'a') as f:
            f.write(json.dumps(session.to_dict()) + '\n')

    def _save_insight(self, insight: DreamInsight):
        """Save dream insight."""
        insights_file = self.data_dir / "insights.jsonl"
        with open(insights_file, 'a') as f:
            f.write(json.dumps(insight.to_dict()) + '\n')

    def _save_consolidated_patterns(self, patterns: List[ConsolidatedPattern]):
        """Save consolidated patterns."""
        patterns_file = self.data_dir / "consolidated_patterns.jsonl"
        with open(patterns_file, 'a') as f:
            for pattern in patterns:
                f.write(json.dumps(pattern.to_dict()) + '\n')

    # ==================== Retrieval ====================

    def get_dream_journal(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent dream journal entries."""
        journal_file = self.data_dir / "dream_journal.jsonl"
        entries = []

        if journal_file.exists():
            with open(journal_file, 'r') as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except:
                        continue

        return entries[-n:]

    def get_insights(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent dream insights."""
        insights_file = self.data_dir / "insights.jsonl"
        insights = []

        if insights_file.exists():
            with open(insights_file, 'r') as f:
                for line in f:
                    try:
                        insights.append(json.loads(line.strip()))
                    except:
                        continue

        return insights[-n:]

    def get_patterns(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get consolidated patterns."""
        patterns_file = self.data_dir / "consolidated_patterns.jsonl"
        patterns = []

        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                for line in f:
                    try:
                        patterns.append(json.loads(line.strip()))
                    except:
                        continue

        return patterns[-n:]

    # ==================== Utilities ====================

    def _get_stopwords(self) -> set:
        """Get common stopwords to filter."""
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'also', 'now', 'here', 'there', 'then', 'once', 'from',
            'into', 'with', 'about', 'against', 'between', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'and', 'but', 'or', 'yet', 'for', 'nor', 'so'
        }

    def set_callbacks(
        self,
        on_phase_change: Optional[Callable[[SleepPhase], None]] = None,
        on_insight: Optional[Callable[[DreamInsight], None]] = None
    ):
        """Set callback functions for events."""
        self._on_phase_change = on_phase_change
        self._on_insight = on_insight


# ==================== Singleton Access ====================

_neurodream_instance: Optional[NeuroDreamEngine] = None


def get_neurodream(**kwargs) -> NeuroDreamEngine:
    """Get or create NeuroDream singleton."""
    global _neurodream_instance
    if _neurodream_instance is None:
        _neurodream_instance = NeuroDreamEngine(**kwargs)
    return _neurodream_instance


def create_neurodream(**kwargs) -> NeuroDreamEngine:
    """Create new NeuroDream instance (replaces singleton)."""
    global _neurodream_instance
    _neurodream_instance = NeuroDreamEngine(**kwargs)
    return _neurodream_instance
