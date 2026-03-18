"""
MCP Tools for AURA Episodic Memory.

Provides agent-accessible tools for memory operations.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .episode import Episode, EpisodeType, EpisodeQuery, EmotionalValence, TemporalContext
from .memory_store import EpisodicMemoryStore
from .timeline import TimelineEngine
from .temporal_parser import TemporalParser, TemporalRange

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable


def create_episodic_tools(
    memory_store: EpisodicMemoryStore,
    timeline_engine: Optional[TimelineEngine] = None
) -> List[MCPTool]:
    """
    Create MCP tools for episodic memory operations.

    Args:
        memory_store: EpisodicMemoryStore instance
        timeline_engine: Optional TimelineEngine (created if not provided)

    Returns:
        List of MCPTool definitions
    """
    if timeline_engine is None:
        timeline_engine = TimelineEngine(memory_store)

    parser = TemporalParser()

    tools = []

    # Tool 1: Remember Episode
    def remember_episode(
        content: str,
        episode_type: str = "conversation",
        importance: float = 0.5,
        title: Optional[str] = None,
        entities: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        emotional_valence: str = "neutral"
    ) -> Dict[str, Any]:
        """Store a new episodic memory."""
        try:
            ep_type = EpisodeType(episode_type)
        except ValueError:
            ep_type = EpisodeType.CONVERSATION

        try:
            valence = EmotionalValence(emotional_valence)
        except ValueError:
            valence = EmotionalValence.NEUTRAL

        episode = Episode(
            content=content,
            episode_type=ep_type,
            temporal_context=TemporalContext(timestamp=datetime.now()),
            title=title,
            importance=importance,
            emotional_valence=valence,
            entities_involved=entities or [],
            tools_used=tools_used or []
        )

        try:
            episode_id = memory_store.store_episode(episode)
        except Exception as e:
            logger.error(f"[EpisodicMemory] Failed to store episode: {e}")
            return {"success": False, "error": f"Storage failed: {e}"}

        preview = title or content[:50]
        suffix = "..." if not title and len(content) > 50 else ""
        return {
            "success": True,
            "episode_id": episode_id,
            "message": f"Stored episode: {preview}{suffix}"
        }

    tools.append(MCPTool(
        name="remember_episode",
        description="Store a new episodic memory. Use this to remember important events, conversations, tasks, or insights.",
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content/description of the memory"
                },
                "episode_type": {
                    "type": "string",
                    "enum": ["conversation", "task_execution", "learning", "error", "milestone", "insight", "user_preference", "system_event"],
                    "description": "Type of episode",
                    "default": "conversation"
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Importance score (0-1)",
                    "default": 0.5
                },
                "title": {
                    "type": "string",
                    "description": "Brief title for the episode"
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entities involved in this episode"
                },
                "tools_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools used in this episode"
                },
                "emotional_valence": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "default": "neutral"
                }
            },
            "required": ["content"]
        },
        handler=remember_episode
    ))

    # Tool 2: Recall Memories
    def recall_memories(
        query: str,
        limit: int = 5,
        episode_types: Optional[List[str]] = None,
        time_filter: Optional[str] = None,
        min_importance: float = 0.0
    ) -> Dict[str, Any]:
        """Search and retrieve episodic memories."""
        # Parse time filter if provided
        start_time = None
        end_time = None

        if time_filter:
            time_range = parser.parse(time_filter)
            if time_range:
                start_time = time_range.start
                end_time = time_range.end

        # Convert episode types
        ep_types = None
        if episode_types:
            ep_types = []
            for t in episode_types:
                try:
                    ep_types.append(EpisodeType(t))
                except ValueError:
                    pass

        search_query = EpisodeQuery(
            query_text=query,
            start_time=start_time,
            end_time=end_time,
            episode_types=ep_types,
            limit=limit,
            min_score=min_importance
        )

        results = memory_store.search(search_query)

        memories = []
        for result in results:
            ep = result.episode
            ts = ep.temporal_context.timestamp if ep.temporal_context else None
            memories.append({
                "id": ep.id,
                "title": ep.title,
                "content": ep.content[:500],
                "type": ep.episode_type.value,
                "timestamp": ts.isoformat() if ts else "",
                "importance": ep.importance,
                "score": result.score,
                "entities": ep.entities_involved[:5],
                "recency": parser.get_recency_description(ts) if ts else "unknown"
            })

        return {
            "success": True,
            "count": len(memories),
            "memories": memories
        }

    tools.append(MCPTool(
        name="recall_memories",
        description="Search and retrieve episodic memories by semantic similarity and filters.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum memories to return"
                },
                "episode_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["conversation", "task_execution", "learning", "error", "milestone", "insight", "user_preference", "system_event"]
                    },
                    "description": "Filter by episode types"
                },
                "time_filter": {
                    "type": "string",
                    "description": "Natural language time filter (e.g., 'yesterday', 'last week')"
                },
                "min_importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0,
                    "description": "Minimum importance threshold"
                }
            },
            "required": ["query"]
        },
        handler=recall_memories
    ))

    # Tool 3: Time Travel
    def time_travel(
        time_reference: str,
        context_episodes: int = 5
    ) -> Dict[str, Any]:
        """Travel to a point in memory and get surrounding context."""
        episodes, narrative = timeline_engine.time_travel(time_reference, context_episodes)

        episode_summaries = []
        for ep in episodes[:10]:
            episode_summaries.append({
                "id": ep.id,
                "title": ep.title or ep.content[:50],
                "type": ep.episode_type.value,
                "timestamp": ep.temporal_context.timestamp.isoformat()
            })

        return {
            "success": True,
            "time_reference": time_reference,
            "episode_count": len(episodes),
            "narrative": narrative,
            "episodes": episode_summaries
        }

    tools.append(MCPTool(
        name="time_travel",
        description="Travel to a specific time in memory. Returns episodes and narrative from that period.",
        parameters={
            "type": "object",
            "properties": {
                "time_reference": {
                    "type": "string",
                    "description": "Natural language time reference (e.g., 'yesterday morning', 'last Monday', '2 hours ago')"
                },
                "context_episodes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Number of surrounding episodes to include"
                }
            },
            "required": ["time_reference"]
        },
        handler=time_travel
    ))

    # Tool 4: Get Timeline
    def get_timeline(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "day",
        episode_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get a timeline view of memories."""
        # Parse times
        if start_time:
            start_range = parser.parse(start_time)
            start = start_range.start if start_range else datetime.now() - timedelta(days=7)
        else:
            start = datetime.now() - timedelta(days=7)

        if end_time:
            end_range = parser.parse(end_time)
            end = end_range.end if end_range else datetime.now()
        else:
            end = datetime.now()

        # Convert episode types
        ep_types = None
        if episode_types:
            ep_types = []
            for t in episode_types:
                try:
                    ep_types.append(EpisodeType(t))
                except ValueError:
                    pass

        time_range = TemporalRange(start=start, end=end, description="timeline query")
        timeline_view = timeline_engine.get_timeline(time_range, granularity, ep_types)

        segments = []
        for seg in timeline_view.segments:
            if seg.episodes:
                segments.append({
                    "label": seg.label,
                    "start": seg.start_time.isoformat(),
                    "end": seg.end_time.isoformat(),
                    "episode_count": len(seg.episodes),
                    "episodes": [
                        {"title": ep.title or ep.content[:40], "type": ep.episode_type.value}
                        for ep in seg.episodes[:3]
                    ]
                })

        return {
            "success": True,
            "total_episodes": timeline_view.total_episodes,
            "granularity": granularity,
            "segments": segments,
            "narrative": timeline_view.to_narrative()[:1000]
        }

    tools.append(MCPTool(
        name="get_timeline",
        description="Get a timeline view of memories within a time range.",
        parameters={
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "Start time (natural language or ISO format)"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time (natural language or ISO format)"
                },
                "granularity": {
                    "type": "string",
                    "enum": ["hour", "day", "week", "month"],
                    "default": "day",
                    "description": "Timeline granularity"
                },
                "episode_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by episode types"
                }
            }
        },
        handler=get_timeline
    ))

    # Tool 5: Get Day Summary
    def get_day_summary(date: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of memories for a specific day."""
        if date:
            time_range = parser.parse(date)
            target_date = time_range.start if time_range else datetime.now()
        else:
            target_date = datetime.now()

        summary = timeline_engine.get_day_summary(target_date)

        return {
            "success": True,
            **summary
        }

    tools.append(MCPTool(
        name="get_day_summary",
        description="Get a summary of all memories for a specific day.",
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Date to summarize (natural language, e.g., 'yesterday', 'last Monday'). Defaults to today."
                }
            }
        },
        handler=get_day_summary
    ))

    # Tool 6: Find Episode Chain
    def find_episode_chain(
        episode_id: str,
        direction: str = "both",
        max_length: int = 10
    ) -> Dict[str, Any]:
        """Find chain of related episodes (story mode)."""
        episode = memory_store.get_episode(episode_id)

        if not episode:
            return {
                "success": False,
                "error": f"Episode not found: {episode_id}"
            }

        chain = timeline_engine.find_episode_chains(episode, direction, max_length)

        chain_data = []
        for ep in chain:
            chain_data.append({
                "id": ep.id,
                "title": ep.title or ep.content[:50],
                "type": ep.episode_type.value,
                "timestamp": ep.temporal_context.timestamp.isoformat(),
                "is_origin": ep.id == episode_id
            })

        return {
            "success": True,
            "origin_episode": episode_id,
            "direction": direction,
            "chain_length": len(chain),
            "chain": chain_data
        }

    tools.append(MCPTool(
        name="find_episode_chain",
        description="Find a chain of related episodes around a starting episode (story mode).",
        parameters={
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": "ID of the starting episode"
                },
                "direction": {
                    "type": "string",
                    "enum": ["before", "after", "both"],
                    "default": "both",
                    "description": "Direction to search for related episodes"
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                    "description": "Maximum chain length"
                }
            },
            "required": ["episode_id"]
        },
        handler=find_episode_chain
    ))

    # Tool 7: Detect Patterns
    def detect_patterns(
        days: int = 30,
        episode_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect temporal patterns in memory."""
        ep_type = None
        if episode_type:
            try:
                ep_type = EpisodeType(episode_type)
            except ValueError:
                pass

        patterns = timeline_engine.detect_patterns(ep_type, days)

        return {
            "success": True,
            "analysis_period_days": days,
            "patterns": patterns
        }

    tools.append(MCPTool(
        name="detect_patterns",
        description="Detect temporal patterns in memory (activity times, recurring themes, etc.).",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30,
                    "description": "Number of days to analyze"
                },
                "episode_type": {
                    "type": "string",
                    "enum": ["conversation", "task_execution", "learning", "error", "milestone", "insight", "user_preference", "system_event"],
                    "description": "Filter by episode type"
                }
            }
        },
        handler=detect_patterns
    ))

    # Tool 8: Get Memory Statistics
    def get_memory_statistics() -> Dict[str, Any]:
        """Get overall memory statistics."""
        stats = memory_store.get_statistics()

        return {
            "success": True,
            **stats
        }

    tools.append(MCPTool(
        name="get_memory_statistics",
        description="Get overall episodic memory statistics.",
        parameters={
            "type": "object",
            "properties": {}
        },
        handler=get_memory_statistics
    ))

    return tools


class QuickEpisodicMemory:
    """Lightweight wrapper for fast episodic memory access in brain.py.

    Provides quick_recall (with LRU cache) and quick_store (best-effort)
    suitable for injection into every LLM call without adding latency.
    """

    def __init__(self, memory_store: EpisodicMemoryStore):
        self._store = memory_store
        self._cache: dict = {}  # Simple LRU-like dict
        self._cache_max = 100

    def quick_recall(self, query: str, limit: int = 3) -> list:
        """Search memories with 500ms timeout and LRU cache on query embeddings."""
        import threading

        cache_key = f"{query[:80]}:{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = []
        done = threading.Event()

        def _search():
            nonlocal result
            try:
                search_query = EpisodeQuery(query_text=query, limit=limit, min_score=0.3)
                hits = self._store.search(search_query)
                result = []
                for hit in hits:
                    ep = hit.episode
                    result.append({
                        "id": ep.id,
                        "title": ep.title or ep.content[:50],
                        "summary": ep.content[:200],
                        "timestamp": ep.temporal_context.timestamp.isoformat() if ep.temporal_context else "",
                        "importance": ep.importance,
                    })
            except Exception as _e:
                logger.debug(f"[QuickEpisodicMemory] Search error (best-effort): {_e}")
            finally:
                done.set()

        t = threading.Thread(target=_search, daemon=True)
        t.start()
        done.wait(timeout=0.5)  # 500ms hard timeout

        # Cache result (evict oldest if too large)
        if len(self._cache) >= self._cache_max:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result

        return result

    def quick_store(self, content: str, title: str = "", importance: float = 0.5) -> None:
        """Store an episode in a background thread (best-effort, never blocks)."""
        import threading

        def _store():
            try:
                episode = Episode(
                    content=content,
                    title=title or content[:60],
                    episode_type=EpisodeType.CONVERSATION,
                    temporal_context=TemporalContext(timestamp=datetime.now()),
                    importance=importance,
                    emotional_valence=EmotionalValence.NEUTRAL,
                )
                self._store.store_episode(episode)
            except Exception:
                pass

        threading.Thread(target=_store, daemon=True).start()


def register_episodic_tools_with_agent(
    agent,
    memory_store: EpisodicMemoryStore,
    timeline_engine: Optional[TimelineEngine] = None
):
    """
    Register episodic memory tools with an agent.

    Args:
        agent: Agent instance with register_tool method
        memory_store: EpisodicMemoryStore instance
        timeline_engine: Optional TimelineEngine
    """
    tools = create_episodic_tools(memory_store, timeline_engine)

    for tool in tools:
        if hasattr(agent, 'register_tool'):
            agent.register_tool(
                name=f"episodic_{tool.name}",
                description=tool.description,
                parameters=tool.parameters,
                handler=tool.handler
            )
        elif hasattr(agent, 'tools'):
            # Alternative registration method
            agent.tools[f"episodic_{tool.name}"] = {
                "description": tool.description,
                "parameters": tool.parameters,
                "handler": tool.handler
            }

    logger.info(f"Registered {len(tools)} episodic memory tools with agent")
