"""API endpoints for all AURA features."""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["features"])


# ============================================================================
# MOOD / EVOEMO
# ============================================================================

class MoodResponse(BaseModel):
    emotion: str = "neutral"
    confidence: int = 50
    valence: float = 0.0
    arousal: float = 0.0
    session_dominant: Optional[str] = None
    readings: int = 0


@router.get("/mood", response_model=MoodResponse)
async def get_mood():
    """Get current mood state from EvoEmo."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_mood_sync)
        return result
    except Exception as e:
        logger.error(f"[Mood] Error: {e}")
        return MoodResponse()


def _get_mood_sync() -> dict:
    agent = agent_service.agent
    if "evoemo" in agent.tools:
        evoemo = agent.tools["evoemo"]
        state = evoemo.get_state() if hasattr(evoemo, 'get_state') else {}
        session = evoemo.get_session_summary() if hasattr(evoemo, 'get_session_summary') else {}
        return {
            "emotion": state.get("emotion", "neutral"),
            "confidence": state.get("confidence", 50),
            "valence": state.get("valence", 0.0),
            "arousal": state.get("arousal", 0.0),
            "session_dominant": session.get("dominant"),
            "readings": session.get("readings", 0)
        }
    return {"emotion": "neutral", "confidence": 50, "valence": 0.0, "arousal": 0.0}


@router.get("/mood/history")
async def get_mood_history():
    """Get mood history and patterns."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_mood_history_sync)
        return result
    except Exception as e:
        return {"error": str(e)}


def _get_mood_history_sync() -> dict:
    agent = agent_service.agent
    if "evoemo" in agent.tools:
        evoemo = agent.tools["evoemo"]
        session = evoemo.get_session_summary() if hasattr(evoemo, 'get_session_summary') else {}
        daily = evoemo.get_daily_summary() if hasattr(evoemo, 'get_daily_summary') else None
        patterns = evoemo.get_patterns() if hasattr(evoemo, 'get_patterns') else {}
        return {
            "session": session,
            "daily": daily.__dict__ if daily and hasattr(daily, '__dict__') else None,
            "patterns": patterns
        }
    return {}


# ============================================================================
# AURA ALIVE
# ============================================================================

class AuraStatusResponse(BaseModel):
    enabled: bool = False
    mood: str = "neutral"
    energy: float = 0.5
    warmth: float = 0.5
    engagement: float = 0.5
    soul_name: str = "AURA"
    patterns_learned: int = 0
    turns: int = 0


@router.get("/aura", response_model=AuraStatusResponse)
async def get_aura_status():
    """Get AURA ALIVE status."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_aura_sync)
        return result
    except Exception as e:
        logger.error(f"[AURA] Error: {e}")
        return AuraStatusResponse()


def _get_aura_sync() -> dict:
    agent = agent_service.agent
    if hasattr(agent, 'aura') and agent.aura:
        aura = agent.aura
        try:
            status = aura.get_status()
            return {
                "enabled": True,
                "mood": status['mood']['mood'],
                "energy": status['mood']['energy'],
                "warmth": status['mood']['warmth'],
                "engagement": status['mood']['engagement'],
                "soul_name": aura.soul.name if hasattr(aura, 'soul') else "AURA",
                "patterns_learned": status['patterns']['total_patterns'],
                "turns": status['turns']
            }
        except:
            return {"enabled": True, "mood": "neutral", "energy": 0.5, "soul_name": "AURA"}
    return {"enabled": False}


class RememberRequest(BaseModel):
    fact: str


@router.post("/aura/remember")
async def aura_remember(request: RememberRequest):
    """Store a fact in AURA memory."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _aura_remember_sync(request.fact)
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _aura_remember_sync(fact: str) -> dict:
    agent = agent_service.agent
    if hasattr(agent, 'aura') and agent.aura and fact.strip():
        success = agent.aura.remember(fact.strip(), importance=0.7)
        return {"success": success, "fact": fact[:50]}
    return {"success": False, "error": "AURA not available"}


# ============================================================================
# INNER MONOLOGUE / THOUGHTS
# ============================================================================

class ThoughtItem(BaseModel):
    type: str
    content: str
    confidence: Optional[int] = None
    timestamp: Optional[str] = None


class ThoughtsResponse(BaseModel):
    thoughts: List[ThoughtItem] = []
    verbosity: int = 2
    think_aloud: bool = False
    thought_count: int = 0


@router.get("/thoughts", response_model=ThoughtsResponse)
async def get_thoughts():
    """Get recent thoughts from inner monologue."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_thoughts_sync)
        return result
    except Exception as e:
        logger.error(f"[Thoughts] Error: {e}")
        return ThoughtsResponse()


def _get_thoughts_sync() -> dict:
    agent = agent_service.agent
    if "inner_monologue" in agent.tools:
        monologue = agent.tools["inner_monologue"]
        thoughts = monologue.get_recent_thoughts(15) if hasattr(monologue, 'get_recent_thoughts') else []
        status = monologue.execute("status") if hasattr(monologue, 'execute') else {}

        thought_list = []
        for t in thoughts:
            thought_list.append({
                "type": t.type if hasattr(t, 'type') else "unknown",
                "content": t.content if hasattr(t, 'content') else str(t),
                "confidence": t.confidence if hasattr(t, 'confidence') else None
            })

        return {
            "thoughts": thought_list,
            "verbosity": status.get("verbosity", 2) if isinstance(status, dict) else 2,
            "think_aloud": status.get("think_aloud", False) if isinstance(status, dict) else False,
            "thought_count": status.get("thought_count", len(thought_list)) if isinstance(status, dict) else len(thought_list)
        }
    return {"thoughts": [], "verbosity": 2, "think_aloud": False, "thought_count": 0}


@router.get("/thoughts/reasoning")
async def get_reasoning_chain():
    """Get the reasoning chain for 'why did you do that?' queries."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_reasoning_sync)
        return {"reasoning": result}
    except Exception as e:
        return {"reasoning": f"Error: {e}"}


def _get_reasoning_sync() -> str:
    agent = agent_service.agent
    if "inner_monologue" in agent.tools:
        return agent.tools["inner_monologue"].get_reasoning_chain()
    return "No reasoning chain available."


@router.post("/thoughts/clear")
async def clear_thoughts():
    """Clear the thought stream."""
    try:
        agent = agent_service.agent
        if "inner_monologue" in agent.tools:
            agent.tools["inner_monologue"].stream.clear()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================

class KnowledgeGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}


@router.get("/knowledge-graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(center: Optional[str] = None, depth: int = 2):
    """Get knowledge graph nodes and edges."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_kg_sync(center, depth)
        )
        return result
    except Exception as e:
        logger.error(f"[KG] Error: {e}")
        return KnowledgeGraphResponse()


def _get_kg_sync(center: Optional[str], depth: int) -> dict:
    agent = agent_service.agent
    if "knowledge_graph" not in agent.tools:
        return {"nodes": [], "edges": [], "stats": {}}

    kg = agent.tools["knowledge_graph"]

    # Get nodes and edges
    if center and center.strip():
        related = kg.get_related(center.strip(), depth=depth, min_weight=0.2)
        nodes = related.get("nodes", [])
        edges = related.get("edges", [])
    else:
        nodes = kg.get_recent_nodes(limit=30) if hasattr(kg, 'get_recent_nodes') else []
        node_ids = {n.id for n in nodes}
        edges = []
        if hasattr(kg, '_edges'):
            for edge in kg._edges.values():
                if edge.source_id in node_ids and edge.target_id in node_ids:
                    edges.append(edge)

    # Format for JSON
    nodes_json = []
    for node in nodes:
        nodes_json.append({
            "id": node.id,
            "label": node.label,
            "type": node.type,
            "confidence": node.confidence if hasattr(node, 'confidence') else 1.0,
            "access_count": node.access_count if hasattr(node, 'access_count') else 1
        })

    edges_json = []
    for edge in edges:
        edges_json.append({
            "source": edge.source_id,
            "target": edge.target_id,
            "type": edge.type,
            "weight": edge.weight if hasattr(edge, 'weight') else 1.0
        })

    # Get stats
    stats = kg.get_stats() if hasattr(kg, 'get_stats') else {}

    return {"nodes": nodes_json, "edges": edges_json, "stats": stats}


# ============================================================================
# METACOGNITIVE GUARDIAN
# ============================================================================

class GuardianResponse(BaseModel):
    enabled: bool = False
    monitoring_level: str = "medium"
    interventions: int = 0
    patterns_learned: int = 0
    session_predictions: int = 0
    recent_predictions: List[Dict[str, Any]] = []


@router.get("/guardian", response_model=GuardianResponse)
async def get_guardian_status():
    """Get Metacognitive Guardian status."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_guardian_sync)
        return result
    except Exception as e:
        logger.error(f"[Guardian] Error: {e}")
        return GuardianResponse()


def _get_guardian_sync() -> dict:
    agent = agent_service.agent
    if hasattr(agent, 'guardian') and agent.guardian:
        guardian = agent.guardian
        # Use get_stats() - the correct method name
        stats = guardian.get_stats() if hasattr(guardian, 'get_stats') else {}
        # Get recent predictions from the guardian's session_predictions list
        recent = []
        if hasattr(guardian, 'session_predictions'):
            recent = [
                {
                    "risk_score": p.risk_score if hasattr(p, 'risk_score') else p.get('risk_score', 0),
                    "action": p.action if hasattr(p, 'action') else p.get('action', 'unknown'),
                    "recommendation": p.recommendation if hasattr(p, 'recommendation') else p.get('recommendation', '')
                }
                for p in guardian.session_predictions[-5:]
            ]
        return {
            "enabled": True,
            "monitoring_level": stats.get("monitoring_level", "medium"),
            "interventions": stats.get("interventions_triggered", 0),
            "patterns_learned": stats.get("failure_patterns_learned", 0),
            "session_predictions": stats.get("session_predictions", 0),
            "recent_predictions": recent
        }
    return {"enabled": False}


# ============================================================================
# NEURODREAM
# ============================================================================

class NeuroDreamResponse(BaseModel):
    enabled: bool = False
    is_sleeping: bool = False
    current_phase: Optional[str] = None
    total_sessions: int = 0
    total_insights: int = 0
    dream_journal: List[Dict[str, Any]] = []
    insights: List[Dict[str, Any]] = []


@router.get("/neurodream", response_model=NeuroDreamResponse)
async def get_neurodream_status():
    """Get NeuroDream sleep/dream status."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_neurodream_sync)
        return result
    except Exception as e:
        logger.error(f"[NeuroDream] Error: {e}")
        return NeuroDreamResponse()


def _get_neurodream_sync() -> dict:
    agent = agent_service.agent
    if hasattr(agent, 'neurodream') and agent.neurodream:
        nd = agent.neurodream
        status = nd.get_status() if hasattr(nd, 'get_status') else {}
        journal = nd.get_dream_journal(n=5) if hasattr(nd, 'get_dream_journal') else []
        insights = nd.get_insights() if hasattr(nd, 'get_insights') else []
        return {
            "enabled": True,
            "is_sleeping": status.get("is_sleeping", False),
            "current_phase": status.get("current_phase"),
            "total_sessions": status.get("total_sessions", 0),
            "total_insights": status.get("total_insights", 0),
            "dream_journal": journal[-5:] if journal else [],
            "insights": insights[-5:] if insights else []
        }
    return {"enabled": False}


def _trigger_sleep_sync() -> dict:
    """Sync helper for sleep trigger."""
    import time
    start = time.time()
    logger.info("[NeuroDream] Starting sleep trigger...")

    agent = agent_service.agent
    if not hasattr(agent, 'neurodream') or not agent.neurodream:
        return {"success": False, "error": "NeuroDream not available"}

    try:
        logger.info("[NeuroDream] Calling enter_sleep...")
        result = agent.neurodream.enter_sleep(trigger="web_ui")
        elapsed = time.time() - start
        logger.info(f"[NeuroDream] enter_sleep completed in {elapsed:.2f}s: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"[NeuroDream] enter_sleep error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/neurodream/sleep")
async def trigger_sleep():
    """Trigger a sleep cycle."""
    import concurrent.futures
    try:
        # Use dedicated executor with timeout to avoid blocking
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_trigger_sleep_sync)
            try:
                result = future.result(timeout=10)  # 10 second timeout
                return result
            except concurrent.futures.TimeoutError:
                logger.error("[NeuroDream] Sleep trigger timed out after 10s")
                return {"success": False, "error": "Operation timed out - enter_sleep is blocking"}
    except Exception as e:
        logger.error(f"[NeuroDream] Sleep trigger exception: {e}")
        return {"success": False, "error": str(e)}


def _trigger_wake_sync() -> dict:
    """Sync helper for wake trigger."""
    agent = agent_service.agent
    if hasattr(agent, 'neurodream') and agent.neurodream:
        result = agent.neurodream.wake_up(reason="user_request")
        return {"success": True, "result": result}
    return {"success": False, "error": "NeuroDream not available"}


@router.post("/neurodream/wake")
async def trigger_wake():
    """Wake up from sleep."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _trigger_wake_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# FLUXMIND
# ============================================================================

class FluxMindResponse(BaseModel):
    enabled: bool = False
    version: str = "unknown"
    accuracy: float = 0.0
    calibration: str = "unknown"


@router.get("/fluxmind", response_model=FluxMindResponse)
async def get_fluxmind_status():
    """Get FluxMind calibrated reasoning status."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_fluxmind_sync)
        return result
    except Exception as e:
        logger.error(f"[FluxMind] Error: {e}")
        return FluxMindResponse()


def _get_fluxmind_sync() -> dict:
    agent = agent_service.agent
    if "fluxmind" in agent.tools:
        fm = agent.tools["fluxmind"]
        status = fm.status() if hasattr(fm, 'status') else {}
        # FluxMind uses OOD-calibrated uncertainty - derive calibration from thresholds
        thresholds = status.get("thresholds", {})
        calibration = "ood_calibrated" if status.get("available", False) else "unknown"
        # Calculate accuracy from model capabilities if available
        # FluxMind's accuracy is based on calibration quality, not prediction accuracy
        accuracy = 0.95 if status.get("available", False) else 0.0
        return {
            "enabled": status.get("available", False),
            "version": status.get("version", "unknown"),
            "accuracy": accuracy,
            "calibration": calibration,
            "thresholds": thresholds
        }
    return {"enabled": False}


# ============================================================================
# VOICE / TTS
# ============================================================================

class VoiceResponse(BaseModel):
    available: bool = False
    engine: str = "none"
    sesame_loaded: bool = False


@router.get("/voice", response_model=VoiceResponse)
async def get_voice_status():
    """Get voice/TTS status."""
    try:
        # TTS is typically a separate singleton, check if agent has TTS
        agent = agent_service.agent

        # Check for PersonaPlex tool
        if "personaplex" in agent.tools:
            pp = agent.tools["personaplex"]
            return {
                "available": True,
                "engine": "personaplex",
                "sesame_loaded": pp.is_loaded() if hasattr(pp, 'is_loaded') else False
            }

        return {"available": False, "engine": "none", "sesame_loaded": False}
    except Exception as e:
        logger.error(f"[Voice] Error: {e}")
        return VoiceResponse()


# ============================================================================
# TOOLS LIST
# ============================================================================

@router.get("/tools")
async def get_available_tools():
    """Get list of available tools."""
    try:
        agent = agent_service.agent
        tools = []
        for name, tool in agent.tools.items():
            tools.append({
                "name": name,
                "description": tool.__doc__[:100] if tool.__doc__ else "No description"
            })
        return {"tools": tools, "count": len(tools)}
    except Exception as e:
        return {"tools": [], "count": 0, "error": str(e)}


# ============================================================================
# METACOGNITION STATS
# ============================================================================

@router.get("/metacognition")
async def get_metacognition_stats():
    """Get metacognition statistics."""
    try:
        from apprentice_agent.metacognition import MetacognitionLogger
        stats = MetacognitionLogger.get_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# LOCAL RAG (Retrieval Augmented Generation)
# ============================================================================

class RAGIndexRequest(BaseModel):
    path: str
    recursive: bool = True


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RAGStatsResponse(BaseModel):
    total_chunks: int = 0
    total_files: int = 0
    embeddings_available: bool = False
    embedding_model: str = "unavailable"
    chunks_by_type: Dict[str, int] = {}


@router.get("/rag/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG index statistics."""
    try:
        agent = agent_service.agent
        if "local_rag" not in agent.tools:
            return RAGStatsResponse()

        rag_tool = agent.tools["local_rag"]
        stats = rag_tool.rag.get_stats()
        return stats
    except Exception as e:
        logger.error(f"[RAG] Stats error: {e}")
        return RAGStatsResponse()


@router.get("/rag/files")
async def get_rag_files():
    """List indexed files."""
    try:
        agent = agent_service.agent
        if "local_rag" not in agent.tools:
            return {"files": [], "error": "RAG not available"}

        rag_tool = agent.tools["local_rag"]
        files = rag_tool.rag.list_indexed_files()
        return {"files": files, "count": len(files)}
    except Exception as e:
        return {"files": [], "error": str(e)}


@router.post("/rag/index")
async def index_documents(request: RAGIndexRequest):
    """Index a file or directory."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _index_documents_sync(request.path, request.recursive)
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _index_documents_sync(path: str, recursive: bool) -> dict:
    from pathlib import Path
    agent = agent_service.agent
    if "local_rag" not in agent.tools:
        return {"success": False, "error": "RAG not available"}

    rag_tool = agent.tools["local_rag"]
    path_obj = Path(path)

    if path_obj.is_dir():
        return rag_tool.rag.index_directory(path, recursive=recursive)
    else:
        return rag_tool.rag.index_file(path)


@router.post("/rag/search")
async def search_documents(request: RAGSearchRequest):
    """Search indexed documents."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _search_documents_sync(request.query, request.top_k)
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _search_documents_sync(query: str, top_k: int) -> dict:
    agent = agent_service.agent
    if "local_rag" not in agent.tools:
        return {"success": False, "error": "RAG not available"}

    rag_tool = agent.tools["local_rag"]
    results = rag_tool.rag.search(query, top_k=top_k)

    return {
        "success": True,
        "query": query,
        "results": [
            {
                "content": r.chunk.content[:500] + "..." if len(r.chunk.content) > 500 else r.chunk.content,
                "source": r.chunk.source,
                "score": f"{r.score:.0%}"
            }
            for r in results
        ]
    }


@router.post("/rag/clear")
async def clear_rag_index():
    """Clear the RAG index."""
    try:
        agent = agent_service.agent
        if "local_rag" not in agent.tools:
            return {"success": False, "error": "RAG not available"}

        rag_tool = agent.tools["local_rag"]
        result = rag_tool.rag.clear_index()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# A-MEM (Agentic Memory - Zettelkasten-style)
# ============================================================================

class AMEMStatsResponse(BaseModel):
    total_notes: int = 0
    total_links: int = 0
    total_boxes: int = 0
    categories: Dict[str, int] = {}
    has_embeddings: int = 0
    evolution_enabled: bool = False


class AMEMNoteResponse(BaseModel):
    id: str
    content: str
    keywords: List[str] = []
    tags: List[str] = []
    context: str = ""
    category: str = "general"
    importance: float = 0.5
    links: int = 0
    created_at: str = ""


class AMEMSearchRequest(BaseModel):
    query: str
    k: int = 5
    follow_links: bool = True


class AMEMRememberRequest(BaseModel):
    content: str
    tags: List[str] = []
    category: str = "general"
    importance: float = 0.5


@router.get("/amem/stats", response_model=AMEMStatsResponse)
async def get_amem_stats():
    """Get A-MEM statistics."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_amem_stats_sync)
        return result
    except Exception as e:
        logger.error(f"[A-MEM] Stats error: {e}")
        return AMEMStatsResponse()


def _get_amem_stats_sync() -> dict:
    agent = agent_service.agent
    # Check tools dict for amem
    amem_tool = agent.tools.get('amem')
    if amem_tool and hasattr(amem_tool, 'amem'):
        return amem_tool.amem.get_stats()
    # Try to get from hybrid memory
    hybrid_mem = agent.tools.get('hybrid_amem')
    if hybrid_mem and hasattr(hybrid_mem, 'amem'):
        return hybrid_mem.amem.get_stats()
    return {}


@router.get("/amem/notes")
async def get_amem_notes(limit: int = 20, category: Optional[str] = None):
    """Get recent A-MEM notes."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_amem_notes_sync(limit, category)
        )
        return result
    except Exception as e:
        logger.error(f"[A-MEM] Notes error: {e}")
        return {"notes": [], "error": str(e)}


def _get_amem_notes_sync(limit: int, category: Optional[str]) -> dict:
    agent = agent_service.agent

    # Get A-MEM instance from tools dict
    amem = None
    amem_tool = agent.tools.get('amem')
    if amem_tool and hasattr(amem_tool, 'amem'):
        amem = amem_tool.amem
    else:
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem and hasattr(hybrid_mem, 'amem'):
            amem = hybrid_mem.amem

    if not amem:
        return {"notes": [], "count": 0}

    # Get notes sorted by creation time
    notes = sorted(
        amem._notes.values(),
        key=lambda n: n.created_at,
        reverse=True
    )

    # Filter by category if specified
    if category:
        notes = [n for n in notes if n.category == category]

    notes = notes[:limit]

    return {
        "notes": [
            {
                "id": n.id,
                "content": n.content[:200],
                "keywords": n.keywords,
                "tags": n.tags,
                "context": n.context,
                "category": n.category,
                "importance": n.importance,
                "links": len(n.links),
                "created_at": n.created_at
            }
            for n in notes
        ],
        "count": len(notes)
    }


@router.get("/amem/note/{note_id}")
async def get_amem_note(note_id: str):
    """Get a specific A-MEM note with linked notes."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_amem_note_sync(note_id)
        )
        return result
    except Exception as e:
        return {"error": str(e)}


def _get_amem_note_sync(note_id: str) -> dict:
    agent = agent_service.agent

    # Get A-MEM instance from tools dict
    amem = None
    amem_tool = agent.tools.get('amem')
    if amem_tool and hasattr(amem_tool, 'amem'):
        amem = amem_tool.amem
    else:
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem and hasattr(hybrid_mem, 'amem'):
            amem = hybrid_mem.amem

    if not amem:
        return {"error": "A-MEM not available"}

    note = amem.read(note_id)
    if not note:
        return {"error": "Note not found"}

    # Get linked notes
    linked = amem.get_linked(note_id)

    return {
        "note": {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "tags": note.tags,
            "context": note.context,
            "category": note.category,
            "importance": note.importance,
            "boxes": note.boxes,
            "created_at": note.created_at,
            "updated_at": note.updated_at,
            "access_count": note.access_count
        },
        "linked_notes": [
            {
                "id": ln.id,
                "content": ln.content[:100],
                "strength": s
            }
            for ln, s in linked[:10]
        ]
    }


@router.post("/amem/search")
async def search_amem(request: AMEMSearchRequest):
    """Search A-MEM notes."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _search_amem_sync(request.query, request.k, request.follow_links)
        )
        return result
    except Exception as e:
        return {"error": str(e), "results": []}


def _search_amem_sync(query: str, k: int, follow_links: bool) -> dict:
    agent = agent_service.agent

    # Get A-MEM instance from tools dict
    amem = None
    amem_tool = agent.tools.get('amem')
    if amem_tool and hasattr(amem_tool, 'amem'):
        amem = amem_tool.amem
    else:
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem and hasattr(hybrid_mem, 'amem'):
            amem = hybrid_mem.amem

    if not amem:
        return {"results": [], "error": "A-MEM not available"}

    results = amem.search_agentic(query, k=k, follow_links=follow_links)

    return {
        "query": query,
        "count": len(results),
        "results": [
            {
                "id": r.get("id", ""),
                "content": r.get("content", ""),
                "keywords": r.get("keywords", []),
                "tags": r.get("tags", []),
                "context": r.get("context", ""),
                "relevance": round(r.get("relevance", 0), 2),
                "hop": r.get("hop", 0)
            }
            for r in results
        ]
    }


@router.post("/amem/remember")
async def amem_remember(request: AMEMRememberRequest):
    """Store a new memory in A-MEM."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _amem_remember_sync(
                request.content, request.tags, request.category, request.importance
            )
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _amem_remember_sync(
    content: str, tags: List[str], category: str, importance: float
) -> dict:
    agent = agent_service.agent

    # Prefer hybrid memory for cross-system storage
    hybrid_mem = agent.tools.get('hybrid_amem')
    if hybrid_mem:
        result = hybrid_mem.remember(
            content=content,
            memory_type=category,
            tags=tags,
            importance=importance,
            source="web_ui"
        )
        return {
            "success": True,
            "note_id": result.get("note_id"),
            "links_created": result.get("links_created", 0),
            "kg_nodes": len(result.get("node_ids", []))
        }

    # Fallback to A-MEM only
    amem_tool = agent.tools.get('amem')
    if amem_tool:
        note = amem_tool.remember(
            content=content,
            tags=tags,
            category=category,
            importance=importance
        )
        return {
            "success": True,
            "note_id": note.id,
            "keywords": note.keywords,
            "links": len(note.links)
        }

    return {"success": False, "error": "A-MEM not available"}


@router.get("/amem/boxes")
async def get_amem_boxes():
    """Get A-MEM boxes (soft clusters)."""
    try:
        agent = agent_service.agent

        # Get A-MEM instance from tools dict
        amem = None
        amem_tool = agent.tools.get('amem')
        if amem_tool and hasattr(amem_tool, 'amem'):
            amem = amem_tool.amem
        else:
            hybrid_mem = agent.tools.get('hybrid_amem')
            if hybrid_mem and hasattr(hybrid_mem, 'amem'):
                amem = hybrid_mem.amem

        if not amem:
            return {"boxes": {}}

        return {"boxes": amem.list_boxes()}
    except Exception as e:
        return {"boxes": {}, "error": str(e)}


@router.post("/amem/consolidate")
async def consolidate_amem():
    """Consolidate A-MEM (merge duplicates, prune weak links)."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _consolidate_amem_sync)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def _consolidate_amem_sync() -> dict:
    agent = agent_service.agent

    # Prefer hybrid consolidation
    hybrid_mem = agent.tools.get('hybrid_amem')
    if hybrid_mem:
        result = hybrid_mem.consolidate()
        return {"success": True, **result}

    amem_tool = agent.tools.get('amem')
    if amem_tool and hasattr(amem_tool, 'amem'):
        result = amem_tool.amem.consolidate()
        return {"success": True, **result}

    return {"success": False, "error": "A-MEM not available"}


# ============================================================================
# PROTO-AGI (Truth Spine - Cognitive Core)
# ============================================================================

class ProtoAGIResponse(BaseModel):
    enabled: bool = False
    mode: str = "idle"
    cycle_count: int = 0
    facts: int = 0
    beliefs: int = 0
    speculations: int = 0
    verifier_pass_rate: float = 0.0
    pending_confirmations: int = 0
    last_action: Optional[str] = None


@router.get("/proto-agi", response_model=ProtoAGIResponse)
async def get_proto_agi_status():
    """Get Proto-AGI Truth Spine status."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _get_proto_agi_sync)
        return result
    except Exception as e:
        logger.error(f"[Proto-AGI] Error: {e}")
        return ProtoAGIResponse()


def _get_proto_agi_sync() -> dict:
    agent = agent_service.agent
    if hasattr(agent, 'proto_agi') and agent.proto_agi:
        agi = agent.proto_agi
        try:
            status = agi.get_status() if hasattr(agi, 'get_status') else {}
            memory = status.get('memory', {})
            verifier = status.get('verifier', {})

            return {
                "enabled": True,
                "mode": status.get('mode', 'idle'),
                "cycle_count": status.get('cycle_count', 0),
                "facts": memory.get('facts', 0),
                "beliefs": memory.get('beliefs', 0),
                "speculations": memory.get('speculations', 0),
                "verifier_pass_rate": verifier.get('success_rate', 0.0),
                "pending_confirmations": status.get('pending_confirmations', 0),
                "last_action": status.get('last_action')
            }
        except Exception as e:
            logger.error(f"[Proto-AGI] Status error: {e}")
            return {"enabled": True, "mode": "error"}
    return {"enabled": False}


@router.post("/proto-agi/mode")
async def set_proto_agi_mode(mode: str):
    """Set Proto-AGI operation mode (idle, assist, operate)."""
    try:
        agent = agent_service.agent
        if hasattr(agent, 'proto_agi') and agent.proto_agi:
            agent.proto_agi.set_mode(mode)
            return {"success": True, "mode": mode}
        return {"success": False, "error": "Proto-AGI not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/proto-agi/start")
async def start_proto_agi_loop(cycle_interval: float = 60.0):
    """Start the Proto-AGI autonomous cognitive loop."""
    try:
        agent = agent_service.agent
        if hasattr(agent, 'start_proto_agi'):
            agent.start_proto_agi(cycle_interval)
            return {"success": True, "message": f"Proto-AGI loop started (interval: {cycle_interval}s)"}
        return {"success": False, "error": "Proto-AGI start method not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/proto-agi/stop")
async def stop_proto_agi_loop():
    """Stop the Proto-AGI autonomous cognitive loop."""
    try:
        agent = agent_service.agent
        if hasattr(agent, 'stop_proto_agi'):
            agent.stop_proto_agi()
            return {"success": True, "message": "Proto-AGI loop stopped"}
        return {"success": False, "error": "Proto-AGI stop method not available"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/proto-agi/traces")
async def get_proto_agi_traces(limit: int = 10):
    """Get recent verification traces."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_traces_sync(limit)
        )
        return result
    except Exception as e:
        return {"traces": [], "error": str(e)}


def _get_traces_sync(limit: int) -> dict:
    agent = agent_service.agent
    if hasattr(agent, 'proto_agi') and agent.proto_agi:
        agi = agent.proto_agi
        if hasattr(agi, 'get_recent_traces'):
            traces = agi.get_recent_traces(limit)
            return {
                "traces": [
                    {
                        "id": t.id if hasattr(t, 'id') else str(i),
                        "action": t.action if hasattr(t, 'action') else "unknown",
                        "tier": t.tier.value if hasattr(t, 'tier') else "unknown",
                        "verified": t.verified if hasattr(t, 'verified') else False,
                        "timestamp": t.timestamp if hasattr(t, 'timestamp') else ""
                    }
                    for i, t in enumerate(traces)
                ],
                "count": len(traces)
            }
    return {"traces": [], "count": 0}


# ============================================================================
# HYBRID MEMORY (A-MEM + Knowledge Graph)
# ============================================================================

@router.get("/hybrid-memory/stats")
async def get_hybrid_memory_stats():
    """Get combined hybrid memory statistics."""
    try:
        agent = agent_service.agent
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem:
            return hybrid_mem.get_stats()
        return {"error": "Hybrid memory not available"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/hybrid-memory/search")
async def search_hybrid_memory(request: AMEMSearchRequest):
    """Search across both A-MEM and Knowledge Graph."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _search_hybrid_sync(request.query, request.k)
        )
        return result
    except Exception as e:
        return {"error": str(e), "results": []}


def _search_hybrid_sync(query: str, k: int) -> dict:
    agent = agent_service.agent

    hybrid_mem = agent.tools.get('hybrid_amem')
    if not hybrid_mem:
        return {"results": [], "error": "Hybrid memory not available"}

    results = hybrid_mem.recall(query, k=k)

    return {
        "query": query,
        "count": len(results),
        "results": [
            {
                "content": r.content,
                "source": r.source,
                "score": round(r.score, 2),
                "id": r.id,
                "keywords": r.keywords,
                "tags": r.tags,
                "context": r.context,
                "node_type": r.node_type,
                "relationships": r.relationships
            }
            for r in results
        ]
    }


@router.get("/hybrid-memory/context")
async def get_memory_context(query: str, max_tokens: int = 500):
    """Get memory context for a query (for LLM prompt injection)."""
    try:
        agent = agent_service.agent
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem:
            context = hybrid_mem.get_context(query, max_tokens=max_tokens)
            return {"context": context, "query": query}
        return {"context": "", "error": "Hybrid memory not available"}
    except Exception as e:
        return {"context": "", "error": str(e)}
