"""API endpoints for all AURA features."""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Response, Depends
from pydantic import BaseModel

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

# Lazy import to avoid blocking event loop at module load
def _get_agent_service():
    """Get agent_service with lazy loading."""
    from api.services.agent_service import agent_service
    return agent_service

router = APIRouter(prefix="/api", tags=["features"], dependencies=[Depends(require_api_key)])


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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_mood_sync)
        return result
    except Exception as e:
        logger.error(f"[Mood] Error: {e}")
        return MoodResponse()


def _get_mood_sync() -> dict:
    agent = _get_agent_service().agent
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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_mood_history_sync)
        return result
    except Exception as e:
        return {"error": safe_error_detail(e)}


def _get_mood_history_sync() -> dict:
    agent = _get_agent_service().agent
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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_aura_sync)
        return result
    except Exception as e:
        logger.error(f"[AURA] Error: {e}")
        return AuraStatusResponse()


def _get_aura_sync() -> dict:
    agent = _get_agent_service().agent
    if not getattr(agent, 'aura_enabled', False):
        return {"enabled": False}
    try:
        mood = "neutral"
        energy = 0.5
        warmth = 0.5
        engagement = 0.5
        if "evoemo" in agent.tools:
            evoemo = agent.tools["evoemo"]
            state = evoemo.get_state() if hasattr(evoemo, 'get_state') else {}
            mood = state.get("emotion", "neutral")
            # Map valence/arousal to warmth/energy
            energy = min(1.0, max(0.0, (state.get("arousal", 0.0) + 1.0) / 2.0))
            warmth = min(1.0, max(0.0, (state.get("valence", 0.0) + 1.0) / 2.0))
            session = evoemo.get_session_summary() if hasattr(evoemo, 'get_session_summary') else {}
            engagement = min(1.0, session.get("readings", 0) / 20.0)
        return {
            "enabled": True,
            "mood": mood,
            "energy": energy,
            "warmth": warmth,
            "engagement": engagement,
            "soul_name": "AURA",
            "patterns_learned": 0,
            "turns": 0,
        }
    except Exception:
        return {"enabled": True, "mood": "neutral", "energy": 0.5, "warmth": 0.5,
                "engagement": 0.5, "soul_name": "AURA", "patterns_learned": 0, "turns": 0}


class RememberRequest(BaseModel):
    fact: str


@router.post("/aura/remember")
async def aura_remember(request: RememberRequest):
    """Store a fact in AURA memory."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _aura_remember_sync(request.fact)
        )
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


def _aura_remember_sync(fact: str) -> dict:
    agent = _get_agent_service().agent
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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_thoughts_sync)
        return result
    except Exception as e:
        logger.error(f"[Thoughts] Error: {e}")
        return ThoughtsResponse()


def _get_thoughts_sync() -> dict:
    agent = _get_agent_service().agent
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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_reasoning_sync)
        return {"reasoning": result}
    except Exception as e:
        return {"reasoning": f"Error: {e}"}


def _get_reasoning_sync() -> str:
    agent = _get_agent_service().agent
    if "inner_monologue" in agent.tools:
        return agent.tools["inner_monologue"].get_reasoning_chain()
    return "No reasoning chain available."


@router.post("/thoughts/clear")
async def clear_thoughts():
    """Clear the thought stream."""
    try:
        agent = _get_agent_service().agent
        if "inner_monologue" in agent.tools:
            agent.tools["inner_monologue"].stream.clear()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_kg_sync(center, depth)
        )
        return result
    except Exception as e:
        logger.error(f"[KG] Error: {e}")
        return KnowledgeGraphResponse()


def _get_kg_sync(center: Optional[str], depth: int) -> dict:
    agent = _get_agent_service().agent
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
# NEURODREAM
# ============================================================================

class NeuroDreamResponse(BaseModel):
    enabled: bool = False
    loading: bool = False
    is_sleeping: bool = False
    current_phase: Optional[str] = None
    total_sessions: int = 0
    total_insights: int = 0
    dream_journal: List[Dict[str, Any]] = []
    insights: List[Dict[str, Any]] = []
    learned_context: Optional[Dict[str, Any]] = None


@router.get("/neurodream", response_model=NeuroDreamResponse)
async def get_neurodream_status():
    """Get NeuroDream sleep/dream status."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_neurodream_sync)
        return result
    except Exception as e:
        logger.error(f"[NeuroDream] Error: {e}")
        return NeuroDreamResponse()


def _get_neurodream_sync() -> dict:
    svc = _get_agent_service()
    if not svc.is_ready:
        return {"enabled": False, "loading": True}
    agent = svc.agent
    if hasattr(agent, 'neurodream') and agent.neurodream:
        nd = agent.neurodream
        status = nd.get_status() if hasattr(nd, 'get_status') else {}
        journal = nd.get_dream_journal(n=5) if hasattr(nd, 'get_dream_journal') else []
        insights = nd.get_insights() if hasattr(nd, 'get_insights') else []
        learned_ctx = None
        if hasattr(nd, 'get_learned_context') and nd.get_learned_context():
            lc = nd.get_learned_context()
            learned_ctx = {
                "version": lc.version,
                "generated_at": lc.generated_at,
                "user_summary": lc.user_summary,
                "key_facts_count": len(lc.key_facts),
                "preferences_count": len(lc.preferences),
                "ongoing_topics": lc.ongoing_topics,
                "conversations_processed": lc.conversations_processed,
            }
        return {
            "enabled": True,
            "loading": False,
            "is_sleeping": status.get("is_sleeping", False),
            "current_phase": status.get("current_phase"),
            "total_sessions": status.get("total_sessions", 0),
            "total_insights": status.get("total_insights", 0),
            "dream_journal": journal[-5:] if journal else [],
            "insights": insights[-5:] if insights else [],
            "learned_context": learned_ctx,
        }
    return {"enabled": False, "loading": False}


def _trigger_sleep_sync() -> dict:
    """Sync helper for sleep trigger."""
    import time
    start = time.time()
    logger.info("[NeuroDream] Starting sleep trigger...")

    agent = _get_agent_service().agent
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
        return {"success": False, "error": safe_error_detail(e)}


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
        return {"success": False, "error": safe_error_detail(e)}


def _trigger_wake_sync() -> dict:
    """Sync helper for wake trigger."""
    agent = _get_agent_service().agent
    if hasattr(agent, 'neurodream') and agent.neurodream:
        result = agent.neurodream.wake_up(reason="user_request")
        return {"success": True, "result": result}
    return {"success": False, "error": "NeuroDream not available"}


@router.post("/neurodream/wake")
async def trigger_wake():
    """Wake up from sleep."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _trigger_wake_sync)
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


@router.get("/neurodream/learned-context")
async def get_learned_context():
    """Get the current Letta-style learned context (Phase 4D)."""
    try:
        agent = _get_agent_service().agent
        if hasattr(agent, 'neurodream') and agent.neurodream:
            nd = agent.neurodream
            ctx = nd.get_learned_context()
            if ctx:
                return {
                    "available": True,
                    "context": ctx.to_dict(),
                    "system_prompt_preview": ctx.to_system_prompt()[:500],
                }
        return {"available": False, "message": "No learned context generated yet"}
    except Exception as e:
        return {"available": False, "error": safe_error_detail(e)}


def _generate_learned_context_sync() -> dict:
    """Sync helper for learned context generation."""
    agent = _get_agent_service().agent
    if hasattr(agent, 'neurodream') and agent.neurodream:
        return agent.neurodream.generate_learned_context()
    return {"success": False, "error": "NeuroDream not available"}


@router.post("/neurodream/learned-context/generate")
async def generate_learned_context():
    """Manually trigger Letta-style learned context generation (Phase 4D)."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _generate_learned_context_sync)
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


# ============================================================================
# FLUXMIND (REMOVED)
# ============================================================================


# ============================================================================
# VOICE / TTS
# ============================================================================

@router.get("/voice")
async def get_voice_status():
    """Get voice/TTS status from VoicePresenceService."""
    try:
        from aura.services.voice_presence import get_voice_presence
        vps = get_voice_presence()
        return vps.get_status()
    except Exception as e:
        logger.error(f"[Voice] Error: {e}")
        return {"available": False, "engine": "none", "enabled": False, "error": safe_error_detail(e)}


class SynthesizeRequest(BaseModel):
    text: str
    emotion: Optional[str] = None


@router.post("/voice/synthesize", dependencies=[Depends(require_api_key)])
async def synthesize_speech(req: SynthesizeRequest):
    """Synthesize speech and return WAV bytes."""
    from aura.services.voice_presence import get_voice_presence
    vps = get_voice_presence()
    if not vps._enabled:
        raise HTTPException(status_code=503, detail="Voice not enabled")

    wav_bytes = await asyncio.get_running_loop().run_in_executor(
        None, vps.synthesize_wav, req.text, req.emotion
    )
    return Response(content=wav_bytes, media_type="audio/wav")


class VoiceToggleRequest(BaseModel):
    enabled: bool


@router.post("/voice/toggle")
async def toggle_voice(req: VoiceToggleRequest):
    """Enable or disable voice output."""
    from aura.services.voice_presence import get_voice_presence
    vps = get_voice_presence()
    vps.set_enabled(req.enabled)
    return {"enabled": vps._enabled}


# ============================================================================
# TOOLS LIST  (Tier 4: categories + search support)
# ============================================================================

# Tool category mapping — drives search/filter in the frontend
_TOOL_CATEGORIES: dict = {
    # Core
    "filesystem": "Core", "web_search": "Core", "brave_search": "Core",
    "tavily": "Core", "firecrawl": "Core", "code_executor": "Core",
    # Memory
    "clipboard": "Memory", "obsidian": "Memory", "amem": "Memory",
    "hybrid_amem": "Memory", "knowledge_graph": "Memory",
    # Communication
    "email": "Communication", "notifications": "Communication",
    # Productivity
    "task_manager": "Productivity", "calendar": "Productivity",
    "task_scheduler": "Productivity", "document_generator": "Productivity",
    "spaced_repetition": "Productivity",
    # Media
    "voice_synth": "Media", "image_gen": "Media",
    "audio_transcriber": "Media", "voice": "Media",
    # Development
    "git": "Development", "github": "Development", "log_analyst": "Development",
    "api_tester": "Development", "database": "Development", "shell_executor": "Development",
    "tool_builder": "Development",
    # AI / Research
    "research": "AI", "arxiv_search": "AI", "mcts_reasoning": "AI",
    "reasoning_tree": "AI",
    # Monitoring
    "meeting_intel": "Monitoring",
    "screen_reader": "Monitoring", "screenshot": "Monitoring", "browser": "Monitoring",
    # System
    "windows_control": "System", "system_control": "System",
    "vision": "System", "pdf_reader": "System",
    # Analytics
    "predictive_tasks": "Analytics", "life_logger": "Analytics",
}


@router.get("/tools")
async def get_available_tools():
    """Get list of available tools with categories for frontend filtering."""
    try:
        agent = _get_agent_service().agent
        tools = []
        for name, tool in agent.tools.items():
            tools.append({
                "name": name,
                "description": (getattr(tool, 'description', None) or tool.__doc__ or "No description")[:120],
                "category": _TOOL_CATEGORIES.get(name, "Other"),
            })
        # Sort: by category then name
        tools.sort(key=lambda t: (t["category"], t["name"]))
        categories = sorted(set(t["category"] for t in tools))
        return {"tools": tools, "count": len(tools), "categories": categories}
    except Exception as e:
        return {"tools": [], "count": 0, "categories": [], "error": safe_error_detail(e)}


# ============================================================================
# PREDICTIVE TASKS FEEDBACK
# ============================================================================

class PredictionFeedbackRequest(BaseModel):
    prediction_id: str
    tool: str
    action: str
    accepted: bool


@router.post("/predictions/feedback")
async def submit_prediction_feedback(request: PredictionFeedbackRequest):
    """Submit feedback on a prediction (accepted/rejected)."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _prediction_feedback_sync(
                request.prediction_id, request.tool, request.action, request.accepted
            )
        )
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


def _prediction_feedback_sync(prediction_id: str, tool: str, action: str, accepted: bool) -> dict:
    agent = _get_agent_service().agent
    if "predictive_tasks" not in agent.tools:
        return {"success": False, "error": "Predictive tasks not available"}
    pred_tool = agent.tools["predictive_tasks"]
    return pred_tool.execute("feedback", prediction_id=prediction_id, tool=tool, action_name=action, accepted=accepted)


# ============================================================================
# TOKEN COST DASHBOARD  (Tier 4)
# ============================================================================

@router.get("/costs/summary")
async def get_costs_summary():
    """Get current session token usage and estimated API cost."""
    try:
        brain = _get_agent_service().agent.brain
        stats = brain.get_session_stats()
        return {"success": True, **stats}
    except Exception as e:
        return {
            "success": False, "error": safe_error_detail(e),
            "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0, "queries": 0,
        }


# ============================================================================
# PLUGIN HOT-RELOAD  (Tier 4)
# ============================================================================

@router.post("/plugins/reload", dependencies=[Depends(require_api_key)])
async def reload_plugins():
    """Reload custom tools from registry without restarting AURA."""
    try:
        agent = _get_agent_service().agent
        before = len(agent.tools)
        agent._load_custom_tools()
        after = len(agent.tools)
        return {
            "success": True,
            "tools_before": before,
            "tools_after": after,
            "new_tools": after - before,
        }
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


@router.get("/plugins")
async def list_plugins():
    """List custom plugins from the tools registry."""
    import json
    from pathlib import Path
    registry_path = Path(__file__).parent.parent.parent / "data" / "custom_tools.json"
    if not registry_path.exists():
        return {"success": True, "plugins": [], "count": 0}
    try:
        with open(registry_path) as f:
            registry = json.load(f)
        plugins = registry.get("tools", [])
        return {"success": True, "plugins": plugins, "count": len(plugins)}
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e), "plugins": []}


# ============================================================================
# METACOGNITION STATS
# ============================================================================

@router.get("/metacognition")
async def get_metacognition_stats():
    """Get metacognition statistics."""
    try:
        from aura.metacognition import MetacognitionLogger
        stats = MetacognitionLogger.get_stats()
        return stats
    except Exception as e:
        return {"error": safe_error_detail(e)}


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
        agent = _get_agent_service().agent
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
        agent = _get_agent_service().agent
        if "local_rag" not in agent.tools:
            return {"files": [], "error": "RAG not available"}

        rag_tool = agent.tools["local_rag"]
        files = rag_tool.rag.list_indexed_files()
        return {"files": files, "count": len(files)}
    except Exception as e:
        return {"files": [], "error": safe_error_detail(e)}


@router.post("/rag/index", dependencies=[Depends(require_api_key)])
async def index_documents(request: RAGIndexRequest):
    """Index a file or directory."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _index_documents_sync(request.path, request.recursive)
        )
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


def _index_documents_sync(path: str, recursive: bool) -> dict:
    from pathlib import Path
    agent = _get_agent_service().agent
    if "local_rag" not in agent.tools:
        return {"success": False, "error": "RAG not available"}

    # Restrict indexing to safe directories (home + data/)
    path_obj = Path(path).resolve()
    safe_roots = [Path.home().resolve(), Path("data").resolve()]
    if not any(str(path_obj).startswith(str(root)) for root in safe_roots):
        return {"success": False, "error": "Path must be within home directory or data/"}

    rag_tool = agent.tools["local_rag"]

    if path_obj.is_dir():
        return rag_tool.rag.index_directory(str(path_obj), recursive=recursive)
    else:
        return rag_tool.rag.index_file(str(path_obj))


@router.post("/rag/search")
async def search_documents(request: RAGSearchRequest):
    """Search indexed documents."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _search_documents_sync(request.query, request.top_k)
        )
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


def _search_documents_sync(query: str, top_k: int) -> dict:
    agent = _get_agent_service().agent
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
        agent = _get_agent_service().agent
        if "local_rag" not in agent.tools:
            return {"success": False, "error": "RAG not available"}

        rag_tool = agent.tools["local_rag"]
        result = rag_tool.rag.clear_index()
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _get_amem_stats_sync)
        return result
    except Exception as e:
        logger.error(f"[A-MEM] Stats error: {e}")
        return AMEMStatsResponse()


def _get_amem_stats_sync() -> dict:
    agent = _get_agent_service().agent
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
    limit = min(limit, 500)  # Cap to prevent OOM
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_amem_notes_sync(limit, category)
        )
        return result
    except Exception as e:
        logger.error(f"[A-MEM] Notes error: {e}")
        return {"notes": [], "error": safe_error_detail(e)}


def _get_amem_notes_sync(limit: int, category: Optional[str]) -> dict:
    agent = _get_agent_service().agent

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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_amem_note_sync(note_id)
        )
        return result
    except Exception as e:
        return {"error": safe_error_detail(e)}


def _get_amem_note_sync(note_id: str) -> dict:
    agent = _get_agent_service().agent

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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _search_amem_sync(request.query, request.k, request.follow_links)
        )
        return result
    except Exception as e:
        return {"error": safe_error_detail(e), "results": []}


def _search_amem_sync(query: str, k: int, follow_links: bool) -> dict:
    agent = _get_agent_service().agent

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
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _amem_remember_sync(
                request.content, request.tags, request.category, request.importance
            )
        )
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


def _amem_remember_sync(
    content: str, tags: List[str], category: str, importance: float
) -> dict:
    agent = _get_agent_service().agent

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
        agent = _get_agent_service().agent

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
        return {"boxes": {}, "error": safe_error_detail(e)}


@router.post("/amem/consolidate")
async def consolidate_amem():
    """Consolidate A-MEM (merge duplicates, prune weak links)."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _consolidate_amem_sync)
        return result
    except Exception as e:
        return {"success": False, "error": safe_error_detail(e)}


def _consolidate_amem_sync() -> dict:
    agent = _get_agent_service().agent

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
# PROTO-AGI (REMOVED)
# ============================================================================


# ============================================================================
# HYBRID MEMORY (A-MEM + Knowledge Graph)
# ============================================================================

@router.get("/hybrid-memory/stats")
async def get_hybrid_memory_stats():
    """Get combined hybrid memory statistics."""
    try:
        agent = _get_agent_service().agent
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem:
            return hybrid_mem.get_stats()
        return {"error": "Hybrid memory not available"}
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.post("/hybrid-memory/search")
async def search_hybrid_memory(request: AMEMSearchRequest):
    """Search across both A-MEM and Knowledge Graph."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _search_hybrid_sync(request.query, request.k)
        )
        return result
    except Exception as e:
        return {"error": safe_error_detail(e), "results": []}


def _search_hybrid_sync(query: str, k: int) -> dict:
    agent = _get_agent_service().agent

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
        agent = _get_agent_service().agent
        hybrid_mem = agent.tools.get('hybrid_amem')
        if hybrid_mem:
            context = hybrid_mem.get_context(query, max_tokens=max_tokens)
            return {"context": context, "query": query}
        return {"context": "", "error": "Hybrid memory not available"}
    except Exception as e:
        return {"context": "", "error": safe_error_detail(e)}


# ============================================================================
# Metacognitive Self-Improvement (Phase 6B)
# ============================================================================

@router.get("/metacognition/status")
async def get_metacognition_status():
    """Get metacognitive engine status: capabilities, goals, improvements."""
    try:
        from aura.consciousness.metacognition import get_metacognitive_engine
        mc = get_metacognitive_engine()
        return mc.get_status()
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/metacognition/capabilities")
async def get_capabilities():
    """Get AURA's self-assessed capability profile."""
    try:
        from aura.consciousness.metacognition import get_metacognitive_engine
        mc = get_metacognitive_engine()
        caps = mc.assess_capabilities()
        return {
            "capabilities": {
                d: {"score": c.score, "confidence": c.confidence,
                    "trend": c.trend, "evidence": c.evidence}
                for d, c in caps.items()
            }
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/metacognition/evaluation")
async def get_evaluation():
    """Get metacognitive evaluation report."""
    try:
        from aura.consciousness.metacognition import get_metacognitive_engine
        mc = get_metacognitive_engine()
        return mc.evaluate_progress()
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.post("/metacognition/cycle", dependencies=[Depends(require_api_key)])
async def run_metacognitive_cycle():
    """Trigger a full metacognitive cycle: assess -> plan -> improve -> evaluate."""
    try:
        from aura.consciousness.metacognition import get_metacognitive_engine
        mc = get_metacognitive_engine()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, mc.run_metacognitive_cycle)
        return result
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/metacognition/self-model")
async def get_self_model():
    """Get AURA's self-model and system prompt injection preview."""
    try:
        from aura.consciousness.metacognition import get_metacognitive_engine
        mc = get_metacognitive_engine()
        model = mc.get_self_model()
        return {
            "strengths": model.strengths,
            "weaknesses": model.weaknesses,
            "active_goals": [
                {"id": g.id, "domain": g.domain, "description": g.description}
                for g in model.learning_goals if g.status in ("pending", "active")
            ],
            "improvement_rate": (
                model.successful_improvements / model.total_improvements
                if model.total_improvements > 0 else 0.0
            ),
            "system_prompt_preview": model.to_system_prompt(),
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


# ============================================================================
# THEORY OF MIND (Phase 6C: User Mental Modeling)
# ============================================================================

@router.get("/theory-of-mind/status")
async def get_tom_status():
    """Get Theory of Mind engine status and summary."""
    try:
        from aura.proactive.theory_of_mind import get_theory_of_mind
        tom = get_theory_of_mind()
        status = tom.get_status()
        return {"active": True, **status}
    except Exception as e:
        return {"active": False, "error": safe_error_detail(e)}


@router.get("/theory-of-mind/model")
async def get_user_model():
    """Get the full user mental model."""
    try:
        from aura.proactive.theory_of_mind import get_theory_of_mind
        tom = get_theory_of_mind()
        return {
            "full_model": tom.get_full_model(),
            "style_guidance": tom.get_style_guidance(),
            "observations_for_inference": tom.get_observations_for_inference(),
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/theory-of-mind/topics")
async def get_topic_knowledge():
    """Get tracked topic knowledge levels."""
    try:
        from aura.proactive.theory_of_mind import get_theory_of_mind
        tom = get_theory_of_mind()
        return {
            "topics": tom.get_knowledge_summary(top_n=20),
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.post("/theory-of-mind/observe", dependencies=[Depends(require_api_key)])
async def observe_message(request: dict):
    """Manually feed a message for Theory of Mind to observe."""
    try:
        from aura.proactive.theory_of_mind import get_theory_of_mind
        tom = get_theory_of_mind()
        message = request.get("message", "")
        role = request.get("role", "user")
        if not message:
            return {"error": "message is required"}
        tom.observe_message(message, role=role)
        return {
            "success": True,
            "emotional_state": tom.get_emotional_state().to_dict(),
            "style": tom.get_communication_style().to_dict(),
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


# ============================================================================
# IDLE PRESENCE (Phase 6D: Genuine Idle Presence)
# ============================================================================

@router.get("/idle-presence/status")
async def get_idle_presence_status():
    """Get genuine idle presence engine status."""
    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine
        ipe = get_idle_presence_engine()
        return ipe.get_status()
    except Exception as e:
        return {"active": False, "error": safe_error_detail(e)}


@router.get("/idle-presence/state")
async def get_idle_presence_state():
    """Get full idle presence state with cognitive load and activities."""
    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine
        ipe = get_idle_presence_engine()
        return ipe.get_state()
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/idle-presence/cognitive-load")
async def get_cognitive_load():
    """Get current cognitive load breakdown."""
    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine
        ipe = get_idle_presence_engine()
        load = ipe.compute_cognitive_load()
        return {
            **load.to_dict(),
            "breath_rate": ipe.get_breath_rate_from_load(),
            "glow_intensity": ipe.get_glow_from_load(),
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/idle-presence/activities")
async def get_idle_activities():
    """Get recent background activities."""
    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine
        ipe = get_idle_presence_engine()
        return {"activities": ipe.get_recent_activities(limit=20)}
    except Exception as e:
        return {"error": safe_error_detail(e)}


# ============================================================================
# INTRINSIC MOTIVATION (Phase 6E: Drive System)
# ============================================================================

@router.get("/motivation/status")
async def get_motivation_status():
    """Get intrinsic motivation engine status."""
    try:
        from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
        im = get_intrinsic_motivation()
        return im.get_status()
    except Exception as e:
        return {"active": False, "error": safe_error_detail(e)}


@router.get("/motivation/drives")
async def get_drives():
    """Get current drive urgency levels."""
    try:
        from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
        im = get_intrinsic_motivation()
        return {"drives": im.get_drives_summary()}
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.post("/motivation/cycle")
async def run_motivation_cycle():
    """Run a full intrinsic motivation cycle."""
    try:
        from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
        im = get_intrinsic_motivation()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, im.run_motivation_cycle)
        return result
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/motivation/actions")
async def get_motivation_actions():
    """Get drive-motivated pending actions."""
    try:
        from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
        im = get_intrinsic_motivation()
        actions = im.generate_actions()
        return {
            "actions": [
                {"drive": a.drive.value, "action": a.action,
                 "description": a.description, "priority": round(a.priority, 2)}
                for a in actions
            ]
        }
    except Exception as e:
        return {"error": safe_error_detail(e)}


@router.get("/motivation/prompt")
async def get_motivation_prompt():
    """Get the motivation system prompt injection preview."""
    try:
        from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
        im = get_intrinsic_motivation()
        return {"prompt": im.get_context_for_prompt()}
    except Exception as e:
        return {"error": safe_error_detail(e)}
