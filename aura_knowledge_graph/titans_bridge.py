"""
Bridge between Titans Neural Memory and Knowledge Graph.

This is the KEY integration point:
- Monitors Titans for high-surprise events
- Triggers entity extraction
- Stores extracted knowledge in graph
- Provides graph context for retrieval
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .entity_extractor import EntityExtractor
from .graph_database import AURAKnowledgeGraph, Entity, Relationship
from .schema import EntityType

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for Titans-KG bridge."""
    surprise_threshold: float = 0.5  # Minimum surprise to trigger extraction
    batch_size: int = 5  # Number of traces to batch before extraction
    auto_extract: bool = True  # Automatically extract on high surprise
    create_co_occurrence: bool = True  # Create relationships between co-occurring entities
    max_queue_age_seconds: int = 300  # Max age before forcing queue flush


class TitansKGBridge:
    """
    Bridge connecting Titans Neural Memory to Knowledge Graph.

    Workflow:
    1. Titans stores a memory trace with surprise score
    2. If surprise > threshold, queue for extraction
    3. Extract entities using LLM
    4. Store in Knowledge Graph
    5. Link back to source memory trace

    This creates a complementary memory system:
    - Titans = Fast, episodic, surprise-based working memory
    - KG = Slow, structured, relationship-based long-term memory
    """

    def __init__(
        self,
        knowledge_graph: AURAKnowledgeGraph,
        llm_func: Callable[[str], str],
        config: Optional[BridgeConfig] = None
    ):
        """
        Initialize the bridge.

        Args:
            knowledge_graph: The Kùzu knowledge graph instance
            llm_func: LLM function for entity extraction
            config: Bridge configuration
        """
        self.kg = knowledge_graph
        self.extractor = EntityExtractor(llm_func)
        self.config = config or BridgeConfig()

        # Extraction queue for batching
        self.extraction_queue: List[Dict] = []
        self._queue_start_time: Optional[float] = None

        # Statistics
        self.total_traces_processed = 0
        self.total_entities_extracted = 0
        self.total_extractions_triggered = 0

    def on_memory_stored(self, trace: Any) -> Optional[List[str]]:
        """
        Called when Titans stores a new memory trace.

        This should be hooked into Titans Memory's store() method:

        ```python
        # In TitansMemory.store():
        trace = self._create_trace(content, context)
        self.long_term[trace.trace_id] = trace

        # Notify bridge
        if self.kg_bridge:
            self.kg_bridge.on_memory_stored(trace)
        ```

        Args:
            trace: The memory trace from Titans (has .content, .surprise_score, .trace_id)

        Returns:
            List of entity IDs if extraction was performed, None otherwise
        """
        self.total_traces_processed += 1

        # Check if surprise exceeds threshold
        surprise = getattr(trace, 'surprise_score', 0)
        if surprise < self.config.surprise_threshold:
            return None

        if not self.config.auto_extract:
            return None

        # Initialize queue timer
        if self._queue_start_time is None:
            self._queue_start_time = time.time()

        # Add to extraction queue
        self.extraction_queue.append({
            "trace_id": getattr(trace, 'trace_id', str(time.time())),
            "content": str(getattr(trace, 'content', trace)),
            "surprise": surprise,
            "timestamp": time.time()
        })

        logger.debug(
            f"[TitansBridge] Queued trace (surprise={surprise:.2f}), "
            f"queue size: {len(self.extraction_queue)}"
        )

        # Process queue if batch size reached or queue is old
        should_process = (
            len(self.extraction_queue) >= self.config.batch_size or
            (self._queue_start_time and
             time.time() - self._queue_start_time > self.config.max_queue_age_seconds)
        )

        if should_process:
            return self._process_extraction_queue()

        return None

    def _process_extraction_queue(self) -> List[str]:
        """Process all queued traces for extraction."""
        if not self.extraction_queue:
            return []

        self.total_extractions_triggered += 1
        logger.info(
            f"[TitansBridge] Processing extraction queue "
            f"({len(self.extraction_queue)} traces)"
        )

        # Combine queued content
        combined_text = "\n\n---\n\n".join([
            item["content"] for item in self.extraction_queue
        ])

        # Get existing entity names for deduplication
        existing = self._get_existing_entity_names()

        # Extract entities
        if existing:
            result = self.extractor.extract_incremental(combined_text, existing)
        else:
            result = self.extractor.extract(combined_text, context="memory traces")

        if not result.success:
            logger.warning(f"[TitansBridge] Extraction failed: {result.error}")
            self.extraction_queue = []
            self._queue_start_time = None
            return []

        # Derive valid_from from earliest trace timestamp in the batch
        earliest_ts = int(min(item["timestamp"] for item in self.extraction_queue))

        # Store entities in graph
        entity_ids = []
        for entity in result.entities:
            eid = self.kg.add_entity(entity)
            entity_ids.append(eid)
            self.total_entities_extracted += 1

        # Store relationships with valid_from from trace timestamps
        for rel in result.relationships:
            rel.valid_from = earliest_ts
            self.kg.add_relationship(rel)

        # Create co-occurrence relationships if enabled
        if self.config.create_co_occurrence and len(entity_ids) > 1:
            self._create_co_occurrences(entity_ids, combined_text, valid_from=earliest_ts)

        # Clear queue
        self.extraction_queue = []
        self._queue_start_time = None

        logger.info(
            f"[TitansBridge] Extracted {len(entity_ids)} entities, "
            f"{len(result.relationships)} relationships"
        )

        return entity_ids

    def _get_existing_entity_names(self, limit: int = 100) -> List[str]:
        """Get names of existing entities for deduplication."""
        return self.kg.get_all_entity_names(limit=limit)

    def _create_co_occurrences(self, entity_ids: List[str], evidence: str, valid_from: Optional[int] = None):
        """Create CO_OCCURS relationships between entities found together."""
        evidence_short = evidence[:200] + "..." if len(evidence) > 200 else evidence

        for i, eid1 in enumerate(entity_ids):
            for eid2 in entity_ids[i + 1:]:
                self.kg.add_relationship(Relationship(
                    source_id=eid1,
                    target_id=eid2,
                    relationship_type="CO_OCCURS",
                    weight=0.5,
                    evidence=evidence_short,
                    valid_from=valid_from,
                ))

    def force_extract(self, text: str, context: str = "manual extraction") -> List[str]:
        """
        Force extraction from text regardless of surprise threshold.
        Useful for importing documents or manual knowledge entry.
        """
        logger.info(f"[TitansBridge] Force extraction from: {context}")

        existing = self._get_existing_entity_names()

        if existing:
            result = self.extractor.extract_incremental(text, existing)
        else:
            result = self.extractor.extract(text, context)

        if not result.success:
            logger.warning(f"[TitansBridge] Force extraction failed: {result.error}")
            return []

        entity_ids = []
        for entity in result.entities:
            eid = self.kg.add_entity(entity)
            entity_ids.append(eid)
            self.total_entities_extracted += 1

        for rel in result.relationships:
            self.kg.add_relationship(rel)

        logger.info(
            f"[TitansBridge] Force extracted {len(entity_ids)} entities, "
            f"{len(result.relationships)} relationships"
        )

        return entity_ids

    def get_context_for_query(self, query: str, max_entities: int = 5) -> str:
        """
        Get relevant graph context for a user query.

        This context should be injected into the LLM prompt to augment
        responses with knowledge graph information.

        Args:
            query: The user's query
            max_entities: Maximum number of entities to include

        Returns:
            Formatted context string for LLM prompt
        """
        # Extract potential entity names from query
        query_words = self._extract_query_entities(query)

        context_parts = []
        seen_entities = set()

        for word in query_words:
            if len(seen_entities) >= max_entities:
                break

            # Search for matching entities
            entities = self.kg.query_entities(word, limit=2)

            for entity in entities:
                if entity["id"] in seen_entities:
                    continue

                seen_entities.add(entity["id"])

                # Boost importance when accessed
                self.kg.boost_importance(entity["id"], boost=0.05)

                # Format entity info
                context_parts.append(
                    f"[{entity['entity_type']}] {entity['name']}: {entity.get('description', 'No description')}"
                )

                # Get related entities
                related = self.kg.get_related_entities(entity["id"], hops=1, limit=3)
                for rel in related:
                    context_parts.append(
                        f"  └─ related to [{rel['entity_type']}] {rel['name']}"
                    )

        if not context_parts:
            return ""

        return "KNOWLEDGE GRAPH CONTEXT:\n" + "\n".join(context_parts)

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract potential entity names from a query.
        Simple implementation - looks for capitalized words and known patterns.
        """
        # Find capitalized words/phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)

        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', query)

        # Find known entity patterns (AURA, AI, etc.)
        known_patterns = re.findall(
            r'\b(?:AURA|AI|AGI|LLM|API|GPU|CPU|RAM|SSD|HDD|USB|HTTP|HTTPS|JSON|XML|HTML|CSS|SQL)\b',
            query,
            re.IGNORECASE
        )

        # Find words longer than 4 characters (potential entity names)
        long_words = re.findall(r'\b[A-Za-z]{5,}\b', query)

        # Combine and deduplicate, preserving order
        all_entities = capitalized + quoted + known_patterns + long_words[:5]
        seen = set()
        unique = []
        for e in all_entities:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique.append(e)

        return unique[:10]  # Limit to 10 potential entities

    def flush(self) -> List[str]:
        """Force process any remaining items in extraction queue."""
        return self._process_extraction_queue()

    def get_statistics(self) -> Dict:
        """Get bridge statistics."""
        return {
            "total_traces_processed": self.total_traces_processed,
            "total_entities_extracted": self.total_entities_extracted,
            "total_extractions_triggered": self.total_extractions_triggered,
            "queue_size": len(self.extraction_queue),
            "config": {
                "surprise_threshold": self.config.surprise_threshold,
                "batch_size": self.config.batch_size,
                "auto_extract": self.config.auto_extract,
                "create_co_occurrence": self.config.create_co_occurrence
            }
        }
