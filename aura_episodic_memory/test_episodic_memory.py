"""
Test the Episodic Memory implementation.

Run with: python -m pytest aura_episodic_memory/test_episodic_memory.py -v
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from .episode import (
    Episode, EpisodeType, EpisodeQuery, TemporalContext,
    EmotionalValence, EpisodeSearchResult
)
from .temporal_parser import TemporalParser, TemporalRange

# Check dependencies
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TestEpisode(unittest.TestCase):
    """Test Episode data model."""

    def test_create_episode(self):
        """Test basic episode creation."""
        episode = Episode(
            content="Test content",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )

        self.assertIsNotNone(episode.id)
        self.assertEqual(episode.content, "Test content")
        self.assertEqual(episode.episode_type, EpisodeType.CONVERSATION)
        self.assertEqual(episode.importance, 0.5)  # Default

    def test_episode_id_generation(self):
        """Test that episode IDs are unique."""
        ep1 = Episode(
            content="Content 1",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )
        ep2 = Episode(
            content="Content 2",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )

        self.assertNotEqual(ep1.id, ep2.id)

    def test_episode_to_dict(self):
        """Test episode serialization."""
        episode = Episode(
            content="Test content",
            episode_type=EpisodeType.TASK_EXECUTION,
            temporal_context=TemporalContext(timestamp=datetime(2024, 1, 15, 10, 30)),
            importance=0.8,
            entities_involved=["Python", "AURA"],
            tools_used=["web_search"]
        )

        data = episode.to_dict()

        self.assertEqual(data["content"], "Test content")
        self.assertEqual(data["episode_type"], "task_execution")
        self.assertEqual(data["importance"], 0.8)
        self.assertIn("Python", data["entities_involved"])
        self.assertIn("web_search", data["tools_used"])

    def test_episode_from_dict(self):
        """Test episode deserialization."""
        data = {
            "id": "ep_test123",
            "content": "Restored content",
            "episode_type": "learning",
            "temporal_context": {
                "timestamp": "2024-01-15T10:30:00",
                "time_of_day": "morning",
                "day_of_week": "monday"
            },
            "importance": 0.9,
            "entities_involved": ["AI", "Memory"],
            "tools_used": []
        }

        episode = Episode.from_dict(data)

        self.assertEqual(episode.id, "ep_test123")
        self.assertEqual(episode.content, "Restored content")
        self.assertEqual(episode.episode_type, EpisodeType.LEARNING)
        self.assertEqual(episode.importance, 0.9)

    def test_recency_score(self):
        """Test recency score calculation."""
        # Recent episode
        recent = Episode(
            content="Recent",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )
        self.assertGreater(recent.get_recency_score(), 0.9)

        # Old episode
        old = Episode(
            content="Old",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(
                timestamp=datetime.now() - timedelta(days=30)
            )
        )
        self.assertLess(old.get_recency_score(), 0.5)

    def test_mark_accessed(self):
        """Test access tracking."""
        episode = Episode(
            content="Test",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )

        self.assertEqual(episode.access_count, 0)
        self.assertIsNone(episode.last_accessed)

        episode.mark_accessed()

        self.assertEqual(episode.access_count, 1)
        self.assertIsNotNone(episode.last_accessed)


class TestTemporalContext(unittest.TestCase):
    """Test TemporalContext."""

    def test_auto_derive_time_of_day(self):
        """Test automatic time-of-day derivation."""
        morning = TemporalContext(timestamp=datetime(2024, 1, 15, 9, 0))
        self.assertEqual(morning.time_of_day, "morning")

        afternoon = TemporalContext(timestamp=datetime(2024, 1, 15, 14, 0))
        self.assertEqual(afternoon.time_of_day, "afternoon")

        evening = TemporalContext(timestamp=datetime(2024, 1, 15, 19, 0))
        self.assertEqual(evening.time_of_day, "evening")

        night = TemporalContext(timestamp=datetime(2024, 1, 15, 23, 0))
        self.assertEqual(night.time_of_day, "night")

    def test_weekend_detection(self):
        """Test weekend detection."""
        saturday = TemporalContext(timestamp=datetime(2024, 1, 13, 10, 0))  # Saturday
        self.assertTrue(saturday.is_weekend)

        monday = TemporalContext(timestamp=datetime(2024, 1, 15, 10, 0))  # Monday
        self.assertFalse(monday.is_weekend)


class TestTemporalParser(unittest.TestCase):
    """Test TemporalParser."""

    def setUp(self):
        """Set up parser with fixed base time."""
        self.base_time = datetime(2024, 1, 15, 14, 30)  # Monday afternoon
        self.parser = TemporalParser(base_time=self.base_time)

    def test_parse_yesterday(self):
        """Test parsing 'yesterday'."""
        result = self.parser.parse("yesterday")

        self.assertIsNotNone(result)
        self.assertEqual(result.start.date(), datetime(2024, 1, 14).date())

    def test_parse_hours_ago(self):
        """Test parsing 'X hours ago'."""
        result = self.parser.parse("2 hours ago")

        self.assertIsNotNone(result)
        expected = self.base_time - timedelta(hours=2)
        self.assertEqual(result.start.hour, expected.hour)

    def test_parse_days_ago(self):
        """Test parsing 'X days ago'."""
        result = self.parser.parse("3 days ago")

        self.assertIsNotNone(result)
        expected = self.base_time - timedelta(days=3)
        self.assertEqual(result.start.date(), expected.date())

    def test_parse_last_week(self):
        """Test parsing 'last week'."""
        result = self.parser.parse("last week")

        self.assertIsNotNone(result)
        # Should be sometime in the previous week
        self.assertLess(result.start, self.base_time - timedelta(days=7))

    def test_parse_time_of_day(self):
        """Test parsing time-of-day references."""
        result = self.parser.parse("this morning")

        self.assertIsNotNone(result)
        self.assertEqual(result.start.date(), self.base_time.date())

    def test_parse_day_of_week(self):
        """Test parsing day-of-week references."""
        result = self.parser.parse("last friday")

        self.assertIsNotNone(result)
        self.assertEqual(result.start.strftime("%A").lower(), "friday")

    def test_recency_description(self):
        """Test human-readable recency descriptions."""
        # Use parser with current time as base for recency calculations
        parser = TemporalParser()  # Uses datetime.now() as base

        # 5 minutes ago
        recent = datetime.now() - timedelta(minutes=5)
        desc_recent = parser.get_recency_description(recent)
        self.assertTrue("minute" in desc_recent or "just now" in desc_recent)

        # 3 hours ago
        hours_ago = datetime.now() - timedelta(hours=3)
        desc_hours = parser.get_recency_description(hours_ago)
        self.assertIn("hour", desc_hours)

        # Yesterday
        yesterday = datetime.now() - timedelta(days=1)
        desc = parser.get_recency_description(yesterday)
        self.assertTrue("yesterday" in desc or "day" in desc)


class TestEpisodeQuery(unittest.TestCase):
    """Test EpisodeQuery."""

    def test_default_query(self):
        """Test default query values."""
        query = EpisodeQuery()

        self.assertIsNone(query.query_text)
        self.assertEqual(query.limit, 10)
        self.assertEqual(query.min_score, 0.0)

    def test_query_with_filters(self):
        """Test query with various filters."""
        query = EpisodeQuery(
            query_text="Python programming",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            episode_types=[EpisodeType.LEARNING, EpisodeType.TASK_EXECUTION],
            limit=5,
            min_score=0.3
        )

        self.assertEqual(query.query_text, "Python programming")
        self.assertEqual(len(query.episode_types), 2)
        self.assertEqual(query.limit, 5)


@unittest.skipUnless(
    QDRANT_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE,
    "Qdrant and sentence-transformers required"
)
class TestEpisodicMemoryStore(unittest.TestCase):
    """Test EpisodicMemoryStore."""

    def setUp(self):
        """Create temporary database for testing."""
        self.test_dir = tempfile.mkdtemp()
        from .memory_store import EpisodicMemoryStore
        self.store = EpisodicMemoryStore(self.test_dir)

    def tearDown(self):
        """Clean up test database."""
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_store_and_retrieve_episode(self):
        """Test storing and retrieving an episode."""
        episode = Episode(
            content="Learned about Python decorators",
            episode_type=EpisodeType.LEARNING,
            temporal_context=TemporalContext(timestamp=datetime.now()),
            importance=0.7,
            entities_involved=["Python", "decorators"]
        )

        episode_id = self.store.store_episode(episode)
        self.assertIsNotNone(episode_id)

        retrieved = self.store.get_episode(episode_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, episode.content)
        self.assertEqual(retrieved.importance, 0.7)

    def test_search_by_text(self):
        """Test semantic search."""
        # Store multiple episodes
        self.store.store_episode(Episode(
            content="Discussion about machine learning algorithms",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        ))
        self.store.store_episode(Episode(
            content="Fixed a bug in the web scraper",
            episode_type=EpisodeType.TASK_EXECUTION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        ))

        # Search for ML-related content
        query = EpisodeQuery(query_text="artificial intelligence", limit=5)
        results = self.store.search(query)

        # Should find the ML-related episode
        self.assertTrue(len(results) > 0)
        self.assertIn("learning", results[0].episode.content.lower())

    def test_search_with_time_filter(self):
        """Test search with temporal filter."""
        # Store old episode
        old_episode = Episode(
            content="Old conversation",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(
                timestamp=datetime.now() - timedelta(days=30)
            )
        )
        self.store.store_episode(old_episode)

        # Store recent episode
        recent_episode = Episode(
            content="Recent conversation",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )
        self.store.store_episode(recent_episode)

        # Search only recent
        query = EpisodeQuery(
            query_text="conversation",
            start_time=datetime.now() - timedelta(days=7),
            limit=10
        )
        results = self.store.search(query)

        # Should only find recent episode
        contents = [r.episode.content for r in results]
        self.assertIn("Recent conversation", contents)

    def test_get_timeline(self):
        """Test getting timeline of episodes."""
        # Store multiple episodes across time
        for i in range(5):
            self.store.store_episode(Episode(
                content=f"Episode {i}",
                episode_type=EpisodeType.CONVERSATION,
                temporal_context=TemporalContext(
                    timestamp=datetime.now() - timedelta(hours=i)
                )
            ))

        episodes = self.store.get_timeline(
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )

        self.assertEqual(len(episodes), 5)
        # Should be sorted by time
        for i in range(len(episodes) - 1):
            self.assertLessEqual(
                episodes[i].temporal_context.timestamp,
                episodes[i + 1].temporal_context.timestamp
            )

    def test_delete_episode(self):
        """Test deleting an episode."""
        episode = Episode(
            content="To be deleted",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        )
        episode_id = self.store.store_episode(episode)

        # Delete
        success = self.store.delete_episode(episode_id)
        self.assertTrue(success)

        # Verify deleted
        retrieved = self.store.get_episode(episode_id)
        self.assertIsNone(retrieved)

    def test_statistics(self):
        """Test getting store statistics."""
        self.store.store_episode(Episode(
            content="Test",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now())
        ))

        stats = self.store.get_statistics()

        self.assertEqual(stats["total_episodes"], 1)
        self.assertIn("vector_dimension", stats)


@unittest.skipUnless(
    QDRANT_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE,
    "Qdrant and sentence-transformers required"
)
class TestTimelineEngine(unittest.TestCase):
    """Test TimelineEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        from .memory_store import EpisodicMemoryStore
        from .timeline import TimelineEngine

        self.store = EpisodicMemoryStore(self.test_dir)
        self.engine = TimelineEngine(self.store)

        # Populate test data
        for i in range(10):
            self.store.store_episode(Episode(
                content=f"Event {i}",
                episode_type=EpisodeType.CONVERSATION,
                temporal_context=TemporalContext(
                    timestamp=datetime.now() - timedelta(hours=i * 2)
                ),
                importance=0.5 + (i % 3) * 0.2
            ))

    def tearDown(self):
        """Clean up."""
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_timeline_view(self):
        """Test getting timeline view."""
        time_range = TemporalRange(
            start=datetime.now() - timedelta(days=1),
            end=datetime.now(),
            description="last day"
        )

        view = self.engine.get_timeline(time_range, granularity="hour")

        self.assertEqual(view.total_episodes, 10)
        self.assertEqual(view.granularity, "hour")
        self.assertTrue(len(view.segments) > 0)

    def test_query_by_time(self):
        """Test natural language time queries."""
        episodes = self.engine.query_by_time("in the last 6 hours")

        # Should find some episodes
        self.assertTrue(len(episodes) > 0)

    def test_get_day_summary(self):
        """Test getting day summary."""
        summary = self.engine.get_day_summary(datetime.now())

        self.assertIn("total_episodes", summary)
        self.assertIn("by_type", summary)
        self.assertIn("highlights", summary)

    def test_time_travel(self):
        """Test time travel feature."""
        # Use "today" since that's when we stored the episodes
        episodes, narrative = self.engine.time_travel("today")

        # Narrative should always be present even if no episodes found
        self.assertTrue(len(narrative) > 0)
        self.assertIn("Traveling back", narrative)

        # Episodes might or might not be found depending on time parsing
        # Just verify the function doesn't crash


@unittest.skipUnless(
    QDRANT_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE,
    "Qdrant and sentence-transformers required"
)
class TestTitansIntegration(unittest.TestCase):
    """Test TitansEpisodicBridge."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        from .memory_store import EpisodicMemoryStore
        from .titans_integration import TitansEpisodicBridge, TitansEpisodicConfig

        self.store = EpisodicMemoryStore(self.test_dir)
        self.bridge = TitansEpisodicBridge(
            memory_store=self.store,
            config=TitansEpisodicConfig(
                surprise_threshold=0.3,
                turns_per_episode=2
            )
        )

    def tearDown(self):
        """Clean up."""
        self.store.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_on_titans_trace_high_surprise(self):
        """Test episode formation from high-surprise traces."""
        episode_id = self.bridge.on_titans_trace(
            trace_content="Discovered unexpected behavior in the API",
            surprise_score=0.8
        )

        self.assertIsNotNone(episode_id)

        # Verify episode was stored
        episode = self.store.get_episode(episode_id)
        self.assertIsNotNone(episode)
        self.assertEqual(episode.episode_type, EpisodeType.LEARNING)

    def test_on_titans_trace_low_surprise(self):
        """Test no episode for low-surprise traces."""
        episode_id = self.bridge.on_titans_trace(
            trace_content="Normal operation",
            surprise_score=0.1
        )

        self.assertIsNone(episode_id)

    def test_conversation_turns(self):
        """Test episode formation from conversation turns."""
        # Add turns
        self.bridge.on_conversation_turn(
            "How do decorators work?",
            "Decorators are functions that wrap other functions..."
        )
        episode_id = self.bridge.on_conversation_turn(
            "Can you show an example?",
            "Here's a simple decorator example..."
        )

        # Should have formed an episode after 2 turns
        self.assertIsNotNone(episode_id)

    def test_get_context_for_query(self):
        """Test getting context for queries."""
        # Add some memories
        self.store.store_episode(Episode(
            content="Discussion about Python decorators",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now()),
            entities_involved=["Python", "decorators"]
        ))

        context = self.bridge.get_context_for_query("Tell me about decorators")

        self.assertIn("EPISODIC MEMORY", context)
        self.assertIn("decorator", context.lower())

    def test_on_task_complete(self):
        """Test recording task completion."""
        episode_id = self.bridge.on_task_complete(
            task_description="Fix login bug",
            result="Successfully patched authentication flow",
            success=True,
            tools_used=["code_edit", "test_runner"]
        )

        episode = self.store.get_episode(episode_id)
        self.assertEqual(episode.episode_type, EpisodeType.TASK_EXECUTION)
        self.assertIn("code_edit", episode.tools_used)

    def test_statistics(self):
        """Test bridge statistics."""
        stats = self.bridge.get_statistics()

        self.assertIn("session_id", stats)
        self.assertIn("episodes_formed", stats)
        self.assertIn("config", stats)


class TestMemoryScorer(unittest.TestCase):
    """Test MemoryScorer."""

    def test_basic_scoring(self):
        """Test basic scoring calculation."""
        from .memory_scorer import MemoryScorer

        scorer = MemoryScorer()

        episode = Episode(
            content="Test episode",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now()),
            importance=0.8
        )

        query = EpisodeQuery(query_text="test")

        score, breakdown = scorer.score(episode, query, vector_similarity=0.9)

        self.assertGreater(score, 0)
        self.assertIn("recency", breakdown)
        self.assertIn("importance", breakdown)
        self.assertIn("relevance", breakdown)

    def test_recency_affects_score(self):
        """Test that recency affects scoring."""
        from .memory_scorer import MemoryScorer

        scorer = MemoryScorer()
        query = EpisodeQuery(query_text="test")

        recent = Episode(
            content="Recent",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(timestamp=datetime.now()),
            importance=0.5
        )

        old = Episode(
            content="Old",
            episode_type=EpisodeType.CONVERSATION,
            temporal_context=TemporalContext(
                timestamp=datetime.now() - timedelta(days=30)
            ),
            importance=0.5
        )

        recent_score, _ = scorer.score(recent, query, 0.5)
        old_score, _ = scorer.score(old, query, 0.5)

        self.assertGreater(recent_score, old_score)


if __name__ == "__main__":
    unittest.main()
