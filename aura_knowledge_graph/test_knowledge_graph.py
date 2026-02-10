"""
Test the Knowledge Graph implementation.

Run with: python -m pytest aura_knowledge_graph/test_knowledge_graph.py -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

# Check if kuzu is available
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

from .schema import EntityType, get_schema_prompt, validate_relationship
from .graph_database import AURAKnowledgeGraph, Entity, Relationship
from .entity_extractor import EntityExtractor, ExtractionResult
from .titans_bridge import TitansKGBridge, BridgeConfig
from .query_engine import KGQueryEngine, QueryMode


class TestSchema(unittest.TestCase):
    """Test schema definitions."""

    def test_entity_types(self):
        """Test all entity types are defined."""
        self.assertEqual(EntityType.PERSON.value, "Person")
        self.assertEqual(EntityType.PROJECT.value, "Project")
        self.assertEqual(EntityType.TECHNOLOGY.value, "Technology")

    def test_get_schema_prompt(self):
        """Test schema prompt generation."""
        prompt = get_schema_prompt()
        self.assertIn("Entity Types:", prompt)
        self.assertIn("Person", prompt)
        self.assertIn("WORKS_ON", prompt)

    def test_validate_relationship(self):
        """Test relationship validation."""
        # Valid relationship
        self.assertTrue(validate_relationship(
            EntityType.PERSON, "WORKS_ON", EntityType.PROJECT
        ))

        # Generic RELATES_TO always allowed
        self.assertTrue(validate_relationship(
            EntityType.CONCEPT, "RELATES_TO", EntityType.TECHNOLOGY
        ))

        # CO_OCCURS always allowed
        self.assertTrue(validate_relationship(
            EntityType.PERSON, "CO_OCCURS", EntityType.PERSON
        ))


@unittest.skipUnless(KUZU_AVAILABLE, "Kùzu not installed")
class TestKnowledgeGraph(unittest.TestCase):
    """Test the Knowledge Graph database."""

    def setUp(self):
        """Create temporary database for testing."""
        # Kuzu needs a non-existent path for database creation
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "kg_db"
        # Remove the directory if it exists (Kuzu will create it)
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self.kg = AURAKnowledgeGraph(str(self.db_path))

    def tearDown(self):
        """Clean up test database."""
        self.kg.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_entity(self):
        """Test adding an entity."""
        entity = Entity(
            name="AURA",
            entity_type=EntityType.PROJECT,
            description="Proto-AGI system"
        )
        entity_id = self.kg.add_entity(entity)

        self.assertIsNotNone(entity_id)
        self.assertEqual(len(entity_id), 12)  # MD5 hash prefix

        # Retrieve entity
        retrieved = self.kg.get_entity_by_id(entity_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["name"], "AURA")
        self.assertEqual(retrieved["entity_type"], "Project")

    def test_add_entity_updates_existing(self):
        """Test that adding same entity updates importance."""
        entity = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            importance=0.5
        )

        # Add twice
        id1 = self.kg.add_entity(entity)
        id2 = self.kg.add_entity(entity)

        self.assertEqual(id1, id2)

        # Check importance increased
        retrieved = self.kg.get_entity_by_id(id1)
        self.assertGreater(retrieved["importance"], 0.5)

    def test_add_relationship(self):
        """Test adding a relationship."""
        # Add two entities
        e1 = Entity(name="Elnur", entity_type=EntityType.PERSON, description="Developer")
        e2 = Entity(name="AURA", entity_type=EntityType.PROJECT, description="AI system")

        id1 = self.kg.add_entity(e1)
        id2 = self.kg.add_entity(e2)

        # Add relationship
        rel = Relationship(
            source_id=id1,
            target_id=id2,
            relationship_type="WORKS_ON",
            evidence="Elnur is building AURA"
        )
        success = self.kg.add_relationship(rel)

        self.assertTrue(success)

        # Check relationship exists
        relationships = self.kg.get_relationships(id1)
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["relationship"], "WORKS_ON")

    def test_query_entities(self):
        """Test querying entities by text."""
        # Add entities
        self.kg.add_entity(Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            description="Programming language"
        ))
        self.kg.add_entity(Entity(
            name="JavaScript",
            entity_type=EntityType.TECHNOLOGY,
            description="Web language"
        ))

        # Query by name
        results = self.kg.query_entities("Python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Python")

        # Query by description
        results = self.kg.query_entities("language")
        self.assertEqual(len(results), 2)

    def test_query_entities_by_type(self):
        """Test filtering query by entity type."""
        self.kg.add_entity(Entity(name="Python", entity_type=EntityType.TECHNOLOGY))
        self.kg.add_entity(Entity(name="Alice", entity_type=EntityType.PERSON))

        results = self.kg.query_entities("", entity_type=EntityType.TECHNOLOGY)
        # Should only find Python
        names = [r["name"] for r in results]
        self.assertIn("Python", names)

    def test_get_related_entities(self):
        """Test getting entities within N hops."""
        # Create a chain: A -> B -> C
        e1 = self.kg.add_entity(Entity(name="A", entity_type=EntityType.CONCEPT))
        e2 = self.kg.add_entity(Entity(name="B", entity_type=EntityType.CONCEPT))
        e3 = self.kg.add_entity(Entity(name="C", entity_type=EntityType.CONCEPT))

        self.kg.add_relationship(Relationship(e1, e2, "RELATES_TO"))
        self.kg.add_relationship(Relationship(e2, e3, "RELATES_TO"))

        # Get related within 2 hops from A
        related = self.kg.get_related_entities(e1, hops=2)

        # Should find B (1 hop) and C (2 hops)
        names = [r["name"] for r in related]
        self.assertIn("B", names)
        self.assertIn("C", names)

    def test_decay_importance(self):
        """Test importance decay."""
        entity = Entity(name="Test", entity_type=EntityType.CONCEPT, importance=1.0)
        entity_id = self.kg.add_entity(entity)

        # Apply decay
        self.kg.decay_importance(decay_rate=0.1)

        # Check importance decreased
        retrieved = self.kg.get_entity_by_id(entity_id)
        self.assertLess(retrieved["importance"], 1.0)
        self.assertAlmostEqual(retrieved["importance"], 0.9, places=2)

    def test_get_statistics(self):
        """Test getting graph statistics."""
        # Add some data
        self.kg.add_entity(Entity(name="Test1", entity_type=EntityType.CONCEPT))
        self.kg.add_entity(Entity(name="Test2", entity_type=EntityType.PERSON))

        stats = self.kg.get_statistics()

        self.assertEqual(stats["total_entities"], 2)
        self.assertIn("entity_type_distribution", stats)
        self.assertEqual(stats["total_entities_added"], 2)

    def test_escape_special_characters(self):
        """Test handling of special characters in names."""
        entity = Entity(
            name="O'Brien's Project",
            entity_type=EntityType.PROJECT,
            description="A project with 'quotes' and \"double quotes\""
        )
        entity_id = self.kg.add_entity(entity)

        retrieved = self.kg.get_entity_by_id(entity_id)
        self.assertEqual(retrieved["name"], "O'Brien's Project")


class TestEntityExtractor(unittest.TestCase):
    """Test entity extraction."""

    def test_extraction_with_mock_llm(self):
        """Test extraction with mock LLM."""
        def mock_llm(prompt):
            return '''
            {
                "entities": [
                    {"name": "AURA", "type": "Project", "description": "AI system"},
                    {"name": "Elnur", "type": "Person", "description": "Developer"}
                ],
                "relationships": [
                    {"source": "Elnur", "relationship": "WORKS_ON", "target": "AURA", "evidence": "building AURA"}
                ]
            }
            '''

        extractor = EntityExtractor(mock_llm)
        result = extractor.extract("Elnur is building AURA, an AI system.")

        self.assertTrue(result.success)
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.relationships), 1)

        # Check entity details
        names = [e.name for e in result.entities]
        self.assertIn("AURA", names)
        self.assertIn("Elnur", names)

    def test_extraction_handles_invalid_json(self):
        """Test extraction handles invalid JSON gracefully."""
        def mock_llm(prompt):
            return "This is not valid JSON"

        extractor = EntityExtractor(mock_llm)
        result = extractor.extract("Some text")

        self.assertFalse(result.success)
        self.assertEqual(len(result.entities), 0)
        self.assertIn("No JSON", result.error)

    def test_extraction_handles_empty_response(self):
        """Test extraction handles empty entities."""
        def mock_llm(prompt):
            return '{"entities": [], "relationships": []}'

        extractor = EntityExtractor(mock_llm)
        result = extractor.extract("Some text")

        self.assertTrue(result.success)
        self.assertEqual(len(result.entities), 0)

    def test_incremental_extraction(self):
        """Test incremental extraction with existing entities."""
        def mock_llm(prompt):
            # Check that existing entities are mentioned in prompt
            if "Python" in prompt:
                return '''
                {
                    "entities": [
                        {"name": "Python", "type": "Technology", "description": "Existing"}
                    ],
                    "relationships": []
                }
                '''
            return '{"entities": [], "relationships": []}'

        extractor = EntityExtractor(mock_llm)
        result = extractor.extract_incremental("Using Python", ["Python", "JavaScript"])

        self.assertTrue(result.success)


@unittest.skipUnless(KUZU_AVAILABLE, "Kùzu not installed")
class TestTitansBridge(unittest.TestCase):
    """Test Titans-KG bridge."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "kg_db"
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self.kg = AURAKnowledgeGraph(str(self.db_path))

        def mock_llm(prompt):
            return '''
            {
                "entities": [
                    {"name": "TestEntity", "type": "Concept", "description": "Test"}
                ],
                "relationships": []
            }
            '''

        self.bridge = TitansKGBridge(
            knowledge_graph=self.kg,
            llm_func=mock_llm,
            config=BridgeConfig(
                surprise_threshold=0.3,
                batch_size=2,
                auto_extract=True
            )
        )

    def tearDown(self):
        """Clean up."""
        self.kg.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_force_extract(self):
        """Test force extraction."""
        entity_ids = self.bridge.force_extract("Test content about something")

        self.assertEqual(len(entity_ids), 1)
        self.assertEqual(self.bridge.total_entities_extracted, 1)

    def test_get_context_for_query(self):
        """Test getting context for a query."""
        # Add an entity first
        self.kg.add_entity(Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            description="Programming language"
        ))

        context = self.bridge.get_context_for_query("Tell me about Python")

        self.assertIn("Python", context)
        self.assertIn("KNOWLEDGE GRAPH", context)

    def test_extract_query_entities(self):
        """Test entity extraction from queries."""
        entities = self.bridge._extract_query_entities(
            'What is AURA and how does "Proto-AGI" work?'
        )

        self.assertIn("AURA", entities)
        self.assertIn("Proto-AGI", entities)

    def test_statistics(self):
        """Test bridge statistics."""
        stats = self.bridge.get_statistics()

        self.assertIn("total_traces_processed", stats)
        self.assertIn("config", stats)
        self.assertEqual(stats["config"]["surprise_threshold"], 0.3)


@unittest.skipUnless(KUZU_AVAILABLE, "Kùzu not installed")
class TestQueryEngine(unittest.TestCase):
    """Test the query engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "kg_db"
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self.kg = AURAKnowledgeGraph(str(self.db_path))
        self.engine = KGQueryEngine(self.kg)

        # Populate test data
        self.kg.add_entity(Entity(
            name="AURA",
            entity_type=EntityType.PROJECT,
            description="Proto-AGI system",
            importance=0.8
        ))
        self.kg.add_entity(Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            description="Programming language",
            importance=0.7
        ))

        # Add relationship
        aura = self.kg.query_entities("AURA", limit=1)[0]
        python = self.kg.query_entities("Python", limit=1)[0]
        self.kg.add_relationship(Relationship(
            source_id=aura["id"],
            target_id=python["id"],
            relationship_type="USES"
        ))

    def tearDown(self):
        """Clean up."""
        self.kg.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_entity_query(self):
        """Test entity query mode."""
        result = self.engine.query("AURA", mode=QueryMode.ENTITY)

        self.assertEqual(result.query_mode, QueryMode.ENTITY)
        self.assertTrue(len(result.entities) > 0)
        self.assertIn("AURA", result.context_string)

    def test_traversal_query(self):
        """Test traversal query mode."""
        result = self.engine.query("AURA", mode=QueryMode.TRAVERSAL, max_hops=2)

        self.assertEqual(result.query_mode, QueryMode.TRAVERSAL)
        # Should find AURA and Python (via relationship)
        names = [e["name"] for e in result.entities]
        self.assertIn("AURA", names)

    def test_global_query(self):
        """Test global query mode."""
        result = self.engine.query("", mode=QueryMode.GLOBAL)

        self.assertEqual(result.query_mode, QueryMode.GLOBAL)
        self.assertIn("OVERVIEW", result.context_string)

    def test_hybrid_query(self):
        """Test hybrid query mode."""
        result = self.engine.query("Python", mode=QueryMode.HYBRID)

        self.assertEqual(result.query_mode, QueryMode.HYBRID)
        self.assertTrue(len(result.entities) > 0)

    def test_answer_graph_question(self):
        """Test answering questions from graph."""
        answer = self.engine.answer_graph_question("What is AURA?")

        self.assertIn("AURA", answer)
        self.assertIn("Proto-AGI", answer)

    def test_get_entity_summary(self):
        """Test entity summary generation."""
        summary = self.engine.get_entity_summary("AURA")

        self.assertIn("AURA", summary)
        self.assertIn("Project", summary)
        self.assertIn("USES", summary)  # Relationship


@unittest.skipUnless(KUZU_AVAILABLE, "Kuzu not installed")
class TestTemporalKnowledgeGraph(unittest.TestCase):
    """Test bi-temporal edge features."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "kg_db"
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self.kg = AURAKnowledgeGraph(str(self.db_path))

        # Add two entities
        self.e1_id = self.kg.add_entity(Entity(
            name="Alice", entity_type=EntityType.PERSON, description="Engineer"
        ))
        self.e2_id = self.kg.add_entity(Entity(
            name="ProjectX", entity_type=EntityType.PROJECT, description="Secret project"
        ))

    def tearDown(self):
        self.kg.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_new_relationship_has_temporal_fields(self):
        """New edges should have valid_from, ingested_at, is_active=true."""
        import time
        before = int(time.time())
        self.kg.add_relationship(Relationship(
            self.e1_id, self.e2_id, "WORKS_ON", evidence="Alice works on ProjectX"
        ))
        after = int(time.time())

        history = self.kg.get_relationship_history(self.e1_id, self.e2_id)
        self.assertEqual(len(history), 1)
        edge = history[0]
        self.assertTrue(edge["is_active"])
        self.assertGreaterEqual(edge["valid_from"], before)
        self.assertLessEqual(edge["valid_from"], after)
        self.assertIsNone(edge["valid_to"])

    def test_invalidate_relationship(self):
        """Invalidated edges should have is_active=false and valid_to set."""
        self.kg.add_relationship(Relationship(self.e1_id, self.e2_id, "WORKS_ON"))
        result = self.kg.invalidate_relationship(self.e1_id, self.e2_id, "WORKS_ON")
        self.assertTrue(result)

        history = self.kg.get_relationship_history(self.e1_id, self.e2_id)
        self.assertEqual(len(history), 1)
        self.assertFalse(history[0]["is_active"])
        self.assertIsNotNone(history[0]["valid_to"])

    def test_invalidated_not_returned_by_default(self):
        """get_relationships should not return inactive edges by default."""
        self.kg.add_relationship(Relationship(self.e1_id, self.e2_id, "WORKS_ON"))
        self.kg.invalidate_relationship(self.e1_id, self.e2_id, "WORKS_ON")

        active = self.kg.get_relationships(self.e1_id)
        self.assertEqual(len(active), 0)

        # But include_inactive should return it
        all_rels = self.kg.get_relationships(self.e1_id, include_inactive=True)
        self.assertEqual(len(all_rels), 1)

    def test_get_relationships_at_time(self):
        """Point-in-time query should return edges valid at that timestamp."""
        import time

        # Use a past timestamp so invalidation (which uses time.time()) is strictly later
        t1 = int(time.time()) - 10
        self.kg.add_relationship(Relationship(
            self.e1_id, self.e2_id, "WORKS_ON", valid_from=t1
        ))

        # Before invalidation, query at t1 should find it
        rels_at_t1 = self.kg.get_relationships_at_time(self.e1_id, t1)
        self.assertGreaterEqual(len(rels_at_t1), 1)

        # Now invalidate (sets valid_to = now, which is > t1)
        self.kg.invalidate_relationship(self.e1_id, self.e2_id, "WORKS_ON")

        # Query at t1 should still find it (valid_from <= t1 AND valid_to > t1)
        rels_at_t1_after = self.kg.get_relationships_at_time(self.e1_id, t1)
        self.assertGreaterEqual(len(rels_at_t1_after), 1)

        # Query well before t1 should not find it
        rels_before = self.kg.get_relationships_at_time(self.e1_id, t1 - 100)
        works_on = [r for r in rels_before if r["relationship"] == "WORKS_ON"]
        self.assertEqual(len(works_on), 0)

    def test_get_relationship_history(self):
        """History should show all versions sorted by valid_from."""
        self.kg.add_relationship(Relationship(self.e1_id, self.e2_id, "WORKS_ON", evidence="v1"))
        self.kg.invalidate_relationship(self.e1_id, self.e2_id, "WORKS_ON")
        self.kg.add_relationship(Relationship(self.e1_id, self.e2_id, "WORKS_ON", evidence="v2"))

        history = self.kg.get_relationship_history(self.e1_id, self.e2_id)
        self.assertEqual(len(history), 2)
        # First should be inactive (old), second should be active (new)
        self.assertFalse(history[0]["is_active"])
        self.assertTrue(history[1]["is_active"])

    def test_supersede_relationship(self):
        """Supersede should invalidate old and create new."""
        self.kg.add_relationship(Relationship(self.e1_id, self.e2_id, "WORKS_ON", evidence="old"))
        result = self.kg.supersede_relationship(
            self.e1_id, self.e2_id, "WORKS_ON",
            new_evidence="new", new_weight=0.8
        )
        self.assertTrue(result)

        history = self.kg.get_relationship_history(self.e1_id, self.e2_id)
        self.assertEqual(len(history), 2)
        self.assertFalse(history[0]["is_active"])
        self.assertTrue(history[1]["is_active"])

    def test_traversal_respects_active_flag(self):
        """get_active_relationships should only return active edges."""
        self.kg.add_relationship(Relationship(self.e1_id, self.e2_id, "WORKS_ON"))
        active = self.kg.get_active_relationships(self.e1_id)
        self.assertEqual(len(active), 1)

        self.kg.invalidate_relationship(self.e1_id, self.e2_id, "WORKS_ON")
        active = self.kg.get_active_relationships(self.e1_id)
        self.assertEqual(len(active), 0)


if __name__ == "__main__":
    unittest.main()
