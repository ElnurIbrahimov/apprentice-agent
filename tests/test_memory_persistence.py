import pytest
import tempfile
from pathlib import Path


def test_store_fact_dedup():
    """store_fact() should not create duplicate entries for same content."""
    from aura.truth_spine import VerifiedMemory, VerificationResult, Artifact, ArtifactType
    with tempfile.TemporaryDirectory() as tmpdir:
        vm = VerifiedMemory(data_dir=Path(tmpdir))
        artifact = Artifact(
            artifact_id="test-art-1",
            artifact_type=ArtifactType.NONE,
            content_hash="",
            raw_data=None,
            metadata={}
        )
        v = VerificationResult(
            is_verified=True,
            artifact=artifact,
            checks_passed=["test"],
            checks_failed=[],
            reasoning="test"
        )
        t1 = vm.store_fact("The sky is blue", v, "test")
        t2 = vm.store_fact("The sky is blue", v, "test")
        assert t1.trace_id == t2.trace_id
        assert len(vm.traces) == 1


def test_knowledge_graph_edge_weight_persists(tmp_path):
    """strengthen_edge() should persist to disk."""
    from aura.tools.knowledge_graph import KnowledgeGraphTool
    kg = KnowledgeGraphTool(db_path=str(tmp_path))
    n1 = kg.add_node("concept", "NodeA")
    n2 = kg.add_node("concept", "NodeB")
    edge = kg.add_edge(n1.id, n2.id, "related", weight=0.5)
    edge_id = edge.id
    original_weight = edge.weight
    kg.strengthen_edge(edge_id, 0.2)
    kg2 = KnowledgeGraphTool(db_path=str(tmp_path))
    new_weight = kg2._edges[edge_id].weight
    assert new_weight > original_weight
