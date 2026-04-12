"""Smoke tests for SOTA deep_research upgrades."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
#  Imports under test
# ---------------------------------------------------------------------------
from aura.tools.deep_research import (
    DeepResearchTool,
    ResultRanker,
    ResearchCache,
    HierarchicalSummarizer,
    deep_research,
)


@pytest.fixture(autouse=True)
def _isolate_research_cache():
    """Restore ResearchCache after each test to prevent mock leakage.

    Earlier test modules may @patch ResearchCache at the module level.
    If those patches' teardown interleaves with our imports, the name
    binding can be left pointing at a MagicMock.
    """
    import aura.tools.deep_research as dr
    real_cls = ResearchCache
    yield
    dr.ResearchCache = real_cls

EXPECTED_KEYS = {
    "success", "topic", "depth", "queries_run", "urls_found", "pages_read",
    "time_seconds", "timed_out", "sources", "content", "summary",
    "synthesis", "page_summaries", "knowledge_gaps", "phases_completed",
    "cached_results",
}


# ---------------------------------------------------------------------------
#  1. Instantiation — bare (no LLM, no searcher)
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_instantiate_bare(mock_cache_cls):
    tool = DeepResearchTool()
    assert tool.name == "deep_research"
    assert tool.llm is None


# ---------------------------------------------------------------------------
#  2. Instantiation — with mock LLM
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_instantiate_with_llm(mock_cache_cls):
    mock_llm = MagicMock(return_value="mock answer")
    tool = DeepResearchTool(llm_func=mock_llm)
    assert tool.llm is mock_llm


# ---------------------------------------------------------------------------
#  3. set_llm post-init
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_set_llm(mock_cache_cls):
    tool = DeepResearchTool()
    assert tool.llm is None
    mock_llm = MagicMock(return_value="hello")
    tool.set_llm(mock_llm)
    assert tool.llm is mock_llm
    assert tool.summarizer.llm is mock_llm


# ---------------------------------------------------------------------------
#  4. research() returns dict with all expected keys
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_research_returns_all_keys(mock_cache_cls):
    mock_cache_cls.return_value.get_research_session.return_value = None
    tool = DeepResearchTool()
    # No search backends → immediate graceful return
    tool.searcher = None
    tool.tavily = None
    tool.brave = None
    tool.firecrawl = None
    result = tool.research("test topic", "quick")
    assert isinstance(result, dict)
    missing = EXPECTED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


# ---------------------------------------------------------------------------
#  5. Graceful fallback — no LLM, no searcher
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_graceful_no_backend(mock_cache_cls):
    mock_cache_cls.return_value.get_research_session.return_value = None
    tool = DeepResearchTool()
    tool.searcher = None
    tool.tavily = None
    tool.brave = None
    tool.firecrawl = None
    result = tool.research("anything", "standard")
    assert result["success"] is False
    assert result["phases_completed"] == 0
    assert result["queries_run"] == 0


# ---------------------------------------------------------------------------
#  6. run() entry point works
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_run_entry_point(mock_cache_cls):
    tool = DeepResearchTool()
    tool.searcher = None
    tool.tavily = None
    result = tool.run("quick test topic")
    assert isinstance(result, dict)
    assert "success" in result
    assert "topic" in result


# ---------------------------------------------------------------------------
#  7. deep_research() convenience function
# ---------------------------------------------------------------------------
@patch("aura.tools.deep_research.ResearchCache")
def test_convenience_function(mock_cache_cls):
    with patch("aura.tools.deep_research.DeepResearchTool") as MockTool:
        mock_inst = MagicMock()
        mock_inst.research.return_value = {"success": True, "topic": "test"}
        MockTool.return_value = mock_inst
        result = deep_research("test", "quick")
        assert result["success"] is True
        mock_inst.research.assert_called_once_with("test", "quick")


# ---------------------------------------------------------------------------
#  8. ResultRanker — instantiate + rank_results
# ---------------------------------------------------------------------------
def test_result_ranker():
    ranker = ResultRanker()
    sample = [
        {"url": "https://arxiv.org/abs/1234", "title": "ML Paper", "snippet": "A study on deep learning results from 2024"},
        {"url": "https://reddit.com/r/test", "title": "Reddit post", "snippet": "short"},
        {"url": "https://example.com", "title": "Example", "snippet": ""},
    ]
    ranked = ranker.rank_results(sample, "deep learning")
    assert len(ranked) == 3
    assert all("_rank_score" in r for r in ranked)
    # arxiv should rank higher than reddit
    arxiv = next(r for r in ranked if "arxiv" in r["url"])
    reddit = next(r for r in ranked if "reddit" in r["url"])
    assert arxiv["_rank_score"] >= reddit["_rank_score"]


# ---------------------------------------------------------------------------
#  9. ResearchCache — set/get roundtrip with temp DB
# ---------------------------------------------------------------------------
def test_research_cache_roundtrip(tmp_path):
    db_path = str(tmp_path / "test_cache.db")
    cache = ResearchCache(db_path=db_path)
    assert cache.get_search("hello") is None

    data = [{"url": "https://example.com", "title": "Test"}]
    cache.set_search("hello", data)
    got = cache.get_search("hello")
    assert got is not None
    assert got[0]["url"] == "https://example.com"

    # Close connection so Windows can clean up tmp_path
    if cache._conn:
        cache._conn.close()


# ---------------------------------------------------------------------------
#  10. HierarchicalSummarizer — no-LLM fallback
# ---------------------------------------------------------------------------
def test_summarizer_no_llm_fallback():
    s = HierarchicalSummarizer(llm_func=None)
    # summarize_page should truncate instead of crashing
    result = s.summarize_page("https://x.com", "A" * 600, "topic")
    assert isinstance(result, str)
    assert len(result) <= 600

    # synthesize with no LLM should concatenate
    summaries = [{"url": "https://a.com", "summary": "Summary A"}]
    synth = s.synthesize(summaries, "topic")
    assert "Summary A" in synth

    # identify_gaps with no LLM returns empty
    gaps = s.identify_gaps(summaries, "topic")
    assert gaps == []
