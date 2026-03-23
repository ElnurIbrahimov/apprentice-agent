"""
Benchmark suite for Aura memory systems at scale.

Tests:
  1. A-MEM write throughput (1000 inserts)
  2. A-MEM search latency at scale (100 queries over 1000 memories)
  3. Write gate throughput (1000 candidates scored)
  4. Episodic memory write + search (500 inserts, 50 queries)

Run:
  python tests/bench_memory_systems.py          # standalone
  pytest tests/bench_memory_systems.py -v -s    # via pytest

Each benchmark uses temp directories and cleans up after itself.
Skips gracefully when backends (qdrant, sentence-transformers) aren't installed.
"""

import os
import sys
import shutil
import statistics
import tempfile
import time
import random
import string
from collections import Counter
from datetime import datetime, timedelta

# Ensure project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_qdrant():
    try:
        import qdrant_client  # noqa: F401
        return True
    except ImportError:
        return False


def _check_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


HAS_QDRANT = _check_qdrant()
HAS_ST = _check_sentence_transformers()

# Try importing the write gate (pure Python, no heavy deps)
try:
    from aura.memory.write_gate import (
        MemoryWriteGate, MemoryCandidate, MemoryDecisionKind,
    )
    HAS_WRITE_GATE = True
except ImportError:
    HAS_WRITE_GATE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TOPICS = [
    "machine learning", "neural networks", "Python programming",
    "database optimization", "REST API design", "user authentication",
    "natural language processing", "computer vision", "data pipelines",
    "distributed systems", "memory management", "caching strategies",
    "microservices architecture", "event-driven design", "CI/CD pipelines",
    "graph databases", "reinforcement learning", "attention mechanisms",
    "transformer models", "embedding spaces", "vector search",
    "knowledge graphs", "causal inference", "Bayesian optimization",
    "genetic algorithms", "sentiment analysis", "speech recognition",
    "image segmentation", "object detection", "recommender systems",
]

_SEARCH_QUERIES = [
    "How does attention work in transformers?",
    "Best practices for database indexing",
    "Python async programming patterns",
    "Memory optimization techniques",
    "Building REST APIs with FastAPI",
    "User preference learning strategies",
    "Distributed training across GPUs",
    "Vector similarity search algorithms",
    "Knowledge graph construction",
    "Caching invalidation strategies",
    "Neural network pruning methods",
    "Data pipeline orchestration tools",
    "Microservice communication patterns",
    "Authentication with JWT tokens",
    "Real-time event processing",
    "Graph neural networks overview",
    "Reinforcement learning reward shaping",
    "Embedding model fine-tuning",
    "Causal discovery algorithms",
    "Bayesian hyperparameter tuning",
]


def _random_content(topic_idx: int) -> str:
    """Generate semi-realistic memory content."""
    topic = _SAMPLE_TOPICS[topic_idx % len(_SAMPLE_TOPICS)]
    filler = "".join(random.choices(string.ascii_lowercase + " ", k=random.randint(40, 120)))
    templates = [
        f"Learned about {topic}: {filler.strip()}.",
        f"The user asked about {topic}. Key insight: {filler.strip()}.",
        f"I prefer using {topic} because {filler.strip()}.",
        f"Task completed: implemented {topic} feature. Details: {filler.strip()}.",
        f"Note to self about {topic} — {filler.strip()}.",
    ]
    return random.choice(templates)


def _random_query() -> str:
    return random.choice(_SEARCH_QUERIES)


def _fake_embedder(dim: int = 128):
    """Return a deterministic fake embedder for benchmarking without sentence-transformers."""
    import hashlib
    import numpy as np

    def embed(text: str):
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    return embed


def _percentile(data, p):
    """Calculate percentile from sorted data using statistics-compatible approach."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}ms"


def _fmt_rate(count: int, seconds: float) -> str:
    if seconds == 0:
        return "inf"
    return f"{count / seconds:.1f}/s"


# ---------------------------------------------------------------------------
# Benchmark 1: Write gate throughput
# ---------------------------------------------------------------------------

def bench_write_gate(n: int = 1000):
    """Score n candidate memories through the write gate."""
    if not HAS_WRITE_GATE:
        print("[SKIP] bench_write_gate: aura.memory.write_gate not importable")
        return None

    gate = MemoryWriteGate()
    # Force gate enabled regardless of Config
    gate._enabled = True

    decisions = Counter()
    timings = []

    for i in range(n):
        content = _random_content(i)
        candidate = MemoryCandidate(
            content=content,
            source=random.choice(["conversation", "task_execution", "explicit_clip", "learning"]),
            user_id="bench_user",
            importance=random.uniform(0.2, 0.9),
            emotional_salience=random.uniform(0.0, 0.6),
            tags=random.sample(["preference", "fact", "goal", "decision", "note"], k=random.randint(0, 2)),
            explicit_save=(random.random() < 0.1),
            confidence=random.uniform(0.5, 1.0),
        )

        # Simulate nearby memories for ~30% of candidates
        nearby = None
        if random.random() < 0.3:
            nearby = [
                {
                    "content": _random_content(i + 500),
                    "source_id": f"fake_{i}",
                    "source": "amem",
                    "score": random.uniform(0.3, 0.95),
                }
            ]

        t0 = time.perf_counter()
        decision = gate.evaluate(candidate, nearby=nearby)
        t1 = time.perf_counter()

        timings.append(t1 - t0)
        decisions[decision.kind.value] += 1

    total = sum(timings)
    avg = statistics.mean(timings)
    return {
        "name": "Write Gate Throughput",
        "count": n,
        "total_s": total,
        "avg_ms": avg * 1000,
        "rate": n / total if total > 0 else float("inf"),
        "p50_ms": _percentile(timings, 50) * 1000,
        "p95_ms": _percentile(timings, 95) * 1000,
        "p99_ms": _percentile(timings, 99) * 1000,
        "decisions": dict(decisions),
    }


# ---------------------------------------------------------------------------
# Benchmark 4: Episodic memory write + search
# ---------------------------------------------------------------------------

def bench_episodic(n_inserts: int = 500, n_queries: int = 50):
    """Insert episodes and search. Uses fake embedder to avoid model load time."""
    if not HAS_QDRANT:
        print("[SKIP] bench_episodic: qdrant-client not installed")
        return None

    try:
        from aura_episodic_memory.episode import (
            Episode, EpisodeType, EpisodeQuery, TemporalContext, EmotionalValence,
        )
        from aura_episodic_memory.memory_store import EpisodicMemoryStore
    except ImportError:
        print("[SKIP] bench_episodic: aura_episodic_memory not importable")
        return None

    tmpdir = tempfile.mkdtemp(prefix="bench_episodic_")
    fake_embed = _fake_embedder(dim=128)

    try:
        store = EpisodicMemoryStore(
            db_path=tmpdir,
            custom_embedder=fake_embed,
        )

        episode_types = list(EpisodeType)
        valences = list(EmotionalValence)

        # --- Write benchmark ---
        write_timings = []
        base_time = datetime.now() - timedelta(days=30)

        for i in range(n_inserts):
            ep = Episode(
                content=_random_content(i),
                episode_type=random.choice(episode_types),
                temporal_context=TemporalContext(
                    timestamp=base_time + timedelta(hours=i),
                ),
                title=f"Episode {i}: {_SAMPLE_TOPICS[i % len(_SAMPLE_TOPICS)]}",
                importance=random.uniform(0.3, 0.9),
                emotional_valence=random.choice(valences),
                entities_involved=[f"entity_{i % 5}"],
                tools_used=[f"tool_{i % 3}"],
            )

            t0 = time.perf_counter()
            store.store_episode(ep)
            t1 = time.perf_counter()
            write_timings.append(t1 - t0)

        write_total = sum(write_timings)
        write_avg = statistics.mean(write_timings)

        # --- Search benchmark ---
        search_timings = []
        for _ in range(n_queries):
            query = EpisodeQuery(
                query_text=_random_query(),
                limit=5,
                min_score=0.0,
            )
            t0 = time.perf_counter()
            store.search(query)
            t1 = time.perf_counter()
            search_timings.append(t1 - t0)

        search_total = sum(search_timings)
        search_avg = statistics.mean(search_timings)

        store.close()

        return {
            "name": "Episodic Memory Write + Search",
            "write_count": n_inserts,
            "write_total_s": write_total,
            "write_avg_ms": write_avg * 1000,
            "write_rate": n_inserts / write_total if write_total > 0 else float("inf"),
            "write_p50_ms": _percentile(write_timings, 50) * 1000,
            "write_p95_ms": _percentile(write_timings, 95) * 1000,
            "search_count": n_queries,
            "search_total_s": search_total,
            "search_avg_ms": search_avg * 1000,
            "search_p50_ms": _percentile(search_timings, 50) * 1000,
            "search_p95_ms": _percentile(search_timings, 95) * 1000,
            "search_p99_ms": _percentile(search_timings, 99) * 1000,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# pytest-compatible test functions
# ---------------------------------------------------------------------------

def test_bench_write_gate():
    result = bench_write_gate(n=1000)
    if result is None:
        import pytest
        pytest.skip("write_gate not importable")
    assert result["count"] == 1000
    assert result["avg_ms"] < 100, f"Write gate too slow: {result['avg_ms']:.1f}ms"
    print(f"\n  {result['name']}: {result['count']} evals in {result['total_s']:.2f}s "
          f"({_fmt_rate(result['count'], result['total_s'])})")
    print(f"  Decisions: {result['decisions']}")


def test_bench_episodic():
    result = bench_episodic(n_inserts=500, n_queries=50)
    if result is None:
        import pytest
        pytest.skip("qdrant-client or aura_episodic_memory not available")
    assert result["write_count"] == 500
    assert result["search_count"] == 50
    print(f"\n  {result['name']}:")
    print(f"    Write: {result['write_count']} eps in {result['write_total_s']:.2f}s "
          f"({_fmt_rate(result['write_count'], result['write_total_s'])})")
    print(f"    Search: avg={_fmt_ms(result['search_avg_ms']/1000)} "
          f"p50={_fmt_ms(result['search_p50_ms']/1000)} "
          f"p95={_fmt_ms(result['search_p95_ms']/1000)}")


# ---------------------------------------------------------------------------
# Standalone runner with summary table
# ---------------------------------------------------------------------------

def _print_separator(char="=", width=72):
    print(char * width)


def _print_summary(results):
    """Print a formatted summary table of all benchmark results."""
    _print_separator()
    print("BENCHMARK SUMMARY")
    _print_separator()

    for r in results:
        if r is None:
            continue
        print(f"\n  {r['name']}")
        print(f"  {'-' * len(r['name'])}")

        if "count" in r:
            print(f"    Items:        {r['count']}")
        if "corpus_size" in r:
            print(f"    Corpus:       {r['corpus_size']} memories")
            print(f"    Queries:      {r['query_count']}")
        if "write_count" in r:
            print(f"    Writes:       {r['write_count']}")
            print(f"    Searches:     {r['search_count']}")

        if "total_s" in r:
            print(f"    Total time:   {r['total_s']:.3f}s")
        if "rate" in r:
            print(f"    Throughput:   {r['rate']:.1f}/s")
        if "avg_ms" in r:
            print(f"    Avg latency:  {r['avg_ms']:.3f}ms")
        if "p50_ms" in r:
            print(f"    p50 latency:  {r['p50_ms']:.3f}ms")
        if "p95_ms" in r:
            print(f"    p95 latency:  {r['p95_ms']:.3f}ms")
        if "p99_ms" in r:
            print(f"    p99 latency:  {r['p99_ms']:.3f}ms")

        # Episodic write/search split
        if "write_total_s" in r:
            print(f"    Write total:  {r['write_total_s']:.3f}s  ({r['write_rate']:.1f}/s)")
            print(f"    Write avg:    {r['write_avg_ms']:.3f}ms")
            print(f"    Write p50:    {r['write_p50_ms']:.3f}ms")
            print(f"    Write p95:    {r['write_p95_ms']:.3f}ms")
            print(f"    Search total: {r['search_total_s']:.3f}s")
            print(f"    Search avg:   {r['search_avg_ms']:.3f}ms")
            print(f"    Search p50:   {r['search_p50_ms']:.3f}ms")
            print(f"    Search p95:   {r['search_p95_ms']:.3f}ms")
            print(f"    Search p99:   {r['search_p99_ms']:.3f}ms")

        if "decisions" in r:
            print(f"    Decisions:    {r['decisions']}")

    _print_separator()
    skipped = sum(1 for r in results if r is None)
    ran = len(results) - skipped
    print(f"  Ran {ran}/{len(results)} benchmarks ({skipped} skipped)")
    _print_separator()


if __name__ == "__main__":
    random.seed(42)
    print()
    _print_separator()
    print("Aura Memory Systems Benchmark Suite")
    _print_separator()
    print(f"  qdrant-client:        {'YES' if HAS_QDRANT else 'NO (skip Episodic)'}")
    print(f"  sentence-transformers: {'YES' if HAS_ST else 'NO'}")
    print(f"  write_gate:           {'YES' if HAS_WRITE_GATE else 'NO (skip)'}")
    _print_separator()

    results = []

    print("\n[1/2] Write Gate Throughput (1000 candidates)...")
    results.append(bench_write_gate(n=1000))

    print("\n[2/2] Episodic Memory Write + Search (500 episodes, 50 queries)...")
    results.append(bench_episodic(n_inserts=500, n_queries=50))

    print()
    _print_summary(results)
