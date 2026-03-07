#!/usr/bin/env python3
"""
AURA Memory Systems - Functional Test Suite
Reusable test script to verify all memory systems are operational.

Usage: python test_memory_systems.py
"""

import os
import sys
import traceback
import tempfile
import shutil
from datetime import datetime

# Suppress progress bars
os.environ['TQDM_DISABLE'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add Aura to path
sys.path.insert(0, os.path.dirname(__file__))


def test_amem():
    """Test A-MEM (Zettelkasten) memory system."""
    print('\n[1] A-MEM (Zettelkasten)')
    try:
        from aura.tools.amem import get_amem
        
        amem = get_amem()
        amem.add('test memory about Python programming', importance=0.8)
        results = amem.search('Python', k=1)
        
        assert len(results) > 0, 'Search returned no results'
        print('    PASS: add + search works')
        return True
    except Exception as e:
        print(f'    FAIL: {type(e).__name__}: {str(e)[:70]}')
        traceback.print_exc()
        return False


def test_knowledge_graph():
    """Test Knowledge Graph system."""
    print('\n[2] Knowledge Graph')
    try:
        from aura.tools.knowledge_graph import KnowledgeGraphTool
        
        kg = KnowledgeGraphTool()
        r = kg.execute('add concept TestConcept')
        assert r.get('success'), f'add_node failed: {r}'
        
        r2 = kg.execute('show TestConcept')
        assert r2 is not None, 'show returned None'
        
        print('    PASS: add_node + show works')
        return True
    except Exception as e:
        print(f'    FAIL: {type(e).__name__}: {str(e)[:70]}')
        traceback.print_exc()
        return False


def test_episodic_memory():
    """Test Episodic Memory (Qdrant-based) system."""
    print('\n[3] Episodic Memory (Qdrant-based)')
    try:
        from aura_episodic_memory.memory_store import EpisodicMemoryStore
        from aura_episodic_memory.episode import (
            Episode, EpisodeType, TemporalContext, EpisodeQuery
        )
        
        tmpdir = tempfile.mkdtemp(prefix='aura_ep_')
        try:
            store = EpisodicMemoryStore(db_path=tmpdir)
            
            # Create and store episode
            ep = Episode(
                content='test episode content',
                episode_type=EpisodeType.CONVERSATION,
                temporal_context=TemporalContext(timestamp=datetime.now())
            )
            ep_id = store.store_episode(ep)
            assert ep_id, 'store_episode returned empty id'
            
            # Retrieve episode
            retrieved = store.get_episode(ep_id)
            assert retrieved is not None, 'get_episode returned None'
            
            # Search
            query = EpisodeQuery(query_text='test')
            results = store.search(query=query)
            assert isinstance(results, list), 'search returned non-list'
            
            # Get stats
            stats = store.get_statistics()
            assert stats.get('total_episodes') == 1, f'Stats wrong: {stats}'
            
            # Cleanup
            store.close()
            print('    PASS: store_episode + get_episode + search + stats works')
            return True
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except:
                pass
    except ImportError as e:
        print(f'    FAIL: Missing dependency: {str(e)[:70]}')
        return False
    except Exception as e:
        print(f'    FAIL: {type(e).__name__}: {str(e)[:70]}')
        traceback.print_exc()
        return False


def test_context_budget():
    """Test Context Budget allocation system."""
    print('\n[4] Context Budget')
    try:
        from aura.memory.context_budget import ContextBudget
        
        b = ContextBudget()
        a = b.allocate('amem', 800)
        k = b.allocate('kg', 600)
        
        assert a == 800, f'amem allocation failed: {a}'
        assert k == 600, f'kg allocation failed: {k}'
        assert b.remaining == 1600, f'remaining calc wrong: {b.remaining}'
        
        print('    PASS: allocate + remaining budget works')
        return True
    except Exception as e:
        print(f'    FAIL: {type(e).__name__}: {str(e)[:70]}')
        traceback.print_exc()
        return False


def main():
    """Run all memory system tests."""
    print('=' * 60)
    print('AURA MEMORY SYSTEMS - FUNCTIONAL TEST')
    print('=' * 60)
    
    results = []
    results.append(('A-MEM', test_amem()))
    results.append(('Knowledge Graph', test_knowledge_graph()))
    results.append(('Episodic Memory', test_episodic_memory()))
    results.append(('Context Budget', test_context_budget()))
    
    # Summary
    print('\n' + '=' * 60)
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    print(f'SUMMARY: {passed} PASS, {failed} FAIL')
    print('=' * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
