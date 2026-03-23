#!/usr/bin/env python3
"""
AURA Memory Systems - Functional Test Suite
Reusable test script to verify all memory systems are operational.

Usage: python test_memory_systems.py
"""

import os
import sys
import traceback

# Suppress progress bars
os.environ['TQDM_DISABLE'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add Aura to path
sys.path.insert(0, os.path.dirname(__file__))


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
    results.append(('Knowledge Graph', test_knowledge_graph()))
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
