"""Interactive chat test with Skill Library integration."""
import sys
import io
import os
from pathlib import Path

# Change to the project directory
os.chdir(Path(__file__).parent)

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from apprentice_agent.agent import ApprenticeAgent

print("Initializing agent with full init...")
agent = ApprenticeAgent(fast_init=False)

print("\n" + "="*60)
print("AURA Skill Library - Chat Integration Test")
print("="*60)

# Show current memory systems status
print("\n[Memory Systems Status]")
print(f"  Skill Library: {'Available' if agent.skill_library else 'Not available'}")
print(f"  Episodic Memory: {'Available' if agent.episodic_memory else 'Not available'}")
print(f"  KG Brain: {'Available' if agent.kg_brain else 'Not available'}")

# Create some useful skills
print("\n[Creating utility skills]")

skills_to_create = [
    {
        "name": "Code Explanation",
        "description": "Explains code in simple terms, breaking down complex logic",
        "category": "coding",
        "trigger_patterns": ["explain this code", "what does this do", "how does this work"],
        "procedure": "1. Identify the language and purpose\n2. Break down into logical sections\n3. Explain each section simply\n4. Summarize the overall flow",
        "tags": ["explanation", "teaching"]
    },
    {
        "name": "Bug Finder",
        "description": "Identifies potential bugs and issues in code",
        "category": "coding",
        "trigger_patterns": ["find bugs", "debug this", "what's wrong with this code"],
        "procedure": "1. Read code carefully\n2. Check for common bugs (off-by-one, null refs)\n3. Trace execution flow\n4. List potential issues with fixes",
        "tags": ["debugging", "quality"]
    },
    {
        "name": "Summary Writer",
        "description": "Creates concise summaries of longer texts",
        "category": "writing",
        "trigger_patterns": ["summarize this", "give me a summary", "tldr"],
        "procedure": "1. Identify main points\n2. Note key details\n3. Write concise summary\n4. Keep under 3 sentences if possible",
        "tags": ["summary", "concise"]
    }
]

for skill_data in skills_to_create:
    result = agent.skill_create(**skill_data)
    if result.get("success"):
        print(f"  Created: {skill_data['name']}")
    else:
        print(f"  Failed: {skill_data['name']} - {result.get('error', 'Unknown')}")

# List all skills
print("\n[All Skills in Library]")
skills = agent.skill_list(sort_by="name")
for skill in skills:
    print(f"  - {skill['name']} ({skill['category']}) - {skill.get('success_rate', 0):.0%} success, {skill.get('total_uses', 0)} uses")

# Test skill finding for different queries
print("\n[Testing Skill Matching]")
test_queries = [
    "Can you explain this code to me?",
    "I think there's a bug in my function",
    "Give me a quick summary of this article",
    "Write a Python script for me"
]

for query in test_queries:
    applicable = agent.skill_find_applicable(query, max_skills=2)
    print(f"\n  Query: '{query[:40]}...'")
    if applicable:
        for skill, score in applicable:
            print(f"    -> {skill.name} (score: {score:.2f})")
    else:
        print("    -> No matching skills")

# Test chat with skill context
print("\n" + "="*60)
print("[Testing Chat with Skill Context]")
print("="*60)

# Get skill context that would be injected into LLM
test_input = "explain this code: def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
context = agent.skill_get_context(test_input)
print(f"\nFor input: '{test_input[:50]}...'")
print(f"Skill context would inject {len(context)} chars into LLM prompt")

if context:
    print("\n[Skill Context Preview]")
    print("-" * 40)
    print(context[:500])
    print("-" * 40)

# Actually chat with the agent
print("\n[Chat Test]")
try:
    response = agent.chat("What can you help me with today?")
    safe_response = response.encode('ascii', errors='replace').decode('ascii')
    print(f"AURA: {safe_response[:400]}...")
except Exception as e:
    print(f"Chat error: {e}")

# Final stats
print("\n" + "="*60)
print("[Final Statistics]")
print("="*60)

sl_stats = agent.get_skill_library_stats()
if sl_stats.get("available", True):
    store = sl_stats.get("store", {})
    learner = sl_stats.get("learner", {})
    executor = sl_stats.get("executor", {})

    print(f"\nSkill Library:")
    print(f"  Total skills: {store.get('total_skills', 0)}")
    print(f"  By category: {store.get('by_category', {})}")
    print(f"  Learned skills: {store.get('learned_skills', 0)}")

    print(f"\nLearner:")
    print(f"  Interactions recorded: {learner.get('interactions_recorded', 0)}")
    print(f"  Skills learned: {learner.get('skills_learned', 0)}")
    print(f"  Patterns in buffer: {learner.get('patterns_in_buffer', 0)}")

    print(f"\nExecutor:")
    print(f"  Executions: {executor.get('executions', 0)}")
    print(f"  With skill: {executor.get('with_skill', 0)}")
    print(f"  Success rate: {executor.get('success_rate', 0):.0%}")

print("\nTest complete!")
agent.shutdown()
