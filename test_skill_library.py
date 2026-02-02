"""Interactive test for AURA Skill Library."""
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

print("Initializing agent...")
agent = ApprenticeAgent(fast_init=False)  # Full init for skill library

print("\n" + "="*60)
print("AURA with Skill Library - Interactive Test")
print("="*60)

# Check initial stats
print("\n[Initial Skill Library Stats]")
sl_stats = agent.get_skill_library_stats()
if sl_stats.get("available", True):
    store_stats = sl_stats.get("store", {})
    print(f"  Total skills: {store_stats.get('total_skills', 0)}")
    print(f"  By category: {store_stats.get('by_category', {})}")
else:
    print(f"  Not available: {sl_stats.get('reason', 'Unknown')}")

# Create a test skill
print("\n[Creating a test skill]")
result = agent.skill_create(
    name="Python Code Reviewer",
    description="Reviews Python code for best practices, style, and potential bugs",
    category="coding",
    trigger_patterns=[
        "review this python code",
        "check my python",
        "code review"
    ],
    procedure="""1. Read the code carefully
2. Check for PEP 8 style compliance
3. Look for potential bugs or edge cases
4. Suggest improvements for readability
5. Check for proper error handling
6. Return a summary of findings""",
    tags=["python", "review", "quality"]
)
print(f"  Created: {result}")

# List skills
print("\n[Listing all skills]")
skills = agent.skill_list()
for skill in skills[:5]:
    print(f"  - {skill['name']} ({skill['category']}) - {skill.get('total_uses', 0)} uses")

# Search for skills
print("\n[Searching for 'python code' skills]")
search_results = agent.skill_search("python code", limit=3)
for skill_id, score in search_results:
    print(f"  - {skill_id}: {score:.2f}")

# Find applicable skills
print("\n[Finding applicable skills for 'review my Python function']")
applicable = agent.skill_find_applicable("review my Python function")
for skill, score in applicable:
    print(f"  - {skill.name}: {score:.2f}")

# Get skill context for LLM
print("\n[Getting skill context for LLM injection]")
context = agent.skill_get_context("I need to review some Python code")
if context:
    print(f"  Context length: {len(context)} chars")
    print(f"  Preview: {context[:200]}...")
else:
    print("  No context available")

# Record skill usage
if result.get("success") and result.get("skill_id"):
    print("\n[Recording skill usage]")
    record_result = agent.skill_record_use(
        skill_id=result["skill_id"],
        input_context="Review this function: def add(a, b): return a + b",
        output="The function is simple and correct. Consider adding type hints.",
        success=True,
        feedback="Helpful review"
    )
    print(f"  Recorded: {record_result}")

# Final stats
print("\n" + "="*60)
print("[Final Skill Library Stats]")
print("="*60)
sl_stats = agent.get_skill_library_stats()
if sl_stats.get("available", True):
    store_stats = sl_stats.get("store", {})
    print(f"  Total skills: {store_stats.get('total_skills', 0)}")
    learner_stats = sl_stats.get("learner", {})
    print(f"  Interactions recorded: {learner_stats.get('interactions_recorded', 0)}")
    print(f"  Skills learned: {learner_stats.get('skills_learned', 0)}")

print("\nTest complete!")
agent.shutdown()
