"""Interactive chat test with Episodic Memory."""
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from apprentice_agent.agent import ApprenticeAgent

print("Initializing agent...")
agent = ApprenticeAgent(fast_init=False)  # Full init for bridge

print("\n" + "="*60)
print("AURA with Episodic Memory - Interactive Test")
print("="*60)

# Check initial stats
print("\n[Initial Memory Stats]")
em_stats = agent.get_episodic_memory_stats()
kg_stats = agent.get_kg_brain_stats()
print(f"  Episodic: {em_stats.get('total_episodes', 0)} episodes")
print(f"  KG Brain: {kg_stats.get('total_entities', 0)} entities")

# Have a conversation
messages = [
    "What's your name and what can you do?",
    "Tell me about Python decorators briefly",
    "Remember that I prefer concise explanations"
]

for msg in messages:
    print(f"\n[User]: {msg}")
    try:
        response = agent.chat(msg)
        # Sanitize for console output
        safe_response = response.encode('ascii', errors='replace').decode('ascii')
        print(f"[AURA]: {safe_response[:300]}...")
    except Exception as e:
        print(f"[Error]: {e}")

# Check memory after conversation
print("\n" + "="*60)
print("[Memory Status After Conversation]")
print("="*60)

em_stats = agent.get_episodic_memory_stats()
print(f"\nEpisodic Memory:")
print(f"  Total episodes: {em_stats.get('total_episodes', 0)}")
print(f"  Episodes formed: {em_stats.get('episodes_formed', 0)}")

# Try recalling what we talked about
print("\n[Recalling 'Python' memories]")
memories = agent.episodic_recall("Python decorators", limit=3)
if memories:
    for m in memories:
        content = m['content'][:80].encode('ascii', errors='replace').decode('ascii')
        print(f"  - [{m['type']}] {content}...")
else:
    print("  No memories found yet (may need more conversation)")

# Time travel
print("\n[Time traveling to 'just now']")
travel = agent.episodic_time_travel("just now")
print(f"  Found {travel.get('episode_count', 0)} episodes")

print("\nTest complete!")
agent.shutdown()
