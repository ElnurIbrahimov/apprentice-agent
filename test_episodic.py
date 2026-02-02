"""Quick test of Episodic Memory integration."""
from apprentice_agent.agent import ApprenticeAgent

print("Initializing agent (fast_init=True)...")
agent = ApprenticeAgent(fast_init=True)

print("\n=== Episodic Memory Stats ===")
stats = agent.get_episodic_memory_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n=== Recording a test episode ===")
result = agent.episodic_record(
    content="Testing the new episodic memory system - it seems to be working great!",
    episode_type="milestone",
    importance=0.8,
    entities=["AURA", "Episodic Memory", "Test"]
)
print(f"  Result: {result}")

print("\n=== Recalling memories about 'episodic' ===")
memories = agent.episodic_recall("episodic memory", limit=3)
for i, mem in enumerate(memories, 1):
    print(f"  {i}. [{mem['type']}] {mem['content'][:60]}... (score: {mem['score']:.2f})")

print("\n=== Time travel to 'today' ===")
travel = agent.episodic_time_travel("today")
print(f"  Found {travel.get('episode_count', 0)} episodes")
if travel.get('narrative'):
    print(f"  Narrative preview: {travel['narrative'][:200]}...")

print("\n=== Memory Health Report ===")
health = agent.episodic_get_health()
print(f"  Status: {health.get('status', 'unknown')}")
if health.get('recommendations'):
    for rec in health['recommendations']:
        print(f"  - {rec}")

print("\nDone! Episodic Memory is working.")
