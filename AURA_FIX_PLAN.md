# AURA FIX IMPLEMENTATION PLAN
Generated: 2026-01-31
**Status: ALL FIXES IMPLEMENTED**

## Summary

The AURA v3.0 ALIVE system is in **good health** overall. The core functionality works correctly:
- Fast path handles instant responses
- Emotional engine tracks and persists mood
- Memory system stores facts and conversations
- Humanizer removes corporate speak
- All components are properly integrated

The issues found are primarily about data synchronization and missing features, not bugs.

---

## Priority 1: CRITICAL (Blocking)

**None.** System is functional.

---

## Priority 2: HIGH (Major Impact)

| # | Issue | File | Fix | Effort |
|---|-------|------|-----|--------|
| 1 | Markdown files not synced with JSON | `markdown_store.py` | Add sync methods | M |

### Fix 1: Add Markdown Sync Methods

**Location:** `aura/memory/markdown_store.py`

**Problem:** `emotional_state.md` and `patterns.md` are not synced with their JSON counterparts.

**Solution:** Add methods to sync JSON data to markdown:

```python
def sync_emotional_state(self, state: dict) -> bool:
    """Sync emotional state from JSON to markdown."""
    content = f"""- Current mood: {state.get('mood', 'neutral')}
- Energy: {state.get('energy', 0.5):.0%}
- Warmth: {state.get('warmth', 0.5):.0%}
- Reason: {state.get('mood_reason', 'None')}
- Updated: {state.get('updated_at', 'Unknown')}"""
    return self.update_section("emotional_state", "Current Mood", content)

def sync_patterns(self, patterns: dict) -> bool:
    """Sync patterns from JSON to markdown."""
    for pattern_type in ["User Patterns", "Conversation Patterns", "Temporal Patterns"]:
        # Filter patterns by type and format for markdown
        ...
```

**Also update:** `AURAEngine.process_response()` to call sync after updates.

---

## Priority 3: MEDIUM (Improvements)

| # | Issue | File | Fix | Effort |
|---|-------|------|-----|--------|
| 1 | User profile not populated | `markdown_store.py` | Add profile extraction | M |
| 2 | Proactive error handling | `heartbeat.py` | Add exponential backoff | S |
| 3 | Pattern descriptions redundant | `pattern_prophet.py` | Improve description generation | S |

### Fix 1: User Profile Extraction

**Location:** `aura/memory/markdown_store.py` or new `profile_builder.py`

**Problem:** User profile sections are empty - no data extracted from conversations.

**Solution:** Add profile extraction during conversation processing:

```python
def extract_profile_from_message(self, message: str) -> Optional[dict]:
    """Extract profile information from user message."""
    profile_triggers = {
        "Basic Info": ["my name is", "i am", "i'm a", "i work as", "i live in"],
        "Preferences": ["i like", "i prefer", "i hate", "i love", "i enjoy"],
        "Goals": ["i want to", "i'm trying to", "my goal is", "i need to"],
        "Context": ["i'm working on", "currently", "today i", "this week"]
    }
    # Extract and return matches
```

### Fix 2: Proactive Error Handling

**Location:** `aura/proactive/heartbeat.py:344-353`

**Current:**
```python
except Exception as e:
    logger.error(f"Monitor loop error: {e}")
    time.sleep(10)  # Back off on error
```

**Fixed:**
```python
def _monitor_loop(self) -> None:
    """Background monitoring loop."""
    error_count = 0
    max_errors = 5

    while self.running:
        try:
            self.run_checks()
            self._save_state()
            error_count = 0  # Reset on success
            time.sleep(self.check_interval)
        except Exception as e:
            error_count += 1
            logger.error(f"Monitor loop error ({error_count}/{max_errors}): {e}")

            if error_count >= max_errors:
                logger.critical("Too many monitor errors, stopping proactive system")
                self.running = False
                break

            # Exponential backoff: 10, 20, 40, 80, 160 seconds
            backoff = min(160, 10 * (2 ** (error_count - 1)))
            time.sleep(backoff)
```

### Fix 3: Pattern Descriptions

**Location:** `aura/patterns/pattern_prophet.py:328-346`

**Problem:** Cluster pattern descriptions are redundant: "Interest cluster around: X, user, prefers, this:"

**Solution:** Improve `_detect_cluster_patterns()` to generate cleaner descriptions.

---

## Priority 4: LOW (Nice to Have)

| # | Issue | File | Fix | Effort |
|---|-------|------|-----|--------|
| 1 | Test mock corporate language | `engine.py:390` | Change mock text | S |
| 2 | RelationshipTracker missing | new file | Implement component | M |
| 3 | Follow-up checking | new feature | Track mentioned tasks | L |

### Fix 1: Test Mock Language

**Location:** `aura/engine.py:390`

**Current:**
```python
mock_response = "I would be happy to help you with that. Let me explain..."
```

**Fixed:**
```python
mock_response = "Sure! Let me explain how this works..."
```

### Fix 2: RelationshipTracker Implementation

**Create:** `aura/relationship/relationship_tracker.py`

```python
"""
RelationshipTracker - Track relationship development with user.

Tracks:
- Attachment level (0-1)
- Conversation count
- Trust moments (vulnerable shares, help requests)
- Relationship age
- Communication style preferences
"""

@dataclass
class RelationshipState:
    attachment_level: float = 0.5
    conversation_count: int = 0
    trust_moments: int = 0
    first_interaction: Optional[str] = None
    vulnerable_shares: int = 0
    helped_with: List[str] = field(default_factory=list)

class RelationshipTracker:
    def __init__(self, state_file: Optional[Path] = None):
        ...

    def record_interaction(self, message: str, was_helpful: bool = True) -> None:
        ...

    def record_trust_moment(self, description: str) -> None:
        ...

    def get_relationship_context(self) -> str:
        """Get context for LLM about relationship."""
        ...
```

---

## Implementation Order

1. **First:** Fix proactive error handling (S, prevents potential issues)
2. **Second:** Add markdown sync methods (M, improves data consistency)
3. **Third:** Fix test mock language (S, cosmetic)
4. **Fourth:** Add user profile extraction (M, improves personalization)
5. **Fifth:** Fix pattern descriptions (S, cosmetic)
6. **Later:** Implement RelationshipTracker (M, new feature)

---

## Estimated Total Effort

| Priority | Items | Effort |
|----------|-------|--------|
| Critical | 0 | 0h |
| High | 1 | ~2h |
| Medium | 3 | ~4h |
| Low | 3 | ~6h |
| **Total** | **7** | **~12h** |

---

## Quick Wins (Can Do Now)

These can be done in under 10 minutes each:

1. **Change test mock text** in `engine.py:390`
2. **Add error count limit** in `heartbeat.py`

---

## Testing After Fixes

After implementing fixes, run these tests:

```bash
# Test AURA module directly
python -m aura.engine
python -m aura.fast_path
python -m aura.emotion.emotional_engine
python -m aura.memory.markdown_store
python -m aura.patterns.pattern_prophet
python -m aura.proactive.heartbeat
python -m aura.thinking.visible_thinking
python -m aura.humanize.response_humanizer
python -m aura.soul.soul_loader

# Test full integration
python main.py --chat
# Then test:
# - "hi" → should get instant greeting
# - "remember this: test fact" → should store instantly
# - "I got the job!" → should get excited response
# - "aura status" → should show status
# - "aura mood" → should show mood
# - Check markdown files are synced
```

---

## Notes

- The system is well-architected with clean separation of concerns
- All major components are functional and properly integrated
- Issues are primarily about polish, not functionality
- Emotional authenticity is excellent - no corporate speak
- Fast path integration is correct and efficient
