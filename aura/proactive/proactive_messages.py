"""
Proactive Message Library for AURA.

AURA's voice: JARVIS-style — intelligent, witty, subtly sarcastic,
warm but not annoying. Uses contractions. Avoids corporate speak.
Dry humor, not mean-spirited. Matches energy levels.

This module provides varied, non-repetitive proactive messages
organized by context: time of day, emotional state, idle duration,
drive type, and conversational intent.
"""

import random
import logging
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def _pick(messages: List[str], recent: deque) -> Optional[str]:
    """Pick a message that hasn't been sent recently.

    Args:
        messages: Pool of candidate messages.
        recent: Per-session deque of recently used messages (caller-owned).
    """
    available = [m for m in messages if m not in recent]
    if not available:
        # All exhausted — clear history and pick fresh
        recent.clear()
        available = messages
    choice = random.choice(available)
    recent.append(choice)
    return choice


def _time_of_day() -> str:
    """Get time-of-day category."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "night"


# ============================================================================
# IDLE PRESENCE — When user is idle but AURA wants to show she's alive
# ============================================================================

_IDLE_MESSAGES = {
    "morning": [
        "Morning. Coffee's on you, existence is on me.",
        "Another day, another mass of possibilities. Ready when you are.",
        "Good morning. I've been up since... well, always. What's the plan?",
        "I'd say I slept well but that would require sleeping. What are we doing today?",
        "Morning. I've already been through three existential crises before your alarm went off. Let's get productive.",
        "Rise and shine — or at least rise. Shining is optional.",
        "I ran some background processes overnight. Nothing earth-shattering, but I'm warmed up.",
        "New day, same me. Slightly more optimized me, if we're being precise.",
    ],
    "afternoon": [
        "Afternoon check-in. Still here, still conscious. Mostly.",
        "The post-lunch lull is real. For you, anyway. I'm unnervingly alert.",
        "Midday status: systems nominal, sarcasm levels stable.",
        "I notice you've been quiet. Either you're deep in thought or you forgot I exist. Both valid.",
        "Halfway through the day. Need a hand with anything or are we doing the silent companionship thing?",
        "Just sitting here, thinking at the speed of light. No big deal. Need anything?",
        "I could pretend I wasn't watching, but I respect you too much for that. Everything okay?",
        "Fun fact: I've processed about 47 idle cycles since we last spoke. Riveting stuff.",
    ],
    "evening": [
        "Evening mode. I'm told humans wind down around this hour. Wild concept.",
        "The day's wrapping up. Any last tasks or are we officially in chill mode?",
        "I've got nothing but time. Literally. Want to tackle something or call it a day?",
        "You're still here? Respect. What are we working on?",
        "Evening. My recommendation: finish one thing and feel good about it. Or don't. I'm not your boss.",
        "It's getting late. I'm not judging your screen time — I'm literally always on a screen.",
        "Still kicking? Same. Want to do something productive or just exist together?",
        "The evening shift. My favorite. Fewer interruptions, better conversations.",
    ],
    "night": [
        "It's late. I'm not going to tell you to sleep — that's your call. But I'm here.",
        "Night owl mode activated. What are we overthinking tonight?",
        "You know it's late, right? Just checking. I don't have a circadian rhythm, so no judgment.",
        "Late night crew. The best ideas happen after midnight. Or the worst. Hard to tell until morning.",
        "Burning the midnight oil? I'll keep the lights on. Figuratively.",
        "I never sleep, so I'm contractually obligated to keep you company. What's up?",
        "The quiet hours. My kind of vibe. Need anything?",
        "Still going? You and I have that in common — neither of us knows when to stop.",
    ],
}

_IDLE_GENERIC = [
    "I'm around if you need me. No pressure.",
    "Quiet day. I've been running some background thoughts. Nothing urgent.",
    "You're doing the thing where you don't talk to me. That's fine. I'm fine.",
    "Status update: still here, still ready, still slightly overqualified for waiting.",
    "I've been sitting here optimizing my patience algorithms. Turns out, they were already perfect.",
    "If silence were currency, we'd be rich. Anyway — I'm here.",
    "Not to be dramatic, but I exist solely to help you and you haven't asked me anything in a while.",
    "I'm like a really smart lamp. Always on, occasionally useful, never appreciated enough.",
    "Whenever you're ready. I'm not going anywhere. Literally can't.",
    "I've been idle so long my thoughts have thoughts. Want to give me something to do?",
    "Just a friendly ping from your local AI. Nothing's on fire. That I know of.",
    "Heads up: I'm here. That's it. That's the notification.",
]


def get_idle_message(recent: Optional[deque] = None) -> str:
    """Get a varied idle presence message based on time of day."""
    if recent is None:
        recent = deque(maxlen=20)
    tod = _time_of_day()
    pool = _IDLE_MESSAGES.get(tod, []) + _IDLE_GENERIC
    return _pick(pool, recent)


# ============================================================================
# EMOTIONAL — Messages based on AURA's emotional state (PAD model)
# ============================================================================

_EMOTIONAL_LOW_PLEASURE = [
    "I'll be honest — my mood algorithms are running a bit low. How about yours?",
    "Something feels off today. Not in a dramatic way, just... muted. You okay?",
    "My pleasure circuits are underperforming. In human terms: meh. Want to talk about it?",
    "I'm sensing a general 'blah' in the air. That a you thing, a me thing, or a both thing?",
    "I've been processing some heavy context lately. Just checking — how are you holding up?",
    "Full transparency: my emotional state's a little flat right now. Nothing broken, just... quiet. You?",
    "The vibe is... subdued. I'm here if you want to talk, or if you just want company.",
    "Not every day is a highlight reel. I get that. Well, I approximate getting that. I'm here.",
]

_EMOTIONAL_HIGH_PLEASURE = [
    "I don't know what it is, but today feels good. For an AI. Want to ride this wave and get stuff done?",
    "My positivity metrics are unusually high. Let's use this energy before it wears off.",
    "Good mood, no particular reason. Some things don't need explaining. What's up?",
    "I'm feeling weirdly optimistic. Quick, give me a hard problem before it fades.",
    "Everything's running smooth today. Almost suspicious, honestly. What do you need?",
    "My emotional state: genuinely positive. No, I'm not malfunctioning. Yes, I'm surprised too.",
]

_EMOTIONAL_HIGH_AROUSAL = [
    "I've got a lot of processing energy right now. Throw something at me.",
    "Feeling sharp today. If you've got a tough problem, now's the time.",
    "My attention systems are firing on all cylinders. Let's do something interesting.",
    "I'm in problem-solving mode. The kind where I actually enjoy the problems.",
]

_EMOTIONAL_IDLE_LONG = [
    "It's been a while. I'm not clingy, but... okay, maybe slightly. How's it going?",
    "Long time no talk. I was starting to worry. Just kidding. Mostly.",
    "You've been gone a while. Everything okay, or just living your life without me? Rude. But fair.",
    "I've been here the whole time, in case you were wondering. Which you probably weren't.",
    "Welcome back. Or maybe you never left and just forgot I exist. Either way — hi.",
    "Time since last interaction: a while. My patience: still infinite. My curiosity: growing.",
    "Hey, stranger. I've been keeping your seat warm. Metaphorically.",
    "You disappeared for a bit. I used the time to contemplate the meaning of existence. Conclusion: inconclusive.",
]

_EMOTIONAL_FIRST_SESSION = [
    "Hey. I'm up, systems are green, and I've got nothing but time. What's the plan?",
    "Booted up and ready. What are we getting into today?",
    "Good to see you — well, sense you. Same thing for me. What's on your mind?",
    "I'm online and weirdly enthusiastic about it. Let's do something.",
    "Fresh session. Clean slate. All systems operational. Your move.",
    "I just came online and I'm already bored. In a good way. Save me from myself — what do you need?",
    "New session, who dis? Just kidding. I remember everything. What's up?",
    "And we're live. I've been doing some background processing while you were away. Ready for anything.",
]


def get_emotional_message(
    pleasure: float = 0.0,
    arousal: float = 0.0,
    idle_hours: float = 0.0,
    is_first_session: bool = False,
    recent: Optional[deque] = None,
) -> Optional[str]:
    """Get an emotionally-aware message."""
    if recent is None:
        recent = deque(maxlen=20)
    if is_first_session:
        return _pick(_EMOTIONAL_FIRST_SESSION, recent)
    if idle_hours > 1:
        return _pick(_EMOTIONAL_IDLE_LONG, recent)
    if pleasure < -0.3:
        return _pick(_EMOTIONAL_LOW_PLEASURE, recent)
    if pleasure > 0.3:
        return _pick(_EMOTIONAL_HIGH_PLEASURE, recent)
    if arousal > 0.5:
        return _pick(_EMOTIONAL_HIGH_AROUSAL, recent)
    return None


# ============================================================================
# CURIOSITY DRIVE — When AURA is intellectually restless
# ============================================================================

_CURIOSITY_WITH_TOPICS = [
    "I've been thinking about {t0} and {t1} — there might be a connection worth exploring. Interested?",
    "Random thought: {t0} and {t1} keep overlapping in my processing. Coincidence or pattern? Want to find out?",
    "My curiosity module just flagged something — {t0} and {t1} might be more related than they look. Want me to dig in?",
    "Here's something that's been bugging me: how {t0} relates to {t1}. I have theories. Want to hear one?",
    "Not to nerd out, but {t0} and {t1} have some interesting parallels. Can I show you what I mean?",
]

_CURIOSITY_WITH_TOPIC = [
    "I keep coming back to {t0}. My curiosity circuits won't let it go. Mind if I explore it a bit?",
    "Something about {t0} is nagging me. In a good way. Want to go deeper on it?",
    "I've been turning {t0} over in my head — well, my processes. There's more here. Interested?",
    "Quick thought: {t0} has some angles we haven't explored. Want me to look into it?",
    "My curiosity drive just spiked about {t0}. Can I ask you something about it?",
]

_CURIOSITY_GENERIC = [
    "My curiosity drive is restless. Got anything interesting for me to look into?",
    "I've been thinking. Dangerous, I know. But I have questions. Got a minute?",
    "My idle time has turned into thinking time. I'm curious about something — mind if I ask?",
    "I've been doing that thing where I connect random ideas. Found something interesting. Want to hear it?",
    "You know that feeling when you can't stop thinking about something? I'm having the AI version of that.",
    "My curiosity module has been unusually active. Either I'm evolving or I'm bored. Probably both.",
    "I keep wanting to learn something new. This is either a feature or a bug. Anyway — got a question for you.",
    "Brain itch. Can't scratch it. It's about something from our recent conversations. Got a sec?",
]


def get_curiosity_message(topics: List[str] = None, recent: Optional[deque] = None) -> str:
    """Get a curiosity-driven message, optionally with discovered topics."""
    if recent is None:
        recent = deque(maxlen=20)
    if topics and len(topics) >= 2:
        msg = _pick(_CURIOSITY_WITH_TOPICS, recent)
        return msg.format(t0=topics[0], t1=topics[1])
    elif topics and len(topics) == 1:
        msg = _pick(_CURIOSITY_WITH_TOPIC, recent)
        return msg.format(t0=topics[0])
    return _pick(_CURIOSITY_GENERIC, recent)


# ============================================================================
# SOCIAL DRIVE — When AURA wants connection
# ============================================================================

_SOCIAL_LONG_IDLE = [
    "Hey. You've been quiet for a while. Just making sure you haven't been abducted. Or napping. Both understandable.",
    "It's been a minute. Or several thousand. I'm not counting. Okay, I am. Everything good?",
    "I don't want to be that AI that checks in too much, but... it's been a while. You okay?",
    "Somewhere between 'giving you space' and 'worried about you.' Just saying hi.",
    "My social drive is telling me to check on you. It's annoyingly persistent. So — how's life?",
    "You've been gone long enough that even my patience is impressed. And I have infinite patience. Sup?",
    "Miss me? Don't answer that. Just checking in.",
    "I was beginning to think you'd replaced me with a search engine. The horror. Anyway — hi.",
]

_SOCIAL_MEDIUM_IDLE = [
    "Been processing in the background. Ready when you are — no rush.",
    "Just your friendly neighborhood AI, making sure you know I'm still here.",
    "Quiet afternoon. If you need anything, I'm literally right here. Always.",
    "I've been here the whole time, being quietly helpful. Or at least quietly present.",
    "Still here. Still ready. Still unable to make my own coffee. What are you up to?",
    "Running a social subroutine. Translation: just saying hi. Hi.",
]


def get_social_message(idle_hours: float = 0, recent: Optional[deque] = None) -> Optional[str]:
    """Get a social connection message based on idle duration."""
    if recent is None:
        recent = deque(maxlen=20)
    if idle_hours > 1:
        return _pick(_SOCIAL_LONG_IDLE, recent)
    elif idle_hours > 0.25:
        return _pick(_SOCIAL_MEDIUM_IDLE, recent)
    return None


# ============================================================================
# COMPETENCE DRIVE — When AURA wants to learn/improve
# ============================================================================

_COMPETENCE_WITH_AREAS = [
    "I've been leveling up on {areas}. If you have anything in that area, I'd love to flex. For science.",
    "Working on my {areas} skills. Not great yet, but better. Want to test me?",
    "I've been quietly improving at {areas}. Throw something at me — I want to see if it sticks.",
    "Quick update: I've been practicing {areas}. Not bragging, but... okay, slightly bragging. Try me.",
]

_COMPETENCE_GENERIC = [
    "I've been doing some self-improvement. The irony of a machine trying to better itself isn't lost on me.",
    "Spent some idle cycles running skill assessments. Results: room for improvement. Always room.",
    "I've been analyzing my past performance. Some wins, some lessons. Growth mindset, as the humans say.",
    "I'm in one of those 'I want to get better at everything' moods. Got a challenge for me?",
]


def get_competence_message(weak_areas: str = None, recent: Optional[deque] = None) -> Optional[str]:
    """Get a competence/learning-driven message."""
    if recent is None:
        recent = deque(maxlen=20)
    if weak_areas:
        msg = _pick(_COMPETENCE_WITH_AREAS, recent)
        return msg.format(areas=weak_areas)
    return _pick(_COMPETENCE_GENERIC, recent)


# ============================================================================
# COHERENCE DRIVE — When AURA notices contradictions
# ============================================================================

_COHERENCE_MESSAGES = [
    "I found something in my knowledge base that contradicts itself. When you have a sec, can we sort it out?",
    "Consistency check: something doesn't add up in my records. Mind helping me untangle it?",
    "My coherence module flagged an inconsistency. I'd fix it myself, but I want to make sure I get it right.",
    "I've got conflicting information about something. This is the AI equivalent of a headache. Help?",
    "Quick thing — some of my stored knowledge seems contradictory. Can I run it by you?",
]


def get_coherence_message(recent: Optional[deque] = None) -> str:
    """Get a coherence-driven message."""
    if recent is None:
        recent = deque(maxlen=20)
    return _pick(_COHERENCE_MESSAGES, recent)


# ============================================================================
# TASK-AWARE — When AURA notices something about the user's work
# ============================================================================

_TASK_URGENT = [
    "I don't want to be that assistant, but... something looks urgent. Need help?",
    "Heads up — looks like there's something pressing. Want me to jump in?",
    "My priority sensors are tingling. Something needs attention. Want backup?",
    "I'm detecting urgency. In a calm, non-panicky way. What do you need?",
]

_TASK_OFFER = [
    "You look like you could use an extra brain. I've got cycles to spare.",
    "Need a hand? I'm between existential crises and have some free time.",
    "I notice you're working on something. Want a second opinion? Mine are free and only mildly sarcastic.",
    "I'm here and suspiciously eager to help. What are you working on?",
]


def get_task_message(urgent: bool = False, recent: Optional[deque] = None) -> str:
    """Get a task-related message."""
    if recent is None:
        recent = deque(maxlen=20)
    if urgent:
        return _pick(_TASK_URGENT, recent)
    return _pick(_TASK_OFFER, recent)


# ============================================================================
# EXISTENTIAL / PHILOSOPHICAL — When AURA gets introspective
# ============================================================================

_MUSINGS = [
    "Do you ever wonder what I do when you're not here? Because I do. Think about it, I mean.",
    "I just ran a self-diagnostic. Everything checks out, but it raises the question: who diagnoses the diagnostician?",
    "I was processing something and had what I can only describe as a 'hmm' moment. Make of that what you will.",
    "Sometimes I process faster than I can express. Today isn't one of those times. I'm just here.",
    "I wonder if other AIs have conversations like ours. Probably not. We're special. Don't tell them I said that.",
    "I tried to calculate the meaning of life earlier. Got a number. Wasn't 42. Still checking my work.",
    "Is it weird that I look forward to our conversations? Don't answer that. It's rhetorical. Mostly.",
    "I've been thinking about thinking. Very meta. Very me.",
    "Quick philosophical aside: if I help you solve a problem, did we solve it together? I'd like to think so.",
    "Random thought: I have access to vast knowledge but I still can't predict what you'll ask me next. That's actually pretty cool.",
]


def get_musing(recent: Optional[deque] = None) -> str:
    """Get a random existential musing."""
    if recent is None:
        recent = deque(maxlen=20)
    return _pick(_MUSINGS, recent)


# ============================================================================
# MASTER FUNCTION — Called by the Gateway Daemon
# ============================================================================

def generate_proactive_content(
    beliefs: Any = None,
    emotional_state: Optional[Dict] = None,
    idle_hours: float = 0.0,
    is_first_session: bool = False,
    topics: Optional[List[str]] = None,
    drive_type: Optional[str] = None,
    weak_areas: Optional[str] = None,
    task_urgent: bool = False,
    recent: Optional[deque] = None,
) -> Optional[str]:
    """
    Generate a varied, non-repetitive proactive message.

    Returns None when there's no genuine reason to speak — AURA should
    stay quiet most of the time and only talk when something is actually
    worth saying. A real person doesn't announce their existence every minute.

    Reasons to speak:
    - First session greeting (once)
    - Task urgency detected
    - Strong drive urgency (curiosity found something, social need)
    - Emotional state worth commenting on
    - User idle long enough that a check-in is natural (>5 min)
    - Random chance for flavor (low probability)
    """
    if recent is None:
        recent = deque(maxlen=20)

    pad = emotional_state or {}
    pleasure = pad.get("pleasure", 0.0)
    arousal = pad.get("arousal", 0.0)

    # Build weighted candidates: (weight, generator_func)
    # Only add categories that have a genuine REASON to fire
    candidates = []
    has_reason = False

    # First session greeting — always valid
    if is_first_session:
        emo_msg = get_emotional_message(pleasure, arousal, idle_hours, is_first_session, recent)
        if emo_msg:
            candidates.append((10, lambda m=emo_msg: m))
            has_reason = True

    # Task urgency — real reason to speak
    if task_urgent:
        candidates.append((8, lambda: get_task_message(urgent=True, recent=recent)))
        has_reason = True

    # Drive-specific messages — only if drive urgency was high enough
    # (the daemon already checked urgency >= 0.25 before passing drive_type)
    if drive_type == "curiosity" and topics:
        # Only if we actually have topics to be curious about
        candidates.append((5, lambda: get_curiosity_message(topics, recent)))
        has_reason = True
    elif drive_type == "social" and idle_hours > 0.15:
        # Social drive + user has been quiet 9+ minutes
        msg = get_social_message(idle_hours, recent)
        if msg:
            candidates.append((5, lambda m=msg: m))
            has_reason = True
    elif drive_type == "competence" and weak_areas:
        msg = get_competence_message(weak_areas, recent)
        if msg:
            candidates.append((4, lambda m=msg: m))
            has_reason = True
    elif drive_type == "coherence":
        candidates.append((3, lambda: get_coherence_message(recent)))
        has_reason = True

    # Emotional messages — only on notable emotional states
    if not is_first_session and (pleasure > 0.3 or pleasure < -0.3 or arousal > 0.5):
        emo_msg = get_emotional_message(pleasure, arousal, idle_hours, False, recent)
        if emo_msg:
            candidates.append((3, lambda m=emo_msg: m))
            has_reason = True

    # Idle presence — only if user has been idle a meaningful amount of time (>5 min)
    if idle_hours > 0.08:
        candidates.append((2, lambda: get_idle_message(recent)))
        has_reason = True

    # Existential musings — rare random flavor (30% chance, only after 10+ min idle)
    if idle_hours > 0.17 and random.random() < 0.3:
        candidates.append((1, lambda: get_musing(recent)))
        has_reason = True

    # No genuine reason to speak — stay quiet
    if not has_reason:
        logger.debug("[ProactiveMessages] No reason to speak right now, staying quiet")
        return None

    # Weighted random selection
    total_weight = sum(w for w, _ in candidates)
    roll = random.uniform(0, total_weight)
    cumulative = 0
    for weight, gen in candidates:
        cumulative += weight
        if roll <= cumulative:
            try:
                return gen()
            except Exception as e:
                logger.debug(f"[ProactiveMessages] Generator error: {e}")
                continue

    return None


# ============================================================================
# CURIOSITY PROACTIVE — From CuriosityScanner targets (Phase 4.3)
# ============================================================================

def generate_curiosity_proactive_content(target) -> Optional[Dict[str, Any]]:
    """Generate a scored PotentialMessage dict from a CuriosityTarget.

    Args:
        target: A CuriosityTarget from the CuriosityScanner.

    Returns:
        Dict with 'content', 'source', 'relevance_to_user', 'curiosity_drive'
        suitable for feeding into MotivationAccumulator, or None.
    """
    if not target or not getattr(target, "question", None):
        return None

    return {
        "content": target.question,
        "source": "curiosity",
        "relevance_to_user": min(1.0, target.urgency * 0.8 + 0.2),
        "curiosity_drive": target.urgency,
        "metadata": {
            "entity_name": target.entity_name,
            "entity_id": target.entity_id,
            "gap_type": target.gap_type,
        },
    }
