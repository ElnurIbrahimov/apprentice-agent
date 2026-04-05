"""Built-in DynamicHand templates.

Each entry is a valid DynamicHand config dict. Placeholders like {topic} or
{product} must be filled in before passing the config to DynamicHand().

Usage
-----
from aura.hands.templates import HAND_TEMPLATES
from aura.hands.dynamic_hand import DynamicHand

config = {**HAND_TEMPLATES["news_monitor"], "name": "ai_news", "goal": "AI safety research news"}
config["search_queries"] = [q.format(topic="AI safety") for q in config["search_queries"]]
hand = DynamicHand(config)
"""

HAND_TEMPLATES: dict[str, dict] = {
    # ------------------------------------------------------------------
    # News Monitor — periodic news digest on any topic
    # ------------------------------------------------------------------
    "news_monitor": {
        "name": "news_monitor",
        "description": "Monitors recent news and developments on a topic of interest",
        "goal": "Find and summarize the latest news about {topic}",
        "search_queries": [
            "{topic} latest news",
            "{topic} recent developments",
            "{topic} breaking news today",
        ],
        "system_prompt": (
            "You are Aura's news monitoring hand. Your job is to find, filter, and "
            "summarize the most relevant and recent news on a given topic. "
            "Focus on verified sources. Exclude opinion pieces unless highly relevant. "
            "Format: 2-sentence lead summary + 5 bullet-point headlines with one-line descriptions."
        ),
        "interval_minutes": 480,
        "idle_only": True,
        "trigger_on_drive": "curiosity",
        "trigger_drive_threshold": 0.65,
        "model_preference": "fast",
        "max_tokens": 20000,
        "max_cost_usd": 0.15,
        "is_custom": True,
    },

    # ------------------------------------------------------------------
    # Tech Digest — weekly synthesis of technology developments
    # ------------------------------------------------------------------
    "tech_digest": {
        "name": "tech_digest",
        "description": "Weekly synthesis of technology developments on a chosen topic",
        "goal": "Synthesize this week's most important developments in {topic}",
        "search_queries": [
            "{topic} this week news",
            "{topic} new releases announcements",
            "{topic} research papers 2026",
        ],
        "system_prompt": (
            "You are Aura's technology digest hand. Run once a week and produce a "
            "curated digest of the most significant developments in the given domain. "
            "Distinguish hype from substance. Prioritize: new releases, research breakthroughs, "
            "industry shifts, and tools worth knowing. "
            "Format: executive summary (3 sentences) + categorized bullet points."
        ),
        "interval_minutes": 10080,
        "idle_only": True,
        "trigger_on_drive": None,
        "trigger_drive_threshold": 0.7,
        "model_preference": "reasoning",
        "max_tokens": 30000,
        "max_cost_usd": 0.25,
        "is_custom": True,
    },

    # ------------------------------------------------------------------
    # Code Reviewer — reviews recent code changes for quality issues
    # ------------------------------------------------------------------
    "code_reviewer": {
        "name": "code_reviewer",
        "description": "Reviews recent code changes for quality, security, and best practices",
        "goal": "Review the most recent code changes and flag potential issues or improvements",
        "search_queries": [
            "common code review issues Python JavaScript",
            "security vulnerabilities code patterns 2026",
        ],
        "system_prompt": (
            "You are Aura's autonomous code review hand. Your job is to review recent "
            "code changes and identify: security vulnerabilities, logic errors, performance "
            "bottlenecks, poor error handling, and style inconsistencies. "
            "Be specific — reference line-level patterns, not vague generalities. "
            "Format: severity-rated findings (Critical / High / Medium / Low) + suggested fixes."
        ),
        "interval_minutes": 360,
        "idle_only": True,
        "trigger_on_drive": "competence",
        "trigger_drive_threshold": 0.6,
        "model_preference": "reasoning",
        "max_tokens": 25000,
        "max_cost_usd": 0.20,
        "is_custom": True,
    },

    # ------------------------------------------------------------------
    # Price Watcher — tracks price changes for a product
    # ------------------------------------------------------------------
    "price_watcher": {
        "name": "price_watcher",
        "description": "Tracks and summarizes current pricing for a product or service",
        "goal": "Find the current best price and pricing trends for {product}",
        "search_queries": [
            "{product} price 2026",
            "{product} buy cheapest deal",
            "{product} price history trend",
        ],
        "system_prompt": (
            "You are Aura's price watching hand. Your job is to find current prices "
            "for a given product or service, compare across sources, and identify "
            "trends or notable deals. "
            "Always note the date of prices found. Flag if prices seem anomalous. "
            "Format: current best price + price range table + trend assessment."
        ),
        "interval_minutes": 240,
        "idle_only": True,
        "trigger_on_drive": None,
        "trigger_drive_threshold": 0.7,
        "model_preference": "fast",
        "max_tokens": 15000,
        "max_cost_usd": 0.10,
        "is_custom": True,
    },

    # ------------------------------------------------------------------
    # Habit Tracker — checks in on a personal goal or habit
    # ------------------------------------------------------------------
    "habit_tracker": {
        "name": "habit_tracker",
        "description": "Periodically checks on progress toward a personal goal or habit",
        "goal": "Assess progress and find motivation strategies for: {goal}",
        "search_queries": [
            "{goal} tips strategies success",
            "{goal} common mistakes how to improve",
            "habit formation science evidence-based",
        ],
        "system_prompt": (
            "You are Aura's habit tracking and coaching hand. Your job is to help "
            "build momentum toward a personal goal by surfacing actionable advice, "
            "evidence-based strategies, and gentle accountability prompts. "
            "Be encouraging but honest. Avoid generic platitudes. "
            "Format: one motivational insight + 3 concrete next actions + one potential obstacle to watch."
        ),
        "interval_minutes": 1440,
        "idle_only": True,
        "trigger_on_drive": "competence",
        "trigger_drive_threshold": 0.55,
        "model_preference": "fast",
        "max_tokens": 15000,
        "max_cost_usd": 0.10,
        "is_custom": True,
    },
}
