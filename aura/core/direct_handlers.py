"""Direct command handlers mixin — monologue, neurodream, git, search, crypto, code.

Extracted from agent.py (2026-03-23) to reduce class size.
These bypass the LLM agent loop for deterministic or direct tool execution.
All methods assume self has: tools, brain, monologue, neurodream, identity.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Safe import for NeuroDream sleep phase
try:
    from aura.tools import SleepPhase
except ImportError:
    SleepPhase = None


class DirectHandlersMixin:
    """Mixin providing direct command handlers that bypass the agent loop."""

    # ------------------------------------------------------------------
    # Inner monologue
    # ------------------------------------------------------------------

    def _handle_monologue_command(self, message: str) -> Optional[str]:
        """Handle inner monologue commands directly, bypassing the LLM.

        Args:
            message: The user's message

        Returns:
            Formatted result string if monologue command, None otherwise
        """
        msg_lower = message.lower()

        monologue_keywords = [
            'show thoughts', 'your thoughts', 'recent thoughts',
            'think aloud', 'verbosity', 'why did you do that',
            'explain your reasoning', 'reasoning chain', 'export thoughts'
        ]

        if not any(kw in msg_lower for kw in monologue_keywords):
            return None

        if "inner_monologue" not in self.tools:
            return "Inner monologue not available."

        monologue = self.tools["inner_monologue"]
        result = monologue.execute(message)

        if result.get("success"):
            if "thoughts" in result:
                return result["thoughts"]
            if "reasoning_chain" in result:
                return result["reasoning_chain"]
            if "message" in result:
                return result["message"]
            return str(result)

        return result.get("error", "Unknown error")

    # ------------------------------------------------------------------
    # NeuroDream sleep/dream
    # ------------------------------------------------------------------

    def _handle_neurodream_command(self, message: str) -> Optional[str]:
        """Handle NeuroDream sleep/dream commands directly.

        Args:
            message: The user's message

        Returns:
            Formatted result string if NeuroDream command, None otherwise
        """
        msg_lower = message.lower()

        neurodream_keywords = [
            'go to sleep', 'sleep now', 'start sleeping', 'enter sleep',
            'dream status', 'sleep status', 'neurodream status',
            'wake up', 'stop sleeping',
            'dream journal', 'show dreams', 'recent dreams',
            'dream insights', 'show insights',
            'sleep patterns', 'consolidated patterns'
        ]

        if not any(kw in msg_lower for kw in neurodream_keywords):
            return None

        if not hasattr(self, 'neurodream') or self.neurodream is None:
            return "NeuroDream not available."

        # Handle specific patterns
        if any(kw in msg_lower for kw in ['go to sleep', 'sleep now', 'start sleeping', 'enter sleep']):
            if SleepPhase and self.neurodream.current_phase != SleepPhase.AWAKE:
                return f"Already in {self.neurodream.current_phase.value} phase."
            result = self.neurodream.enter_sleep(trigger="manual")
            if result.get("success"):
                return "Entering sleep mode... Beginning memory consolidation cycle."
            return f"Could not enter sleep: {result.get('error', 'Unknown error')}"

        if any(kw in msg_lower for kw in ['dream status', 'sleep status', 'neurodream status']):
            status = self.neurodream.get_status()
            phase_emoji = {
                "awake": "Awake",
                "light": "Light Sleep",
                "deep": "Deep Sleep",
                "rem": "REM Sleep",
                "waking": "Waking Up"
            }
            return (f"**NeuroDream Status**\n"
                   f"- Phase: {phase_emoji.get(status['phase'], status['phase'])}\n"
                   f"- Total Sessions: {status['total_sessions']}\n"
                   f"- Total Insights: {status['total_insights']}\n"
                   f"- Idle Minutes: {status['idle_minutes']:.1f}\n"
                   f"- Last Sleep: {status['last_sleep'] or 'Never'}")

        if any(kw in msg_lower for kw in ['wake up', 'stop sleeping']):
            if SleepPhase and self.neurodream.current_phase == SleepPhase.AWAKE:
                return "Already awake."
            result = self.neurodream.wake_up(reason="manual")
            summary = result.get("summary", {})
            return (f"Waking up...\n"
                   f"- Phases completed: {', '.join(summary.get('phases_completed', []))}\n"
                   f"- Insights generated: {summary.get('insights_generated', 0)}\n"
                   f"- Patterns found: {summary.get('patterns_found', 0)}")

        if any(kw in msg_lower for kw in ['dream journal', 'show dreams', 'recent dreams']):
            entries = self.neurodream.get_dream_journal(n=5)
            if not entries:
                return "No dream journal entries yet."
            lines = ["**Recent Dream Sessions:**"]
            for entry in entries[-5:]:
                phases = ', '.join(entry.get('phases_completed', []))
                insights = entry.get('insights_generated', 0)
                lines.append(f"- {entry.get('start_time', 'Unknown')[:16]}: {phases} ({insights} insights)")
            return '\n'.join(lines)

        if any(kw in msg_lower for kw in ['dream insights', 'show insights']):
            insights = self.neurodream.get_insights(n=5)
            if not insights:
                return "No dream insights generated yet."
            lines = ["**Recent Dream Insights:**"]
            for insight in insights[-5:]:
                lines.append(f"- [{insight.get('insight_type', 'unknown')}] {insight.get('content', '')[:100]}...")
            return '\n'.join(lines)

        if any(kw in msg_lower for kw in ['sleep patterns', 'consolidated patterns']):
            patterns = self.neurodream.get_patterns(n=5)
            if not patterns:
                return "No patterns consolidated yet."
            lines = ["**Consolidated Patterns:**"]
            for pattern in patterns[-5:]:
                lines.append(f"- [{pattern.get('pattern_type', 'unknown')}] {pattern.get('description', '')[:80]}...")
            return '\n'.join(lines)

        return None

    # ------------------------------------------------------------------
    # Git commands
    # ------------------------------------------------------------------

    def _handle_git_command(self, message: str) -> Optional[str]:
        """Handle Git commands directly, bypassing the LLM.

        Args:
            message: The user's message

        Returns:
            Formatted result string if Git command, None otherwise
        """
        message_lower = message.lower()

        # Check if this is a Git command - expanded natural language patterns
        git_keywords = [
            # Explicit git commands
            'git status', 'git log', 'git diff', 'git branch', 'git stash',
            # Branch queries
            'what branch', 'which branch', 'current branch', 'show branches', 'list branches',
            # Commit queries
            'show commits', 'recent commits', 'commit history', 'last commit',
            # Status queries
            'staged files', 'unstaged', 'untracked', 'show changes',
            'what changed', 'pending changes', 'working tree'
        ]
        if not any(kw in message_lower for kw in git_keywords):
            return None  # Not a Git command

        # Get the git tool
        git_tool = self.tools.get('git')
        if not git_tool:
            return "Git tool is not available."

        # Map natural language to specific git actions
        if any(kw in message_lower for kw in ['what branch', 'which branch', 'current branch', 'show branches', 'list branches']):
            result = git_tool.branch('.')
        elif any(kw in message_lower for kw in ['show commits', 'recent commits', 'commit history', 'last commit', 'git log']):
            result = git_tool.log('.', count=5)
        elif any(kw in message_lower for kw in ['staged files', 'unstaged', 'untracked', 'git status', 'what changed', 'pending changes', 'working tree']):
            result = git_tool.status('.')
        elif any(kw in message_lower for kw in ['show changes', 'git diff']):
            result = git_tool.diff('.')
        elif 'git stash' in message_lower:
            if 'list' in message_lower:
                result = git_tool.stash('.', 'list')
            elif 'pop' in message_lower:
                result = git_tool.stash('.', 'pop')
            else:
                result = git_tool.stash('.', 'push')
        else:
            # Fallback to execute() for other commands
            result = git_tool.execute(message)

        if result.get('success'):
            return result.get('output', str(result))
        else:
            return f"Git error: {result.get('error', 'Unknown error')}"

    # ------------------------------------------------------------------
    # Direct web search
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Search intent classifier — signal-scoring approach
    # ------------------------------------------------------------------

    # Words that HARD-BLOCK search (user is talking about local work)
    _SEARCH_BLOCKERS = frozenset([
        'my code', 'my file', 'my project', 'this file', 'this code',
        'this function', 'this class', 'this module', 'this bug',
        'fix this', 'edit this', 'run this', 'explain this',
        'refactor', 'debug this', 'compile', 'deploy this',
        'write a function', 'write a script', 'write a class',
        'write code', 'create a file', 'make a component',
        'merge conflict', 'git commit', 'git push', 'pull request',
        'unit test', 'test case', 'run tests', 'npm', 'pip install',
        'import error', 'syntax error', 'type error', 'traceback',
        'stack trace', 'error message',
    ])

    # Explicit search commands (+10 score, instant trigger)
    _SEARCH_COMMANDS = re.compile(
        r'^(?:search|google|look\s*up|find\s+(?:online|on\s+the\s+web|info|information|out\s+about))\b', re.I
    )

    # Temporal/recency words (+3 each)
    _TEMPORAL_WORDS = frozenset([
        'latest', 'newest', 'recent', 'current', 'new', 'now',
        'today', 'yesterday', 'this week', 'this month', 'this year',
        'just released', 'just announced', 'just launched', 'just dropped',
        'breaking', 'trending', 'upcoming', 'right now',
        '2024', '2025', '2026', '2027',
    ])

    # Information-seeking nouns (+2 each)
    _INFO_VERBS = frozenset([
        'news', 'updates', 'update', 'developments', 'announcements',
        'announcement', 'releases', 'release', 'changelog',
        'review', 'reviews', 'comparison', 'benchmark', 'benchmarks',
        'pricing', 'price', 'cost', 'specs', 'specifications',
        'features', 'roadmap', 'schedule', 'timeline',
        'status', 'availability', 'eta',
        'info', 'information', 'details', 'overview', 'summary',
        'breakthroughs', 'breakthrough', 'progress', 'results',
        'rumors', 'rumor', 'rumours', 'rumour', 'leak', 'leaks',
        'drama', 'controversy', 'scandal', 'crash', 'outage',
        'version', 'changelog', 'compatibility',
    ])

    # Real-world knowledge domains (+2 if topic is about these)
    _WORLD_DOMAINS = frozenset([
        # AI/ML
        'ai', 'ml', 'gpt', 'llm', 'llms', 'model', 'models', 'claude',
        'openai', 'google', 'meta', 'microsoft', 'apple', 'nvidia', 'amd', 'intel',
        'anthropic', 'deepmind', 'mistral', 'gemma', 'gemini', 'llama', 'phi',
        'qwen', 'deepseek', 'sora', 'midjourney', 'dall-e', 'chatgpt',
        'copilot', 'cursor', 'windsurf', 'codeium', 'tabnine', 'replit',
        # Tech / frameworks
        'python', 'javascript', 'typescript', 'rust', 'go', 'java', 'swift', 'kotlin',
        'react', 'nextjs', 'vue', 'angular', 'svelte', 'remix', 'astro',
        'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'vercel', 'cloudflare',
        'supabase', 'firebase', 'mongodb', 'postgres', 'redis',
        # Crypto / finance
        'crypto', 'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol',
        'binance', 'coinbase', 'ftx', 'defi', 'nft',
        'stock', 'stocks', 'market', 'economy', 'inflation', 'recession',
        'nasdaq', 'sp500', 'dow', 'fed', 'interest rate',
        # Geopolitics / society
        'war', 'election', 'elections', 'politics', 'climate', 'covid', 'pandemic',
        'trump', 'biden', 'putin', 'china', 'russia', 'ukraine', 'taiwan',
        'eu', 'nato', 'un', 'sanctions', 'tariff', 'tariffs',
        'law', 'regulation', 'policy', 'supreme court', 'congress',
        # Space / tech companies
        'spacex', 'nasa', 'tesla', 'starlink', 'boeing', 'amazon',
        # Consumer tech
        'iphone', 'android', 'samsung', 'pixel', 'macbook', 'windows',
        'steam', 'playstation', 'xbox', 'nintendo', 'switch',
        # Entertainment
        'movie', 'film', 'series', 'show', 'album', 'song', 'spotify', 'netflix',
        # Sports
        'tournament', 'championship', 'olympics', 'world cup', 'nba', 'nfl', 'fifa',
        # Science
        'research', 'paper', 'study', 'breakthrough', 'discovery',
        'vaccine', 'treatment', 'fda', 'clinical trial',
    ])

    # Sentence-level patterns that strongly imply web search (+3)
    _IMPLICIT_SEARCH = re.compile(
        r'(?:'
        r'what\s+happened\s+(?:to|with|at|in)\b'       # "what happened to X"
        r'|.+\s+(?:drama|controversy|scandal|backlash|outage|hack|breach|leak|layoffs?|acquisition|merger|ipo|bankrupt)'  # "X drama/scandal"
        r'|.+\s+(?:vs\.?|versus|compared\s+to|better\s+than|or)\s+.+'  # "X vs Y" comparisons
        r'|.+\b(?:latest|update|news|release|version)\s*$'  # trailing "X latest/news"
        r')', re.I
    )

    # Question patterns that signal factual queries (+2)
    _FACTUAL_Q = re.compile(
        r'(?:^|\b)(?:'
        r'whats\s+\w+'                                                              # "whats happening"
        r'|what(?:\'s|\s+is|\s+are|\s+was|\s+were|\s+happened|\s+about)'            # "what is/are/was..."
        r'|who(?:\'s|\s+is|\s+are|\s+was|\s+were|\s+won|\s+lost|\s+got|\s+made)'   # "who won/lost..."
        r'|when(?:\'s|\s+is|\s+was|\s+did|\s+does|\s+will)'                         # "when is/did..."
        r'|where(?:\'s|\s+is|\s+are|\s+can|\s+did)'                                 # "where is/can..."
        r'|how\s+(?:much|many|long|far|old|big|fast|well|good)'                     # "how much/many..."
        r'|is\s+there|are\s+there|has\s+there|have\s+there'                         # "is/are there..."
        r'|(?:has|have|did|does|will)\s+.+?\s+(?:been|come|released|launched|announced|started|happened|changed|dropped|shipped|crashed|failed|closed|collapsed|surged|spiked|won|lost)'  # "has X been released" / "did X crash"
        r'|(?:is|are)\s+.+?\s+(?:out|available|released|live|ready|dead|gone|down)'  # "is X out/available"
        r'|any\s+(?:news|updates?|info|word|details?|sign|chance)'                   # "any news/updates..."
        r'|show\s+me'                                                                 # "show me..."
        r')\b', re.I
    )

    # Patterns that extract a clean topic from common sentence structures
    _TOPIC_EXTRACTORS = [
        # "X news/updates/release" → X
        re.compile(r'^(?:any|the|latest|recent|new|give\s+me)?\s*(?:news|updates?|info|information|details?|developments?|word)\s+(?:on|about|for|regarding|from)\s+(.+)', re.I),
        # "whats/what's new/happening with X"
        re.compile(r'whats?\s+(?:is\s+)?(?:new|happening|going\s+on|up|the\s+(?:deal|story|latest|status|situation))\s+(?:with|in|at|on|for|about)\s+(.+)', re.I),
        re.compile(r'what(?:\'s|\s+is)\s+(?:new|happening|going\s+on|up|the\s+(?:deal|story|latest|status|situation))\s+(?:with|in|at|on|for|about)\s+(.+)', re.I),
        # "tell me about X" / "tell me the latest on X"
        re.compile(r'tell\s+me\s+(?:about\s+)?(?:the\s+)?(?:latest|newest|recent|current)?\s*(?:on|about|for|with|regarding)?\s*(.+)', re.I),
        # "has/did X been released/announced" (handles hyphens, dots, numbers in names)
        re.compile(r'(?:has|have|did|when\s+(?:did|does|will|is))\s+(.+?)\s+(?:been\s+)?(?:released?|launched?|announced?|come\s+out|dropped|shipped|started|arrived)', re.I),
        # "is X out/available/released"
        re.compile(r'(?:is|are)\s+(.+?)\s+(?:out|available|released|live|ready|here|dead|gone|down)\s*(?:yet|now|already)?', re.I),
        # "when is/does X come out / release"
        re.compile(r'when\s+(?:is|does|will|did)\s+(.+?)\s+(?:come\s+out|release|launch|drop|ship|start|arrive|happen)', re.I),
        # "X latest version / release"
        re.compile(r'(?:what(?:\'?s|\s+is)\s+)?(?:the\s+)?(?:latest|newest|current|most\s+recent)\s+(?:version|release|build|edition)\s+(?:of\s+)?(.+)', re.I),
        # "how much is X / price of X"
        re.compile(r'(?:what(?:\'?s|\s+is)|how\s+much\s+is)\s+(?:the\s+)?(?:current\s+)?(?:price|cost|value|rate)\s+(?:of|for)\s+(.+)', re.I),
        # "status of X"
        re.compile(r'(?:what(?:\'?s|\s+is)\s+)?(?:the\s+)?(?:current\s+)?(?:status|state|situation|progress)\s+(?:of|on|with|for)\s+(.+)', re.I),
        # "what happened to/with X"
        re.compile(r'what\s+happened\s+(?:to|with|at|in)\s+(.+)', re.I),
        # "who won/lost/got X" → extract the event/thing
        re.compile(r'who\s+(?:won|lost|got|made|created|invented|discovered|started|founded)\s+(?:the\s+)?(.+)', re.I),
        # "show me X" / "get me X"
        re.compile(r'(?:show|get|give|bring)\s+me\s+(?:the\s+)?(?:latest\s+|recent\s+|current\s+)?(.+)', re.I),
        # "research/search X" (explicit command — extract topic)
        re.compile(r'^(?:search|google|look\s*up|research|find)\s+(?:online\s+)?(?:the\s+web\s+)?(?:for\s+|about\s+|on\s+)?(.+)', re.I),
        # "updates/news on X" (noun-first)
        re.compile(r'^(?:updates?|news|developments?|announcements?|releases?|info|information)\s+(?:on|about|for|regarding|from)\s+(.+)', re.I),
        # "check X" / "check on X"
        re.compile(r'^check\s+(?:online\s+)?(?:the\s+web\s+)?(?:for\s+|on\s+|about\s+)?(.+)', re.I),
        # "find X" / "find out about X"
        re.compile(r'^find\s+(?:out\s+)?(?:about\s+|info\s+(?:on|about)\s+)?(.+)', re.I),
    ]

    def _score_search_intent(self, message: str) -> tuple[int, Optional[str]]:
        """Score how likely a message needs web search. Returns (score, extracted_query).

        Score interpretation:
            >= 6: definitely search (explicit command or strong multi-signal)
            4-5:  probably search (temporal + factual question)
            <= 3: probably not search
        """
        msg = message.lower().strip()
        words = set(re.split(r'\W+', msg))
        score = 0

        # --- Hard blockers: return immediately ---
        if len(msg) < 8:
            return (0, None)
        for blocker in self._SEARCH_BLOCKERS:
            if blocker in msg:
                return (0, None)

        # --- Explicit search command: +10 (instant win) ---
        if self._SEARCH_COMMANDS.search(msg):
            score += 10

        # --- Temporal/recency signals: +3 each (max +6) ---
        temporal_hits = 0
        for tw in self._TEMPORAL_WORDS:
            if tw in msg:
                temporal_hits += 1
        score += min(temporal_hits * 3, 6)

        # --- Info-seeking nouns: +2 each (max +4) ---
        info_hits = sum(1 for w in self._INFO_VERBS if w in words)
        score += min(info_hits * 2, 4)

        # --- Real-world domain overlap: +2 if any match ---
        if words & self._WORLD_DOMAINS:
            score += 2

        # --- Factual question pattern: +2 ---
        if self._FACTUAL_Q.search(msg):
            score += 2

        # --- Implicit search sentence patterns: +3 ---
        if self._IMPLICIT_SEARCH.search(msg):
            score += 3

        # --- Ends with "?" on a non-trivial message: +1 ---
        if msg.endswith('?') and len(msg) > 15:
            score += 1

        # --- "online" / "web" / "internet" / "google" explicit hint: +4 ---
        if any(h in msg for h in ('online', 'on the web', 'on the internet', 'the web')):
            score += 4

        # --- Extract query topic ---
        query = None
        for extractor in self._TOPIC_EXTRACTORS:
            m = extractor.search(msg)
            if m:
                query = m.group(1).strip()
                query = re.sub(r'^(?:the|a|an)\s+', '', query)  # strip leading articles
                query = re.sub(r'[.,!?"\';:]+$', '', query).strip()
                if len(query) > 2:
                    break
                query = None

        # If no extractor matched but score is high, use cleaned full message
        if not query and score >= 4:
            query = message.strip()
            query = re.sub(r'[.,!?"\';:]+$', '', query).strip()

        return (score, query)

    def _needs_web_search(self, message: str) -> Optional[str]:
        """Detect if a message needs web search via signal scoring.

        Returns the search query if web search is needed, None otherwise.
        """
        score, query = self._score_search_intent(message)
        if score >= 4 and query and len(query) > 2:
            return query
        return None

    def _handle_direct_search(self, message: str, synthesize: bool = True) -> Optional[str]:
        """Handle search requests directly, bypassing agent loop.

        Uses signal-scoring to detect search intent. Explicit commands
        ("search for X") and strong multi-signal queries ("any news on
        gemma 4 models") both trigger immediate search with synthesis.

        Args:
            message: The user's message
            synthesize: Whether to use LLM to synthesize results

        Returns:
            Formatted search results if search request, None otherwise
        """
        message_lower = message.lower().strip()

        # If user wants comprehensive/detailed research, let it go through full agent loop
        comprehensive_keywords = ['comprehensive', 'detailed', 'in-depth', 'thorough', 'deep dive', 'extensive', 'full analysis']
        if any(kw in message_lower for kw in comprehensive_keywords):
            return None

        # Score the message
        score, query = self._score_search_intent(message)

        # Score < 4: not a search request
        if score < 4 or not query or len(query) < 3:
            return None

        logger.debug(f"[DIRECT SEARCH] score={score} query='{query}'")

        try:
            from aura.tools.search_fallback import web_search_with_fallback
            result = web_search_with_fallback(query, max_results=5, tool_registry=self.tools)

            if not result.get("success"):
                return f"Search failed: {result.get('error', 'Unknown error')}"

            results = result.get("results", [])
            if not results:
                return f"No results found for '{query}'."

            # Format raw results
            raw_results = ""
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "No description")
                url = r.get("url", "")
                raw_results += f"{i}. {title}\n   {snippet}\n   URL: {url}\n\n"

            # Synthesize with LLM if available and requested
            if synthesize and hasattr(self, 'brain'):
                try:
                    synthesis_prompt = f"""You are summarizing REAL web search results for the query: '{query}'

SEARCH RESULTS:
{raw_results}

STRICT RULES:
- ONLY use information that appears in the search results above
- Do NOT add information from your own knowledge — if it's not in the results, don't include it
- Do NOT make up or fabricate any facts, dates, or details
- Include the actual URLs from the results as clickable references
- If the results don't fully answer the query, say so honestly
- Format with markdown for readability
- Be concise but accurate"""

                    synthesized = self.brain.think(synthesis_prompt)
                    if synthesized and len(synthesized) > 50:
                        return synthesized
                except (AttributeError, TypeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
                    logger.debug(f"[DIRECT SEARCH] Synthesis failed, returning raw: {e}")

            # Fallback to formatted raw results
            formatted = f"Here's what I found for '{query}':\n\n"
            for i, r in enumerate(results[:5], 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "No description")
                url = r.get("url", "")
                formatted += f"{i}. **{title}**\n   {snippet}\n   {url}\n\n"

            return formatted.strip()

        except (AttributeError, KeyError, TypeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
            logger.debug(f"[DIRECT SEARCH] Error: {e}")
            return f"Search error: {e}"

    # ------------------------------------------------------------------
    # Direct crypto price
    # ------------------------------------------------------------------

    def _handle_direct_crypto(self, message: str) -> Optional[str]:
        """Handle crypto price requests directly, bypassing agent loop.

        This prevents the LLM from hallucinating crypto prices.
        User says "BTC price" -> fetches real BTC price from API.

        Args:
            message: The user's message

        Returns:
            Formatted price info if crypto request, None otherwise
        """
        message_lower = message.lower().strip()

        # Crypto symbols and names mapping
        crypto_map = {
            'btc': 'bitcoin', 'bitcoin': 'bitcoin',
            'eth': 'ethereum', 'ethereum': 'ethereum',
            'sol': 'solana', 'solana': 'solana',
            'ada': 'cardano', 'cardano': 'cardano',
            'doge': 'dogecoin', 'dogecoin': 'dogecoin',
            'xrp': 'ripple', 'ripple': 'ripple',
            'dot': 'polkadot', 'polkadot': 'polkadot',
            'bnb': 'binancecoin', 'binance': 'binancecoin',
            'avax': 'avalanche-2', 'avalanche': 'avalanche-2',
            'matic': 'matic-network', 'polygon': 'matic-network',
        }

        # Patterns for crypto price requests
        crypto_patterns = [
            r'(?:what(?:\'s| is) )?(?:the )?(?:current )?(?:price (?:of )?)?(\w+)\s*price',
            r'(?:what(?:\'s| is) )?(?:the )?(?:current )?price (?:of )?(\w+)',
            r'how much (?:is|does) (\w+)(?: cost)?',
            r'^(\w+)\s*price$',
            r'^price\s*(?:of\s+)?(\w+)$',
            r'(\w+) (?:price|value|cost)',
        ]

        # Try to extract crypto name
        crypto_id = None
        for pattern in crypto_patterns:
            match = re.search(pattern, message_lower)
            if match:
                potential_crypto = match.group(1).strip()
                if potential_crypto in crypto_map:
                    crypto_id = crypto_map[potential_crypto]
                    break

        if not crypto_id:
            return None  # Not a crypto price request

        # Check if crypto_price tool is available
        if 'crypto_price' not in self.tools:
            return "Crypto price tool not available."

        logger.debug(f"[DIRECT CRYPTO] Fetching price for: {crypto_id}")

        try:
            tool = self.tools['crypto_price']
            result = tool.get_price(crypto_id)

            if not result.get("success"):
                return f"Failed to get price: {result.get('error', 'Unknown error')}"

            # Format the response
            price = result.get("price", 0)
            change_24h = result.get("change_24h", 0)
            name = result.get("name", crypto_id.title())
            symbol = result.get("symbol", "").upper()

            change_emoji = "\U0001f4c8" if change_24h >= 0 else "\U0001f4c9"
            change_sign = "+" if change_24h >= 0 else ""

            formatted = f"**{name} ({symbol})** {change_emoji}\n"
            formatted += f"\U0001f4b0 Current Price: **${price:,.2f}**\n"
            formatted += f"\U0001f4ca 24h Change: {change_sign}{change_24h:.2f}%"

            return formatted

        except (AttributeError, KeyError, TypeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
            logger.debug(f"[DIRECT CRYPTO] Error: {e}")
            return f"Crypto price error: {e}"

    # ------------------------------------------------------------------
    # Direct code execution
    # ------------------------------------------------------------------

    def _handle_direct_code(self, message: str) -> Optional[str]:
        """Handle code execution requests directly, bypassing agent loop.

        This ensures code is actually executed when the user asks for it.
        Handles: "calculate X", "run python for X", "execute code for X", "what is X!" (factorial)

        Args:
            message: The user's message

        Returns:
            Formatted execution result if code request, None otherwise
        """
        message_lower = message.lower().strip()

        # Check if code_executor tool is available
        if 'code_executor' not in self.tools:
            return None

        # Patterns that indicate code execution intent
        execute_patterns = [
            r'^(?:please\s+)?(?:run|execute)\s+(?:python\s+)?(?:code\s+)?(?:for|to)\s+(.+)$',
            r'^(?:please\s+)?(?:write\s+and\s+)?(?:run|execute)\s+(?:python\s+)?(?:code\s+)?(?:for|to)\s+(.+)$',
            r'^(?:please\s+)?calculate\s+(.+)$',
            r'^(?:please\s+)?compute\s+(.+)$',
            r'^what\s+is\s+(\d+)\s*[!]$',  # "what is 20!" -> factorial
            r'^(\d+)\s*[!]$',  # "20!" -> factorial
            r'^(?:please\s+)?(?:find|generate|show)\s+(?:the\s+)?(?:first\s+)?(\d+)\s+(?:fibonacci|fib)\s*(?:numbers?)?$',
            r'^(?:please\s+)?(?:fibonacci|fib)\s+(?:sequence\s+)?(?:of\s+)?(\d+)$',
        ]

        task_description = None
        code_to_run = None

        for pattern in execute_patterns:
            match = re.match(pattern, message_lower, re.IGNORECASE)
            if match:
                task_description = match.group(1).strip()
                break

        # Also check for explicit code in the message (```python ... ```)
        # When attachment context is present, only scan the user's actual request — not document content
        scan_target = message
        if '[FILE_ATTACHMENT_CONTEXT]' in message and 'User request:' in message:
            scan_target = message.split('User request:', 1)[-1]
        code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', scan_target, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            code_to_run = code_block_match.group(1).strip()
            task_description = "provided code"

        if not task_description and not code_to_run:
            return None  # Not a code execution request

        logger.debug(f"[DIRECT CODE] Task: '{task_description}'")

        # Generate code if not provided
        if not code_to_run:
            # Handle common patterns directly without LLM
            if re.match(r'^\d+\s*!?$', task_description) or 'factorial' in task_description:
                # Factorial
                num_match = re.search(r'(\d+)', task_description)
                if num_match:
                    n = num_match.group(1)
                    code_to_run = f"import math\nresult = math.factorial({n})\nprint(f'{n}! = {{result}}')"

            elif 'fibonacci' in task_description or 'fib' in task_description:
                # Fibonacci
                num_match = re.search(r'(\d+)', task_description)
                n = num_match.group(1) if num_match else "10"
                code_to_run = f"""def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]

result = fibonacci({n})
print(f"First {n} Fibonacci numbers: {{result}}")"""

            elif 'prime' in task_description:
                # Prime numbers or prime check
                num_match = re.search(r'(\d+)', task_description)
                if num_match:
                    n = num_match.group(1)
                    if 'first' in task_description or 'generate' in task_description:
                        code_to_run = f"""def sieve_of_eratosthenes(limit):
    primes = []
    is_prime = [True] * (limit + 1)
    for num in range(2, limit + 1):
        if is_prime[num]:
            primes.append(num)
            for multiple in range(num * num, limit + 1, num):
                is_prime[multiple] = False
    return primes

# Generate enough primes
primes = sieve_of_eratosthenes({int(n) * 15})[:{int(n)}]
print(f"First {int(n)} prime numbers: " + str(primes))"""
                    else:
                        code_to_run = f"""def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

result = is_prime({n})
print(f"{n} is{{'' if result else ' not'}} a prime number")"""

            else:
                # Use LLM to generate code for complex requests
                from aura.brain import TaskType
                code_prompt = f"""Write Python code to: {task_description}

Requirements:
- Include print statements to show the output
- Keep it simple and readable
- Only output the Python code, nothing else

Python code:"""

                generated = self.brain.think(code_prompt, use_history=False, task_type=TaskType.CODE)

                # Extract code from response
                code_match = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', generated, re.DOTALL)
                if code_match:
                    code_to_run = code_match.group(1).strip()
                else:
                    # Try to use the whole response if it looks like code
                    lines = generated.strip().split('\n')
                    code_lines = [l for l in lines if any(c in l for c in ['print', 'def ', 'import ', '=', 'for ', 'if ', 'return'])]
                    if code_lines:
                        code_to_run = '\n'.join(code_lines)
                    else:
                        code_to_run = generated.strip()

        if not code_to_run:
            return None

        # SECURITY: Validate LLM-generated code before execution.
        from aura.security.tool_validator import validate_script_code
        is_safe, reason = validate_script_code(code_to_run, "<llm_generated>")
        if not is_safe:
            logger.warning(f"[Agent] Blocked unsafe LLM-generated code: {reason}")
            return None

        # Execute the code
        try:
            tool = self.tools['code_executor']
            result = tool.execute(code_to_run)

            # Format the response
            formatted = "**Code Execution Result**\n\n"
            formatted += f"```python\n{code_to_run}\n```\n\n"

            if result.get("success"):
                output = result.get("output", "").strip()
                if output:
                    formatted += f"**Output:**\n```\n{output}\n```"
                else:
                    formatted += "**Output:** (no output)"
            else:
                error = result.get("errors", result.get("error", "Unknown error"))
                formatted += f"**Error:**\n```\n{error}\n```"

            return formatted

        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, OSError) as e:
            logger.debug(f"[DIRECT CODE] Error: {e}")
            return f"Code execution error: {e}"
