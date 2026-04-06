"""Direct command handlers mixin — monologue, neurodream, git, search, crypto, code.

Extracted from agent.py (2026-03-23) to reduce class size.
These bypass the LLM agent loop for deterministic or direct tool execution.
All methods assume self has: tools, brain, monologue, neurodream, identity.
"""

import json
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

    def _handle_direct_search(self, message: str, synthesize: bool = True) -> Optional[str]:
        """Handle explicit search requests directly, bypassing agent loop.

        This prevents the LLM's planning phase from hallucinating different queries.
        User says "search for AI news" -> searches for "AI news" exactly.
        Results are then synthesized by the LLM for better presentation.

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
            return None  # Let full agent loop handle this

        # Strip common greeting/name prefixes to allow "hey aura search for X"
        prefix_patterns = [
            r'^(?:hey\s+)?(?:aura|assistant|ai|bot)[,!.]?\s*',
            r'^(?:hi|hello|hey)[,!.]?\s*',
            r'^(?:okay|ok|yo)[,!.]?\s*',
            r'^(?:alright|sure|yeah|yep|yes)[,!.]?\s*',
            r'^(?:let\'?s|lets|can\s+you|could\s+you|please|pls)[,!.]?\s*',
            r'^(?:i\s+want\s+(?:you\s+)?to|i\s+need\s+(?:you\s+)?to)[,!.]?\s*',
            r'^(?:go\s+ahead\s+and|now)[,!.]?\s*',
        ]
        for prefix in prefix_patterns:
            message_lower = re.sub(prefix, '', message_lower, flags=re.IGNORECASE).strip()

        # Keywords that indicate "search online" intent
        online_keywords = ['online', 'web', 'internet', 'google', 'latest', 'current', 'recent', 'news', 'today']

        # Check if this is an ambiguous research request (no topic or unclear intent)
        ambiguous_patterns = [
            r'^(?:do\s+)?(?:a\s+)?research$',
            r'^(?:do\s+)?(?:a\s+)?(?:deep\s+)?search$',
            r'^(?:can\s+you\s+)?research$',
            r'^(?:please\s+)?research$',
            r'^look\s+(?:something\s+)?up$',
            r'^find\s+(?:something|info|information)$',
        ]

        for pattern in ambiguous_patterns:
            if re.match(pattern, message_lower, re.IGNORECASE):
                return ("I'd be happy to help with research! \U0001f50d\n\n"
                        "**What would you like me to do?**\n"
                        "1. **Search online** - Get the latest info from the web\n"
                        "2. **Use my knowledge** - Answer from what I already know\n\n"
                        "Just tell me the topic! For example:\n"
                        "- \"Search online for quantum computing\"\n"
                        "- \"Tell me about quantum computing\"\n"
                        "- \"Research latest AI news online\"")

        # Patterns for EXPLICIT ONLINE search requests
        search_patterns = [
            # Direct search commands
            r'^search\s+(?:online\s+)?(?:the\s+web\s+)?(?:for\s+)?["\']?(.+?)["\']?$',
            r'^(?:web\s+)?search[:\s]+["\']?(.+?)["\']?$',
            r'^look\s+up\s+["\']?(.+?)["\']?$',
            r'^google\s+["\']?(.+?)["\']?$',
            r'^find\s+(?:online|on the web)\s+["\']?(.+?)["\']?$',
            r'^search\s+for\s+["\']?(.+?)["\']?[.,!?]?$',
            # Flexible patterns
            r'^do\s+(?:a\s+)?(?:deep\s+)?search\s+(?:online\s+)?(?:for\s+|about\s+|on\s+)?["\']?(.+?)["\']?$',
            r'^(?:please\s+)?(?:can\s+you\s+)?search\s+(?:online\s+)?(?:for\s+|about\s+)?["\']?(.+?)["\']?$',
            r'^(?:deep\s+)?search\s+(?:online\s+)?(?:for\s+|about\s+|on\s+)?["\']?(.+?)["\']?$',
            # Research with online intent
            r'^research\s+(?:online\s+)?(?:about\s+|on\s+)?["\']?(.+?)["\']?\s+online$',
            r'^research\s+online\s+(?:about\s+|on\s+|for\s+)?["\']?(.+?)["\']?$',
            # News/latest patterns (always online)
            r'^(?:get|find|show)\s+(?:me\s+)?(?:the\s+)?(?:latest|recent|current)\s+(?:news\s+)?(?:on|about|for)\s+["\']?(.+?)["\']?$',
            r'^what(?:\'s|\s+is)\s+(?:the\s+)?(?:latest|recent|current)\s+(?:news\s+)?(?:on|about)\s+["\']?(.+?)["\']?$',
            # Lookup patterns
            r'^look\s+(?:this\s+)?up[:\s]+["\']?(.+?)["\']?$',
            r'^(?:can\s+you\s+)?(?:please\s+)?look\s+up\s+["\']?(.+?)["\']?$',
            # Find info patterns
            r'^find\s+(?:me\s+)?(?:info|information)\s+(?:on|about)\s+["\']?(.+?)["\']?$',
            r'^get\s+(?:me\s+)?(?:info|information)\s+(?:on|about)\s+["\']?(.+?)["\']?$',
        ]

        # Extract the search query
        query = None
        for pattern in search_patterns:
            match = re.match(pattern, message_lower, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Remove trailing punctuation
                query = re.sub(r'[.,!?]+$', '', query).strip()
                break

        # If no explicit pattern matched, check for "research X" with online keywords
        if not query:
            research_match = re.match(r'^(?:do\s+)?(?:a\s+)?research\s+(?:about\s+|on\s+)?["\']?(.+?)["\']?$', message_lower)
            if research_match:
                potential_query = research_match.group(1).strip()
                # Check if any online keyword is present
                if any(kw in message_lower for kw in online_keywords):
                    query = potential_query
                else:
                    # Ambiguous - has topic but unclear if online or knowledge
                    return (f"I can help you research **{potential_query}**! \U0001f50d\n\n"
                            f"Would you like me to:\n"
                            f"1. **Search online** - \"search online for {potential_query}\"\n"
                            f"2. **Use my knowledge** - \"tell me about {potential_query}\"\n\n"
                            f"Which would you prefer?")

        if not query:
            return None  # Not an explicit search request

        # Check if web_search tool is available
        if 'web_search' not in self.tools:
            return "Web search tool not available."

        logger.debug(f"[DIRECT SEARCH] User query: '{query}'")

        try:
            # Use fallback chain (Tavily → Brave → SearXNG) instead of SearXNG-only
            try:
                from aura.tools.search_fallback import web_search_with_fallback
                result = web_search_with_fallback(query, max_results=5, tool_registry=self.tools)
            except ImportError:
                # Fallback to direct web_search if search_fallback not available
                tool = self.tools['web_search']
                result = tool.search(query, num_results=5)

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
                    synthesis_prompt = f"""Based on these web search results for '{query}', provide a helpful summary:

{raw_results}

Instructions:
- Summarize the key information from these results
- Include relevant URLs as references
- Keep it concise but informative
- Format nicely with markdown"""

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
            formatted = f"**Code Execution Result**\n\n"
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
