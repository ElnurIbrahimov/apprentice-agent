"""Main agent implementation with observe/plan/act/evaluate/remember loop."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

from .brain import OllamaBrain, TaskType
from .identity import load_identity, get_identity_prompt, detect_name_change, detect_personality_change, update_name, update_personality
from .memory import MemorySystem
from .metacognition import MetacognitionLogger
from .config import Config
from .tools import FileSystemTool, WebSearchTool, CodeExecutorTool, ScreenshotTool, VisionTool, PDFReaderTool, ClipboardTool, ArxivSearchTool, BrowserTool, SystemControlTool, NotificationTool, ToolBuilderTool, MarketplaceTool, FluxMindTool, FLUXMIND_AVAILABLE, RegexBuilderTool, GitTool, PersonaPlexTool, ClawdbotTool, EvoEmoTool, get_tone_modifier, build_adaptive_system_prompt, get_monologue, KnowledgeGraphTool, get_knowledge_graph, MetacognitiveGuardian, GuardianConfig, NeuroDreamEngine, SleepPhase


class AgentPhase(Enum):
    """Phases of the agent loop."""
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    EVALUATE = "evaluate"
    REMEMBER = "remember"


@dataclass
class AgentState:
    """Current state of the agent."""
    goal: str = ""
    phase: AgentPhase = AgentPhase.OBSERVE
    observations: str = ""
    current_plan: str = ""
    last_action: Optional[dict] = None
    last_result: Optional[dict] = None
    evaluation: Optional[dict] = None
    iteration: int = 0
    completed: bool = False
    history: list = field(default_factory=list)
    gathered_content: str = ""  # Store content gathered from searches for summarization


class ApprenticeAgent:
    """An AI agent that learns and acts using observe/plan/act/evaluate/remember."""

    def __init__(self):
        self.brain = OllamaBrain()
        self.memory = MemorySystem()
        self.tools = {
            "filesystem": FileSystemTool(),
            "web_search": WebSearchTool(),
            "code_executor": CodeExecutorTool(),
            "screenshot": ScreenshotTool(),
            "vision": VisionTool(),
            "pdf_reader": PDFReaderTool(),
            "clipboard": ClipboardTool(),
            "arxiv_search": ArxivSearchTool(),
            "browser": BrowserTool(),
            "system_control": SystemControlTool(),
            "notifications": NotificationTool(),
            "tool_builder": ToolBuilderTool(),
            "marketplace": MarketplaceTool(),
            "regex_builder": RegexBuilderTool(),
            "git": GitTool(),
            "clawdbot": ClawdbotTool(),
            "evoemo": EvoEmoTool(),
            "inner_monologue": get_monologue(),
            "knowledge_graph": get_knowledge_graph()
        }
        # Connect inner monologue to EvoEmo for emotional awareness
        self.monologue = self.tools["inner_monologue"]
        if "evoemo" in self.tools:
            self.monologue.connect_evoemo(self.tools["evoemo"])
        # Add PersonaPlex if enabled (Tool #17)
        if Config.PERSONAPLEX_ENABLED:
            self.tools["personaplex"] = PersonaPlexTool()
            print("[LOADED] PersonaPlex - Real-time full-duplex voice (requires HF_TOKEN)")
        # Add FluxMind if available
        if FLUXMIND_AVAILABLE:
            models_path = Path(__file__).parent.parent / "models" / "fluxmind_v0751.pt"
            self.tools["fluxmind"] = FluxMindTool(str(models_path))
            if self.tools["fluxmind"].is_available():
                print("[LOADED] FluxMind v0.75.1 - Calibrated reasoning engine")
        self.state = AgentState()
        self.max_iterations = 10
        self.metacognition = MetacognitionLogger()
        self.use_fastpath = True  # Enable fast-path by default
        self.identity = load_identity()  # Load agent identity
        self.custom_tool_keywords = {}  # Map of keyword -> tool_name for custom tools
        self._load_custom_tools()  # Load active custom tools

        # Initialize Metacognitive Guardian (Tool #23)
        self.guardian = MetacognitiveGuardian(
            inner_monologue=self.monologue,
            evoemo=self.tools.get("evoemo"),
            knowledge_graph=self.tools.get("knowledge_graph"),
            config=GuardianConfig(monitoring_level="medium")
        )
        self._pending_prediction = None  # Track for outcome recording
        print("[LOADED] Metacognitive Guardian - Failure prediction system")

        # Initialize NeuroDream (Tool #24) - Sleep/Dream Memory Consolidation
        self.neurodream = NeuroDreamEngine(
            knowledge_graph=self.tools.get("knowledge_graph"),
            hybrid_memory=None,  # Will be set if hybrid_memory is available
            evoemo=self.tools.get("evoemo"),
            inner_monologue=self.monologue,
            chromadb=self.memory.collection if hasattr(self.memory, 'collection') else None,
            idle_threshold_minutes=30
        )
        print("[LOADED] NeuroDream - Sleep/dream memory consolidation")

    def _load_custom_tools(self) -> None:
        """Load active custom tools from registry."""
        import json
        import importlib.util

        registry_path = Path(__file__).parent.parent / "data" / "custom_tools.json"
        if not registry_path.exists():
            return

        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)

            for tool_entry in registry.get("tools", []):
                if tool_entry.get("status") != "active":
                    continue

                tool_name = tool_entry["name"]
                tool_file = Path(tool_entry.get("file", ""))
                if not tool_file.exists():
                    continue

                try:
                    # Dynamic import of custom tool
                    spec = importlib.util.spec_from_file_location(
                        tool_name,
                        tool_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Get the tool class
                        class_name = tool_entry.get("class_name")
                        if hasattr(module, class_name):
                            tool_class = getattr(module, class_name)
                            self.tools[tool_name] = tool_class()
                            print(f"[LOADED] Custom tool: {tool_name}")

                            # Load keywords for this tool
                            keywords = tool_entry.get("keywords", [])
                            if not keywords:
                                # Generate default keywords from name and description
                                keywords = self._generate_default_keywords(
                                    tool_name,
                                    tool_entry.get("description", ""),
                                    tool_entry.get("functions", [])
                                )
                            for kw in keywords:
                                self.custom_tool_keywords[kw.lower()] = tool_name
                except Exception as e:
                    print(f"[ERROR] Failed to load custom tool {tool_name}: {e}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ERROR] Failed to load custom tools registry: {e}")

    def _generate_default_keywords(self, name: str, description: str, functions: list) -> list[str]:
        """Generate default keywords for a custom tool if not provided.

        Args:
            name: Tool name
            description: Tool description
            functions: List of function names

        Returns:
            List of keywords for tool detection
        """
        import re
        keywords = set()

        # Add words from tool name
        for word in name.lower().split('_'):
            if len(word) > 2:
                keywords.add(word)
        keywords.add(name.lower().replace('_', ' '))

        # Add words from description
        stop_words = {'a', 'an', 'the', 'is', 'are', 'to', 'of', 'in', 'for', 'on',
                      'with', 'at', 'by', 'from', 'and', 'or', 'but', 'can', 'will'}
        desc_words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        for word in desc_words:
            if word not in stop_words and len(word) > 2:
                keywords.add(word)

        # Add function names
        for func in functions:
            for word in func.lower().split('_'):
                if len(word) > 2:
                    keywords.add(word)

        # Add common variations
        if 'bmi' in name.lower():
            keywords.update(['body mass index', 'height and weight'])
        if 'temperature' in name.lower():
            keywords.update(['celsius', 'fahrenheit', 'temp'])

        return list(keywords)

    def _is_simple_query(self, goal: str) -> bool:
        """Check if the goal is a simple conversational query.

        Simple queries include:
        - Greetings (hello, hi, hey, thanks, bye, etc.)
        - Questions about the agent itself (who are you, what can you do)
        - Yes/no questions and short conversational queries
        - Opinion/preference questions
        - General knowledge questions without tool needs
        """
        goal_lower = goal.lower().strip()
        words = goal_lower.split()

        # Greeting patterns - always fast-path
        greetings = [
            'hello', 'hi', 'hey', 'greetings', 'howdy', 'yo',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'thanks', 'thank you', 'thx', 'bye', 'goodbye', 'see you',
            'how are you', "what's up", 'whats up', 'sup', "how's it going",
            'nice to meet you', 'pleased to meet you'
        ]
        for greeting in greetings:
            if goal_lower.startswith(greeting) or goal_lower == greeting:
                return True

        # Questions about the agent itself - always fast-path
        agent_patterns = [
            'who are you', 'what are you', 'are you an ai', 'are you a bot',
            'are you real', 'are you human', 'what can you do', 'what do you do',
            'how do you work', 'what is your name', "what's your name",
            'tell me about yourself', 'introduce yourself', 'your capabilities',
            'what are your abilities', 'can you help', 'how can you help',
            'what should i call you', 'who made you', 'who created you',
            'are you chatgpt', 'are you gpt', 'are you claude', 'are you llama',
            'what model are you', 'what llm are you'
        ]
        for pattern in agent_patterns:
            if pattern in goal_lower:
                return True

        # Yes/no question starters - usually conversational
        yesno_starters = [
            'is it', 'is this', 'is that', 'are you', 'are there', 'are we',
            'can you', 'can i', 'could you', 'would you', 'should i', 'should we',
            'do you', 'does it', 'does this', 'did you', 'have you', 'has it',
            'will you', 'will it', 'was it', 'were you', 'is there'
        ]

        # Opinion/conversational starters - usually fast-path
        conversational_starters = [
            'what do you think', 'what is your opinion', 'do you like',
            'do you prefer', 'which is better', "what's the best",
            'tell me a joke', 'tell me something', 'say something',
            'how should i', 'what should i', 'why is', 'why do', 'why are',
            'when is', 'when do', 'when should', 'where is', 'where do',
            'explain', 'describe', 'define', 'what is a', 'what is the',
            'who is', 'who was', 'how many', 'how much', 'how long',
            'what happened', 'what does', 'what means', 'meaning of'
        ]

        # Tool keywords that REQUIRE the full agent loop
        # Be specific - only trigger for clear tool actions
        tool_keywords = [
            'search the web', 'search online', 'look up online', 'google',
            'take a screenshot', 'capture screen', 'screenshot',
            'read file', 'open file', 'list files', 'list directory',
            'run code', 'execute code', 'run python', 'execute python',
            'calculate', 'compute', 'factorial', 'fibonacci',
            'read pdf', 'open pdf', 'extract from pdf',
            'clipboard', 'copy to clipboard', 'paste from clipboard',
            'analyze image', 'look at image', 'describe image',
            'current weather', 'weather in', 'news about', 'latest news',
            'stock price', 'bitcoin price', 'crypto price',
            'arxiv', 'research paper', 'academic paper', 'find papers',
            'download paper', 'search papers', 'summarize papers', 'compare papers',
            'browse', 'open website', 'go to', 'visit url', 'visit site',
            'click', 'navigate to', 'google search', 'open page',
            'volume', 'brightness', 'open app', 'launch app', 'open notepad',
            'open calculator', 'open browser', 'open chrome', 'open firefox',
            'open vscode', 'open terminal', 'system info', 'cpu usage',
            'ram usage', 'memory usage', 'lock screen', 'set volume', 'set brightness',
            'remind', 'reminder', 'notify', 'notification', 'alert', 'schedule',
            'every day', 'every morning', 'every evening', 'daily at', 'weekly',
            'in 5 minutes', 'in 10 minutes', 'in 30 minutes', 'in an hour',
            'set reminder', 'set alarm', 'remind me',
            'create tool', 'make tool', 'build tool', 'new tool', 'i need a tool',
            'custom tool', 'generate tool', 'tool builder', 'list tools', 'test tool',
            'enable tool', 'disable tool', 'delete tool', 'remove tool',
            'marketplace', 'plugin', 'browse plugins', 'search plugins', 'install plugin',
            'download tool', 'share tool', 'publish tool', 'uninstall plugin', 'my plugins',
            'installed plugins', 'rate plugin', 'update plugin', 'plugin marketplace',
            'fluxmind', 'ask fluxmind', 'confidence check', 'calibrated', 'uncertainty',
            'verify sequence', 'how confident',
            'regex', 'regular expression', 'pattern', 'match pattern', 'regex pattern',
            'build regex', 'test regex', 'explain regex', 'validate regex',
            'git', 'commit', 'push', 'pull', 'branch', 'repository', 'repo',
            'git status', 'git log', 'git diff', 'git stash', 'clone',
            'what branch', 'which branch', 'current branch', 'show commits',
            'recent commits', 'staged files', 'unstaged', 'untracked',
            'show changes', 'list branches', 'show branches',
            'personaplex', 'duplex', 'realtime voice', 'real-time voice',
            'natural voice', 'voice server', 'start voice', 'stop voice',
            'set voice', 'change voice', 'list voices', 'set persona',
            'clawdbot', 'send message', 'send whatsapp', 'send telegram',
            'whatsapp message', 'telegram message', 'discord message',
            'signal message', 'imessage', 'text to', 'message to',
            'evoemo', 'mood', 'emotion', 'how am i feeling', 'my mood',
            'mood history', 'emotional state', 'clear mood',
            'inner monologue', 'show thoughts', 'think aloud', 'verbosity',
            'why did you do that', 'export thoughts', 'your thoughts',
            'reasoning chain', 'what were you thinking',
            'guardian stats', 'guardian status', 'guardian level',
            'set guardian', 'monitoring level', 'show predictions',
            'failure patterns', 'reset guardian',
            # NeuroDream keywords (Tool #24)
            'go to sleep', 'sleep now', 'enter sleep', 'start sleep',
            'dream status', 'sleep status', 'neurodream status',
            'wake up', 'stop sleeping', 'interrupt sleep',
            'dream journal', 'show dreams', 'recent dreams',
            'dream insights', 'show insights', 'sleep insights',
            'sleep patterns', 'consolidated patterns'
        ]

        # Check if it clearly needs a tool
        needs_tool = any(kw in goal_lower for kw in tool_keywords)
        if needs_tool:
            return False  # Use full agent loop

        # Check if query matches any custom tool keywords
        for kw in self.custom_tool_keywords:
            if kw in goal_lower:
                return False  # Use full agent loop for custom tools

        # Check for yes/no questions (usually don't need tools)
        for starter in yesno_starters:
            if goal_lower.startswith(starter):
                # But not if it's asking to perform an action
                action_words = ['search', 'find', 'calculate', 'screenshot', 'read', 'analyze']
                if not any(aw in goal_lower for aw in action_words):
                    return True

        # Check for conversational starters
        for starter in conversational_starters:
            if goal_lower.startswith(starter):
                return True

        # Short queries (less than 8 words) without tool needs are likely conversational
        if len(words) < 8:
            return True

        # Medium queries (8-12 words) - check for question patterns
        if len(words) < 12:
            question_words = ['what', 'who', 'why', 'how', 'when', 'where', 'which']
            if any(goal_lower.startswith(qw) for qw in question_words):
                return True

        return False

    def _fast_path_response(self, goal: str) -> dict:
        """Handle simple queries without the full agent loop."""
        print(f"\n{'='*60}")
        print(f"Agent responding (fast-path): {goal}")
        print(f"{'='*60}\n")

        # Build a context-aware system prompt with identity
        identity_prompt = get_identity_prompt()
        system_prompt = f"""{identity_prompt}

You are a helpful AI assistant running locally via Ollama.

About yourself:
- You are an AI agent that can search the web, take screenshots, read files, execute Python code, and more
- You run locally using Ollama with the Llama and Qwen models
- You were created to help users with various tasks through conversation and tool use

Guidelines:
- Be friendly, helpful, and concise
- For greetings, respond warmly but briefly
- For questions about yourself, be informative but humble
- For general knowledge questions, give accurate, helpful answers
- Keep responses short (1-3 sentences for simple queries)
- If asked to do something that requires tools (search, screenshot, files, code), say you can help with that"""

        # Check if this is an identity question - use reasoning model for better system prompt adherence
        goal_lower = goal.lower()
        identity_patterns = [
            'what is your name', "what's your name", 'who are you', 'your name',
            'are you called', 'what should i call you', 'introduce yourself',
            'tell me about yourself', 'what are you', 'are you an ai', 'are you a bot',
            'what model are you', 'are you qwen', 'are you llama', 'are you deepseek'
        ]
        is_identity_question = any(pattern in goal_lower for pattern in identity_patterns)
        task_type = TaskType.REASONING if is_identity_question else TaskType.SIMPLE

        # Use appropriate model based on query type
        response = self.brain.think(
            goal,
            system_prompt=system_prompt,
            use_history=True,  # Enable history for conversational context
            task_type=task_type
        )

        model_used = self.brain.get_last_model_used()
        print(f"[FAST-PATH] Using {model_used}")
        print(f"Response: {response}\n")

        # Log to metacognition
        self.metacognition.start_goal(goal)
        self.metacognition.increment_iteration()
        self.metacognition.log_evaluation(
            tool="fast_path",
            action="direct_response",
            confidence=100,
            success=True,
            progress="Responded directly without tool execution",
            next_step="complete",
            result_summary=response[:500],
            model_used=model_used
        )

        return {
            "goal": goal,
            "completed": True,
            "iterations": 0,
            "fast_path": True,
            "response": response,
            "final_evaluation": {
                "success": True,
                "confidence": 100,
                "progress": response
            },
            "history": []
        }

    def _check_identity_update(self, goal: str) -> Optional[dict]:
        """Check if user is trying to update agent identity.

        Args:
            goal: The user's message/goal

        Returns:
            Response dict if identity was updated, None otherwise
        """
        # Check for name change
        new_name = detect_name_change(goal)
        if new_name:
            self.identity = update_name(new_name)
            response = f"Got it! I'll remember that. You can call me {new_name} from now on."
            print(f"\n[IDENTITY] Name updated to: {new_name}")
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "identity_update": True,
                "response": response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": response
                },
                "history": []
            }

        # Check for personality change
        new_personality = detect_personality_change(goal)
        if new_personality:
            # Append to existing personality or replace
            current = self.identity.get("personality", "")
            updated = f"{current}, {new_personality}" if current else new_personality
            self.identity = update_personality(updated)
            response = f"I'll try to be more {new_personality}. Thanks for the feedback!"
            print(f"\n[IDENTITY] Personality updated to: {updated}")
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "identity_update": True,
                "response": response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": response
                },
                "history": []
            }

        return None

    def run(self, goal: str, context: Optional[dict] = None, use_fastpath: Optional[bool] = None) -> dict:
        """Run the agent loop to achieve a goal.

        Args:
            goal: The goal to achieve
            context: Optional context dictionary
            use_fastpath: Override fast-path behavior (None uses self.use_fastpath)
        """
        # Record user activity for NeuroDream idle detection
        if hasattr(self, 'neurodream') and self.neurodream:
            self.neurodream.record_activity()

        # Check for identity updates first
        identity_response = self._check_identity_update(goal)
        if identity_response:
            return identity_response

        # Check for FluxMind commands - handle directly without LLM
        fluxmind_response = self._handle_fluxmind_command(goal)
        if fluxmind_response:
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "fluxmind_direct": True,
                "response": fluxmind_response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": fluxmind_response
                },
                "history": []
            }

        # Check for Git commands - handle directly without LLM hallucination
        git_response = self._handle_git_command(goal)
        if git_response:
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "git_direct": True,
                "response": git_response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": git_response
                },
                "history": []
            }

        # Check for Inner Monologue commands - handle directly
        monologue_response = self._handle_monologue_command(goal)
        if monologue_response:
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "monologue_direct": True,
                "response": monologue_response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": monologue_response
                },
                "history": []
            }

        # Check for Knowledge Graph commands - handle directly
        kg_response = self._handle_knowledge_graph_command(goal)
        if kg_response:
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "kg_direct": True,
                "response": kg_response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": kg_response
                },
                "history": []
            }

        # Check for Guardian commands - handle directly
        guardian_response = self._handle_guardian_command(goal)
        if guardian_response:
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "guardian_direct": True,
                "response": guardian_response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": guardian_response
                },
                "history": []
            }

        # Check for NeuroDream commands - handle directly
        neurodream_response = self._handle_neurodream_command(goal)
        if neurodream_response:
            return {
                "goal": goal,
                "completed": True,
                "iterations": 0,
                "fast_path": True,
                "neurodream_direct": True,
                "response": neurodream_response,
                "final_evaluation": {
                    "success": True,
                    "confidence": 100,
                    "progress": neurodream_response
                },
                "history": []
            }

        # === METACOGNITIVE GUARDIAN PRE-CHECK ===
        # Assess risk before processing (works with or without FluxMind)
        if hasattr(self, 'guardian') and self.guardian:
            prediction = self.guardian.assess_risk(
                task=goal,
                tool=None,  # No tool selected yet
                context=context or {}
            )

            if prediction:
                self._pending_prediction = prediction

                # If very high risk, intervene before even starting
                if prediction.probability >= self.guardian.config.intervention_threshold:
                    intervention = self.guardian.execute_intervention(prediction)

                    if intervention["should_abort"]:
                        return {
                            "goal": goal,
                            "completed": False,
                            "iterations": 0,
                            "guardian_abort": True,
                            "response": intervention["message"],
                            "final_evaluation": {
                                "success": False,
                                "confidence": int((1 - prediction.probability) * 100),
                                "progress": f"Guardian intervention: {prediction.failure_type.value}"
                            },
                            "history": []
                        }

        # Check for fast-path eligibility
        fastpath_enabled = use_fastpath if use_fastpath is not None else self.use_fastpath
        if fastpath_enabled and self._is_simple_query(goal):
            return self._fast_path_response(goal)

        self.state = AgentState(goal=goal)
        context = context or {}
        # Clear screenshot path for new task
        self.brain._last_screenshot_path = None
        # Start metacognition tracking for this goal
        self.metacognition.start_goal(goal)

        # Start inner monologue session
        self.monologue.start_session()
        self.monologue.think("perceive", f"Received: '{goal[:80]}{'...' if len(goal) > 80 else ''}'")

        # Check for user emotional state if EvoEmo available
        if "evoemo" in self.tools:
            try:
                evoemo = self.tools["evoemo"]
                mood_result = evoemo.analyze_text(goal)
                if mood_result.get("success"):
                    user_mood = mood_result.get("emotion", "calm")
                    confidence = mood_result.get("confidence", 50)
                    if user_mood in ["frustrated", "stressed"]:
                        self.monologue.think("perceive", f"User seems {user_mood} ({confidence}% confidence). Being careful and clear.")
            except Exception:
                pass

        print(f"\n{'='*60}")
        print(f"Agent starting with goal: {goal}")
        print(f"{'='*60}\n")

        while not self.state.completed and self.state.iteration < self.max_iterations:
            self.state.iteration += 1
            self.metacognition.increment_iteration()
            print(f"\n--- Iteration {self.state.iteration} ---")

            # Phase 1: OBSERVE
            self._observe(context)

            # Phase 2: PLAN
            self._plan()

            # Phase 3: ACT
            self._act()

            # Phase 4: EVALUATE
            self._evaluate()

            # Phase 5: REMEMBER
            self._remember()

        return self._get_final_result()

    def _observe(self, context: dict) -> None:
        """Phase 1: Observe the current state and context."""
        self.state.phase = AgentPhase.OBSERVE
        print(f"[OBSERVE] Analyzing current context...")

        # Gather context including relevant memories
        relevant_memories = self.memory.recall(self.state.goal, n_results=3)

        # Emit recall thought if memories found
        if relevant_memories:
            memory_preview = ", ".join([m["content"][:30] for m in relevant_memories[:2]])
            self.monologue.think("recall", f"Found {len(relevant_memories)} relevant memories: {memory_preview}...")

        observation_context = {
            "goal": self.state.goal,
            "iteration": self.state.iteration,
            "relevant_memories": [m["content"] for m in relevant_memories],
            "last_action": self.state.last_action,
            "last_result": self.state.last_result,
            **context
        }

        self.state.observations = self.brain.observe(observation_context)
        print(f"Observations: {self.state.observations[:200]}...")

    def _plan(self) -> None:
        """Phase 2: Create or update the plan."""
        self.state.phase = AgentPhase.PLAN
        print(f"[PLAN] Creating action plan...")

        # Include summarize as an available tool
        available_tools = list(self.tools.keys()) + ["summarize"]
        self.state.current_plan = self.brain.plan(
            self.state.goal,
            self.state.observations,
            available_tools
        )

        # Emit reasoning thought about the plan
        plan_preview = self.state.current_plan[:100].replace('\n', ' ')
        self.monologue.think("reason", f"Planning approach: {plan_preview}...")

        print(f"Plan: {self.state.current_plan[:200]}...")

    def _detect_custom_tool(self, goal: str) -> Optional[tuple[str, str]]:
        """Detect if the goal matches a custom tool.

        Args:
            goal: The user's goal

        Returns:
            (tool_name, action) tuple if a custom tool matches, None otherwise
        """
        goal_lower = goal.lower()

        # Marketplace keywords have highest priority - skip custom tool detection
        marketplace_keywords = [
            'publish', 'marketplace', 'browse plugins', 'install plugin',
            'uninstall plugin', 'my plugins', 'installed plugins', 'rate plugin',
            'update plugin', 'plugin marketplace', 'search plugins', 'download tool',
            'share tool'
        ]
        if any(kw in goal_lower for kw in marketplace_keywords):
            return None  # Let the agent route to marketplace instead

        # Check each custom tool's keywords
        best_match = None
        best_match_score = 0

        for keyword, tool_name in self.custom_tool_keywords.items():
            if keyword in goal_lower:
                # Score based on keyword length (longer = more specific)
                score = len(keyword)
                if score > best_match_score:
                    best_match_score = score
                    best_match = tool_name

        if best_match:
            # Return the tool name and the goal as the action
            return (best_match, goal)

        return None

    def _detect_marketplace_action(self, goal: str) -> Optional[tuple[str, str]]:
        """Detect if the goal requires the marketplace tool.

        Args:
            goal: The user's goal

        Returns:
            (tool_name, action) tuple if marketplace matches, None otherwise
        """
        import re
        goal_lower = goal.lower()

        marketplace_keywords = [
            'publish', 'marketplace', 'browse plugin', 'install plugin',
            'uninstall plugin', 'my plugins', 'rate plugin', 'update plugin',
            'search plugin', 'plugin marketplace', 'download tool', 'share tool'
        ]

        # Check if any marketplace keyword is present
        if not any(kw in goal_lower for kw in marketplace_keywords):
            return None

        # Route to marketplace with the full goal as action
        print(f"[MARKETPLACE] Detected marketplace action in: {goal[:50]}...")
        return ("marketplace", goal)

    def _act(self) -> None:
        """Phase 3: Execute an action."""
        self.state.phase = AgentPhase.ACT
        print(f"[ACT] Deciding and executing action...")

        # PRIORITY 1: Check for marketplace actions FIRST (highest priority)
        marketplace_match = self._detect_marketplace_action(self.state.goal)
        if marketplace_match and self.state.iteration == 1:
            tool_name, action = marketplace_match
            action_decision = {
                "tool": tool_name,
                "action": action,
                "reasoning": "Using marketplace tool based on goal keywords"
            }
        # PRIORITY 2: Check for FluxMind actions
        elif self._detect_fluxmind_action(self.state.goal) and self.state.iteration == 1:
            fluxmind_match = self._detect_fluxmind_action(self.state.goal)
            tool_name, action = fluxmind_match
            action_decision = {
                "tool": tool_name,
                "action": action,
                "reasoning": "Using FluxMind calibrated reasoning based on goal keywords"
            }
        # PRIORITY 3: Check for Git actions
        elif self._detect_git_action(self.state.goal) and self.state.iteration == 1:
            git_match = self._detect_git_action(self.state.goal)
            tool_name, action = git_match
            action_decision = {
                "tool": tool_name,
                "action": action,
                "reasoning": "Using Git tool based on goal keywords"
            }
        # PRIORITY 4: Check if a custom tool should be used
        elif self._detect_custom_tool(self.state.goal) and self.state.iteration == 1:
            custom_tool_match = self._detect_custom_tool(self.state.goal)
            tool_name, action = custom_tool_match
            print(f"[CUSTOM TOOL] Detected custom tool: {tool_name}")
            action_decision = {
                "tool": tool_name,
                "action": action,
                "reasoning": f"Using custom tool {tool_name} based on goal keywords"
            }
        # PRIORITY 5: Check for Inner Monologue actions
        elif self._detect_monologue_action(self.state.goal) and self.state.iteration == 1:
            monologue_match = self._detect_monologue_action(self.state.goal)
            tool_name, action = monologue_match
            action_decision = {
                "tool": tool_name,
                "action": action,
                "reasoning": "Using inner monologue tool based on goal keywords"
            }
        else:
            # Include summarize as an available tool
            available_tools = list(self.tools.keys()) + ["summarize"]
            action_decision = self.brain.decide_action(
                self.state.current_plan,
                available_tools
            )

        self.state.last_action = action_decision
        tool_name = action_decision.get('tool', 'unknown')
        action_str = action_decision.get('action', '')[:50]

        # Emit decision thought
        self.monologue.think(
            "decide",
            f"Selected tool: {tool_name}",
            confidence=85,  # Default high confidence for tool selection
            metadata={"tool": tool_name, "action_preview": action_str}
        )

        print(f"Action: {tool_name} - {action_decision.get('action')}")

        # Emit execute thought
        self.monologue.think("execute", f"Running {tool_name}...")

        # Execute the action
        self.state.last_result = self._execute_action(action_decision)
        print(f"Result: {str(self.state.last_result)[:200]}...")

    def _evaluate(self) -> None:
        """Phase 4: Evaluate the result."""
        self.state.phase = AgentPhase.EVALUATE
        print(f"[EVALUATE] Assessing result...")

        # Get tool info
        last_tool = self.state.last_action.get('tool', '').lower() if self.state.last_action else ''
        result_success = self.state.last_result.get('success', False) if self.state.last_result else False

        # SPECIAL HANDLING: FluxMind results bypass LLM evaluation to prevent hallucination
        if last_tool == 'fluxmind' and result_success:
            result = self.state.last_result
            # Build evaluation directly from tool results (no LLM interpretation)
            if 'next_state' in result:
                progress = f"FluxMind step result: next_state={result['next_state']}, confidence={result['confidence']:.2%}"
            elif 'confidence' in result:
                progress = f"FluxMind confidence: {result['confidence']:.2%} - {result.get('interpretation', '')}"
            elif 'trajectory' in result:
                progress = f"FluxMind program executed: {len(result['trajectory'])} states, mean_confidence={result.get('mean_confidence', 0):.2%}"
            else:
                progress = f"FluxMind result: {result.get('formatted', str(result))}"

            self.state.evaluation = {
                'success': True,
                'confidence': 100,
                'progress': progress,
                'next': 'complete'
            }
            self.state.completed = True
            print(f"Success: True")
            print(f"Confidence: 100%")
            print(f"Progress: {progress}")
            print("[FLUXMIND] Direct result (no LLM interpretation)")
            return

        # SPECIAL HANDLING: Git tool results bypass LLM evaluation to prevent hallucination
        if last_tool == 'git' and result_success:
            result = self.state.last_result
            # Use the 'output' field directly - it contains formatted git output
            git_output = result.get('output', str(result))

            self.state.evaluation = {
                'success': True,
                'confidence': 100,
                'progress': git_output,
                'next': 'complete'
            }
            self.state.completed = True
            print(f"Success: True")
            print(f"Confidence: 100%")
            print(f"Git Output:\n{git_output[:500]}...")
            print("[GIT] Direct result (no LLM interpretation)")
            return

        action_str = f"{self.state.last_action.get('tool')}: {self.state.last_action.get('action')}"
        result_str = str(self.state.last_result)

        self.state.evaluation = self.brain.evaluate(
            action_str,
            result_str,
            self.state.goal
        )

        eval_success = self.state.evaluation.get('success', False)
        eval_confidence = self.state.evaluation.get('confidence', 0)
        eval_progress = self.state.evaluation.get('progress', '')[:100]

        print(f"Success: {eval_success}")
        print(f"Confidence: {eval_confidence}%")
        print(f"Progress: {self.state.evaluation.get('progress')}")

        # Emit reflection thought
        if eval_success:
            self.monologue.think(
                "reflect",
                f"Result looks good: {eval_progress}...",
                confidence=eval_confidence
            )
            # Check for eureka moment (high confidence success)
            if eval_confidence >= 90:
                self.monologue.think("eureka", "High confidence result achieved!")
        else:
            # Low confidence or failure
            if eval_confidence < 60:
                self.monologue.think(
                    "uncertain",
                    f"Low confidence ({eval_confidence}%). May need to try different approach.",
                    confidence=eval_confidence
                )
            else:
                self.monologue.think(
                    "reflect",
                    f"Result needs work: {eval_progress}...",
                    confidence=eval_confidence
                )

        # Get model used for this evaluation
        model_used = self.brain.get_last_model_used()
        print(f"Model: {model_used}")

        # Log to metacognition system
        self.metacognition.log_evaluation(
            tool=self.state.last_action.get('tool'),
            action=self.state.last_action.get('action'),
            confidence=self.state.evaluation.get('confidence', 0),
            success=self.state.evaluation.get('success', False),
            progress=self.state.evaluation.get('progress'),
            next_step=self.state.evaluation.get('next'),
            result_summary=str(self.state.last_result)[:500],
            model_used=model_used
        )

        # Check for marketplace read-only actions - complete immediately on success
        last_tool = self.state.last_action.get('tool', '').lower() if self.state.last_action else ''
        last_action = self.state.last_action.get('action', '').lower() if self.state.last_action else ''
        result_success = self.state.last_result.get('success', False) if self.state.last_result else False

        if last_tool == 'marketplace' and result_success:
            read_only_actions = ['browse', 'list', 'my_plugins', 'my plugins', 'search', 'info', 'installed']
            if any(action_word in last_action for action_word in read_only_actions):
                self.state.completed = True
                print("Marketplace query completed successfully!")
                return

        # Check if goal is achieved
        if self.state.evaluation.get("success") and "complete" in self.state.evaluation.get("next", "").lower():
            # Override for combined screenshot+vision tasks
            is_screenshot_and_vision = getattr(self.brain, '_current_goal_is_screenshot_and_vision', False)

            if is_screenshot_and_vision and last_tool == 'screenshot':
                # Don't mark as complete - still need to do vision analysis
                print("Screenshot done, continuing to vision analysis...")
                self.state.evaluation['next'] = 'continue'
            else:
                self.state.completed = True
                print("Goal achieved!")

    def _remember(self) -> None:
        """Phase 5: Store important information in memory."""
        self.state.phase = AgentPhase.REMEMBER
        print(f"[REMEMBER] Storing experience...")

        # Record this iteration in history
        episode = {
            "iteration": self.state.iteration,
            "goal": self.state.goal,
            "action": self.state.last_action,
            "result": self.state.last_result,
            "evaluation": self.state.evaluation,
            "timestamp": datetime.now().isoformat()
        }
        self.state.history.append(episode)

        # Store significant learnings in long-term memory
        if self.state.evaluation:
            memory_content = self.brain.summarize_for_memory({
                "goal": self.state.goal,
                "actions": [self.state.last_action],
                "outcome": self.state.evaluation.get("progress", ""),
                "lessons": self.state.evaluation.get("next", "")
            })

            self.memory.remember(
                memory_content,
                memory_type="episodic",
                metadata={
                    "goal": self.state.goal,
                    "success": self.state.evaluation.get("success", False),
                    "iteration": self.state.iteration
                }
            )
            print(f"Stored memory: {memory_content[:100]}...")

    def _execute_action(self, action_decision: dict) -> dict:
        """Execute an action using the appropriate tool."""
        tool_name = action_decision.get("tool", "").lower()
        action = action_decision.get("action", "")

        # Handle summarize tool specially (it's not in self.tools)
        if tool_name == "summarize":
            return self._execute_summarize()

        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()) + ["summarize"]
            }

        tool = self.tools[tool_name]

        # Parse the action into tool method and arguments
        try:
            result = self._parse_and_execute_tool_action(tool, tool_name, action)
            # Store successful search results for later summarization
            if tool_name == "web_search" and result.get("success"):
                self._store_search_results(result)
            # Store screenshot path for combined screenshot+vision tasks
            if tool_name == "screenshot" and result.get("success"):
                self.brain._last_screenshot_path = result.get("path")
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_summarize(self) -> dict:
        """Execute the summarize action on gathered content."""
        if not self.state.gathered_content:
            return {
                "success": False,
                "error": "No content to summarize. Run a web search first."
            }

        summary = self.brain.summarize(self.state.gathered_content, self.state.goal)
        return {
            "success": True,
            "summary": summary,
            "message": "Content summarized successfully"
        }

    def _store_search_results(self, result: dict) -> None:
        """Store search results for later summarization."""
        results = result.get("results", [])
        if results:
            content_parts = []
            for r in results[:5]:  # Limit to top 5 results
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                if title or snippet:
                    content_parts.append(f"- {title}: {snippet}")

            new_content = "\n".join(content_parts)
            # Append to existing content
            if self.state.gathered_content:
                self.state.gathered_content += "\n\n" + new_content
            else:
                self.state.gathered_content = new_content

    def _parse_and_execute_tool_action(self, tool: Any, tool_name: str, action: str) -> dict:
        """Parse action string and execute on tool."""
        action_lower = action.lower()

        if tool_name == "filesystem":
            if "read" in action_lower:
                # Extract file path from action
                path = self._extract_path(action)
                return tool.read_file(path) if path else {"success": False, "error": "No path specified"}
            elif "write" in action_lower:
                return {"success": False, "error": "Write action needs path and content"}
            elif "list" in action_lower:
                path = self._extract_path(action) or "."
                return tool.list_directory(path)
            elif "search" in action_lower:
                pattern = self._extract_pattern(action)
                return tool.search_files(pattern) if pattern else {"success": False, "error": "No pattern"}
            else:
                return {"success": False, "error": f"Unknown filesystem action: {action}"}

        elif tool_name == "web_search":
            if "news" in action_lower:
                query = self._extract_query(action)
                return tool.news(query) if query else {"success": False, "error": "No query"}
            elif "search" in action_lower or "find" in action_lower:
                query = self._extract_query(action)
                return tool.search(query) if query else {"success": False, "error": "No query"}
            elif "answer" in action_lower:
                query = self._extract_query(action)
                return tool.instant_answer(query) if query else {"success": False, "error": "No query"}
            else:
                # Default to search
                return tool.search(action)

        elif tool_name == "code_executor":
            # Extract Python code from the action
            code = self._extract_code(action)
            if code:
                return tool.execute(code)
            else:
                return {"success": False, "error": "No code provided"}

        elif tool_name == "screenshot":
            # Handle screenshot actions
            if "region" in action_lower:
                # Extract region coordinates if specified
                import re
                numbers = re.findall(r'\d+', action)
                if len(numbers) >= 4:
                    x, y, w, h = int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])
                    return tool.take_screenshot_region(x, y, w, h)
                return {"success": False, "error": "Region needs x, y, width, height"}
            elif "monitor" in action_lower or "list" in action_lower:
                return tool.list_monitors()
            else:
                # Default: capture full screen (primary monitor)
                return tool.take_screenshot(monitor=1)

        elif tool_name == "vision":
            # Handle vision/image analysis actions
            image_path = self._extract_path(action)
            if not image_path:
                # Check if there's a recent screenshot
                screenshots_dir = Path("screenshots")
                if screenshots_dir.exists():
                    screenshots = sorted(screenshots_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if screenshots:
                        image_path = str(screenshots[0])

            if not image_path:
                return {"success": False, "error": "No image path found. Take a screenshot first or specify an image path."}

            # Determine question from action
            if "read" in action_lower or "text" in action_lower:
                return tool.read_text(image_path)
            elif "screen" in action_lower:
                return tool.describe_screen(image_path)
            else:
                return tool.analyze_image(image_path)

        elif tool_name == "pdf_reader":
            # Handle PDF reading actions
            pdf_path = self._extract_path(action)
            if not pdf_path:
                # Try extracting from original goal
                pdf_path = self._extract_path(self.state.goal)
            if not pdf_path:
                return {"success": False, "error": "No PDF path specified"}

            if "info" in action_lower or "metadata" in action_lower:
                return tool.info(pdf_path)
            elif "search" in action_lower or "find" in action_lower:
                # Extract search query
                query = self._extract_query(action)
                if not query:
                    return {"success": False, "error": "No search query provided"}
                return tool.search(pdf_path, query)
            elif "extract" in action_lower:
                # Extract pages specification from action first, then from goal
                pages = self._extract_pages(action)
                if not pages:
                    pages = self._extract_pages(self.state.goal)
                pages = pages or "all"
                return tool.extract_text(pdf_path, pages)
            else:
                # Default: read PDF
                # Extract pages specification from action first, then from goal
                pages = self._extract_pages(action)
                if not pages:
                    pages = self._extract_pages(self.state.goal)
                pages = pages or "all"
                summarize = "summar" in action_lower
                return tool.read(pdf_path, pages, summarize)

        elif tool_name == "clipboard":
            # Handle clipboard actions
            if "analyze" in action_lower or "detect" in action_lower or "type" in action_lower:
                return tool.analyze()
            elif "write" in action_lower or "copy" in action_lower:
                # Extract text to copy
                text = self._extract_clipboard_text(action)
                if not text:
                    # Try from original goal
                    text = self._extract_clipboard_text(self.state.goal)
                if text:
                    return tool.write(text)
                return {"success": False, "error": "No text specified to copy"}
            else:
                # Default: read clipboard (paste, read, what's in clipboard)
                return tool.read()

        elif tool_name == "arxiv_search":
            # Handle arXiv search actions
            if "download" in action_lower:
                arxiv_id = self._extract_arxiv_id(action)
                if not arxiv_id:
                    return {"success": False, "error": "No arXiv ID specified for download"}
                return tool.download_pdf(arxiv_id)
            elif "abstract" in action_lower:
                arxiv_id = self._extract_arxiv_id(action)
                if not arxiv_id:
                    return {"success": False, "error": "No arXiv ID specified"}
                return tool.get_abstract(arxiv_id)
            elif "paper" in action_lower and ("get" in action_lower or "details" in action_lower):
                arxiv_id = self._extract_arxiv_id(action)
                if not arxiv_id:
                    return {"success": False, "error": "No arXiv ID specified"}
                return tool.get_paper(arxiv_id)
            elif "author" in action_lower:
                author = self._extract_query(action)
                if not author:
                    return {"success": False, "error": "No author name specified"}
                return tool.search_by_author(author)
            elif "category" in action_lower or "recent" in action_lower:
                category = self._extract_arxiv_category(action)
                query = self._extract_query(action) if "category" in action_lower else None
                if not category:
                    return {"success": False, "error": "No category specified (e.g., cs.AI, physics.quant-ph)"}
                if "recent" in action_lower:
                    return tool.get_recent(category)
                return tool.search_by_category(category, query)
            elif "summarize" in action_lower or "summary" in action_lower or "compare" in action_lower:
                # AI-powered research summary
                query = self._extract_query(action)
                if not query:
                    query = action.replace("summarize", "").replace("summary", "").replace("compare", "").strip()
                if not query:
                    return {"success": False, "error": "No query specified for research summary"}
                return tool.summarize_search(query)
            else:
                # Default: search by query
                query = self._extract_query(action)
                if not query:
                    query = action  # Use the whole action as query
                return tool.search(query)

        elif tool_name == "browser":
            # Handle browser actions
            if "open" in action_lower or "go to" in action_lower or "navigate" in action_lower or "visit" in action_lower:
                url = self._extract_url(action)
                if not url:
                    return {"success": False, "error": "No URL specified"}
                return tool.open(url)
            elif "screenshot" in action_lower or "capture" in action_lower:
                return tool.screenshot()
            elif "text" in action_lower or "content" in action_lower:
                return tool.get_text()
            elif "links" in action_lower:
                return tool.get_links()
            elif "click" in action_lower:
                selector = self._extract_selector(action)
                if not selector:
                    return {"success": False, "error": "No selector specified for click"}
                return tool.click(selector)
            elif "fill" in action_lower or "type" in action_lower:
                # Extract selector and text
                parts = action.split(" ", 2)
                if len(parts) >= 3:
                    return tool.fill(parts[1], parts[2])
                return {"success": False, "error": "Fill requires selector and text"}
            elif "google" in action_lower or "search" in action_lower:
                query = self._extract_query(action)
                if not query:
                    return {"success": False, "error": "No search query specified"}
                return tool.search_google(query)
            elif "close" in action_lower or "quit" in action_lower:
                return tool.close()
            else:
                # Default: try to open as URL
                url = self._extract_url(action)
                if url:
                    return tool.open(url)
                return {"success": False, "error": f"Unknown browser action: {action}"}

        elif tool_name == "system_control":
            # Handle system control actions
            if "get" in action_lower and "volume" in action_lower:
                return tool.get_volume()
            elif "set" in action_lower and "volume" in action_lower:
                level = self._extract_number(action)
                if level is None:
                    return {"success": False, "error": "No volume level specified"}
                return tool.set_volume(level)
            elif "get" in action_lower and "brightness" in action_lower:
                return tool.get_brightness()
            elif "set" in action_lower and "brightness" in action_lower:
                level = self._extract_number(action)
                if level is None:
                    return {"success": False, "error": "No brightness level specified"}
                return tool.set_brightness(level)
            elif "open" in action_lower or "launch" in action_lower or "start" in action_lower:
                app_name = self._extract_app_name(action)
                if not app_name:
                    return {"success": False, "error": "No app name specified"}
                return tool.open_app(app_name)
            elif "system" in action_lower or "info" in action_lower or "cpu" in action_lower or "ram" in action_lower or "memory" in action_lower:
                return tool.get_system_info()
            elif "lock" in action_lower:
                return tool.lock_screen()
            else:
                # Try using the execute method for natural language
                return tool.execute(action)

        elif tool_name == "notifications":
            # Handle notification actions
            if "list" in action_lower:
                return tool.list_tasks()
            elif "clear" in action_lower:
                return tool.clear_all()
            elif "remove" in action_lower or "delete" in action_lower or "cancel" in action_lower:
                task_id = self._extract_task_id(action)
                if not task_id:
                    return {"success": False, "error": "No task ID specified for removal"}
                return tool.remove_task(task_id)
            elif "condition" in action_lower or "alert when" in action_lower or "notify when" in action_lower:
                # Conditional alert: "alert when CPU exceeds 80%"
                message, condition_type, threshold = self._extract_condition_params(action)
                if not condition_type or threshold is None:
                    # Try extracting from goal
                    message, condition_type, threshold = self._extract_condition_params(self.state.goal)
                if not condition_type:
                    return {"success": False, "error": "Could not parse condition type (cpu/ram/disk)"}
                if threshold is None:
                    return {"success": False, "error": "Could not parse threshold value"}
                return tool.add_condition(message or f"High {condition_type} alert", condition_type, threshold)
            elif "schedule" in action_lower or "every day" in action_lower or "daily" in action_lower or "weekday" in action_lower or "weekly" in action_lower:
                # Scheduled notification: "notify every day at 9 AM for standup"
                message, time_of_day, repeat = self._extract_scheduled_params(action)
                if not time_of_day:
                    # Try extracting from goal
                    message, time_of_day, repeat = self._extract_scheduled_params(self.state.goal)
                if not time_of_day:
                    return {"success": False, "error": "Could not parse time of day (e.g., 9:00 AM)"}
                return tool.add_scheduled(message or "Scheduled notification", time_of_day, repeat)
            else:
                # Default: reminder - "remind me in 30 minutes to take a break"
                message, time_str = self._extract_reminder_params(action)
                if not time_str:
                    # Try extracting from goal
                    message, time_str = self._extract_reminder_params(self.state.goal)
                if not time_str:
                    return {"success": False, "error": "Could not parse time (e.g., 'in 30 minutes')"}
                if not message:
                    message = "Reminder"
                return tool.add_reminder(message, time_str)

        elif tool_name == "tool_builder":
            # Handle tool builder actions
            if "list" in action_lower:
                return tool.list_custom_tools()
            elif "test" in action_lower:
                return tool.execute(action)
            elif "enable" in action_lower:
                return tool.execute(action)
            elif "disable" in action_lower:
                return tool.execute(action)
            elif "rollback" in action_lower or "delete" in action_lower or "remove" in action_lower:
                return tool.execute(action)
            elif "create" in action_lower:
                # Generate tool spec from user request using LLM
                tool_spec = self._generate_tool_spec(self.state.goal)
                if not tool_spec:
                    return {
                        "success": False,
                        "error": "Failed to generate tool specification from request."
                    }
                if "error" in tool_spec:
                    return {"success": False, "error": tool_spec["error"]}

                # Create the tool with generated spec
                return tool.create_tool(
                    name=tool_spec["name"],
                    description=tool_spec["description"],
                    functions_spec=tool_spec["functions_spec"]
                )
            else:
                return tool.execute(action)

        elif tool_name == "marketplace":
            # Handle marketplace actions
            return tool.execute(action)

        elif tool_name == "fluxmind":
            # Handle FluxMind calibrated reasoning actions
            return self._execute_fluxmind_action(tool, action)

        elif tool_name == "regex_builder":
            # Handle regex builder actions
            return tool.execute(action)

        elif tool_name == "git":
            # Handle git operations
            return tool.execute(action)

        elif tool_name == "personaplex":
            # Handle PersonaPlex real-time voice actions (Tool #17)
            if "status" in action_lower or "running" in action_lower or "check" in action_lower:
                return tool.status()
            elif "start" in action_lower or "launch" in action_lower:
                # Extract optional voice from action
                voice = None
                for v in tool.VOICES.keys():
                    if v.lower() in action_lower:
                        voice = v
                        break
                return tool.start_server(voice=voice)
            elif "stop" in action_lower or "shutdown" in action_lower or "kill" in action_lower:
                return tool.stop_server()
            elif "list" in action_lower and "voice" in action_lower:
                return tool.list_voices()
            elif "set" in action_lower and "voice" in action_lower:
                # Extract voice ID from action
                for v in tool.VOICES.keys():
                    if v.lower() in action_lower:
                        return tool.set_voice(v)
                return {"success": False, "error": "No voice ID found. Use list_voices to see options."}
            elif "set" in action_lower and "persona" in action_lower:
                # Extract persona text (everything after "persona")
                import re
                match = re.search(r'persona[:\s]+["\']?(.+?)["\']?$', action, re.I)
                if match:
                    return tool.set_persona(match.group(1).strip())
                return {"success": False, "error": "No persona text provided."}
            elif "reset" in action_lower or "default" in action_lower:
                return tool.reset_to_defaults()
            else:
                return tool.execute(action)

        elif tool_name == "clawdbot":
            # Handle Clawdbot messaging (Tool #19)
            import re
            if "status" in action_lower or "check" in action_lower:
                return tool.get_status()
            elif "start" in action_lower and "gateway" in action_lower:
                return tool.start_gateway()
            elif "stop" in action_lower and "gateway" in action_lower:
                return tool.stop_gateway()
            elif "list" in action_lower and "channel" in action_lower:
                return tool.list_channels()
            elif "send" in action_lower or "message" in action_lower:
                # Extract recipient and message
                # Pattern: send "message" to +1234567890 via whatsapp
                to_match = re.search(r'to\s+([+\d]+|\w+@\w+)', action, re.I)
                msg_match = re.search(r'["\']([^"\']+)["\']|message[:\s]+(.+?)(?:\s+to|\s+via|$)', action, re.I)
                channel_match = re.search(r'via\s+(\w+)', action, re.I)

                to = to_match.group(1) if to_match else None
                message = (msg_match.group(1) or msg_match.group(2)).strip() if msg_match else None
                channel = channel_match.group(1).lower() if channel_match else "whatsapp"

                if not to:
                    return {"success": False, "error": "No recipient specified. Use: send 'message' to +1234567890"}
                if not message:
                    return {"success": False, "error": "No message specified. Use: send 'message' to +1234567890"}

                return tool.send_message(to, message, channel)
            else:
                return tool.execute(action, **{})

        # Handle custom tools - they all have an execute() method
        if tool_name in self.custom_tool_keywords.values() or hasattr(tool, 'execute'):
            return tool.execute(action)

        return {"success": False, "error": f"Cannot parse action for {tool_name}"}

    def _extract_path(self, action: str) -> Optional[str]:
        """Extract file path from action string."""
        import re
        # Look for quoted strings first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Look for Windows paths (C:/... or C:\...)
        win_paths = re.findall(r'[A-Za-z]:[/\\][\w./\\-]+', action)
        if win_paths:
            return win_paths[0]
        # Look for Unix paths or relative paths
        unix_paths = re.findall(r'(?:^|[ ])([./][\w./\\-]+)', action)
        if unix_paths:
            return unix_paths[0].strip()
        # Look for path patterns with extensions
        paths = re.findall(r'[\w./\\-]+\.\w+', action)
        return paths[0] if paths else None

    def _extract_pattern(self, action: str) -> Optional[str]:
        """Extract search pattern from action string."""
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        patterns = re.findall(r'\*[\w.*]+|\w+\.\w+', action)
        return patterns[0] if patterns else "*.py"

    def _extract_query(self, action: str) -> Optional[str]:
        """Extract search query from action string."""
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Remove common command words
        query = re.sub(r'\b(search|find|look|for|up|query|news|about)\b', '', action, flags=re.I)
        return query.strip() or None

    def _extract_code(self, action: str) -> Optional[str]:
        """Extract Python code from action string."""
        import re

        # Look for code in triple backticks
        triple_backtick = re.search(r'```(?:python)?\s*(.*?)```', action, re.DOTALL)
        if triple_backtick:
            return triple_backtick.group(1).strip()

        # Look for code in single backticks
        single_backtick = re.search(r'`([^`]+)`', action)
        if single_backtick:
            return single_backtick.group(1).strip()

        # Look for code after common prefixes
        prefixes = ['run:', 'execute:', 'code:', 'python:']
        action_lower = action.lower()
        for prefix in prefixes:
            if prefix in action_lower:
                idx = action_lower.index(prefix) + len(prefix)
                return action[idx:].strip()

        # If it looks like code (has print, =, import, def, etc.), use as-is
        code_indicators = ['print(', 'import ', 'def ', 'class ', '=', 'for ', 'while ', 'if ']
        if any(ind in action for ind in code_indicators):
            return action.strip()

        # Otherwise return the whole action as potential code
        return action.strip() if action.strip() else None

    def _extract_pages(self, action: str) -> Optional[str]:
        """Extract page specification from action string."""
        import re
        action_lower = action.lower()

        # Check for "first page" or "last page" first
        if 'first page' in action_lower or 'first' in action_lower and 'page' in action_lower:
            return "first"
        if 'last page' in action_lower or 'last' in action_lower and 'page' in action_lower:
            return "last"
        if 'all page' in action_lower:
            return "all"

        # Look for page patterns like "pages 1-3", "page 5", "pages 1,3,5"
        # Pattern: "pages 1-3" or "page 1-3" or "pages 1, 3, 5"
        page_patterns = [
            r'pages?\s+([0-9]+(?:\s*[-,]\s*[0-9]+)*)',  # "pages 1-3" or "pages 1,3,5"
            r'page\s+([0-9]+(?:\s*-\s*[0-9]+)?)',       # "page 5" or "page 1-3"
        ]
        for pattern in page_patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                # Clean up any extra spaces in the result
                result = re.sub(r'\s+', '', match.group(1))
                return result
        return None

    def _extract_clipboard_text(self, action: str) -> Optional[str]:
        """Extract text to copy to clipboard from action string."""
        import re
        # Look for quoted text first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        # Look for text after common patterns
        patterns = [
            r'copy\s+(?:this\s+)?(?:to\s+clipboard)?[:\s]+(.+)',
            r'write\s+(?:to\s+clipboard)?[:\s]+(.+)',
            r'clipboard[:\s]+(.+)',
            r'copy[:\s]+(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Remove common suffixes
                text = re.sub(r'\s+to\s+clipboard.*$', '', text, flags=re.IGNORECASE)
                return text.strip('"\'')
        return None

    def _extract_arxiv_id(self, action: str) -> Optional[str]:
        """Extract arXiv ID from action string."""
        import re
        # arXiv ID patterns: 2301.00001, 2301.00001v1, cs.AI/0001001
        patterns = [
            r'(\d{4}\.\d{4,5}(?:v\d+)?)',  # New format: 2301.00001 or 2301.00001v1
            r'([a-z-]+(?:\.[A-Z]{2})?/\d{7})',  # Old format: cs.AI/0001001
        ]
        for pattern in patterns:
            match = re.search(pattern, action)
            if match:
                return match.group(1)
        return None

    def _extract_arxiv_category(self, action: str) -> Optional[str]:
        """Extract arXiv category from action string."""
        import re
        # Category patterns: cs.AI, physics.quant-ph, math.CO, etc.
        match = re.search(r'([a-z-]+\.[A-Z]{2,}(?:-[a-z]+)?)', action)
        if match:
            return match.group(1)
        # Also try lowercase
        match = re.search(r'([a-z-]+\.[a-z]{2,}(?:-[a-z]+)?)', action, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_url(self, action: str) -> Optional[str]:
        """Extract URL from action string."""
        import re
        # Look for full URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, action)
        if match:
            return match.group()
        # Look for domain-like patterns
        domain_pattern = r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        match = re.search(domain_pattern, action)
        if match:
            return match.group()
        # Look for quoted text that might be a URL
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        for q in quoted:
            if '.' in q and ' ' not in q:
                return q
        return None

    def _extract_selector(self, action: str) -> Optional[str]:
        """Extract CSS selector from action string."""
        import re
        # Look for quoted selectors
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Look for common selector patterns
        selector_pattern = r'(#[\w-]+|\.[\w-]+|\[[\w-]+=?[^\]]*\])'
        match = re.search(selector_pattern, action)
        if match:
            return match.group()
        return None

    def _extract_number(self, action: str) -> Optional[int]:
        """Extract a number from action string."""
        import re
        numbers = re.findall(r'\d+', action)
        if numbers:
            return int(numbers[0])
        return None

    def _extract_app_name(self, action: str) -> Optional[str]:
        """Extract app name from action string."""
        import re
        action_lower = action.lower()

        # App names from SystemControlTool allowlist
        app_names = [
            'notepad', 'calculator', 'browser', 'chrome', 'firefox',
            'explorer', 'vscode', 'terminal', 'cmd', 'powershell'
        ]

        # Check for each app name
        for app_name in app_names:
            if app_name in action_lower:
                return app_name

        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        return None

    def _extract_task_id(self, action: str) -> Optional[str]:
        """Extract task ID from action string."""
        import re
        # Look for hex-like IDs (8 characters)
        match = re.search(r'\b([a-f0-9]{8})\b', action, re.IGNORECASE)
        if match:
            return match.group(1)
        # Look for quoted ID
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        return None

    def _extract_reminder_params(self, action: str) -> tuple[Optional[str], Optional[str]]:
        """Extract message and time_str from reminder action.

        Examples:
        - "remind me in 30 minutes to take a break" -> ("take a break", "in 30 minutes")
        - "add_reminder 'test' in 1 minute" -> ("test", "in 1 minute")
        - "in 5 minutes remind me to check email" -> ("check email", "in 5 minutes")

        Returns:
            (message, time_str) tuple
        """
        import re
        action_lower = action.lower()

        # Extract time pattern: "in X minutes/hours/seconds"
        time_pattern = r'(in\s+\d+\s+(?:minute|minutes|min|hour|hours|hr|second|seconds|sec)s?)'
        time_match = re.search(time_pattern, action_lower)
        time_str = time_match.group(1) if time_match else None

        # Extract message - look for quoted text first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            message = quoted[0]
        else:
            # Try to extract message from common patterns
            message = None

            # Pattern: "remind me to <message>" or "reminder to <message>"
            to_pattern = r'(?:remind(?:er)?(?:\s+me)?|notification)\s+to\s+(.+?)(?:\s+in\s+\d+|$)'
            to_match = re.search(to_pattern, action_lower)
            if to_match:
                message = to_match.group(1).strip()

            # Pattern: "in X minutes to <message>"
            if not message:
                in_to_pattern = r'in\s+\d+\s+\w+\s+to\s+(.+?)$'
                in_to_match = re.search(in_to_pattern, action_lower)
                if in_to_match:
                    message = in_to_match.group(1).strip()

            # Pattern: "<message> in X minutes" - message before time
            if not message and time_str:
                before_time = action_lower.split(time_str)[0].strip()
                # Remove common prefixes
                before_time = re.sub(r'^(add_reminder|remind(?:er)?(?:\s+me)?|set\s+(?:a\s+)?reminder)\s*', '', before_time, flags=re.IGNORECASE)
                if before_time and len(before_time) > 2:
                    message = before_time.strip(' "\'')

            # Pattern: "in X minutes <message>" - message after time
            if not message and time_str:
                after_time = action_lower.split(time_str)[-1].strip()
                # Remove common connectors
                after_time = re.sub(r'^(to|for|about|that)\s+', '', after_time)
                if after_time and len(after_time) > 2:
                    message = after_time.strip(' "\'')

        return (message, time_str)

    def _extract_scheduled_params(self, action: str) -> tuple[Optional[str], Optional[str], str]:
        """Extract message, time_of_day, and repeat from scheduled notification action.

        Examples:
        - "schedule notification at 9 AM daily for standup" -> ("standup", "9:00 AM", "daily")
        - "every day at 10:30 remind me to check reports" -> ("check reports", "10:30", "daily")

        Returns:
            (message, time_of_day, repeat) tuple
        """
        import re
        action_lower = action.lower()

        # Extract time of day patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:am|pm)?)',  # 9:00 AM, 10:30, 14:00
            r'(\d{1,2}\s*(?:am|pm))',          # 9 AM, 10pm
            r'at\s+(\d{1,2}(?::\d{2})?)\s*(?:am|pm)?',  # at 9, at 10:30
        ]
        time_of_day = None
        for pattern in time_patterns:
            match = re.search(pattern, action_lower)
            if match:
                time_of_day = match.group(1).strip()
                break

        # Determine repeat pattern
        repeat = "daily"  # default
        if "weekday" in action_lower:
            repeat = "weekdays"
        elif "weekly" in action_lower:
            repeat = "weekly"

        # Extract message - look for quoted text first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            message = quoted[0]
        else:
            # Try to extract message from patterns
            message = None

            # Pattern: "for <message>" or "saying <message>" or "about <message>"
            for_pattern = r'(?:for|saying|about|message)\s+(.+?)(?:\s+at\s+\d|$)'
            for_match = re.search(for_pattern, action_lower)
            if for_match:
                message = for_match.group(1).strip()

            # Pattern: "to <message>"
            if not message:
                to_pattern = r'(?:remind(?:\s+me)?|notification)\s+to\s+(.+?)(?:\s+(?:at|every|daily)|$)'
                to_match = re.search(to_pattern, action_lower)
                if to_match:
                    message = to_match.group(1).strip()

        return (message, time_of_day, repeat)

    def _extract_condition_params(self, action: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """Extract message, condition_type, and threshold from conditional alert action.

        Examples:
        - "alert when CPU exceeds 80%" -> ("High CPU alert", "cpu", 80)
        - "notify when RAM above 90%" -> ("High RAM alert", "ram", 90)

        Returns:
            (message, condition_type, threshold) tuple
        """
        import re
        action_lower = action.lower()

        # Extract condition type
        condition_type = None
        if "cpu" in action_lower:
            condition_type = "cpu"
        elif "ram" in action_lower or "memory" in action_lower:
            condition_type = "ram"
        elif "disk" in action_lower:
            condition_type = "disk"

        # Extract threshold percentage
        threshold = None
        threshold_patterns = [
            r'(\d+)\s*%',           # 80%, 90 %
            r'(?:above|over|exceeds?|greater\s+than|>)\s*(\d+)',  # above 80, exceeds 90
        ]
        for pattern in threshold_patterns:
            match = re.search(pattern, action_lower)
            if match:
                threshold = int(match.group(1))
                break

        # Extract message - look for quoted text first
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        message = quoted[0] if quoted else None

        return (message, condition_type, threshold)

    def _generate_tool_spec(self, user_request: str) -> Optional[dict]:
        """Generate a tool specification from a natural language request.

        Uses the LLM to parse the user's request and generate a structured
        specification with name, description, and functions_spec.

        Args:
            user_request: The user's natural language tool creation request

        Returns:
            Dict with name, description, and functions_spec, or None if parsing fails
        """
        import json
        import re

        prompt = f"""You are a tool specification generator. Given a user's request, generate a JSON specification for a new tool.

User request: {user_request}

Generate a JSON object with EXACTLY this structure:
{{
    "name": "tool_name_in_snake_case",
    "description": "Brief description of what the tool does",
    "functions_spec": [
        {{
            "name": "function_name",
            "params": ["param1", "param2"],
            "description": "What this function does",
            "body": "Python code that returns a dict with 'success' key"
        }}
    ]
}}

Rules for the code in "body":
1. Must be valid Python code
2. Must return a dict with "success": True/False
3. Can use: json, re, datetime, math, random, hashlib, base64, urllib.parse
4. CANNOT use: os, subprocess, sys, socket, requests, eval, exec, __import__
5. Parameters are available as local variables
6. Use float() or int() to convert numeric parameters

Example for a temperature converter:
{{
    "name": "temperature_converter",
    "description": "Convert temperatures between Celsius and Fahrenheit",
    "functions_spec": [
        {{
            "name": "celsius_to_fahrenheit",
            "params": ["celsius"],
            "description": "Convert Celsius to Fahrenheit",
            "body": "temp = float(celsius)\\nfahrenheit = (temp * 9/5) + 32\\nreturn {{\\"success\\": True, \\"fahrenheit\\": round(fahrenheit, 2)}}"
        }},
        {{
            "name": "fahrenheit_to_celsius",
            "params": ["fahrenheit"],
            "description": "Convert Fahrenheit to Celsius",
            "body": "temp = float(fahrenheit)\\ncelsius = (temp - 32) * 5/9\\nreturn {{\\"success\\": True, \\"celsius\\": round(celsius, 2)}}"
        }}
    ]
}}

Output ONLY the JSON object, no other text."""

        try:
            # Use reasoning model for this complex task
            from .brain import TaskType
            response = self.brain.think(
                prompt,
                task_type=TaskType.REASONING,
                use_history=False
            )

            # Extract JSON from response
            # Try to find JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return {"error": f"No JSON found in response: {response[:200]}"}

            json_str = json_match.group()

            # Parse the JSON
            spec = json.loads(json_str)

            # Validate required fields
            if "name" not in spec:
                return {"error": "Missing 'name' in tool specification"}
            if "description" not in spec:
                return {"error": "Missing 'description' in tool specification"}
            if "functions_spec" not in spec or not spec["functions_spec"]:
                return {"error": "Missing or empty 'functions_spec' in tool specification"}

            # Validate each function spec
            for func in spec["functions_spec"]:
                if "name" not in func:
                    return {"error": "Function missing 'name'"}
                if "params" not in func:
                    func["params"] = []  # Default to no params
                if "description" not in func:
                    func["description"] = f"Function {func['name']}"
                if "body" not in func:
                    return {"error": f"Function '{func['name']}' missing 'body'"}

            return spec

        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON in response: {e}"}
        except Exception as e:
            return {"error": f"Failed to generate tool spec: {e}"}

    def _execute_fluxmind_action(self, tool, action: str) -> dict:
        """Execute FluxMind calibrated reasoning actions.

        Handles commands like:
        - "Ask FluxMind about state [5,3,7,2]"
        - "Ask FluxMind how confident it is about state [5,3,7,2]"
        - "FluxMind step [5,3,7,2] op 0 context 0"
        - "FluxMind verify sequence"
        - "FluxMind status"

        Args:
            tool: The FluxMindTool instance
            action: The action string to parse

        Returns:
            dict with success status and results
        """
        import re
        action_lower = action.lower()

        # Check if tool is available
        if not tool.is_available():
            return {
                "success": False,
                "error": "FluxMind model not loaded. Train it first with train_fluxmind().",
                "status": tool.status()
            }

        # Status check
        if "status" in action_lower:
            status = tool.status()
            return {"success": True, "status": status, "formatted": tool.format_for_aura(status)}

        # Extract state from various formats: [5,3,7,2], (5,3,7,2), 5,3,7,2
        state_match = re.search(r'\[?\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\)?', action)
        state = None
        if state_match:
            state = [int(state_match.group(i)) for i in range(1, 5)]

        # Extract operation and context if provided
        op_match = re.search(r'op(?:eration)?\s*[=:]?\s*(\d+)', action_lower)
        ctx_match = re.search(r'(?:context|ctx|dsl)\s*[=:]?\s*(\d+)', action_lower)
        operation = int(op_match.group(1)) if op_match else 0
        context = int(ctx_match.group(1)) if ctx_match else 0

        # "Ask FluxMind" or "how confident" - get confidence check
        if "ask fluxmind" in action_lower or "how confident" in action_lower or "confidence" in action_lower:
            if not state:
                return {
                    "success": False,
                    "error": "No state provided. Use format: Ask FluxMind about state [5,3,7,2]"
                }
            result = tool.get_confidence(state, operation, context)
            result["success"] = True
            result["formatted"] = tool.format_for_aura(result)
            return result

        # "step" command - execute single step
        if "step" in action_lower:
            if not state:
                return {
                    "success": False,
                    "error": "No state provided. Use format: FluxMind step [5,3,7,2] op 0 context 0"
                }
            result = tool.step(state, operation, context)
            result["success"] = True
            result["formatted"] = tool.format_for_aura(result)
            return result

        # "execute" or "program" - run full program
        if "execute" in action_lower or "program" in action_lower:
            if not state:
                return {
                    "success": False,
                    "error": "No initial state provided."
                }
            # Extract operations list
            ops_match = re.search(r'ops?\s*[=:]?\s*\[([0-9,\s]+)\]', action_lower)
            ctxs_match = re.search(r'(?:contexts?|dsls?)\s*[=:]?\s*\[([0-9,\s]+)\]', action_lower)
            if ops_match:
                operations = [int(x.strip()) for x in ops_match.group(1).split(',')]
                contexts = [int(x.strip()) for x in ctxs_match.group(1).split(',')] if ctxs_match else [0] * len(operations)
                result = tool.execute(state, operations, contexts)
                result["success"] = True
                result["formatted"] = tool.format_for_aura(result)
                return result
            return {"success": False, "error": "No operations list provided. Use format: ops=[0,2,4]"}

        # "verify" - verify action sequence
        if "verify" in action_lower:
            return {
                "success": False,
                "error": "Verify requires a sequence of actions. Use tool.verify_sequence() directly."
            }

        # Default: if state provided, do confidence check
        if state:
            result = tool.get_confidence(state, operation, context)
            result["success"] = True
            result["formatted"] = tool.format_for_aura(result)
            return result

        return {
            "success": False,
            "error": "Could not parse FluxMind action. Try: 'Ask FluxMind about state [5,3,7,2]'"
        }

    def _detect_fluxmind_action(self, goal: str) -> Optional[tuple[str, str]]:
        """Detect if the goal requires the FluxMind tool.

        Args:
            goal: The user's goal

        Returns:
            (tool_name, action) tuple if FluxMind matches, None otherwise
        """
        goal_lower = goal.lower()

        fluxmind_keywords = [
            'ask fluxmind', 'fluxmind', 'how confident', 'calibrated',
            'confidence check', 'verify sequence', 'uncertainty'
        ]

        # Check if any FluxMind keyword is present
        if any(kw in goal_lower for kw in fluxmind_keywords):
            print(f"[FLUXMIND] Detected FluxMind action in: {goal[:50]}...")
            return ("fluxmind", goal)

    def _detect_git_action(self, goal: str) -> Optional[tuple[str, str]]:
        """Detect if the goal requires the Git tool.

        Args:
            goal: The user's goal

        Returns:
            (tool_name, action) tuple if Git matches, None otherwise
        """
        goal_lower = goal.lower()

        git_keywords = [
            # Explicit git commands
            'git status', 'git log', 'git diff', 'git branch', 'git stash',
            'git add', 'git commit', 'git push', 'git pull', 'git clone',
            # Branch queries
            'what branch', 'which branch', 'current branch', 'show branches', 'list branches',
            # Commit queries
            'show commits', 'recent commits', 'commit history', 'last commit',
            # Status queries
            'staged files', 'unstaged', 'untracked', 'show changes',
            'what changed', 'pending changes', 'working tree'
        ]

        # Check if any Git keyword is present
        if any(kw in goal_lower for kw in git_keywords):
            print(f"[GIT] Detected Git action in: {goal[:50]}...")
            return ("git", goal)

        return None

    def _detect_monologue_action(self, goal: str) -> Optional[tuple[str, str]]:
        """Detect if the goal requires the Inner Monologue tool.

        Args:
            goal: The user's goal

        Returns:
            (tool_name, action) tuple if monologue matches, None otherwise
        """
        goal_lower = goal.lower()

        monologue_keywords = [
            'show thoughts', 'your thoughts', 'recent thoughts',
            'think aloud on', 'think aloud off',
            'verbosity 0', 'verbosity 1', 'verbosity 2', 'verbosity 3',
            'why did you do that', 'explain your reasoning', 'reasoning chain',
            'what were you thinking', 'how did you decide',
            'export thoughts', 'inner monologue'
        ]

        # Check if any monologue keyword is present
        if any(kw in goal_lower for kw in monologue_keywords):
            print(f"[MONOLOGUE] Detected monologue action in: {goal[:50]}...")
            return ("inner_monologue", goal)

        return None

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

    def _detect_knowledge_graph_action(self, goal: str) -> Optional[Tuple[str, str]]:
        """Detect knowledge graph commands in user goals."""
        goal_lower = goal.lower()

        kg_keywords = [
            'knowledge graph', 'show graph', 'what do you know about',
            'how is', 'related to', 'connected to', 'find path',
            'show knowledge', 'graph memory', 'what have you learned',
            'consolidate memory', 'forget about'
        ]

        if any(kw in goal_lower for kw in kg_keywords):
            print(f"[KG] Detected knowledge graph action in: {goal[:50]}...")
            return ("knowledge_graph", goal)

        return None

    def _handle_knowledge_graph_command(self, message: str) -> Optional[str]:
        """Handle knowledge graph commands directly, bypassing the LLM.

        Args:
            message: The user's message

        Returns:
            Formatted result string if KG command, None otherwise
        """
        msg_lower = message.lower()

        kg_keywords = [
            'what do you know about', 'knowledge graph', 'show graph',
            'how is', 'related to', 'connected to', 'find path between',
            'what have you learned', 'consolidate memory', 'graph stats'
        ]

        if not any(kw in msg_lower for kw in kg_keywords):
            return None

        if "knowledge_graph" not in self.tools:
            return "Knowledge graph not available."

        kg = self.tools["knowledge_graph"]

        # Handle specific patterns
        if "what do you know about" in msg_lower:
            topic = msg_lower.split("what do you know about")[-1].strip().rstrip("?")
            result = kg.execute(f"query {topic}")
            if result.get("success") and result.get("results"):
                return "Here's what I know:\n" + "\n".join(result["results"])
            return f"I don't have much knowledge about '{topic}' yet."

        if "how is" in msg_lower and "related to" in msg_lower:
            # Extract "how is X related to Y"
            parts = msg_lower.replace("?", "").split("related to")
            if len(parts) == 2:
                source = parts[0].replace("how is", "").strip()
                target = parts[1].strip()
                result = kg.execute(f"path {source} to {target}")
                if result.get("success") and result.get("path"):
                    return f"Connection: {result['path']}"
                return f"No direct connection found between '{source}' and '{target}'."

        if "consolidate memory" in msg_lower:
            result = kg.execute("consolidate")
            return f"Memory consolidated: {result.get('merged_nodes', 0)} nodes merged, {result.get('pruned_edges', 0)} edges pruned."

        if "graph stats" in msg_lower or "knowledge graph" in msg_lower:
            result = kg.execute("stats")
            if result.get("success"):
                return f"Knowledge Graph: {result['total_nodes']} nodes, {result['total_edges']} edges, {result['clusters']} clusters"

        # Generic query
        result = kg.execute(message)
        if result.get("success"):
            if result.get("results"):
                return "\n".join(result["results"])
            return str(result)

        return result.get("error", "Unknown error")

    def _handle_guardian_command(self, message: str) -> Optional[str]:
        """Handle Metacognitive Guardian commands directly.

        Args:
            message: The user's message

        Returns:
            Formatted result string if guardian command, None otherwise
        """
        msg_lower = message.lower()

        guardian_keywords = [
            'guardian stats', 'guardian status', 'guardian level',
            'set guardian', 'monitoring level', 'show predictions',
            'how confident', 'failure patterns', 'reset guardian'
        ]

        if not any(kw in msg_lower for kw in guardian_keywords):
            return None

        if not hasattr(self, 'guardian') or self.guardian is None:
            return "Metacognitive Guardian not available."

        # Handle specific patterns
        if "guardian stats" in msg_lower or "guardian status" in msg_lower:
            stats = self.guardian.get_stats()
            return (f"**Metacognitive Guardian Status**\n"
                   f"- Monitoring Level: {stats['monitoring_level']}\n"
                   f"- Session Predictions: {stats['session_predictions']}\n"
                   f"- Interventions Triggered: {stats['interventions_triggered']}\n"
                   f"- Failure Patterns Learned: {stats['failure_patterns_learned']}\n"
                   f"- Thresholds: Warning={stats['thresholds']['warning']:.0%}, "
                   f"Intervention={stats['thresholds']['intervention']:.0%}, "
                   f"Abort={stats['thresholds']['abort']:.0%}")

        if "set guardian" in msg_lower or "monitoring level" in msg_lower:
            for level in ["low", "medium", "high", "critical"]:
                if level in msg_lower:
                    self.guardian.set_monitoring_level(level)
                    return f"Guardian monitoring level set to **{level}**."
            return ("Available monitoring levels:\n"
                   "- **low**: Only critical risks\n"
                   "- **medium**: Balanced monitoring (default)\n"
                   "- **high**: Sensitive monitoring\n"
                   "- **critical**: Maximum sensitivity")

        if "show predictions" in msg_lower:
            predictions = self.guardian.session_predictions
            if not predictions:
                return "No predictions made in this session yet."
            recent = predictions[-5:]
            lines = ["**Recent Predictions:**"]
            for p in recent:
                lines.append(f"- [{p.failure_type.value}] {p.probability:.0%} - {p.reasoning[:50]}...")
            return "\n".join(lines)

        if "failure patterns" in msg_lower:
            patterns = self.guardian.failure_patterns
            if not patterns:
                return "No failure patterns learned yet."
            return f"Guardian has learned from **{len(patterns)}** past failures."

        if "reset guardian" in msg_lower:
            self.guardian.reset_session()
            return "Guardian session stats reset."

        if "how confident" in msg_lower:
            stats = self.guardian.get_stats()
            level = stats['monitoring_level']
            return (f"I'm monitoring at **{level}** sensitivity. "
                   f"This session: {stats['session_predictions']} risk assessments, "
                   f"{stats['interventions_triggered']} interventions.")

        return None

    def record_user_feedback(self, was_helpful: bool, feedback_text: str = None):
        """Record user feedback to improve guardian predictions.

        Args:
            was_helpful: Whether the response was helpful
            feedback_text: Optional text feedback
        """
        if self._pending_prediction and hasattr(self, 'guardian'):
            self.guardian.record_outcome(
                prediction_id=self._pending_prediction.id,
                was_successful=was_helpful,
                user_feedback=feedback_text
            )
            self._pending_prediction = None

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
            if self.neurodream.current_phase != SleepPhase.AWAKE:
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
            if self.neurodream.current_phase == SleepPhase.AWAKE:
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

    def _get_final_result(self) -> dict:
        """Compile the final result of the agent run."""
        # End inner monologue session
        if self.state.completed:
            self.monologue.think("reflect", "Task completed successfully!")
        else:
            self.monologue.think("reflect", f"Stopping after {self.state.iteration} iterations.")
        session_summary = self.monologue.end_session()

        result = {
            "goal": self.state.goal,
            "completed": self.state.completed,
            "iterations": self.state.iteration,
            "final_evaluation": self.state.evaluation,
            "history": self.state.history,
            "monologue_summary": session_summary.get("summary", {})
        }
        # Include actual tool result for FluxMind (prevents LLM hallucination in output)
        if self.state.last_action and self.state.last_action.get('tool', '').lower() == 'fluxmind':
            result["fluxmind_result"] = self.state.last_result
            if self.state.last_result and self.state.last_result.get('formatted'):
                result["response"] = self.state.last_result['formatted']

        # Include actual tool result for Git (prevents LLM hallucination in output)
        if self.state.last_action and self.state.last_action.get('tool', '').lower() == 'git':
            result["git_result"] = self.state.last_result
            if self.state.last_result and self.state.last_result.get('output'):
                result["response"] = self.state.last_result['output']

        return result

    def _handle_fluxmind_command(self, message: str) -> Optional[str]:
        """Handle FluxMind commands directly, bypassing the LLM.

        Args:
            message: The user's message

        Returns:
            Formatted result string if FluxMind command, None otherwise
        """
        import re
        message_lower = message.lower()

        # Check if this is a FluxMind command
        fluxmind_keywords = ['fluxmind', 'ask fluxmind', 'how confident']
        if not any(kw in message_lower for kw in fluxmind_keywords):
            return None  # Not a FluxMind command

        # Check if FluxMind tool is available
        if 'fluxmind' not in self.tools:
            return "FluxMind tool not available. Make sure it's installed and the model is trained."

        tool = self.tools['fluxmind']
        if not tool.is_available():
            return "FluxMind model not loaded. Train it first with: python -c \"from tools.fluxmind import train_fluxmind; train_fluxmind('models/fluxmind_v0751.pt')\""

        # Status command
        if "status" in message_lower:
            status = tool.status()
            return f"FluxMind Status:\n  Available: {status['available']}\n  Version: {status['version']}\n  Capabilities: {', '.join(status['capabilities'])}"

        # Extract state from message: [5,3,7,2] or (5,3,7,2)
        state_match = re.search(r'\[?\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\)?', message)
        state = [int(state_match.group(i)) for i in range(1, 5)] if state_match else None

        # Extract operation and context
        op_match = re.search(r'op(?:eration)?\s*[=:]?\s*(\d+)', message_lower)
        ctx_match = re.search(r'(?:context|ctx|dsl)\s*[=:]?\s*(\d+)', message_lower)
        operation = int(op_match.group(1)) if op_match else 0
        context = int(ctx_match.group(1)) if ctx_match else 0

        # Step command
        if "step" in message_lower:
            if not state:
                return "FluxMind step requires a state. Example: FluxMind step [5,3,7,2] op 0 context 0"
            result = tool.step(state, operation, context)
            return f"FluxMind Step Result:\n  Input: {state}\n  Next State: {result['next_state']}\n  Confidence: {result['confidence']:.2%}\n  Should Trust: {result['should_trust']}"

        # Confidence check (ask fluxmind, how confident, about state)
        if state and ("confident" in message_lower or "about" in message_lower or "ask" in message_lower):
            result = tool.get_confidence(state, operation, context)
            trust_msg = "Yes, proceed" if result['should_trust'] else ("No, abstain" if result['should_abstain'] else "Verify recommended")
            return f"FluxMind Confidence Check:\n  State: {state}\n  Confidence: {result['confidence']:.2%}\n  Should Trust: {trust_msg}\n  Interpretation: {result['interpretation']}"

        # Default: if we have a state, do confidence check
        if state:
            result = tool.get_confidence(state, operation, context)
            return f"FluxMind Confidence: {result['confidence']:.2%} - {result['interpretation']}"

        # Fallback: Any FluxMind question that doesn't match specific commands
        # Returns a helpful summary of capabilities
        status = tool.status()
        return f"""FluxMind Capabilities:

Version: {status['version']}
Status: {'Online' if status['available'] else 'Offline'}

What I can do with FluxMind:
- Calibrated confidence (I actually know when I don't know)
- 99.86% accuracy on familiar inputs
- 0.06% confidence on unfamiliar inputs (1664x drop!)
- Sub-millisecond inference (<1ms vs 500ms+ for LLMs)

Try these commands:
- "FluxMind status"
- "Ask FluxMind about state [5,3,7,2]"
- "FluxMind step [5,3,7,2] op 0 context 0"
- "How confident is FluxMind about [25,25,25,25]?\""""

    def chat(self, message: str, speak: bool = False) -> str:
        """Simple chat interface for one-off interactions.

        Args:
            message: User message
            speak: If True, speak the response using TTS

        Returns:
            Agent response text
        """
        # Analyze emotional state (EvoEmo - Tool #20)
        emotion_reading = self._analyze_emotion(message)

        # Check for FluxMind commands FIRST, before LLM
        fluxmind_result = self._handle_fluxmind_command(message)
        if fluxmind_result:
            if speak:
                self._speak(fluxmind_result, emotion=emotion_reading.emotion if emotion_reading else None)
            return fluxmind_result

        # Check for EvoEmo commands
        evoemo_result = self._handle_evoemo_command(message)
        if evoemo_result:
            if speak:
                self._speak(evoemo_result)
            return evoemo_result

        # Use fast model for simple queries (greetings, etc.)
        if self._is_simple_query(message):
            task_type = TaskType.SIMPLE
        else:
            task_type = None  # Let brain auto-detect

        # Apply emotional tone modifier if detected with high confidence
        tone_modifier = None
        if emotion_reading and emotion_reading.confidence >= 50:
            tone_modifier = get_tone_modifier(emotion_reading.emotion)

        response = self.brain.think(message, task_type=task_type, tone_modifier=tone_modifier)

        if speak:
            self._speak(response, emotion=emotion_reading.emotion if emotion_reading else None)

        return response

    def _analyze_emotion(self, message: str):
        """Analyze emotional state from user message."""
        try:
            if "evoemo" in self.tools and self.tools["evoemo"].is_enabled():
                return self.tools["evoemo"].analyze_text(message)
        except Exception as e:
            print(f"[EvoEmo] Analysis error: {e}")
        return None

    def _handle_evoemo_command(self, message: str) -> Optional[str]:
        """Handle EvoEmo-specific commands."""
        message_lower = message.lower()

        evoemo_commands = [
            "my mood", "how am i feeling", "current mood", "mood status",
            "mood history", "emotion history", "clear mood", "disable mood",
            "enable mood", "mood patterns"
        ]

        if not any(cmd in message_lower for cmd in evoemo_commands):
            return None

        try:
            evoemo = self.tools.get("evoemo")
            if not evoemo:
                return None

            if "clear" in message_lower:
                result = evoemo.clear_history()
                return "Mood history cleared." if result.get("success") else "Failed to clear history."

            elif "disable" in message_lower:
                evoemo.set_enabled(False)
                return "Mood tracking disabled."

            elif "enable" in message_lower:
                evoemo.set_enabled(True)
                return "Mood tracking enabled."

            elif "history" in message_lower:
                history = evoemo.get_history(days=7)
                if not history:
                    return "No mood history yet."
                # Summarize
                from collections import Counter
                emotions = [h["emotion"] for h in history]
                dist = Counter(emotions)
                summary = ", ".join(f"{e}: {c}" for e, c in dist.most_common())
                return f"Mood history (7 days, {len(history)} readings): {summary}"

            elif "pattern" in message_lower:
                patterns = evoemo.get_patterns()
                if patterns.get("status") == "insufficient_data":
                    return f"Not enough data for patterns yet ({patterns.get('readings', 0)} readings)."
                dominant = patterns.get("dominant_emotion", "calm")
                stress_hours = patterns.get("stress_hours", [])
                stress_info = f" Stress tends to peak around: {stress_hours}" if stress_hours else ""
                return f"Your dominant mood: {dominant}.{stress_info}"

            else:
                # Current mood
                mood = evoemo.get_current_mood()
                if mood:
                    emoji = evoemo.get_mood_emoji()
                    return f"Current mood: {emoji} {mood.emotion} ({mood.confidence}% confidence)"
                return "No mood data yet. Keep chatting and I'll pick up on how you're feeling."

        except Exception as e:
            print(f"[EvoEmo] Command error: {e}")
            return None

    def get_current_mood(self):
        """Get current emotional state (for external use)."""
        try:
            if "evoemo" in self.tools:
                return self.tools["evoemo"].get_current_mood()
        except:
            pass
        return None

    def get_mood_emoji(self) -> str:
        """Get emoji for current mood (for GUI)."""
        try:
            if "evoemo" in self.tools:
                return self.tools["evoemo"].get_mood_emoji()
        except:
            pass
        return "😐"

    def _speak(self, text: str, emotion: Optional[str] = None):
        """Speak text using TTS with optional emotional adaptation."""
        try:
            from .tools.voice import VoiceTool
            voice = VoiceTool()

            # Adapt voice parameters based on emotion if available
            if emotion:
                from .tools.evoemo_prompts import get_voice_params
                params = get_voice_params(emotion)
                # Voice tools may support rate/pitch adjustment
                voice.speak(text, rate=params.get("rate", 1.0))
            else:
                voice.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")

    def recall_memories(self, query: str, n: int = 5) -> list:
        """Recall relevant memories."""
        return self.memory.recall(query, n_results=n)
