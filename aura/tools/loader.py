"""Tool discovery, keyword matching, and loading logic for ApprenticeAgent.

Extracted from aura.agent (2026-03-23).

Contains:
- _TOOL_KEYWORDS / _TOOL_KEYWORDS_RE — keyword-based tool detection
- load_core_tools() — instantiate core (lightweight) tools + tool_search
- load_heavy_tools() — register heavy tools in DeferredToolRegistry (lazy)
- load_synthesized_tools() — auto-load validated synthesized tools
- ensure_tool() — lazy-load a single tool by name (checks deferred registry)
- get_deferred_tool_list() — names/descriptions for system prompt injection
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# ============================================================================
#                    TOOL KEYWORD SET (for fast-path bypass)
# ============================================================================

_TOOL_KEYWORDS = frozenset([
    # --- Tool 0a: code_search (grep/glob/definitions) ---
    'grep', 'search code', 'find in code', 'search in files', 'code search',
    'find definition', 'find class', 'find function', 'find method',
    'find references', 'where is', 'where does', 'which file',
    'project structure', 'show structure', 'codebase', 'repo map',
    'detect project', 'project type', 'what stack',
    'glob', 'find files', 'search files', 'file pattern',

    # --- Tool 0b: code_edit (surgical edits) ---
    'edit file', 'edit code', 'modify file', 'change code', 'update code',
    'replace in file', 'find and replace', 'search replace',
    'refactor', 'rename', 'fix bug', 'fix error', 'patch',
    'add import', 'add function', 'add class', 'add method',
    'remove line', 'delete line', 'insert line',
    'rollback edit', 'undo edit',

    # --- Tool 1: filesystem ---
    'list files', 'show files', 'what files', 'read file', 'write file',
    'open file', 'save file', 'delete file', 'create file', 'file contents',
    'list directory', 'show directory', 'folder contents', 'dir contents',
    'find file', 'search file', 'file system', 'filesystem',

    # --- Tool 2: web_search ---
    'search', 'google', 'look up', 'lookup', 'find online', 'search online',
    'search the web', 'web search', 'search for', 'find out',
    'stock price',
    'weather', 'current weather', 'weather in', 'forecast',
    'news', 'latest news', 'news about', 'headlines',
    'who is', 'what is the current', 'how much does',

    # --- Tool 2b: crypto_price (real-time crypto prices) ---
    'bitcoin price', 'btc price', 'ethereum price', 'eth price',
    'crypto price', 'cryptocurrency price', 'price of bitcoin',
    'price of ethereum', 'price of btc', 'price of eth', 'price of crypto',
    'how much is bitcoin', 'how much is ethereum', 'how much is btc',
    'current bitcoin', 'current ethereum', 'current btc', 'current eth',
    'solana price', 'sol price', 'dogecoin price', 'doge price',
    'cardano price', 'ada price', 'xrp price', 'ripple price',

    # --- Tool 3: code_executor ---
    'run code', 'execute code', 'run python', 'execute python', 'run this',
    'calculate', 'compute', 'factorial', 'fibonacci', 'prime number',
    'write code', 'code to', 'python code', 'script', 'program',

    # --- Tool 4: screenshot ---
    'screenshot', 'take screenshot', 'capture screen', 'capture my screen',
    'screen capture', 'grab screen', 'print screen', 'snapshot', 'screen shot',
    "what's on my screen", 'what is on my screen', 'show my screen',

    # --- Tool 5: vision ---
    'analyze image', 'analyze this image', 'analyze the image',
    'describe image', 'describe this image', 'describe the image',
    'look at image', 'look at this image', 'look at the image',
    "what's in this image", 'what is in this image', 'read image',
    'image analysis', 'picture analysis', 'photo analysis',
    'ocr', 'read text from image', 'extract text from image',

    # --- Tool 6: pdf_reader ---
    'read pdf', 'open pdf', 'pdf file', 'extract pdf', 'pdf contents',
    "what's in this pdf", 'summarize pdf', 'summarize the pdf',
    'search pdf', 'pdf document',

    # --- Tool 7: browser ---
    'browse', 'open website', 'go to website', 'visit website', 'open url',
    'go to url', 'visit url', 'navigate to', 'open page', 'web page',
    'click on', 'click the', 'scroll', 'browser',

    # --- Tool 8: git ---
    'git', 'commit', 'git commit', 'git push', 'git pull', 'git status',
    'git log', 'git diff', 'git stash', 'git branch', 'clone repo',
    'repository', 'repo', 'staged files', 'unstaged', 'untracked',
    'what branch', 'which branch', 'current branch', 'show commits',
    'recent commits', 'show changes', 'list branches',

    # --- Tool 9: arxiv_search ---
    'arxiv', 'research paper', 'academic paper', 'find papers',
    'search papers', 'download paper', 'summarize paper', 'compare papers',
    'scientific paper', 'journal article', 'academic research',

    # --- Tool 10: system_control ---
    'volume', 'set volume', 'get volume', 'brightness', 'set brightness',
    'system info', 'cpu usage', 'ram usage', 'memory usage', 'gpu usage',
    'disk usage', 'disk space', 'open app', 'launch app', 'start app',
    'open notepad', 'open calculator', 'open browser', 'open chrome',
    'open firefox', 'open vscode', 'open terminal', 'lock screen',

    # --- Tool 11: clipboard ---
    'clipboard', 'copy to clipboard', 'paste from clipboard', 'read clipboard',
    'write clipboard', 'clipboard contents', 'what is in clipboard',
    "what's in my clipboard", 'copy this', 'paste this',

    # --- Tool 12: notifications ---
    'remind me', 'reminder', 'set reminder', 'create reminder',
    'notification', 'notify me', 'alert me', 'schedule',
    'in 5 minutes', 'in 10 minutes', 'in 30 minutes', 'in an hour',
    'every day', 'every morning', 'every evening', 'daily at', 'weekly',
    'set alarm', 'timer',

    # --- Tool 13: knowledge_graph / KG Brain ---
    'remember this', 'store this', 'save this fact', 'add to knowledge',
    'what do you know about', 'recall', 'knowledge graph',
    'kg brain', 'kg stats', 'kg query', 'extract entities',
    'learn that', 'remember that', 'consolidate memory', 'graph stats',

    # --- Tool 14: tool_builder ---
    'create tool', 'build tool', 'make tool', 'new tool', 'custom tool',
    'generate tool', 'tool builder', 'design tool',

    # --- Tool 15: marketplace ---
    'marketplace', 'plugin', 'download tool', 'install plugin',
    'browse plugins', 'search plugins', 'uninstall plugin',
    'my plugins', 'installed plugins', 'share tool', 'publish tool',

    # --- Tool 16: regex_builder ---
    'regex', 'regular expression', 'build regex', 'test regex',
    'regex pattern', 'match pattern', 'validate regex', 'explain regex',

    # --- Cognitive System 23: NeuroDream (memory consolidation) ---
    'neurodream', 'go to sleep', 'sleep now', 'dream status',
    'dream journal', 'show dreams', 'sleep insights', 'dream insights',
    'sleep patterns', 'memory consolidation',

    # --- Cognitive System 24: EvoEmo (emotional) ---
    'evoemo', 'my mood', 'how am i feeling', 'mood history',
    'emotional state', 'analyze emotion', 'detect emotion',

    # --- Cognitive System 25: InnerMonologue ---
    'inner monologue', 'show thoughts', 'your thoughts', 'think aloud',
    'reasoning chain', 'what were you thinking', 'export thoughts',

    # --- Tool: deep_research ---
    'deep research', 'research thoroughly', 'thorough research',
    'in-depth research', 'comprehensive research', 'research topic',
    'investigate thoroughly', 'deep dive',

    # --- Tool: image_gen ---
    'generate image', 'create image', 'make image', 'draw image',
    'image generation', 'create picture', 'generate picture',

    # --- Tool: voice/tts ---
    'text to speech', 'speak this', 'say this', 'read aloud',
    'tts', 'sesame tts',

    # --- Tool: load_skill (progressive skill loading) ---
    'load skill', 'get skill', 'skill procedure', 'skill details',
    'show skill', 'use skill',

    # --- Tool: visual_feedback ---
    'render', 'screenshot', 'preview', 'visual', 'feedback', 'iterate',

    # --- Tool: component_registry ---
    'component', 'template', 'hero', 'pricing', 'table', 'form', 'sidebar', 'ui',

    # --- Tool: deploy ---
    'deploy', 'vercel', 'netlify', 'github pages', 'publish site',

    # --- Tool: scaffold ---
    'scaffold', 'boilerplate', 'project template', 'generate project',
])

# Pre-compiled combined regex for _TOOL_KEYWORDS (avoids re-compiling ~150 patterns per message)
_TOOL_KEYWORDS_RE = re.compile(r'\b(?:' + '|'.join(re.escape(kw) for kw in _TOOL_KEYWORDS) + r')\b')


# ============================================================================
#                    TOOL LOADING HELPERS
# ============================================================================

def load_core_tools(brain=None) -> dict:
    """Instantiate core (always-loaded) tools.

    Only ~15 essential tools are loaded immediately.  Everything else is
    registered in the DeferredToolRegistry via ``load_heavy_tools()``.

    Returns a dict mapping tool_name -> tool_instance.
    Tools that fail to import or construct are silently skipped.
    """
    from aura.tools import (
        BraveSearchTool,
        CalendarTool,
        ClipboardTool,
        CodeEditTool,
        CodeExecutorTool,
        CodeSearchTool,
        FileSystemTool,
        GitTool,
        LoadSkillTool,
        NotificationTool,
        TaskManagerTool,
        WebSearchTool,
        get_monologue,
    )
    from aura.tools.tool_search import ToolSearchTool

    tools: dict = {}
    _core = [
        ("code_search", CodeSearchTool),
        ("code_edit", CodeEditTool),
        ("filesystem", FileSystemTool),
        ("web_search", WebSearchTool),
        ("brave_search", BraveSearchTool),
        ("code_executor", CodeExecutorTool),
        ("clipboard", ClipboardTool),
        ("notifications", NotificationTool),
        ("git", GitTool),
        ("calendar", CalendarTool),
        ("task_manager", TaskManagerTool),
        ("tool_search", ToolSearchTool),
        ("load_skill", LoadSkillTool),
    ]
    for name, cls in _core:
        try:
            if cls is None:
                logger.warning(f"[TOOLS] {name} skipped — import failed (class is None)")
                continue
            tools[name] = cls()
            logger.debug(f"[TOOLS] {name} loaded")
        except Exception as e:
            logger.warning(f"[TOOLS] {name} skipped — init failed: {e}")

    # inner_monologue is a factory function, not a class
    try:
        if get_monologue is not None:
            tools["inner_monologue"] = get_monologue()
    except Exception as e:
        logger.warning(f"[TOOLS] inner_monologue skipped — init failed: {e}")

    logger.info(f"[TOOLS] {len(tools)} core tools loaded")
    return tools


def load_heavy_tools(tools: dict, brain=None) -> None:
    """Register heavy/optional tools in the DeferredToolRegistry.

    Instead of instantiating them eagerly, we store name + description +
    a lazy loader.  The LLM sees names/descriptions in the system prompt;
    ``ensure_tool()`` or ``tool_search`` resolves them on demand.

    Called when ``fast_init=False`` (or always — registration is cheap).
    """
    from aura.tools.deferred_registry import deferred_registry as _dr

    # ------------------------------------------------------------------
    #  Deferred tool definitions: (name, description, loader_lambda)
    #
    #  The lambdas capture ``brain`` from the enclosing scope where needed.
    #  None of the imports happen until the lambda is actually called.
    # ------------------------------------------------------------------

    def _make_loader(module_path: str, class_name: str, post_init=None, **kwargs):
        """Return a zero-arg loader that imports + instantiates on demand."""
        def _loader():
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            inst = cls(**kwargs) if kwargs else cls()
            if post_init:
                post_init(inst)
            return inst
        return _loader

    _deferred_defs = [
        # --- Previously in load_core_tools (moved to deferred) ---
        ("tavily_search",       "Search the web using Tavily (AI-optimized). Advanced depth for deep research. Can extract clean content from URLs.",
                                _make_loader("aura.tools.tavily_tool", "TavilyTool")),
        ("firecrawl",           "Scrape a URL or search/crawl the web using Firecrawl. Returns clean markdown content.",
                                _make_loader("aura.tools.firecrawl_tool", "FirecrawlTool")),
        ("crypto_price",        "Get real-time cryptocurrency prices from CoinGecko.",
                                _make_loader("aura.tools.crypto_price", "CryptoPriceTool")),
        ("evoemo",              "Track emotional state and adapt responses.",
                                _make_loader("aura.tools.evoemo", "EvoEmoTool")),
        ("spaced_repetition",   "Flashcard-based learning with SM-2 spaced repetition.",
                                _make_loader("aura.tools.spaced_repetition", "SpacedRepetitionTool")),
        ("research",            "Save and search research notes, findings, and skills.",
                                _make_loader("aura.tools.research_tool", "ResearchTool")),
        ("obsidian",            "Search and index your Obsidian vault notes.",
                                _make_loader("aura.tools.obsidian_tool", "ObsidianTool")),
        ("github",              "GitHub awareness — weekly dev summaries, open PRs/issues, recent commits, repo stats.",
                                _make_loader("aura.tools.github_tool", "GitHubTool")),
        ("log_analyst",         "Analyze terminal output, error logs, tracebacks — extract errors and suggest fixes.",
                                _make_loader("aura.tools.log_analyst", "LogAnalystTool")),
        ("document_generator",  "Generate Word (.docx) and PDF documents from text or markdown.",
                                _make_loader("aura.tools.document_generator", "DocumentGeneratorTool")),
        ("windows_control",     "Control Windows apps via UI Automation — click buttons, fill forms, type text in any app.",
                                _make_loader("aura.tools.windows_control", "WindowsControlTool")),
        ("task_scheduler",      "Schedule AURA tools to run automatically — cron jobs, intervals, one-time tasks.",
                                _make_loader("aura.tools.task_scheduler", "TaskSchedulerTool")),
        ("predictive_tasks",    "Learn workflow patterns from tool usage and predict upcoming tasks.",
                                _make_loader("aura.tools.predictive_tasks", "PredictiveTaskTool")),
        ("meeting_intel",       "Record and analyze meetings locally — Whisper transcription, action items, decisions.",
                                _make_loader("aura.tools.meeting_intel", "MeetingIntelTool")),
        ("voice_synth",         "High-quality neural TTS with 54 voice presets — speak text aloud or save audio files.",
                                _make_loader("aura.tools.voice_synth", "VoiceSynthTool")),
        ("life_logger",         "Unified life timeline — cross-source search and daily/weekly summaries.",
                                _make_loader("aura.tools.life_logger", "LifeLoggerTool")),

        # --- Previously in load_heavy_tools (still deferred) ---
        ("screenshot",          "Capture screenshots of the screen or specific windows.",
                                _make_loader("aura.tools.screenshot", "ScreenshotTool")),
        ("vision",              "Analyze images using vision LLM — OCR, description, object detection.",
                                lambda: __import__("aura.tools.vision", fromlist=["VisionTool"]).VisionTool(brain=brain)),
        ("pdf_reader",          "Extract text and metadata from PDF documents.",
                                _make_loader("aura.tools.pdf_reader", "PDFReaderTool")),
        ("arxiv_search",        "Search arXiv for academic papers, download PDFs, and extract abstracts.",
                                _make_loader("aura.tools.arxiv_search", "ArxivSearchTool")),
        ("browser",             "Control a web browser to navigate, interact with pages, and extract content.",
                                _make_loader("aura.tools.browser", "BrowserTool")),
        ("system_control",      "Control system volume, brightness, launch apps, and get system info.",
                                _make_loader("aura.tools.system_control", "SystemControlTool")),
        ("tool_builder",        "Create, test, and manage custom tools with VOYAGER-style composition.",
                                _make_loader("aura.tools.tool_builder", "ToolBuilderTool")),
        ("marketplace",         "Browse, install, and manage plugins from the marketplace.",
                                _make_loader("aura.tools.marketplace", "MarketplaceTool")),
        ("knowledge_graph",     "Store and query entities and relationships in the local knowledge graph.",
                                lambda: __import__("aura.tools.knowledge_graph", fromlist=["get_knowledge_graph"]).get_knowledge_graph()),
        ("api_tester",          "Test REST APIs — collections, auth, chaining, diff, JSON extraction.",
                                _make_loader("aura.tools.api_tester", "APITesterTool")),
        ("database",            "Query SQLite databases, inspect schemas, import/export CSV.",
                                _make_loader("aura.tools.database_tool", "DatabaseTool")),
        ("audio_transcriber",   "Transcribe audio and video files to text using Whisper.",
                                _make_loader("aura.tools.audio_transcriber", "AudioTranscriberTool")),

        # --- Dynamic-import heavy tools ---
        ("deep_research",       "Deep research: STORM-pattern outline-first, citation-anchored pipeline.",
                                _make_loader("aura.tools.deep_research", "DeepResearchTool",
                                             post_init=lambda t: _inject_deep_research_llm(t, brain))),
        ("codebase_index",      "Semantic codebase index with hybrid search and incremental updates.",
                                _make_loader("aura.tools.codebase_index", "CodebaseIndex")),
        ("image_gen",           "Generate images using Stable Diffusion.",
                                _make_loader("aura.tools.image_gen", "ImageGenerationTool")),
        ("voice",               "Voice input/output and voice conversations.",
                                _make_loader("aura.tools.voice", "VoiceTool")),
        ("local_rag",           "Index and search local documents (PDFs, text, code, markdown).",
                                _make_loader("aura.tools.local_rag", "LocalRAGTool")),
        ("shell_executor",      "Execute shell commands with persistent sessions and security sandboxing.",
                                _make_loader("aura.tools.shell_executor", "ShellExecutorTool")),
        ("screen_reader",       "Monitor screen for changes, extract text, detect active application.",
                                _make_loader("aura.tools.screen_reader", "ScreenReaderTool")),
        ("email",               "Read and send emails via IMAP/SMTP or Gmail API.",
                                _make_loader("aura.tools.email_tool", "EmailTool")),
        ("visual_feedback",     "Render UI code in headless browser, screenshot, iterate for quality.",
                                _make_loader("aura.tools.visual_feedback", "VisualFeedbackLoop")),
        ("component_registry",  "Fetch production-ready UI component templates (hero, pricing, table, etc.).",
                                _make_loader("aura.tools.component_registry", "ComponentRegistryTool")),
        ("deploy",              "Deploy projects to Vercel, Netlify, or GitHub Pages.",
                                _make_loader("aura.tools.deploy_tool", "DeployTool")),
        ("scaffold",            "Generate project templates and boilerplate.",
                                _make_loader("aura.tools.scaffold", "ScaffoldTool")),
    ]

    for name, description, loader in _deferred_defs:
        _dr.register(name, description, loader)

    logger.info(f"[TOOLS] {len(_deferred_defs)} tools registered in DeferredToolRegistry")


def load_synthesized_tools(tools: dict) -> None:
    """Auto-load validated synthesized tools from aura/tools/synthesized/."""
    from aura.security.tool_validator import validate_custom_tool_code

    synth_path = os.path.join(os.path.dirname(__file__), 'synthesized')
    if not os.path.exists(synth_path):
        return

    for file in os.listdir(synth_path):
        if not file.endswith('.py') or file == '__init__.py':
            continue
        tool_name = file[:-3]
        tool_file_path = os.path.join(synth_path, file)
        try:
            with open(tool_file_path, 'r', encoding='utf-8') as f:
                tool_code = f.read()

            is_valid, validation_msg = validate_custom_tool_code(tool_code, tool_file_path)
            if not is_valid:
                logger.warning(f"synthesized/{tool_name} BLOCKED: {validation_msg}")
                continue

            module = __import__(f'aura.tools.synthesized.{tool_name}', fromlist=[tool_name])
            # Try different class name patterns
            class_name = ''.join(word.title() for word in tool_name.split('_')) + 'Tool'
            tool_class = getattr(module, class_name, None)
            if not tool_class:
                tool_class = getattr(module, f'{tool_name}Tool', None)
            if tool_class:
                tools[tool_name] = tool_class()
                logger.info(f"[LOADED] synthesized/{tool_name}")
            elif hasattr(module, 'execute'):
                exec_fn = module.execute
                wrapper = type(f'{tool_name}_tool', (), {
                    'execute': staticmethod(exec_fn),
                    'run': staticmethod(exec_fn),
                    'name': tool_name,
                    'description': getattr(module, '__doc__', '') or tool_name,
                })()
                tools[tool_name] = wrapper
                logger.info(f"[LOADED] synthesized/{tool_name}")
        except Exception as e:
            logger.warning(f"synthesized/{tool_name} not loaded: {e}")


def ensure_tool(tools: dict, tool_name: str, brain=None):
    """Lazily load a tool if not already in *tools*. Returns the tool or None.

    Checks the DeferredToolRegistry first — if the tool is registered there,
    it resolves (instantiates) it and adds it to *tools* for future use.
    """
    if tool_name in tools:
        return tools[tool_name]

    # --- Try the deferred registry first ---
    from aura.tools.deferred_registry import deferred_registry as _dr

    if _dr.has(tool_name):
        inst = _dr.resolve(tool_name)
        if inst is not None:
            tools[tool_name] = inst
            logger.debug(f"[TOOLS] {tool_name} resolved from DeferredToolRegistry")
            return inst

    # --- Fallback: legacy lazy map for tools not yet in deferred registry ---
    from aura.tools import (
        APITesterTool,
        ArxivSearchTool,
        AudioTranscriberTool,
        BrowserTool,
        CalendarTool,
        DatabaseTool,
        MarketplaceTool,
        PDFReaderTool,
        ResearchTool,
        ScreenshotTool,
        SpacedRepetitionTool,
        SystemControlTool,
        TaskManagerTool,
        ToolBuilderTool,
        VisionTool,
        get_knowledge_graph,
    )

    _lazy_map = {
        "knowledge_graph": lambda: get_knowledge_graph(),
        "screenshot": lambda: ScreenshotTool(),
        "vision": lambda: VisionTool(brain=brain),
        "pdf_reader": lambda: PDFReaderTool(),
        "arxiv_search": lambda: ArxivSearchTool(),
        "browser": lambda: BrowserTool(),
        "system_control": lambda: SystemControlTool(),
        "tool_builder": lambda: ToolBuilderTool(),
        "marketplace": lambda: MarketplaceTool(),
        "calendar": lambda: CalendarTool(),
        "spaced_repetition": lambda: SpacedRepetitionTool(),
        "task_manager": lambda: TaskManagerTool(),
        "api_tester": lambda: APITesterTool(),
        "database": lambda: DatabaseTool(),
        "audio_transcriber": lambda: AudioTranscriberTool(),
        "research": lambda: ResearchTool(),
    }

    # Some require dynamic imports
    _lazy_dynamic = {
        "shell_executor": ("aura.tools.shell_executor", "ShellExecutorTool"),
        "screen_reader": ("aura.tools.screen_reader", "ScreenReaderTool"),
        "email": ("aura.tools.email_tool", "EmailTool"),
    }

    if tool_name in _lazy_map:
        try:
            tools[tool_name] = _lazy_map[tool_name]()
        except Exception as e:
            logger.warning(f"[TOOLS] {tool_name} lazy-load failed: {e}")
    elif tool_name in _lazy_dynamic:
        try:
            mod_path, cls_name = _lazy_dynamic[tool_name]
            import importlib
            mod = importlib.import_module(mod_path)
            tools[tool_name] = getattr(mod, cls_name)()
        except Exception as e:
            logger.warning(f"[TOOLS] {tool_name} dynamic lazy-load failed: {e}")

    return tools.get(tool_name)


# ============================================================================
#                    INTERNAL HELPERS
# ============================================================================

def _load_optional(tools: dict, name: str, module_path: str, class_name: str,
                   post_init=None) -> None:
    """Try to import *class_name* from *module_path* and instantiate it."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        inst = cls()
        tools[name] = inst
        if post_init:
            post_init(inst)
        logger.info(f"[LOADED] {name}")
    except Exception as e:
        logger.warning(f"{name} not loaded: {e}")


def _inject_deep_research_llm(tool, brain) -> None:
    """Wire brain.think into DeepResearchTool's LLM hook."""
    if brain and hasattr(tool, 'set_llm'):
        tool.set_llm(lambda p, s=None: brain.think(p, s, use_history=False))


# ============================================================================
#                    DEFERRED TOOL LIST (for system prompt injection)
# ============================================================================

def get_deferred_tool_list() -> list[dict]:
    """Return ``[{name, description}]`` for all deferred (not yet loaded) tools.

    Intended for injection into the LLM system prompt so it knows which
    tools are available on demand via ``tool_search``.
    """
    from aura.tools.deferred_registry import deferred_registry
    return deferred_registry.list_all()
