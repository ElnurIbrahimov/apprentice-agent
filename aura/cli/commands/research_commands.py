import logging
from pathlib import Path
from typing import Optional

from ..context import get_ctx

logger = logging.getLogger(__name__)


def handle_research(agent, arg, context) -> Optional[str]:
    from ..display import console as _research_console
    ctx = get_ctx()
    research_ctx = ctx.research_ctx if ctx else None
    if research_ctx is None:
        from ..research_mode import ResearchContext
        research_ctx = ResearchContext()
        if ctx:
            ctx.research_ctx = research_ctx
    topic = arg.strip()
    if not topic:
        if research_ctx.is_active:
            research_ctx.stop()
            _research_console.print("[dim]Research mode ended.[/dim]")
        else:
            _research_console.print("[dim]Usage: /research <topic>[/dim]")
        return
    research_ctx.start(topic)
    _research_console.print(f"[magenta]Research mode: {topic}[/magenta]")


def handle_sources(agent, arg, context) -> Optional[str]:
    from ..display import console as _sources_console
    ctx = get_ctx()
    research_ctx = ctx.research_ctx if ctx else None
    if research_ctx and research_ctx.is_active:
        from ..research_mode import render_sources
        render_sources(_sources_console, research_ctx)
    else:
        _sources_console.print("[dim]No research session active.[/dim]")


def handle_export(agent, arg, context) -> Optional[str]:
    import re as _re_export

    from ..display import console as _export_console
    ctx = get_ctx()
    research_ctx = ctx.research_ctx if ctx else None
    if research_ctx and research_ctx.is_active:
        md = research_ctx.export_markdown()
        safe_topic = _re_export.sub(r'[^\w\-]', '_', research_ctx.topic)[:30]
        out_path = Path(f"research_{safe_topic}.md")
        out_path.write_text(md)
        _export_console.print(f"[green]Exported to {out_path}[/green]")
    else:
        _export_console.print("[dim]No active research session to export.[/dim]")


def handle_browse(agent, arg, context) -> Optional[str]:
    if not arg:
        from ..display import console
        console.print("Usage: /browse <url> | /browse search <query> | /browse text | /browse screenshot | /browse click <selector> | /browse links")
    else:
        _handle_browse_command(agent, arg)


def handle_recall(agent, arg, context) -> Optional[str]:
    from ..display import console
    if arg:
        memories = agent.recall_memories(arg)
        console.print(f"\nRecalled {len(memories)} memories:")
        for m in memories:
            console.print(f"  - {m.get('content', str(m))[:100]}...")
    else:
        console.print("Usage: /recall <query>")


def _handle_browse_command(agent, arg: str):
    if 'browser' not in agent.tools:
        try:
            from aura.tools.browser import BrowserTool
            agent.tools['browser'] = BrowserTool(headless=False)
        except ImportError:
            print("Browser tool not available. Install playwright: pip install playwright && playwright install")
            return

    browser = agent.tools['browser']
    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower()
    subarg = parts[1] if len(parts) > 1 else ""

    if subcmd == "search":
        if not subarg:
            print("Usage: /browse search <query>")
            return
        from urllib.parse import urlencode
        query_url = f"https://www.google.com/search?{urlencode({'q': subarg})}"
        result = browser.open(query_url)
        if result.get("success"):
            print(f"  Searched: {subarg}")
            print(f"  Title: {result.get('title', 'N/A')}")
            links = browser.get_links()
            if links.get("success"):
                print("  Top results:")
                count = 0
                for link in links.get("links", []):
                    href = link.get("href", "")
                    text = link.get("text", "").strip()
                    if text and "google" not in href and len(text) > 5:
                        print(f"    - {text[:80]}")
                        print(f"      {href}")
                        count += 1
                        if count >= 5:
                            break
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")

    elif subcmd == "text":
        result = browser.get_text()
        if result.get("success"):
            print(f"  Page: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('url', 'N/A')}")
            text = result.get("text", "")
            print(f"  Text ({result.get('length', 0)} chars):\n")
            print(text[:2000])
            if len(text) > 2000:
                print(f"\n  ... ({len(text) - 2000} more chars)")
        else:
            print(f"  Error: {result.get('error', 'No page loaded')}")

    elif subcmd == "screenshot":
        result = browser.screenshot(subarg if subarg else None)
        if result.get("success"):
            print(f"  Screenshot saved: {result.get('path', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Screenshot failed')}")

    elif subcmd == "click":
        if not subarg:
            print("Usage: /browse click <css-selector>")
            return
        result = browser.click(subarg)
        if result.get("success"):
            print(f"  Clicked: {subarg}")
            print(f"  Now at: {result.get('url', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Click failed')}")

    elif subcmd == "links":
        result = browser.get_links()
        if result.get("success"):
            print(f"  URL: {result.get('url', 'N/A')}")
            print(f"  Links ({result.get('count', 0)}):")
            for link in result.get("links", [])[:20]:
                text = link.get("text", "").strip()
                href = link.get("href", "")
                if text:
                    print(f"    [{text[:60]}] {href}")
        else:
            print(f"  Error: {result.get('error', 'No page loaded')}")

    else:
        result = browser.open(arg)
        if result.get("success"):
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('url', 'N/A')}")
            print(f"  Status: {result.get('status', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Navigation failed')}")
