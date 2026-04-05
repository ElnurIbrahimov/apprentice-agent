"""
Telegram MarkdownV2 formatting utilities — shared by TelegramBot and TelegramChannel.

Handles code blocks, inline code, bold, italic, links.
Escapes all MarkdownV2 special chars outside of formatting spans.
Splits long messages into <= 4096-char chunks at paragraph boundaries.
"""

import re
from typing import Dict, List

_MARKDOWNV2_ESCAPE = re.compile(r'([_*\[\]()~`>#+\-=|{}.!\\])')
_TELEGRAM_MSG_LIMIT = 4096


def escape_mdv2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MARKDOWNV2_ESCAPE.sub(r'\\\1', text)


def format_telegram_response(text: str) -> List[str]:
    """Convert markdown text to Telegram MarkdownV2 and split into chunks.

    Handles code blocks, inline code, bold, italic, links.
    Escapes all MarkdownV2 special chars outside of formatting spans.
    Returns a list of strings, each <= 4096 chars, split at paragraph boundaries.
    """
    if not text:
        return [""]

    # --- Phase 1: Extract code blocks and inline code to protect them ---
    placeholders: Dict[str, str] = {}
    counter = [0]

    def _save_code_block(m: re.Match) -> str:
        key = f"\x00CB{counter[0]}\x00"
        counter[0] += 1
        lang = m.group(1) or ""
        code = m.group(2)
        placeholders[key] = f"```{lang}\n{code}\n```"
        return key

    def _save_inline_code(m: re.Match) -> str:
        key = f"\x00IC{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = f"`{m.group(1)}`"
        return key

    result = re.sub(r'```(\w*)\n?(.*?)```', _save_code_block, text, flags=re.DOTALL)
    result = re.sub(r'`([^`\n]+)`', _save_inline_code, result)

    # --- Phase 2: Extract links, bold, italic before escaping ---
    link_phs: Dict[str, str] = {}

    def _save_link(m: re.Match) -> str:
        key = f"\x00LK{counter[0]}\x00"
        counter[0] += 1
        link_text = escape_mdv2(m.group(1))
        url = m.group(2)
        link_phs[key] = f"[{link_text}]({url})"
        return key

    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _save_link, result)

    fmt_phs: Dict[str, str] = {}

    def _save_bold(m: re.Match) -> str:
        key = f"\x00BD{counter[0]}\x00"
        counter[0] += 1
        inner = escape_mdv2(m.group(1))
        fmt_phs[key] = f"*{inner}*"
        return key

    def _save_italic(m: re.Match) -> str:
        key = f"\x00IT{counter[0]}\x00"
        counter[0] += 1
        inner = escape_mdv2(m.group(1))
        fmt_phs[key] = f"_{inner}_"
        return key

    # Bold: **text** or __text__
    result = re.sub(r'\*\*(.+?)\*\*', _save_bold, result)
    result = re.sub(r'__(.+?)__', _save_bold, result)
    # Italic: *text* or _text_ (single, non-greedy)
    result = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', _save_italic, result)
    result = re.sub(r'(?<!_)_([^_]+?)_(?!_)', _save_italic, result)

    # --- Phase 3: Escape remaining text ---
    result = escape_mdv2(result)

    # --- Phase 4: Restore placeholders (reverse order of extraction) ---
    for key, val in fmt_phs.items():
        result = result.replace(escape_mdv2(key), val)
    for key, val in link_phs.items():
        result = result.replace(escape_mdv2(key), val)
    for key, val in placeholders.items():
        result = result.replace(escape_mdv2(key), val)

    # --- Phase 5: Split into <= 4096-char chunks ---
    return split_message(result, _TELEGRAM_MSG_LIMIT)


def split_message(text: str, limit: int = _TELEGRAM_MSG_LIMIT) -> List[str]:
    """Split text into chunks of at most *limit* chars at paragraph breaks."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Try paragraph break, then newline, then space, then hard cut
        split_at = remaining.rfind('\n\n', 0, limit)
        if split_at == -1:
            split_at = remaining.rfind('\n', 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(' ', 0, limit)
        if split_at == -1:
            split_at = limit

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip('\n')

    return chunks if chunks else [""]


def format_research_citations(text: str, sources: List[Dict]) -> str:
    """Format research results with numbered citations and clickable source list.

    Args:
        text: The research report body (may already contain [N] refs).
        sources: List of dicts with 'url' and 'title' keys.

    Returns:
        Formatted text with a numbered source list appended.
    """
    if not sources:
        return text

    # De-duplicate by URL
    seen_urls: set = set()
    unique: List[Dict] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(s)

    lines = ["\n\n---\n**Sources:**"]
    for i, src in enumerate(unique, 1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        if url:
            lines.append(f"[{i}] [{title}]({url})")
        else:
            lines.append(f"[{i}] {title}")

    return text + "\n".join(lines)
