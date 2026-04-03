"""Browser control tool using Playwright — world-class automation."""

import base64
import json
import logging
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry decorator for stale element / transient failures
# ---------------------------------------------------------------------------

def _auto_retry(max_retries: int = 3, delay: float = 0.3):
    """Retry on playwright errors (stale element, detached frame, etc.).

    When a selector-based method fails, attempts self-healing by calling
    _heal_selector before retrying with an alternative selector.
    """
    # Error substrings that suggest a selector failed to match
    _SELECTOR_ERRORS = [
        "waiting for selector",
        "no element matches selector",
        "failed to find element",
        "element not found",
        "selector resolved to hidden",
    ]

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            last_err = None
            current_args = list(args)
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(self, *current_args, **kwargs)
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    retryable = any(k in err_str for k in [
                        "element is not attached",
                        "element is detached",
                        "frame was detached",
                        "execution context was destroyed",
                        "target closed",
                        "navigation",
                    ])

                    # Try selector healing if the first arg looks like a selector
                    is_selector_error = any(k in err_str for k in _SELECTOR_ERRORS)
                    if is_selector_error and current_args and isinstance(current_args[0], str):
                        retryable = True  # selector errors are retryable via healing
                        healer = getattr(self, "_heal_selector", None)
                        if healer:
                            healed = healer(current_args[0], str(e)[:200])
                            if healed:
                                logger.debug(
                                    "[BrowserTool] Healed selector %r -> %r",
                                    current_args[0], healed,
                                )
                                current_args[0] = healed

                    if not retryable or attempt == max_retries:
                        raise
                    logger.debug(
                        "[BrowserTool] Retry %d/%d for %s: %s",
                        attempt, max_retries, fn.__name__, str(e)[:100],
                    )
                    time.sleep(delay * attempt)
            raise last_err  # unreachable but satisfies type checker
        return wrapper
    return decorator


class BrowserTool:
    """Tool for browser automation using Playwright.

    Capabilities: multi-tab, smart waits, rich element interaction,
    page analysis, session/cookie persistence, auto-retry, popup dismissal.
    """

    name = "browser"
    description = "Control a web browser to navigate, interact with pages, and extract content"

    # Safety blocklist
    BLOCKED_PATTERNS = [
        "login", "signin", "sign-in", "sign_in",
        "checkout", "payment", "pay.",
        "bank", "banking",
        "password", "passwd", "pwd",
    ]

    # Common popup selectors to auto-dismiss
    POPUP_DISMISS_SELECTORS = [
        # Cookie banners
        "button:has-text('Accept all')",
        "button:has-text('Accept All')",
        "button:has-text('Accept cookies')",
        "button:has-text('Accept Cookies')",
        "button:has-text('I agree')",
        "button:has-text('Got it')",
        "button:has-text('OK')",
        "[id*='cookie'] button",
        "[class*='cookie'] button",
        "[id*='consent'] button",
        "[class*='consent'] button",
        # Notification prompts
        "button:has-text('No thanks')",
        "button:has-text('Not now')",
        "button:has-text('Maybe later')",
        "button:has-text('Dismiss')",
        # Generic close buttons on overlays
        "[class*='modal'] [class*='close']",
        "[class*='overlay'] [class*='close']",
        "[class*='popup'] [class*='close']",
        "[aria-label='Close']",
        "[aria-label='Dismiss']",
    ]

    def __init__(
        self,
        output_dir: str = "screenshots",
        headless: bool = True,
        user_agent: Optional[str] = None,
        viewport: Optional[Dict[str, int]] = None,
        cookies_path: Optional[str] = None,
    ):
        """Initialize browser tool.

        Args:
            output_dir: Directory to save screenshots
            headless: Run browser in headless mode
            user_agent: Custom user-agent string
            viewport: Custom viewport, e.g. {"width": 1920, "height": 1080}
            cookies_path: Path to persist cookies JSON
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.user_agent = user_agent
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.cookies_path = Path(cookies_path) if cookies_path else self.output_dir / "cookies.json"

        self._playwright = None
        self._browser = None
        self._context = None

        # Multi-tab state: tab_id -> Page
        self._tabs: Dict[str, Any] = {}
        self._active_tab: Optional[str] = None
        self._tab_counter: int = 0

        # World-class upgrades
        self._vision_tool = None       # Lazy VisionTool
        self._downloads: Dict[str, dict] = {}  # Download tracking
        self.session_file = self.output_dir / "session.json"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _page(self):
        """Current active page. Backward compat."""
        if self._active_tab and self._active_tab in self._tabs:
            return self._tabs[self._active_tab]
        return None

    @_page.setter
    def _page(self, value):
        """Setter kept for backward compat with _ensure_browser."""
        if value is not None and self._active_tab:
            self._tabs[self._active_tab] = value

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    def _ensure_browser(self) -> None:
        """Ensure browser is initialized with at least one tab."""
        if self._page is not None:
            return

        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)

        context_opts = {
            "viewport": self.viewport,
        }
        if self.user_agent:
            context_opts["user_agent"] = self.user_agent

        self._context = self._browser.new_context(**context_opts)

        # Restore cookies if saved
        self._try_load_cookies()

        # Block notification permission prompts
        self._context.grant_permissions([])

        page = self._context.new_page()
        tab_id = self._make_tab_id()
        self._tabs[tab_id] = page
        self._active_tab = tab_id

    def _make_tab_id(self) -> str:
        self._tab_counter += 1
        return f"tab_{self._tab_counter}"

    def _is_blocked_url(self, url: str) -> bool:
        """Check if URL contains blocked patterns for safety."""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.BLOCKED_PATTERNS)

    def _dismiss_popups(self) -> None:
        """Try to dismiss common popups/cookie banners on current page."""
        if self._page is None:
            return
        for sel in self.POPUP_DISMISS_SELECTORS:
            try:
                el = self._page.query_selector(sel)
                if el and el.is_visible():
                    el.click(timeout=1000)
                    time.sleep(0.2)
                    return  # one popup dismissed is enough per call
            except Exception as e:
                logger.debug(f"[BrowserTool] Popup dismiss failed: {e}")
                continue

    # ------------------------------------------------------------------
    # 1. MULTI-TAB SUPPORT
    # ------------------------------------------------------------------

    def open_tab(self, url: Optional[str] = None) -> dict:
        """Open a new tab, optionally navigating to url. Returns tab_id.

        Args:
            url: Optional URL to navigate to in the new tab

        Returns:
            dict with tab_id, success status
        """
        try:
            self._ensure_browser()
            page = self._context.new_page()
            tab_id = self._make_tab_id()
            self._tabs[tab_id] = page
            self._active_tab = tab_id

            result = {
                "success": True,
                "tab_id": tab_id,
                "message": f"Opened new tab {tab_id}",
            }

            if url:
                nav = self.open(url)
                result.update({"url": nav.get("url"), "title": nav.get("title")})

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def switch_tab(self, tab_id: str) -> dict:
        """Switch to a specific tab by tab_id.

        Args:
            tab_id: The tab identifier returned by open_tab or list_tabs

        Returns:
            dict with success status and current tab info
        """
        if tab_id not in self._tabs:
            return {"success": False, "error": f"Tab {tab_id} not found. Use list_tabs() to see available tabs."}
        try:
            self._active_tab = tab_id
            page = self._tabs[tab_id]
            return {
                "success": True,
                "tab_id": tab_id,
                "url": page.url,
                "title": page.title(),
                "message": f"Switched to {tab_id}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close_tab(self, tab_id: Optional[str] = None) -> dict:
        """Close a tab. If tab_id is None, closes active tab.

        Args:
            tab_id: Tab to close. Defaults to active tab.

        Returns:
            dict with success status
        """
        tab_id = tab_id or self._active_tab
        if not tab_id or tab_id not in self._tabs:
            return {"success": False, "error": f"Tab {tab_id} not found"}
        try:
            self._tabs[tab_id].close()
            del self._tabs[tab_id]

            # Switch to another tab if we closed the active one
            if tab_id == self._active_tab:
                if self._tabs:
                    self._active_tab = next(iter(self._tabs))
                else:
                    self._active_tab = None

            return {
                "success": True,
                "closed": tab_id,
                "active_tab": self._active_tab,
                "remaining_tabs": len(self._tabs),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_tabs(self) -> dict:
        """Return all open tabs with their titles, URLs, and active status.

        Returns:
            dict with tabs list
        """
        tabs = []
        for tid, page in self._tabs.items():
            try:
                tabs.append({
                    "tab_id": tid,
                    "url": page.url,
                    "title": page.title(),
                    "active": tid == self._active_tab,
                })
            except Exception as e:
                logger.debug(f"[BrowserTool] Tab info fetch failed for {tid}: {e}")
                tabs.append({
                    "tab_id": tid,
                    "url": "unknown",
                    "title": "unknown",
                    "active": tid == self._active_tab,
                })
        return {"success": True, "count": len(tabs), "tabs": tabs}

    # ------------------------------------------------------------------
    # 2. SMART WAITS
    # ------------------------------------------------------------------

    def wait_for_selector(self, selector: str, timeout: float = 10) -> dict:
        """Wait until element matching selector appears on page.

        Args:
            selector: CSS selector to wait for
            timeout: Max wait in seconds

        Returns:
            dict with success status and whether element was found
        """
        try:
            self._ensure_browser()
            self._page.wait_for_selector(selector, timeout=int(timeout * 1000))
            return {
                "success": True,
                "selector": selector,
                "message": f"Element '{selector}' found",
            }
        except Exception as e:
            return {"success": False, "error": f"Timeout waiting for '{selector}': {e}", "selector": selector}

    def wait_for_navigation(self, timeout: float = 10) -> dict:
        """Wait for page navigation/load to complete.

        Args:
            timeout: Max wait in seconds

        Returns:
            dict with success status and page info after navigation
        """
        try:
            self._ensure_browser()
            self._page.wait_for_load_state("domcontentloaded", timeout=int(timeout * 1000))
            return {
                "success": True,
                "url": self._page.url,
                "title": self._page.title(),
                "message": "Navigation complete",
            }
        except Exception as e:
            return {"success": False, "error": f"Navigation wait timed out: {e}"}

    def wait_for_text(self, text: str, timeout: float = 10) -> dict:
        """Wait until specific text appears on the page.

        Args:
            text: Text string to wait for
            timeout: Max wait in seconds

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            # Use Playwright's built-in text selector with page.wait_for_selector
            # Escape quotes in text for the selector
            escaped = text.replace("'", "\\'")
            self._page.wait_for_selector(
                f"text='{escaped}'",
                timeout=int(timeout * 1000),
            )
            return {
                "success": True,
                "text": text,
                "message": f"Text '{text[:50]}' appeared on page",
            }
        except Exception as e:
            return {"success": False, "error": f"Text '{text[:50]}' not found within {timeout}s: {e}"}

    def wait_for_network_idle(self, timeout: float = 5) -> dict:
        """Wait until no network requests for 500ms.

        Args:
            timeout: Max wait in seconds

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._page.wait_for_load_state("networkidle", timeout=int(timeout * 1000))
            return {"success": True, "message": "Network idle"}
        except Exception as e:
            return {"success": False, "error": f"Network not idle within {timeout}s: {e}"}

    # ------------------------------------------------------------------
    # 3. ELEMENT INTERACTION
    # ------------------------------------------------------------------

    @_auto_retry()
    def fill_form(self, fields: Dict[str, str]) -> dict:
        """Fill multiple form fields at once.

        Args:
            fields: Dict of {selector: value} pairs

        Returns:
            dict with success status and per-field results
        """
        try:
            self._ensure_browser()
            results = []
            for selector, value in fields.items():
                try:
                    self._page.wait_for_selector(selector, timeout=5000)
                    self._page.fill(selector, value, timeout=5000)
                    results.append({"selector": selector, "filled": True})
                except Exception as e:
                    results.append({"selector": selector, "filled": False, "error": str(e)[:100]})
            succeeded = sum(1 for r in results if r["filled"])
            return {
                "success": succeeded == len(fields),
                "total": len(fields),
                "filled": succeeded,
                "failed": len(fields) - succeeded,
                "details": results,
                "message": f"Filled {succeeded}/{len(fields)} fields",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @_auto_retry()
    def select_option(self, selector: str, value: str, by: str = "value") -> dict:
        """Select an option from a dropdown.

        Args:
            selector: CSS selector for the <select> element
            value: The value/label/index to select
            by: Selection method — "value", "label", or "index"

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._page.wait_for_selector(selector, timeout=5000)

            if by == "label":
                selected = self._page.select_option(selector, label=value)
            elif by == "index":
                selected = self._page.select_option(selector, index=int(value))
            else:
                selected = self._page.select_option(selector, value=value)

            return {
                "success": True,
                "selector": selector,
                "selected": selected,
                "message": f"Selected '{value}' in {selector}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    @_auto_retry()
    def upload_file(self, selector: str, path: str) -> dict:
        """Upload a file to a file input element.

        Args:
            selector: CSS selector for the file input
            path: Absolute path to the file to upload

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            file_path = Path(path)
            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            self._page.wait_for_selector(selector, timeout=5000)
            self._page.set_input_files(selector, str(file_path))
            return {
                "success": True,
                "selector": selector,
                "file": file_path.name,
                "message": f"Uploaded {file_path.name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    @_auto_retry()
    def scroll_to(self, selector: str) -> dict:
        """Scroll an element into view.

        Args:
            selector: CSS selector for element to scroll to

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._page.wait_for_selector(selector, timeout=5000)
            self._page.eval_on_selector(
                selector,
                "el => el.scrollIntoView({behavior: 'smooth', block: 'center'})"
            )
            time.sleep(0.3)  # let smooth scroll finish
            return {
                "success": True,
                "selector": selector,
                "message": f"Scrolled to {selector}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    @_auto_retry()
    def hover(self, selector: str) -> dict:
        """Hover over an element.

        Args:
            selector: CSS selector for element to hover

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._page.wait_for_selector(selector, timeout=5000)
            self._page.hover(selector, timeout=5000)
            return {
                "success": True,
                "selector": selector,
                "message": f"Hovered over {selector}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    # ------------------------------------------------------------------
    # 4. PAGE ANALYSIS
    # ------------------------------------------------------------------

    def get_page_text(self) -> dict:
        """Extract readable text content from the current page.

        Returns:
            dict with page text (cleaned, deduplicated whitespace)
        """
        try:
            self._ensure_browser()
            text = self._page.inner_text("body")
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
            return {
                "success": True,
                "url": self._page.url,
                "title": self._page.title(),
                "text": text[:10000],
                "length": len(text),
                "truncated": len(text) > 10000,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_links(self) -> dict:
        """Return all links on the page with text and href.

        Returns:
            dict with list of {text, href} dicts
        """
        try:
            self._ensure_browser()
            links = self._page.eval_on_selector_all(
                "a[href]",
                """elements => elements.map(el => ({
                    text: el.innerText.trim().substring(0, 100),
                    href: el.href,
                    target: el.target || '_self'
                }))"""
            )
            links = [
                link for link in links
                if link["href"] and not link["href"].startswith("javascript:")
            ]
            return {
                "success": True,
                "url": self._page.url,
                "count": len(links),
                "links": links[:200],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_forms(self) -> dict:
        """Return all forms on the page with their fields.

        Returns:
            dict with forms list, each containing action, method, and field details
        """
        try:
            self._ensure_browser()
            forms = self._page.evaluate("""() => {
                return Array.from(document.querySelectorAll('form')).map((form, idx) => {
                    const fields = Array.from(form.querySelectorAll(
                        'input, select, textarea, button[type="submit"]'
                    )).map(el => ({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        name: el.name || '',
                        id: el.id || '',
                        placeholder: el.placeholder || '',
                        required: el.required || false,
                        value: el.type === 'password' ? '***' : (el.value || '').substring(0, 50),
                        options: el.tagName === 'SELECT'
                            ? Array.from(el.options).map(o => ({
                                value: o.value,
                                text: o.text.substring(0, 50),
                                selected: o.selected
                              }))
                            : undefined
                    }));
                    return {
                        index: idx,
                        action: form.action || '',
                        method: (form.method || 'GET').toUpperCase(),
                        id: form.id || '',
                        name: form.name || '',
                        fields: fields
                    };
                });
            }""")
            return {
                "success": True,
                "url": self._page.url,
                "count": len(forms),
                "forms": forms[:20],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_tables(self) -> dict:
        """Extract table data as lists of dicts (header row becomes keys).

        Returns:
            dict with tables list, each table is a list of row dicts
        """
        try:
            self._ensure_browser()
            tables = self._page.evaluate("""() => {
                return Array.from(document.querySelectorAll('table')).slice(0, 10).map((table, idx) => {
                    const rows = Array.from(table.querySelectorAll('tr'));
                    if (rows.length === 0) return { index: idx, headers: [], rows: [] };

                    // Try to get headers from <thead> or first <tr>
                    let headers = [];
                    const thead = table.querySelector('thead');
                    if (thead) {
                        headers = Array.from(thead.querySelectorAll('th, td'))
                            .map(c => c.innerText.trim().substring(0, 50));
                    }
                    if (headers.length === 0 && rows.length > 0) {
                        headers = Array.from(rows[0].querySelectorAll('th, td'))
                            .map(c => c.innerText.trim().substring(0, 50));
                    }

                    // Get data rows (skip header row if we pulled headers from first tr)
                    const startIdx = thead ? 0 : 1;
                    const dataRows = rows.slice(startIdx).map(row => {
                        const cells = Array.from(row.querySelectorAll('td, th'))
                            .map(c => c.innerText.trim().substring(0, 100));
                        if (headers.length > 0) {
                            const obj = {};
                            headers.forEach((h, i) => { obj[h || ('col_' + i)] = cells[i] || ''; });
                            return obj;
                        }
                        return cells;
                    });

                    return {
                        index: idx,
                        headers: headers,
                        row_count: dataRows.length,
                        rows: dataRows.slice(0, 100)  // limit rows
                    };
                });
            }""")
            return {
                "success": True,
                "url": self._page.url,
                "count": len(tables),
                "tables": tables,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def screenshot(self, filename: Optional[str] = None, full_page: bool = True) -> dict:
        """Take a screenshot, return file path and base64.

        Args:
            filename: Optional filename (without extension). If None, auto-generated.
            full_page: Capture full scrollable page (default True)

        Returns:
            dict with file path and base64 encoded image
        """
        try:
            self._ensure_browser()

            if filename:
                safe_name = Path(filename).stem
                if not safe_name or '/' in filename or '\\' in filename or '..' in filename:
                    return {"success": False, "error": "Invalid filename: must not contain path separators"}
                filepath = self.output_dir / f"{safe_name}.png"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"browser_{timestamp}.png"

            raw_bytes = self._page.screenshot(path=str(filepath), full_page=full_page)
            b64 = base64.b64encode(raw_bytes).decode("ascii") if raw_bytes else None

            return {
                "success": True,
                "path": str(filepath.absolute()),
                "filename": filepath.name,
                "url": self._page.url,
                "base64": b64,
                "message": f"Screenshot saved to {filepath}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # 5. SESSION / COOKIE MANAGEMENT
    # ------------------------------------------------------------------

    def save_cookies(self, path: Optional[str] = None) -> dict:
        """Save browser cookies to a JSON file for later restoration.

        Args:
            path: Optional file path. Uses default cookies_path if None.

        Returns:
            dict with success status and cookie count
        """
        try:
            self._ensure_browser()
            cookies = self._context.cookies()
            save_path = Path(path) if path else self.cookies_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
            return {
                "success": True,
                "count": len(cookies),
                "path": str(save_path.absolute()),
                "message": f"Saved {len(cookies)} cookies",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_cookies(self, path: Optional[str] = None) -> dict:
        """Load cookies from a JSON file into the browser context.

        Args:
            path: Optional file path. Uses default cookies_path if None.

        Returns:
            dict with success status and cookie count
        """
        try:
            self._ensure_browser()
            load_path = Path(path) if path else self.cookies_path
            if not load_path.exists():
                return {"success": False, "error": f"Cookie file not found: {load_path}"}

            cookies = json.loads(load_path.read_text(encoding="utf-8"))
            self._context.add_cookies(cookies)
            return {
                "success": True,
                "count": len(cookies),
                "path": str(load_path.absolute()),
                "message": f"Loaded {len(cookies)} cookies",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _try_load_cookies(self) -> None:
        """Silently try to load cookies on browser init."""
        try:
            if self.cookies_path and self.cookies_path.exists():
                cookies = json.loads(self.cookies_path.read_text(encoding="utf-8"))
                if cookies:
                    self._context.add_cookies(cookies)
                    logger.debug("[BrowserTool] Restored %d cookies from %s", len(cookies), self.cookies_path)
        except Exception as e:
            logger.debug(f"[BrowserTool] Cookie restore failed (non-critical): {e}")

    def set_user_agent(self, user_agent: str) -> dict:
        """Update user-agent. Takes effect on next new context/tab.

        Args:
            user_agent: New user-agent string

        Returns:
            dict with success status
        """
        self.user_agent = user_agent
        return {"success": True, "user_agent": user_agent, "message": "User-agent set. Takes effect on next browser restart."}

    def set_viewport(self, width: int, height: int) -> dict:
        """Resize viewport on the active page.

        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._page.set_viewport_size({"width": width, "height": height})
            self.viewport = {"width": width, "height": height}
            return {
                "success": True,
                "viewport": self.viewport,
                "message": f"Viewport set to {width}x{height}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # 6. ERROR RECOVERY
    # ------------------------------------------------------------------

    def dismiss_popups(self) -> dict:
        """Manually trigger popup/cookie banner dismissal.

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._dismiss_popups()
            return {"success": True, "message": "Popup dismissal attempted"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # BACKWARD-COMPATIBLE METHODS (original API preserved)
    # ------------------------------------------------------------------

    def open(self, url: str) -> dict:
        """Navigate to a URL and return page info.

        Args:
            url: URL to navigate to

        Returns:
            dict with success status, title, and status code
        """
        _url_lower = url.lower().strip()
        if _url_lower.startswith(('file://', 'data:', 'javascript:', 'ftp://', 'blob:')):
            return {
                "success": False,
                "error": "URL scheme blocked for safety: only http(s) URLs are allowed",
                "url": url,
            }

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        if self._is_blocked_url(url):
            return {
                "success": False,
                "error": "URL blocked for safety: contains sensitive pattern (login/payment/bank)",
                "url": url,
            }

        # SSRF protection: block private IPs and DNS rebinding
        try:
            from aura.security.ssrf_guard import validate_url_safe
            validate_url_safe(url)
        except ValueError as e:
            return {"success": False, "error": f"SSRF blocked: {e}", "url": url}
        except ImportError:
            pass

        try:
            self._ensure_browser()
            response = self._page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Auto-dismiss popups after navigation
            try:
                self._dismiss_popups()
            except Exception as e:
                logger.debug(f"[BrowserTool] Post-navigation popup dismiss failed: {e}")

            return {
                "success": True,
                "url": self._page.url,
                "title": self._page.title(),
                "status": response.status if response else None,
                "tab_id": self._active_tab,
                "message": f"Navigated to {self._page.title()}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}

    def get_text(self) -> dict:
        """Extract visible text from current page. Alias for get_page_text."""
        return self.get_page_text()

    def get_links(self) -> dict:
        """Extract all links from current page. Alias for get_page_links."""
        return self.get_page_links()

    @_auto_retry()
    def click(self, selector: str) -> dict:
        """Click an element by CSS selector.

        Args:
            selector: CSS selector for element to click

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()

            href = self._page.get_attribute(selector, "href")
            if href and self._is_blocked_url(href):
                return {
                    "success": False,
                    "error": "Click blocked: link leads to sensitive page",
                    "selector": selector,
                }

            self._page.click(selector, timeout=10000)
            self._page.wait_for_load_state("domcontentloaded", timeout=10000)

            return {
                "success": True,
                "selector": selector,
                "url": self._page.url,
                "title": self._page.title(),
                "message": f"Clicked element: {selector}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    @_auto_retry()
    def fill(self, selector: str, text: str) -> dict:
        """Fill text into an input field.

        Args:
            selector: CSS selector for input element
            text: Text to type

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()
            self._page.wait_for_selector(selector, timeout=5000)
            self._page.fill(selector, text, timeout=10000)

            return {
                "success": True,
                "selector": selector,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "message": f"Filled text into {selector}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

    def search_google(self, query: str) -> dict:
        """Search Google and return results.

        Args:
            query: Search query

        Returns:
            dict with success status and search results
        """
        try:
            self._ensure_browser()

            self._page.goto("https://www.google.com", wait_until="domcontentloaded")

            # Accept cookies if prompted
            try:
                self._page.click("button:has-text('Accept all')", timeout=3000)
            except Exception as e:
                logger.debug(f"[BrowserTool] Google cookie accept skipped: {e}")

            self._page.fill("textarea[name='q'], input[name='q']", query)
            self._page.keyboard.press("Enter")
            self._page.wait_for_load_state("domcontentloaded")

            results = self._page.eval_on_selector_all(
                "#search .g",
                """elements => elements.slice(0, 10).map(el => {
                    const titleEl = el.querySelector('h3');
                    const linkEl = el.querySelector('a');
                    const snippetEl = el.querySelector('[data-sncf], .VwiC3b');
                    return {
                        title: titleEl ? titleEl.innerText : '',
                        url: linkEl ? linkEl.href : '',
                        snippet: snippetEl ? snippetEl.innerText : ''
                    };
                }).filter(r => r.title && r.url)"""
            )

            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    # ==================================================================
    # WORLD-CLASS UPGRADES
    # ==================================================================

    def _heal_selector(self, failed_selector: str, error_context: str = "") -> Optional[str]:
        """Try alternative selectors when CSS selector fails."""
        if self._page is None:
            return None
        try:
            for tag in ["button", "a", "input", "select", "textarea", "*"]:
                candidates = [
                    f"{tag}[aria-label]",
                    f"{tag}[role='button']",
                    f"{tag}[role='link']",
                ]
                for sel in candidates:
                    try:
                        el = self._page.query_selector(sel)
                        if el and el.is_visible():
                            return sel
                    except Exception:
                        pass
        except Exception:
            pass
        return None

    def visual_query(self, question: str) -> dict:
        """Screenshot current page and ask vision model a question about it."""
        try:
            self._ensure_browser()
            ss = self.screenshot()
            if not ss.get("success"):
                return {"success": False, "error": "Screenshot failed"}

            if self._vision_tool is None:
                from aura.tools.vision import VisionTool
                self._vision_tool = VisionTool()

            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(base64.b64decode(ss["base64"]))
                tmp_path = tmp.name

            result = self._vision_tool.analyze_image(tmp_path, question)
            os.unlink(tmp_path)

            return {
                "success": True,
                "analysis": result.get("description", result.get("content", str(result))),
                "question": question,
            }
        except ImportError:
            return {"success": False, "error": "VisionTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def visual_click(self, description: str) -> dict:
        """Use vision to locate an element by description and click it."""
        try:
            self._ensure_browser()
            ss = self.screenshot()
            if not ss.get("success"):
                return {"success": False, "error": "Screenshot failed"}

            if self._vision_tool is None:
                from aura.tools.vision import VisionTool
                self._vision_tool = VisionTool()

            import tempfile, os, re as _re
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(base64.b64decode(ss["base64"]))
                tmp_path = tmp.name

            prompt = (
                f"Find the UI element matching: '{description}'. "
                "Return ONLY the pixel coordinates as: x,y (from top-left). "
                "If not found, return 'not_found'."
            )
            result = self._vision_tool.analyze_image(tmp_path, prompt)
            os.unlink(tmp_path)

            text = result.get("description", result.get("content", str(result)))
            if "not_found" in text.lower():
                return {"success": False, "error": f"Element not found: {description}"}

            match = _re.search(r'(\d+)\s*,\s*(\d+)', text)
            if not match:
                return {"success": False, "error": f"Could not parse coordinates from: {text[:100]}"}

            x, y = int(match.group(1)), int(match.group(2))
            self._page.mouse.click(x, y)
            return {"success": True, "description": description, "coordinates": [x, y]}
        except ImportError:
            return {"success": False, "error": "VisionTool not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_page_context(self) -> dict:
        """Extract current page context for LLM planning."""
        try:
            self._ensure_browser()
            return {
                "url": self._page.url if self._page else "about:blank",
                "title": self._page.title() if self._page else "",
                "text": self.get_page_text().get("text", "")[:2000],
                "forms_count": len(self.get_page_forms().get("forms", [])),
                "links_count": len(self.get_page_links().get("links", [])),
            }
        except Exception:
            return {"url": "unknown", "title": "", "text": "", "forms_count": 0, "links_count": 0}

    def plan_and_execute(self, goal: str, max_steps: int = 10) -> dict:
        """Decompose a goal into browser steps via LLM, execute with re-planning."""
        try:
            self._ensure_browser()
            try:
                from aura.core.brain import OllamaBrain
                brain = OllamaBrain()
            except Exception:
                return {"success": False, "error": "LLM not available for planning"}

            execution_log = []
            for replan in range(3):
                ctx = self._get_page_context()
                prompt = (
                    f"Goal: {goal}\n\nCurrent page: {ctx['url']} - {ctx['title']}\n"
                    f"Text: {ctx['text'][:1000]}\n\n"
                    f"Return a JSON array of steps. Each step: "
                    f'[{{"action":"click|fill|wait|screenshot|get_text","selector":"...","value":"...","description":"..."}}]'
                )
                try:
                    import re as _re
                    resp = brain.chat(prompt) if hasattr(brain, 'chat') else brain.think(prompt)
                    match = _re.search(r'\[.*\]', resp, _re.DOTALL)
                    if not match:
                        return {"success": False, "error": f"Plan not JSON: {resp[:200]}"}
                    plan = json.loads(match.group())
                except Exception as e:
                    return {"success": False, "error": f"Planning failed: {e}"}

                failed = False
                for i, step in enumerate(plan[:max_steps]):
                    act = step.get("action", "").lower()
                    try:
                        if act == "click":
                            r = self.click(step.get("selector", ""))
                        elif act == "fill":
                            r = self.fill(step.get("selector", ""), step.get("value", ""))
                        elif act == "wait":
                            r = self.wait_for_text(step.get("value", ""), timeout=5)
                        elif act == "screenshot":
                            r = self.screenshot()
                        elif act == "get_text":
                            r = self.get_page_text()
                        else:
                            r = {"success": False, "error": f"Unknown: {act}"}

                        execution_log.append({"step": i + 1, "desc": step.get("description", act), "success": r.get("success", False)})
                        if not r.get("success"):
                            failed = True
                            break
                    except Exception as e:
                        execution_log.append({"step": i + 1, "desc": step.get("description", act), "success": False, "error": str(e)})
                        failed = True
                        break

                if not failed:
                    return {"success": True, "goal": goal, "steps": len(execution_log), "log": execution_log}

            return {"success": False, "goal": goal, "log": execution_log, "error": "Plan failed after re-planning"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download(self, url_or_selector: str) -> dict:
        """Download a file by URL or by clicking a download link."""
        try:
            self._ensure_browser()
            is_url = url_or_selector.startswith(("http://", "https://"))

            with self._page.expect_download(timeout=30000) as dl_info:
                if is_url:
                    self._page.goto(url_or_selector)
                else:
                    self._page.click(url_or_selector, timeout=10000)

            dl = dl_info.value
            filename = dl.suggested_filename or "download"
            target = self.output_dir / filename
            dl.save_as(str(target))
            size = target.stat().st_size if target.exists() else 0
            self._downloads[filename] = {"path": str(target), "filename": filename, "size": size}
            return {"success": True, "filename": filename, "path": str(target), "size": size}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_downloads(self) -> dict:
        """List all downloads from this session."""
        return {"success": True, "count": len(self._downloads), "downloads": list(self._downloads.values())}

    def save_session(self, path: Optional[str] = None) -> dict:
        """Save tab URLs and scroll positions to JSON."""
        try:
            self._ensure_browser()
            tabs = []
            for tid, page in self._tabs.items():
                try:
                    scroll = page.evaluate("() => ({x: window.scrollX, y: window.scrollY})")
                except Exception:
                    scroll = {"x": 0, "y": 0}
                tabs.append({"tab_id": tid, "url": page.url, "title": page.title(), "scroll": scroll})

            save_path = Path(path) if path else self.session_file
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps({"active_tab": self._active_tab, "tabs": tabs}, indent=2))
            return {"success": True, "path": str(save_path), "tab_count": len(tabs)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def restore_session(self, path: Optional[str] = None) -> dict:
        """Restore tabs and scroll positions from saved session."""
        try:
            self._ensure_browser()
            load_path = Path(path) if path else self.session_file
            if not load_path.exists():
                return {"success": False, "error": f"Session file not found: {load_path}"}

            session = json.loads(load_path.read_text())
            restored = 0
            for i, tab_data in enumerate(session.get("tabs", [])):
                try:
                    if i == 0 and self._page:
                        self._page.goto(tab_data["url"], wait_until="domcontentloaded", timeout=15000)
                    else:
                        page = self._context.new_page()
                        tid = self._make_tab_id()
                        self._tabs[tid] = page
                        page.goto(tab_data["url"], wait_until="domcontentloaded", timeout=15000)

                    scroll = tab_data.get("scroll", {})
                    self._page.evaluate(f"window.scrollTo({scroll.get('x', 0)}, {scroll.get('y', 0)})")
                    restored += 1
                except Exception as e:
                    logger.debug(f"[BrowserTool] Tab restore failed: {e}")

            return {"success": True, "tabs_restored": restored}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close(self) -> dict:
        """Close the browser and all tabs.

        Returns:
            dict with success status
        """
        try:
            # Save cookies before closing
            try:
                if self._context:
                    cookies = self._context.cookies()
                    if cookies:
                        self.cookies_path.parent.mkdir(parents=True, exist_ok=True)
                        self.cookies_path.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
            except Exception as e:
                logger.debug(f"[BrowserTool] Cookie save on close failed: {e}")

            # Close all tab references
            self._tabs.clear()
            self._active_tab = None

            if self._context:
                self._context.close()
                self._context = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None

            return {"success": True, "message": "Browser closed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # EXECUTE — action dispatcher (backward compatible + new actions)
    # ------------------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a browser action by name.

        Supports all original actions plus new ones:
        - open_tab, switch_tab, close_tab, list_tabs
        - wait_for_selector, wait_for_navigation, wait_for_text, wait_for_network_idle
        - fill_form, select_option, upload_file, scroll_to, hover
        - get_page_text, get_page_links, get_page_forms, get_page_tables
        - screenshot (enhanced), save_cookies, load_cookies, dismiss_popups
        - set_viewport

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower().strip()

        # --- Multi-tab ---
        if action_lower == "open_tab":
            return self.open_tab(url=kwargs.get("url"))
        elif action_lower == "switch_tab":
            return self.switch_tab(tab_id=kwargs.get("tab_id", ""))
        elif action_lower == "close_tab":
            return self.close_tab(tab_id=kwargs.get("tab_id"))
        elif action_lower == "list_tabs":
            return self.list_tabs()

        # --- Smart waits ---
        elif action_lower == "wait_for_selector":
            return self.wait_for_selector(
                selector=kwargs.get("selector", ""),
                timeout=kwargs.get("timeout", 10),
            )
        elif action_lower == "wait_for_navigation":
            return self.wait_for_navigation(timeout=kwargs.get("timeout", 10))
        elif action_lower == "wait_for_text":
            return self.wait_for_text(
                text=kwargs.get("text", ""),
                timeout=kwargs.get("timeout", 10),
            )
        elif action_lower in ("wait_for_network_idle", "wait_network_idle"):
            return self.wait_for_network_idle(timeout=kwargs.get("timeout", 5))

        # --- Element interaction ---
        elif action_lower == "fill_form":
            return self.fill_form(fields=kwargs.get("fields", {}))
        elif action_lower == "select_option":
            return self.select_option(
                selector=kwargs.get("selector", ""),
                value=kwargs.get("value", ""),
                by=kwargs.get("by", "value"),
            )
        elif action_lower == "upload_file":
            return self.upload_file(
                selector=kwargs.get("selector", ""),
                path=kwargs.get("path", ""),
            )
        elif action_lower == "scroll_to":
            return self.scroll_to(selector=kwargs.get("selector", ""))
        elif action_lower == "hover":
            return self.hover(selector=kwargs.get("selector", ""))

        # --- Page analysis ---
        elif action_lower in ("get_page_text", "page_text"):
            return self.get_page_text()
        elif action_lower in ("get_page_links", "page_links"):
            return self.get_page_links()
        elif action_lower in ("get_page_forms", "page_forms"):
            return self.get_page_forms()
        elif action_lower in ("get_page_tables", "page_tables"):
            return self.get_page_tables()

        # --- Session ---
        elif action_lower == "save_cookies":
            return self.save_cookies(path=kwargs.get("path"))
        elif action_lower == "load_cookies":
            return self.load_cookies(path=kwargs.get("path"))
        elif action_lower == "dismiss_popups":
            return self.dismiss_popups()
        elif action_lower == "set_viewport":
            return self.set_viewport(
                width=kwargs.get("width", 1280),
                height=kwargs.get("height", 720),
            )

        # --- World-class upgrades ---
        elif action_lower == "visual_query":
            return self.visual_query(question=kwargs.get("question", ""))
        elif action_lower == "visual_click":
            return self.visual_click(description=kwargs.get("description", ""))
        elif action_lower == "plan_and_execute":
            return self.plan_and_execute(goal=kwargs.get("goal", ""))
        elif action_lower == "download":
            return self.download(url_or_selector=kwargs.get("url") or kwargs.get("selector", ""))
        elif action_lower == "get_downloads":
            return self.get_downloads()
        elif action_lower == "save_session":
            return self.save_session(path=kwargs.get("path"))
        elif action_lower == "restore_session":
            return self.restore_session(path=kwargs.get("path"))

        # --- Original actions (backward compat) ---
        elif "open" in action_lower or "goto" in action_lower or "navigate" in action_lower:
            url = kwargs.get("url") or self._extract_url(action)
            if not url:
                return {"success": False, "error": "No URL provided"}
            return self.open(url)

        elif "screenshot" in action_lower or "capture" in action_lower:
            filename = kwargs.get("filename")
            full_page = kwargs.get("full_page", True)
            return self.screenshot(filename, full_page=full_page)

        elif "text" in action_lower or "content" in action_lower:
            return self.get_page_text()

        elif "links" in action_lower:
            return self.get_page_links()

        elif "forms" in action_lower:
            return self.get_page_forms()

        elif "tables" in action_lower:
            return self.get_page_tables()

        elif "click" in action_lower:
            selector = kwargs.get("selector") or self._extract_selector(action)
            if not selector:
                return {"success": False, "error": "No selector provided"}
            return self.click(selector)

        elif "fill" in action_lower or "type" in action_lower:
            selector = kwargs.get("selector")
            text = kwargs.get("text")
            if not selector or not text:
                return {"success": False, "error": "Selector and text required"}
            return self.fill(selector, text)

        elif "select" in action_lower:
            selector = kwargs.get("selector")
            value = kwargs.get("value")
            if not selector or not value:
                return {"success": False, "error": "Selector and value required"}
            return self.select_option(selector, value)

        elif "google" in action_lower or "search" in action_lower:
            query = kwargs.get("query") or self._extract_query(action)
            if not query:
                return {"success": False, "error": "No search query provided"}
            return self.search_google(query)

        elif "close" in action_lower or "quit" in action_lower:
            return self.close()

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    # ------------------------------------------------------------------
    # Private helpers (backward compat)
    # ------------------------------------------------------------------

    def _extract_url(self, action: str) -> Optional[str]:
        """Extract URL from action string."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, action)
        if match:
            return match.group()
        domain_pattern = r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        match = re.search(domain_pattern, action)
        if match:
            return match.group()
        return None

    def _extract_selector(self, action: str) -> Optional[str]:
        """Extract CSS selector from action string."""
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        selector_pattern = r'(#[\w-]+|\.[\w-]+|\[[\w-]+=?[^\]]*\]|button|input|a\b)'
        match = re.search(selector_pattern, action)
        if match:
            return match.group()
        return None

    def _extract_query(self, action: str) -> Optional[str]:
        """Extract search query from action string."""
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        patterns = [
            r'(?:search|google)\s+(?:for\s+)?["\']?([^"\']+)',
            r'for\s+["\']?([^"\']+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')
        return None
