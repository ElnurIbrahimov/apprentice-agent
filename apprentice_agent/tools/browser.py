"""Browser control tool using Playwright."""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
from urllib.parse import urlparse


class BrowserTool:
    """Tool for browser automation using Playwright."""

    name = "browser"
    description = "Control a web browser to navigate, interact with pages, and extract content"

    # Safety blocklist - URLs containing these patterns are blocked
    BLOCKED_PATTERNS = [
        "login", "signin", "sign-in", "sign_in",
        "checkout", "payment", "pay.",
        "bank", "banking",
        "password", "passwd", "pwd"
    ]

    def __init__(self, output_dir: str = "screenshots", headless: bool = True):
        """Initialize browser tool.

        Args:
            output_dir: Directory to save screenshots
            headless: Run browser in headless mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self._browser = None
        self._context = None
        self._page = None

    def _is_blocked_url(self, url: str) -> bool:
        """Check if URL contains blocked patterns for safety."""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.BLOCKED_PATTERNS)

    def _ensure_browser(self) -> None:
        """Ensure browser is initialized."""
        if self._page is None:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._context = self._browser.new_context()
            self._page = self._context.new_page()

    def open(self, url: str) -> dict:
        """Navigate to a URL and return page info.

        Args:
            url: URL to navigate to

        Returns:
            dict with success status, title, and status code
        """
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Safety check
        if self._is_blocked_url(url):
            return {
                "success": False,
                "error": f"URL blocked for safety: contains sensitive pattern (login/payment/bank)",
                "url": url
            }

        try:
            self._ensure_browser()
            response = self._page.goto(url, wait_until="domcontentloaded", timeout=30000)

            return {
                "success": True,
                "url": self._page.url,
                "title": self._page.title(),
                "status": response.status if response else None,
                "message": f"Navigated to {self._page.title()}"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}

    def screenshot(self, filename: Optional[str] = None) -> dict:
        """Capture screenshot of current page.

        Args:
            filename: Optional filename (without extension)

        Returns:
            dict with success status and file path
        """
        try:
            self._ensure_browser()

            if filename:
                filepath = self.output_dir / f"{filename}.png"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"browser_{timestamp}.png"

            self._page.screenshot(path=str(filepath), full_page=True)

            return {
                "success": True,
                "path": str(filepath.absolute()),
                "filename": filepath.name,
                "url": self._page.url,
                "message": f"Screenshot saved to {filepath}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_text(self) -> dict:
        """Extract visible text from current page.

        Returns:
            dict with success status and page text
        """
        try:
            self._ensure_browser()

            # Get visible text content
            text = self._page.inner_text("body")
            # Clean up excessive whitespace
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

            return {
                "success": True,
                "url": self._page.url,
                "title": self._page.title(),
                "text": text[:10000],  # Limit to 10k chars
                "length": len(text)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_links(self) -> dict:
        """Extract all links from current page.

        Returns:
            dict with success status and list of links
        """
        try:
            self._ensure_browser()

            links = self._page.eval_on_selector_all(
                "a[href]",
                """elements => elements.map(el => ({
                    text: el.innerText.trim().substring(0, 100),
                    href: el.href
                }))"""
            )

            # Filter out empty and javascript links
            links = [
                link for link in links
                if link["href"] and not link["href"].startswith("javascript:")
            ]

            return {
                "success": True,
                "url": self._page.url,
                "count": len(links),
                "links": links[:100]  # Limit to 100 links
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def click(self, selector: str) -> dict:
        """Click an element by CSS selector.

        Args:
            selector: CSS selector for element to click

        Returns:
            dict with success status
        """
        try:
            self._ensure_browser()

            # Check if clicking would navigate to blocked URL
            href = self._page.get_attribute(selector, "href")
            if href and self._is_blocked_url(href):
                return {
                    "success": False,
                    "error": f"Click blocked: link leads to sensitive page",
                    "selector": selector
                }

            self._page.click(selector, timeout=10000)
            self._page.wait_for_load_state("domcontentloaded", timeout=10000)

            return {
                "success": True,
                "selector": selector,
                "url": self._page.url,
                "title": self._page.title(),
                "message": f"Clicked element: {selector}"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "selector": selector}

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

            self._page.fill(selector, text, timeout=10000)

            return {
                "success": True,
                "selector": selector,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "message": f"Filled text into {selector}"
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

            # Navigate to Google
            self._page.goto("https://www.google.com", wait_until="domcontentloaded")

            # Accept cookies if prompted (for EU users)
            try:
                self._page.click("button:has-text('Accept all')", timeout=3000)
            except:
                pass  # No cookie prompt

            # Fill search box and submit
            self._page.fill("textarea[name='q'], input[name='q']", query)
            self._page.keyboard.press("Enter")
            self._page.wait_for_load_state("domcontentloaded")

            # Extract search results
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
                "results": results
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def close(self) -> dict:
        """Close the browser.

        Returns:
            dict with success status
        """
        try:
            if self._page:
                self._page.close()
                self._page = None
            if self._context:
                self._context.close()
                self._context = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if hasattr(self, '_playwright') and self._playwright:
                self._playwright.stop()
                self._playwright = None

            return {"success": True, "message": "Browser closed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a browser action by name.

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        if "open" in action_lower or "goto" in action_lower or "navigate" in action_lower:
            url = kwargs.get("url") or self._extract_url(action)
            if not url:
                return {"success": False, "error": "No URL provided"}
            return self.open(url)

        elif "screenshot" in action_lower or "capture" in action_lower:
            filename = kwargs.get("filename")
            return self.screenshot(filename)

        elif "text" in action_lower or "content" in action_lower:
            return self.get_text()

        elif "links" in action_lower:
            return self.get_links()

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

        elif "google" in action_lower or "search" in action_lower:
            query = kwargs.get("query") or self._extract_query(action)
            if not query:
                return {"success": False, "error": "No search query provided"}
            return self.search_google(query)

        elif "close" in action_lower or "quit" in action_lower:
            return self.close()

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    def _extract_url(self, action: str) -> Optional[str]:
        """Extract URL from action string."""
        import re
        # Look for URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, action)
        if match:
            return match.group()
        # Look for domain-like patterns
        domain_pattern = r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        match = re.search(domain_pattern, action)
        if match:
            return match.group()
        return None

    def _extract_selector(self, action: str) -> Optional[str]:
        """Extract CSS selector from action string."""
        import re
        # Look for quoted selectors
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Look for common selector patterns
        selector_pattern = r'(#[\w-]+|\.[\w-]+|\[[\w-]+=?[^\]]*\]|button|input|a\b)'
        match = re.search(selector_pattern, action)
        if match:
            return match.group()
        return None

    def _extract_query(self, action: str) -> Optional[str]:
        """Extract search query from action string."""
        import re
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]
        # Look for text after "for" or "search"
        patterns = [
            r'(?:search|google)\s+(?:for\s+)?["\']?([^"\']+)',
            r'for\s+["\']?([^"\']+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')
        return None
