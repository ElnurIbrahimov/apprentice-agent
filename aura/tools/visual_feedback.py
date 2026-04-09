"""Visual Feedback Loop for UI code generation.

Generates UI code, renders it in a headless browser via Playwright,
screenshots the result, and sends it back to the LLM for iterative
improvement. This is the #1 lever for design quality.

Flow:
  1. User asks "build a pricing page"
  2. LLM generates React/HTML code
  3. Code rendered in headless Chromium (Playwright)
  4. Screenshot taken
  5. Screenshot + code sent back to model: "fix visual issues"
  6. Model iterates (up to max_iterations)
  7. Final code + screenshot returned
"""

import base64
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
TAILWIND_CDN = '<script src="https://cdn.tailwindcss.com"></script>'
REACT_CDN = (
    '<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>\n'
    '<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>\n'
    '<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>'
)
LUCIDE_CDN = '<script src="https://unpkg.com/lucide@latest"></script>'

# Models known to accept images in their message content
VISION_MODELS = {
    "kimi-k2.5:cloud", "kimi-k2.5",
    "chatgpt:gpt-5.4", "chatgpt:gpt-5.4-thinking", "chatgpt:gpt-5.3-codex",
    "chatgpt:gpt-4o",
    "gemini:gemini-2.5-pro", "gemini:gemini-2.0-flash",
    "openai:gpt-4o", "openai:gpt-4-turbo",
    "anthropic:claude-sonnet-4-20250514",
}

SCREENSHOT_DIR = Path(tempfile.gettempdir()) / "aura_visual_feedback"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  HTML wrapping helpers
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> str:
    """Extract code from markdown fenced block (```html ... ``` or ```jsx ... ```)."""
    # Try ```html or ```jsx or ```tsx or bare ``` blocks
    pattern = r"```(?:html|jsx|tsx|react|javascript|js)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fenced block found, return the whole text (might be raw HTML)
    return text.strip()


def _is_full_html(code: str) -> bool:
    """Check if code is a complete HTML document."""
    lower = code.lower()
    return "<!doctype" in lower or "<html" in lower


def _is_react_component(code: str) -> bool:
    """Check if code looks like a React/JSX component."""
    indicators = [
        "function ", "const ", "export default",
        "useState", "useEffect", "className=",
        "React.", "jsx", "</>",
    ]
    return any(ind in code for ind in indicators)


def _wrap_react_component(code: str) -> str:
    """Wrap a React component in an HTML shell with Tailwind + React CDN."""
    # Try to extract the component name
    name_match = re.search(r"(?:function|const)\s+(\w+)", code)
    component_name = name_match.group(1) if name_match else "App"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  {TAILWIND_CDN}
  {REACT_CDN}
  {LUCIDE_CDN}
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    {code}
    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(React.createElement({component_name}));
  </script>
</body>
</html>"""


def _wrap_html_snippet(code: str) -> str:
    """Wrap an HTML snippet in a full document with Tailwind CDN."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  {TAILWIND_CDN}
  {LUCIDE_CDN}
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  </style>
</head>
<body>
  {code}
  <script>lucide.createIcons();</script>
</body>
</html>"""


def _prepare_html(code: str) -> str:
    """Convert any code variant (React, HTML snippet, full HTML) to a renderable document."""
    if _is_full_html(code):
        # Inject Tailwind CDN if missing
        if "tailwindcss" not in code.lower():
            code = code.replace("</head>", f"  {TAILWIND_CDN}\n</head>", 1)
        return code
    if _is_react_component(code):
        return _wrap_react_component(code)
    return _wrap_html_snippet(code)


# ---------------------------------------------------------------------------
#  Playwright rendering
# ---------------------------------------------------------------------------

def _render_and_screenshot(html: str, viewport_width: int = 1280, viewport_height: int = 800) -> dict:
    """Render HTML in headless Chromium and take a screenshot.

    Returns:
        dict with keys: success, path, base64, width, height, console_errors, error
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"success": False, "error": "Playwright not installed. Run: pip install playwright && python -m playwright install chromium"}

    tmp_html = SCREENSHOT_DIR / f"render_{int(time.time() * 1000)}.html"
    screenshot_path = SCREENSHOT_DIR / f"screenshot_{int(time.time() * 1000)}.png"
    console_errors = []

    try:
        tmp_html.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})

            # Capture console errors
            page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type in ("error", "warning") else None)

            # Navigate and wait for rendering
            page.goto(f"file:///{tmp_html.as_posix()}", wait_until="networkidle", timeout=30000)
            # Extra wait for Tailwind CDN + React hydration
            page.wait_for_timeout(2000)

            # Get actual page height for full-page screenshot
            body_height = page.evaluate("document.body.scrollHeight")
            screenshot_height = min(body_height, 4000)  # cap at 4000px

            page.screenshot(
                path=str(screenshot_path),
                full_page=True,
                clip={"x": 0, "y": 0, "width": viewport_width, "height": screenshot_height} if screenshot_height < body_height else None,
            )

            browser.close()

        # Read screenshot as base64 for vision models
        img_data = screenshot_path.read_bytes()
        img_b64 = base64.b64encode(img_data).decode("utf-8")

        return {
            "success": True,
            "path": str(screenshot_path),
            "base64": img_b64,
            "width": viewport_width,
            "height": screenshot_height,
            "console_errors": console_errors,
        }

    except Exception as e:
        logger.error(f"[VisualFeedback] Render/screenshot failed: {e}")
        return {"success": False, "error": str(e), "console_errors": console_errors}

    finally:
        # Clean up temp HTML (keep screenshot for reference)
        try:
            tmp_html.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Main class
# ---------------------------------------------------------------------------

class VisualFeedbackLoop:
    """Generate UI code, render it, screenshot it, iterate for quality.

    Usage:
        vfl = VisualFeedbackLoop(brain)
        result = vfl.generate_with_feedback("build a pricing page", max_iterations=2)
        # result = {code, html, screenshot_path, screenshot_base64, iterations, model_used}
    """

    def __init__(self, brain=None):
        """Initialize with an OllamaBrain instance for LLM calls.

        Args:
            brain: OllamaBrain instance. If None, will try to get one from the agent.
        """
        self._brain = brain

    def _get_brain(self):
        """Lazy-load brain if not provided."""
        if self._brain is not None:
            return self._brain
        try:
            from api.services.agent_service import get_agent_service
            self._brain = get_agent_service().agent.brain
        except Exception:
            pass
        return self._brain

    def _model_supports_vision(self, model: str) -> bool:
        """Check if the current model can accept images."""
        if not model:
            return False
        # Check exact match or prefix match
        if model in VISION_MODELS:
            return True
        for vm in VISION_MODELS:
            if model.startswith(vm.split(":")[0] + ":"):
                # Check if it's a known vision-capable provider
                prefix = model.split(":")[0]
                if prefix in ("kimi", "chatgpt", "openai", "gemini", "anthropic"):
                    return True
        return False

    def _call_llm(self, prompt: str, system_prompt: str | None = None, image_b64: str | None = None) -> tuple:
        """Call the LLM, optionally with an image.

        Returns:
            (response_text, model_used)
        """
        brain = self._get_brain()
        if brain is None:
            raise RuntimeError("No brain available for LLM calls")

        # Determine current model
        model = getattr(brain, '_model_override', None) or "auto"

        # If we have an image and the model supports vision, send multimodal
        if image_b64 and self._model_supports_vision(model):
            return self._call_llm_with_image(brain, prompt, system_prompt, image_b64, model)
        else:
            # Text-only call (or describe the screenshot textually)
            full_prompt = prompt
            if system_prompt:
                response = brain.think(full_prompt, system_prompt=system_prompt, use_history=False, model_override=model if model != "auto" else None)
            else:
                response = brain.think(full_prompt, use_history=False, model_override=model if model != "auto" else None)
            last_model = brain.get_last_model_used() if hasattr(brain, 'get_last_model_used') else model
            return response, last_model

    def _call_llm_with_image(self, brain, prompt: str, system_prompt: str, image_b64: str, model: str) -> tuple:
        """Send prompt + image to a vision-capable model via Ollama-style API."""
        try:
            client, actual_model = brain._get_client_for_model(model)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            })

            response = client.chat(model=actual_model, messages=messages)
            content = response.get("message", {}).get("content", "")
            return content, actual_model
        except Exception as e:
            logger.warning(f"[VisualFeedback] Vision call failed ({e}), falling back to text-only")
            # Fallback: text-only with layout description
            response = brain.think(prompt, system_prompt=system_prompt, use_history=False)
            last_model = brain.get_last_model_used() if hasattr(brain, 'get_last_model_used') else model
            return response, last_model

    def generate_with_feedback(
        self,
        prompt: str,
        max_iterations: int = 2,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        progress_callback=None,
    ) -> dict:
        """Full visual feedback loop: generate -> render -> screenshot -> iterate.

        Args:
            prompt: User's UI request (e.g., "build a pricing page")
            max_iterations: How many review cycles (0 = generate only, no feedback)
            viewport_width: Browser viewport width for rendering
            viewport_height: Browser viewport height for rendering
            progress_callback: Optional callable(event_dict) for streaming progress

        Returns:
            dict with: code, html, screenshot_path, screenshot_base64,
                       iterations, model_used, console_errors, improvements
        """
        def _emit(event_type: str, detail: str):
            if progress_callback:
                try:
                    progress_callback({
                        "type": "tool_trace",
                        "event": event_type,
                        "tool": "visual_feedback",
                        "detail": detail,
                        "timestamp": time.time(),
                    })
                except Exception:
                    pass

        _emit("start", f"Generating UI: {prompt[:60]}")

        # ── Step 1: Initial code generation ──
        system_prompt = (
            "You are an expert frontend developer specializing in beautiful, modern UI design. "
            "Generate complete, production-quality code based on the user's request. "
            "Use Tailwind CSS for styling. Output ONLY code in a ```html or ```jsx fenced block. "
            "No explanations before or after the code block. "
            "Make the design visually polished: proper spacing, shadows, gradients, hover states, "
            "and responsive layout. Use modern design patterns (cards, rounded corners, subtle shadows). "
            "Include realistic placeholder content (not lorem ipsum). "
            "IMPORTANT: Include ALL content in a single file — inline styles/scripts are fine."
        )

        _emit("progress", "Generating initial code...")
        raw_response, model_used = self._call_llm(
            f"Generate a complete UI for: {prompt}",
            system_prompt=system_prompt,
        )

        code = _extract_code_block(raw_response)
        html = _prepare_html(code)
        all_improvements = []

        # If max_iterations is 0, just return the initial code
        if max_iterations == 0:
            return {
                "code": code,
                "html": html,
                "screenshot_path": None,
                "screenshot_base64": None,
                "iterations": 0,
                "model_used": model_used,
                "console_errors": [],
                "improvements": [],
            }

        # ── Steps 2-6: Render, screenshot, iterate ──
        for iteration in range(max_iterations):
            iter_label = f"iteration {iteration + 1}/{max_iterations}"
            _emit("progress", f"Rendering code ({iter_label})...")

            # Render and screenshot
            render_result = _render_and_screenshot(html, viewport_width, viewport_height)

            if not render_result["success"]:
                logger.warning(f"[VisualFeedback] Render failed on {iter_label}: {render_result.get('error')}")
                _emit("error", f"Render failed: {render_result.get('error', 'unknown')}")
                break

            _emit("progress", f"Reviewing screenshot ({iter_label})...")

            # Build review prompt
            console_errors_text = ""
            if render_result.get("console_errors"):
                console_errors_text = "\n\nConsole errors detected:\n" + "\n".join(render_result["console_errors"][:10])

            review_system = (
                "You are an expert UI reviewer. You are looking at a screenshot of a rendered web page. "
                "Your job is to identify visual issues and produce an improved version of the code. "
                "Common issues to check: alignment, spacing, color contrast, missing hover states, "
                "text readability, responsive design, visual hierarchy, overall polish. "
                "If the design looks good, make only minor refinements. "
                "Output the COMPLETE improved code in a ```html or ```jsx fenced block. "
                "Do NOT output partial code or diffs — output the full file."
            )

            # If vision model: send image. Otherwise: describe layout textually.
            if self._model_supports_vision(model_used):
                review_prompt = (
                    f"Here is the code I generated for: \"{prompt}\"\n\n"
                    f"```\n{code}\n```\n\n"
                    f"I've attached a screenshot of how it renders. "
                    f"Review the visual result and fix any issues. "
                    f"Focus on: visual polish, spacing, alignment, readability.{console_errors_text}\n\n"
                    f"Output the complete improved code."
                )
                review_response, model_used = self._call_llm(
                    review_prompt,
                    system_prompt=review_system,
                    image_b64=render_result["base64"],
                )
            else:
                # Text-only: describe the rendering context
                review_prompt = (
                    f"Here is the code I generated for: \"{prompt}\"\n\n"
                    f"```\n{code}\n```\n\n"
                    f"The page rendered at {render_result['width']}x{render_result['height']}px. "
                    f"{console_errors_text}\n\n"
                    f"Review this code carefully for visual issues: "
                    f"spacing, alignment, color contrast, text readability, responsiveness, visual hierarchy. "
                    f"Output the complete improved code."
                )
                review_response, model_used = self._call_llm(
                    review_prompt,
                    system_prompt=review_system,
                )

            # Extract improved code
            improved_code = _extract_code_block(review_response)

            # Check if the model actually changed anything
            if improved_code.strip() == code.strip():
                _emit("progress", f"No changes needed ({iter_label})")
                all_improvements.append(f"Iteration {iteration + 1}: No changes — design approved")
                # Keep the last successful screenshot
                code = improved_code
                html = _prepare_html(code)
                break
            else:
                # Summarize what changed (brief diff description)
                len_diff = len(improved_code) - len(code)
                change_desc = f"Iteration {iteration + 1}: Code {'expanded' if len_diff > 0 else 'refined'} by {abs(len_diff)} chars"
                all_improvements.append(change_desc)
                _emit("progress", change_desc)

                code = improved_code
                html = _prepare_html(code)

        # ── Final render for the screenshot to return ──
        _emit("progress", "Final render...")
        final_render = _render_and_screenshot(html, viewport_width, viewport_height)

        _emit("done", f"Complete — {len(all_improvements)} iteration(s)")

        return {
            "code": code,
            "html": html,
            "screenshot_path": final_render.get("path") if final_render["success"] else None,
            "screenshot_base64": final_render.get("base64") if final_render["success"] else None,
            "iterations": len(all_improvements),
            "model_used": model_used,
            "console_errors": final_render.get("console_errors", []),
            "improvements": all_improvements,
        }

    def generate_stream(
        self,
        prompt: str,
        max_iterations: int = 2,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ):
        """Streaming version — yields progress events + final result.

        Yields dicts compatible with agent_service chat_stream format:
            {"type": "tool_trace", "event": ..., "tool": "visual_feedback", ...}
            {"type": "chunk", "content": "..."}  (for the final code)
        """
        events = []

        def collect_event(evt):
            events.append(evt)

        # Emit a starting message
        yield {
            "type": "tool_trace",
            "event": "start",
            "tool": "visual_feedback",
            "detail": f'Visual feedback loop for "{prompt[:50]}"',
            "timestamp": time.time(),
        }
        yield {"type": "tool_status", "tool_name": "visual_feedback", "tool_action": "generating UI"}

        result = self.generate_with_feedback(
            prompt=prompt,
            max_iterations=max_iterations,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            progress_callback=collect_event,
        )

        # Yield collected progress events
        for evt in events:
            yield evt

        # Yield the final code as chat content
        code = result.get("code", "")
        iterations = result.get("iterations", 0)
        improvements = result.get("improvements", [])
        model_used = result.get("model_used", "unknown")

        # Build the response message
        response_parts = []
        if improvements:
            response_parts.append(f"**Visual Feedback Loop** completed {iterations} iteration(s) using `{model_used}`:")
            for imp in improvements:
                response_parts.append(f"- {imp}")
            response_parts.append("")

        # Wrap code in a fenced block
        if _is_react_component(code):
            response_parts.append(f"```jsx\n{code}\n```")
        else:
            response_parts.append(f"```html\n{code}\n```")

        # Add screenshot info
        if result.get("screenshot_path"):
            response_parts.append(f"\n*Screenshot saved to: `{result['screenshot_path']}`*")

        if result.get("console_errors"):
            response_parts.append(f"\n**Console warnings:** {len(result['console_errors'])}")
            for err in result["console_errors"][:5]:
                response_parts.append(f"- `{err}`")

        yield {"type": "chunk", "content": "\n".join(response_parts)}
        yield {"type": "tool_status", "tool_name": "", "tool_action": ""}  # clear status

        # Yield screenshot as an artifact if available
        if result.get("screenshot_base64"):
            yield {
                "type": "visual_feedback_screenshot",
                "screenshot_base64": result["screenshot_base64"],
                "screenshot_path": result.get("screenshot_path", ""),
                "width": 1280,
                "height": 800,
            }

        yield {
            "type": "tool_trace",
            "event": "done",
            "tool": "visual_feedback",
            "detail": f'{iterations} iteration(s), model: {model_used}',
            "elapsed_ms": 0,
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
#  Module-level convenience
# ---------------------------------------------------------------------------
_instance: Optional[VisualFeedbackLoop] = None


def get_visual_feedback(brain=None) -> VisualFeedbackLoop:
    """Get or create a singleton VisualFeedbackLoop."""
    global _instance
    if _instance is None or (brain is not None and _instance._brain is None):
        _instance = VisualFeedbackLoop(brain=brain)
    return _instance
