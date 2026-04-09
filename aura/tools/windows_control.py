"""Windows UI Control Tool — control any Windows app via MS UI Automation.

Uses pywinauto (UI Automation backend) to find windows, click buttons,
fill text fields, and automate any Windows application without APIs.
Falls back to pyautogui for pixel-level operations.

This is AURA's "Computer Use" capability for Windows — lets AURA act
like a human operating desktop apps (Excel, Outlook, browsers, legacy software).

Config: No tokens needed — works with any running Windows app.
"""

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pywinauto
    from pywinauto import Application, Desktop
    from pywinauto.findwindows import ElementAmbiguousError, ElementNotFoundError
    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to top-left corner to abort
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


def _not_available() -> Dict:
    return {"success": False, "error": "pywinauto not installed. Run: pip install pywinauto"}


class WindowsControlTool:
    """Control Windows apps via UI Automation — click buttons, fill forms, navigate any app."""

    name = "windows_control"
    description = "Control Windows apps via UI Automation — click buttons, fill forms, type text in any app"

    # Strict allowlist of executables that run_app() can launch
    APP_ALLOWLIST = {
        "notepad": "notepad.exe",
        "notepad.exe": "notepad.exe",
        "calculator": "calc.exe",
        "calc.exe": "calc.exe",
        "explorer": "explorer.exe",
        "explorer.exe": "explorer.exe",
        "code": "code",
        "vscode": "code",
        "chrome": "chrome",
        "firefox": "firefox",
        "msedge": "msedge",
        "edge": "msedge",
        "terminal": "wt.exe",
        "wt.exe": "wt.exe",
        "mspaint": "mspaint.exe",
        "mspaint.exe": "mspaint.exe",
        "wordpad": "wordpad.exe",
        "snippingtool": "SnippingTool.exe",
        "taskmgr": "taskmgr.exe",
        "control": "control.exe",
    }

    # ------------------------------------------------------------------ #
    # Window Management
    # ------------------------------------------------------------------ #

    def list_windows(self, visible_only: bool = True) -> Dict:
        """List all open windows.

        Args:
            visible_only: Only include visible windows (default True)
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            result = []
            for w in windows:
                try:
                    title = w.window_text()
                    if not title and visible_only:
                        continue
                    rect = w.rectangle()
                    result.append({
                        "title": title,
                        "class_name": w.class_name(),
                        "handle": w.handle,
                        "visible": w.is_visible(),
                        "width": rect.width(),
                        "height": rect.height(),
                    })
                except Exception:
                    continue
            if visible_only:
                result = [w for w in result if w["visible"] and w["title"]]
            result.sort(key=lambda x: x["title"])
            return {"success": True, "windows": result, "count": len(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def find_window(self, title: str, exact: bool = False) -> Dict:
        """Find a window by title.

        Args:
            title: Window title to search for (partial match by default)
            exact: Require exact title match
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            desktop = Desktop(backend="uia")
            if exact:
                windows = desktop.windows(title=title)
            else:
                windows = desktop.windows(title_re=f".*{re.escape(title)}.*")
            if not windows:
                return {"success": False, "error": f"No window found matching '{title}'"}
            w = windows[0]
            rect = w.rectangle()
            return {
                "success": True,
                "title": w.window_text(),
                "class_name": w.class_name(),
                "handle": w.handle,
                "x": rect.left,
                "y": rect.top,
                "width": rect.width(),
                "height": rect.height(),
                "count_found": len(windows),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_window_elements(self, title: str) -> Dict:
        """Get all interactive UI elements in a window.

        Args:
            title: Window title (partial match)
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            app = Application(backend="uia")
            app.connect(title_re=f".*{re.escape(title)}.*")
            dlg = app.top_window()
            elements = []
            for ctrl in dlg.descendants():
                try:
                    ctrl_text = ctrl.window_text()
                    ctrl_type = ctrl.element_info.control_type
                    if ctrl_text or ctrl_type:
                        elements.append({
                            "text": ctrl_text,
                            "type": str(ctrl_type),
                            "enabled": ctrl.is_enabled(),
                            "visible": ctrl.is_visible(),
                            "auto_id": ctrl.element_info.automation_id or "",
                        })
                except Exception:
                    continue
            return {
                "success": True,
                "window": dlg.window_text(),
                "elements": elements[:50],  # cap at 50
                "total": len(elements),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def click_element(self, window_title: str, element_text: str, double_click: bool = False) -> Dict:
        """Click a UI element in a window by its text label.

        Args:
            window_title: Window title (partial match)
            element_text: Text of the button/element to click
            double_click: Double-click instead of single click
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            app = Application(backend="uia")
            app.connect(title_re=f".*{re.escape(window_title)}.*")
            dlg = app.top_window()
            dlg.set_focus()
            time.sleep(0.2)

            # Try finding by text first
            try:
                ctrl = dlg.child_window(title=element_text, found_index=0)
                if double_click:
                    ctrl.double_click_input()
                else:
                    ctrl.click_input()
                return {"success": True, "clicked": element_text, "window": window_title}
            except ElementNotFoundError:
                pass

            # Try by automation_id
            try:
                ctrl = dlg.child_window(auto_id=element_text)
                if double_click:
                    ctrl.double_click_input()
                else:
                    ctrl.click_input()
                return {"success": True, "clicked": element_text, "window": window_title, "method": "auto_id"}
            except ElementNotFoundError:
                pass

            # Try partial text match
            for ctrl in dlg.descendants():
                try:
                    if element_text.lower() in ctrl.window_text().lower() and ctrl.is_enabled():
                        if double_click:
                            ctrl.double_click_input()
                        else:
                            ctrl.click_input()
                        return {"success": True, "clicked": ctrl.window_text(), "window": window_title, "method": "partial_match"}
                except Exception:
                    continue

            return {"success": False, "error": f"Element '{element_text}' not found in '{window_title}'"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fill_text(self, window_title: str, field_identifier: str, text: str, clear_first: bool = True) -> Dict:
        """Type text into a text field in a window.

        Args:
            window_title: Window title (partial match)
            field_identifier: Text field label, auto_id, or placeholder text
            text: Text to type into the field
            clear_first: Clear field contents before typing
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            app = Application(backend="uia")
            app.connect(title_re=f".*{re.escape(window_title)}.*")
            dlg = app.top_window()
            dlg.set_focus()
            time.sleep(0.2)

            edit = None
            # Try by auto_id
            try:
                edit = dlg.child_window(auto_id=field_identifier, control_type="Edit")
            except ElementNotFoundError:
                pass

            # Try by label text (look for Edit near a label)
            if not edit:
                try:
                    edit = dlg.child_window(title=field_identifier, control_type="Edit")
                except ElementNotFoundError:
                    pass

            # Try any Edit control
            if not edit:
                edits = dlg.descendants(control_type="Edit")
                if edits:
                    edit = edits[0]

            if not edit:
                return {"success": False, "error": f"No text field found matching '{field_identifier}'"}

            edit.click_input()
            if clear_first:
                edit.select()
                time.sleep(0.1)
            edit.type_keys(text, with_spaces=True)
            return {"success": True, "typed": text, "field": field_identifier, "window": window_title}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_text(self, window_title: str, element_identifier: Optional[str] = None) -> Dict:
        """Get text content from a window or specific element.

        Args:
            window_title: Window title (partial match)
            element_identifier: Specific element identifier (optional — gets all text if None)
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            app = Application(backend="uia")
            app.connect(title_re=f".*{re.escape(window_title)}.*")
            dlg = app.top_window()

            if element_identifier:
                try:
                    ctrl = dlg.child_window(auto_id=element_identifier)
                    return {"success": True, "text": ctrl.window_text(), "element": element_identifier}
                except ElementNotFoundError:
                    pass
                # Try by title
                try:
                    ctrl = dlg.child_window(title=element_identifier)
                    return {"success": True, "text": ctrl.window_text(), "element": element_identifier}
                except ElementNotFoundError:
                    return {"success": False, "error": f"Element '{element_identifier}' not found"}
            else:
                # Get all text from window
                texts = []
                for ctrl in dlg.descendants():
                    try:
                        t = ctrl.window_text().strip()
                        if t and len(t) > 1:
                            texts.append(t)
                    except Exception:
                        continue
                return {"success": True, "window": window_title, "texts": list(dict.fromkeys(texts))[:100]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def press_keys(self, window_title: str, keys: str) -> Dict:
        """Send keyboard input to a window.

        Args:
            window_title: Window title (partial match)
            keys: Key sequence e.g. '{ENTER}', '^s' (Ctrl+S), '%{F4}' (Alt+F4)
                  Pywinauto notation: ^ = Ctrl, % = Alt, + = Shift, {KEY} = special key
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()
        try:
            app = Application(backend="uia")
            app.connect(title_re=f".*{re.escape(window_title)}.*")
            dlg = app.top_window()
            dlg.set_focus()
            time.sleep(0.2)
            dlg.type_keys(keys)
            return {"success": True, "keys_sent": keys, "window": window_title}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_app(self, executable: str, args: Optional[str] = None, wait_ready: bool = True) -> Dict:
        """Launch a Windows application from the allowlist.

        Args:
            executable: App name from allowlist (e.g. 'notepad', 'calc.exe', 'chrome')
            args: Command-line arguments (optional)
            wait_ready: Wait for app to be ready before returning
        """
        if not PYWINAUTO_AVAILABLE:
            return _not_available()

        # SECURITY: Only allow apps from the allowlist
        exe_lower = executable.lower().strip()
        if exe_lower not in self.APP_ALLOWLIST:
            allowed = ", ".join(sorted(set(self.APP_ALLOWLIST.values())))
            return {
                "success": False,
                "error": f"App '{executable}' not in allowlist. Allowed: {allowed}"
            }
        safe_executable = self.APP_ALLOWLIST[exe_lower]

        try:
            cmd_parts = [safe_executable]
            if args:
                cmd_parts.extend(args.split())
            # Use subprocess.list2cmdline for safe Windows command quoting
            cmd_str = subprocess.list2cmdline(cmd_parts)
            app = Application(backend="uia")
            app.start(cmd_str)
            if wait_ready:
                try:
                    app.top_window().wait("ready", timeout=10)
                except Exception:
                    pass
            title = ""
            try:
                title = app.top_window().window_text()
            except Exception:
                pass
            return {"success": True, "launched": executable, "window_title": title}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Pixel-level fallbacks (pyautogui)
    # ------------------------------------------------------------------ #

    def screenshot_region(self, x: int, y: int, width: int, height: int, output: Optional[str] = None) -> Dict:
        """Capture a screen region to file.

        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
            output: Output file path (defaults to Desktop)
        """
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui not installed"}
        try:
            from datetime import datetime
            img = pyautogui.screenshot(region=(x, y, width, height))
            out = output or str(Path.home() / "Desktop" / f"capture_{datetime.now().strftime('%H%M%S')}.png")
            img.save(out)
            return {"success": True, "path": out, "region": f"{x},{y} {width}x{height}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def mouse_click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> Dict:
        """Click at specific screen coordinates.

        Args:
            x, y: Screen coordinates
            button: 'left', 'right', or 'middle'
            clicks: Number of clicks (2 for double-click)
        """
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui not installed"}
        try:
            pyautogui.click(x, y, button=button, clicks=clicks, interval=0.1)
            return {"success": True, "clicked": f"({x}, {y})", "button": button, "clicks": clicks}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def type_text(self, text: str, interval: float = 0.02) -> Dict:
        """Type text at current cursor position.

        Args:
            text: Text to type
            interval: Delay between keystrokes in seconds
        """
        if not PYAUTOGUI_AVAILABLE:
            return {"success": False, "error": "pyautogui not installed"}
        try:
            pyautogui.write(text, interval=interval)
            return {"success": True, "typed": text[:50] + ("..." if len(text) > 50 else "")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Execute dispatcher
    # ------------------------------------------------------------------ #

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a Windows control action."""
        a = action.lower().strip()

        if "list" in a and "window" in a:
            return self.list_windows(kwargs.get("visible_only", True))
        if ("find" in a or "search" in a) and "window" in a:
            return self.find_window(kwargs.get("title") or kwargs.get("window") or "", kwargs.get("exact", False))
        if "element" in a and ("list" in a or "get" in a):
            return self.get_window_elements(kwargs.get("title") or kwargs.get("window") or "")
        if "click" in a:
            if kwargs.get("x") is not None:
                return self.mouse_click(kwargs["x"], kwargs["y"], kwargs.get("button", "left"), kwargs.get("clicks", 1))
            return self.click_element(
                kwargs.get("window") or kwargs.get("window_title") or "",
                kwargs.get("element") or kwargs.get("element_text") or "",
                kwargs.get("double_click", False),
            )
        if "fill" in a or "type" in a or "input" in a:
            window = kwargs.get("window") or kwargs.get("window_title") or ""
            if window:
                return self.fill_text(window, kwargs.get("field") or kwargs.get("field_identifier") or "", kwargs.get("text") or "")
            return self.type_text(kwargs.get("text") or "")
        if "text" in a or "read" in a:
            return self.get_text(kwargs.get("window") or kwargs.get("window_title") or "", kwargs.get("element"))
        if "key" in a or "press" in a or "shortcut" in a:
            return self.press_keys(kwargs.get("window") or kwargs.get("window_title") or "", kwargs.get("keys") or "")
        if "run" in a or "launch" in a or "open" in a:
            return self.run_app(kwargs.get("executable") or kwargs.get("app") or "", kwargs.get("args"), kwargs.get("wait_ready", True))
        if "screenshot" in a or "capture" in a:
            return self.screenshot_region(kwargs.get("x", 0), kwargs.get("y", 0), kwargs.get("width", 800), kwargs.get("height", 600), kwargs.get("output"))

        return self.list_windows()
