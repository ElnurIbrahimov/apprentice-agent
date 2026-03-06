"""Screen Reader tool for monitoring screen content, OCR text extraction, and change detection."""

import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

from aura.tools._shared_models import get_florence2


def _load_florence2():
    """Load Florence-2 via shared singleton."""
    model, processor = get_florence2()
    if model is None:
        raise RuntimeError("Florence-2 unavailable (disabled or failed to load)")
    return model, processor


@dataclass
class ScreenState:
    """Represents a snapshot of the screen state."""
    text: str                             # OCR'd text
    active_window: str = ""               # Window title
    active_process: str = ""              # Process name
    timestamp: str = ""                   # ISO timestamp
    screenshot_path: str = ""             # Saved screenshot path
    regions: List[dict] = field(default_factory=list)  # Detected text regions

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ChangeDetector:
    """Detects significant visual changes between screen frames."""

    def __init__(self, threshold: float = 0.05):
        self._last_frame_hash: Optional[str] = None
        self._last_frame = None
        self.threshold = threshold

    def has_changed(self, current_frame) -> Tuple[bool, float]:
        """Compare current frame against the last one.

        Returns (changed, difference_ratio).
        """
        try:
            from PIL import Image, ImageChops
            import numpy as np
        except ImportError:
            # Fallback: hash-based comparison
            return self._hash_compare(current_frame)

        if self._last_frame is None:
            self._last_frame = current_frame
            return False, 0.0

        try:
            # Ensure both frames are PIL Images
            if not isinstance(current_frame, Image.Image):
                current_frame = Image.frombytes("RGB", current_frame.size, current_frame.rgb)
            if not isinstance(self._last_frame, Image.Image):
                self._last_frame = Image.frombytes("RGB", self._last_frame.size, self._last_frame.rgb)

            # Resize to same dimensions if needed
            if current_frame.size != self._last_frame.size:
                current_frame = current_frame.resize(self._last_frame.size)

            diff = ImageChops.difference(current_frame, self._last_frame)
            diff_array = np.array(diff)
            diff_ratio = float(np.mean(diff_array > 30) if diff_array.size > 0 else 0.0)

            changed = diff_ratio > self.threshold
            self._last_frame = current_frame
            return changed, round(diff_ratio, 4)

        except Exception as e:
            logger.debug(f"[ScreenReader] Change detection error: {e}")
            self._last_frame = current_frame
            return True, 1.0

    def _hash_compare(self, frame) -> Tuple[bool, float]:
        """Simple hash-based comparison fallback."""
        try:
            if hasattr(frame, 'rgb'):
                frame_hash = hashlib.md5(frame.rgb).hexdigest()
            elif hasattr(frame, 'tobytes'):
                frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            else:
                frame_hash = hashlib.md5(str(frame).encode()).hexdigest()
        except Exception:
            return True, 1.0

        if self._last_frame_hash is None:
            self._last_frame_hash = frame_hash
            return False, 0.0

        changed = frame_hash != self._last_frame_hash
        self._last_frame_hash = frame_hash
        return changed, 1.0 if changed else 0.0


class ScreenReaderTool:
    """Monitor screen for changes, extract text via OCR, detect active application."""

    name = "screen_reader"
    description = "Monitor screen for changes, extract text, detect active application"

    def __init__(self):
        self._change_detector = ChangeDetector()
        self._screenshot_dir = Path(__file__).parent.parent.parent / "screenshots"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _capture_screen(self, region: dict = None):
        """Capture the screen (or region) using mss. Returns (screenshot, path)."""
        try:
            import mss
            import mss.tools
        except ImportError:
            return None, None

        try:
            with mss.mss() as sct:
                if region:
                    monitor = {
                        "left": region.get("x", 0),
                        "top": region.get("y", 0),
                        "width": region.get("width", 800),
                        "height": region.get("height", 600),
                    }
                else:
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

                screenshot = sct.grab(monitor)

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screen_read_{timestamp}.png"
                filepath = self._screenshot_dir / filename
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filepath))

                return screenshot, str(filepath)
        except Exception as e:
            logger.error(f"[ScreenReader] Capture failed: {e}")
            return None, None

    def _ocr_image(self, image_path: str) -> str:
        """Extract text from an image using Florence-2 OCR."""
        try:
            model, processor = _load_florence2()
        except RuntimeError as e:
            logger.warning(f"[ScreenReader] Florence-2 not available: {e}")
            return self._fallback_ocr(image_path)

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            device = next(model.parameters()).device
            inputs = processor(text="<OCR>", images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Clean up Florence-2 OCR output
            text = text.replace("<OCR>", "").strip()
            return text

        except Exception as e:
            logger.error(f"[ScreenReader] OCR failed: {e}")
            return self._fallback_ocr(image_path)

    def _fallback_ocr(self, image_path: str) -> str:
        """Fallback OCR using pytesseract if available."""
        try:
            import pytesseract
            from PIL import Image
            image = Image.open(image_path)
            return pytesseract.image_to_string(image)
        except ImportError:
            return "[OCR unavailable: install pytesseract or ensure Florence-2 is available]"
        except Exception as e:
            return f"[OCR error: {e}]"

    def read_screen(self) -> dict:
        """Capture and OCR the current screen."""
        screenshot, path = self._capture_screen()
        if not screenshot:
            return {"success": False, "error": "Failed to capture screen"}

        text = self._ocr_image(path) if path else ""
        window_info = self.get_active_window()

        state = ScreenState(
            text=text,
            active_window=window_info.get("title", ""),
            active_process=window_info.get("process", ""),
            screenshot_path=path or "",
        )

        return {
            "success": True,
            "text": state.text,
            "active_window": state.active_window,
            "active_process": state.active_process,
            "screenshot_path": state.screenshot_path,
            "timestamp": state.timestamp,
            "response": f"Screen text ({len(state.text)} chars):\n{state.text[:500]}"
                        + (f"\n[Active: {state.active_window}]" if state.active_window else "")
        }

    def read_region(self, x: int, y: int, width: int, height: int) -> dict:
        """OCR a specific screen region."""
        screenshot, path = self._capture_screen(region={"x": x, "y": y, "width": width, "height": height})
        if not screenshot:
            return {"success": False, "error": "Failed to capture region"}

        text = self._ocr_image(path) if path else ""

        return {
            "success": True,
            "text": text,
            "region": {"x": x, "y": y, "width": width, "height": height},
            "screenshot_path": path or "",
            "response": f"Region text:\n{text[:500]}"
        }

    def detect_changes(self, interval_s: float = 2.0, duration_s: float = 10.0) -> dict:
        """Monitor screen for visual changes over a time period."""
        changes = []
        start = time.time()
        checks = 0

        while time.time() - start < duration_s:
            screenshot, path = self._capture_screen()
            if screenshot:
                changed, diff_ratio = self._change_detector.has_changed(screenshot)
                checks += 1
                if changed:
                    text = self._ocr_image(path) if path else ""
                    changes.append({
                        "timestamp": datetime.now().isoformat(),
                        "diff_ratio": diff_ratio,
                        "text_snippet": text[:200],
                        "screenshot_path": path or "",
                    })
            time.sleep(interval_s)

        return {
            "success": True,
            "changes_detected": len(changes),
            "total_checks": checks,
            "duration": round(time.time() - start, 1),
            "changes": changes,
            "response": f"Detected {len(changes)} change(s) in {checks} checks over {duration_s}s"
        }

    def get_active_window(self) -> dict:
        """Return the active window title and process name."""
        title = ""
        process = ""

        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                length = user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value

                # Get process name
                import ctypes.wintypes
                pid = ctypes.wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                try:
                    import psutil
                    proc = psutil.Process(pid.value)
                    process = proc.name()
                except (ImportError, Exception):
                    process = f"PID:{pid.value}"

            except Exception as e:
                logger.debug(f"[ScreenReader] Win32 active window error: {e}")

        elif sys.platform == "linux":
            try:
                import subprocess
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    title = result.stdout.strip()

                result2 = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowpid"],
                    capture_output=True, text=True, timeout=5
                )
                if result2.returncode == 0:
                    pid = int(result2.stdout.strip())
                    try:
                        import psutil
                        proc = psutil.Process(pid)
                        process = proc.name()
                    except (ImportError, Exception):
                        process = f"PID:{pid}"
            except (FileNotFoundError, Exception) as e:
                logger.debug(f"[ScreenReader] Linux active window error: {e}")

        elif sys.platform == "darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to get name of first process whose frontmost is true'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    process = result.stdout.strip()
                    title = process
            except Exception as e:
                logger.debug(f"[ScreenReader] macOS active window error: {e}")

        return {
            "success": True,
            "title": title,
            "process": process,
            "response": f"Active window: {title}" + (f" ({process})" if process else "")
        }

    def watch(self, keywords: List[str] = None, interval_s: float = 3.0,
              duration_s: float = 30.0) -> dict:
        """Watch screen for keyword appearance."""
        if not keywords:
            return {"success": False, "error": "No keywords specified to watch for"}

        keyword_lower = [k.lower() for k in keywords]
        found = []
        start = time.time()
        checks = 0

        while time.time() - start < duration_s:
            screenshot, path = self._capture_screen()
            if screenshot and path:
                text = self._ocr_image(path).lower()
                checks += 1

                for kw in keyword_lower:
                    if kw in text:
                        found.append({
                            "keyword": kw,
                            "timestamp": datetime.now().isoformat(),
                            "screenshot_path": path,
                        })

                if found:
                    break

            time.sleep(interval_s)

        return {
            "success": True,
            "found": len(found) > 0,
            "matches": found,
            "checks": checks,
            "duration": round(time.time() - start, 1),
            "response": (f"Found keyword(s): {', '.join(m['keyword'] for m in found)}"
                         if found else f"No keywords found after {checks} checks")
        }

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a screen reader action."""
        action_lower = action.lower().strip()

        # Read screen
        if action_lower in ("read", "read_screen", "capture", "ocr", "screen"):
            return self.read_screen()

        # Read region
        if action_lower.startswith("read_region") or action_lower.startswith("region"):
            x = kwargs.get("x", 0)
            y = kwargs.get("y", 0)
            width = kwargs.get("width", 800)
            height = kwargs.get("height", 600)
            # Try to parse from action: "read_region x y w h"
            parts = re.findall(r'\d+', action)
            if len(parts) >= 4:
                x, y, width, height = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return self.read_region(x, y, width, height)

        # Active window
        if "active" in action_lower or "window" in action_lower or "focus" in action_lower:
            return self.get_active_window()

        # Detect changes
        if "change" in action_lower or "monitor" in action_lower or "detect" in action_lower:
            interval = kwargs.get("interval_s", 2.0)
            duration = kwargs.get("duration_s", 10.0)
            return self.detect_changes(interval_s=interval, duration_s=duration)

        # Watch for keywords
        if "watch" in action_lower:
            keywords_str = action.replace("watch", "").strip()
            keywords = kwargs.get("keywords") or [k.strip() for k in keywords_str.split(",") if k.strip()]
            interval = kwargs.get("interval_s", 3.0)
            duration = kwargs.get("duration_s", 30.0)
            return self.watch(keywords=keywords, interval_s=interval, duration_s=duration)

        # Default: read screen
        return self.read_screen()


# Singleton
screen_reader_tool = ScreenReaderTool()
