"""Screenshot tool for capturing screen images."""

import mss
import mss.tools
from pathlib import Path
from datetime import datetime
from typing import Optional


class ScreenshotTool:
    """Tool for capturing screenshots."""

    def __init__(self, output_dir: str = "screenshots"):
        """Initialize screenshot tool.

        Args:
            output_dir: Directory to save screenshots (relative to project root)
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, prefix: str = "screenshot") -> str:
        """Generate a unique filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.png"

    def take_screenshot(self, monitor: int = 0) -> dict:
        """Capture full screen screenshot.

        Args:
            monitor: Monitor index (0 = all monitors, 1 = first, 2 = second, etc.)

        Returns:
            dict with success status, path, and dimensions
        """
        try:
            with mss.mss() as sct:
                # Get monitor info
                if monitor == 0:
                    # Capture all monitors combined
                    mon = sct.monitors[0]
                elif monitor < len(sct.monitors):
                    mon = sct.monitors[monitor]
                else:
                    return {
                        "success": False,
                        "error": f"Monitor {monitor} not found. Available: {len(sct.monitors) - 1}"
                    }

                # Capture screenshot
                screenshot = sct.grab(mon)

                # Save to file
                filename = self._generate_filename("screenshot")
                filepath = self.output_dir / filename
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filepath))

                return {
                    "success": True,
                    "path": str(filepath.absolute()),
                    "filename": filename,
                    "width": screenshot.width,
                    "height": screenshot.height,
                    "message": f"Screenshot saved to {filepath}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to capture screenshot: {str(e)}"
            }

    def take_screenshot_region(self, x: int, y: int, width: int, height: int) -> dict:
        """Capture a specific region of the screen.

        Args:
            x: Left position
            y: Top position
            width: Width of region
            height: Height of region

        Returns:
            dict with success status, path, and dimensions
        """
        try:
            with mss.mss() as sct:
                # Define region
                region = {
                    "left": x,
                    "top": y,
                    "width": width,
                    "height": height
                }

                # Capture screenshot
                screenshot = sct.grab(region)

                # Save to file
                filename = self._generate_filename("region")
                filepath = self.output_dir / filename
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filepath))

                return {
                    "success": True,
                    "path": str(filepath.absolute()),
                    "filename": filename,
                    "width": screenshot.width,
                    "height": screenshot.height,
                    "region": {"x": x, "y": y, "width": width, "height": height},
                    "message": f"Region screenshot saved to {filepath}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to capture region: {str(e)}"
            }

    def list_monitors(self) -> dict:
        """List available monitors and their dimensions.

        Returns:
            dict with monitor information
        """
        try:
            with mss.mss() as sct:
                monitors = []
                for i, mon in enumerate(sct.monitors):
                    monitors.append({
                        "index": i,
                        "left": mon["left"],
                        "top": mon["top"],
                        "width": mon["width"],
                        "height": mon["height"],
                        "is_primary": i == 1
                    })

                return {
                    "success": True,
                    "monitors": monitors,
                    "count": len(monitors) - 1  # Exclude combined monitor (index 0)
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list monitors: {str(e)}"
            }

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a screenshot action.

        Args:
            action: Action to perform (screenshot, region, monitors)
            **kwargs: Additional arguments for the action

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        if "region" in action_lower:
            # Extract region coordinates if provided
            x = kwargs.get("x", 0)
            y = kwargs.get("y", 0)
            width = kwargs.get("width", 800)
            height = kwargs.get("height", 600)
            return self.take_screenshot_region(x, y, width, height)

        elif "monitor" in action_lower or "list" in action_lower:
            return self.list_monitors()

        else:
            # Default: full screenshot
            monitor = kwargs.get("monitor", 1)  # Default to primary monitor
            return self.take_screenshot(monitor)
