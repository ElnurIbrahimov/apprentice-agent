"""System control tool for volume, brightness, apps, and system info."""

import subprocess
import ctypes
from typing import Optional


class SystemControlTool:
    """Tool for controlling system settings and launching apps."""

    name = "system_control"
    description = "Control system volume, brightness, launch apps, and get system info"

    # Strict allowlist of apps that can be opened
    APP_ALLOWLIST = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "browser": "start msedge",
        "chrome": "start chrome",
        "firefox": "start firefox",
        "explorer": "explorer.exe",
        "vscode": "code",
        "terminal": "wt.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
    }

    def __init__(self):
        self._volume_interface = None

    def _get_volume_interface(self):
        """Lazy load volume interface."""
        if self._volume_interface is None:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            self._volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        return self._volume_interface

    def get_volume(self) -> dict:
        """Get current system volume level.

        Returns:
            dict with success status and volume level 0-100
        """
        try:
            volume = self._get_volume_interface()
            level = volume.GetMasterVolumeLevelScalar()
            muted = volume.GetMute()

            return {
                "success": True,
                "volume": round(level * 100),
                "muted": bool(muted),
                "message": f"Volume is at {round(level * 100)}%" + (" (muted)" if muted else "")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_volume(self, level: int) -> dict:
        """Set system volume level.

        Args:
            level: Volume level 0-100

        Returns:
            dict with success status
        """
        try:
            level = max(0, min(100, level))
            volume = self._get_volume_interface()
            volume.SetMasterVolumeLevelScalar(level / 100, None)

            return {
                "success": True,
                "volume": level,
                "message": f"Volume set to {level}%"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_brightness(self) -> dict:
        """Get current screen brightness level.

        Returns:
            dict with success status and brightness level 0-100
        """
        try:
            import screen_brightness_control as sbc

            brightness = sbc.get_brightness()
            # Returns list for multiple monitors, get first
            level = brightness[0] if isinstance(brightness, list) else brightness

            return {
                "success": True,
                "brightness": level,
                "message": f"Brightness is at {level}%"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_brightness(self, level: int) -> dict:
        """Set screen brightness level.

        Args:
            level: Brightness level 0-100

        Returns:
            dict with success status
        """
        try:
            import screen_brightness_control as sbc

            level = max(0, min(100, level))
            sbc.set_brightness(level)

            return {
                "success": True,
                "brightness": level,
                "message": f"Brightness set to {level}%"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_app(self, name: str) -> dict:
        """Open an application from the allowlist.

        Args:
            name: Application name (notepad, calculator, browser, explorer, vscode, terminal)

        Returns:
            dict with success status
        """
        name_lower = name.lower().strip()

        # Check allowlist
        if name_lower not in self.APP_ALLOWLIST:
            allowed = ", ".join(self.APP_ALLOWLIST.keys())
            return {
                "success": False,
                "error": f"App '{name}' not in allowlist. Allowed: {allowed}"
            }

        try:
            command = self.APP_ALLOWLIST[name_lower]

            # Use shell=True for 'start' commands, otherwise direct execution
            if command.startswith("start "):
                subprocess.Popen(command, shell=True)
            else:
                subprocess.Popen(command, shell=True)

            return {
                "success": True,
                "app": name_lower,
                "message": f"Opened {name_lower}"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "app": name_lower}

    def get_system_info(self) -> dict:
        """Get system resource usage.

        Returns:
            dict with CPU, RAM, and GPU usage
        """
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # RAM usage
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            ram_used_gb = memory.used / (1024 ** 3)
            ram_total_gb = memory.total / (1024 ** 3)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # GPU info (if available)
            gpu_info = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = {
                        "name": gpu.name,
                        "load": round(gpu.load * 100),
                        "memory_used": round(gpu.memoryUsed),
                        "memory_total": round(gpu.memoryTotal),
                        "temperature": gpu.temperature
                    }
            except:
                pass  # GPUtil not installed or no GPU

            return {
                "success": True,
                "cpu": {
                    "usage_percent": cpu_percent,
                    "cores": cpu_count
                },
                "ram": {
                    "usage_percent": ram_percent,
                    "used_gb": round(ram_used_gb, 1),
                    "total_gb": round(ram_total_gb, 1)
                },
                "disk": {
                    "usage_percent": disk_percent
                },
                "gpu": gpu_info,
                "message": f"CPU: {cpu_percent}%, RAM: {ram_percent}%, Disk: {disk_percent}%"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def lock_screen(self) -> dict:
        """Lock the Windows screen.

        Returns:
            dict with success status
        """
        try:
            ctypes.windll.user32.LockWorkStation()
            return {
                "success": True,
                "message": "Screen locked"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a system control action.

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        if "get" in action_lower and "volume" in action_lower:
            return self.get_volume()
        elif "set" in action_lower and "volume" in action_lower:
            level = kwargs.get("level") or self._extract_number(action)
            if level is None:
                return {"success": False, "error": "No volume level specified"}
            return self.set_volume(int(level))
        elif "get" in action_lower and "brightness" in action_lower:
            return self.get_brightness()
        elif "set" in action_lower and "brightness" in action_lower:
            level = kwargs.get("level") or self._extract_number(action)
            if level is None:
                return {"success": False, "error": "No brightness level specified"}
            return self.set_brightness(int(level))
        elif "open" in action_lower or "launch" in action_lower or "start" in action_lower:
            app_name = kwargs.get("name") or self._extract_app_name(action)
            if not app_name:
                return {"success": False, "error": "No app name specified"}
            return self.open_app(app_name)
        elif "system" in action_lower or "info" in action_lower or "cpu" in action_lower or "ram" in action_lower:
            return self.get_system_info()
        elif "lock" in action_lower:
            return self.lock_screen()
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    def _extract_number(self, action: str) -> Optional[int]:
        """Extract a number from action string."""
        import re
        numbers = re.findall(r'\d+', action)
        if numbers:
            return int(numbers[0])
        return None

    def _extract_app_name(self, action: str) -> Optional[str]:
        """Extract app name from action string."""
        action_lower = action.lower()

        # Check for each app in allowlist
        for app_name in self.APP_ALLOWLIST.keys():
            if app_name in action_lower:
                return app_name

        # Look for quoted text
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        return None
