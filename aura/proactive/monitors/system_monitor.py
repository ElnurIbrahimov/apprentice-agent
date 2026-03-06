"""
System Monitor - Monitors system-level events.

Events generated:
- battery_low: Battery level is low
- disk_space_low: Disk space is running low
- memory_high: Memory usage is high
- cpu_high: CPU usage is high
- network_change: Network connectivity changed
- system_alert: General system alert
"""

import asyncio
import logging
import platform
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base_monitor import BaseMonitor
from ..event_bus import Event, EventPriority

logger = logging.getLogger(__name__)

# Try to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("[SystemMonitor] psutil not available - limited functionality")


class SystemMonitor(BaseMonitor):
    """
    Monitor for system-level events.

    Features:
    - Battery level monitoring
    - Disk space monitoring
    - Memory usage monitoring
    - CPU usage monitoring
    - Network connectivity monitoring
    """

    def __init__(
        self,
        event_bus=None,
        poll_interval: float = 30.0,  # Check every 30 seconds
        battery_threshold: int = 20,  # Percent
        disk_threshold: int = 90,     # Percent used
        memory_threshold: int = 85,   # Percent used
        cpu_threshold: int = 90       # Percent sustained
    ):
        """
        Initialize system monitor.

        Args:
            event_bus: EventBus to publish to
            poll_interval: Seconds between polls
            battery_threshold: Battery percent to trigger low warning
            disk_threshold: Disk usage percent to trigger warning
            memory_threshold: Memory usage percent to trigger warning
            cpu_threshold: CPU usage percent to trigger warning
        """
        super().__init__(event_bus, poll_interval)

        self._battery_threshold = battery_threshold
        self._disk_threshold = disk_threshold
        self._memory_threshold = memory_threshold
        self._cpu_threshold = cpu_threshold

        # State tracking to avoid repeated alerts
        self._battery_warned = False
        self._disk_warned: Dict[str, bool] = {}
        self._memory_warned = False
        self._cpu_high_count = 0  # Sustained high CPU counter
        self._last_network_status: Optional[bool] = None

        # CPU history for sustained detection
        self._cpu_history: List[float] = []

        logger.info(f"[SystemMonitor] Initialized (psutil={PSUTIL_AVAILABLE})")

    @property
    def source(self) -> str:
        return "system"

    async def _poll(self) -> List[Event]:
        """Poll for system events."""
        if not PSUTIL_AVAILABLE:
            return []

        events = []

        # Check battery
        battery_event = self._check_battery()
        if battery_event:
            events.append(battery_event)

        # Check disk space
        disk_events = self._check_disk_space()
        events.extend(disk_events)

        # Check memory
        memory_event = self._check_memory()
        if memory_event:
            events.append(memory_event)

        # Check CPU (sustained high usage)
        cpu_event = self._check_cpu()
        if cpu_event:
            events.append(cpu_event)

        # Check network
        network_event = self._check_network()
        if network_event:
            events.append(network_event)

        return events

    def _check_battery(self) -> Optional[Event]:
        """Check battery level."""
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return None  # No battery (desktop)

            percent = battery.percent
            plugged = battery.power_plugged

            # Only warn if not plugged in and below threshold
            if not plugged and percent <= self._battery_threshold:
                if not self._battery_warned:
                    self._battery_warned = True
                    return self.create_event(
                        "battery_low",
                        {
                            "percent": percent,
                            "plugged": plugged,
                            "time_remaining": battery.secsleft if battery.secsleft != -1 else None
                        },
                        priority=EventPriority.HIGH if percent <= 10 else EventPriority.MEDIUM
                    )
            else:
                self._battery_warned = False

        except Exception as e:
            logger.debug(f"[SystemMonitor] Battery check failed: {e}")

        return None

    def _check_disk_space(self) -> List[Event]:
        """Check disk space on all partitions."""
        events = []

        try:
            partitions = psutil.disk_partitions()

            for partition in partitions:
                try:
                    # Skip removable drives
                    if "removable" in partition.opts.lower():
                        continue

                    usage = psutil.disk_usage(partition.mountpoint)
                    percent_used = usage.percent

                    if percent_used >= self._disk_threshold:
                        if not self._disk_warned.get(partition.device, False):
                            self._disk_warned[partition.device] = True
                            events.append(self.create_event(
                                "disk_space_low",
                                {
                                    "device": partition.device,
                                    "mountpoint": partition.mountpoint,
                                    "percent_used": percent_used,
                                    "free_gb": round(usage.free / (1024**3), 2),
                                    "total_gb": round(usage.total / (1024**3), 2)
                                },
                                priority=EventPriority.MEDIUM
                            ))
                    else:
                        self._disk_warned[partition.device] = False

                except (PermissionError, OSError):
                    continue

        except Exception as e:
            logger.debug(f"[SystemMonitor] Disk check failed: {e}")

        return events

    def _check_memory(self) -> Optional[Event]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent

            if percent_used >= self._memory_threshold:
                if not self._memory_warned:
                    self._memory_warned = True
                    return self.create_event(
                        "memory_high",
                        {
                            "percent_used": percent_used,
                            "available_gb": round(memory.available / (1024**3), 2),
                            "total_gb": round(memory.total / (1024**3), 2)
                        },
                        priority=EventPriority.MEDIUM
                    )
            else:
                self._memory_warned = False

        except Exception as e:
            logger.debug(f"[SystemMonitor] Memory check failed: {e}")

        return None

    def _check_cpu(self) -> Optional[Event]:
        """Check CPU usage (sustained high)."""
        try:
            # Get CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)

            # Track history
            self._cpu_history.append(cpu_percent)
            if len(self._cpu_history) > 5:
                self._cpu_history.pop(0)

            # Check for sustained high CPU (3+ consecutive high readings)
            if len(self._cpu_history) >= 3:
                recent = self._cpu_history[-3:]
                if all(c >= self._cpu_threshold for c in recent):
                    self._cpu_high_count += 1

                    # Only alert once per sustained period
                    if self._cpu_high_count == 1:
                        return self.create_event(
                            "cpu_high",
                            {
                                "percent": cpu_percent,
                                "sustained_readings": len(recent),
                                "average": round(sum(recent) / len(recent), 1)
                            },
                            priority=EventPriority.LOW
                        )
                else:
                    self._cpu_high_count = 0

        except Exception as e:
            logger.debug(f"[SystemMonitor] CPU check failed: {e}")

        return None

    def _check_network(self) -> Optional[Event]:
        """Check network connectivity."""
        try:
            # Check if we have any network interfaces up
            stats = psutil.net_if_stats()
            connected = any(
                s.isup for name, s in stats.items()
                if not name.startswith("lo")  # Skip loopback
            )

            # Check for change
            if self._last_network_status is not None:
                if connected != self._last_network_status:
                    self._last_network_status = connected
                    return self.create_event(
                        "network_change",
                        {
                            "connected": connected,
                            "status": "connected" if connected else "disconnected"
                        },
                        priority=EventPriority.MEDIUM
                    )

            self._last_network_status = connected

        except Exception as e:
            logger.debug(f"[SystemMonitor] Network check failed: {e}")

        return None

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information.

        Returns:
            Dict with system metrics
        """
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "hostname": platform.node(),
        }

        if PSUTIL_AVAILABLE:
            try:
                info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                info["cpu_count"] = psutil.cpu_count()

                memory = psutil.virtual_memory()
                info["memory_percent"] = memory.percent
                info["memory_available_gb"] = round(memory.available / (1024**3), 2)

                battery = psutil.sensors_battery()
                if battery:
                    info["battery_percent"] = battery.percent
                    info["battery_plugged"] = battery.power_plugged

            except Exception as e:
                logger.debug(f"[SystemMonitor] System info error: {e}")

        return info
