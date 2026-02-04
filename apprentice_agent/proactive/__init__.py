"""
Proactive System for AURA - Gateway Daemon and Background Monitoring.

The proactive system enables AURA to:
- Monitor the user's context (screen, calendar, system)
- Make proactive decisions about when to intervene
- Send timely notifications and suggestions
- Balance helpfulness with not being annoying

Architecture:
    Monitors -> EventBus -> SalienceFilter -> GatewayDaemon -> AURA
                                                   |
                                        ActiveInferenceEngine

Components:
- EventBus: Pub/Sub system for events (Redis or in-memory)
- SalienceFilter: Filters events by relevance and importance
- ActiveInferenceEngine: Makes proactive decisions using Free Energy Principle
- GatewayDaemon: Central coordinator for proactive behavior
- Monitors: Watch various sources (screen, calendar, system)
"""

# Legacy heartbeat (still used for basic notifications)
from .heartbeat import HeartbeatMonitor, Notification

# Event Bus
from .event_bus import (
    Event,
    EventPriority,
    EventBus,
    create_calendar_event,
    create_screen_event,
    create_system_event,
)

# Salience Filter
from .salience_filter import (
    SalienceFilter,
    SalienceWeights,
    FilteredEvent,
)

# Active Inference
from .active_inference import (
    ActiveInferenceEngine,
    ProactiveAction,
    ProactiveDecision,
    BeliefState,
)

# Gateway Daemon
from .gateway_daemon import (
    GatewayDaemon,
    DaemonState,
    UserContext,
    ProactiveMessage,
    get_gateway_daemon,
    start_gateway_daemon,
    stop_gateway_daemon,
)

# Monitors
from .monitors import (
    BaseMonitor,
    MonitorState,
    ScreenMonitor,
    CalendarMonitor,
    SystemMonitor,
)

__all__ = [
    # Legacy
    "HeartbeatMonitor",
    "Notification",
    # Event Bus
    "Event",
    "EventPriority",
    "EventBus",
    "create_calendar_event",
    "create_screen_event",
    "create_system_event",
    # Salience
    "SalienceFilter",
    "SalienceWeights",
    "FilteredEvent",
    # Active Inference
    "ActiveInferenceEngine",
    "ProactiveAction",
    "ProactiveDecision",
    "BeliefState",
    # Gateway Daemon
    "GatewayDaemon",
    "DaemonState",
    "UserContext",
    "ProactiveMessage",
    "get_gateway_daemon",
    "start_gateway_daemon",
    "stop_gateway_daemon",
    # Monitors
    "BaseMonitor",
    "MonitorState",
    "ScreenMonitor",
    "CalendarMonitor",
    "SystemMonitor",
]
