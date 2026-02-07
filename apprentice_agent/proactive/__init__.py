"""
Proactive System - Gateway Daemon

Enables AURA to initiate conversations and act proactively.

Architecture:
    Monitors -> EventBus -> SalienceFilter -> GatewayDaemon -> AURA

Components:
- EventBus: Pub/sub system for events (in-memory or Redis)
- SalienceFilter: Determines which events deserve attention
- ActiveInference: Decision-making using Free Energy Principle
- GatewayDaemon: Orchestrates proactive behavior
"""

from .event_bus import Event, EventBus, EventPriority
from .salience_filter import SalienceFilter, FilteredEvent
from .active_inference import (
    ActiveInferenceEngine,
    ProactiveAction,
    ProactiveDecision,
    BeliefState
)
from .gateway_daemon import (
    GatewayDaemon,
    ProactiveMessage,
    get_gateway_daemon,
    start_gateway_daemon,
    stop_gateway_daemon
)
from .persistence import ProactivePersistence, get_persistence

__all__ = [
    # Event Bus
    "Event",
    "EventBus",
    "EventPriority",
    # Salience Filter
    "SalienceFilter",
    "FilteredEvent",
    # Active Inference
    "ActiveInferenceEngine",
    "ProactiveAction",
    "ProactiveDecision",
    "BeliefState",
    # Gateway Daemon
    "GatewayDaemon",
    "ProactiveMessage",
    "get_gateway_daemon",
    "start_gateway_daemon",
    "stop_gateway_daemon",
    # Persistence
    "ProactivePersistence",
    "get_persistence",
]

