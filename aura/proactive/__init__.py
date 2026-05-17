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
- MotivationAccumulator: 5-factor scoring + threshold learning (Phase 4.2)
- CuriosityScanner: KG gap detection + natural questions (Phase 4.3)
"""

from .active_inference import ActiveInferenceEngine, BeliefState, ProactiveAction, ProactiveDecision
from .curiosity_scanner import (
    CuriosityScanner,
    CuriosityTarget,
    GapType,
    get_curiosity_scanner,
)
from .event_bus import Event, EventBus, EventPriority
from .gateway_daemon import (
    GatewayDaemon,
    ProactiveMessage,
    get_gateway_daemon,
    start_gateway_daemon,
    stop_gateway_daemon,
)
from .motivation_accumulator import (
    MotivationAccumulator,
    PotentialMessage,
    get_motivation_accumulator,
)
from .persistence import ProactivePersistence, get_persistence
from .salience_filter import FilteredEvent, SalienceFilter

__all__ = [
    # Active Inference
    "ActiveInferenceEngine",
    "BeliefState",
    # Phase 4.3: Curiosity Scanner
    "CuriosityScanner",
    "CuriosityTarget",
    # Event Bus
    "Event",
    "EventBus",
    "EventPriority",
    "FilteredEvent",
    "GapType",
    # Gateway Daemon
    "GatewayDaemon",
    # Phase 4.2: Motivation Accumulator
    "MotivationAccumulator",
    "PotentialMessage",
    "ProactiveAction",
    "ProactiveDecision",
    "ProactiveMessage",
    # Persistence
    "ProactivePersistence",
    # Salience Filter
    "SalienceFilter",
    "get_curiosity_scanner",
    "get_gateway_daemon",
    "get_motivation_accumulator",
    "get_persistence",
    "start_gateway_daemon",
    "stop_gateway_daemon",
]

