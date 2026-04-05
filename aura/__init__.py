"""AURA - Autonomous Universal Reasoning Agent."""

from aura._version import __version__
__all__ = ["ApprenticeAgent", "OllamaBrain"]


def __getattr__(name):
    if name == "ApprenticeAgent":
        from .agent import ApprenticeAgent
        return ApprenticeAgent
    if name == "OllamaBrain":
        from .brain import OllamaBrain
        return OllamaBrain
    raise AttributeError(f"module 'aura' has no attribute {name!r}")
