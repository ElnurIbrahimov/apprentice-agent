"""
Mixin classes for TelegramBot.

Each mixin provides a specific set of handler methods that are mixed into
the TelegramBot class via multiple inheritance.
"""
from .commands import CommandsMixin
from .research import ResearchMixin
from .sessions import SessionsMixin
from .media import MediaMixin
from .agent_core import AgentCoreMixin
from .skills import SkillsMixin
from .scheduling import SchedulingMixin
from .location import LocationMixin
from .social import SocialMixin
from .payments import PaymentsMixin
from .misc import MiscMixin

__all__ = [
    "CommandsMixin",
    "ResearchMixin",
    "SessionsMixin",
    "MediaMixin",
    "AgentCoreMixin",
    "SkillsMixin",
    "SchedulingMixin",
    "LocationMixin",
    "SocialMixin",
    "PaymentsMixin",
    "MiscMixin",
]
