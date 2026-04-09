"""
Mixin classes for TelegramBot.

Each mixin provides a specific set of handler methods that are mixed into
the TelegramBot class via multiple inheritance.
"""
from .agent_core import AgentCoreMixin
from .commands import CommandsMixin
from .location import LocationMixin
from .media import MediaMixin
from .misc import MiscMixin
from .payments import PaymentsMixin
from .research import ResearchMixin
from .scheduling import SchedulingMixin
from .sessions import SessionsMixin
from .skills import SkillsMixin
from .social import SocialMixin

__all__ = [
    "AgentCoreMixin",
    "CommandsMixin",
    "LocationMixin",
    "MediaMixin",
    "MiscMixin",
    "PaymentsMixin",
    "ResearchMixin",
    "SchedulingMixin",
    "SessionsMixin",
    "SkillsMixin",
    "SocialMixin",
]
