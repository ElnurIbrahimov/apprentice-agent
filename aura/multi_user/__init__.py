"""
Multi-User Consciousness Module (ADV-04).

Provides per-user mental models, a layered AI identity, cross-user
learning with privacy, and session management.

Usage:
    from aura.multi_user import get_multi_user_manager

    manager = get_multi_user_manager()
    user_model = manager.get_active_user_model()
"""

from .identity_core import IdentityCore, get_identity_core
from .manager import MultiUserManager, get_multi_user_manager
from .schemas import IdentityLayer, TrustLevel
from .user_mind_model import UserMindModel

__all__ = [
    "IdentityCore",
    "IdentityLayer",
    "MultiUserManager",
    "TrustLevel",
    "UserMindModel",
    "get_identity_core",
    "get_multi_user_manager",
]
