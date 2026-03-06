"""
Multi-User Consciousness Module (ADV-04).

Provides per-user mental models, a layered AI identity, cross-user
learning with privacy, and session management.

Usage:
    from aura.multi_user import get_multi_user_manager

    manager = get_multi_user_manager()
    user_model = manager.get_active_user_model()
"""

from .schemas import TrustLevel, IdentityLayer
from .user_mind_model import UserMindModel
from .identity_core import IdentityCore, get_identity_core
from .manager import MultiUserManager, get_multi_user_manager

__all__ = [
    "MultiUserManager",
    "get_multi_user_manager",
    "UserMindModel",
    "TrustLevel",
    "IdentityLayer",
    "IdentityCore",
    "get_identity_core",
]
