"""
Backward-compatibility shim.

The Telegram bot has been refactored into the aura.messaging.telegram package.
This module re-exports TelegramBot and the public formatting helpers so that
existing imports continue to work without modification.

    from aura.messaging.telegram_bot import TelegramBot   # still works
    from aura.messaging.telegram_bot import _split_message  # still works
"""
from aura.messaging.telegram.bot import (
    TelegramBot,
    _active_bot_instance,
    _active_event_loop,
    _check_rate_limit,
    _run_telegram_scheduled_task,
    _send_telegram_reminder,
    notify_hand_approval_request,
    notify_hand_result,
)
from aura.messaging.telegram_formatting import (
    format_telegram_response,
)
from aura.messaging.telegram_formatting import (
    split_message as _split_message,
)

__all__ = [
    "TelegramBot",
    "_active_bot_instance",
    "_active_event_loop",
    "_check_rate_limit",
    "_run_telegram_scheduled_task",
    "_send_telegram_reminder",
    "_split_message",
    "format_telegram_response",
    "notify_hand_approval_request",
    "notify_hand_result",
]
