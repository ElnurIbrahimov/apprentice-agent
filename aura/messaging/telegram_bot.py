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
    _check_rate_limit,
    _send_telegram_reminder,
    _run_telegram_scheduled_task,
    notify_hand_result,
    notify_hand_approval_request,
    _active_bot_instance,
    _active_event_loop,
)
from aura.messaging.telegram_formatting import (
    format_telegram_response,
    split_message as _split_message,
)

__all__ = [
    "TelegramBot",
    "_check_rate_limit",
    "_send_telegram_reminder",
    "_run_telegram_scheduled_task",
    "notify_hand_result",
    "notify_hand_approval_request",
    "_active_bot_instance",
    "_active_event_loop",
    "format_telegram_response",
    "_split_message",
]
