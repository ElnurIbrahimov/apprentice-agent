"""
Natural-language time/schedule expression parsers and code extraction helpers.
No dependencies on bot.py or any mixin.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Dict


def _parse_time_expression(text: str) -> Optional[datetime]:
    """Parse natural language time expressions into an absolute datetime.

    Handles:
        - "in Xh", "in Xm", "in X hours", "in X minutes", "in X days"
        - "at HH:MM", "at H:MMam/pm", "at Ham/pm"
        - "tomorrow", "tomorrow HH:MM", "tomorrow at HH:MM"
        - "next monday", "next tuesday at HH:MM"

    Returns None if parsing fails.
    """
    original = text.strip()
    t = original.lower().strip()
    now = datetime.now()

    # --- Relative: "in X hours/minutes/seconds/days" or shorthand "in 2h" ---
    m = re.match(r'in\s+(\d+)\s*h(?:ours?)?$', t)
    if m:
        return now + timedelta(hours=int(m.group(1)))

    m = re.match(r'in\s+(\d+)\s*m(?:in(?:utes?)?)?$', t)
    if m:
        return now + timedelta(minutes=int(m.group(1)))

    m = re.match(r'in\s+(\d+)\s*s(?:ec(?:onds?)?)?$', t)
    if m:
        return now + timedelta(seconds=int(m.group(1)))

    m = re.match(r'in\s+(\d+)\s*d(?:ays?)?$', t)
    if m:
        return now + timedelta(days=int(m.group(1)))

    # Compound relative: "in 1h 30m", "in 2 hours 15 minutes"
    m = re.match(r'in\s+(\d+)\s*h(?:ours?)?\s+(\d+)\s*m(?:in(?:utes?)?)?$', t)
    if m:
        return now + timedelta(hours=int(m.group(1)), minutes=int(m.group(2)))

    # --- "at HH:MM" or "at H:MMam/pm" or "at Ham/pm" ---
    m = re.match(r'at\s+(\d{1,2}):(\d{2})\s*(am|pm)?$', t)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        ampm = m.group(3)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    m = re.match(r'at\s+(\d{1,2})\s*(am|pm)$', t)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    # --- "tomorrow" optionally with time ---
    m = re.match(r'tomorrow(?:\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?)?$', t)
    if m:
        tomorrow = now + timedelta(days=1)
        if m.group(1):
            hour, minute = int(m.group(1)), int(m.group(2))
            ampm = m.group(3)
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)

    m = re.match(r'tomorrow(?:\s+(?:at\s+)?(\d{1,2})\s*(am|pm))?$', t)
    if m and m.group(1):
        tomorrow = now + timedelta(days=1)
        hour = int(m.group(1))
        ampm = m.group(2)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)

    # --- "next <weekday>" optionally with time ---
    days_of_week = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6,
        'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3,
        'fri': 4, 'sat': 5, 'sun': 6,
    }
    m = re.match(
        r'next\s+(\w+?)(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?$', t
    )
    if m:
        day_name = m.group(1)
        if day_name in days_of_week:
            target_weekday = days_of_week[day_name]
            days_ahead = (target_weekday - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target = now + timedelta(days=days_ahead)

            hour, minute = 9, 0  # default 9am
            if m.group(2):
                hour = int(m.group(2))
                minute = int(m.group(3)) if m.group(3) else 0
                ampm = m.group(4)
                if ampm == 'pm' and hour != 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0

            return target.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return None


def _parse_schedule_expression(text: str) -> Optional[Dict]:
    """Parse a schedule expression into APScheduler trigger parameters.

    Handles:
        - "every Xh", "every X hours", "every X minutes", "every Xm"
        - "daily at HH:MM", "daily at Ham/pm"
        - "every monday at HH:MM", "every <weekday> at H:MMam/pm"
        - "every 30 minutes", "every 2 hours"

    Returns a dict with:
        - "type": "interval" | "cron"
        - For interval: "hours", "minutes", "seconds"
        - For cron: "cron_expression" (5-part)
    Returns None if parsing fails.
    """
    t = text.lower().strip()

    # --- "every Xh" / "every X hours" / "every Xm" / "every X minutes" ---
    m = re.match(r'every\s+(\d+)\s*h(?:ours?)?$', t)
    if m:
        return {"type": "interval", "hours": int(m.group(1)), "minutes": 0, "seconds": 0}

    m = re.match(r'every\s+(\d+)\s*m(?:in(?:utes?)?)?$', t)
    if m:
        return {"type": "interval", "hours": 0, "minutes": int(m.group(1)), "seconds": 0}

    m = re.match(r'every\s+(\d+)\s*s(?:ec(?:onds?)?)?$', t)
    if m:
        return {"type": "interval", "hours": 0, "minutes": 0, "seconds": int(m.group(1))}

    # --- "daily at HH:MM" or "daily at Ham/pm" ---
    m = re.match(r'daily\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$', t)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return {"type": "cron", "cron_expression": f"{minute} {hour} * * *"}

    # --- "every <weekday> at HH:MM" ---
    days_of_week = {
        'monday': '1', 'tuesday': '2', 'wednesday': '3', 'thursday': '4',
        'friday': '5', 'saturday': '6', 'sunday': '0',
        'mon': '1', 'tue': '2', 'wed': '3', 'thu': '4',
        'fri': '5', 'sat': '6', 'sun': '0',
    }
    m = re.match(
        r'every\s+(\w+)\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$', t
    )
    if m:
        day_name = m.group(1)
        if day_name in days_of_week:
            dow = days_of_week[day_name]
            hour = int(m.group(2))
            minute = int(m.group(3)) if m.group(3) else 0
            ampm = m.group(4)
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            return {"type": "cron", "cron_expression": f"{minute} {hour} * * {dow}"}

    return None


def _extract_code_from_message(text: str) -> str:
    """Extract Python code from a /code message.

    Supports:
      /code print("hello")
      /code ```python\\nprint("hello")\\n```
      /code ```\\nprint("hello")\\n```
    """
    # Remove the /code prefix (may include @botname)
    stripped = re.sub(r"^/code(@\w+)?\s*", "", text, count=1)

    # Check for fenced code block: ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()

    return stripped.strip()
