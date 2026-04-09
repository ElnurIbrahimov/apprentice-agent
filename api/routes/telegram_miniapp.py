"""Telegram Mini App — initData validation endpoint.

Validates the cryptographic hash Telegram includes in initData so the
backend can verify a request genuinely comes from the Mini App.

See: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
"""

import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/telegram",
    tags=["telegram-miniapp"],
    dependencies=[Depends(require_api_key)],
)


class InitDataRequest(BaseModel):
    init_data: str  # raw window.Telegram.WebApp.initData string


class InitDataResponse(BaseModel):
    valid: bool
    user_id: int | None = None
    first_name: str | None = None
    username: str | None = None
    auth_date: int | None = None


def _validate_init_data(init_data: str, bot_token: str) -> dict | None:
    """Validate Telegram Mini App initData using HMAC-SHA256.

    Returns parsed data dict if valid, None if invalid.
    """
    if not init_data or not bot_token:
        return None

    try:
        parsed = parse_qs(init_data, keep_blank_values=True)
        # Extract hash
        received_hash = parsed.get("hash", [None])[0]
        if not received_hash:
            return None

        # Build data-check-string: sorted key=value pairs, excluding "hash"
        data_pairs = []
        for key, values in parsed.items():
            if key == "hash":
                continue
            data_pairs.append(f"{key}={values[0]}")
        data_pairs.sort()
        data_check_string = "\n".join(data_pairs)

        # HMAC-SHA256: secret_key = HMAC-SHA256("WebAppData", bot_token)
        secret_key = hmac.new(
            b"WebAppData", bot_token.encode(), hashlib.sha256
        ).digest()

        # Compute hash
        computed_hash = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(computed_hash, received_hash):
            return None

        # Check auth_date freshness (reject if older than 1 hour)
        auth_date = int(parsed.get("auth_date", [0])[0])
        if time.time() - auth_date > 3600:
            logger.warning(f"[MiniApp] initData too old: auth_date={auth_date}")
            return None

        # Parse user object
        user_raw = parsed.get("user", [None])[0]
        user = json.loads(user_raw) if user_raw else {}

        return {
            "user_id": user.get("id"),
            "first_name": user.get("first_name"),
            "username": user.get("username"),
            "auth_date": auth_date,
        }

    except Exception as e:
        logger.error(f"[MiniApp] initData validation error: {e}")
        return None


@router.post("/validate-init", response_model=InitDataResponse)
async def validate_init_data(request: InitDataRequest):
    """Validate Telegram Mini App initData.

    The Mini App should call this on startup to prove it's running
    inside Telegram and get the authenticated user info.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(status_code=500, detail="Bot token not configured")

    result = _validate_init_data(request.init_data, bot_token)
    if result is None:
        return InitDataResponse(valid=False)

    return InitDataResponse(
        valid=True,
        user_id=result.get("user_id"),
        first_name=result.get("first_name"),
        username=result.get("username"),
        auth_date=result.get("auth_date"),
    )
