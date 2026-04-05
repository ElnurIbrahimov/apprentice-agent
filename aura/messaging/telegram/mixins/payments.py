"""
PaymentsMixin — /premium, /donate, /stars, _handle_callback,
                _handle_pre_checkout, _handle_successful_payment, _handle_stars_callback
"""
from __future__ import annotations

import logging
import os
import time as _time

from aura.messaging.telegram.constants import PREMIUM_TIERS

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class PaymentsMixin:
    """Payment and premium subscription handlers."""

    _STARS_TIERS = [
        {"stars": 50, "label": "\u2b50 50 Stars", "description": "Small support"},
        {"stars": 150, "label": "\u2b50 150 Stars", "description": "Medium support"},
        {"stars": 500, "label": "\u2b50 500 Stars", "description": "Big support — unlocks priority"},
    ]

    async def _handle_premium(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /premium — show available premium tiers."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        keyboard = []
        for tier_id, tier in PREMIUM_TIERS.items():
            price_str = f"${tier['price'] / 100:.2f}"
            keyboard.append([InlineKeyboardButton(
                f"{tier['title']} — {price_str}",
                callback_data=f"buy_{tier_id}"
            )])

        text = "AURA Premium\n\n"
        for tier_id, tier in PREMIUM_TIERS.items():
            benefits = "\n".join(f"  • {b}" for b in tier["benefits"])
            text += f"{tier['title']} (${tier['price']/100:.2f}/mo)\n{benefits}\n\n"

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(text, reply_markup=reply_markup)

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries for premium purchase buttons."""
        query = update.callback_query
        await query.answer()

        tier_id = query.data.replace("buy_", "")
        tier = PREMIUM_TIERS.get(tier_id)
        if not tier:
            return

        provider_token = os.getenv("TELEGRAM_PAYMENT_TOKEN", "")
        if not provider_token:
            await query.message.reply_text("Payments not configured yet. Contact the admin.")
            return

        await context.bot.send_invoice(
            chat_id=query.from_user.id,
            title=tier["title"],
            description=tier["description"],
            payload=f"premium_{tier_id}_{query.from_user.id}",
            provider_token=provider_token,
            currency=tier["currency"],
            prices=[LabeledPrice(tier["title"], tier["price"])],
            start_parameter=f"premium_{tier_id}",
        )

    async def _handle_pre_checkout(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Must answer pre-checkout query within 10 seconds."""
        query = update.pre_checkout_query
        await query.answer(ok=True)

    async def _handle_successful_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Record successful payment and grant premium status."""
        payment = update.message.successful_payment
        user_id = str(update.effective_user.id)
        tier_id = payment.invoice_payload.split("_")[1] if "_" in payment.invoice_payload else "supporter"

        self._premium_users[user_id] = {
            "tier": tier_id,
            "paid_at": _time.time(),
            "amount": payment.total_amount,
            "currency": payment.currency,
        }

        self.store.set_premium(
            user_id=user_id, tier=tier_id,
            stars_amount=payment.total_amount,
            metadata={"currency": payment.currency, "paid_at": _time.time()},
        )

        tier = PREMIUM_TIERS.get(tier_id, {})
        await update.message.reply_text(
            f"Thank you for your support!\n\n"
            f"You now have {tier.get('title', 'Premium')} access.\n"
            f"Benefits:\n" + "\n".join(f"  • {b}" for b in tier.get("benefits", []))
        )

    async def _handle_donate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /donate — one-time support payment."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        args = context.args
        amount = 500  # Default $5
        if args:
            try:
                amount = int(float(args[0]) * 100)
            except ValueError:
                pass

        provider_token = os.getenv("TELEGRAM_PAYMENT_TOKEN", "")
        if not provider_token:
            await update.message.reply_text("Payments not configured. Set TELEGRAM_PAYMENT_TOKEN.")
            return

        await context.bot.send_invoice(
            chat_id=update.effective_chat.id,
            title="Support AURA",
            description="One-time donation to support AURA development",
            payload=f"donate_{update.effective_user.id}_{amount}",
            provider_token=provider_token,
            currency="USD",
            prices=[LabeledPrice("Donation", amount)],
        )

    async def _handle_stars(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stars — support AURA with Telegram Stars (XTR currency)."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        keyboard = []
        for tier in self._STARS_TIERS:
            keyboard.append([InlineKeyboardButton(
                f"{tier['label']} — {tier['description']}",
                callback_data=f"stars_{tier['stars']}"
            )])
        text = (
            "\u2b50 Support AURA with Telegram Stars\n\n"
            "Telegram Stars are an in-app currency. "
            "Choose a tier below to send stars as a thank-you!"
        )
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def _handle_stars_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle stars_<amount> callback — send a Stars invoice."""
        query = update.callback_query
        await query.answer()
        try:
            amount = int(query.data.replace("stars_", ""))
        except ValueError:
            return
        await context.bot.send_invoice(
            chat_id=query.from_user.id,
            title=f"Support AURA — {amount} Stars",
            description=f"Send {amount} Telegram Stars to support AURA development",
            payload=f"stars_{amount}_{query.from_user.id}",
            provider_token="",  # Empty for Telegram Stars
            currency="XTR",
            prices=[LabeledPrice(f"{amount} Stars", amount)],
        )
