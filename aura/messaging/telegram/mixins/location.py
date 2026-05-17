"""
LocationMixin — _handle_location, _get_location_info, _handle_nearby, _handle_fleet
"""
from __future__ import annotations

import asyncio
import logging

try:
    from telegram import Update
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class LocationMixin:
    """Location sharing and nearby-search handlers."""

    async def _handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle shared location — provide contextual local info."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        location = update.message.location
        lat = location.latitude
        lon = location.longitude

        placeholder = await update.message.reply_text("\U0001f4cd Getting info for your location...")

        try:
            info, _results = await self._get_location_info(lat, lon)

            # Store last known location for /nearby (persisted to SQLite)
            self.store.set_user_location(str(update.effective_user.id), lat, lon)

            await self._edit_or_send_response(placeholder, str(update.effective_chat.id), info, update)
        except Exception as e:
            logger.error(f"[Location] Error getting location info: {e}")
            await self._edit_or_send_response(
                placeholder, str(update.effective_chat.id),
                f"Couldn't get location info: {e}", update
            )

    async def _get_location_info(self, lat: float, lon: float) -> tuple:
        """Fetch contextual info for a location.

        Returns (formatted_text, raw_results_dict).
        """
        import aiohttp

        results: dict = {}

        async with aiohttp.ClientSession() as session:
            # 1. Reverse geocoding via Nominatim
            try:
                url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=14"
                headers = {"User-Agent": "AURA-Bot/4.5"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        address = data.get("address", {})
                        results["city"] = (
                            address.get("city")
                            or address.get("town")
                            or address.get("village")
                            or "Unknown"
                        )
                        results["country"] = address.get("country", "")
                        results["display"] = data.get("display_name", "")[:100]
            except Exception:
                pass

            # 2. Weather from wttr.in
            try:
                url = f"https://wttr.in/{lat},{lon}?format=j1"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current = data.get("current_condition", [{}])[0]
                        results["temp"] = current.get("temp_C", "?")
                        results["feels_like"] = current.get("FeelsLikeC", "?")
                        results["condition"] = current.get("weatherDesc", [{}])[0].get("value", "")
                        results["humidity"] = current.get("humidity", "?")
                        results["wind"] = current.get("windspeedKmph", "?")

                        # Astronomy
                        astro = data.get("weather", [{}])[0].get("astronomy", [{}])[0]
                        results["sunrise"] = astro.get("sunrise", "")
                        results["sunset"] = astro.get("sunset", "")
            except Exception:
                pass

            # 3. Timezone from timeapi.io
            try:
                url = f"https://timeapi.io/api/Time/current/coordinate?latitude={lat}&longitude={lon}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results["timezone"] = data.get("timeZone", "")
                        results["local_time"] = data.get("time", "")
                        results["day_of_week"] = data.get("dayOfWeek", "")
            except Exception:
                pass

        # Build response text
        city = results.get("city", "Unknown")
        country = results.get("country", "")

        lines = [f"\U0001f4cd {city}, {country}"]

        if "temp" in results:
            lines.append(f"\n\U0001f321 Weather: {results['temp']}\u00b0C (feels like {results['feels_like']}\u00b0C)")
            lines.append(f"   {results['condition']}")
            lines.append(f"   Humidity: {results['humidity']}% | Wind: {results['wind']} km/h")

        if "sunrise" in results:
            lines.append(f"\n\U0001f305 Sunrise: {results['sunrise']} | Sunset: {results['sunset']}")

        if "timezone" in results:
            lines.append(f"\n\U0001f550 Local time: {results.get('local_time', '')} ({results['timezone']})")

        lines.append(f"\n\U0001f4cc Coordinates: {lat:.4f}, {lon:.4f}")

        lines.append("\nTip: Send me a message about what you want to do here, and I can help with local recommendations!")

        return "\n".join(lines), results

    async def _handle_nearby(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /nearby <query> — search for nearby places using last shared location."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        user_id = str(update.effective_user.id)
        last_location = self.store.get_user_location(user_id)

        if not last_location:
            await update.message.reply_text("Please share your location first, then use /nearby <query>")
            return

        query = " ".join(context.args) if context.args else "restaurants"
        lat, lon = last_location["latitude"], last_location["longitude"]
        city = "the area"

        prompt = (
            f"Find {query} near coordinates {lat},{lon} (in {city}). "
            f"Give me top 3-5 recommendations with brief descriptions."
        )

        placeholder = await update.message.reply_text(f"\U0001f50d Searching for {query} nearby...")

        try:
            response_text, _ = await asyncio.to_thread(self._run_agent_sync, prompt)
            await self._edit_or_send_response(
                placeholder, str(update.effective_chat.id),
                response_text or "Couldn't find results.", update
            )
        except Exception as e:
            logger.error(f"[Nearby] Error: {e}")
            await self._edit_or_send_response(
                placeholder, str(update.effective_chat.id),
                f"Couldn't search nearby: {e}", update
            )

    async def _handle_fleet(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /fleet <goal> — parallel multi-agent decomposition."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /fleet <goal>\n\nExample:\n/fleet Analyze the codebase for bugs, optimizations, and security issues")
            return

        goal = " ".join(args)
        placeholder = await update.message.reply_text(f"Fleet dispatched: {goal[:100]}...\n\nRunning all specialists in parallel...")

        try:
            orch = getattr(self.aura, "orchestrator", None)
            if not orch:
                await self._edit_or_send_response(placeholder, str(update.effective_chat.id), "Multi-agent orchestrator not available.", update)
                return

            result = await asyncio.to_thread(orch.chat, goal)
            response = result if isinstance(result, str) else str(result.get("response", result))
            await self._edit_or_send_response(placeholder, str(update.effective_chat.id), f"Fleet Results:\n\n{response}", update)
        except Exception as e:
            await self._edit_or_send_response(placeholder, str(update.effective_chat.id), f"Fleet error: {e}", update)
