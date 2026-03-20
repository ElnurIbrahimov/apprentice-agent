"""Real-time crypto prices from CoinGecko API (free, no API key needed)."""

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CryptoPriceTool:
    """Get real-time crypto prices from CoinGecko API (free, no API key)."""

    name = "crypto_price"
    description = "Get real-time cryptocurrency prices from CoinGecko"

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Common crypto name mappings
    CRYPTO_IDS = {
        "bitcoin": "bitcoin", "btc": "bitcoin",
        "ethereum": "ethereum", "eth": "ethereum",
        "solana": "solana", "sol": "solana",
        "cardano": "cardano", "ada": "cardano",
        "dogecoin": "dogecoin", "doge": "dogecoin",
        "xrp": "ripple", "ripple": "ripple",
        "polkadot": "polkadot", "dot": "polkadot",
        "bnb": "binancecoin", "binance": "binancecoin",
        "avalanche": "avalanche-2", "avax": "avalanche-2",
        "polygon": "matic-network", "matic": "matic-network",
        "litecoin": "litecoin", "ltc": "litecoin",
        "chainlink": "chainlink", "link": "chainlink",
        "uniswap": "uniswap", "uni": "uniswap",
        "toncoin": "the-open-network", "ton": "the-open-network",
        "tron": "tron", "trx": "tron",
        "shiba": "shiba-inu", "shib": "shiba-inu",
        "pepe": "pepe",
    }

    def __init__(self):
        self.timeout = 10

    def get_price(self, crypto: str, currency: str = "usd") -> Dict:
        """
        Get current price for a cryptocurrency.

        Args:
            crypto: Crypto name or symbol (bitcoin, btc, ethereum, eth, etc.)
            currency: Fiat currency (usd, eur, gbp, etc.)

        Returns:
            Dict with price info
        """
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests library not available"}

        # Normalize crypto name
        crypto_lower = crypto.lower().strip()
        crypto_id = self.CRYPTO_IDS.get(crypto_lower, crypto_lower)

        logger.info(f"[CRYPTO] Getting price for {crypto_id} in {currency}")

        try:
            response = requests.get(
                f"{self.BASE_URL}/simple/price",
                params={
                    "ids": crypto_id,
                    "vs_currencies": currency,
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()

                if crypto_id in data:
                    price_data = data[crypto_id]
                    price = price_data.get(currency, 0)
                    change_24h = price_data.get(f"{currency}_24h_change", 0)
                    market_cap = price_data.get(f"{currency}_market_cap", 0)

                    logger.info(f"[CRYPTO] {crypto_id}: ${price:,.2f}")

                    return {
                        "success": True,
                        "crypto": crypto_id,
                        "symbol": crypto_lower.upper(),
                        "currency": currency.upper(),
                        "price": price,
                        "price_formatted": f"${price:,.2f}" if currency == "usd" else f"{price:,.2f} {currency.upper()}",
                        "change_24h": round(change_24h, 2) if change_24h else 0,
                        "change_24h_formatted": f"{change_24h:+.2f}%" if change_24h else "N/A",
                        "market_cap": market_cap,
                        "market_cap_formatted": f"${market_cap:,.0f}" if market_cap else "N/A",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Crypto '{crypto}' not found. Try: bitcoin, ethereum, solana, dogecoin, etc.",
                    }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                }

        except requests.Timeout:
            return {"success": False, "error": "Request timed out"}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_multiple_prices(self, cryptos: List[str], currency: str = "usd") -> Dict:
        """Get prices for multiple cryptocurrencies at once."""
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests library not available"}

        crypto_ids = [self.CRYPTO_IDS.get(c.lower(), c.lower()) for c in cryptos]

        try:
            response = requests.get(
                f"{self.BASE_URL}/simple/price",
                params={
                    "ids": ",".join(crypto_ids),
                    "vs_currencies": currency,
                    "include_24hr_change": "true",
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                results = {}

                for crypto_id in crypto_ids:
                    if crypto_id in data:
                        price = data[crypto_id].get(currency, 0)
                        change = data[crypto_id].get(f"{currency}_24h_change", 0)
                        results[crypto_id] = {
                            "price": price,
                            "price_formatted": f"{{'usd': '$', 'eur': '€', 'gbp': '£', 'jpy': '¥'}.get(currency.lower(), currency.upper() + ' ')}{price:,.2f}",
                            "change_24h": f"{change:+.2f}%" if change else "N/A",
                        }

                return {"success": True, "prices": results}
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run(self, query: str) -> Dict:
        """Main entry point - parse query and get price."""
        query_lower = query.lower()

        # Try to find a crypto name in the query
        for crypto_name, crypto_id in self.CRYPTO_IDS.items():
            if crypto_name in query_lower:
                return self.get_price(crypto_id)

        # If no match, try the first word as crypto name
        words = query.split()
        if words:
            return self.get_price(words[0])

        return {"success": False, "error": "Please specify a cryptocurrency (bitcoin, ethereum, etc.)"}


# Convenience function for direct use
def crypto_price(crypto: str, currency: str = "usd") -> Dict:
    """Get crypto price."""
    tool = CryptoPriceTool()
    return tool.get_price(crypto, currency)
