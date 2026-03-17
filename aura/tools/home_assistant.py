"""Home Assistant Bridge — control your smart home from AURA.

Connects to Home Assistant via its REST API to control lights, switches,
thermostats, cameras, and any other entity. AURA can respond to context
by adjusting your physical environment ("starting focus mode" → dims lights).

Config (set in .env):
    HASS_URL   — Home Assistant URL, e.g. http://homeassistant.local:8123
    HASS_TOKEN — Long-lived access token (HA → Profile → Long-lived tokens)
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10


class HomeAssistantTool:
    """Control Home Assistant smart home — lights, switches, climate, scenes, automations."""

    name = "home_assistant"
    description = "Control your smart home via Home Assistant — lights, temperature, switches, scenes, automations"

    def __init__(self):
        self._url = os.getenv("HASS_URL", "").rstrip("/")
        self._token = os.getenv("HASS_TOKEN", "")
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if self._url and self._token:
            logger.info(f"[HomeAssistant] Connected to {self._url}")
        else:
            logger.info("[HomeAssistant] No HASS_URL/HASS_TOKEN set — configure in .env")

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _check_config(self) -> Optional[Dict]:
        if not self._url:
            return {"success": False, "error": "HASS_URL not set in .env (e.g. http://homeassistant.local:8123)"}
        if not self._token:
            return {"success": False, "error": "HASS_TOKEN not set in .env — get it from HA → Profile → Long-lived access tokens"}
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests not installed"}
        return None

    def _get(self, path: str) -> tuple[Optional[Any], Optional[str]]:
        err = self._check_config()
        if err:
            return None, err["error"]
        try:
            resp = requests.get(f"{self._url}/api{path}", headers=self._headers, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError:
            return None, f"Cannot connect to Home Assistant at {self._url} — is it running?"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return None, str(e)

    def _post(self, path: str, data: Optional[Dict] = None) -> tuple[Optional[Any], Optional[str]]:
        err = self._check_config()
        if err:
            return None, err["error"]
        try:
            resp = requests.post(
                f"{self._url}/api{path}",
                headers=self._headers,
                json=data or {},
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {}, None
        except requests.exceptions.ConnectionError:
            return None, f"Cannot connect to Home Assistant at {self._url}"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return None, str(e)

    # ------------------------------------------------------------------ #
    # State & Info
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict:
        """Check Home Assistant connection and version."""
        data, err = self._get("/")
        if err:
            return {"success": False, "error": err}
        return {
            "success": True,
            "url": self._url,
            "version": data.get("version"),
            "location": data.get("location_name"),
            "message": data.get("message", "Connected"),
        }

    def get_states(self, domain: Optional[str] = None) -> Dict:
        """Get states of all entities (or filter by domain).

        Args:
            domain: Filter by domain e.g. 'light', 'switch', 'climate', 'sensor'
        """
        data, err = self._get("/states")
        if err:
            return {"success": False, "error": err}
        entities = data or []
        if domain:
            entities = [e for e in entities if e["entity_id"].startswith(f"{domain}.")]
        result = [
            {
                "entity_id": e["entity_id"],
                "state": e["state"],
                "friendly_name": e.get("attributes", {}).get("friendly_name", ""),
                "last_changed": e.get("last_changed", "")[:16],
            }
            for e in entities
        ]
        return {
            "success": True,
            "domain": domain or "all",
            "count": len(result),
            "entities": result,
        }

    def get_entity(self, entity_id: str) -> Dict:
        """Get detailed state of a specific entity.

        Args:
            entity_id: Entity ID e.g. 'light.living_room', 'switch.tv'
        """
        data, err = self._get(f"/states/{entity_id}")
        if err:
            return {"success": False, "error": err}
        return {
            "success": True,
            "entity_id": data["entity_id"],
            "state": data["state"],
            "attributes": data.get("attributes", {}),
            "last_changed": data.get("last_changed", "")[:19],
            "last_updated": data.get("last_updated", "")[:19],
        }

    def search_entities(self, query: str) -> Dict:
        """Search entities by name or id.

        Args:
            query: Search term e.g. 'kitchen', 'bedroom light'
        """
        data, err = self._get("/states")
        if err:
            return {"success": False, "error": err}
        q = query.lower()
        matches = [
            e for e in (data or [])
            if q in e["entity_id"].lower()
            or q in e.get("attributes", {}).get("friendly_name", "").lower()
        ]
        return {
            "success": True,
            "query": query,
            "count": len(matches),
            "entities": [
                {
                    "entity_id": e["entity_id"],
                    "state": e["state"],
                    "friendly_name": e.get("attributes", {}).get("friendly_name", ""),
                }
                for e in matches[:20]
            ],
        }

    # ------------------------------------------------------------------ #
    # Controls
    # ------------------------------------------------------------------ #

    def turn_on(self, entity_id: str, **kwargs) -> Dict:
        """Turn on a light, switch, or any entity.

        Args:
            entity_id: Entity to turn on e.g. 'light.bedroom', 'switch.fan'
            brightness: Light brightness 0-255 (lights only)
            color_temp: Color temperature in mireds (lights only)
            rgb_color: RGB color as [r, g, b] list (lights only)
        """
        domain = entity_id.split(".")[0]
        service_data = {"entity_id": entity_id}
        if "brightness" in kwargs:
            service_data["brightness"] = kwargs["brightness"]
        if "color_temp" in kwargs:
            service_data["color_temp"] = kwargs["color_temp"]
        if "rgb_color" in kwargs:
            service_data["rgb_color"] = kwargs["rgb_color"]
        data, err = self._post(f"/services/{domain}/turn_on", service_data)
        if err:
            return {"success": False, "error": err}
        return {"success": True, "action": "turn_on", "entity_id": entity_id, "extras": kwargs}

    def turn_off(self, entity_id: str) -> Dict:
        """Turn off a light, switch, or any entity.

        Args:
            entity_id: Entity to turn off e.g. 'light.living_room'
        """
        domain = entity_id.split(".")[0]
        data, err = self._post(f"/services/{domain}/turn_off", {"entity_id": entity_id})
        if err:
            return {"success": False, "error": err}
        return {"success": True, "action": "turn_off", "entity_id": entity_id}

    def toggle(self, entity_id: str) -> Dict:
        """Toggle an entity (on→off or off→on).

        Args:
            entity_id: Entity to toggle
        """
        domain = entity_id.split(".")[0]
        data, err = self._post(f"/services/{domain}/toggle", {"entity_id": entity_id})
        if err:
            return {"success": False, "error": err}
        return {"success": True, "action": "toggle", "entity_id": entity_id}

    # Domains that could execute arbitrary code or damage HA
    _BLOCKED_HA_DOMAINS = frozenset({
        "shell_command", "python_script", "recorder",
        "homeassistant", "persistent_notification",
    })

    def call_service(self, domain: str, service: str, data: Optional[Dict] = None) -> Dict:
        """Call any Home Assistant service.

        Args:
            domain: Service domain e.g. 'light', 'climate', 'script', 'scene'
            service: Service name e.g. 'turn_on', 'set_temperature', 'turn'
            data: Service data / parameters
        """
        if domain.lower() in self._BLOCKED_HA_DOMAINS:
            return {"success": False, "error": f"Domain '{domain}' is blocked for security reasons"}
        result, err = self._post(f"/services/{domain}/{service}", data or {})
        if err:
            return {"success": False, "error": err}
        return {"success": True, "domain": domain, "service": service, "data": data}

    def set_climate(self, entity_id: str, temperature: Optional[float] = None,
                    hvac_mode: Optional[str] = None) -> Dict:
        """Control a climate/thermostat entity.

        Args:
            entity_id: Climate entity e.g. 'climate.living_room'
            temperature: Target temperature
            hvac_mode: 'heat', 'cool', 'auto', 'off'
        """
        data: Dict = {"entity_id": entity_id}
        results = []
        if temperature is not None:
            _, err = self._post("/services/climate/set_temperature", {**data, "temperature": temperature})
            results.append({"set_temperature": temperature, "error": err})
        if hvac_mode is not None:
            _, err = self._post("/services/climate/set_hvac_mode", {**data, "hvac_mode": hvac_mode})
            results.append({"set_hvac_mode": hvac_mode, "error": err})
        errors = [r for r in results if r.get("error")]
        return {
            "success": len(errors) == 0,
            "entity_id": entity_id,
            "actions": results,
            "errors": errors,
        }

    def activate_scene(self, scene_id: str) -> Dict:
        """Activate a Home Assistant scene.

        Args:
            scene_id: Scene entity ID e.g. 'scene.movie_time', 'scene.focus_mode'
        """
        if not scene_id.startswith("scene."):
            scene_id = f"scene.{scene_id}"
        data, err = self._post("/services/scene/turn_on", {"entity_id": scene_id})
        if err:
            return {"success": False, "error": err}
        return {"success": True, "scene_activated": scene_id}

    # ------------------------------------------------------------------ #
    # Automations & Scripts
    # ------------------------------------------------------------------ #

    def list_automations(self) -> Dict:
        """List all automations."""
        data, err = self._get("/states")
        if err:
            return {"success": False, "error": err}
        automations = [
            e for e in (data or [])
            if e["entity_id"].startswith("automation.")
        ]
        return {
            "success": True,
            "count": len(automations),
            "automations": [
                {
                    "entity_id": a["entity_id"],
                    "state": a["state"],
                    "friendly_name": a.get("attributes", {}).get("friendly_name", ""),
                    "last_triggered": a.get("attributes", {}).get("last_triggered", "never"),
                }
                for a in automations
            ],
        }

    def trigger_automation(self, automation_id: str) -> Dict:
        """Manually trigger an automation.

        Args:
            automation_id: Automation entity ID e.g. 'automation.morning_routine'
        """
        if not automation_id.startswith("automation."):
            automation_id = f"automation.{automation_id}"
        data, err = self._post("/services/automation/trigger", {"entity_id": automation_id})
        if err:
            return {"success": False, "error": err}
        return {"success": True, "triggered": automation_id}

    def run_script(self, script_id: str) -> Dict:
        """Run a Home Assistant script.

        Args:
            script_id: Script entity ID e.g. 'script.good_morning'
        """
        if not script_id.startswith("script."):
            script_id = f"script.{script_id}"
        data, err = self._post("/services/script/turn_on", {"entity_id": script_id})
        if err:
            return {"success": False, "error": err}
        return {"success": True, "script_run": script_id}

    # ------------------------------------------------------------------ #
    # Convenience: Focus Mode
    # ------------------------------------------------------------------ #

    def focus_mode(self, enable: bool = True) -> Dict:
        """Enable/disable focus mode (dim lights, set cool temperature).

        Calls scene.focus_mode if it exists, otherwise applies direct control.
        """
        if enable:
            # Try scene first
            data, err = self._post("/services/scene/turn_on", {"entity_id": "scene.focus_mode"})
            if not err:
                return {"success": True, "focus_mode": "enabled", "method": "scene"}
            # Fallback: dim all lights to 30%
            lights_data, _ = self._get("/states")
            lights = [e["entity_id"] for e in (lights_data or []) if e["entity_id"].startswith("light.")]
            results = []
            for light in lights[:10]:
                self.turn_on(light, brightness=77)  # 30% of 255
                results.append(light)
            return {"success": True, "focus_mode": "enabled", "method": "direct", "dimmed": results}
        else:
            data, err = self._post("/services/scene/turn_on", {"entity_id": "scene.normal"})
            if not err:
                return {"success": True, "focus_mode": "disabled", "method": "scene"}
            return {"success": True, "focus_mode": "disabled", "note": "No 'scene.normal' found — adjust manually"}

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a Home Assistant action."""
        a = action.lower().strip()
        entity = kwargs.get("entity_id") or kwargs.get("entity") or ""

        if "status" in a or "connect" in a or "check" in a:
            return self.get_status()
        if "search" in a or "find" in a:
            return self.search_entities(kwargs.get("query") or action)
        if "state" in a or "list" in a:
            return self.get_states(kwargs.get("domain"))
        if "get" in a and entity:
            return self.get_entity(entity)
        if "turn_on" in a or ("on" in a and "turn" in a):
            return self.turn_on(entity, **{k: v for k, v in kwargs.items() if k not in ("entity_id", "entity")})
        if "turn_off" in a or ("off" in a and "turn" in a):
            return self.turn_off(entity)
        if "toggle" in a:
            return self.toggle(entity)
        if "scene" in a or "activate" in a:
            return self.activate_scene(kwargs.get("scene") or entity)
        if "climate" in a or "temperature" in a or "thermostat" in a:
            return self.set_climate(entity, kwargs.get("temperature"), kwargs.get("hvac_mode"))
        if "automation" in a and ("list" in a or "get" in a):
            return self.list_automations()
        if "automation" in a:
            return self.trigger_automation(entity)
        if "script" in a:
            return self.run_script(entity)
        if "focus" in a:
            return self.focus_mode(kwargs.get("enable", True))
        if "service" in a:
            return self.call_service(kwargs.get("domain", ""), kwargs.get("service", ""), kwargs.get("data"))
        return self.get_status()
