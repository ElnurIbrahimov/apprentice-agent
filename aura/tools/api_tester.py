"""API Tester tool — HTTP client for testing REST APIs (like Postman)."""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Maximum response body to store/return
MAX_RESPONSE_BODY = 10_000

HISTORY_FILE = Path(__file__).parent.parent.parent / "data" / "api_tester_history.json"


@dataclass
class APIRequest:
    """A recorded API request/response pair."""
    id: str
    method: str
    url: str
    status_code: Optional[int] = None
    elapsed_ms: float = 0.0
    request_headers: Dict[str, str] = field(default_factory=dict)
    request_body: Optional[str] = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    response_body: str = ""
    content_type: str = ""
    timestamp: str = ""
    error: Optional[str] = None
    name: str = ""  # Optional label

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class APITesterTool:
    """HTTP client for testing REST APIs."""

    name = "api_tester"
    description = "Test REST APIs with HTTP requests (GET, POST, PUT, DELETE)"

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._history = []

    def _save_history(self):
        try:
            # Keep last 100 entries
            if len(self._history) > 100:
                self._history = self._history[-100:]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except IOError:
            pass

    def _truncate(self, text: str, max_len: int = MAX_RESPONSE_BODY) -> str:
        if len(text) <= max_len:
            return text
        half = max_len // 2
        return text[:half] + f"\n...[truncated {len(text) - max_len} chars]...\n" + text[-half:]

    def request(self, method: str, url: str, headers: Dict[str, str] = None,
                body: str = None, json_body: Any = None, auth: tuple = None,
                timeout: int = 30, name: str = "") -> dict:
        """Send an HTTP request."""
        try:
            import requests as req_lib
        except ImportError:
            return {"success": False, "error": "requests library not installed. Run: pip install requests"}

        method = method.upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
            return {"success": False, "error": f"Invalid method: {method}"}

        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
        if not urlparse(url).netloc:
            return {"success": False, "error": f"Invalid URL: {url}"}

        # SECURITY: Block SSRF targets — private/loopback ranges and metadata endpoints
        import ipaddress
        _parsed = urlparse(url)
        _hostname = _parsed.hostname or ""
        _ssrf_blocked = [
            "169.254.169.254",  # AWS/GCP/Azure metadata
            "metadata.google.internal",
            "169.254.170.2",    # ECS metadata
        ]
        if _hostname in _ssrf_blocked:
            return {"success": False, "error": "Blocked: metadata endpoint"}
        try:
            _ip = ipaddress.ip_address(_hostname)
            if _ip.is_private or _ip.is_loopback or _ip.is_link_local:
                return {"success": False, "error": "Blocked: private/loopback IP addresses not allowed"}
        except ValueError:
            pass  # hostname is a DNS name, not an IP — allow

        req_headers = dict(headers or {})
        req_body = None

        if json_body is not None:
            req_body = json.dumps(json_body) if not isinstance(json_body, str) else json_body
            req_headers.setdefault("Content-Type", "application/json")
        elif body:
            req_body = body

        record = APIRequest(
            id=uuid.uuid4().hex[:8],
            method=method,
            url=url,
            request_headers=req_headers,
            request_body=req_body,
            name=name,
        )

        try:
            start = time.time()
            resp = req_lib.request(
                method=method,
                url=url,
                headers=req_headers if req_headers else None,
                data=req_body if (req_headers.get("Content-Type") != "application/json" and req_body) else None,
                json=json.loads(req_body) if (req_headers.get("Content-Type") == "application/json" and req_body) else None,
                auth=auth,
                timeout=timeout,
                allow_redirects=True,
            )
            elapsed = (time.time() - start) * 1000

            record.status_code = resp.status_code
            record.elapsed_ms = round(elapsed, 2)
            record.response_headers = dict(resp.headers)
            record.content_type = resp.headers.get("Content-Type", "")

            # Get response body
            try:
                resp_body = resp.text
            except Exception:
                resp_body = repr(resp.content[:500])

            record.response_body = self._truncate(resp_body)

        except Exception as e:
            record.error = str(e)
            record.elapsed_ms = round((time.time() - start) * 1000, 2)

        # Save to history
        self._history.append(record.to_dict())
        self._save_history()

        if record.error:
            return {
                "success": False,
                "error": record.error,
                "elapsed_ms": record.elapsed_ms,
                "request_id": record.id,
                "response": f"Request failed: {record.error}"
            }

        # Try to parse JSON for pretty display
        body_preview = record.response_body
        if "json" in record.content_type.lower():
            try:
                parsed_json = json.loads(record.response_body)
                body_preview = json.dumps(parsed_json, indent=2)[:2000]
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "success": True,
            "request_id": record.id,
            "method": method,
            "url": url,
            "status_code": record.status_code,
            "elapsed_ms": record.elapsed_ms,
            "content_type": record.content_type,
            "response_headers": record.response_headers,
            "body": record.response_body,
            "body_length": len(record.response_body),
            "response": f"{method} {url} -> {record.status_code} ({record.elapsed_ms:.0f}ms)\n{body_preview[:1000]}"
        }

    def get(self, url: str, headers: Dict[str, str] = None, **kwargs) -> dict:
        return self.request("GET", url, headers=headers, **kwargs)

    def post(self, url: str, body: str = None, json_body: Any = None,
             headers: Dict[str, str] = None, **kwargs) -> dict:
        return self.request("POST", url, headers=headers, body=body, json_body=json_body, **kwargs)

    def put(self, url: str, body: str = None, json_body: Any = None,
            headers: Dict[str, str] = None, **kwargs) -> dict:
        return self.request("PUT", url, headers=headers, body=body, json_body=json_body, **kwargs)

    def delete(self, url: str, headers: Dict[str, str] = None, **kwargs) -> dict:
        return self.request("DELETE", url, headers=headers, **kwargs)

    def history(self, limit: int = 10) -> dict:
        """Show recent request history."""
        recent = self._history[-limit:]
        recent.reverse()
        formatted = []
        for r in recent:
            status = r.get("status_code") or "ERR"
            formatted.append(f"[{r['id']}] {r['method']} {r['url']} -> {status} ({r.get('elapsed_ms', 0):.0f}ms)")

        return {
            "success": True,
            "count": len(recent),
            "history": recent,
            "response": f"Last {len(recent)} request(s):\n" + "\n".join(formatted)
        }

    def inspect(self, request_id: str) -> dict:
        """Get full details of a past request."""
        for r in reversed(self._history):
            if r.get("id") == request_id:
                return {"success": True, "request": r, "response": json.dumps(r, indent=2)[:3000]}
        return {"success": False, "error": f"Request not found: {request_id}"}

    def clear_history(self) -> dict:
        count = len(self._history)
        self._history = []
        self._save_history()
        return {"success": True, "cleared": count, "response": f"Cleared {count} request(s)"}

    # -- Dispatch -----------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # History
        if action_lower.startswith("history"):
            limit = kwargs.get("limit", 10)
            return self.history(limit=limit)

        # Inspect
        if action_lower.startswith("inspect") or action_lower.startswith("detail"):
            req_id = kwargs.get("request_id")
            if not req_id:
                m = re.search(r'\b([a-f0-9]{8})\b', action)
                req_id = m.group(1) if m else None
            if req_id:
                return self.inspect(req_id)
            return {"success": False, "error": "No request ID specified"}

        # Clear history
        if "clear" in action_lower and "history" in action_lower:
            return self.clear_history()

        # Parse method + URL from action
        method = kwargs.get("method")
        url = kwargs.get("url")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body")
        json_body = kwargs.get("json_body") or kwargs.get("json")
        auth = kwargs.get("auth")
        timeout = kwargs.get("timeout", 30)

        if not method or not url:
            # Try: "GET https://api.example.com/data"
            parts = action.split(None, 1)
            if len(parts) >= 2 and parts[0].upper() in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                method = parts[0].upper()
                url_part = parts[1].strip()
                # Extract URL (may have trailing params)
                url_match = re.match(r'(https?://\S+|[\w.-]+\.\w+\S*)', url_part)
                url = url_match.group(1) if url_match else url_part
            elif len(parts) >= 1:
                # Just a URL — default to GET
                url_match = re.match(r'(https?://\S+|[\w.-]+\.\w+\S*)', action.strip())
                if url_match:
                    method = "GET"
                    url = url_match.group(1)

        # Parse headers from action: header:Key=Value
        hdr_matches = re.findall(r'header:\s*(\S+?)=(\S+)', action, re.IGNORECASE)
        for k, v in hdr_matches:
            headers[k] = v

        # Parse body from action: body:{...}
        body_match = re.search(r'body:\s*(\{.+\})', action, re.IGNORECASE)
        if body_match and not body and not json_body:
            try:
                json_body = json.loads(body_match.group(1))
            except json.JSONDecodeError:
                body = body_match.group(1)

        if method and url:
            return self.request(method=method, url=url, headers=headers,
                                body=body, json_body=json_body, auth=auth, timeout=timeout)

        return {
            "success": False,
            "error": f"Could not parse: {action}. "
                     "Try: 'GET https://api.example.com/data' or 'POST https://api.example.com/data body:{\"key\":\"value\"}'"
        }


# Singleton
api_tester_tool = APITesterTool()
