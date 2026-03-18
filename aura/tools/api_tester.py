"""API Tester tool — World-class HTTP client for testing REST APIs.

Features:
- Request collections (save, organize, replay)
- Auth management (bearer, basic, api_key)
- Response analysis (diff, extract via JSON path)
- Chain requests with variable interpolation
- History with search (last 100 requests)
"""

import base64
import copy
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MAX_RESPONSE_BODY = 10_000

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "api_tester"
HISTORY_FILE = DATA_DIR / "history.json"
COLLECTIONS_FILE = DATA_DIR / "collections.json"
AUTH_FILE = DATA_DIR / "auth.json"
VARIABLES_FILE = DATA_DIR / "variables.json"

# Legacy history location — migrate on first load
LEGACY_HISTORY = Path(__file__).parent.parent.parent / "data" / "api_tester_history.json"


# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

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
    name: str = ""
    collection: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SavedRequest:
    """A request saved in a collection."""
    name: str
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    json_body: Any = None
    auth_type: Optional[str] = None  # override collection auth
    collection: str = "default"
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedRequest":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
#  JSON path extraction helper
# ---------------------------------------------------------------------------

def _json_path_extract(data: Any, path: str) -> Any:
    """Simple JSON path extractor. Supports dot notation and array indices.

    Examples:
        "user.name"          -> data["user"]["name"]
        "items[0].id"        -> data["items"][0]["id"]
        "results[*].name"    -> [item["name"] for item in data["results"]]
    """
    if not path:
        return data

    wildcard_segments = path.split("[*]")
    if len(wildcard_segments) > 1:
        # Handle wildcard: extract prefix, recurse on each element
        prefix = wildcard_segments[0].rstrip(".")
        suffix = "[*]".join(wildcard_segments[1:]).lstrip(".")
        base = _json_path_extract(data, prefix) if prefix else data
        if not isinstance(base, list):
            return None
        results = []
        for item in base:
            val = _json_path_extract(item, suffix) if suffix else item
            if val is not None:
                results.append(val)
        return results

    tokens = re.split(r'\.|\[', path)
    current = data
    for token in tokens:
        if not token:
            continue
        token = token.rstrip("]")
        if isinstance(current, dict):
            current = current.get(token)
        elif isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


# ---------------------------------------------------------------------------
#  Response differ
# ---------------------------------------------------------------------------

def _diff_responses(old: Dict[str, Any], new: Dict[str, Any]) -> str:
    """Compare two response dicts, highlighting differences."""
    lines: List[str] = []

    # Status
    os, ns = old.get("status_code"), new.get("status_code")
    if os != ns:
        lines.append(f"  Status: {os} -> {ns}")

    # Timing
    ot, nt = old.get("elapsed_ms", 0), new.get("elapsed_ms", 0)
    diff_ms = nt - ot
    sign = "+" if diff_ms >= 0 else ""
    lines.append(f"  Time: {ot:.0f}ms -> {nt:.0f}ms ({sign}{diff_ms:.0f}ms)")

    # Headers diff
    oh = old.get("response_headers", {})
    nh = new.get("response_headers", {})
    all_keys = set(oh.keys()) | set(nh.keys())
    header_diffs = []
    for k in sorted(all_keys):
        ov, nv = oh.get(k), nh.get(k)
        if ov != nv:
            if ov is None:
                header_diffs.append(f"    + {k}: {nv}")
            elif nv is None:
                header_diffs.append(f"    - {k}: {ov}")
            else:
                header_diffs.append(f"    ~ {k}: {ov} -> {nv}")
    if header_diffs:
        lines.append("  Headers changed:")
        lines.extend(header_diffs)

    # Body diff (JSON-aware)
    ob = old.get("response_body", "")
    nb = new.get("response_body", "")
    if ob == nb:
        lines.append("  Body: identical")
    else:
        try:
            oj = json.loads(ob)
            nj = json.loads(nb)
            body_diffs = _json_diff(oj, nj, prefix="    ")
            if body_diffs:
                lines.append("  Body (JSON diff):")
                lines.extend(body_diffs[:30])
                if len(body_diffs) > 30:
                    lines.append(f"    ... and {len(body_diffs) - 30} more changes")
            else:
                lines.append("  Body: identical (JSON)")
        except (json.JSONDecodeError, TypeError):
            # Plain text diff — just report size change
            lines.append(f"  Body: changed ({len(ob)} -> {len(nb)} chars)")

    return "\n".join(lines)


def _json_diff(old: Any, new: Any, prefix: str = "", path: str = "$") -> List[str]:
    """Recursive JSON diff, returns list of change descriptions."""
    diffs: List[str] = []
    if type(old) != type(new):
        diffs.append(f"{prefix}{path}: type {type(old).__name__} -> {type(new).__name__}")
        return diffs
    if isinstance(old, dict):
        all_keys = set(old.keys()) | set(new.keys())
        for k in sorted(all_keys):
            kpath = f"{path}.{k}"
            if k not in old:
                diffs.append(f"{prefix}+ {kpath}: {_short(new[k])}")
            elif k not in new:
                diffs.append(f"{prefix}- {kpath}: {_short(old[k])}")
            else:
                diffs.extend(_json_diff(old[k], new[k], prefix, kpath))
    elif isinstance(old, list):
        for i in range(max(len(old), len(new))):
            ipath = f"{path}[{i}]"
            if i >= len(old):
                diffs.append(f"{prefix}+ {ipath}: {_short(new[i])}")
            elif i >= len(new):
                diffs.append(f"{prefix}- {ipath}: {_short(old[i])}")
            else:
                diffs.extend(_json_diff(old[i], new[i], prefix, ipath))
    else:
        if old != new:
            diffs.append(f"{prefix}~ {path}: {_short(old)} -> {_short(new)}")
    return diffs


def _short(val: Any, maxlen: int = 60) -> str:
    s = json.dumps(val) if not isinstance(val, str) else val
    return s if len(s) <= maxlen else s[:maxlen] + "..."


# ---------------------------------------------------------------------------
#  Main Tool
# ---------------------------------------------------------------------------

class APITesterTool:
    """World-class HTTP client for testing REST APIs."""

    name = "api_tester"
    description = "Test REST APIs — collections, auth, chaining, diff, JSON extraction"

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []
        self._collections: Dict[str, List[Dict[str, Any]]] = {}  # name -> list of saved requests
        self._auth: Dict[str, Dict[str, Any]] = {}  # profile_name -> {type, credentials}
        self._variables: Dict[str, str] = {}  # key -> value for interpolation
        self._load_all()

    # -----------------------------------------------------------------------
    #  Persistence
    # -----------------------------------------------------------------------

    def _load_all(self):
        self._load_history()
        self._load_collections()
        self._load_auth()
        self._load_variables()

    def _load_history(self):
        # Migrate from legacy location
        if LEGACY_HISTORY.exists() and not HISTORY_FILE.exists():
            try:
                import shutil
                shutil.move(str(LEGACY_HISTORY), str(HISTORY_FILE))
            except Exception:
                pass

        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._history = []

    def _save_history(self):
        try:
            if len(self._history) > 100:
                self._history = self._history[-100:]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save history: %s", e)

    def _load_collections(self):
        if COLLECTIONS_FILE.exists():
            try:
                with open(COLLECTIONS_FILE, "r", encoding="utf-8") as f:
                    self._collections = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._collections = {}

    def _save_collections(self):
        try:
            with open(COLLECTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._collections, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save collections: %s", e)

    def _load_auth(self):
        if AUTH_FILE.exists():
            try:
                with open(AUTH_FILE, "r", encoding="utf-8") as f:
                    self._auth = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._auth = {}

    def _save_auth(self):
        try:
            with open(AUTH_FILE, "w", encoding="utf-8") as f:
                json.dump(self._auth, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save auth: %s", e)

    def _load_variables(self):
        if VARIABLES_FILE.exists():
            try:
                with open(VARIABLES_FILE, "r", encoding="utf-8") as f:
                    self._variables = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._variables = {}

    def _save_variables(self):
        try:
            with open(VARIABLES_FILE, "w", encoding="utf-8") as f:
                json.dump(self._variables, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save variables: %s", e)

    # -----------------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------------

    def _truncate(self, text: str, max_len: int = MAX_RESPONSE_BODY) -> str:
        if len(text) <= max_len:
            return text
        half = max_len // 2
        return text[:half] + f"\n...[truncated {len(text) - max_len} chars]...\n" + text[-half:]

    def _interpolate(self, text: str) -> str:
        """Replace {{var_name}} with stored variable values."""
        if not text or "{{" not in text:
            return text

        def _replace(m):
            key = m.group(1).strip()
            return self._variables.get(key, m.group(0))

        return re.sub(r"\{\{(.+?)\}\}", _replace, text)

    def _interpolate_dict(self, d: Dict[str, str]) -> Dict[str, str]:
        return {self._interpolate(k): self._interpolate(v) for k, v in d.items()}

    def _apply_auth(self, headers: Dict[str, str], auth_profile: str = None) -> Dict[str, str]:
        """Apply auth from a profile (or 'default') to headers."""
        profile_name = auth_profile or "default"
        auth = self._auth.get(profile_name)
        if not auth:
            return headers

        headers = dict(headers)
        auth_type = auth.get("type", "").lower()

        if auth_type == "bearer":
            token = self._interpolate(auth.get("token", ""))
            headers.setdefault("Authorization", f"Bearer {token}")
        elif auth_type == "basic":
            username = self._interpolate(auth.get("username", ""))
            password = self._interpolate(auth.get("password", ""))
            cred = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers.setdefault("Authorization", f"Basic {cred}")
        elif auth_type == "api_key":
            key_name = auth.get("header", "X-API-Key")
            key_value = self._interpolate(auth.get("key", ""))
            headers.setdefault(key_name, key_value)

        return headers

    def _validate_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Validate and normalize URL. Returns (url, error)."""
        parsed = urlparse(url)
        if not parsed.scheme:
            url = "https://" + url
        if not urlparse(url).netloc:
            return None, f"Invalid URL: {url}"

        # SSRF protection
        import ipaddress
        import socket as _socket
        _parsed = urlparse(url)
        _hostname = _parsed.hostname or ""
        _ssrf_blocked = [
            "169.254.169.254", "metadata.google.internal",
            "169.254.170.2", "168.63.129.16", "0.0.0.0", "10.0.0.1",
        ]
        if _hostname in _ssrf_blocked:
            return None, "Blocked: metadata endpoint"

        _hostname_lower = _hostname.lower()
        if _hostname_lower in ("localhost", "[::1]", "::1"):
            return None, "Blocked: localhost/loopback addresses not allowed"

        try:
            _ip = ipaddress.ip_address(_hostname)
            if _ip.is_private or _ip.is_loopback or _ip.is_link_local or _ip.is_reserved:
                return None, "Blocked: private/loopback IP addresses not allowed"
        except ValueError:
            try:
                _resolved = _socket.getaddrinfo(_hostname, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
                for _family, _type, _proto, _canonname, _sockaddr in _resolved:
                    _resolved_ip = ipaddress.ip_address(_sockaddr[0])
                    if _resolved_ip.is_private or _resolved_ip.is_loopback or _resolved_ip.is_link_local or _resolved_ip.is_reserved:
                        return None, "Blocked: hostname resolves to private/loopback IP"
            except (_socket.gaierror, OSError):
                return None, "Blocked: DNS resolution failed for hostname"

        return url, None

    # -----------------------------------------------------------------------
    #  Core: request()
    # -----------------------------------------------------------------------

    def request(self, method: str, url: str, headers: Dict[str, str] = None,
                body: str = None, json_body: Any = None, auth: tuple = None,
                auth_profile: str = None, timeout: int = 30, name: str = "",
                extract_vars: Dict[str, str] = None) -> dict:
        """Send an HTTP request.

        Args:
            method: HTTP method.
            url: Target URL.
            headers: Request headers.
            body: Raw body string.
            json_body: JSON body (will be serialized).
            auth: (username, password) tuple for requests lib.
            auth_profile: Name of stored auth profile to auto-apply.
            timeout: Request timeout in seconds.
            name: Label for this request.
            extract_vars: Dict of {var_name: json_path} to extract from response
                         and store as variables for later use.
        """
        try:
            import requests as req_lib
        except ImportError:
            return {"success": False, "error": "requests library not installed. Run: pip install requests"}

        method = method.upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
            return {"success": False, "error": f"Invalid method: {method}"}

        # Interpolate variables
        url = self._interpolate(url)
        url, url_err = self._validate_url(url)
        if url_err:
            return {"success": False, "error": url_err}

        req_headers = self._interpolate_dict(dict(headers or {}))

        # Apply stored auth
        req_headers = self._apply_auth(req_headers, auth_profile)

        req_body = None
        if json_body is not None:
            if isinstance(json_body, str):
                json_body = self._interpolate(json_body)
                req_body = json_body
            else:
                # Interpolate string values inside the dict/list
                raw = json.dumps(json_body)
                raw = self._interpolate(raw)
                req_body = raw
                json_body = json.loads(raw)
            req_headers.setdefault("Content-Type", "application/json")
        elif body:
            req_body = self._interpolate(body)

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
            send_json = None
            send_data = None
            if req_headers.get("Content-Type") == "application/json" and req_body:
                try:
                    send_json = json.loads(req_body)
                except (json.JSONDecodeError, TypeError):
                    send_data = req_body
            elif req_body:
                send_data = req_body

            resp = req_lib.request(
                method=method,
                url=url,
                headers=req_headers if req_headers else None,
                data=send_data,
                json=send_json,
                auth=auth,
                timeout=timeout,
                allow_redirects=True,
            )
            elapsed = (time.time() - start) * 1000

            record.status_code = resp.status_code
            record.elapsed_ms = round(elapsed, 2)
            record.response_headers = dict(resp.headers)
            record.content_type = resp.headers.get("Content-Type", "")

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
                "response": f"Request failed: {record.error}",
            }

        # Extract variables from response if requested
        extracted = {}
        if extract_vars and record.response_body:
            try:
                resp_json = json.loads(record.response_body)
                for var_name, json_path in extract_vars.items():
                    val = _json_path_extract(resp_json, json_path)
                    if val is not None:
                        self._variables[var_name] = str(val) if not isinstance(val, str) else val
                        extracted[var_name] = self._variables[var_name]
                if extracted:
                    self._save_variables()
            except (json.JSONDecodeError, TypeError):
                pass

        # Pretty format JSON response
        body_preview = record.response_body
        if "json" in record.content_type.lower():
            try:
                parsed_json = json.loads(record.response_body)
                body_preview = json.dumps(parsed_json, indent=2)[:2000]
            except (json.JSONDecodeError, TypeError):
                pass

        # Build status line with color hints
        sc = record.status_code
        status_tag = "OK" if 200 <= sc < 300 else "REDIRECT" if 300 <= sc < 400 else "CLIENT_ERR" if 400 <= sc < 500 else "SERVER_ERR"

        result = {
            "success": True,
            "request_id": record.id,
            "method": method,
            "url": url,
            "status_code": record.status_code,
            "status_tag": status_tag,
            "elapsed_ms": record.elapsed_ms,
            "content_type": record.content_type,
            "response_headers": record.response_headers,
            "body": record.response_body,
            "body_length": len(record.response_body),
            "response": f"{method} {url} -> {record.status_code} {status_tag} ({record.elapsed_ms:.0f}ms)\n{body_preview[:1000]}",
        }
        if extracted:
            result["extracted_vars"] = extracted

        return result

    # -----------------------------------------------------------------------
    #  Convenience methods (unchanged interface)
    # -----------------------------------------------------------------------

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

    def patch(self, url: str, body: str = None, json_body: Any = None,
              headers: Dict[str, str] = None, **kwargs) -> dict:
        return self.request("PATCH", url, headers=headers, body=body, json_body=json_body, **kwargs)

    # -----------------------------------------------------------------------
    #  Collections
    # -----------------------------------------------------------------------

    def save_request(self, name: str, method: str, url: str,
                     headers: Dict[str, str] = None, body: str = None,
                     json_body: Any = None, collection: str = "default",
                     description: str = "", auth_type: str = None) -> dict:
        """Save a request to a named collection."""
        if not name:
            return {"success": False, "error": "Request name is required"}

        saved = SavedRequest(
            name=name,
            method=method.upper(),
            url=url,
            headers=headers or {},
            body=body,
            json_body=json_body,
            collection=collection,
            description=description,
            auth_type=auth_type,
        )

        if collection not in self._collections:
            self._collections[collection] = []

        # Replace if same name exists in collection
        coll = self._collections[collection]
        replaced = False
        for i, r in enumerate(coll):
            if r.get("name") == name:
                coll[i] = saved.to_dict()
                replaced = True
                break
        if not replaced:
            coll.append(saved.to_dict())

        self._save_collections()
        action = "updated" if replaced else "saved"
        return {
            "success": True,
            "name": name,
            "collection": collection,
            "response": f"Request '{name}' {action} in collection '{collection}'",
        }

    def list_requests(self, collection: str = None) -> dict:
        """List saved requests, optionally filtered by collection."""
        if collection:
            items = self._collections.get(collection, [])
            if not items:
                return {"success": True, "count": 0, "requests": [],
                        "response": f"Collection '{collection}' is empty or doesn't exist"}
            formatted = [f"  [{r['method']}] {r['name']} -> {r['url']}" for r in items]
            return {
                "success": True,
                "count": len(items),
                "collection": collection,
                "requests": items,
                "response": f"Collection '{collection}' ({len(items)} requests):\n" + "\n".join(formatted),
            }

        # All collections
        all_formatted = []
        total = 0
        for coll_name, items in self._collections.items():
            all_formatted.append(f"\n[{coll_name}] ({len(items)} requests)")
            for r in items:
                all_formatted.append(f"  [{r['method']}] {r['name']} -> {r['url']}")
            total += len(items)

        if not all_formatted:
            return {"success": True, "count": 0, "collections": {},
                    "response": "No saved requests. Use save_request() to create one."}

        return {
            "success": True,
            "count": total,
            "collections": self._collections,
            "response": f"{total} saved request(s) in {len(self._collections)} collection(s):" + "\n".join(all_formatted),
        }

    def run_saved(self, name: str, collection: str = None,
                  override_headers: Dict[str, str] = None,
                  override_body: Any = None,
                  extract_vars: Dict[str, str] = None) -> dict:
        """Execute a saved request by name."""
        # Search all collections or specific one
        target = None
        search_in = {collection: self._collections.get(collection, [])} if collection else self._collections

        for coll_name, items in search_in.items():
            for r in items:
                if r.get("name") == name:
                    target = r
                    break
            if target:
                break

        if not target:
            return {"success": False, "error": f"Saved request '{name}' not found"}

        headers = dict(target.get("headers", {}))
        if override_headers:
            headers.update(override_headers)

        body = override_body or target.get("body")
        json_body = target.get("json_body") if not override_body else None
        if override_body and isinstance(override_body, (dict, list)):
            json_body = override_body
            body = None

        return self.request(
            method=target["method"],
            url=target["url"],
            headers=headers,
            body=body,
            json_body=json_body,
            auth_profile=target.get("auth_type"),
            name=name,
            extract_vars=extract_vars,
        )

    def delete_request(self, name: str, collection: str = "default") -> dict:
        """Delete a saved request from a collection."""
        coll = self._collections.get(collection, [])
        original_len = len(coll)
        coll = [r for r in coll if r.get("name") != name]
        if len(coll) == original_len:
            return {"success": False, "error": f"Request '{name}' not found in collection '{collection}'"}
        self._collections[collection] = coll
        if not coll:
            del self._collections[collection]
        self._save_collections()
        return {"success": True, "response": f"Request '{name}' deleted from '{collection}'"}

    def delete_collection(self, collection: str) -> dict:
        """Delete an entire collection."""
        if collection not in self._collections:
            return {"success": False, "error": f"Collection '{collection}' not found"}
        count = len(self._collections[collection])
        del self._collections[collection]
        self._save_collections()
        return {"success": True, "response": f"Collection '{collection}' deleted ({count} requests removed)"}

    # -----------------------------------------------------------------------
    #  Auth Management
    # -----------------------------------------------------------------------

    def set_auth(self, auth_type: str, credentials: Dict[str, str],
                 profile: str = "default") -> dict:
        """Set authentication credentials for a profile.

        Args:
            auth_type: "bearer", "basic", or "api_key"
            credentials: Depends on type:
                bearer:  {"token": "..."}
                basic:   {"username": "...", "password": "..."}
                api_key: {"key": "...", "header": "X-API-Key"}  (header is optional)
            profile: Auth profile name (default: "default")
        """
        auth_type = auth_type.lower()
        if auth_type not in ("bearer", "basic", "api_key"):
            return {"success": False, "error": f"Invalid auth type: {auth_type}. Use: bearer, basic, api_key"}

        if auth_type == "bearer" and "token" not in credentials:
            return {"success": False, "error": "Bearer auth requires 'token' in credentials"}
        if auth_type == "basic" and ("username" not in credentials or "password" not in credentials):
            return {"success": False, "error": "Basic auth requires 'username' and 'password' in credentials"}
        if auth_type == "api_key" and "key" not in credentials:
            return {"success": False, "error": "API key auth requires 'key' in credentials"}

        self._auth[profile] = {"type": auth_type, **credentials}
        self._save_auth()
        return {
            "success": True,
            "profile": profile,
            "type": auth_type,
            "response": f"Auth profile '{profile}' set ({auth_type})",
        }

    def get_auth(self, profile: str = "default") -> dict:
        """Get auth profile info (masks secrets)."""
        auth = self._auth.get(profile)
        if not auth:
            return {"success": False, "error": f"Auth profile '{profile}' not found"}

        masked = dict(auth)
        for key in ("token", "password", "key"):
            if key in masked and masked[key]:
                val = masked[key]
                masked[key] = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"

        return {"success": True, "profile": profile, "auth": masked,
                "response": f"Auth '{profile}': {json.dumps(masked)}"}

    def list_auth(self) -> dict:
        """List all auth profiles (secrets masked)."""
        profiles = []
        for name, auth in self._auth.items():
            profiles.append(f"  [{name}] type={auth.get('type', '?')}")
        if not profiles:
            return {"success": True, "count": 0, "response": "No auth profiles configured"}
        return {
            "success": True,
            "count": len(self._auth),
            "profiles": list(self._auth.keys()),
            "response": f"{len(self._auth)} auth profile(s):\n" + "\n".join(profiles),
        }

    def remove_auth(self, profile: str = "default") -> dict:
        """Remove an auth profile."""
        if profile not in self._auth:
            return {"success": False, "error": f"Auth profile '{profile}' not found"}
        del self._auth[profile]
        self._save_auth()
        return {"success": True, "response": f"Auth profile '{profile}' removed"}

    # -----------------------------------------------------------------------
    #  Variables
    # -----------------------------------------------------------------------

    def set_var(self, name: str, value: str) -> dict:
        """Set a variable for interpolation in requests."""
        self._variables[name] = value
        self._save_variables()
        return {"success": True, "response": f"Variable '{name}' set"}

    def get_var(self, name: str) -> dict:
        """Get a variable value."""
        if name not in self._variables:
            return {"success": False, "error": f"Variable '{name}' not found"}
        return {"success": True, "name": name, "value": self._variables[name],
                "response": f"{{{{name}}}} = {self._variables[name]}"}

    def list_vars(self) -> dict:
        """List all stored variables."""
        if not self._variables:
            return {"success": True, "count": 0, "response": "No variables set"}
        formatted = [f"  {{{{{k}}}}} = {v}" for k, v in self._variables.items()]
        return {
            "success": True,
            "count": len(self._variables),
            "variables": dict(self._variables),
            "response": f"{len(self._variables)} variable(s):\n" + "\n".join(formatted),
        }

    def clear_vars(self) -> dict:
        """Clear all variables."""
        count = len(self._variables)
        self._variables.clear()
        self._save_variables()
        return {"success": True, "response": f"Cleared {count} variable(s)"}

    # -----------------------------------------------------------------------
    #  Response Analysis
    # -----------------------------------------------------------------------

    def extract(self, request_id: str = None, json_path: str = "",
                response_body: str = None) -> dict:
        """Extract values from a response using JSON path notation.

        Args:
            request_id: ID of a previous request (from history).
            json_path: Dot-notation path (e.g. "data.users[0].name", "items[*].id").
            response_body: Raw JSON string to extract from (alternative to request_id).
        """
        body = response_body
        if not body and request_id:
            for r in reversed(self._history):
                if r.get("id") == request_id:
                    body = r.get("response_body", "")
                    break
            if body is None:
                return {"success": False, "error": f"Request '{request_id}' not found"}

        if not body:
            return {"success": False, "error": "No response body to extract from"}

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return {"success": False, "error": "Response body is not valid JSON"}

        result = _json_path_extract(data, json_path)
        if result is None:
            return {"success": False, "error": f"Path '{json_path}' not found in response"}

        formatted = json.dumps(result, indent=2) if not isinstance(result, str) else result
        return {
            "success": True,
            "path": json_path,
            "value": result,
            "response": f"Extracted '{json_path}':\n{formatted[:2000]}",
        }

    def compare(self, request_id_old: str, request_id_new: str) -> dict:
        """Compare two request/response records, showing what changed."""
        old_rec = new_rec = None
        for r in self._history:
            if r.get("id") == request_id_old:
                old_rec = r
            if r.get("id") == request_id_new:
                new_rec = r

        # Also allow comparing by name (find latest two runs)
        if not old_rec or not new_rec:
            return {"success": False, "error": "One or both request IDs not found in history"}

        diff = _diff_responses(old_rec, new_rec)
        return {
            "success": True,
            "old_id": request_id_old,
            "new_id": request_id_new,
            "response": f"Diff [{request_id_old}] vs [{request_id_new}]:\n{diff}",
        }

    def compare_by_name(self, name: str) -> dict:
        """Compare the last two runs of a named request."""
        matches = [r for r in self._history if r.get("name") == name]
        if len(matches) < 2:
            return {"success": False, "error": f"Need at least 2 runs of '{name}' in history, found {len(matches)}"}
        return self.compare(matches[-2]["id"], matches[-1]["id"])

    # -----------------------------------------------------------------------
    #  Chain Requests
    # -----------------------------------------------------------------------

    def run_chain(self, steps: List[Dict[str, Any]]) -> dict:
        """Run a sequence of requests, passing variables between them.

        Each step is a dict with:
            - name: (optional) saved request name to run, OR
            - method + url: for inline requests
            - headers: (optional) additional headers
            - body / json_body: (optional) request body
            - extract_vars: (optional) dict {var_name: json_path} to extract and store
            - auth_profile: (optional) auth profile to use

        Variables extracted from step N are available in step N+1 via {{var_name}}.

        Example:
            run_chain([
                {"name": "login", "extract_vars": {"token": "data.access_token"}},
                {"name": "get_profile"},  # Uses {{token}} in saved request headers
            ])
        """
        if not steps:
            return {"success": False, "error": "No steps provided"}

        results = []
        for i, step in enumerate(steps):
            step_name = step.get("name", f"step_{i+1}")

            if "name" in step and not step.get("method"):
                # Run saved request
                result = self.run_saved(
                    name=step["name"],
                    override_headers=step.get("headers"),
                    override_body=step.get("body") or step.get("json_body"),
                    extract_vars=step.get("extract_vars"),
                )
            else:
                # Inline request
                result = self.request(
                    method=step.get("method", "GET"),
                    url=step.get("url", ""),
                    headers=step.get("headers"),
                    body=step.get("body"),
                    json_body=step.get("json_body"),
                    auth_profile=step.get("auth_profile"),
                    name=step_name,
                    extract_vars=step.get("extract_vars"),
                )

            results.append({
                "step": i + 1,
                "name": step_name,
                "success": result.get("success", False),
                "status_code": result.get("status_code"),
                "elapsed_ms": result.get("elapsed_ms", 0),
                "error": result.get("error"),
                "extracted_vars": result.get("extracted_vars", {}),
            })

            # Stop chain on failure unless step says continue_on_error
            if not result.get("success") and not step.get("continue_on_error"):
                break

        total_ms = sum(r.get("elapsed_ms", 0) for r in results)
        all_ok = all(r["success"] for r in results)

        formatted = [f"Chain: {len(results)}/{len(steps)} steps, {total_ms:.0f}ms total"]
        for r in results:
            status = f"{r['status_code']}" if r["status_code"] else "ERR"
            icon = "+" if r["success"] else "X"
            formatted.append(f"  [{icon}] Step {r['step']} ({r['name']}): {status} ({r['elapsed_ms']:.0f}ms)")
            if r.get("extracted_vars"):
                for k, v in r["extracted_vars"].items():
                    formatted.append(f"       -> {{{{{k}}}}} = {v[:60]}")
            if r.get("error"):
                formatted.append(f"       ERROR: {r['error'][:100]}")

        return {
            "success": all_ok,
            "steps_run": len(results),
            "steps_total": len(steps),
            "total_ms": round(total_ms, 2),
            "results": results,
            "variables": dict(self._variables),
            "response": "\n".join(formatted),
        }

    # -----------------------------------------------------------------------
    #  History (enhanced)
    # -----------------------------------------------------------------------

    def history(self, limit: int = 10, method: str = None,
                url_contains: str = None, status_code: int = None) -> dict:
        """Show recent request history with optional filters."""
        recent = list(self._history)

        if method:
            recent = [r for r in recent if r.get("method", "").upper() == method.upper()]
        if url_contains:
            q = url_contains.lower()
            recent = [r for r in recent if q in r.get("url", "").lower()]
        if status_code is not None:
            recent = [r for r in recent if r.get("status_code") == status_code]

        recent = recent[-limit:]
        recent.reverse()

        formatted = []
        for r in recent:
            status = r.get("status_code") or "ERR"
            name_tag = f" ({r['name']})" if r.get("name") else ""
            formatted.append(
                f"[{r['id']}] {r['method']} {r['url']} -> {status} ({r.get('elapsed_ms', 0):.0f}ms){name_tag}"
            )

        return {
            "success": True,
            "count": len(recent),
            "history": recent,
            "response": f"Last {len(recent)} request(s):\n" + "\n".join(formatted),
        }

    def search_history(self, query: str) -> dict:
        """Search history by URL, method, name, or status code."""
        q = query.lower().strip()
        matches = []
        for r in self._history:
            if (q in r.get("url", "").lower() or
                q in r.get("method", "").lower() or
                q in r.get("name", "").lower() or
                q == str(r.get("status_code", ""))):
                matches.append(r)

        matches = matches[-20:]  # Cap at 20
        formatted = []
        for r in matches:
            status = r.get("status_code") or "ERR"
            formatted.append(f"[{r['id']}] {r['method']} {r['url']} -> {status}")

        return {
            "success": True,
            "count": len(matches),
            "history": matches,
            "response": f"{len(matches)} match(es) for '{query}':\n" + "\n".join(formatted),
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

    # -----------------------------------------------------------------------
    #  Dispatch (extended)
    # -----------------------------------------------------------------------

    def execute(self, action: str, **kwargs) -> dict:
        action_lower = action.lower().strip()

        # History
        if action_lower.startswith("history"):
            return self.history(
                limit=kwargs.get("limit", 10),
                method=kwargs.get("method"),
                url_contains=kwargs.get("url_contains"),
                status_code=kwargs.get("status_code"),
            )

        # Search history
        if action_lower.startswith("search_history") or action_lower.startswith("search history"):
            query = kwargs.get("query") or (action.split(None, 2)[-1] if len(action.split()) > 2 else "")
            return self.search_history(query)

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

        # Collections
        if action_lower.startswith("save_request") or action_lower.startswith("save request"):
            return self.save_request(
                name=kwargs.get("name", ""),
                method=kwargs.get("method", "GET"),
                url=kwargs.get("url", ""),
                headers=kwargs.get("headers"),
                body=kwargs.get("body"),
                json_body=kwargs.get("json_body"),
                collection=kwargs.get("collection", "default"),
                description=kwargs.get("description", ""),
            )

        if action_lower.startswith("list_requests") or action_lower.startswith("list requests") or action_lower == "collections":
            return self.list_requests(collection=kwargs.get("collection"))

        if action_lower.startswith("run_saved") or action_lower.startswith("run saved"):
            name = kwargs.get("name") or (action.split(None, 2)[-1] if len(action.split()) > 2 else "")
            return self.run_saved(
                name=name.strip(),
                collection=kwargs.get("collection"),
                extract_vars=kwargs.get("extract_vars"),
            )

        # Auth
        if action_lower.startswith("set_auth") or action_lower.startswith("set auth"):
            return self.set_auth(
                auth_type=kwargs.get("auth_type", kwargs.get("type", "")),
                credentials=kwargs.get("credentials", {}),
                profile=kwargs.get("profile", "default"),
            )

        if action_lower.startswith("list_auth") or action_lower.startswith("list auth"):
            return self.list_auth()

        # Variables
        if action_lower.startswith("set_var") or action_lower.startswith("set var"):
            return self.set_var(name=kwargs.get("name", ""), value=kwargs.get("value", ""))

        if action_lower.startswith("list_vars") or action_lower.startswith("list vars") or action_lower == "vars":
            return self.list_vars()

        # Extract
        if action_lower.startswith("extract"):
            return self.extract(
                request_id=kwargs.get("request_id"),
                json_path=kwargs.get("json_path", kwargs.get("path", "")),
                response_body=kwargs.get("response_body"),
            )

        # Compare
        if action_lower.startswith("compare") or action_lower.startswith("diff"):
            old_id = kwargs.get("old_id") or kwargs.get("request_id_old")
            new_id = kwargs.get("new_id") or kwargs.get("request_id_new")
            name = kwargs.get("name")
            if name:
                return self.compare_by_name(name)
            if old_id and new_id:
                return self.compare(old_id, new_id)
            # Try to parse IDs from action
            ids = re.findall(r'\b([a-f0-9]{8})\b', action)
            if len(ids) >= 2:
                return self.compare(ids[0], ids[1])
            return {"success": False, "error": "Provide old_id and new_id, or a request name"}

        # Chain
        if action_lower.startswith("chain") or action_lower.startswith("run_chain"):
            steps = kwargs.get("steps", [])
            return self.run_chain(steps)

        # Parse method + URL from action (original behavior)
        method = kwargs.get("method")
        url = kwargs.get("url")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body")
        json_body = kwargs.get("json_body") or kwargs.get("json")
        auth = kwargs.get("auth")
        auth_profile = kwargs.get("auth_profile")
        timeout = kwargs.get("timeout", 30)
        extract_vars = kwargs.get("extract_vars")

        if not method or not url:
            parts = action.split(None, 1)
            if len(parts) >= 2 and parts[0].upper() in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                method = parts[0].upper()
                url_part = parts[1].strip()
                url_match = re.match(r'(https?://\S+|[\w.-]+\.\w+\S*)', url_part)
                url = url_match.group(1) if url_match else url_part
            elif len(parts) >= 1:
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
            return self.request(
                method=method, url=url, headers=headers,
                body=body, json_body=json_body, auth=auth,
                auth_profile=auth_profile, timeout=timeout,
                extract_vars=extract_vars,
            )

        return {
            "success": False,
            "error": f"Could not parse: {action}. "
                     "Try: 'GET https://...', 'save_request', 'list_requests', "
                     "'run_saved <name>', 'chain', 'set_auth', 'compare', 'extract'"
        }


# Singleton
api_tester_tool = APITesterTool()
