"""SSRF & DNS rebinding protection.

Stolen from OpenFang's approach: resolve DNS once, check IP against blocklist,
pin resolved IP for the actual request. Blocks private IPs, loopback, link-local,
and DNS rebinding attacks.

Fixes the vulnerability flagged in ENGINEERING_REVIEW_2026-03-20.md:
  "validate_url_scheme resolves at check time, HTTP client again at request time"
"""

import ipaddress
import logging
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# RFC1918 + loopback + link-local + documentation + benchmarking + carrier-grade NAT
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),      # RFC1918
    ipaddress.ip_network("192.168.0.0/16"),     # RFC1918
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("100.64.0.0/10"),      # Carrier-grade NAT
    ipaddress.ip_network("192.0.0.0/24"),       # IETF protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),       # Documentation (TEST-NET-1)
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),    # Documentation (TEST-NET-2)
    ipaddress.ip_network("203.0.113.0/24"),     # Documentation (TEST-NET-3)
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("fc00::/7"),           # Unique local
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("ff00::/8"),           # Multicast
]

# Allowed schemes
_ALLOWED_SCHEMES = {"http", "https"}

# Max URL length
_MAX_URL_LENGTH = 4096

# Dangerous ports (common internal services)
_BLOCKED_PORTS = {
    6379,   # Redis
    11211,  # Memcached
    27017,  # MongoDB
    3306,   # MySQL
    5432,   # PostgreSQL
    9200,   # Elasticsearch
    2379,   # etcd
    8500,   # Consul
    6443,   # Kubernetes API
}


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private/reserved/blocked."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Invalid IP = blocked

    # Fast path: stdlib catches most private/reserved/loopback/link-local
    if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local or addr.is_multicast:
        return True

    # Extended checks (carrier-grade NAT, documentation, benchmarking)
    for network in _BLOCKED_NETWORKS:
        if addr in network:
            return True
    return False


def _resolve_hostname(hostname: str, timeout: float = 5.0) -> list[str]:
    """Resolve hostname to IP addresses. Raises on failure.

    Uses a timeout to prevent DNS hanging (e.g., on attacker-controlled resolvers).
    """
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        ips = list({r[4][0] for r in results})
        if not ips:
            raise ValueError(f"DNS resolution returned no results for {hostname}")
        return ips
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for {hostname}: {e}")
    except socket.timeout:
        raise ValueError(f"DNS resolution timed out for {hostname} ({timeout}s)")
    finally:
        socket.setdefaulttimeout(old_timeout)


def validate_url_safe(url: str, allow_redirects: bool = False) -> Tuple[str, Optional[str]]:
    """Validate a URL is safe from SSRF attacks.

    Returns:
        (pinned_url, original_hostname) — pinned_url has the hostname replaced
        with the resolved IP to prevent DNS rebinding (TOCTOU). original_hostname
        is provided so callers can set the Host header if needed.

        If the URL already contains an IP literal, pinned_url == original url
        and original_hostname is None.

    Raises:
        ValueError: if URL is blocked (private IP, bad scheme, bad port, etc.)
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    if len(url) > _MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds max length ({_MAX_URL_LENGTH})")

    parsed = urlparse(url)

    # Scheme check
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Blocked scheme: {parsed.scheme}. Only http/https allowed.")

    # Hostname extraction
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")

    # Port check
    port = parsed.port
    if port and port in _BLOCKED_PORTS:
        raise ValueError(f"Blocked port: {port} (internal service)")

    # Check if hostname is already an IP
    try:
        addr = ipaddress.ip_address(hostname)
        if is_private_ip(str(addr)):
            raise ValueError(f"Blocked: {hostname} is a private/reserved IP")
        # Already an IP literal — no DNS rebinding possible
        return url, None
    except ValueError as e:
        if "private" in str(e).lower() or "blocked" in str(e).lower():
            raise
        # Not an IP — it's a hostname, resolve it

    # DNS resolution + IP validation (the core SSRF defense)
    resolved_ips = _resolve_hostname(hostname)

    for ip in resolved_ips:
        if is_private_ip(ip):
            raise ValueError(
                f"Blocked: {hostname} resolves to private IP {ip}. "
                f"Possible SSRF/DNS rebinding attack."
            )

    # Pin to first safe IP — replace hostname with IP in the URL to
    # eliminate the TOCTOU gap between validation and request time.
    pinned_ip = resolved_ips[0]
    logger.debug(f"[SSRF] {hostname} → {pinned_ip} (safe, pinned)")

    pinned_url = _pin_url(parsed, pinned_ip)
    return pinned_url, hostname


def _pin_url(parsed, pinned_ip: str) -> str:
    """Replace hostname with pinned IP in a parsed URL, handling IPv6 brackets."""
    from urllib.parse import urlunparse

    port_str = f":{parsed.port}" if parsed.port else ""
    # IPv6 addresses need brackets in URLs
    if ":" in pinned_ip:
        netloc = f"[{pinned_ip}]{port_str}"
    else:
        netloc = f"{pinned_ip}{port_str}"

    return urlunparse((
        parsed.scheme, netloc, parsed.path,
        parsed.params, parsed.query, parsed.fragment,
    ))


def safe_request(url: str, method: str = "GET", **kwargs) -> "requests.Response":
    """Make an HTTP request with SSRF protection.

    Resolves DNS once, validates IP, then makes request with pinned IP.
    Drop-in replacement for requests.get/post/etc.
    """
    import requests
    from urllib.parse import urlparse, urljoin

    pinned_url, original_hostname = validate_url_safe(url)

    # Set Host header to original hostname so the server sees the right vhost
    headers = kwargs.pop("headers", {})
    if original_hostname:
        headers.setdefault("Host", original_hostname)
    kwargs["headers"] = headers
    # Disable redirect following — we'll validate each redirect
    kwargs.setdefault("allow_redirects", False)
    # Timeout to prevent hanging on slow/malicious servers
    kwargs.setdefault("timeout", 30)

    response = requests.request(method, pinned_url, **kwargs)

    # Handle redirects manually with SSRF checks on each hop
    max_redirects = 5
    current_url = url  # Track original URL for resolving relative redirects
    while response.is_redirect and max_redirects > 0:
        redirect_url = response.headers.get("Location", "")
        if not redirect_url:
            break
        # Handle relative redirects (e.g., Location: /path)
        if not redirect_url.startswith(("http://", "https://")):
            redirect_url = urljoin(current_url, redirect_url)
        # Validate redirect target against SSRF (returns pinned URL)
        pinned_redirect, redir_hostname = validate_url_safe(redirect_url)
        headers_r = kwargs.get("headers", {}).copy()
        if redir_hostname:
            headers_r["Host"] = redir_hostname
        else:
            headers_r.pop("Host", None)  # IP literal, no Host override needed
        kwargs["headers"] = headers_r
        response = requests.request("GET", pinned_redirect, **kwargs)
        current_url = redirect_url
        max_redirects -= 1

    return response
