package com.aura.core.url

import java.net.InetAddress
import java.net.URI
import java.net.UnknownHostException

/**
 * Shared SSRF guard for all tools that fetch arbitrary URLs.
 *
 * Extracted from [com.aura.tools.FirecrawlFetchTool] so that every
 * URL-fetching tool applies the same scheme + private-IP checks
 * without duplicating the logic. The guard is intentionally strict:
 * only http/https schemes, no localhost, no private/loopback/link-local
 * addresses.
 *
 * Usage:
 * ```
 * val error = SsrfGuard.validate(url)
 * if (error != null) return ToolResult.Error(error, "ssrf_guard")
 * // ... proceed with fetch ...
 * ```
 */
object SsrfGuard {

    /**
     * Validate [url] against the SSRF guard.
     *
     * @return null if the URL is safe to fetch, or an error message
     *   explaining why it was rejected.
     */
    fun validate(url: String): String? {
        val uri = try {
            URI(url)
        } catch (e: Exception) {
            return "invalid URL: ${e.message}"
        }

        val scheme = uri.scheme ?: ""
        if (scheme != "http" && scheme != "https") {
            return "only http/https URLs are allowed"
        }

        val host = uri.host ?: return "URL has no host"

        // Reject bare localhost hostnames
        if (host == "localhost" || host == "localhost.localdomain") {
            return "access to localhost is not allowed"
        }

        // Resolve and check for private / loopback / link-local IPs
        try {
            val addr = InetAddress.getByName(host)
            if (isPrivateAddress(addr)) {
                return "access to private IP is not allowed"
            }
        } catch (_: UnknownHostException) {
            // If we can't resolve the host, let the HTTP client try.
            // It will fail with its own error if the host doesn't exist.
            // We do NOT reject unresolved hosts because DNS may be flaky
            // in the moment and a legitimate external host could fail
            // to resolve temporarily.
        }

        return null
    }

    /**
     * Returns true if [addr] is a private, loopback, link-local, or
     * unique-local IPv6 address — i.e. something that should never be
     * reachable from a cloud service.
     */
    private fun isPrivateAddress(addr: InetAddress): Boolean {
        // Covers 127.x.x.x, ::1
        if (addr.isLoopbackAddress) return true
        // Covers 169.254.x.x, fe80::/10
        if (addr.isLinkLocalAddress) return true
        // Covers 10.x.x.x, 172.16-31.x.x, 192.168.x.x, fec0::/10
        if (addr.isSiteLocalAddress) return true

        // IPv6 unique local address range fc00::/7
        val raw = addr.address ?: return false
        if (raw.size == 16 && (raw[0].toInt() and 0xfe) == 0xfc) return true

        return false
    }
}