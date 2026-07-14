package com.aura.core.url

import okhttp3.Dns
import okhttp3.OkHttpClient
import java.net.InetAddress
import java.net.URI
import java.net.UnknownHostException
import java.util.Locale

sealed interface SsrfValidation {
    data class Safe(
        val url: String,
        val host: String,
        val addresses: List<InetAddress>,
    ) : SsrfValidation

    data class Blocked(val reason: String) : SsrfValidation
}

/**
 * Shared SSRF boundary for every tool that accepts an arbitrary URL.
 *
 * Validation is fail-closed and checks every DNS answer. Direct callers must
 * use [pinnedClient] so the HTTP request cannot resolve the hostname a second
 * time to a different address. Redirects are disabled; a redirect target is
 * never followed without a fresh validation/pinning cycle.
 */
object SsrfGuard {

    fun validate(url: String): String? = when (val result = inspect(url)) {
        is SsrfValidation.Safe -> null
        is SsrfValidation.Blocked -> result.reason
    }

    fun inspect(
        url: String,
        resolver: (String) -> List<InetAddress> = { host ->
            InetAddress.getAllByName(host).toList()
        },
    ): SsrfValidation {
        val uri = try {
            URI(url)
        } catch (error: Exception) {
            return SsrfValidation.Blocked("invalid URL: ${error.message}")
        }

        val scheme = uri.scheme?.lowercase(Locale.US).orEmpty()
        if (scheme != "http" && scheme != "https") {
            return SsrfValidation.Blocked("only http/https URLs are allowed")
        }
        if (uri.userInfo != null) {
            return SsrfValidation.Blocked("URL credentials are not allowed")
        }

        val host = uri.host?.removeSuffix(".")?.lowercase(Locale.US)
            ?: return SsrfValidation.Blocked("URL has no host")
        if (host == "localhost" || host == "localhost.localdomain") {
            return SsrfValidation.Blocked("access to localhost is not allowed")
        }

        val addresses = try {
            resolver(host)
        } catch (_: UnknownHostException) {
            return SsrfValidation.Blocked("could not resolve URL host")
        } catch (_: Exception) {
            return SsrfValidation.Blocked("could not resolve URL host")
        }
        if (addresses.isEmpty()) {
            return SsrfValidation.Blocked("could not resolve URL host")
        }
        if (addresses.any(::isNonPublicAddress)) {
            return SsrfValidation.Blocked("access to private IP is not allowed")
        }

        return SsrfValidation.Safe(url = url, host = host, addresses = addresses.distinct())
    }

    /**
     * Clone [base] with DNS pinned to the addresses that [inspect] approved.
     * Any unexpected hostname lookup fails, and redirects are never followed.
     */
    fun pinnedClient(base: OkHttpClient, target: SsrfValidation.Safe): OkHttpClient {
        val expectedHost = target.host
        val pinnedAddresses = target.addresses.toList()
        val pinnedDns = object : Dns {
            override fun lookup(hostname: String): List<InetAddress> {
                val normalized = hostname.removeSuffix(".").lowercase(Locale.US)
                if (normalized != expectedHost) {
                    throw UnknownHostException("unexpected redirect host")
                }
                return pinnedAddresses
            }
        }
        return base.newBuilder()
            .dns(pinnedDns)
            .followRedirects(false)
            .followSslRedirects(false)
            .build()
    }

    private fun isNonPublicAddress(address: InetAddress): Boolean {
        if (address.isAnyLocalAddress || address.isLoopbackAddress) return true
        if (address.isLinkLocalAddress || address.isSiteLocalAddress) return true
        if (address.isMulticastAddress) return true

        val raw = address.address ?: return true
        if (raw.size == 16) {
            // IPv6 unique-local fc00::/7 and documentation 2001:db8::/32.
            if ((raw[0].toInt() and 0xfe) == 0xfc) return true
            if ((raw[0].toInt() and 0xff) == 0x20 &&
                (raw[1].toInt() and 0xff) == 0x01 &&
                (raw[2].toInt() and 0xff) == 0x0d &&
                (raw[3].toInt() and 0xff) == 0xb8
            ) return true

            // IPv4-mapped IPv6 ::ffff:a.b.c.d.
            val mapped = raw.take(10).all { it.toInt() == 0 } &&
                (raw[10].toInt() and 0xff) == 0xff &&
                (raw[11].toInt() and 0xff) == 0xff
            if (mapped) return isNonPublicIpv4(raw.copyOfRange(12, 16))
            return false
        }
        return raw.size != 4 || isNonPublicIpv4(raw)
    }

    private fun isNonPublicIpv4(raw: ByteArray): Boolean {
        val a = raw[0].toInt() and 0xff
        val b = raw[1].toInt() and 0xff
        val c = raw[2].toInt() and 0xff
        return when {
            a == 0 || a == 10 || a == 127 -> true
            a == 100 && b in 64..127 -> true // carrier-grade NAT
            a == 169 && b == 254 -> true
            a == 172 && b in 16..31 -> true
            a == 192 && b == 168 -> true
            a == 192 && b == 0 && c in 0..2 -> true
            a == 198 && b in 18..19 -> true // benchmark networks
            a == 198 && b == 51 && c == 100 -> true
            a == 203 && b == 0 && c == 113 -> true
            a >= 224 -> true
            else -> false
        }
    }
}
