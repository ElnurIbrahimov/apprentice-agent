package com.aura.core.url

import okhttp3.OkHttpClient
import org.junit.Test
import java.net.InetAddress
import java.net.UnknownHostException
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull

/**
 * Unit tests for the shared [SsrfGuard] URL validator.
 *
 * Covers the scheme, localhost, and private-IP rejection paths.
 * External-host behavior is tested with an injected resolver so this suite
 * never depends on live DNS.
 */
class SsrfGuardTest {

    @Test
    fun `rejects non-http schemes`() {
        assertNotNull(SsrfGuard.validate("file:///etc/passwd"))
        assertNotNull(SsrfGuard.validate("javascript:alert(1)"))
        assertNotNull(SsrfGuard.validate("data:text/html,<script>alert(1)</script>"))
        assertNotNull(SsrfGuard.validate("ftp://example.com/file"))
    }

    @Test
    fun `rejects localhost hostnames`() {
        assertNotNull(SsrfGuard.validate("http://localhost/"))
        assertNotNull(SsrfGuard.validate("http://localhost:8080/"))
        assertNotNull(SsrfGuard.validate("http://localhost.localdomain/"))
        assertNotNull(SsrfGuard.validate("https://localhost/admin"))
    }

    @Test
    fun `rejects private IP addresses`() {
        assertNotNull(SsrfGuard.validate("http://127.0.0.1/"))
        assertNotNull(SsrfGuard.validate("http://10.0.0.1/"))
        assertNotNull(SsrfGuard.validate("http://192.168.1.1/"))
        assertNotNull(SsrfGuard.validate("http://172.16.0.1/"))
        assertNotNull(SsrfGuard.validate("http://169.254.169.254/latest/meta-data/"))
        assertNotNull(SsrfGuard.validate("http://[::1]/"))
    }

    @Test
    fun `rejects invalid URLs`() {
        assertNotNull(SsrfGuard.validate("not a url"))
        assertNotNull(SsrfGuard.validate("://missing-scheme"))
    }

    @Test
    fun `allows external http URLs`() {
        val public = InetAddress.getByAddress(byteArrayOf(93, 184.toByte(), 216.toByte(), 34))

        val result = SsrfGuard.inspect("https://example.com/page") { listOf(public) }

        assertIs<SsrfValidation.Safe>(result)
        assertEquals("example.com", result.host)
    }

    @Test
    fun `rejects when any resolved address is private`() {
        val public = InetAddress.getByAddress(byteArrayOf(93, 184.toByte(), 216.toByte(), 34))
        val private = InetAddress.getByAddress(byteArrayOf(10, 0, 0, 1))

        val result = SsrfGuard.inspect("https://mixed.example") { listOf(public, private) }

        assertIs<SsrfValidation.Blocked>(result)
        assertEquals("access to private IP is not allowed", result.reason)
    }

    @Test
    fun `resolution failure is blocked instead of delegated to http client`() {
        val result = SsrfGuard.inspect("https://missing.example") {
            throw UnknownHostException("missing")
        }

        assertIs<SsrfValidation.Blocked>(result)
        assertEquals("could not resolve URL host", result.reason)
    }

    @Test
    fun `pinned client disables redirects and serves only validated addresses`() {
        val address = InetAddress.getByAddress(byteArrayOf(93, 184.toByte(), 216.toByte(), 34))
        val safe = assertIs<SsrfValidation.Safe>(
            SsrfGuard.inspect("https://public.example/page") { listOf(address) },
        )

        val client = SsrfGuard.pinnedClient(OkHttpClient(), safe)

        assertFalse(client.followRedirects)
        assertFalse(client.followSslRedirects)
        assertEquals(listOf(address), client.dns.lookup("public.example"))
        assertFailsWith<UnknownHostException> { client.dns.lookup("redirect.example") }
    }

    @Test
    fun `rejects URL with no host`() {
        assertNotNull(SsrfGuard.validate("http:///path"))
    }
}
