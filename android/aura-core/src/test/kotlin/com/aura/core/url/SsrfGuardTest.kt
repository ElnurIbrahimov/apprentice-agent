package com.aura.core.url

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Unit tests for the shared [SsrfGuard] URL validator.
 *
 * Covers the scheme, localhost, and private-IP rejection paths.
 * External hosts (google.com) are expected to pass — we can't
 * guarantee DNS resolution in a unit test environment, but the
 * guard intentionally allows unresolved hosts to fall through to
 * the HTTP client.
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
        // External hosts that resolve to public IPs should pass.
        // We can't guarantee DNS in a unit test, but the guard
        // allows unresolved hosts to fall through. If DNS IS
        // available, external hosts resolve to public IPs and pass.
        // If DNS is NOT available, the guard returns null (allow).
        val result = SsrfGuard.validate("https://example.com/page")
        // Either null (allowed) or a DNS error message — both are
        // acceptable in a test environment. The key assertion is
        // that it's NOT a private-IP rejection.
        if (result != null) {
            assertEquals(true, result.contains("could not resolve") || result.contains("invalid URL"))
        }
    }

    @Test
    fun `rejects URL with no host`() {
        assertNotNull(SsrfGuard.validate("http:///path"))
    }
}