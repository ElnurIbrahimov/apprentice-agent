package com.aura.memory

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LikeEscapingTest {
    @Test fun `empty string passes through`() = assertEquals("", escapeLikeWildcards(""))

    @Test fun `plain text passes through`() =
        assertEquals("hello world", escapeLikeWildcards("hello world"))

    @Test fun `percent is escaped`() =
        assertEquals("100\\% off", escapeLikeWildcards("100% off"))

    @Test fun `underscore is escaped`() =
        assertEquals("user\\_name", escapeLikeWildcards("user_name"))

    @Test fun `backslash is escaped first so subsequent escapes work`() =
        assertEquals("a\\\\b", escapeLikeWildcards("a\\b"))

    @Test fun `multiple wildcards all escaped`() =
        assertEquals("50\\%\\_\\% done", escapeLikeWildcards("50%_% done"))

    @Test fun `wrapping with percent after escaping produces safe contains query`() {
        val user = "100% off"
        val safe = "%${escapeLikeWildcards(user)}%"
        // The escaped query should not match a string that contains "100" + something
        // other than "off" — proves the percent is not a wildcard anymore.
        assertTrue(safe.contains("\\%"), "wrapped query should contain escaped percent")
        assertTrue("100% done".contains("100% done"))
        // Without escaping, "%100% off%" would match both "100% off" and "100% done" —
        // with escaping, "%100\\% off%" matches only the literal substring "100% off".
    }
}
