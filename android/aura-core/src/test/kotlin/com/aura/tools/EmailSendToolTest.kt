package com.aura.tools

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class EmailSendToolTest {

    @Test
    fun `accepts ordinary recipient addresses`() {
        assertTrue(isValidEmailRecipient("user@example.com"))
        assertTrue(isValidEmailRecipient("first.last+tag@example.co.uk"))
    }

    @Test
    fun `rejects mailto field injection and malformed recipients`() {
        assertFalse(isValidEmailRecipient("user@example.com?cc=attacker@example.com"))
        assertFalse(isValidEmailRecipient("user@example.com#fragment"))
        assertFalse(isValidEmailRecipient("user@example.com&bcc=attacker@example.com"))
        assertFalse(isValidEmailRecipient("not-an-email"))
        assertFalse(isValidEmailRecipient(""))
    }
}
