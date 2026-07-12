package com.aura.notifications

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NotificationCaptureStoreTest {

    @Test
    fun `upsert replaces matching notification and orders newest first`() {
        val store = NotificationCaptureStore()
        store.setConnected(true)
        store.upsert(CapturedNotification("a", "pkg.one", "Old", "first", 1L))
        store.upsert(CapturedNotification("b", "pkg.two", "Second", "second", 2L))
        store.upsert(CapturedNotification("a", "pkg.one", "Updated", "latest", 3L))

        val rows = store.snapshot(10)
        assertEquals(listOf("a", "b"), rows.map { it.key })
        assertEquals("Updated", rows.first().title)
        assertEquals(2, rows.size)
    }

    @Test
    fun `remove and clear keep listener state coherent`() {
        val store = NotificationCaptureStore()
        assertFalse(store.connected.value)
        store.setConnected(true)
        store.upsert(CapturedNotification("a", "pkg", "Title", "Text", 1L))
        store.remove("a")
        assertTrue(store.snapshot(10).isEmpty())

        store.upsert(CapturedNotification("b", "pkg", "Title", "Text", 2L))
        store.clear()
        assertTrue(store.snapshot(10).isEmpty())
        assertTrue(store.connected.value)
    }

    @Test
    fun `snapshot clamps limits and returns immutable copy`() {
        val store = NotificationCaptureStore()
        store.setConnected(true)
        repeat(60) { index ->
            store.upsert(CapturedNotification("$index", "pkg", "Title $index", "", index.toLong()))
        }

        val rows = store.snapshot(500)
        assertEquals(50, rows.size)
        assertEquals("59", rows.first().key)
    }
}
