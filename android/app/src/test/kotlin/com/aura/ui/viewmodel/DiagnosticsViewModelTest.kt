package com.aura.ui.viewmodel

import android.content.Context
import com.aura.core.error.CrashLogEntry
import com.aura.core.error.CrashLogger
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import java.io.File
import java.nio.file.Files
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class DiagnosticsViewModelTest {

    private lateinit var logger: CrashLogger
    private lateinit var context: Context
    private lateinit var cacheDir: File

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        logger = mockk(relaxed = true)
        context = mockk()
        cacheDir = Files.createTempDirectory("diagnostics-vm").toFile()
        every { context.cacheDir } returns cacheDir
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
        cacheDir.deleteRecursively()
    }

    @Test
    fun `initial load exposes newest entries`() = runTest {
        val entries = listOf(
            CrashLogEntry(2L, "new", "newest"),
            CrashLogEntry(1L, "old", "oldest"),
        )
        every { logger.entries() } returns entries

        val vm = DiagnosticsViewModel(logger, context)

        assertFalse(vm.state.value.loading)
        assertEquals(entries, vm.state.value.entries)
    }

    @Test
    fun `clear removes logger history and refreshes state`() = runTest {
        every { logger.entries() } returnsMany listOf(
            listOf(CrashLogEntry(1L, "error", "boom")),
            emptyList(),
        )
        val vm = DiagnosticsViewModel(logger, context)
        assertTrue(vm.state.value.entries.isNotEmpty())

        vm.clearAll()

        verify(exactly = 1) { logger.clear() }
        assertTrue(vm.state.value.entries.isEmpty())
    }

    @Test
    fun `prepareExport publishes share file once`() = runTest {
        val exported = File(cacheDir, "aura-diagnostics.jsonl").apply { writeText("{}\n") }
        every { logger.entries() } returns emptyList()
        every { logger.exportTo(cacheDir, any()) } returns exported
        val vm = DiagnosticsViewModel(logger, context)

        vm.prepareExport()

        assertEquals(exported, vm.state.value.exportFile)
        vm.consumeExport()
        assertNull(vm.state.value.exportFile)
    }

    @Test
    fun `load failure is visible and dismissible`() = runTest {
        every { logger.entries() } throws IllegalStateException("disk unavailable")
        val vm = DiagnosticsViewModel(logger, context)

        assertTrue(vm.state.value.error?.contains("disk unavailable") == true)
        vm.clearError()
        assertNull(vm.state.value.error)
    }
}
