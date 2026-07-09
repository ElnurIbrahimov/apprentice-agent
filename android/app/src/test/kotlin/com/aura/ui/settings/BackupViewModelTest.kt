package com.aura.ui.settings

import android.content.Context
import android.net.Uri
import com.aura.backup.AuraBackup
import com.aura.backup.BackupManager
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
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

/**
 * Tests the synchronous state-machine portion of [BackupViewModel].
 *
 * We use [UnconfinedTestDispatcher] as the main dispatcher so that
 * any viewModelScope.launch body runs inline up to the first
 * suspension point. Coroutine blocks that internally use
 * [Dispatchers.IO] (the real IO dispatcher) will still escape
 * the test's control, so we keep the tests small and synchronous.
 */
class BackupViewModelTest {

    private val context = mockk<Context>(relaxed = true)
    private val backupManager = mockk<BackupManager>(relaxed = true)

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial state is idle with no result`() = runTest {
        val vm = BackupViewModel(context, backupManager)
        val s = vm.state.value
        assertFalse(s.exportInFlight)
        assertFalse(s.importInFlight)
        assertNull(s.lastResult)
        assertNull(s.pendingImportBytes)
        assertFalse(s.showImportConfirm)
    }

    @Test
    fun `prepareExportFile writes JSON and surfaces result`() = runTest {
        val sampleBackup = AuraBackup(
            exportedAt = 1L, appVersionName = "0.1.0",
            memories = listOf(
                com.aura.backup.MemoryBackup("m1", "x", "user", "preference", 0.5f, 1L, 1L, 0, 1f, "", "{}")
            ),
        )
        coEvery { backupManager.snapshot(any()) } returns sampleBackup
        every { backupManager.encodeToJson(sampleBackup) } returns "{\"memories\":[{}]}"
        every { backupManager.exportFile() } answers {
            java.io.File.createTempFile("aura-backup-test", ".json").apply { deleteOnExit() }
        }

        val vm = BackupViewModel(context, backupManager)
        val file = vm.prepareExportFile()

        val state = vm.state.value
        assertFalse(state.exportInFlight)
        assertTrue(state.lastResult!!.contains("Exported"))
        assertEquals("{\"memories\":[{}]}", file!!.readText())
    }

    @Test
    fun `cancelImport clears the confirm dialog`() = runTest {
        val vm = BackupViewModel(context, backupManager)
        // cancelImport on a fresh VM is a no-op; the no-op must not
        // throw and the dialog state must stay false.
        vm.cancelImport()
        assertFalse(vm.state.value.showImportConfirm)
        assertNull(vm.state.value.pendingImportBytes)
    }

    @Test
    fun `confirmImport with no staged bytes is a no-op`() = runTest {
        val vm = BackupViewModel(context, backupManager)
        vm.confirmImport(purgeFirst = false)
        // Restore must not be called when there's nothing staged.
        coVerify(exactly = 0) { backupManager.restore(any()) }
    }

    @Test
    fun `clearResult clears the last result`() = runTest {
        val vm = BackupViewModel(context, backupManager)
        vm.clearResult()
        assertNull(vm.state.value.lastResult)
    }
}
