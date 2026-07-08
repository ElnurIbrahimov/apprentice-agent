package com.aura.ui.settings

import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests the app-lock toggle in [SettingsViewModel]. The toggle
 * was previously a stored-but-no-UI gap: the value lived in
 * [UserPreferences.appLockEnabled] and round-tripped through
 * backup, but there was no Settings UI to ever set it.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class SettingsViewModelAppLockTest {

    private val providerRegistry = mockk<ProviderRegistry>(relaxed = true)
    private val providerKeys = mockk<ProviderKeys>(relaxed = true)
    private val userPreferences = mockk<UserPreferences>(relaxed = true)

    private val appLockFlow = MutableStateFlow(false)
    private val morningBriefFlow = MutableStateFlow(true)
    private val calendarMonitorFlow = MutableStateFlow(true)

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        every { providerRegistry.configured() } returns emptyList()
        every { providerRegistry.all() } returns emptyList()
        every { providerKeys.keyFor(any()) } returns null
        // The SettingsViewModel now waits for the ProviderKeys initial
        // load to finish before reading configured providers. In
        // production this flips to true in ProviderKeys.init once
        // DataStore resolves; in tests we just pretend it's already
        // done so reload() doesn't block.
        every { providerKeys.loaded } returns MutableStateFlow(true)
        every { userPreferences.defaultModel } returns flowOf("ollama:deepseek-v4-pro:cloud")
        every { userPreferences.firstRunComplete } returns flowOf(true)
        every { userPreferences.appLockEnabled } returns appLockFlow
        every { userPreferences.morningBriefEnabled } returns morningBriefFlow
        every { userPreferences.calendarMonitorEnabled } returns calendarMonitorFlow
        every { userPreferences.themeMode } returns flowOf("system")
        every { userPreferences.customIdentity } returns flowOf("")
        every { userPreferences.specialistOverrides } returns flowOf("{}")
        every { userPreferences.ttsEnabled } returns flowOf(true)
        every { userPreferences.incognitoDefault } returns flowOf(false)
        every { userPreferences.lastSeenProactiveAt } returns flowOf(0L)
        every { userPreferences.morningBriefHour } returns flowOf(7)
        coEvery { userPreferences.setAppLockEnabled(any()) } answers {
            appLockFlow.value = firstArg()
        }
        coEvery { userPreferences.setMorningBriefEnabled(any()) } answers {
            morningBriefFlow.value = firstArg()
        }
        coEvery { userPreferences.setCalendarMonitorEnabled(any()) } answers {
            calendarMonitorFlow.value = firstArg()
        }
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `appLockEnabled starts false in the default state`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        assertFalse(vm.state.value.appLockEnabled)
    }

    @Test
    fun `appLockEnabled reflects the persisted value after reload`() = runTest {
        appLockFlow.value = true
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        vm.reload()
        assertTrue(vm.state.value.appLockEnabled)
    }

    @Test
    fun `setAppLockEnabled true persists and updates the state`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        vm.setAppLockEnabled(true)
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(true) }
        assertTrue(vm.state.value.appLockEnabled)
    }

    @Test
    fun `setAppLockEnabled false persists and updates the state`() = runTest {
        appLockFlow.value = true
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        // Sanity: we did read the initial value as true.
        assertTrue(vm.state.value.appLockEnabled)

        vm.setAppLockEnabled(false)
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(false) }
        assertFalse(vm.state.value.appLockEnabled)
        assertEquals(false, appLockFlow.value)
    }

    @Test
    fun `toggling twice lands on the last value`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        vm.setAppLockEnabled(true)
        vm.setAppLockEnabled(false)
        assertFalse(vm.state.value.appLockEnabled)
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(true) }
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(false) }
    }

    @Test
    fun `morning brief defaults to enabled`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        assertTrue(vm.state.value.morningBriefEnabled, "fresh install should default to morning brief on")
    }

    @Test
    fun `setMorningBriefEnabled false persists and updates state`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        vm.setMorningBriefEnabled(false)
        coVerify(exactly = 1) { userPreferences.setMorningBriefEnabled(false) }
        assertFalse(vm.state.value.morningBriefEnabled)
        assertEquals(false, morningBriefFlow.value)
    }

    @Test
    fun `setMorningBriefEnabled re-enabling flips state back`() = runTest {
        morningBriefFlow.value = false
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        assertFalse(vm.state.value.morningBriefEnabled, "should read the persisted false on init")

        vm.setMorningBriefEnabled(true)
        assertTrue(vm.state.value.morningBriefEnabled)
        coVerify(exactly = 1) { userPreferences.setMorningBriefEnabled(true) }
    }

    @Test
    fun `calendar monitor defaults to enabled`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        assertTrue(vm.state.value.calendarMonitorEnabled, "fresh install should default to calendar monitor on")
    }

    @Test
    fun `setCalendarMonitorEnabled false persists and updates state`() = runTest {
        val vm = SettingsViewModel(providerRegistry, providerKeys, userPreferences)
        vm.setCalendarMonitorEnabled(false)
        coVerify(exactly = 1) { userPreferences.setCalendarMonitorEnabled(false) }
        assertFalse(vm.state.value.calendarMonitorEnabled)
        assertEquals(false, calendarMonitorFlow.value)
    }
}
