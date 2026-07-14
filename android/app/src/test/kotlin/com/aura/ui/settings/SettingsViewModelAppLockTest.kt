package com.aura.ui.settings

import com.aura.agent.IdentityStore
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderRegistry
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ModelDescriptor
import com.aura.providers.ProviderModelList
import com.aura.providers.ProviderStatus
import com.aura.providers.ProviderCredentialState
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
    private val identityStore = mockk<IdentityStore>(relaxed = true)
    private val catalogFlow = MutableStateFlow(ModelCatalog(emptyMap(), emptyList()))
    private val credentialFlow = MutableStateFlow(
        ProviderKeys.PREFIXES.associateWith { ProviderCredentialState.NotConfigured },
    )
    private val modelCatalogRepository = mockk<ModelCatalogRepository>(relaxed = true)

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
        every { providerKeys.credentialStates } returns credentialFlow
        every { providerKeys.embeddingModel } returns ""
        every { modelCatalogRepository.catalog } returns catalogFlow
        every { userPreferences.defaultModel } returns flowOf("test:chat-model")
        every { userPreferences.visionModel } returns flowOf(null)
        every { userPreferences.backgroundModel } returns flowOf(null)
        every { userPreferences.deepModeModel } returns flowOf(null)
        every { userPreferences.moaReferenceModels } returns flowOf(emptyList())
        every { userPreferences.moaAggregatorModel } returns flowOf(null)
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

    private fun newViewModel() = SettingsViewModel(
        providerRegistry,
        providerKeys,
        userPreferences,
        identityStore,
        modelCatalogRepository,
    )

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `refreshModels consumes shared catalog state`() = runTest {
        every { modelCatalogRepository.refresh(any()) } answers {
            val models = listOf(
                ModelDescriptor("test:model-a", "model-a", "test"),
                ModelDescriptor("test:model-b", "model-b", "test"),
            )
            catalogFlow.value = ModelCatalog(
                providers = mapOf(
                    "test" to ProviderModelList(
                        providerPrefix = "test",
                        status = ProviderStatus.Ready,
                        models = models,
                    ),
                ),
                allModels = models,
            )
        }

        val vm = newViewModel()
        vm.refreshModels()

        assertEquals(listOf("test:model-a", "test:model-b"), vm.state.value.availableModels)
        assertEquals(null, vm.state.value.modelsError)
    }

    @Test
    fun `typing provider draft does not persist credential`() = runTest {
        val vm = newViewModel()

        vm.updateCredentialDraft("ollama", "draft-key")

        assertEquals("draft-key", vm.state.value.keyDrafts["ollama"])
        coVerify(exactly = 0) { providerKeys.set(any(), any()) }
    }

    @Test
    fun `save and test persists then verifies from catalog`() = runTest {
        coEvery { providerKeys.set("ollama", "draft-key") } answers {
            credentialFlow.value = credentialFlow.value +
                ("ollama" to ProviderCredentialState.Saved)
        }
        coEvery { providerKeys.markValidation("ollama", true) } answers {
            credentialFlow.value = credentialFlow.value +
                ("ollama" to ProviderCredentialState.Valid)
        }
        coEvery { modelCatalogRepository.refreshProvider("ollama", any()) } answers {
            val models = listOf(ModelDescriptor("ollama:test-model", "test-model", "ollama"))
            catalogFlow.value = ModelCatalog(
                providers = mapOf(
                    "ollama" to ProviderModelList(
                        "ollama",
                        ProviderStatus.Ready,
                        models,
                    ),
                ),
                allModels = models,
            )
        }
        val vm = newViewModel()
        vm.updateCredentialDraft("ollama", "draft-key")

        vm.saveAndTestProvider("ollama")

        assertEquals(ProviderTestPhase.Verified, vm.state.value.providerTests["ollama"]?.phase)
        assertEquals(ProviderCredentialState.Valid, vm.state.value.credentialStates["ollama"])
        assertEquals(listOf("ollama:test-model"), vm.state.value.availableModels)
        coVerify(exactly = 1) { providerKeys.set("ollama", "draft-key") }
    }

    @Test
    fun `typed catalog failure leaves provider invalid`() = runTest {
        coEvery { providerKeys.set("ollama", "bad-key") } answers {
            credentialFlow.value = credentialFlow.value +
                ("ollama" to ProviderCredentialState.Saved)
        }
        coEvery { providerKeys.markValidation("ollama", false) } answers {
            credentialFlow.value = credentialFlow.value +
                ("ollama" to ProviderCredentialState.Invalid)
        }
        coEvery { modelCatalogRepository.refreshProvider("ollama", any()) } answers {
            catalogFlow.value = ModelCatalog(
                providers = mapOf(
                    "ollama" to ProviderModelList(
                        providerPrefix = "ollama",
                        status = ProviderStatus.Unauthorized,
                        errorMessage = "Authentication failed",
                    ),
                ),
                allModels = emptyList(),
            )
        }
        val vm = newViewModel()
        vm.updateCredentialDraft("ollama", "bad-key")

        vm.saveAndTestProvider("ollama")

        assertEquals(ProviderTestPhase.Failed, vm.state.value.providerTests["ollama"]?.phase)
        assertEquals(ProviderCredentialState.Invalid, vm.state.value.credentialStates["ollama"])
    }

    @Test
    fun `settings credential specs cover every provider and tool prefix`() {
        assertEquals(
            ProviderKeys.PREFIXES.toSet(),
            SETTINGS_CREDENTIAL_SPECS.map { it.prefix }.toSet(),
        )
        assertEquals(ProviderKeys.PREFIXES.size, SETTINGS_CREDENTIAL_SPECS.size)
    }

    @Test
    fun `tool-only credential saves without pretending to verify a model catalog`() = runTest {
        coEvery { providerKeys.set("brave", "tool-key") } answers {
            credentialFlow.value = credentialFlow.value +
                ("brave" to ProviderCredentialState.Saved)
        }
        val vm = newViewModel()
        vm.updateCredentialDraft("brave", "tool-key")

        vm.saveAndTestProvider("brave")

        assertEquals("Saved securely", vm.state.value.verifyResults["brave"])
        coVerify(exactly = 1) { providerKeys.set("brave", "tool-key") }
        coVerify(exactly = 0) { modelCatalogRepository.refreshProvider("brave", any()) }
        coVerify(exactly = 0) { providerKeys.markValidation("brave", any()) }
    }

    @Test
    fun `appLockEnabled starts false in the default state`() = runTest {
        val vm = newViewModel()
        assertFalse(vm.state.value.appLockEnabled)
    }

    @Test
    fun `appLockEnabled reflects the persisted value after reload`() = runTest {
        appLockFlow.value = true
        val vm = newViewModel()
        vm.reload()
        assertTrue(vm.state.value.appLockEnabled)
    }

    @Test
    fun `setAppLockEnabled true persists and updates the state`() = runTest {
        val vm = newViewModel()
        vm.setAppLockEnabled(true)
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(true) }
        assertTrue(vm.state.value.appLockEnabled)
    }

    @Test
    fun `setAppLockEnabled false persists and updates the state`() = runTest {
        appLockFlow.value = true
        val vm = newViewModel()
        // Sanity: we did read the initial value as true.
        assertTrue(vm.state.value.appLockEnabled)

        vm.setAppLockEnabled(false)
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(false) }
        assertFalse(vm.state.value.appLockEnabled)
        assertEquals(false, appLockFlow.value)
    }

    @Test
    fun `toggling twice lands on the last value`() = runTest {
        val vm = newViewModel()
        vm.setAppLockEnabled(true)
        vm.setAppLockEnabled(false)
        assertFalse(vm.state.value.appLockEnabled)
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(true) }
        coVerify(exactly = 1) { userPreferences.setAppLockEnabled(false) }
    }

    @Test
    fun `morning brief defaults to enabled`() = runTest {
        val vm = newViewModel()
        assertTrue(vm.state.value.morningBriefEnabled, "fresh install should default to morning brief on")
    }

    @Test
    fun `setMorningBriefEnabled false persists and updates state`() = runTest {
        val vm = newViewModel()
        vm.setMorningBriefEnabled(false)
        coVerify(exactly = 1) { userPreferences.setMorningBriefEnabled(false) }
        assertFalse(vm.state.value.morningBriefEnabled)
        assertEquals(false, morningBriefFlow.value)
    }

    @Test
    fun `setMorningBriefEnabled re-enabling flips state back`() = runTest {
        morningBriefFlow.value = false
        val vm = newViewModel()
        assertFalse(vm.state.value.morningBriefEnabled, "should read the persisted false on init")

        vm.setMorningBriefEnabled(true)
        assertTrue(vm.state.value.morningBriefEnabled)
        coVerify(exactly = 1) { userPreferences.setMorningBriefEnabled(true) }
    }

    @Test
    fun `calendar monitor defaults to enabled`() = runTest {
        val vm = newViewModel()
        assertTrue(vm.state.value.calendarMonitorEnabled, "fresh install should default to calendar monitor on")
    }

    @Test
    fun `setCalendarMonitorEnabled false persists and updates state`() = runTest {
        val vm = newViewModel()
        vm.setCalendarMonitorEnabled(false)
        coVerify(exactly = 1) { userPreferences.setCalendarMonitorEnabled(false) }
        assertFalse(vm.state.value.calendarMonitorEnabled)
        assertEquals(false, calendarMonitorFlow.value)
    }
}
