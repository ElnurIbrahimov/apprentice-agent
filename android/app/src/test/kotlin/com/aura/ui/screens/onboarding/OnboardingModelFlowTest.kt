package com.aura.ui.screens.onboarding

import com.aura.FirstRunGate
import com.aura.data.UserPreferences
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelCatalogRepository
import com.aura.providers.ModelDescriptor
import com.aura.providers.ProviderKeys
import com.aura.providers.ProviderModelList
import com.aura.providers.ProviderStatus
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class OnboardingModelFlowTest {
    private val dispatcher = StandardTestDispatcher()
    private lateinit var gate: FirstRunGate
    private lateinit var keys: ProviderKeys
    private lateinit var repository: ModelCatalogRepository
    private lateinit var preferences: UserPreferences
    private lateinit var catalog: MutableStateFlow<ModelCatalog>

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
        gate = mockk(relaxed = true)
        keys = mockk(relaxed = true)
        repository = mockk(relaxed = true)
        preferences = mockk(relaxed = true)
        catalog = MutableStateFlow(ModelCatalog(emptyMap(), emptyList()))
        every { repository.catalog } returns catalog
        every { keys.loaded } returns MutableStateFlow(true)
    }

    @After
    fun tearDown() = Dispatchers.resetMain()

    private fun viewModel() = OnboardingViewModel(gate, keys, repository, preferences)

    @Test
    fun `save and test persists key then exposes verified catalog models`() = runTest(dispatcher) {
        val descriptor = ModelDescriptor("ollama:model-a", "model-a", "ollama")
        coEvery { repository.refreshProvider("ollama", true) } coAnswers {
            catalog.value = ModelCatalog(
                providers = mapOf(
                    "ollama" to ProviderModelList(
                        providerPrefix = "ollama",
                        status = ProviderStatus.Ready,
                        models = listOf(descriptor),
                    ),
                ),
                allModels = listOf(descriptor),
            )
        }
        val vm = viewModel()
        vm.updateKeyDraft("ollama", "test-key")
        vm.saveAndTest("ollama")
        advanceUntilIdle()

        coVerify(exactly = 1) { keys.set("ollama", "test-key") }
        coVerify(exactly = 1) { repository.refreshProvider("ollama", true) }
        assertEquals(OnboardingCredentialStatus.Verified, vm.state.value.credentialStatus["ollama"])
        assertEquals(listOf("ollama:model-a"), vm.state.value.catalog.allModels.map { it.id })
    }

    @Test
    fun `default model selection persists only catalog model`() = runTest(dispatcher) {
        val descriptor = ModelDescriptor("ollama:model-a", "model-a", "ollama")
        catalog.value = ModelCatalog(
            mapOf("ollama" to ProviderModelList("ollama", ProviderStatus.Ready, listOf(descriptor))),
            listOf(descriptor),
        )
        val vm = viewModel()
        advanceUntilIdle()

        vm.selectDefaultModel("ollama:model-a")
        advanceUntilIdle()
        coVerify(exactly = 1) { preferences.setDefaultModel("ollama:model-a") }
        assertEquals("ollama:model-a", vm.state.value.selectedDefaultModel)

        vm.selectDefaultModel("made-up:model")
        advanceUntilIdle()
        coVerify(exactly = 0) { preferences.setDefaultModel("made-up:model") }
        assertTrue(vm.state.value.error?.contains("available", ignoreCase = true) == true)
    }

    @Test
    fun `skip completes without fabricating a model`() = runTest(dispatcher) {
        var completed = false
        val vm = viewModel()
        vm.skip { completed = true }
        advanceUntilIdle()

        coVerify(exactly = 1) { gate.markComplete() }
        coVerify(exactly = 0) { preferences.setDefaultModel(any()) }
        assertTrue(completed)
        assertEquals(null, vm.state.value.selectedDefaultModel)
    }
}
