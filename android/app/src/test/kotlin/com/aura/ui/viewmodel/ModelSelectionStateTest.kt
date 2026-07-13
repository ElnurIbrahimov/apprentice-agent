package com.aura.ui.viewmodel

import com.aura.providers.ModelCatalog
import com.aura.providers.ModelDescriptor
import com.aura.providers.ProviderModelList
import com.aura.providers.ProviderStatus
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ModelSelectionStateTest {

    @Test
    fun `blank selection is missing even when models exist`() {
        assertIs<ModelSelectionState.Missing>(resolveModelSelection("", readyCatalog()))
    }

    @Test
    fun `loading provider produces loading selection`() {
        val catalog = catalog(ProviderStatus.Loading, emptyList())
        assertIs<ModelSelectionState.Loading>(resolveModelSelection("test:model-a", catalog))
    }

    @Test
    fun `selected catalog model is ready`() {
        val state = resolveModelSelection("test:model-a", readyCatalog())
        assertIs<ModelSelectionState.Ready>(state)
        assertEquals("test:model-a", state.activeModel)
    }

    @Test
    fun `cached models remain ready and report stale provider`() {
        val models = listOf(model("model-a"))
        val catalog = ModelCatalog(
            providers = mapOf(
                "test" to ProviderModelList(
                    "test",
                    ProviderStatus.Ready,
                    models,
                    errorMessage = "Offline; using cache",
                ),
            ),
            allModels = models,
        )
        val state = assertIs<ModelSelectionState.Ready>(
            resolveModelSelection("test:model-a", catalog),
        )
        assertTrue("test" in state.staleProviders)
    }

    @Test
    fun `removed selected model fails with recovery`() {
        val state = assertIs<ModelSelectionState.Failed>(
            resolveModelSelection("test:removed", readyCatalog()),
        )
        assertTrue(state.message.contains("no longer available"))
    }

    @Test
    fun `typed provider failure is surfaced`() {
        val catalog = ModelCatalog(
            providers = mapOf(
                "test" to ProviderModelList(
                    providerPrefix = "test",
                    status = ProviderStatus.Unauthorized,
                    errorMessage = "Authentication failed",
                ),
            ),
            allModels = emptyList(),
        )
        val state = assertIs<ModelSelectionState.Failed>(
            resolveModelSelection("test:model-a", catalog),
        )
        assertEquals("Authentication failed", state.message)
    }

    private fun readyCatalog(): ModelCatalog {
        val models = listOf(model("model-a"), model("model-b"))
        return ModelCatalog(
            providers = mapOf("test" to ProviderModelList("test", ProviderStatus.Ready, models)),
            allModels = models,
        )
    }

    private fun catalog(status: ProviderStatus, models: List<ModelDescriptor>): ModelCatalog =
        ModelCatalog(
            providers = mapOf("test" to ProviderModelList("test", status, models)),
            allModels = models,
        )

    private fun model(name: String) = ModelDescriptor("test:$name", name, "test")
}
