package com.aura.ui.screens.onboarding

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.aura.providers.ModelCatalog
import com.aura.providers.ModelDescriptor
import com.aura.providers.ProviderModelList
import com.aura.providers.ProviderStatus
import com.aura.ui.theme.AuraTheme
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.assertEquals

class OnboardingContentTest {
    @get:Rule
    val composeRule = createComposeRule()

    private val model = ModelDescriptor("ollama:model-a", "model-a", "ollama")

    @Test
    fun providerStepShowsProgressStatusAndImeSafeActions() {
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                OnboardingContent(
                    state = OnboardingUiState(
                        step = OnboardingStep.Provider,
                        keyDrafts = mapOf("ollama" to "test-key"),
                        credentialStatus = mapOf("ollama" to OnboardingCredentialStatus.Verified),
                    ),
                    onBack = {}, onSkip = {}, onNext = {},
                    onKeyDraftChanged = { _, _ -> }, onSaveAndTest = {},
                    onModelSelected = {}, onFinish = {},
                )
            }
        }

        composeRule.onNodeWithTag("onboarding-progress").assertIsDisplayed()
        composeRule.onNodeWithText("Ollama Cloud").assertIsDisplayed()
        composeRule.onNodeWithText("Verified").assertIsDisplayed()
        composeRule.onNodeWithTag("onboarding-bottom-actions").assertIsDisplayed()
    }

    @Test
    fun modelStepSelectsVisibleCatalogModel() {
        var selected: String? = null
        composeRule.setContent {
            AuraTheme(themeMode = "dark") {
                ModelSelectionStep(
                    catalog = ModelCatalog(
                        mapOf("ollama" to ProviderModelList("ollama", ProviderStatus.Ready, listOf(model))),
                        listOf(model),
                    ),
                    selectedModel = null,
                    error = null,
                    onModelSelected = { selected = it },
                )
            }
        }

        composeRule.onNodeWithTag("onboarding-model-ollama:model-a").performClick()
        assertEquals("ollama:model-a", selected)
    }

    @Test
    fun completionShowsSelectedDefaultBeforeFinish() {
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                OnboardingContent(
                    state = OnboardingUiState(
                        step = OnboardingStep.Complete,
                        selectedDefaultModel = "ollama:model-a",
                    ),
                    onBack = {}, onSkip = {}, onNext = {},
                    onKeyDraftChanged = { _, _ -> }, onSaveAndTest = {},
                    onModelSelected = {}, onFinish = {},
                )
            }
        }

        composeRule.onNodeWithTag("onboarding-selected-model").assertIsDisplayed()
        composeRule.onNodeWithText("Start chatting").assertIsDisplayed()
    }
}
