package com.aura

import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.core.app.ApplicationProvider
import androidx.work.Configuration
import androidx.work.WorkManager
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import com.aura.data.UserPreferences
import com.aura.providers.ProviderKeys
import com.aura.testing.FakeProviderController
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import javax.inject.Inject
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.first
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.After
import org.junit.runner.RunWith

@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class ModelSelectionFlowTest {

    @get:Rule(order = 0)
    val hiltRule = HiltAndroidRule(this)

    @get:Rule(order = 1)
    val composeRule = createEmptyComposeRule()

    private lateinit var scenario: ActivityScenario<MainActivity>

    @Inject lateinit var preferences: UserPreferences
    @Inject lateinit var providerKeys: ProviderKeys

    @Before
    fun resetState() {
        hiltRule.inject()
        runCatching {
            WorkManager.initialize(
                ApplicationProvider.getApplicationContext(),
                Configuration.Builder().build(),
            )
        }
        FakeProviderController.reset()
        runBlocking {
            providerKeys.awaitLoaded()
            providerKeys.set("ollama", "")
            preferences.setDefaultModel(null)
            preferences.setFirstRunComplete(true)
        }
        scenario = ActivityScenario.launch(MainActivity::class.java)
        composeRule.waitForIdle()
    }

    @After
    fun closeActivity() = scenario.close()

    @Test
    fun credentialToChatSelectionSurvivesRecreationAndChatOverrideStaysLocal() {
        composeRule.onNodeWithContentDescription("Settings").performClick()
        composeRule.onNodeWithTag("provider-key-ollama-cloud")
            .performScrollTo()
            .performTextReplacement("test-key-valid")
        composeRule.onNodeWithTag("provider-test-ollama-cloud").performClick()
        composeRule.waitUntil(10_000) {
            composeRule.onAllNodesWithText("Verified").fetchSemanticsNodes().isNotEmpty()
        }

        composeRule.onNodeWithText("Chat default").performScrollTo()
        composeRule.onAllNodesWithText("Choose")[0].performClick()
        composeRule.onNodeWithTag("model-row-ollama:model-a").performClick()
        composeRule.waitForIdle()

        scenario.recreate()
        composeRule.waitForIdle()
        composeRule.onNodeWithContentDescription("Chat").performClick()
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("chat-model-pill").assertTextContains("Model A", substring = true)
        composeRule.onNodeWithText("Message AURA…").performTextReplacement("hello")
        composeRule.onNodeWithContentDescription("Send").assertIsEnabled()

        composeRule.onNodeWithTag("chat-model-pill").performClick()
        composeRule.onNodeWithTag("model-row-ollama:model-b").performClick()
        composeRule.onNodeWithTag("chat-model-pill").assertTextContains("Model B", substring = true)

        assertEquals("ollama:model-a", runBlocking { preferences.defaultModel.first() })
    }

}
