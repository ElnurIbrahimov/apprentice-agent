package com.aura.ui.nav

import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.ui.test.assertHeightIsAtLeast
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import com.aura.ui.theme.AuraDimensions
import com.aura.ui.theme.AuraTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import androidx.compose.ui.unit.dp

class BottomNavigationTest {

    @get:Rule
    val composeRule = createComposeRule()

    @Test
    fun queryRouteSelectsChatAndClickNavigatesHome() {
        var selected: String? = null
        composeRule.setContent {
            AuraTheme(themeMode = "light") {
                AuraBottomNavigation(
                    currentRoute = "chat?draft=hello",
                    onRouteSelected = { selected = it.route },
                    navigationBarInsets = WindowInsets(bottom = 24.dp),
                )
            }
        }

        composeRule.onNodeWithTag("bottom-navigation-row")
            .assertHeightIsAtLeast(AuraDimensions.bottomNavigationHeight)
        val rowBounds = composeRule.onNodeWithTag("bottom-navigation-row")
            .fetchSemanticsNode().boundsInRoot
        val barBounds = composeRule.onNodeWithTag("bottom-navigation")
            .fetchSemanticsNode().boundsInRoot
        assertTrue(
            "navigation inset must live below the interactive row: row=$rowBounds bar=$barBounds",
            rowBounds.bottom < barBounds.bottom,
        )
        composeRule.onNodeWithContentDescription("Chat").assertIsSelected()
        composeRule.onNodeWithContentDescription("Home").assertIsNotSelected().performClick()
        assertEquals("home", selected)
    }
}
