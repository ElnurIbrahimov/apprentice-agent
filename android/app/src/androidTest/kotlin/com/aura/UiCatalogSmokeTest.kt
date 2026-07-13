package com.aura

import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.aura.debug.CatalogSurfaces
import com.aura.debug.CatalogStates
import com.aura.debug.CatalogThemes
import com.aura.debug.UiCatalogActivity
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.assertTrue

@RunWith(AndroidJUnit4::class)
class UiCatalogSmokeTest {

    @Test
    fun catalogExposesEveryPlannedSurface() {
        val required = setOf(
            "startup", "onboarding", "home", "chat", "model-picker", "settings",
            "memory", "history", "tasks", "reminders", "hands", "tools",
            "proactive", "graph", "profile", "identity", "diagnostics", "voice",
            "quick-ask", "widget-config",
        )
        assertTrue(CatalogSurfaces.containsAll(required))
        assertTrue(
            CatalogStates.containsAll(
                setOf("content", "loading", "empty", "error", "no-provider", "selected", "partial-error"),
            ),
        )
        assertTrue(CatalogThemes == setOf("light", "dark"))
    }

    @Test
    fun catalogActivityLaunchesWithDeterministicExtras() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        CatalogThemes.forEach { theme ->
            val intent = Intent(context, UiCatalogActivity::class.java).apply {
                putExtra(UiCatalogActivity.EXTRA_SURFACE, "home")
                putExtra(UiCatalogActivity.EXTRA_STATE, "empty")
                putExtra(UiCatalogActivity.EXTRA_THEME, theme)
            }
            ActivityScenario.launch<UiCatalogActivity>(intent).use { scenario ->
                scenario.onActivity { activity -> assertTrue(!activity.isFinishing) }
            }
        }
    }
}
