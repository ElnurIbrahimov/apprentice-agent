package com.aura

import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.Assert.assertEquals

/**
 * The simplest possible smoke test: the app context exists and the
 * package name matches. Run with:
 *   ./gradlew :app:connectedAndroidTest
 */
class SmokeTest {

    @Test
    fun useAppContext() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals(BuildConfig.APPLICATION_ID, appContext.packageName)
    }

    @Test
    fun appLaunchesMainActivity() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                assertEquals(MainActivity::class.java, activity::class.java)
            }
        }
    }
}
