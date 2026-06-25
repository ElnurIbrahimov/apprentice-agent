package com.aura

import androidx.test.ext.junit.runAndroidJUnit4
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
        assertEquals("com.aura", appContext.packageName)
    }

    @Test
    fun appLaunchesMainActivity() {
        val intent = androidx.test.core.app.launcher.ActivityLauncher.launch(
            "com.aura.MainActivity"
        )
        // If we got here, the activity launched
        assert(intent != null)
    }
}
