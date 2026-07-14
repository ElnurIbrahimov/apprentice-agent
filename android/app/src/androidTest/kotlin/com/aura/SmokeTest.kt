package com.aura

import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.Configuration
import androidx.work.WorkManager
import com.aura.data.UserPreferences
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import javax.inject.Inject
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The simplest possible smoke test: the app context exists and the
 * package name matches. Run with:
 *   ./gradlew :app:connectedAndroidTest
 */
@HiltAndroidTest
@RunWith(AndroidJUnit4::class)
class SmokeTest {

    @get:Rule
    val hiltRule = HiltAndroidRule(this)

    @Inject lateinit var preferences: UserPreferences

    @Before
    fun setUp() {
        hiltRule.inject()
        runCatching {
            WorkManager.initialize(
                ApplicationProvider.getApplicationContext(),
                Configuration.Builder().build(),
            )
        }
        runBlocking {
            preferences.setAppLockEnabled(false)
            preferences.setFirstRunComplete(true)
        }
    }

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
