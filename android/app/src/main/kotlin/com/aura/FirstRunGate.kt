package com.aura

import javax.inject.Inject
import javax.inject.Singleton

/**
 * Gate that blocks the main UI until the user has completed onboarding.
 *
 * The actual storage is in [com.aura.data.UserPreferences.firstRunComplete]
 * — historically this class had its own `Context.auraPrefs` DataStore
 * extension, but that collided with two other declarations of the
 * same extension (and worse, used a different value type for the
 * same logical key — string vs boolean), so writes from
 * SettingsViewModel never reached this reader. The flag is now
 * centralized on [com.aura.data.UserPreferences].
 */
@Singleton
class FirstRunGate @Inject constructor(
    private val userPreferences: com.aura.data.UserPreferences,
) {
    suspend fun isFirstRunComplete(): Boolean = userPreferences.isFirstRunComplete()

    suspend fun markComplete() {
        userPreferences.setFirstRunComplete(true)
    }
}
