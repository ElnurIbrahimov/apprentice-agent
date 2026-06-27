package com.aura.security

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStoreFile
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import java.security.KeyStore
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object SecurityModule {

    @Provides
    @Singleton
    fun provideDataStore(
        @ApplicationContext context: Context,
    ): DataStore<Preferences> = PreferenceDataStoreFactory.create(
        produceFile = { context.preferencesDataStoreFile("aura_secure_settings") }
    )

    @Provides
    @Singleton
    fun provideKeyManager(): KeyManager {
        return try {
            val ks = KeyStore.getInstance("AndroidKeyStore")
            ks.load(null)
            KeyManager(ks)
        } catch (_: Exception) {
            // JVM / Robolectric test fallback: in-memory AES key
            KeyManager(null)
        }
    }
}
