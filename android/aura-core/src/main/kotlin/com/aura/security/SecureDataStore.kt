package com.aura.security

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Provider
import javax.inject.Singleton

/**
 * Thrown when decryption fails (e.g. the underlying key in the Android
 * Keystore was lost or the stored ciphertext is corrupted).
 */
class DecryptionFailedException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause)

/**
 * A thin wrapper around [DataStore] that encrypts values at rest using
 * AES-256-GCM via the Android Keystore.
 *
 * The underlying [DataStore] is accessed through a [Provider] so that tests
 * can supply an in-memory store without needing the Android framework.
 */
@Singleton
class SecureDataStore @Inject constructor(
    private val dataStoreProvider: Provider<DataStore<Preferences>>,
    private val keyManager: KeyManager,
) {
    private val dataStore: DataStore<Preferences>
        get() = dataStoreProvider.get()

    /** The encryption key, retrieved or created lazily. */
    private val key by lazy { keyManager.getOrCreateKey() }

    /**
     * Encrypts [value] and writes it under the given [key] in the DataStore.
     */
    suspend fun putString(key: String, value: String) {
        val encrypted = keyManager.encrypt(value, this.key)
        dataStore.edit { prefs ->
            prefs[stringPreferencesKey(key)] = encrypted
        }
    }

    /**
     * Reads and decrypts the value stored under the given [key].
     *
     * Returns `null` if the key does not exist in the DataStore.
     *
     * @throws DecryptionFailedException if the stored value exists but
     *         cannot be decrypted (e.g. the Android Keystore key was
     *         invalidated).
     */
    suspend fun getString(key: String): String? {
        val encrypted = dataStore.data
            .map { prefs -> prefs[stringPreferencesKey(key)] }
            .first() ?: return null
        return keyManager.decrypt(encrypted, this.key)
            ?: throw DecryptionFailedException(
                "Failed to decrypt stored value for key '$key' — " +
                    "the Android Keystore key may have been invalidated."
            )
    }

    /**
     * Removes the entry stored under the given [key] from the DataStore.
     */
    suspend fun removeString(key: String) {
        dataStore.edit { prefs ->
            prefs.remove(stringPreferencesKey(key))
        }
    }
}
