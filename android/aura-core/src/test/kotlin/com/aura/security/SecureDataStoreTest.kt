package com.aura.security

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import kotlinx.coroutines.test.runTest
import org.junit.Test
import java.io.File
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * Unit tests for [SecureDataStore] that run on the JVM using an in-memory
 * DataStore and an in-memory AES key.
 *
 * DataStore cleanup note: PreferenceDataStore does not implement
 * [java.io.Closeable]. The internal actor is cancelled when the test
 * scope from [runTest] completes, which is sufficient — no manual
 * close is needed.
 */
class SecureDataStoreTest {

    @Test
    fun `roundtrip via fake datastore`() = runTest {
        // DataStore requires the file to end in .preferences_pb. The tempdir
        // file is cleaned up via deleteOnExit().
        val file = File(
            System.getProperty("java.io.tmpdir"),
            "sds_${UUID.randomUUID()}.preferences_pb",
        )
        file.deleteOnExit()
        val dataStore = PreferenceDataStoreFactory.create(produceFile = { file })
        val store = SecureDataStore(
            dataStoreProvider = javax.inject.Provider<DataStore<androidx.datastore.preferences.core.Preferences>> { dataStore },
            keyManager = KeyManager(null),
        )

        assertNull(store.getString("test_key"), "initially null")
        store.putString("test_key", "my-secret-value")
        assertEquals("my-secret-value", store.getString("test_key"))

        store.putString("test_key", "updated-value")
        assertEquals("updated-value", store.getString("test_key"))

        store.removeString("test_key")
        assertNull(store.getString("test_key"), "null after removal")
    }
}
