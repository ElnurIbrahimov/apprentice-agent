package com.aura.security

import org.junit.Test
import java.util.Base64
import javax.crypto.KeyGenerator
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Unit tests for [KeyManager] that run on the JVM without any Android
 * dependencies. An in-memory AES key is used instead of the Android
 * Keystore.
 */
class KeyManagerTest {

    private val keyManager = KeyManager(null) // in-memory key
    private val key = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()

    @Test
    fun `roundtrip encrypts and decrypts a string`() {
        val original = "sk-ant-my-secret-api-key-abc123"
        val encrypted = keyManager.encrypt(original, key)
        val decrypted = keyManager.decrypt(encrypted, key)

        assertNotNull(decrypted)
        assertEquals(original, decrypted)
    }

    @Test
    fun `different IVs produce different ciphertexts`() {
        val plaintext = "same-value-every-time"

        val ciphertext1 = keyManager.encrypt(plaintext, key)
        val ciphertext2 = keyManager.encrypt(plaintext, key)

        // The Base64-encoded ciphertexts must differ because a fresh
        // random IV is used on every encryption call.
        assertTrue(
            ciphertext1 != ciphertext2,
            "Two encryptions of the same plaintext should produce different ciphertexts (different IV)"
        )
    }

    @Test
    fun `decrypt returns null on corrupted ciphertext`() {
        val encrypted = keyManager.encrypt("valid", key)
        val bytes = Base64.getDecoder().decode(encrypted)
        // Corrupt the first byte (part of the IV — will cause AEAD tag mismatch)
        bytes[0] = (bytes[0].toInt() xor 0xFF).toByte()
        val corrupted = Base64.getEncoder().encodeToString(bytes)

        assertNull(keyManager.decrypt(corrupted, key))
    }

    @Test
    fun `decrypt returns null on garbage input`() {
        assertNull(keyManager.decrypt("this-is-not-valid-base64!!!", key))
        assertNull(keyManager.decrypt("", key))
        assertNull(keyManager.decrypt("AAAA", key)) // valid base64 but too short
    }
}
