package com.aura.security

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import java.util.Base64
import javax.crypto.AEADBadTagException
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Cryptographic helper that manages a 256-bit AES-GCM key and provides
 * encrypt/decrypt operations.
 *
 * In production, the key lives in the Android Keystore under the alias
 * "aura_secure_prefs". For testing, pass `keyStore = null` to use an
 * in-memory key generated via `KeyGenerator("AES")` — no Robolectric
 * or Android dependencies required.
 */
class KeyManager(private val keyStore: KeyStore? = null) {

    companion object {
        private const val KEY_ALIAS = "aura_secure_prefs"
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
        private const val GCM_IV_LENGTH = 12
        private const val GCM_TAG_LENGTH_BITS = 128
    }

    /**
     * Returns a 256-bit AES [SecretKey].
     *
     * - If [keyStore] is non-null and contains an entry for [KEY_ALIAS], that
     *   key is returned.
     * - If [keyStore] is non-null but has no such entry, a new key is
     *   generated inside the Android Keystore via [KeyGenParameterSpec] with
     *   AES + GCM + NoPadding, [setUserAuthenticationRequired] = false.
     * - If [keyStore] is null (test mode), an in-memory AES-256 key is
     *   generated.
     */
    fun getOrCreateKey(): SecretKey {
        if (keyStore != null) {
            val existing = keyStore.getEntry(KEY_ALIAS, null)
            if (existing is KeyStore.SecretKeyEntry) {
                return existing.secretKey
            }
            val keyGenerator = KeyGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_AES,
                ANDROID_KEYSTORE
            )
            keyGenerator.init(
                KeyGenParameterSpec.Builder(
                    KEY_ALIAS,
                    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
                )
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256)
                    .setUserAuthenticationRequired(false)
                    .build()
            )
            return keyGenerator.generateKey()
        }
        // In-memory fallback for testing
        val keyGen = KeyGenerator.getInstance("AES")
        keyGen.init(256)
        return keyGen.generateKey()
    }

    /**
     * Encrypts [plaintext] with AES-256/GCM/NoPadding using the given [key].
     *
     * The ciphertext is returned as a Base64-encoded string containing the
     * 12-byte IV followed by the GCM ciphertext (which includes the 128-bit
     * authentication tag).
     */
    fun encrypt(plaintext: String, key: SecretKey): String {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key)
        val iv = cipher.iv
        val ciphertext = cipher.doFinal(plaintext.toByteArray(Charsets.UTF_8))
        val combined = ByteArray(iv.size + ciphertext.size)
        System.arraycopy(iv, 0, combined, 0, iv.size)
        System.arraycopy(ciphertext, 0, combined, iv.size, ciphertext.size)
        return Base64.getEncoder().encodeToString(combined)
    }

    /**
     * Decrypts a Base64-encoded ciphertext (IV + AES/GCM payload) produced
     * by [encrypt].
     *
     * Returns `null` gracefully if the AEAD authentication tag check fails
     * (e.g. the keystore was wiped and the key no longer matches the stored
     * ciphertext), rather than throwing.
     */
    fun decrypt(ciphertextB64: String, key: SecretKey): String? {
        return try {
            val combined = Base64.getDecoder().decode(ciphertextB64)
            if (combined.size < GCM_IV_LENGTH + 1) return null
            val iv = combined.copyOfRange(0, GCM_IV_LENGTH)
            val ciphertext = combined.copyOfRange(GCM_IV_LENGTH, combined.size)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(GCM_TAG_LENGTH_BITS, iv))
            String(cipher.doFinal(ciphertext), Charsets.UTF_8)
        } catch (_: AEADBadTagException) {
            null // auth tag mismatch — caller can decide to fall back gracefully
        } catch (_: IllegalArgumentException) {
            null // invalid Base64 input
        } catch (_: Exception) {
            null // any other unexpected error; return null gracefully in v1
        }
    }
}
