package com.aura.providers

/**
 * Represents the lifecycle state of a single provider's API key credential.
 *
 * Terminal states that block further automatic transitions:
 * - [NotConfigured]: no key has been stored (DataStore has no entry).
 * - [Saved]: a key has been successfully stored and loaded from secure storage.
 * - [Valid]: the credential has been validated against the provider's endpoint.
 * - [Invalid]: the credential was rejected by the provider's endpoint.
 * - [StorageError]: a storage or decryption error occurred; the credential
 *   cannot be read. Once reached this state is terminal for the load cycle.
 *
 * Non-terminal:
 * - [Loading]: the initial DataStore load is in progress (only meaningful
 *   during the first async init of [ProviderKeys]).
 */
enum class ProviderCredentialState {
    NotConfigured,
    Loading,
    Saved,
    Valid,
    Invalid,
    StorageError,
}
