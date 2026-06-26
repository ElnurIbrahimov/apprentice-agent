package com.aura.providers

import okhttp3.OkHttpClient

/**
 * OpenAI-compatible chat completions client for Ollama Cloud.
 *
 * This is a thin wrapper around [OpenAiCompatProvider]; all the streaming,
 * tool-call parsing, and request-building logic lives in the parent class.
 *
 * The API key is read from [ProviderKeys] on every [chat] call, not at
 * construction time, so changes the user makes in the Settings UI take
 * effect immediately without restarting the app.
 */
class OllamaCloudProvider(
    override val prefix: String,
    override val displayName: String,
    private val baseUrl: String,
    providerKeys: ProviderKeys,
    httpClient: OkHttpClient,
) : OpenAiCompatProvider(
    prefix = prefix,
    displayName = displayName,
    baseUrl = baseUrl,
    providerKeys = providerKeys,
    httpClient = httpClient,
)
