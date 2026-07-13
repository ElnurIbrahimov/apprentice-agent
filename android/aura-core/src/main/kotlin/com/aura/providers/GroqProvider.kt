package com.aura.providers

import okhttp3.OkHttpClient

/**
 * Groq provider — a thin wrapper around [OpenAiCompatProvider].
 *
 * Base URL: https://api.groq.com/openai/v1/
 * Default models: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
 *
 * The API key is read from [ProviderKeys] on every [chat] call, not at
 * construction time, so changes the user makes in the Settings UI take
 * effect immediately without restarting the app.
 */
class GroqProvider(
    providerKeys: ProviderKeys,
    httpClient: OkHttpClient,
    baseUrl: String = "https://api.groq.com/openai/v1",
) : OpenAiCompatProvider(
    prefix = "groq",
    displayName = "Groq",
    baseUrl = baseUrl,
    providerKeys = providerKeys,
    httpClient = httpClient,
)
