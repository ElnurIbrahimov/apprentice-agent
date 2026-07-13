package com.aura.providers

import okhttp3.Interceptor
import okhttp3.OkHttpClient

/**
 * OpenRouter provider — a thin wrapper around [OpenAiCompatProvider].
 *
 * OpenRouter is a unified API gateway that provides access to many LLMs
 * through a single OpenAI-compatible endpoint. It requires specific headers
 * for identification.
 *
 * Base URL: https://openrouter.ai/api/v1
 * Default models: derived from live /v1/models endpoint
 *
 * Required headers:
 * - HTTP-Referer: "https://aura-android"
 * - X-Title: "Aura Android"
 *
 * The API key is read from [ProviderKeys] on every [chat] call, not at
 * construction time, so changes the user makes in the Settings UI take
 * effect immediately without restarting the app.
 */
class OpenRouterProvider(
    providerKeys: ProviderKeys,
    httpClient: OkHttpClient,
    baseUrl: String = "https://openrouter.ai/api/v1",
) : OpenAiCompatProvider(
    prefix = "openrouter",
    displayName = "OpenRouter",
    baseUrl = baseUrl,
    providerKeys = providerKeys,
    httpClient = httpClient.newBuilder()
        .addInterceptor(Interceptor { chain ->
            val request = chain.request().newBuilder()
                .addHeader("HTTP-Referer", "https://aura-android")
                .addHeader("X-Title", "Aura Android")
                .build()
            chain.proceed(request)
        })
        .build(),
)
