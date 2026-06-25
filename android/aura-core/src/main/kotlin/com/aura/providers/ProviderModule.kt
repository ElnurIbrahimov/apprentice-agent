package com.aura.providers

import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import dagger.multibindings.IntoMap
import dagger.multibindings.StringKey
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ProviderModule {

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    @Provides
    @IntoMap
    @StringKey("ollama")
    fun provideOllama(client: OkHttpClient): Provider = OllamaCloudProvider(
        prefix = "ollama",
        displayName = "Ollama Cloud",
        baseUrl = "https://ollama.com/v1",
        apiKey = System.getenv("OLLAMA_API_KEY") ?: "",
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("anthropic")
    fun provideAnthropic(client: OkHttpClient): Provider = AnthropicProvider(
        apiKey = System.getenv("ANTHROPIC_API_KEY") ?: "",
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("openai")
    fun provideOpenAI(client: OkHttpClient): Provider = OllamaCloudProvider(
        prefix = "openai",
        displayName = "OpenAI",
        baseUrl = "https://api.openai.com/v1",
        apiKey = System.getenv("OPENAI_API_KEY") ?: "",
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("deepseek")
    fun provideDeepSeek(client: OkHttpClient): Provider = OllamaCloudProvider(
        prefix = "deepseek",
        displayName = "DeepSeek",
        baseUrl = "https://api.deepseek.com/v1",
        apiKey = System.getenv("DEEPSEEK_API_KEY") ?: "",
        httpClient = client,
    )

    @Provides
    @IntoMap
    @StringKey("local")
    fun provideLocal(): Provider = LocalProvider(
        modelDir = java.io.File(System.getProperty("user.home") ?: "/sdcard", "Aura/models"),
    )
}
