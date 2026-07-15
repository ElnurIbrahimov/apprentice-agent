package com.aura.capabilities.worldlabs

import com.aura.capabilities.ImageResult
import com.aura.capabilities.ImageProvider
import com.aura.capabilities.CapabilityCatalogException
import com.aura.capabilities.CapabilityKind
import com.aura.capabilities.CapabilityProvider
import com.aura.capabilities.http.CapabilityHttp
import com.aura.capabilities.http.asJsonObjectOrNull
import com.aura.capabilities.http.stringOrNull
import com.aura.providers.ProviderKeys
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import javax.inject.Inject
import javax.inject.Singleton

/**
 * World Labs Marble API. Two-step flow:
 *   1. POST /marble/v1/worlds/generate with a text prompt -> operation_id
 *   2. Poll GET /marble/v1/worlds/{operation_id} until assets are ready, return asset URL.
 * https://docs.worldlabs.ai/api
 */
@Singleton
class WorldLabs3DProvider @Inject constructor(
    private val client: OkHttpClient,
    private val providerKeys: ProviderKeys,
) : CapabilityProvider {
    override val prefix = "worldlabs"
    override val displayName = "World Labs"
    override val kind = CapabilityKind.World3DGeneration
    private val apiKey: String get() = providerKeys.keyFor(prefix).orEmpty()
    override fun isConfigured(): Boolean = apiKey.isNotBlank()

    data class WorldResult(val worldUrl: String?, val operationId: String)

    suspend fun generateWorld(prompt: String): WorldResult = withContext(Dispatchers.IO) {
        val body = CapabilityHttp.buildJsonBody(
            "prompt" to prompt,
            "model" to "Marble",
        )
        val request = okhttp3.Request.Builder()
            .url("https://api.worldlabs.ai/marble/v1/worlds/generate")
            .header("WLT-Api-Key", apiKey)
            .header("Content-Type", "application/json")
            .post(body.toRequestBody("application/json".toMediaType()))
            .build()
        val createResp = client.newCall(request).execute()
        val createRaw = createResp.use { it.body?.string().orEmpty() }
        CapabilityHttp.classify(createResp, createRaw)
        val opId = CapabilityHttp.json.parseToJsonElement(createRaw)
            .asJsonObjectOrNull()?.get("operation_id").stringOrNull()
            ?: throw CapabilityCatalogException.MalformedResponseException("World Labs response missing operation_id")

        repeat(40) {
            delay(4_000L)
            val pollReq = okhttp3.Request.Builder()
                .url("https://api.worldlabs.ai/marble/v1/worlds/$opId")
                .header("WLT-Api-Key", apiKey)
                .build()
            val pollResp = client.newCall(pollReq).execute()
            val pollRaw = pollResp.use { it.body?.string().orEmpty() }
            CapabilityHttp.classify(pollResp, pollRaw)
            val data = CapabilityHttp.json.parseToJsonElement(pollRaw).asJsonObjectOrNull() ?: return@repeat
            val status = data["status"].stringOrNull()
            if (status == "completed" || status == "succeeded" || status == "success") {
                val url = data["world_url"].stringOrNull()
                    ?: data["url"].stringOrNull()
                return@withContext WorldResult(worldUrl = url, operationId = opId)
            }
            if (status == "failed" || status == "error") {
                throw CapabilityCatalogException.NetworkException("World Labs task $opId failed: $pollRaw")
            }
        }
        throw CapabilityCatalogException.MalformedResponseException("World Labs task $opId timed out")
    }
}
