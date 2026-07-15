package com.aura.capabilities.stability

import com.aura.capabilities.ImageProvider
import com.aura.capabilities.ImageRequest
import com.aura.capabilities.ImageResult
import com.aura.capabilities.http.CapabilityHttp
import com.aura.providers.ProviderKeys
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Stability AI v2beta stable-image-generate endpoint.
 * https://platform.stability.ai/docs/api-reference
 */
@Singleton
class StabilityImageProvider @Inject constructor(
    private val client: OkHttpClient,
    private val providerKeys: ProviderKeys,
) : ImageProvider {
    override val prefix = "stability"
    override val displayName = "Stability AI"
    private val apiKey: String get() = providerKeys.keyFor(prefix).orEmpty()
    override fun isConfigured(): Boolean = apiKey.isNotBlank()

    override suspend fun generate(req: ImageRequest): ImageResult = withContext(Dispatchers.IO) {
        val model = req.model.ifBlank { "stable-image-ultra" }
        val url = "https://api.stability.ai/v2beta/stable-image/generate/$model"
        val body = okhttp3.MultipartBody.Builder()
            .setType(okhttp3.MultipartBody.FORM)
            .addFormDataPart("prompt", req.prompt)
            .addFormDataPart("output_format", "png")
            .apply { req.negativePrompt?.let { addFormDataPart("negative_prompt", it) } }
            .build()
        val request = okhttp3.Request.Builder()
            .url(url)
            .header("authorization", "Bearer $apiKey")
            .header("accept", "image/*")
            .post(body)
            .build()
        val response = client.newCall(request).execute()
        val bytes = response.use { it.body?.bytes() ?: ByteArray(0) }
        if (!response.isSuccessful) {
            val msg = String(bytes, Charsets.UTF_8)
            throw com.aura.capabilities.CapabilityCatalogException.NetworkException(
                message = "Stability returned ${response.code}: $msg",
                statusCode = response.code,
            )
        }
        if (bytes.isEmpty()) {
            throw com.aura.capabilities.CapabilityCatalogException.MalformedResponseException("Stability returned empty image body")
        }
        ImageResult(bytes = bytes, mimeType = "image/png")
    }
}
