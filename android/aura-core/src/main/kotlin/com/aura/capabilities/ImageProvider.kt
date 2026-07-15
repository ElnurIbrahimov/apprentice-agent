package com.aura.capabilities

data class ImageRequest(
    val prompt: String,
    val model: String = "",
    val width: Int = 1024,
    val height: Int = 1024,
    val negativePrompt: String? = null,
)

/**
 * [url] is provided when the provider returns a hosted image URL.
 * [bytes] is provided when the provider returns raw bytes (Stability, OpenAI b64).
 * At least one is set. [mimeType] defaults to image/png.
 */
data class ImageResult(
    val url: String? = null,
    val bytes: ByteArray? = null,
    val mimeType: String = "image/png",
) {
    override fun equals(other: Any?): Boolean = other is ImageResult &&
        url == other.url &&
        bytes?.contentEquals(other.bytes ?: ByteArray(0)) == true &&
        mimeType == other.mimeType
    override fun hashCode(): Int {
        var result = url?.hashCode() ?: 0
        result = 31 * result + (bytes?.contentHashCode() ?: 0)
        result = 31 * result + mimeType.hashCode()
        return result
    }
}

interface ImageProvider : CapabilityProvider {
    override val kind: CapabilityKind get() = CapabilityKind.ImageGeneration
    suspend fun generate(req: ImageRequest): ImageResult
}
