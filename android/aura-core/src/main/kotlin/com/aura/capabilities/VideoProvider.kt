package com.aura.capabilities

import kotlinx.coroutines.Job

data class VideoRequest(
    val prompt: String,
    val model: String = "",
    val durationSeconds: Int = 5,
    val aspectRatio: String = "16:9",
    val referenceImageUrl: String? = null,
)

data class VideoResult(
    val videoUrl: String? = null,
    val bytes: ByteArray? = null,
    val mimeType: String = "video/mp4",
    val taskId: String? = null,
) {
    override fun equals(other: Any?): Boolean = other is VideoResult &&
        videoUrl == other.videoUrl &&
        bytes?.contentEquals(other.bytes ?: ByteArray(0)) == true &&
        mimeType == other.mimeType &&
        taskId == other.taskId
    override fun hashCode(): Int {
        var result = videoUrl?.hashCode() ?: 0
        result = 31 * result + (bytes?.contentHashCode() ?: 0)
        result = 31 * result + mimeType.hashCode()
        result = 31 * result + (taskId?.hashCode() ?: 0)
        return result
    }
}

interface VideoProvider : CapabilityProvider {
    override val kind: CapabilityKind get() = CapabilityKind.VideoGeneration
    /**
     * Synchronous-by-default implementation: submits, polls, and returns the
     * completed video in a single call. Implementations must respect the
     * calling coroutine's [Job] for cooperative cancellation.
     */
    suspend fun generate(req: VideoRequest): VideoResult
}
