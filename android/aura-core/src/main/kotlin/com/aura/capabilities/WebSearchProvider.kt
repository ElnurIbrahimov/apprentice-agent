package com.aura.capabilities

data class WebSearchRequest(
    val query: String,
    val numResults: Int = 5,
    val includeText: Boolean = true,
)

data class WebSearchResult(
    val title: String,
    val url: String,
    val snippet: String,
    val text: String? = null,
    val score: Double? = null,
)

interface WebSearchProvider : CapabilityProvider {
    override val kind: CapabilityKind get() = CapabilityKind.WebSearch
    suspend fun search(req: WebSearchRequest): List<WebSearchResult>
}
