package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.core.url.SsrfGuard
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.OkHttpClient
import okhttp3.Request
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Weather tool using Open-Meteo (free, no API key required).
 *
 * API: https://api.open-meteo.com/v1/forecast?latitude=X&longitude=Y
 * &current=temperature_2m,wind_speed_10m,weather_code
 *
 * If no lat/lon is provided, the tool returns an error asking the agent
 * to get the device location first via the location_now tool.
 *
 * Risk: READ_ONLY (network egress only, no phone permissions).
 */
@Singleton
class WeatherTool @Inject constructor(
    private val httpClient: OkHttpClient,
) {
    private val json = Json { ignoreUnknownKeys = true }

    fun definition() = ToolDefinition(
        name = "weather",
        description = "Get current weather for a location. Requires latitude and longitude (get them from the location_now tool if the user doesn't provide coordinates).",
        parameters = ToolParameters(
            properties = mapOf(
                "latitude" to ToolProperty(
                    type = "number",
                    description = "Latitude of the location (e.g. 40.4093 for Baku)",
                ),
                "longitude" to ToolProperty(
                    type = "number",
                    description = "Longitude of the location (e.g. 49.8671 for Baku)",
                ),
            ),
            required = listOf("latitude", "longitude"),
        ),
    )

    val tool = Tool(
        name = "weather",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, _ ->
            val lat = (call.arguments["latitude"] as? Number)?.toDouble()
                ?: return@Tool ToolResult.Error("missing 'latitude' argument", "bad_args")
            val lon = (call.arguments["longitude"] as? Number)?.toDouble()
                ?: return@Tool ToolResult.Error("missing 'longitude' argument", "bad_args")

            try {
                val result = fetchWeather(lat, lon)
                ToolResult.Ok(result)
            } catch (e: Exception) {
                ToolResult.Error("Weather fetch failed: ${e.message}", "weather_error")
            }
        },
        category = "media",
    )

    private fun fetchWeather(lat: Double, lon: Double): String {
        val url = "https://api.open-meteo.com/v1/forecast?latitude=$lat&longitude=$lon" +
            "&current=temperature_2m,wind_speed_10m,weather_code"

        // SSRF guard — Open-Meteo is a public API but we validate anyway
        // to maintain the invariant that every httpClient.newCall goes
        // through the guard.
        val ssrfError = SsrfGuard.validate(url)
        if (ssrfError != null) throw RuntimeException(ssrfError)

        val req = Request.Builder()
            .url(url)
            .header("Accept", "application/json")
            .build()

        httpClient.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("Open-Meteo HTTP ${resp.code}")
            val body = resp.body?.string() ?: throw RuntimeException("Empty response")
            val root = json.parseToJsonElement(body).jsonObject
            val current = root["current"]?.jsonObject
                ?: throw RuntimeException("Missing 'current' in weather response")

            val temp = current["temperature_2m"]?.jsonPrimitive?.contentOrNull ?: "?"
            val wind = current["wind_speed_10m"]?.jsonPrimitive?.contentOrNull ?: "?"
            val code = current["weather_code"]?.jsonPrimitive?.contentOrNull?.toIntOrNull() ?: 0

            val description = weatherCodeToText(code)
            return "$temp°C, $description, wind $wind km/h"
        }
    }

    companion object {
        fun weatherCodeToText(code: Int): String = when (code) {
            0 -> "clear sky"
            1 -> "mainly clear"
            2 -> "partly cloudy"
            3 -> "overcast"
            45 -> "fog"
            48 -> "depositing rime fog"
            51 -> "light drizzle"
            53 -> "moderate drizzle"
            55 -> "dense drizzle"
            61 -> "slight rain"
            63 -> "moderate rain"
            65 -> "heavy rain"
            66 -> "freezing rain"
            67 -> "heavy freezing rain"
            71 -> "slight snow"
            73 -> "moderate snow"
            75 -> "heavy snow"
            77 -> "snow grains"
            80 -> "slight rain showers"
            81 -> "moderate rain showers"
            82 -> "violent rain showers"
            85 -> "slight snow showers"
            86 -> "heavy snow showers"
            95 -> "thunderstorm"
            96 -> "thunderstorm with slight hail"
            99 -> "thunderstorm with heavy hail"
            else -> "unknown ($code)"
        }
    }
}