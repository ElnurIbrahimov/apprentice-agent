package com.aura.tools

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import androidx.core.content.ContextCompat
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Get current GPS location. v1: last-known fix from any provider (fast, no GPS lock).
 * v1.5: real-time fused location. This is phone-native (no Aura equivalent).
 * Risk: PRIVACY (ACCESS_FINE_LOCATION).
 */
@Singleton
class LocationNowTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "location_now",
        description = "Get the device's current location as latitude/longitude/accuracy.",
        parameters = ToolParameters(properties = emptyMap(), required = emptyList()),
    )

    val tool = Tool(
        name = "location_now",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        requiredPermissions = listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
        parameters = definition().parameters,
        execute = { call, ctx ->
            val granted = requiredPermissionsMet()
            if (!granted) {
                return@Tool ToolResult.NeedsPermission(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    "Location access is needed to report your position.",
                )
            }
            val loc = lastKnown()
            if (loc == null) {
                ToolResult.Ok("Location unavailable. Make sure GPS is on and a fix has been acquired.")
            } else {
                ToolResult.Ok(formatLocation(loc))
            }
        },
    )

    private fun requiredPermissionsMet(): Boolean = listOf(
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION,
    ).all { perm ->
        ContextCompat.checkSelfPermission(context, perm) == PackageManager.PERMISSION_GRANTED
    }

    @SuppressLint("MissingPermission")
    private fun lastKnown(): Location? {
        val mgr = ContextCompat.getSystemService(context, LocationManager::class.java) ?: return null
        val providers = mgr.getProviders(true)
        var best: Location? = null
        for (p in providers) {
            val l = try { mgr.getLastKnownLocation(p) } catch (e: SecurityException) { null } ?: continue
            if (best == null || l.accuracy < best.accuracy) best = l
        }
        return best
    }

    private fun formatLocation(loc: Location): String {
        val time = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.US).format(java.util.Date(loc.time))
        return "lat=${loc.latitude}, lon=${loc.longitude}, accuracy=${loc.accuracy}m, provider=${loc.provider}, time=$time"
    }
}
