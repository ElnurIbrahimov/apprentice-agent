package com.aura.tools

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
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
 * Get current network connectivity state. Mirrors aura/proactive/monitors/ (none — new).
 * Risk: READ_ONLY.
 */
@Singleton
class NetworkStateTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "network_state",
        description = "Get current network state: connected, type (wifi/cellular/ether/none), and whether the active network is metered (e.g. mobile data).",
        parameters = ToolParameters(properties = emptyMap(), required = emptyList()),
    )

    val tool = Tool(
        name = "network_state",
        description = definition().description,
        risk = ToolRisk.READ_ONLY,
        parameters = definition().parameters,
        execute = { call, ctx ->
            val mgr = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return@Tool ToolResult.Error("no ConnectivityManager", "system_error")
            val network = mgr.activeNetwork ?: return@Tool ToolResult.Ok("Offline — no active network.")
            val caps = mgr.getNetworkCapabilities(network) ?: return@Tool ToolResult.Ok("Offline (no capabilities).")
            val type = when {
                caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "wifi"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_VPN) -> "vpn"
                else -> "unknown"
            }
            val metered = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) !caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) else false
            val validated = caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
            ToolResult.Ok("Online: $type, ${if (metered) "metered" else "unmetered"}, ${if (validated) "validated" else "not yet validated"}")
        },
    )
}
