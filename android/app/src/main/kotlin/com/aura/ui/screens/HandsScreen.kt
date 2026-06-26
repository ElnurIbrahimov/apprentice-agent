package com.aura.ui.screens

import androidx.compose.foundation.layout.*; import androidx.compose.foundation.lazy.LazyColumn; import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape; import androidx.compose.material3.*
import androidx.compose.material.icons.Icons; import androidx.compose.material.icons.filled.Build
import androidx.compose.runtime.*; import androidx.compose.ui.Alignment; import androidx.compose.ui.Modifier; import androidx.compose.ui.text.font.FontWeight; import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel; import com.aura.hands.Hand; import com.aura.ui.viewmodel.HandsViewModel
import java.text.SimpleDateFormat; import java.util.*

@Composable
fun HandsScreen(viewModel: HandsViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsState()
    Column(Modifier.fillMaxSize().padding(16.dp)) {
        Text("Hands", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
        Text("Automation macros. ${state.hands.size} configured.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f))
        Spacer(Modifier.height(12.dp)); HorizontalDivider(); Spacer(Modifier.height(8.dp))
        if (state.loading) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
        else if (state.hands.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Column(horizontalAlignment = Alignment.CenterHorizontally) { Text("🤖", style = MaterialTheme.typography.displayLarge); Spacer(Modifier.height(8.dp)); Text("No hands yet", style = MaterialTheme.typography.titleMedium); Text("Hands are automation macros. Say 'run X' to trigger one.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.6f)) } }
        else LazyColumn(contentPadding = PaddingValues(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { items(state.hands, key = { it.id }) { h -> HandRow(h) } }
    }
}

@Composable
private fun HandRow(hand: Hand) {
    val fmt = SimpleDateFormat("MMM d", Locale.US)
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Build, null, tint = if (hand.enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f), modifier = Modifier.size(24.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) { Text(hand.name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold); Text(hand.triggerPhrase.ifBlank { "no trigger phrase" }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)) }
            Text(if (hand.enabled) "on" else "off", style = MaterialTheme.typography.labelSmall, color = if (hand.enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f))
            Spacer(Modifier.width(8.dp))
            Text(fmt.format(Date(hand.createdAt)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f))
        }
    }
}
