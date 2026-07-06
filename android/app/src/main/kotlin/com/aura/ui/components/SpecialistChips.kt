package com.aura.ui.components

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.aura.agent.Specialist

/**
 * Horizontal scrollable row of chips for each specialist.
 *
 * - [selected] is the currently active specialist (tapped by user).
 * - [suggested] is the auto-detected specialist from the draft text (may be null).
 * - [onSelect] is called when a chip is tapped; passing `null` clears the selection.
 */
@Composable
fun SpecialistChips(
    selected: Specialist?,
    suggested: Specialist?,
    onSelect: (Specialist?) -> Unit,
    modifier: Modifier = Modifier,
    specialists: List<Specialist> = Specialist.ALL,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for (specialist in specialists) {
            val isSelected = specialist == selected
            val isSuggested = !isSelected && specialist == suggested

            FilterChip(
                selected = isSelected,
                onClick = {
                    // Toggle: if already selected, clear; otherwise select this one
                    onSelect(if (isSelected) null else specialist)
                },
                label = {
                    Text(
                        text = "${specialist.icon} ${specialist.name.replaceFirstChar { it.uppercase() }}",
                        style = MaterialTheme.typography.labelMedium,
                    )
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                    selectedLabelColor = MaterialTheme.colorScheme.onPrimaryContainer,
                ),
                border = if (isSuggested && !isSelected) {
                    FilterChipDefaults.filterChipBorder(
                        enabled = true,
                        selected = false,
                        borderColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f),
                    )
                } else {
                    FilterChipDefaults.filterChipBorder(enabled = true, selected = isSelected)
                },
            )
        }
    }
}
