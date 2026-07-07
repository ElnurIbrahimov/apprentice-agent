package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.text.Regex
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Alignment
import androidx.compose.foundation.shape.CircleShape

/**
 * Lightweight markdown renderer. Parses the 80% case:
 * - Code blocks (```lang ... ```)
 * - Bold (**text**)
 * - Italic (*text* / _text_)
 * - Inline code (`code`)
 * - Headers (# ## ###)
 * - Bullet lists (- item / * item)
 *
 * No external library. Renders to AnnotatedString for Text composable.
 */

private val codeBlockRegex = Regex("```(\\w*)\\n([\\s\\S]*?)```")
private val boldRegex = Regex("\\*\\*(.+?)\\*\\*")
private val italicRegex = Regex("(?<!\\*)\\*(?!\\s)(.+?)(?<!\\s)\\*(?!\\*)|_(.+?)_")
private val inlineCodeRegex = Regex("`([^`]+)`")
private val headerRegex = Regex("^(#{1,3})\\s+(.+)$", RegexOption.MULTILINE)
private val bulletRegex = Regex("^[\\-*]\\s+(.+)$", RegexOption.MULTILINE)

/**
 * Parse markdown into an AnnotatedString for inline text rendering.
 * Handles bold, italic, inline code, headers, and bullet prefixes.
 */
fun parseMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    val lines = text.split("\n")
    for ((index, line) in lines.withIndex()) {
        val headerMatch = headerRegex.find(line)
        if (headerMatch != null) {
            val level = headerMatch.groupValues[1].length
            val headerText = headerMatch.groupValues[2]
            pushStyle(SpanStyle(fontWeight = FontWeight.Bold, fontSize = (16 - level).sp))
            append(headerText)
            pop()
        } else {
            val bulletMatch = bulletRegex.find(line)
            if (bulletMatch != null) {
                append("• ")
                appendInlineMarkdown(bulletMatch.groupValues[1])
            } else {
                appendInlineMarkdown(line)
            }
        }
        if (index < lines.lastIndex) append("\n")
    }
}

private fun AnnotatedString.Builder.appendInlineMarkdown(text: String) {
    var pos = 0
    val combined = Regex("(\\*\\*(.+?)\\*\\*)|(`([^`]+)`)|(\\*([^*]+?)\\*)|(_([^_]+?)_)")
    combined.findAll(text).forEach { match ->
        if (match.range.first > pos) append(text.substring(pos, match.range.first))
        when {
            match.groupValues[2].isNotEmpty() -> {
                pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                append(match.groupValues[2])
                pop()
            }
            match.groupValues[4].isNotEmpty() -> {
                pushStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = Color(0x1A808080)))
                append(match.groupValues[4])
                pop()
            }
            match.groupValues[6].isNotEmpty() -> {
                pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                append(match.groupValues[6])
                pop()
            }
            match.groupValues[8].isNotEmpty() -> {
                pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                append(match.groupValues[8])
                pop()
            }
        }
        pos = match.range.last + 1
    }
    if (pos < text.length) append(text.substring(pos))
}

/**
 * Check if text contains markdown code blocks.
 */
fun hasCodeBlock(text: String): Boolean = codeBlockRegex.containsMatchIn(text)

/**
 * Extract code blocks from text. Returns list of (language, code) pairs.
 */
fun extractCodeBlocks(text: String): List<Pair<String, String>> {
    return codeBlockRegex.findAll(text).map { match ->
        Pair(match.groupValues[1], match.groupValues[2])
    }.toList()
}

/**
 * Render a markdown string as a Compose Column.
 * Code blocks are rendered as separate boxed sections; the rest is
 * rendered as AnnotatedString in Text composables.
 */
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    style: androidx.compose.ui.text.TextStyle = MaterialTheme.typography.bodyMedium,
) {
    Column(modifier = modifier) {
        val blocks = splitMarkdownBlocks(text)
        for (block in blocks) {
            when (block) {
                is MarkdownBlock.Code -> {
                    CodeBlock(language = block.language, code = block.code)
                    Spacer(Modifier.height(6.dp))
                }
                is MarkdownBlock.Text -> {
                    Text(
                        text = parseMarkdown(block.content),
                        style = style,
                        overflow = TextOverflow.Visible,
                    )
                    Spacer(Modifier.height(4.dp))
                }
            }
        }
    }
}

private sealed class MarkdownBlock {
    data class Text(val content: String) : MarkdownBlock()
    data class Code(val language: String, val code: String) : MarkdownBlock()
}

private fun splitMarkdownBlocks(text: String): List<MarkdownBlock> {
    val blocks = mutableListOf<MarkdownBlock>()
    var pos = 0
    codeBlockRegex.findAll(text).forEach { match ->
        if (match.range.first > pos) {
            val before = text.substring(pos, match.range.first).trim('\n')
            if (before.isNotBlank()) blocks.add(MarkdownBlock.Text(before))
        }
        blocks.add(MarkdownBlock.Code(match.groupValues[1], match.groupValues[2]))
        pos = match.range.last + 1
    }
    if (pos < text.length) {
        val remaining = text.substring(pos).trim('\n')
        if (remaining.isNotBlank()) blocks.add(MarkdownBlock.Text(remaining))
    }
    return blocks
}

@Composable
private fun CodeBlock(language: String, code: String) {
    // Premium code block: dark surface regardless of theme, mono
    // font, subtle border. Mirrors the look of a real IDE.
    val codeBg = MaterialTheme.colorScheme.surface
    val codeFg = MaterialTheme.colorScheme.onSurface
    val borderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(codeBg, RoundedCornerShape(12.dp))
            .border(1.dp, borderColor, RoundedCornerShape(12.dp))
            .padding(14.dp),
    ) {
        Column {
            if (language.isNotBlank()) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 8.dp),
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                MaterialTheme.colorScheme.primary,
                                androidx.compose.foundation.shape.CircleShape,
                            ),
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = language,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
            }
            Text(
                text = code,
                style = MaterialTheme.typography.bodySmall.copy(
                    fontFamily = FontFamily.Monospace,
                ),
                color = codeFg,
                overflow = TextOverflow.Visible,
            )
        }
    }
}