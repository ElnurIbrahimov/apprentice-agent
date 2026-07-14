package com.aura.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.ClickableText
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.text.Regex
import androidx.compose.foundation.border

/**
 * Lightweight markdown renderer. Handles the 80% case with regex
 * — no external library.
 *
 * Supported:
 * - Code blocks (```lang ... ```)
 * - Bold (**text**) and italic (*text* / _text_)
 * - Bold + italic (***text***)
 * - Inline code (`code`)
 * - Headers (# ## ###)
 * - Bullet lists (- item / * item)
 * - Ordered lists (1. item)
 * - Tables (| col1 | col2 |  +  |---|---|)
 * - Links ([label](url)) — clickable; taps open the URL via the system URI handler
 *
 * Trade-off vs a real markdown lib: edge cases like nested code
 * inside link text, HTML, and image embeds aren't supported. Good
 * enough for chat output where 99% of LLM responses are short
 * prose with bold/italic/code/headers/lists/links.
 */

private val codeBlockRegex = Regex("```(\\w*)\\n([\\s\\S]*?)```")
private val headerRegex = Regex("^(#{1,3})\\s+(.+)$", RegexOption.MULTILINE)
private val orderedListRegex = Regex("^(\\d+)\\.\\s+(.+)$", RegexOption.MULTILINE)
private val bulletRegex = Regex("^[-*]\\s+(.+)$", RegexOption.MULTILINE)
// Table delimiter row: pipes + dashes, optional colons for alignment.
private val tableDelimiterRegex = Regex("^\\s*\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$", RegexOption.MULTILINE)
// Detect a table header row: at least one pipe on a line, not a code block, not a bullet/header.
private val tableRowRegex = Regex("^\\s*\\|.*\\|\\s*$", RegexOption.MULTILINE)
// Combined inline patterns. Bold (group 1) takes priority over
// italic. Bold-italic (***text***) is its own pattern, handled
// before plain bold so the asterisks aren't consumed as plain bold.
private val boldItalicRegex = Regex("\\*\\*\\*(?=\\S)([^*]+?)(?<=\\S)\\*\\*\\*")
private val boldRegex = Regex("\\*\\*(?=\\S)([^*]+?)(?<=\\S)\\*\\*")
// Italic with single * — negative lookbehind/lookahead to avoid
// matching inside **bold** and to require non-whitespace on both
// sides of the content (prevents `* foo *` from being italic).
private val italicStarRegex = Regex("(?<!\\*)\\*(?!\\s)(?!\\*)([^*]+?)(?<!\\s)(?<!\\*)\\*(?!\\*)")
// Underscore italic per CommonMark: the opening underscore must
// not be preceded by a non-whitespace char (so `_x` and `(_x)`
// match, but `a_x` does not); the closing underscore must not be
// followed by a non-whitespace char (so `x_` and `x_)` match, but
// `x_a` does not). \w includes underscore, so we use a custom
// set: a "non-emphasis char" is anything that's not whitespace
// and not a markdown emphasis marker (* or _).
private val italicUnderscoreRegex = Regex("(?<![A-Za-z0-9*])_(?!\\s)([^_\\n]+?)(?<!\\s)_(?![A-Za-z0-9*])")
private val inlineCodeRegex = Regex("`([^`\\n]+)`")
private val linkRegex = Regex("\\[([^\\]]+)\\]\\(([^)]+)\\)")
private val citationMarkerRegex = Regex("\\[(\\d{1,3})](?!\\()")
private val superscriptDigits = mapOf(
    '0' to '⁰', '1' to '¹', '2' to '²', '3' to '³', '4' to '⁴',
    '5' to '⁵', '6' to '⁶', '7' to '⁷', '8' to '⁸', '9' to '⁹',
)

/**
 * Convert only known, standalone citation markers to compact superscripts.
 * Markdown links and fenced code are deliberately left untouched.
 */
internal fun renderCitationMarkers(text: String, validIndices: Set<Int>): String {
    if (text.isBlank() || validIndices.isEmpty()) return text
    fun transform(segment: String): String = citationMarkerRegex.replace(segment) { match ->
        val index = match.groupValues[1].toIntOrNull()
        if (index !in validIndices) {
            match.value
        } else {
            "⁽" + match.groupValues[1].map { superscriptDigits[it] ?: it }.joinToString("") + "⁾"
        }
    }

    val output = StringBuilder(text.length)
    var cursor = 0
    for (codeBlock in codeBlockRegex.findAll(text)) {
        output.append(transform(text.substring(cursor, codeBlock.range.first)))
        output.append(codeBlock.value)
        cursor = codeBlock.range.last + 1
    }
    output.append(transform(text.substring(cursor)))
    return output.toString()
}

/** Only web links from untrusted model output may reach the system URI handler. */
internal fun isSafeMarkdownUrl(url: String): Boolean = runCatching {
    val uri = java.net.URI(url.trim())
    (uri.scheme == "https" || uri.scheme == "http") && !uri.host.isNullOrBlank()
}.getOrDefault(false)

// Colors used for inline markdown rendering. Captured at
// composition time (the only legal place to read
// MaterialTheme.colorScheme) and passed into the non-composable
// parser/builder. Using Compose colors in a non-composable function
// is a compile error — capturing them is the workaround.
data class MarkdownColors(
    val link: Color,
    val linkDim: Color,
)

/**
 * Composable helper that captures the current theme's colors and
 * returns a [MarkdownColors] bundle. Callers that need to render
 * markdown from a non-composable context (e.g. building an
 * AnnotatedString) can capture this once at composition and pass
 * it in.
 */
@Composable
fun rememberMarkdownColors(): MarkdownColors = MarkdownColors(
    link = MaterialTheme.colorScheme.primary,
    linkDim = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
)

/**
 * Parse markdown into an AnnotatedString for inline text rendering.
 * Handles bold, italic, inline code, links, headers, and bullet prefixes.
 */
@Composable
fun parseMarkdown(text: String): AnnotatedString {
    val colors = MarkdownColors(
        link = MaterialTheme.colorScheme.primary,
        linkDim = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
    )
    return parseMarkdown(text, colors)
}

/**
 * Non-composable parser used internally. Public @Composable
 * [parseMarkdown] captures theme colors and delegates here. Also
 * exposed as internal so [com.aura.ui.components.StreamingText]
 * can call it from a non-composable AnnotatedString builder.
 */
internal fun parseMarkdown(text: String, colors: MarkdownColors): AnnotatedString = buildAnnotatedString {
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
            val orderedMatch = orderedListRegex.find(line)
            if (orderedMatch != null) {
                val n = orderedMatch.groupValues[1]
                val content = orderedMatch.groupValues[2]
                append("${n}. ")
                appendInlineMarkdown(content, colors)
            } else {
                val bulletMatch = bulletRegex.find(line)
                if (bulletMatch != null) {
                    append("• ")
                    appendInlineMarkdown(bulletMatch.groupValues[1], colors)
                } else {
                    appendInlineMarkdown(line, colors)
                }
            }
        }
        if (index < lines.lastIndex) append("\n")
    }
}

/**
 * Same as [parseMarkdown] but returns an [AnnotatedString] with link annotations
 * attached so that [ClickableText] can open URLs.
 */
internal fun parseMarkdownClickable(text: String, colors: MarkdownColors): AnnotatedString = buildAnnotatedString {
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
            val orderedMatch = orderedListRegex.find(line)
            if (orderedMatch != null) {
                val n = orderedMatch.groupValues[1]
                val content = orderedMatch.groupValues[2]
                append("${n}. ")
                appendInlineMarkdownClickable(content, colors)
            } else {
                val bulletMatch = bulletRegex.find(line)
                if (bulletMatch != null) {
                    append("• ")
                    appendInlineMarkdownClickable(bulletMatch.groupValues[1], colors)
                } else {
                    appendInlineMarkdownClickable(line, colors)
                }
            }
        }
        if (index < lines.lastIndex) append("\n")
    }
}

private fun AnnotatedString.Builder.appendInlineMarkdown(text: String, colors: MarkdownColors) {
    if (text.isEmpty()) return
    // Run all inline patterns. To avoid double-processing, mark
    // matched ranges and skip them in later patterns. The simpler
    // approach below uses a single pass through text positions,
    // applying patterns in priority order at each position.
    var pos = 0
    while (pos < text.length) {
        val remaining = text.substring(pos)
        // Find the next match of any pattern.
        val candidates = listOfNotNull(
            boldItalicRegex.find(remaining)?.let { Triple("bi", it, 1) },
            boldRegex.find(remaining)?.let { Triple("b", it, 1) },
            inlineCodeRegex.find(remaining)?.let { Triple("c", it, 1) },
            linkRegex.find(remaining)?.let { Triple("l", it, 1) },
            italicStarRegex.find(remaining)?.let { Triple("is", it, 1) },
            italicUnderscoreRegex.find(remaining)?.let { Triple("iu", it, 1) },
        )
        val next = candidates.minByOrNull { it.second.range.first }
        if (next == null) {
            append(text.substring(pos))
            return
        }
        val (kind, match, contentGroup) = next
        val startInRemaining = match.range.first
        if (startInRemaining > 0) {
            append(text.substring(pos, pos + startInRemaining))
        }
        when (kind) {
            "bi" -> {
                pushStyle(SpanStyle(fontWeight = FontWeight.Bold, fontStyle = FontStyle.Italic))
                append(match.groupValues[contentGroup])
                pop()
            }
            "b" -> {
                pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                append(match.groupValues[contentGroup])
                pop()
            }
            "c" -> {
                pushStyle(SpanStyle(
                    fontFamily = FontFamily.Monospace,
                    background = Color(0x1A808080),
                ))
                append(match.groupValues[contentGroup])
                pop()
            }
            "l" -> {
                val label = match.groupValues[1]
                val url = match.groupValues[2]
                pushStyle(SpanStyle(
                    color = colors.link,
                    textDecoration = TextDecoration.Underline,
                ))
                append(label)
                pop()
                append(" (")
                pushStyle(SpanStyle(
                    color = colors.linkDim,
                ))
                append(url)
                pop()
                append(")")
            }
            "is" -> {
                pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                append(match.groupValues[contentGroup])
                pop()
            }
            "iu" -> {
                pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                append(match.groupValues[contentGroup])
                pop()
            }
        }
        pos += match.range.last + 1
    }
}

private fun AnnotatedString.Builder.appendInlineMarkdownClickable(text: String, colors: MarkdownColors) {
    if (text.isEmpty()) return
    var pos = 0
    while (pos < text.length) {
        val remaining = text.substring(pos)
        val candidates = listOfNotNull(
            boldItalicRegex.find(remaining)?.let { Triple("bi", it, 1) },
            boldRegex.find(remaining)?.let { Triple("b", it, 1) },
            inlineCodeRegex.find(remaining)?.let { Triple("c", it, 1) },
            linkRegex.find(remaining)?.let { Triple("l", it, 1) },
            italicStarRegex.find(remaining)?.let { Triple("is", it, 1) },
            italicUnderscoreRegex.find(remaining)?.let { Triple("iu", it, 1) },
        )
        val next = candidates.minByOrNull { it.second.range.first }
        if (next == null) {
            append(text.substring(pos))
            return
        }
        val (kind, match, contentGroup) = next
        val startInRemaining = match.range.first
        if (startInRemaining > 0) {
            append(text.substring(pos, pos + startInRemaining))
        }
        when (kind) {
            "bi" -> {
                pushStyle(SpanStyle(fontWeight = FontWeight.Bold, fontStyle = FontStyle.Italic))
                append(match.groupValues[contentGroup])
                pop()
            }
            "b" -> {
                pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                append(match.groupValues[contentGroup])
                pop()
            }
            "c" -> {
                pushStyle(SpanStyle(
                    fontFamily = FontFamily.Monospace,
                    background = Color(0x1A808080),
                ))
                append(match.groupValues[contentGroup])
                pop()
            }
            "l" -> {
                val label = match.groupValues[1]
                val url = match.groupValues[2]
                val start = this.length
                pushStyle(SpanStyle(
                    color = colors.link,
                    textDecoration = TextDecoration.Underline,
                ))
                append(label)
                pop()
                if (isSafeMarkdownUrl(url)) {
                    addStringAnnotation(
                        tag = "URL",
                        annotation = url,
                        start = start,
                        end = start + label.length,
                    )
                }
            }
            "is" -> {
                pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                append(match.groupValues[contentGroup])
                pop()
            }
            "iu" -> {
                pushStyle(SpanStyle(fontStyle = FontStyle.Italic))
                append(match.groupValues[contentGroup])
                pop()
            }
        }
        pos += match.range.last + 1
    }
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
 * rendered as AnnotatedString in Text composables. Tables are
 * detected by a row containing pipes followed by a delimiter row
 * and rendered as a basic grid.
 */
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    style: androidx.compose.ui.text.TextStyle = MaterialTheme.typography.bodyMedium,
) {
    val colors = MarkdownColors(
        link = MaterialTheme.colorScheme.primary,
        linkDim = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
    )
    val annotated = parseMarkdownClickable(text, colors)
    val uriHandler = LocalUriHandler.current
    ClickableText(
        text = annotated,
        style = style,
        modifier = modifier,
        onClick = { offset ->
            annotated.getStringAnnotations(tag = "URL", start = offset, end = offset)
                .firstOrNull()
                ?.let { annotation ->
                    runCatching { uriHandler.openUri(annotation.item) }
                }
        },
    )
}

private sealed class MarkdownBlock {
    data class Text(val content: String) : MarkdownBlock()
    data class Code(val language: String, val code: String) : MarkdownBlock()
    data class Table(val headers: List<String>, val rows: List<List<String>>) : MarkdownBlock()
}

/**
 * Render a markdown string as a Compose Column.
 * Code blocks are rendered as separate boxed sections; the rest is
 * rendered as clickable AnnotatedString so that inline links open URLs.
 */
@Composable
fun MarkdownColumn(
    text: String,
    modifier: Modifier = Modifier,
    style: androidx.compose.ui.text.TextStyle = MaterialTheme.typography.bodyMedium,
) {
    val colors = MarkdownColors(
        link = MaterialTheme.colorScheme.primary,
        linkDim = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.5f),
    )
    Column(modifier = modifier) {
        val blocks = splitMarkdownBlocks(text)
        for (block in blocks) {
            when (block) {
                is MarkdownBlock.Code -> {
                    CodeBlock(language = block.language, code = block.code)
                    Spacer(Modifier.height(6.dp))
                }
                is MarkdownBlock.Table -> {
                    TableBlock(headers = block.headers, rows = block.rows)
                    Spacer(Modifier.height(6.dp))
                }
                is MarkdownBlock.Text -> {
                    ClickableMarkdownBlock(text = block.content, colors = colors, style = style)
                    Spacer(Modifier.height(4.dp))
                }
            }
        }
    }
}

@Composable
private fun ClickableMarkdownBlock(
    text: String,
    colors: MarkdownColors,
    style: androidx.compose.ui.text.TextStyle,
) {
    val annotated = parseMarkdownClickable(text, colors)
    val uriHandler = LocalUriHandler.current
    ClickableText(
        text = annotated,
        style = style,
        overflow = TextOverflow.Visible,
        onClick = { offset ->
            annotated.getStringAnnotations(tag = "URL", start = offset, end = offset)
                .firstOrNull()
                ?.let { annotation ->
                    runCatching { uriHandler.openUri(annotation.item) }
                }
        },
    )
}

private fun splitMarkdownBlocks(text: String): List<MarkdownBlock> {
    val blocks = mutableListOf<MarkdownBlock>()
    val codeRanges = codeBlockRegex.findAll(text).map { it.range }.toList()
    var pos = 0
    fun appendText(s: String) {
        if (s.isBlank()) return
        // Detect a table: a row of pipes, a delimiter row, then 1+
        // more pipe rows. Tables must be contiguous and not cross
        // a code block boundary.
        val lines = s.split("\n")
        val tableBlocks = mutableListOf<Pair<IntRange, MarkdownBlock.Table>>()
        var i = 0
        while (i < lines.size - 1) {
            val headerLine = lines[i]
            val delimLine = lines[i + 1]
            if (headerLine.contains("|") &&
                tableDelimiterRegex.matches(delimLine)
            ) {
                val headers = parseTableCells(headerLine)
                val rows = mutableListOf<List<String>>()
                var j = i + 2
                while (j < lines.size && lines[j].contains("|") && !tableDelimiterRegex.matches(lines[j])) {
                    rows.add(parseTableCells(lines[j]))
                    j++
                }
                if (rows.isNotEmpty()) {
                    val start = lines.subList(0, i).joinToString("\n").length
                    val end = lines.subList(0, j).joinToString("\n").length
                    tableBlocks.add((start..end) to MarkdownBlock.Table(headers, rows))
                    i = j
                    continue
                }
            }
            i++
        }
        // Walk lines, replacing table ranges with the table block
        // and emitting non-table text as MarkdownBlock.Text.
        val tableRanges = tableBlocks.map { it.first }.sortedBy { it.first }
        var charPos = 0
        val out = mutableListOf<MarkdownBlock>()
        for ((range, table) in tableBlocks) {
            if (charPos < range.first) {
                out.add(MarkdownBlock.Text(s.substring(charPos, range.first).trim('\n')))
            }
            out.add(table)
            charPos = range.last + 1
        }
        if (charPos < s.length) {
            out.add(MarkdownBlock.Text(s.substring(charPos).trim('\n')))
        }
        blocks.addAll(out.filter { it !is MarkdownBlock.Text || it.content.isNotBlank() })
    }

    codeRanges.forEach { range ->
        if (range.first > pos) {
            appendText(text.substring(pos, range.first))
        }
        val match = codeBlockRegex.find(text, range.first) ?: return@forEach
        blocks.add(MarkdownBlock.Code(match.groupValues[1], match.groupValues[2]))
        pos = range.last + 1
    }
    if (pos < text.length) {
        appendText(text.substring(pos))
    }
    return blocks
}

private fun parseTableCells(line: String): List<String> {
    // Trim leading/trailing pipes then split on pipes. Trim each cell.
    val trimmed = line.trim().trim('|').trim()
    return trimmed.split("|").map { it.trim() }
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

@Composable
private fun TableBlock(headers: List<String>, rows: List<List<String>>) {
    val borderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
    val headerBg = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, borderColor, RoundedCornerShape(8.dp)),
    ) {
        // Header row
        Row(modifier = Modifier.background(headerBg)) {
            for (h in headers) {
                Text(
                    text = h,
                    modifier = Modifier
                        .weight(1f)
                        .padding(8.dp),
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
        // Body rows
        for ((idx, row) in rows.withIndex()) {
            val rowBg = if (idx % 2 == 1)
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.2f)
            else Color.Transparent
            Row(modifier = Modifier.background(rowBg)) {
                for (cell in row) {
                    Text(
                        text = cell,
                        modifier = Modifier
                            .weight(1f)
                            .padding(8.dp),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}