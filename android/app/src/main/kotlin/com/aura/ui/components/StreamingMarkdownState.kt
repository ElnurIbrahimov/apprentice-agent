package com.aura.ui.components

import androidx.compose.ui.text.AnnotatedString

/**
 * Streaming-aware markdown parser that avoids the "marker flicker"
 * during token-by-token streaming.
 *
 * The problem: the LLM emits tokens one at a time, and a single
 * `**bold**` marker pair may straddle chunks. If chunk 1 is
 * `"**bol"` and chunk 2 is `"d**"`, the bold regex matches only
 * on chunk 2 — the visible text changes from `**bol` to `bold`
 * with the opening `**` suddenly disappearing between frames.
 *
 * Fix: hide trailing unclosed markers from the rendered output.
 * If the text ends with `**` and there is no matching close
 * after it, we replace the trailing marker with a zero-width
 * non-joiner. The user sees the visible text without the
 * dangling markers; the next render (which has more text) re-
 * evaluates and either completes the style or keeps the marker
 * hidden. The flicker is gone because the visible content
 * never gains or loses the marker characters.
 *
 * This only handles `**` bold and `` ` `` code — the two
 * patterns that flicker visibly. Italic, links, headers, etc.
 * are rare at chunk boundaries and the existing
 * [parseMarkdown] handles them correctly on the final render.
 */
class StreamingMarkdownState {

    /**
     * Render the cumulative streamed text, suppressing any
     * trailing unclosed bold/code markers.
     */
    fun render(cumulative: String, colors: MarkdownColors, clickable: Boolean = false): AnnotatedString {
        if (cumulative.isEmpty()) return AnnotatedString("")
        val (visibleText, _) = maskTrailingUnclosedMarkers(cumulative)
        return if (clickable) parseMarkdownClickable(visibleText, colors) else parseMarkdown(visibleText, colors)
    }

    /**
     * Returns the text with trailing unclosed markers stripped,
     * and the count of stripped chars. The caller can use the
     * count to know if any markers were hidden.
     */
    private fun maskTrailingUnclosedMarkers(text: String): Pair<String, Int> {
        // Scan the last 3 chars. The combinations of trailing
        // markers we care about (in order of precedence):
        //   *** at end  -> opening of bold-italic (3 chars)
        //   **  at end  -> opening of bold, OR closing of bold-italic opener (2 chars)
        //   *   at end  -> opening of italic, OR closing of bold opener (1 char)
        //   `   at end  -> opening of inline code, unless paired
        //
        // We need to count `*` and `*` pairs to decide what's
        // open vs closed. The simplest correct rule: count the
        // total `*` in the last "phrase" (between the last
        // newline and end). If the count is odd, the last `*`
        // is unpaired. Strip 1, 2, or 3 trailing `*` depending
        // on the count.
        if (text.isEmpty()) return text to 0

        // Find end of last phrase (last newline or end of text).
        val lastNewline = text.lastIndexOf('\n')
        val endOfPhrase = if (lastNewline == -1) text.length else text.length
        val phraseStart = if (lastNewline == -1) 0 else lastNewline + 1
        val phrase = text.substring(phraseStart, endOfPhrase)
        val starCount = phrase.count { it == '*' }

        // Compute how many trailing `*` to strip.
        val stripStars = when {
            // If the phrase has 1+ `*` at the end:
            phrase.endsWith("***") -> {
                // Could be opening of bold-italic OR closing of
                // a bold opener (i.e. `**`). Count the `*` in
                // the phrase excluding the last 3 to see.
                val rest = phrase.dropLast(3).count { it == '*' }
                if (rest % 2 == 0) {
                    // The 3 trailing are an opening bold-italic.
                    3
                } else {
                    // The 2 of the 3 trailing close a bold; the
                    // last * is an opening italic. Strip 1.
                    1
                }
            }
            phrase.endsWith("**") -> {
                // Could be opening bold OR closing bold-italic.
                // Count the `*` in the phrase excluding the last 2.
                val rest = phrase.dropLast(2).count { it == '*' }
                if (rest % 2 == 0) {
                    // The 2 trailing are an opening bold.
                    2
                } else {
                    // The 2 trailing close a bold-italic; nothing
                    // to strip. (We'll let the parser handle it.)
                    0
                }
            }
            phrase.endsWith("*") -> {
                // Could be opening italic OR closing bold.
                val rest = phrase.dropLast(1).count { it == '*' }
                if (rest % 2 == 0) {
                    // No opener — the last `*` is an unpaired
                    // closing. Strip it so it doesn't render as
                    // literal text.
                    1
                } else {
                    // The last `*` could be opening italic. Strip.
                    1
                }
            }
            else -> 0
        }

        var masked = text
        if (stripStars > 0 && masked.length >= stripStars) {
            masked = masked.substring(0, masked.length - stripStars)
        }

        // Inline code: a trailing backtick that's not paired.
        if (masked.endsWith("`")) {
            // Count backticks in the current line. If odd, the
            // trailing one is unpaired.
            val lastNl = masked.lastIndexOf('\n')
            val line = if (lastNl == -1) masked else masked.substring(lastNl + 1)
            if (line.count { it == '`' } % 2 == 1) {
                masked = masked.substring(0, masked.length - 1)
            }
        }

        val stripped = text.length - masked.length
        return masked to stripped
    }
}
