package com.aura.documents

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.text.Html
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream
import javax.inject.Inject
import javax.inject.Singleton

data class ExtractedDocument(
    val id: String,
    val name: String,
    val mimeType: String,
    val sourceUri: String,
    val text: String,
)

@Singleton
class DocumentTextExtractor @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    init {
        PDFBoxResourceLoader.init(context.applicationContext)
    }

    suspend fun extract(uri: Uri): ExtractedDocument = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }?.take(240)?.ifBlank { null } ?: uri.lastPathSegment?.takeLast(120) ?: "Imported document"
        val mimeType = resolver.getType(uri).orEmpty().ifBlank { mimeFromName(name) }
        val bytes = resolver.openInputStream(uri)?.use(::readLimited)
            ?: error("Aura could not open this document.")
        val text = when {
            mimeType == "application/pdf" || name.endsWith(".pdf", true) -> extractPdf(bytes)
            mimeType == DOCX_MIME || name.endsWith(".docx", true) -> DocumentTextParsers.extractDocx(bytes)
            mimeType == "text/html" || name.endsWith(".html", true) || name.endsWith(".htm", true) ->
                Html.fromHtml(bytes.toString(Charsets.UTF_8), Html.FROM_HTML_MODE_LEGACY).toString()
            mimeType.startsWith("text/") || mimeType == "application/json" || isPlainTextName(name) ->
                bytes.toString(Charsets.UTF_8)
            else -> error("Unsupported document type. Use PDF, DOCX, TXT, MD, CSV, JSON, YAML, HTML, or source-code files.")
        }.replace('\u0000', ' ').trim()
        require(text.isNotBlank()) {
            "No readable text was found. Scanned PDFs need OCR, which is not enabled yet."
        }
        ExtractedDocument(
            id = sha256(bytes),
            name = name,
            mimeType = mimeType,
            sourceUri = uri.toString(),
            text = text,
        )
    }

    private fun extractPdf(bytes: ByteArray): String =
        PDDocument.load(bytes).use { document -> PDFTextStripper().getText(document) }

    private fun readLimited(input: java.io.InputStream): ByteArray {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(16 * 1024)
        var total = 0
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            total += read
            require(total <= MAX_FILE_BYTES) { "Document is larger than 20 MB." }
            output.write(buffer, 0, read)
        }
        return output.toByteArray()
    }

    private fun mimeFromName(name: String): String = when {
        name.endsWith(".pdf", true) -> "application/pdf"
        name.endsWith(".docx", true) -> DOCX_MIME
        name.endsWith(".json", true) -> "application/json"
        name.endsWith(".html", true) || name.endsWith(".htm", true) -> "text/html"
        else -> "text/plain"
    }

    private fun isPlainTextName(name: String): Boolean {
        val extension = name.substringAfterLast('.', "").lowercase()
        return extension in PLAIN_EXTENSIONS
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

    companion object {
        const val MAX_FILE_BYTES = 20 * 1024 * 1024
        const val DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        private val PLAIN_EXTENSIONS = setOf(
            "txt", "md", "markdown", "csv", "json", "yaml", "yml", "xml", "log",
            "kt", "kts", "java", "py", "js", "ts", "tsx", "jsx", "css", "sql", "sh",
        )
    }
}

internal object DocumentTextParsers {
    private val paragraphEnd = Regex("</w:p\\s*>", RegexOption.IGNORE_CASE)
    private val lineBreak = Regex("<w:(?:br|cr)(?:\\s[^>]*)?/?>(?:</w:(?:br|cr)>)?", RegexOption.IGNORE_CASE)
    private val tab = Regex("<w:tab(?:\\s[^>]*)?/?>(?:</w:tab>)?", RegexOption.IGNORE_CASE)
    private val xmlTag = Regex("<[^>]+>")
    private val numericEntity = Regex("&#(x?[0-9A-Fa-f]+);")

    fun extractDocx(bytes: ByteArray): String {
        ZipInputStream(ByteArrayInputStream(bytes)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (entry.name == "word/document.xml") {
                    val xml = zip.readBytes().toString(Charsets.UTF_8)
                        .replace(Regex(">\\s+<"), "><")
                    return decodeXmlEntities(
                        xml
                            .replace(paragraphEnd, "\n\n")
                            .replace(lineBreak, "\n")
                            .replace(tab, "\t")
                            .replace(xmlTag, ""),
                    ).replace(Regex("[ \\t]+\\n"), "\n")
                        .replace(Regex("\\n{3,}"), "\n\n")
                        .trim()
                }
            }
        }
        error("This DOCX file has no word/document.xml content.")
    }

    private fun decodeXmlEntities(value: String): String {
        val named = value
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&")
        return numericEntity.replace(named) { match ->
            val raw = match.groupValues[1]
            val codePoint = if (raw.startsWith('x', true)) {
                raw.drop(1).toIntOrNull(16)
            } else {
                raw.toIntOrNull()
            }
            codePoint?.let { String(Character.toChars(it)) } ?: match.value
        }
    }
}