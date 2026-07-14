package com.aura.documents

import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals

class DocumentTextExtractorTest {
    @Test
    fun `docx parser preserves paragraphs and decoded text`() {
        val xml = """
            <?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>Aura &amp; the glass city</w:t></w:r></w:p>
                <w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>
              </w:body>
            </w:document>
        """.trimIndent()
        val bytes = ByteArrayOutputStream().also { out ->
            ZipOutputStream(out).use { zip ->
                zip.putNextEntry(ZipEntry("word/document.xml"))
                zip.write(xml.toByteArray())
                zip.closeEntry()
            }
        }.toByteArray()

        val text = DocumentTextParsers.extractDocx(bytes)

        assertEquals("Aura & the glass city\n\nSecond paragraph.", text)
    }
}