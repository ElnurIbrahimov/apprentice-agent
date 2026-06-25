package com.aura.tools

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.ContactsContract
import androidx.core.content.ContextCompat
import com.aura.agent.Tool
import com.aura.agent.ToolContext
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Search contacts by name. Mirrors aura/tools/contacts_search.py (none — new).
 * Risk: PRIVACY (READ_CONTACTS).
 */
@Singleton
class ContactsSearchTool @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    fun definition() = ToolDefinition(
        name = "contacts_search",
        description = "Search contacts by name. Returns name, phone, and email for up to 10 matches.",
        parameters = ToolParameters(
            properties = mapOf(
                "query" to ToolProperty(type = "string", description = "Name to search for (case-insensitive, partial match)"),
            ),
            required = listOf("query"),
        ),
    )

    val tool = Tool(
        name = "contacts_search",
        description = definition().description,
        risk = ToolRisk.PRIVACY,
        requiredPermissions = listOf(Manifest.permission.READ_CONTACTS),
        parameters = definition().parameters,
        execute = { call, ctx ->
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) != PackageManager.PERMISSION_GRANTED) {
                return@Tool ToolResult.NeedsPermission(Manifest.permission.READ_CONTACTS, "Contacts access required.")
            }
            val query = call.arguments["query"] as? String ?: return@Tool ToolResult.Error("missing 'query'", "bad_args")
            try {
                val results = search(query)
                if (results.isEmpty()) ToolResult.Ok("No contacts found matching: $query")
                else ToolResult.Ok(formatResults(results))
            } catch (e: SecurityException) {
                ToolResult.NeedsPermission(Manifest.permission.READ_CONTACTS, "Contacts permission revoked.")
            } catch (e: Exception) {
                ToolResult.Error("contacts search failed: ${e.message}", "exception")
            }
        },
    )

    private data class Contact(val name: String, val phone: String?, val email: String?)

    private fun search(query: String): List<Contact> {
        val out = mutableListOf<Contact>()
        val selection = "${ContactsContract.Contacts.DISPLAY_NAME} LIKE ?"
        val args = arrayOf("%$query%")
        context.contentResolver.query(
            ContactsContract.Contacts.CONTENT_URI,
            arrayOf(ContactsContract.Contacts._ID, ContactsContract.Contacts.DISPLAY_NAME, ContactsContract.Contacts.HAS_PHONE_NUMBER),
            selection, args, "${ContactsContract.Contacts.DISPLAY_NAME} ASC LIMIT 10"
        )?.use { c ->
            while (c.moveToNext()) {
                val id = c.getLong(0)
                val name = c.getString(1) ?: continue
                val hasPhone = c.getInt(2) == 1
                val phone = if (hasPhone) primaryPhone(id) else null
                val email = primaryEmail(id)
                out += Contact(name, phone, email)
            }
        }
        return out
    }

    private fun primaryPhone(contactId: Long): String? {
        context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            arrayOf(ContactsContract.CommonDataKinds.Phone.NUMBER),
            "${ContactsContract.CommonDataKinds.Phone.CONTACT_ID} = ?",
            arrayOf(contactId.toString()),
            null
        )?.use { c -> if (c.moveToFirst()) return c.getString(0) }
        return null
    }

    private fun primaryEmail(contactId: Long): String? {
        context.contentResolver.query(
            ContactsContract.CommonDataKinds.Email.CONTENT_URI,
            arrayOf(ContactsContract.CommonDataKinds.Email.ADDRESS),
            "${ContactsContract.CommonDataKinds.Email.CONTACT_ID} = ?",
            arrayOf(contactId.toString()),
            null
        )?.use { c -> if (c.moveToFirst()) return c.getString(0) }
        return null
    }

    private fun formatResults(results: List<Contact>): String =
        results.mapIndexed { i, c ->
            val phone = c.phone?.let { " · $it" } ?: ""
            val email = c.email?.let { " · $it" } ?: ""
            "${i + 1}. ${c.name}$phone$email"
        }.joinToString("\n")
}
