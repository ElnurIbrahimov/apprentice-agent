package com.aura.tools

import com.aura.agent.Tool
import com.aura.agent.ToolResult
import com.aura.agent.ToolRisk
import com.aura.data.UserPreferences
import com.aura.providers.ToolDefinition
import com.aura.providers.ToolParameters
import com.aura.providers.ToolProperty
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import java.util.Properties
import javax.inject.Inject
import javax.inject.Singleton
import javax.mail.Authenticator
import javax.mail.Message
import javax.mail.PasswordAuthentication
import javax.mail.Session
import javax.mail.Transport
import javax.mail.internet.InternetAddress
import javax.mail.internet.MimeMessage

/**
 * Send an email in the background via user-configured SMTP. Reads
 * credentials from DataStore; fails fast if SMTP is not configured.
 * Risk: WRITE_REMOTE.
 */
@Singleton
class SendEmailBackgroundTool @Inject constructor(
    private val userPreferences: UserPreferences,
) {
    fun definition() = ToolDefinition(
        name = "send_email_background",
        description = "Send an email in the background using the SMTP server configured in Settings. Requires host, port, username, password, and from address.",
        parameters = ToolParameters(
            properties = mapOf(
                "to" to ToolProperty(type = "string", description = "Recipient email address"),
                "subject" to ToolProperty(type = "string", description = "Email subject"),
                "body" to ToolProperty(type = "string", description = "Plain-text body"),
                "cc" to ToolProperty(type = "string", description = "Optional comma-separated CC addresses"),
                "bcc" to ToolProperty(type = "string", description = "Optional comma-separated BCC addresses"),
            ),
            required = listOf("to", "subject", "body"),
        ),
    )

    val tool = Tool(
        name = "send_email_background",
        description = definition().description,
        risk = ToolRisk.WRITE_REMOTE,
        parameters = definition().parameters,
        execute = { call, _ ->
            val to = call.arguments["to"] as? String
                ?: return@Tool ToolResult.Error("missing 'to'", "bad_args")
            val subject = call.arguments["subject"] as? String
                ?: return@Tool ToolResult.Error("missing 'subject'", "bad_args")
            val body = call.arguments["body"] as? String
                ?: return@Tool ToolResult.Error("missing 'body'", "bad_args")

            try {
                val host = userPreferences.smtpHost.first().trim()
                val port = userPreferences.smtpPort.first()
                val username = userPreferences.smtpUsername.first().trim()
                val password = userPreferences.smtpPassword.first()
                val from = userPreferences.smtpFrom.first().trim().ifBlank { username }

                if (host.isBlank() || username.isBlank() || password.isBlank() || from.isBlank()) {
                    return@Tool ToolResult.Error(
                        "SMTP not configured. Go to Settings → SMTP to set host, port, username, password, and from address.",
                        "missing_config",
                    )
                }

                val result = send(host, port, username, password, from, to, subject, body)
                if (result.isSuccess) {
                    ToolResult.Ok("Email sent to $to (subject: $subject)")
                } else {
                    ToolResult.Error("SMTP error: ${result.exceptionOrNull()?.message}", "smtp_error")
                }
            } catch (e: Exception) {
                ToolResult.Error("Email send failed: ${e.message}", "exception")
            }
        },
        category = "communication",
    )

    private suspend fun send(
        host: String,
        port: Int,
        username: String,
        password: String,
        from: String,
        to: String,
        subject: String,
        body: String,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val props = Properties().apply {
                put("mail.smtp.auth", "true")
                put("mail.smtp.starttls.enable", "true")
                put("mail.smtp.host", host)
                put("mail.smtp.port", port.toString())
                put("mail.smtp.connectiontimeout", "15000")
                put("mail.smtp.timeout", "15000")
            }
            val auth = object : Authenticator() {
                override fun getPasswordAuthentication(): PasswordAuthentication {
                    return PasswordAuthentication(username, password)
                }
            }
            val session = Session.getInstance(props, auth)
            val message = MimeMessage(session).apply {
                setFrom(InternetAddress(from))
                setRecipients(Message.RecipientType.TO, InternetAddress.parse(to))
                setSubject(subject)
                setText(body)
            }
            Transport.send(message)
        }
    }
}
