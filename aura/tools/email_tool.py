"""Email tool for reading and sending emails via IMAP/SMTP."""

import email
import email.mime.text
import email.mime.multipart
import hashlib
import imaplib
import json
import logging
import os
import re
import smtplib
import ssl
from dataclasses import dataclass, field, asdict
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

EMAIL_CONFIG_FILE = Path(__file__).parent.parent.parent / "data" / "email_config.json"


def _derive_encryption_key() -> tuple:
    """Return (key_bytes, error_msg). key_bytes is None on error."""
    import base64
    secret = os.environ.get("AURA_EMAIL_KEY")
    if not secret:
        return None, (
            "AURA_EMAIL_KEY environment variable is not set. "
            "Set it to a strong random secret before storing email credentials."
        )
    key_bytes = hashlib.pbkdf2_hmac("sha256", secret.encode(), b"aura-email-v1", 100_000)
    return base64.urlsafe_b64encode(key_bytes), None


def _encrypt(plaintext: str) -> tuple:
    """Return (ciphertext, error_msg). ciphertext is None on error."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None, "cryptography package not installed. Run: pip install cryptography"
    key, err = _derive_encryption_key()
    if err:
        return None, err
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8"), None


def _decrypt(ciphertext: str) -> tuple:
    """Return (plaintext, error_msg). plaintext is None on error."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None, "cryptography package not installed. Run: pip install cryptography"
    key, err = _derive_encryption_key()
    if err:
        return None, err
    try:
        f = Fernet(key)
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8"), None
    except Exception as e:
        logger.warning(f"[Email] Decryption failed (check AURA_EMAIL_KEY): {e}")
        return None, f"Decryption failed — check AURA_EMAIL_KEY: {e}"


def _sanitize_imap_string(value: str) -> str:
    """Strip characters that could break IMAP search string grammar."""
    return value.replace('"', '').replace('\\', '').replace('\r', '').replace('\n', '').strip()


def _sanitize_error(error: Exception) -> str:
    """Strip credentials from error messages."""
    msg = str(error)
    # Remove anything that looks like a password or token
    msg = re.sub(r'(?i)(password|passwd|pass|token|secret|key)\s*[=:]\s*\S+', r'\1=***', msg)
    msg = re.sub(r'b\'[^\']{20,}\'', "b'***'", msg)
    return msg


@dataclass
class EmailMessage:
    """Represents an email message."""
    id: str
    subject: str
    sender: str
    to: List[str] = field(default_factory=list)
    date: str = ""
    body_text: str = ""
    body_html: Optional[str] = None
    attachments: List[dict] = field(default_factory=list)
    is_read: bool = True
    folder: str = "INBOX"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmailTool:
    """Read and send emails via IMAP/SMTP."""

    name = "email"
    description = "Read and send emails via IMAP/SMTP"

    def __init__(self):
        self._config: Optional[dict] = None
        self._load_config()

    def _load_config(self):
        """Load email configuration from encrypted file."""
        if EMAIL_CONFIG_FILE.exists():
            try:
                with open(EMAIL_CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # Decrypt password
                if config.get("app_password"):
                    plaintext, err = _decrypt(config["app_password"])
                    if err:
                        logger.warning(f"[Email] Decryption error: {err}")
                        self._config = None
                        return
                    config["app_password"] = plaintext
                self._config = config
            except Exception as e:
                logger.warning(f"[Email] Config load error: {_sanitize_error(e)}")
                self._config = None

    def _save_config(self, config: dict) -> dict | None:
        """Save configuration with encrypted password. Returns error dict on failure, None on success."""
        EMAIL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        save_config = dict(config)
        if save_config.get("app_password"):
            encrypted, err = _encrypt(save_config["app_password"])
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            save_config["app_password"] = encrypted
        try:
            with open(EMAIL_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(save_config, f, indent=4)
            self._config = config
            return None
        except IOError as e:
            logger.error(f"[Email] Config save error: {e}")
            return {"success": False, "error": f"Config save error: {e}"}

    def _sanitize_header(self, value: str) -> str:
        """Remove characters that could inject MIME headers."""
        return value.replace('\r', '').replace('\n', '').replace('\0', '')

    def _is_configured(self) -> bool:
        return (self._config is not None
                and self._config.get("email")
                and self._config.get("app_password"))

    def get_config_status(self) -> dict:
        """Check if email is configured."""
        if self._is_configured():
            return {
                "success": True,
                "configured": True,
                "email": self._config.get("email", ""),
                "imap_server": self._config.get("imap_server", ""),
                "smtp_server": self._config.get("smtp_server", ""),
                "response": f"Email configured: {self._config['email']}"
            }
        return {
            "success": True,
            "configured": False,
            "response": "Email not configured. Use 'setup' action to configure."
        }

    def setup(self, email_addr: str = None, app_password: str = None,
              imap_server: str = None, smtp_server: str = None,
              display_name: str = None) -> dict:
        """Configure email settings."""
        if not email_addr or not app_password:
            return {
                "success": False,
                "error": "Email address and app password are required.\n"
                         "For Gmail: use an App Password (not your regular password).\n"
                         "Usage: setup email:<your@email.com> password:<app_password>"
            }

        # Auto-detect servers from email domain
        domain = email_addr.split("@")[-1].lower()
        server_map = {
            "gmail.com": ("imap.gmail.com", "smtp.gmail.com"),
            "googlemail.com": ("imap.gmail.com", "smtp.gmail.com"),
            "outlook.com": ("outlook.office365.com", "smtp.office365.com"),
            "hotmail.com": ("outlook.office365.com", "smtp.office365.com"),
            "yahoo.com": ("imap.mail.yahoo.com", "smtp.mail.yahoo.com"),
            "icloud.com": ("imap.mail.me.com", "smtp.mail.me.com"),
        }

        detected_imap, detected_smtp = server_map.get(domain, (f"imap.{domain}", f"smtp.{domain}"))

        config = {
            "imap_server": imap_server or detected_imap,
            "imap_port": 993,
            "smtp_server": smtp_server or detected_smtp,
            "smtp_port": 587,
            "email": email_addr,
            "app_password": app_password,
            "display_name": display_name or email_addr.split("@")[0],
        }

        # Test connection
        try:
            ctx = ssl.create_default_context()
            mail = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"], ssl_context=ctx)
            mail.login(config["email"], config["app_password"])
            mail.logout()
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection test failed: {_sanitize_error(e)}. Check credentials and server settings."
            }

        save_err = self._save_config(config)
        if save_err:
            return save_err
        return {
            "success": True,
            "email": config["email"],
            "imap_server": config["imap_server"],
            "smtp_server": config["smtp_server"],
            "response": f"Email configured successfully for {config['email']}"
        }

    def _connect_imap(self) -> tuple:
        """Return (mail_connection, error_msg). mail is None on error."""
        if not self._is_configured():
            return None, "Email not configured. Use 'setup' first."

        try:
            ctx = ssl.create_default_context()
            mail = imaplib.IMAP4_SSL(
                self._config["imap_server"],
                self._config.get("imap_port", 993),
                ssl_context=ctx
            )
            mail.login(self._config["email"], self._config["app_password"])
            return mail, None
        except Exception as e:
            return None, f"IMAP connection failed: {_sanitize_error(e)}"

    def _decode_header_value(self, value: str) -> str:
        """Decode an email header value."""
        if not value:
            return ""
        decoded_parts = decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                result.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                result.append(str(part))
        return " ".join(result)

    def _parse_email(self, raw_data: bytes, msg_id: str, folder: str = "INBOX") -> EmailMessage:
        """Parse raw email data into EmailMessage."""
        msg = email.message_from_bytes(raw_data)

        subject = self._decode_header_value(msg.get("Subject", ""))
        sender = self._decode_header_value(msg.get("From", ""))
        to_addrs = [self._decode_header_value(a) for a in (msg.get("To", "").split(","))]
        date = msg.get("Date", "")

        body_text = ""
        body_html = None
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        attachments.append({
                            "filename": self._decode_header_value(filename),
                            "size": len(part.get_payload(decode=True) or b""),
                            "content_type": content_type,
                        })
                elif content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text = payload.decode("utf-8", errors="replace")
                elif content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_html = payload.decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                if msg.get_content_type() == "text/html":
                    body_html = payload.decode("utf-8", errors="replace")
                else:
                    body_text = payload.decode("utf-8", errors="replace")

        return EmailMessage(
            id=str(msg_id),
            subject=subject,
            sender=sender,
            to=to_addrs,
            date=date,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            folder=folder,
        )

    def fetch_emails(self, folder: str = "INBOX", limit: int = 10,
                     unread_only: bool = False, since_date: str = None) -> dict:
        """Fetch emails from a folder."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        try:
            mail, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            folder = _sanitize_imap_string(folder)
            mail.select(folder, readonly=True)

            # Build search criteria
            criteria = []
            if unread_only:
                criteria.append("UNSEEN")
            if since_date:
                safe_date = _sanitize_imap_string(since_date)
                criteria.append(f'SINCE "{safe_date}"')
            if not criteria:
                criteria.append("ALL")

            search_str = " ".join(criteria)
            status, messages = mail.search(None, search_str)

            if status != "OK":
                mail.logout()
                return {"success": False, "error": "IMAP search failed"}

            msg_ids = messages[0].split()
            # Get most recent N
            msg_ids = msg_ids[-limit:]
            msg_ids.reverse()  # newest first

            emails = []
            for msg_id in msg_ids:
                status, data = mail.fetch(msg_id, "(RFC822)")
                if status == "OK" and data[0]:
                    parsed = self._parse_email(data[0][1], msg_id.decode(), folder)
                    emails.append({
                        "id": parsed.id,
                        "subject": parsed.subject,
                        "sender": parsed.sender,
                        "date": parsed.date,
                        "has_attachments": len(parsed.attachments) > 0,
                        "preview": parsed.body_text[:100],
                    })

            mail.logout()

            formatted = []
            for em in emails:
                attach_icon = " [+]" if em["has_attachments"] else ""
                formatted.append(f"[{em['id']}] {em['sender']}: {em['subject']}{attach_icon}")

            return {
                "success": True,
                "count": len(emails),
                "emails": emails,
                "folder": folder,
                "formatted": "\n".join(formatted) if formatted else "No emails found",
                "response": f"Found {len(emails)} email(s) in {folder}\n" + "\n".join(formatted)
            }

        except Exception as e:
            return {"success": False, "error": f"Fetch failed: {_sanitize_error(e)}"}

    def read_email(self, email_id: str) -> dict:
        """Read full email content."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        try:
            mail, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            mail.select("INBOX")

            status, data = mail.fetch(email_id.encode(), "(RFC822)")
            if status != "OK" or not data[0]:
                mail.logout()
                return {"success": False, "error": f"Email not found: {email_id}"}

            parsed = self._parse_email(data[0][1], email_id)
            mail.logout()

            return {
                "success": True,
                "email": parsed.to_dict(),
                "response": f"From: {parsed.sender}\nSubject: {parsed.subject}\nDate: {parsed.date}\n\n{parsed.body_text[:2000]}"
            }

        except Exception as e:
            return {"success": False, "error": f"Read failed: {_sanitize_error(e)}"}

    def send_email(self, to: str, subject: str, body: str,
                   cc: str = None, bcc: str = None) -> dict:
        """Send an email via SMTP."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        if not to or not subject or not body:
            return {"success": False, "error": "Missing required fields: to, subject, body"}

        try:
            to = self._sanitize_header(to)
            subject = self._sanitize_header(subject)
            if cc:
                cc = self._sanitize_header(cc)
            if bcc:
                bcc = self._sanitize_header(bcc)

            msg = email.mime.multipart.MIMEMultipart()
            msg["From"] = f"{self._config.get('display_name', '')} <{self._config['email']}>"
            msg["To"] = to
            msg["Subject"] = subject
            if cc:
                msg["Cc"] = cc
            if bcc:
                msg["Bcc"] = bcc

            msg.attach(email.mime.text.MIMEText(body, "plain"))

            # Build recipient list
            recipients = [to]
            if cc:
                recipients.extend([a.strip() for a in cc.split(",")])
            if bcc:
                recipients.extend([a.strip() for a in bcc.split(",")])

            ctx = ssl.create_default_context()
            with smtplib.SMTP(self._config["smtp_server"], self._config.get("smtp_port", 587)) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.ehlo()
                server.login(self._config["email"], self._config["app_password"])
                server.sendmail(self._config["email"], recipients, msg.as_string())

            return {
                "success": True,
                "to": to,
                "subject": subject,
                "response": f"Email sent to {to}: {subject}"
            }

        except Exception as e:
            return {"success": False, "error": f"Send failed: {_sanitize_error(e)}"}

    def reply(self, email_id: str, body: str) -> dict:
        """Reply to an email."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        # First fetch the original email
        original = self.read_email(email_id)
        if not original.get("success"):
            return original

        orig_email = original["email"]
        to = orig_email.get("sender", "")

        # Validate the extracted sender address
        # Strip display name if present: "Name <addr>" -> "addr"
        addr_match = re.search(r'<([^>]+)>', to)
        to_addr = addr_match.group(1).strip() if addr_match else to.strip()
        if not re.match(r'^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$', to_addr):
            return {"success": False, "error": f"Invalid reply-to address extracted from sender: {to!r}"}

        subject = orig_email.get("subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        reply_body = f"{body}\n\n---\nOn {orig_email.get('date', '')}, {to} wrote:\n{orig_email.get('body_text', '')[:500]}"

        return self.send_email(to=to_addr, subject=subject, body=reply_body)

    def search_emails(self, query: str, folder: str = "INBOX") -> dict:
        """Search emails using IMAP search."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        try:
            mail, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            mail.select(folder, readonly=True)

            # IMAP search by subject or body
            safe_query = _sanitize_imap_string(query)
            status, messages = mail.search(None, f'(OR SUBJECT "{safe_query}" BODY "{safe_query}")')

            if status != "OK":
                mail.logout()
                return {"success": False, "error": "Search failed"}

            msg_ids = messages[0].split()[-20:]  # Last 20 matches
            msg_ids.reverse()

            results = []
            for msg_id in msg_ids:
                status, data = mail.fetch(msg_id, "(RFC822)")
                if status == "OK" and data[0]:
                    parsed = self._parse_email(data[0][1], msg_id.decode(), folder)
                    results.append({
                        "id": parsed.id,
                        "subject": parsed.subject,
                        "sender": parsed.sender,
                        "date": parsed.date,
                    })

            mail.logout()

            return {
                "success": True,
                "count": len(results),
                "results": results,
                "query": query,
                "response": f"Found {len(results)} email(s) matching '{query}'"
            }

        except Exception as e:
            return {"success": False, "error": f"Search failed: {_sanitize_error(e)}"}

    def list_folders(self) -> dict:
        """List mailbox folders."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        try:
            mail, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            status, folders = mail.list()
            mail.logout()

            if status != "OK":
                return {"success": False, "error": "Could not list folders"}

            folder_names = []
            for f in folders:
                # Parse IMAP folder response
                if isinstance(f, bytes):
                    parts = f.decode().split('"')
                    if len(parts) >= 3:
                        folder_names.append(parts[-2])
                    else:
                        folder_names.append(f.decode().split()[-1])

            return {
                "success": True,
                "folders": folder_names,
                "count": len(folder_names),
                "response": f"Mailbox folders:\n" + "\n".join(folder_names)
            }

        except Exception as e:
            return {"success": False, "error": f"List folders failed: {_sanitize_error(e)}"}

    def mark_read(self, email_id: str) -> dict:
        """Mark an email as read."""
        return self._set_flag(email_id, "\\Seen", add=True)

    def mark_unread(self, email_id: str) -> dict:
        """Mark an email as unread."""
        return self._set_flag(email_id, "\\Seen", add=False)

    def _set_flag(self, email_id: str, flag: str, add: bool = True) -> dict:
        """Set or remove an IMAP flag on an email."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured."}

        try:
            mail, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            mail.select("INBOX")
            action = "+FLAGS" if add else "-FLAGS"
            status, _ = mail.store(email_id.encode(), action, flag)
            mail.logout()

            if status == "OK":
                label = "read" if add else "unread"
                return {"success": True, "response": f"Email {email_id} marked as {label}"}
            return {"success": False, "error": "Flag update failed"}

        except Exception as e:
            return {"success": False, "error": f"Flag failed: {_sanitize_error(e)}"}

    def execute(self, action: str, **kwargs) -> dict:
        """Execute an email action."""
        action_lower = action.lower().strip()

        # Setup
        if action_lower == "setup" or action_lower == "configure":
            return self.setup(
                email_addr=kwargs.get("email"),
                app_password=kwargs.get("password") or kwargs.get("app_password"),
                imap_server=kwargs.get("imap_server"),
                smtp_server=kwargs.get("smtp_server"),
                display_name=kwargs.get("display_name"),
            )

        # Status
        if action_lower in ("status", "config", "config_status"):
            return self.get_config_status()

        # List folders
        if action_lower in ("folders", "list_folders"):
            return self.list_folders()

        # Fetch emails
        if action_lower.startswith("fetch") or action_lower in ("inbox", "check", "check mail"):
            unread_only = "unread" in action_lower
            limit = kwargs.get("limit", 10)
            folder = kwargs.get("folder", "INBOX")
            return self.fetch_emails(folder=folder, limit=limit, unread_only=unread_only)

        # Read specific email
        if action_lower.startswith("read"):
            email_id = kwargs.get("email_id")
            if not email_id:
                parts = action.split()
                email_id = parts[-1] if len(parts) > 1 else None
            if email_id:
                return self.read_email(email_id)
            return {"success": False, "error": "No email ID specified"}

        # Send email
        if action_lower.startswith("send"):
            to = kwargs.get("to")
            subject = kwargs.get("subject")
            body = kwargs.get("body")

            if not all([to, subject, body]):
                # Try parsing from action: "send to:<addr> subject:<subj> body:<text>"
                to_match = re.search(r'to:\s*(\S+)', action, re.IGNORECASE)
                subj_match = re.search(r'subject:\s*(.+?)(?:\s+body:|\s*$)', action, re.IGNORECASE)
                body_match = re.search(r'body:\s*(.+)', action, re.IGNORECASE)

                to = to or (to_match.group(1) if to_match else None)
                subject = subject or (subj_match.group(1).strip() if subj_match else None)
                body = body or (body_match.group(1).strip() if body_match else None)

            if to and subject and body:
                return self.send_email(to=to, subject=subject, body=body,
                                       cc=kwargs.get("cc"), bcc=kwargs.get("bcc"))
            return {"success": False, "error": "Usage: send to:<addr> subject:<subj> body:<text>"}

        # Reply
        if action_lower.startswith("reply"):
            email_id = kwargs.get("email_id")
            body = kwargs.get("body")
            if not email_id:
                parts = action.split()
                email_id = parts[1] if len(parts) > 1 else None
            if email_id and body:
                return self.reply(email_id, body)
            return {"success": False, "error": "Usage: reply <email_id> body:<text>"}

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            folder = kwargs.get("folder", "INBOX")
            return self.search_emails(query, folder=folder)

        # Mark read/unread
        if "mark" in action_lower:
            email_id = kwargs.get("email_id")
            if not email_id:
                id_match = re.search(r'\b(\d+)\b', action)
                email_id = id_match.group(1) if id_match else None
            if email_id:
                if "unread" in action_lower:
                    return self.mark_unread(email_id)
                return self.mark_read(email_id)
            return {"success": False, "error": "No email ID specified"}

        return {
            "success": False,
            "error": f"Unknown email action: {action}. "
                     "Try: 'fetch', 'fetch unread', 'read <id>', 'send to:<addr> subject:<subj> body:<text>', 'search <query>', 'setup'"
        }


# Singleton
email_tool = EmailTool()
