"""Email tool for reading and sending emails via IMAP/SMTP or Gmail API.

Supports:
- IMAP/SMTP (default, with connection pooling)
- Gmail API (optional, graceful fallback)
- HTML→text conversion
- Attachment handling
- AI-powered: inbox summary, reply drafting, categorization, action item extraction
"""

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
import sqlite3
import ssl
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.header import decode_header
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

EMAIL_CONFIG_FILE = Path(__file__).parent.parent.parent / "data" / "email_config.json"

# ---------------------------------------------------------------------------
#  Optional imports — never break if missing
# ---------------------------------------------------------------------------
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GRequest
    from googleapiclient.discovery import build as google_build
    HAS_GMAIL_API = True
except ImportError:
    HAS_GMAIL_API = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html2text as _html2text_mod
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

# Gmail API config
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]
GMAIL_CREDS_PATH = Path.home() / ".aura" / "gmail_creds.json"
GMAIL_TOKEN_PATH = Path.home() / ".aura" / "gmail_token.json"


# ---------------------------------------------------------------------------
#  Encryption helpers (unchanged)
# ---------------------------------------------------------------------------
def _derive_encryption_key() -> tuple:
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
    return value.replace('"', '').replace('\\', '').replace('\r', '').replace('\n', '').strip()


def _sanitize_error(error: Exception) -> str:
    msg = str(error)
    msg = re.sub(r'(?i)(password|passwd|pass|token|secret|key)\s*[=:]\s*\S+', r'\1=***', msg)
    msg = re.sub(r'b\'[^\']{20,}\'', "b'***'", msg)
    return msg


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using the best available method."""
    if not html:
        return ""
    if HAS_HTML2TEXT:
        h = _html2text_mod.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        return h.handle(html)
    if HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    # Minimal fallback: strip tags
    clean = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'&nbsp;', ' ', clean)
    clean = re.sub(r'&amp;', '&', clean)
    clean = re.sub(r'&lt;', '<', clean)
    clean = re.sub(r'&gt;', '>', clean)
    return clean.strip()


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------
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
    labels: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
#  Gmail API backend
# ---------------------------------------------------------------------------
class GmailBackend:
    """Optional Gmail API integration."""

    def __init__(self):
        self._service = None

    @property
    def available(self) -> bool:
        return HAS_GMAIL_API and GMAIL_CREDS_PATH.exists()

    def _get_service(self):
        if self._service is not None:
            return self._service

        creds = None
        if GMAIL_TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(GMAIL_TOKEN_PATH), GMAIL_SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GRequest())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(GMAIL_CREDS_PATH), GMAIL_SCOPES)
                creds = flow.run_local_server(port=0)
            GMAIL_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(GMAIL_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())

        self._service = google_build("gmail", "v1", credentials=creds)
        return self._service

    def inbox(self, limit: int = 20, unread_only: bool = False,
              label: str = None) -> List[Dict[str, Any]]:
        """Fetch messages from Gmail inbox."""
        service = self._get_service()
        query = "in:inbox"
        if unread_only:
            query += " is:unread"
        if label:
            query += f" label:{label}"

        results = service.users().messages().list(
            userId="me", q=query, maxResults=limit
        ).execute()

        messages = []
        for msg_ref in results.get("messages", []):
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            ).execute()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            is_unread = "UNREAD" in msg.get("labelIds", [])

            messages.append({
                "id": msg["id"],
                "thread_id": msg.get("threadId", ""),
                "subject": headers.get("Subject", "(No subject)"),
                "sender": headers.get("From", ""),
                "to": headers.get("To", ""),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
                "is_read": not is_unread,
                "labels": msg.get("labelIds", []),
            })
        return messages

    def read(self, message_id: str) -> Dict[str, Any]:
        """Read full message content."""
        service = self._get_service()
        msg = service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()

        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body_text, body_html, attachments = self._extract_body(msg.get("payload", {}))

        # If only HTML, convert
        if not body_text and body_html:
            body_text = _html_to_text(body_html)

        return {
            "id": msg["id"],
            "thread_id": msg.get("threadId", ""),
            "subject": headers.get("Subject", ""),
            "sender": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "body_text": body_text,
            "body_html": body_html,
            "attachments": attachments,
            "labels": msg.get("labelIds", []),
        }

    def _extract_body(self, payload: dict) -> Tuple[str, str, List[dict]]:
        """Recursively extract body text, HTML, and attachments from payload."""
        import base64
        body_text = ""
        body_html = ""
        attachments = []

        mime_type = payload.get("mimeType", "")
        body_data = payload.get("body", {}).get("data", "")

        if mime_type == "text/plain" and body_data:
            body_text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        elif mime_type == "text/html" and body_data:
            body_html = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

        for part in payload.get("parts", []):
            filename = part.get("filename", "")
            if filename:
                attachments.append({
                    "filename": filename,
                    "mimeType": part.get("mimeType", ""),
                    "size": part.get("body", {}).get("size", 0),
                    "attachmentId": part.get("body", {}).get("attachmentId", ""),
                })
            else:
                sub_text, sub_html, sub_attach = self._extract_body(part)
                if sub_text and not body_text:
                    body_text = sub_text
                if sub_html and not body_html:
                    body_html = sub_html
                attachments.extend(sub_attach)

        return body_text, body_html, attachments

    def send(self, to: str, subject: str, body: str,
             cc: str = None, bcc: str = None) -> Dict[str, Any]:
        """Send email via Gmail API."""
        import base64

        service = self._get_service()
        msg = email.mime.multipart.MIMEMultipart()

        # Get user's email for From header
        profile = service.users().getProfile(userId="me").execute()
        msg["From"] = profile.get("emailAddress", "me")
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc
        msg.attach(email.mime.text.MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        sent = service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()

        return {"id": sent["id"], "thread_id": sent.get("threadId", "")}

    def reply(self, message_id: str, body: str) -> Dict[str, Any]:
        """Reply to a message."""
        import base64

        service = self._get_service()
        original = service.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["From", "To", "Subject", "Message-ID"],
        ).execute()

        headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}
        thread_id = original.get("threadId", "")
        to = headers.get("From", "")
        subject = headers.get("Subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        msg = email.mime.multipart.MIMEMultipart()
        profile = service.users().getProfile(userId="me").execute()
        msg["From"] = profile.get("emailAddress", "me")
        msg["To"] = to
        msg["Subject"] = subject
        msg["In-Reply-To"] = headers.get("Message-ID", "")
        msg["References"] = headers.get("Message-ID", "")
        msg.attach(email.mime.text.MIMEText(body, "plain"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        sent = service.users().messages().send(
            userId="me", body={"raw": raw, "threadId": thread_id}
        ).execute()

        return {"id": sent["id"], "thread_id": sent.get("threadId", "")}

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search Gmail messages."""
        service = self._get_service()
        results = service.users().messages().list(
            userId="me", q=query, maxResults=limit
        ).execute()

        messages = []
        for msg_ref in results.get("messages", []):
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            messages.append({
                "id": msg["id"],
                "subject": headers.get("Subject", ""),
                "sender": headers.get("From", ""),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
            })
        return messages

    def get_labels(self) -> List[Dict[str, str]]:
        """Get Gmail labels."""
        service = self._get_service()
        results = service.users().labels().list(userId="me").execute()
        return [{"id": l["id"], "name": l["name"]} for l in results.get("labels", [])]

    def download_attachment(self, message_id: str, attachment_id: str) -> bytes:
        """Download an attachment by ID."""
        import base64
        service = self._get_service()
        attachment = service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id
        ).execute()
        return base64.urlsafe_b64decode(attachment["data"])


# ---------------------------------------------------------------------------
#  IMAP connection pool
# ---------------------------------------------------------------------------
class IMAPConnectionPool:
    """Reusable IMAP connections to avoid re-auth on every operation."""

    def __init__(self, max_connections: int = 3, idle_timeout: int = 300):
        self._pool: List[Tuple[imaplib.IMAP4_SSL, float]] = []
        self._lock = threading.Lock()
        self._max = max_connections
        self._idle_timeout = idle_timeout
        self._config: Optional[dict] = None

    def configure(self, config: dict):
        self._config = config

    def get(self) -> Tuple[Optional[imaplib.IMAP4_SSL], Optional[str]]:
        """Get a connection from the pool or create a new one."""
        if not self._config:
            return None, "Email not configured."

        now = datetime.now().timestamp()

        with self._lock:
            # Try to reuse an existing connection
            while self._pool:
                conn, created_at = self._pool.pop()
                if now - created_at > self._idle_timeout:
                    try:
                        conn.logout()
                    except Exception:
                        pass
                    continue
                # Test if connection is still alive
                try:
                    conn.noop()
                    return conn, None
                except Exception:
                    try:
                        conn.logout()
                    except Exception:
                        pass
                    continue

        # Create new connection
        try:
            ctx = ssl.create_default_context()
            conn = imaplib.IMAP4_SSL(
                self._config["imap_server"],
                self._config.get("imap_port", 993),
                ssl_context=ctx,
            )
            conn.login(self._config["email"], self._config["app_password"])
            return conn, None
        except Exception as e:
            return None, f"IMAP connection failed: {_sanitize_error(e)}"

    def release(self, conn: imaplib.IMAP4_SSL):
        """Return a connection to the pool instead of closing it."""
        with self._lock:
            if len(self._pool) < self._max:
                self._pool.append((conn, datetime.now().timestamp()))
            else:
                try:
                    conn.logout()
                except Exception:
                    pass

    def close_all(self):
        """Close all pooled connections."""
        with self._lock:
            for conn, _ in self._pool:
                try:
                    conn.logout()
                except Exception:
                    pass
            self._pool.clear()


# ---------------------------------------------------------------------------
#  Main EmailTool
# ---------------------------------------------------------------------------
class EmailTool:
    """Read and send emails via IMAP/SMTP or Gmail API.

    Backends (in priority order):
    1. Gmail API (if credentials exist at ~/.aura/gmail_creds.json)
    2. IMAP/SMTP (traditional, with connection pooling)
    """

    name = "email"
    description = "Read and send emails via IMAP/SMTP or Gmail API"

    def __init__(self):
        self._config: Optional[dict] = None
        self._pool = IMAPConnectionPool()
        self._load_config()

        # Optional backends
        self._gmail = GmailBackend() if HAS_GMAIL_API else None

        # Brain reference for AI features (set externally)
        self._brain = None

    def set_brain(self, brain):
        """Set brain reference for AI-powered features."""
        self._brain = brain

    @property
    def gmail_available(self) -> bool:
        return self._gmail is not None and self._gmail.available

    # ------------------------------------------------------------------
    #  Config (unchanged)
    # ------------------------------------------------------------------
    def _load_config(self):
        if EMAIL_CONFIG_FILE.exists():
            try:
                with open(EMAIL_CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if config.get("app_password"):
                    plaintext, err = _decrypt(config["app_password"])
                    if err:
                        logger.warning(f"[Email] Decryption error: {err}")
                        self._config = None
                        return
                    config["app_password"] = plaintext
                self._config = config
                self._pool.configure(config)
            except Exception as e:
                logger.warning(f"[Email] Config load error: {_sanitize_error(e)}")
                self._config = None

    def _save_config(self, config: dict) -> dict | None:
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
            self._pool.configure(config)
            return None
        except IOError as e:
            logger.error(f"[Email] Config save error: {e}")
            return {"success": False, "error": f"Config save error: {e}"}

    def _sanitize_header(self, value: str) -> str:
        return value.replace('\r', '').replace('\n', '').replace('\0', '')

    def _is_configured(self) -> bool:
        return (self._config is not None
                and self._config.get("email")
                and self._config.get("app_password"))

    def get_config_status(self) -> dict:
        gmail_status = "available" if self.gmail_available else "not configured"
        if self._is_configured():
            return {
                "success": True,
                "configured": True,
                "email": self._config.get("email", ""),
                "imap_server": self._config.get("imap_server", ""),
                "smtp_server": self._config.get("smtp_server", ""),
                "gmail_api": gmail_status,
                "response": f"Email configured: {self._config['email']} (Gmail API: {gmail_status})"
            }
        return {
            "success": True,
            "configured": False,
            "gmail_api": gmail_status,
            "response": f"Email not configured via IMAP. Gmail API: {gmail_status}. Use 'setup' action to configure IMAP."
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

    # ------------------------------------------------------------------
    #  IMAP helpers (with connection pooling)
    # ------------------------------------------------------------------
    def _connect_imap(self) -> tuple:
        """Get IMAP connection from pool."""
        if not self._is_configured():
            return None, "Email not configured. Use 'setup' first."
        return self._pool.get()

    def _release_imap(self, conn: imaplib.IMAP4_SSL):
        """Return connection to pool."""
        if conn:
            self._pool.release(conn)

    def _decode_header_value(self, value: str) -> str:
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
        """Parse raw email data into EmailMessage with HTML→text conversion."""
        msg = email.message_from_bytes(raw_data)

        subject = self._decode_header_value(msg.get("Subject", ""))
        sender = self._decode_header_value(msg.get("From", ""))
        to_addrs = [self._decode_header_value(a) for a in (msg.get("To", "").split(","))]
        date_str = msg.get("Date", "")

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

        # If we only have HTML, convert to text
        if not body_text and body_html:
            body_text = _html_to_text(body_html)

        return EmailMessage(
            id=str(msg_id),
            subject=subject,
            sender=sender,
            to=to_addrs,
            date=date_str,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            folder=folder,
        )

    # ------------------------------------------------------------------
    #  Gmail API convenience wrappers
    # ------------------------------------------------------------------
    def gmail_inbox(self, limit: int = 20, unread_only: bool = False,
                    label: str = None) -> dict:
        """Fetch inbox via Gmail API."""
        if not self.gmail_available:
            return {"success": False, "error": "Gmail API not configured. Place OAuth2 credentials at ~/.aura/gmail_creds.json"}
        try:
            messages = self._gmail.inbox(limit=limit, unread_only=unread_only, label=label)
            formatted = []
            for m in messages:
                unread = " [NEW]" if not m.get("is_read") else ""
                formatted.append(f"[{m['id'][:8]}] {m['sender']}: {m['subject']}{unread}")
            return {
                "success": True, "count": len(messages), "emails": messages,
                "source": "gmail_api",
                "formatted": "\n".join(formatted) if formatted else "No emails",
                "response": f"Found {len(messages)} email(s) via Gmail API\n" + "\n".join(formatted),
            }
        except Exception as e:
            return {"success": False, "error": f"Gmail API error: {e}"}

    def gmail_read(self, message_id: str) -> dict:
        """Read full message via Gmail API."""
        if not self.gmail_available:
            return {"success": False, "error": "Gmail API not configured."}
        try:
            msg = self._gmail.read(message_id)
            return {
                "success": True, "email": msg, "source": "gmail_api",
                "response": f"From: {msg['sender']}\nSubject: {msg['subject']}\nDate: {msg['date']}\n\n{msg['body_text'][:2000]}"
            }
        except Exception as e:
            return {"success": False, "error": f"Gmail read error: {e}"}

    def gmail_send(self, to: str, subject: str, body: str,
                   cc: str = None, bcc: str = None) -> dict:
        """Send email via Gmail API."""
        if not self.gmail_available:
            return {"success": False, "error": "Gmail API not configured."}
        try:
            result = self._gmail.send(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
            return {"success": True, **result, "source": "gmail_api",
                    "response": f"Email sent via Gmail API to {to}: {subject}"}
        except Exception as e:
            return {"success": False, "error": f"Gmail send error: {e}"}

    def gmail_reply(self, message_id: str, body: str) -> dict:
        """Reply via Gmail API."""
        if not self.gmail_available:
            return {"success": False, "error": "Gmail API not configured."}
        try:
            result = self._gmail.reply(message_id, body)
            return {"success": True, **result, "source": "gmail_api",
                    "response": f"Reply sent via Gmail API (thread: {result.get('thread_id', '')})"}
        except Exception as e:
            return {"success": False, "error": f"Gmail reply error: {e}"}

    def gmail_search(self, query: str, limit: int = 20) -> dict:
        """Search via Gmail API."""
        if not self.gmail_available:
            return {"success": False, "error": "Gmail API not configured."}
        try:
            results = self._gmail.search(query, limit=limit)
            return {
                "success": True, "count": len(results), "results": results,
                "source": "gmail_api", "query": query,
                "response": f"Found {len(results)} email(s) matching '{query}' via Gmail API"
            }
        except Exception as e:
            return {"success": False, "error": f"Gmail search error: {e}"}

    # ------------------------------------------------------------------
    #  Core IMAP operations (upgraded with pooling + HTML conversion)
    # ------------------------------------------------------------------
    def inbox(self, limit: int = 20, unread_only: bool = False) -> dict:
        """Fetch inbox — uses Gmail API if available, falls back to IMAP."""
        if self.gmail_available:
            return self.gmail_inbox(limit=limit, unread_only=unread_only)
        return self.fetch_emails(folder="INBOX", limit=limit, unread_only=unread_only)

    def read(self, message_id: str) -> dict:
        """Read message — uses Gmail API if available, falls back to IMAP."""
        if self.gmail_available:
            return self.gmail_read(message_id)
        return self.read_email(message_id)

    def send(self, to: str, subject: str, body: str,
             cc: str = None, bcc: str = None) -> dict:
        """Send email — uses Gmail API if available, falls back to SMTP."""
        if self.gmail_available:
            return self.gmail_send(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
        return self.send_email(to=to, subject=subject, body=body, cc=cc, bcc=bcc)

    def search(self, query: str, limit: int = 20) -> dict:
        """Search — uses Gmail API if available, falls back to IMAP."""
        if self.gmail_available:
            return self.gmail_search(query, limit=limit)
        return self.search_emails(query)

    def fetch_emails(self, folder: str = "INBOX", limit: int = 10,
                     unread_only: bool = False, since_date: str = None) -> dict:
        """Fetch emails from a folder via IMAP."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        conn = None
        try:
            conn, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            folder = _sanitize_imap_string(folder)
            conn.select(folder, readonly=True)

            criteria = []
            if unread_only:
                criteria.append("UNSEEN")
            if since_date:
                safe_date = _sanitize_imap_string(since_date)
                criteria.append(f'SINCE "{safe_date}"')
            if not criteria:
                criteria.append("ALL")

            search_str = " ".join(criteria)
            status, messages = conn.search(None, search_str)

            if status != "OK":
                self._release_imap(conn)
                return {"success": False, "error": "IMAP search failed"}

            msg_ids = messages[0].split()
            msg_ids = msg_ids[-limit:]
            msg_ids.reverse()

            emails = []
            for msg_id in msg_ids:
                status, data = conn.fetch(msg_id, "(RFC822)")
                if status == "OK" and data[0]:
                    parsed = self._parse_email(data[0][1], msg_id.decode(), folder)
                    emails.append({
                        "id": parsed.id,
                        "subject": parsed.subject,
                        "sender": parsed.sender,
                        "date": parsed.date,
                        "has_attachments": len(parsed.attachments) > 0,
                        "attachments": parsed.attachments,
                        "preview": parsed.body_text[:200],
                    })

            self._release_imap(conn)

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
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
            return {"success": False, "error": f"Fetch failed: {_sanitize_error(e)}"}

    def read_email(self, email_id: str) -> dict:
        """Read full email content via IMAP."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        conn = None
        try:
            conn, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            conn.select("INBOX")

            status, data = conn.fetch(email_id.encode(), "(RFC822)")
            if status != "OK" or not data[0]:
                self._release_imap(conn)
                return {"success": False, "error": f"Email not found: {email_id}"}

            parsed = self._parse_email(data[0][1], email_id)
            self._release_imap(conn)

            return {
                "success": True,
                "email": parsed.to_dict(),
                "response": f"From: {parsed.sender}\nSubject: {parsed.subject}\nDate: {parsed.date}\n\n{parsed.body_text[:2000]}"
            }

        except Exception as e:
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
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

    def reply_email(self, email_id: str, body: str) -> dict:
        """Reply to an email — uses Gmail API if available, falls back to IMAP/SMTP."""
        if self.gmail_available:
            return self.gmail_reply(email_id, body)

        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        original = self.read_email(email_id)
        if not original.get("success"):
            return original

        orig_email = original["email"]
        to = orig_email.get("sender", "")

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

        conn = None
        try:
            conn, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            conn.select(folder, readonly=True)

            safe_query = _sanitize_imap_string(query)
            status, messages = conn.search(None, f'(OR SUBJECT "{safe_query}" BODY "{safe_query}")')

            if status != "OK":
                self._release_imap(conn)
                return {"success": False, "error": "Search failed"}

            msg_ids = messages[0].split()[-20:]
            msg_ids.reverse()

            results = []
            for msg_id in msg_ids:
                status, data = conn.fetch(msg_id, "(RFC822)")
                if status == "OK" and data[0]:
                    parsed = self._parse_email(data[0][1], msg_id.decode(), folder)
                    results.append({
                        "id": parsed.id,
                        "subject": parsed.subject,
                        "sender": parsed.sender,
                        "date": parsed.date,
                    })

            self._release_imap(conn)

            return {
                "success": True,
                "count": len(results),
                "results": results,
                "query": query,
                "response": f"Found {len(results)} email(s) matching '{query}'"
            }

        except Exception as e:
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
            return {"success": False, "error": f"Search failed: {_sanitize_error(e)}"}

    def list_folders(self) -> dict:
        """List mailbox folders."""
        if not self._is_configured():
            return {"success": False, "error": "Email not configured. Use 'setup' first."}

        conn = None
        try:
            conn, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            status, folders = conn.list()
            self._release_imap(conn)

            if status != "OK":
                return {"success": False, "error": "Could not list folders"}

            folder_names = []
            for f in folders:
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
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
            return {"success": False, "error": f"List folders failed: {_sanitize_error(e)}"}

    def list_attachments(self, email_id: str) -> dict:
        """List attachments for a given email."""
        result = self.read(email_id)
        if not result.get("success"):
            return result

        email_data = result.get("email", {})
        attachments = email_data.get("attachments", [])

        formatted = []
        for i, att in enumerate(attachments):
            size_kb = att.get("size", 0) / 1024
            formatted.append(f"  [{i}] {att.get('filename', 'unknown')} ({att.get('content_type', '?')}, {size_kb:.1f}KB)")

        return {
            "success": True,
            "attachments": attachments,
            "count": len(attachments),
            "response": f"Attachments for email {email_id}:\n" + ("\n".join(formatted) if formatted else "No attachments")
        }

    def download_attachment(self, email_id: str, attachment_index: int = 0,
                            save_dir: str = None) -> dict:
        """Download an attachment from an email.

        For Gmail API: uses attachmentId directly.
        For IMAP: extracts from the raw message.
        """
        if self.gmail_available:
            return self._gmail_download_attachment(email_id, attachment_index, save_dir)
        return self._imap_download_attachment(email_id, attachment_index, save_dir)

    def _gmail_download_attachment(self, email_id: str, index: int, save_dir: str = None) -> dict:
        try:
            msg = self._gmail.read(email_id)
            attachments = msg.get("attachments", [])
            if index >= len(attachments):
                return {"success": False, "error": f"Attachment index {index} out of range (have {len(attachments)})"}

            att = attachments[index]
            att_id = att.get("attachmentId", "")
            if not att_id:
                return {"success": False, "error": "No attachmentId available"}

            data = self._gmail.download_attachment(email_id, att_id)
            target_dir = Path(save_dir) if save_dir else Path.home() / "Downloads"
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(att["filename"]).name  # Strip directory components
            filepath = (target_dir / safe_name).resolve()
            try:
                filepath.relative_to(target_dir.resolve())
            except ValueError:
                return {"success": False, "error": "Invalid attachment filename"}
            filepath.write_bytes(data)

            return {"success": True, "path": str(filepath), "filename": safe_name,
                    "size": len(data), "response": f"Downloaded {safe_name} to {filepath}"}
        except Exception as e:
            return {"success": False, "error": f"Download failed: {e}"}

    def _imap_download_attachment(self, email_id: str, index: int, save_dir: str = None) -> dict:
        if not self._is_configured():
            return {"success": False, "error": "Email not configured."}

        conn = None
        try:
            conn, err = self._connect_imap()
            if err:
                return {"success": False, "error": err}
            conn.select("INBOX")

            status, data = conn.fetch(email_id.encode(), "(RFC822)")
            if status != "OK" or not data[0]:
                self._release_imap(conn)
                return {"success": False, "error": f"Email not found: {email_id}"}

            msg = email.message_from_bytes(data[0][1])
            self._release_imap(conn)

            attachments = []
            for part in msg.walk():
                if "attachment" in str(part.get("Content-Disposition", "")):
                    filename = part.get_filename()
                    if filename:
                        attachments.append((self._decode_header_value(filename), part))

            if index >= len(attachments):
                return {"success": False, "error": f"Attachment index {index} out of range (have {len(attachments)})"}

            filename, part = attachments[index]
            payload = part.get_payload(decode=True)
            target_dir = Path(save_dir) if save_dir else Path.home() / "Downloads"
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(filename).name  # Strip directory components
            filepath = (target_dir / safe_name).resolve()
            try:
                filepath.relative_to(target_dir.resolve())
            except ValueError:
                return {"success": False, "error": "Invalid attachment filename"}
            filepath.write_bytes(payload)

            return {"success": True, "path": str(filepath), "filename": safe_name,
                    "size": len(payload), "response": f"Downloaded {safe_name} to {filepath}"}
        except Exception as e:
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
            return {"success": False, "error": f"Download failed: {_sanitize_error(e)}"}

    def mark_read(self, email_id: str) -> dict:
        return self._set_flag(email_id, "\\Seen", add=True)

    def mark_unread(self, email_id: str) -> dict:
        return self._set_flag(email_id, "\\Seen", add=False)

    def _set_flag(self, email_id: str, flag: str, add: bool = True) -> dict:
        if not self._is_configured():
            return {"success": False, "error": "Email not configured."}

        conn = None
        try:
            conn, err = self._connect_imap()
            if err:
                return {"success": False, "error": err, "blocked_by": "configuration"}
            conn.select("INBOX")
            action = "+FLAGS" if add else "-FLAGS"
            status, _ = conn.store(email_id.encode(), action, flag)
            self._release_imap(conn)

            if status == "OK":
                label = "read" if add else "unread"
                return {"success": True, "response": f"Email {email_id} marked as {label}"}
            return {"success": False, "error": "Flag update failed"}

        except Exception as e:
            if conn:
                try:
                    conn.logout()
                except Exception:
                    pass
            return {"success": False, "error": f"Flag failed: {_sanitize_error(e)}"}

    # ------------------------------------------------------------------
    #  Email Intelligence (AI-powered)
    # ------------------------------------------------------------------
    def _get_brain(self):
        """Get brain reference for AI features."""
        if self._brain is not None:
            return self._brain
        # Try to import and get global brain
        try:
            from aura.brain import get_brain
            self._brain = get_brain()
            return self._brain
        except Exception:
            return None

    def summarize_inbox(self, limit: int = 10) -> dict:
        """Generate an AI summary of recent emails."""
        inbox_result = self.inbox(limit=limit)
        if not inbox_result.get("success"):
            return inbox_result

        emails = inbox_result.get("emails", [])
        if not emails:
            return {"success": True, "summary": "Inbox is empty.", "response": "Inbox is empty."}

        brain = self._get_brain()
        if not brain:
            # Fallback: structured summary without AI
            lines = []
            for em in emails:
                lines.append(f"- {em.get('sender', '?')}: {em.get('subject', '?')} ({em.get('date', '?')})")
            summary = f"Recent {len(emails)} emails:\n" + "\n".join(lines)
            return {"success": True, "summary": summary, "response": summary}

        # Build context for AI
        email_list = []
        for em in emails:
            email_list.append(
                f"From: {em.get('sender', '?')}\n"
                f"Subject: {em.get('subject', '?')}\n"
                f"Date: {em.get('date', '?')}\n"
                f"Preview: {em.get('preview', em.get('snippet', ''))[:150]}"
            )

        prompt = (
            "Summarize this inbox concisely. Group by priority. "
            "Highlight anything urgent or requiring action.\n\n"
            + "\n---\n".join(email_list)
        )

        try:
            summary = brain._quick_generate(prompt, timeout=30)
            return {"success": True, "summary": summary, "count": len(emails),
                    "response": f"Inbox summary ({len(emails)} emails):\n{summary}"}
        except Exception as e:
            logger.warning(f"[Email] AI summarize failed: {e}")
            lines = [f"- {em.get('sender', '?')}: {em.get('subject', '?')}" for em in emails]
            fallback = "\n".join(lines)
            return {"success": True, "summary": fallback, "response": fallback}

    def draft_reply(self, message_id: str, intent: str = "") -> dict:
        """Generate a reply draft using AI.

        Args:
            message_id: Email to reply to
            intent: What the user wants to say (e.g. "accept the meeting", "decline politely")
        """
        msg_result = self.read(message_id)
        if not msg_result.get("success"):
            return msg_result

        email_data = msg_result.get("email", {})
        body = email_data.get("body_text", "")[:1500]
        subject = email_data.get("subject", "")
        sender = email_data.get("sender", "")

        brain = self._get_brain()
        if not brain:
            return {"success": False, "error": "AI brain not available for draft generation."}

        intent_instruction = f"The user's intent: {intent}" if intent else "Write a professional and helpful reply."

        prompt = (
            f"Draft a reply to this email.\n\n"
            f"From: {sender}\nSubject: {subject}\n\n{body}\n\n"
            f"{intent_instruction}\n\n"
            f"Write ONLY the reply body text, nothing else."
        )

        try:
            draft = brain._quick_generate(prompt, timeout=30)
            return {
                "success": True,
                "draft": draft,
                "to": sender,
                "subject": f"Re: {subject}" if not subject.lower().startswith("re:") else subject,
                "original_id": message_id,
                "response": f"Draft reply to {sender}:\n\n{draft}"
            }
        except Exception as e:
            return {"success": False, "error": f"Draft generation failed: {e}"}

    # ------------------------------------------------------------------
    #  Sender profile cache (SQLite-backed)
    # ------------------------------------------------------------------

    def _get_sender_cache_db(self):
        """Open (and bootstrap) the sender profile cache."""
        cache_path = Path(__file__).parent.parent.parent / "data" / "email_sender_cache.db"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(cache_path))
        conn.execute("""CREATE TABLE IF NOT EXISTS sender_profiles (
            sender TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            last_classified TEXT NOT NULL)""")
        conn.commit()
        return conn

    def _lookup_sender_cache(self, sender: str) -> Optional[str]:
        """Check if sender is in the cache. Returns category or None."""
        try:
            conn = self._get_sender_cache_db()
            try:
                row = conn.execute(
                    "SELECT category FROM sender_profiles WHERE sender=?", (sender,)
                ).fetchone()
                return row[0] if row else None
            finally:
                conn.close()
        except Exception:
            return None

    def _store_sender_cache(self, sender: str, category: str) -> None:
        """Store sender -> category mapping in the cache."""
        try:
            conn = self._get_sender_cache_db()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO sender_profiles (sender, category, last_classified) VALUES (?,?,?)",
                    (sender, category, datetime.now().isoformat()))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[Email] Sender cache store error: {e}")

    def categorize(self, message_id: str) -> dict:
        """Auto-categorize an email into: urgent, action_needed, fyi, newsletter, spam, personal.

        Uses sender cache first, then AI if available, otherwise heuristics.
        """
        msg_result = self.read(message_id)
        if not msg_result.get("success"):
            return msg_result

        email_data = msg_result.get("email", {})
        subject = email_data.get("subject", "").lower()
        body = email_data.get("body_text", "")[:500].lower()
        sender = email_data.get("sender", "").lower()

        # Check sender cache first — skip LLM for known senders
        cached = self._lookup_sender_cache(sender)
        if cached:
            return {
                "success": True,
                "message_id": message_id,
                "category": cached,
                "method": "sender_cache",
                "response": f"Email categorized as: {cached} (cached sender profile)"
            }

        brain = self._get_brain()
        if brain:
            prompt = (
                f"Categorize this email into exactly ONE category: "
                f"urgent, action_needed, fyi, newsletter, spam, personal.\n\n"
                f"From: {sender}\nSubject: {subject}\n\n{body[:500]}\n\n"
                f"Respond with ONLY the category name, nothing else."
            )
            try:
                raw = brain._quick_generate(prompt, timeout=15)
                category = raw.strip().lower().replace(" ", "_")
                valid = {"urgent", "action_needed", "fyi", "newsletter", "spam", "personal"}
                if category not in valid:
                    # Try to find a valid category in the response
                    for v in valid:
                        if v in category:
                            category = v
                            break
                    else:
                        category = "fyi"  # default

                # Store in sender cache for future lookups
                if sender:
                    self._store_sender_cache(sender, category)

                return {
                    "success": True,
                    "message_id": message_id,
                    "category": category,
                    "method": "ai",
                    "response": f"Email categorized as: {category}"
                }
            except Exception as e:
                logger.warning(f"[Email] AI categorize failed: {e}")

        # Heuristic fallback
        category = "fyi"

        urgent_keywords = ["urgent", "asap", "emergency", "immediately", "critical", "deadline"]
        action_keywords = ["please", "action required", "respond", "rsvp", "confirm", "approve", "review"]
        newsletter_keywords = ["unsubscribe", "newsletter", "digest", "weekly update", "no-reply", "noreply"]
        spam_keywords = ["winner", "congratulations", "free money", "act now", "limited time"]

        combined = subject + " " + body

        if any(kw in combined for kw in urgent_keywords):
            category = "urgent"
        elif any(kw in combined for kw in action_keywords):
            category = "action_needed"
        elif any(kw in combined for kw in spam_keywords):
            category = "spam"
        elif any(kw in combined for kw in newsletter_keywords) or "noreply" in sender or "no-reply" in sender:
            category = "newsletter"

        # Store heuristic result in sender cache too
        if sender:
            self._store_sender_cache(sender, category)

        return {
            "success": True,
            "message_id": message_id,
            "category": category,
            "method": "heuristic",
            "response": f"Email categorized as: {category}"
        }

    def extract_action_items(self, message_id: str) -> dict:
        """Extract action items / tasks from an email body.

        Uses AI if available, otherwise uses pattern matching.
        """
        msg_result = self.read(message_id)
        if not msg_result.get("success"):
            return msg_result

        email_data = msg_result.get("email", {})
        body = email_data.get("body_text", "")[:2000]
        subject = email_data.get("subject", "")

        brain = self._get_brain()
        if brain:
            prompt = (
                f"Extract all action items and tasks from this email. "
                f"Return them as a numbered list. If no action items, say 'No action items found.'\n\n"
                f"Subject: {subject}\n\n{body}"
            )
            try:
                raw = brain._quick_generate(prompt, timeout=20)
                items = [line.strip() for line in raw.strip().split("\n") if line.strip()]
                # Clean up numbering
                cleaned = []
                for item in items:
                    item = re.sub(r'^\d+[\.\)]\s*', '', item).strip()
                    if item and item.lower() != "no action items found.":
                        cleaned.append(item)

                return {
                    "success": True,
                    "message_id": message_id,
                    "action_items": cleaned,
                    "count": len(cleaned),
                    "method": "ai",
                    "response": f"Action items from '{subject}':\n" + (
                        "\n".join(f"  - {item}" for item in cleaned) if cleaned else "No action items found."
                    ),
                }
            except Exception as e:
                logger.warning(f"[Email] AI extract_action_items failed: {e}")

        # Heuristic fallback: look for imperative sentences and todo patterns
        items = []
        patterns = [
            r'(?:please|kindly|could you|can you|need you to|make sure to)\s+(.+?)(?:\.|$)',
            r'(?:TODO|TO DO|Action|Task):\s*(.+?)(?:\.|$)',
            r'(?:deadline|due|by)\s+(.+?)(?:\.|$)',
            r'(?:^|\n)\s*[-*]\s+(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, body, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) > 10 and cleaned not in items:
                    items.append(cleaned)

        return {
            "success": True,
            "message_id": message_id,
            "action_items": items[:10],
            "count": len(items[:10]),
            "method": "heuristic",
            "response": f"Action items from '{subject}':\n" + (
                "\n".join(f"  - {item}" for item in items[:10]) if items else "No action items found."
            ),
        }

    # ------------------------------------------------------------------
    #  execute() — extended with new actions
    # ------------------------------------------------------------------
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

        # Summarize inbox
        if action_lower.startswith("summarize") or action_lower.startswith("summary"):
            limit = kwargs.get("limit", 10)
            return self.summarize_inbox(limit=limit)

        # Categorize
        if action_lower.startswith("categorize") or action_lower.startswith("classify"):
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            if not email_id:
                parts = action.split()
                email_id = parts[-1] if len(parts) > 1 else None
            if email_id:
                return self.categorize(email_id)
            return {"success": False, "error": "No email ID specified for categorization"}

        # Extract action items
        if "action item" in action_lower or "extract task" in action_lower or "extract action" in action_lower:
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            if not email_id:
                parts = action.split()
                email_id = parts[-1] if len(parts) > 1 else None
            if email_id:
                return self.extract_action_items(email_id)
            return {"success": False, "error": "No email ID specified"}

        # Draft reply
        if action_lower.startswith("draft"):
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            intent = kwargs.get("intent", "")
            if not email_id:
                parts = action.split()
                email_id = parts[-1] if len(parts) > 1 else None
            if email_id:
                return self.draft_reply(email_id, intent=intent)
            return {"success": False, "error": "No email ID specified for draft"}

        # Attachments
        if action_lower.startswith("attachment") or action_lower.startswith("list attachment"):
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            if not email_id:
                parts = action.split()
                email_id = parts[-1] if len(parts) > 1 else None
            if email_id:
                return self.list_attachments(email_id)
            return {"success": False, "error": "No email ID specified"}

        if action_lower.startswith("download"):
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            index = kwargs.get("attachment_index", 0)
            save_dir = kwargs.get("save_dir")
            if email_id:
                return self.download_attachment(email_id, attachment_index=index, save_dir=save_dir)
            return {"success": False, "error": "No email ID specified for download"}

        # Gmail-specific commands
        if action_lower.startswith("gmail"):
            sub = action_lower.replace("gmail", "").strip()
            if sub.startswith("inbox") or sub.startswith("list"):
                limit = kwargs.get("limit", 20)
                unread = "unread" in sub
                label = kwargs.get("label")
                return self.gmail_inbox(limit=limit, unread_only=unread, label=label)
            elif sub.startswith("read"):
                email_id = kwargs.get("email_id") or kwargs.get("message_id")
                if email_id:
                    return self.gmail_read(email_id)
                return {"success": False, "error": "No email ID specified"}
            elif sub.startswith("send"):
                return self.gmail_send(
                    to=kwargs.get("to", ""),
                    subject=kwargs.get("subject", ""),
                    body=kwargs.get("body", ""),
                    cc=kwargs.get("cc"),
                    bcc=kwargs.get("bcc"),
                )
            elif sub.startswith("search"):
                query = kwargs.get("query", "")
                return self.gmail_search(query)
            elif sub.startswith("reply"):
                email_id = kwargs.get("email_id") or kwargs.get("message_id")
                body = kwargs.get("body", "")
                if email_id and body:
                    return self.gmail_reply(email_id, body)
                return {"success": False, "error": "Need email_id and body for reply"}
            return self.gmail_inbox()

        # Fetch emails (inbox)
        if action_lower.startswith("fetch") or action_lower in ("inbox", "check", "check mail"):
            unread_only = "unread" in action_lower
            limit = kwargs.get("limit", 10)
            folder = kwargs.get("folder", "INBOX")
            return self.inbox(limit=limit, unread_only=unread_only)

        # Read specific email
        if action_lower.startswith("read"):
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            if not email_id:
                parts = action.split()
                email_id = parts[-1] if len(parts) > 1 else None
            if email_id:
                return self.read(email_id)
            return {"success": False, "error": "No email ID specified"}

        # Send email
        if action_lower.startswith("send"):
            to = kwargs.get("to")
            subject = kwargs.get("subject")
            body = kwargs.get("body")

            if not all([to, subject, body]):
                to_match = re.search(r'to:\s*(\S+)', action, re.IGNORECASE)
                subj_match = re.search(r'subject:\s*(.+?)(?:\s+body:|\s*$)', action, re.IGNORECASE)
                body_match = re.search(r'body:\s*(.+)', action, re.IGNORECASE)

                to = to or (to_match.group(1) if to_match else None)
                subject = subject or (subj_match.group(1).strip() if subj_match else None)
                body = body or (body_match.group(1).strip() if body_match else None)

            if to and subject and body:
                return self.send(to=to, subject=subject, body=body,
                                 cc=kwargs.get("cc"), bcc=kwargs.get("bcc"))
            return {"success": False, "error": "Usage: send to:<addr> subject:<subj> body:<text>"}

        # Reply
        if action_lower.startswith("reply"):
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
            body = kwargs.get("body")
            if not email_id:
                parts = action.split()
                email_id = parts[1] if len(parts) > 1 else None
            if email_id and body:
                return self.reply_email(email_id, body)
            return {"success": False, "error": "Usage: reply <email_id> body:<text>"}

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            folder = kwargs.get("folder", "INBOX")
            return self.search(query)

        # Mark read/unread
        if "mark" in action_lower:
            email_id = kwargs.get("email_id") or kwargs.get("message_id")
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
                     "Try: 'inbox', 'fetch unread', 'read <id>', 'send to:<addr> subject:<subj> body:<text>', "
                     "'reply <id>', 'search <query>', 'summarize', 'categorize <id>', 'draft <id>', "
                     "'extract action items <id>', 'attachments <id>', 'download <id>', 'gmail inbox', 'setup'"
        }


# Singleton
email_tool = EmailTool()
