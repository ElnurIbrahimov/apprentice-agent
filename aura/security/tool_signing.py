"""Ed25519 tool signing and verification.

Stolen from OpenFang: every agent manifest is cryptographically signed.
Prevents tampering with custom tools loaded from the marketplace/custom directory.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# hmac is always needed (verify_tool must handle hmac-sha256 sigs even when nacl is available)
import hmac

# Try to import Ed25519 — fallback to HMAC-SHA256 if nacl not available
_HAS_NACL = False
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError
    _HAS_NACL = True
except ImportError:
    logger.debug("[ToolSign] PyNaCl not installed, falling back to HMAC-SHA256")


_KEY_DIR = os.path.join(os.environ.get("AURA_DATA_DIR", "data"), "keys")
_PRIVATE_KEY_FILE = os.path.join(_KEY_DIR, "tool_signing.key")
_PUBLIC_KEY_FILE = os.path.join(_KEY_DIR, "tool_signing.pub")


def _ensure_keypair() -> Tuple[bytes, bytes]:
    """Generate or load the signing keypair."""
    os.makedirs(_KEY_DIR, exist_ok=True)

    if os.path.exists(_PRIVATE_KEY_FILE):
        with open(_PRIVATE_KEY_FILE, "rb") as f:
            private = f.read()
        if os.path.exists(_PUBLIC_KEY_FILE):
            with open(_PUBLIC_KEY_FILE, "rb") as f:
                public = f.read()
        else:
            # HMAC mode: private key is also the verify key
            public = private
        return private, public

    if _HAS_NACL:
        sk = SigningKey.generate()
        private = bytes(sk)
        public = bytes(sk.verify_key)
    else:
        # HMAC fallback: shared secret
        private = os.urandom(32)
        public = private  # HMAC uses same key for sign/verify

    with open(_PRIVATE_KEY_FILE, "wb") as f:
        f.write(private)
    try:
        os.chmod(_PRIVATE_KEY_FILE, 0o600)
    except OSError:
        pass  # Windows: chmod doesn't work the same way

    if _HAS_NACL:
        # Only write .pub when it's a real public key (Ed25519 verify key)
        with open(_PUBLIC_KEY_FILE, "wb") as f:
            f.write(public)
    else:
        # HMAC mode: do NOT write a .pub file — the "public" key IS the secret.
        # Both sign and verify read from _PRIVATE_KEY_FILE.
        if os.path.exists(_PUBLIC_KEY_FILE):
            try:
                os.unlink(_PUBLIC_KEY_FILE)
            except OSError:
                pass

    logger.info(f"[ToolSign] Generated new {'Ed25519' if _HAS_NACL else 'HMAC'} keypair")
    return private, public


def _hash_file(path: str) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_tool(tool_path: str) -> str:
    """Sign a tool file. Creates a .sig file alongside it.

    Returns the signature file path.
    """
    tool_path = str(Path(tool_path).resolve())
    if not os.path.isfile(tool_path):
        raise FileNotFoundError(f"Tool file not found: {tool_path}")

    private_key, _ = _ensure_keypair()
    content_hash = _hash_file(tool_path)

    # Sign the hash
    if _HAS_NACL:
        sk = SigningKey(private_key)
        signed = sk.sign(content_hash.encode("utf-8"))
        signature = signed.signature.hex()
    else:
        sig = hmac.new(private_key, content_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        signature = sig

    # Write signature file
    sig_path = tool_path + ".sig"
    sig_data = {
        "version": 1,
        "file": os.path.basename(tool_path),
        "hash": content_hash,
        "signature": signature,
        "algorithm": "ed25519" if _HAS_NACL else "hmac-sha256",
        "signed_at": time.time(),
    }
    with open(sig_path, "w") as f:
        json.dump(sig_data, f, indent=2)

    logger.info(f"[ToolSign] Signed {os.path.basename(tool_path)} → {os.path.basename(sig_path)}")
    return sig_path


def verify_tool(tool_path: str) -> Tuple[bool, Optional[str]]:
    """Verify a tool file's signature.

    Returns:
        (is_valid, error_message) — is_valid=True if signature checks out
    """
    tool_path = str(Path(tool_path).resolve())
    sig_path = tool_path + ".sig"

    if not os.path.isfile(tool_path):
        return False, f"Tool file not found: {tool_path}"

    if not os.path.isfile(sig_path):
        return False, f"No signature file found: {sig_path}"

    _, public_key = _ensure_keypair()

    # Read signature file
    try:
        with open(sig_path, "r") as f:
            sig_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Invalid signature file: {e}"

    stored_hash = sig_data.get("hash", "")
    signature = sig_data.get("signature", "")
    algorithm = sig_data.get("algorithm", "")

    # Verify content hash matches current file
    current_hash = _hash_file(tool_path)
    if current_hash != stored_hash:
        return False, f"File has been modified since signing (hash mismatch)"

    # Verify cryptographic signature
    if algorithm == "ed25519" and _HAS_NACL:
        try:
            vk = VerifyKey(public_key)
            vk.verify(stored_hash.encode("utf-8"), bytes.fromhex(signature))
            return True, None
        except BadSignatureError:
            return False, "Ed25519 signature verification failed"
        except Exception as e:
            return False, f"Signature verification error: {e}"
    elif algorithm == "hmac-sha256":
        expected = hmac.new(public_key, stored_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True, None
        else:
            return False, "HMAC signature verification failed"
    else:
        return False, f"Unknown algorithm: {algorithm}"


def is_tool_signed(tool_path: str) -> bool:
    """Quick check if a signature file exists for a tool."""
    return os.path.isfile(str(Path(tool_path).resolve()) + ".sig")
