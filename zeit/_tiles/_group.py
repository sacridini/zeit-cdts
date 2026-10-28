"""The group of computers that share tiles: its secret, kept in the user's configuration,
and the proofs that a coordinator and a helper both know it (or, while a helper joins,
the invitation code).

The short invitation code is never the long-term secret. It authenticates one pairing,
over TLS, in which the helper receives the group's random 256-bit secret. Every later
connection proves knowledge of that secret both ways, bound to the coordinator's TLS
certificate, so a machine in the middle can neither impersonate one side nor relay.
"""

import datetime
import functools
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
from pathlib import Path
from typing import Optional, Tuple

from .._config import config_dir

PROTOCOL = 1
TCP_PORT = 47801
UDP_PORT = 47800

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"   # Crockford's base32: no I, L, O, U


def _group_file() -> Path:
    return config_dir() / "group.json"


def load_secret() -> Optional[bytes]:
    try:
        return bytes.fromhex(json.loads(_group_file().read_text())["secret"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_secret(secret: bytes) -> None:
    path = _group_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"secret": secret.hex()}, f)
    os.replace(tmp, path)


def ensure_secret() -> bytes:
    """The group's secret, created (a group of one) on first use."""
    secret = load_secret()
    if secret is None:
        secret = secrets.token_bytes(32)
        save_secret(secret)
    return secret


def forget_secret() -> bool:
    try:
        _group_file().unlink()
        return True
    except FileNotFoundError:
        return False


def group_id(secret: bytes) -> str:
    """A public name of the group, for discovery: it reveals nothing of the secret."""
    return hmac.new(secret, b"zeit-group-id", hashlib.sha256).hexdigest()[:16]


def new_code() -> str:
    chars = "".join(secrets.choice(_ALPHABET) for _ in range(8))   # 40 bits
    return f"{chars[:4]}-{chars[4:]}"


def normalize_code(code: str) -> str:
    text = code.upper().replace("-", "").replace(" ", "").translate(str.maketrans("OIL", "011"))
    if len(text) != 8 or any(c not in _ALPHABET for c in text):
        raise ValueError(f"not an invitation code: {code!r} (it looks like K7QM-2XDP)")
    return text


def code_key(code: str, salt: bytes) -> bytes:
    """The pairing key: scrypt makes guessing the code offline from a captured proof
    take months instead of minutes, and the invitation expires long before."""
    return hashlib.scrypt(normalize_code(code).encode(), salt=salt, n=2 ** 15, r=8, p=1,
                          maxmem=2 ** 26, dklen=32)


def proof(key: bytes, role: str, *parts: str) -> str:
    message = "|".join(("zeit-tiles", str(PROTOCOL), role) + parts).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(str(a), str(b))


@functools.lru_cache(maxsize=None)
def version() -> str:
    """zeit's version. In a source checkout (an editable install) the installed metadata
    can be stale, so the checkout's pyproject.toml is read instead."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = pyproject.read_text()
        if re.search(r'^name\s*=\s*"zeit-cdts"', text, re.M):
            found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
            if found:
                return found.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version as _version

        return _version("zeit-cdts")
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=None)
def code_fingerprint() -> str:
    """A hash of zeit's Python code: two computers with the same version number but other
    code (a development install on one of them) would give other results."""
    package = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(package).as_posix().encode() + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def machine_name() -> str:
    return socket.gethostname()


def certificate(directory: Path) -> Tuple[str, str, str]:
    """A throwaway self-signed certificate: (cert file, key file, SHA-256 fingerprint).
    Its trust comes from the proofs, which are bound to the fingerprint, not from a CA."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "zeit")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30)).sign(key, hashes.SHA256()))
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return str(cert_path), str(key_path), fingerprint
