"""HTTP over TLS between a coordinator and its helpers, the handshake that authenticates
both, and the UDP broadcasts with which helpers find a coordinator on the local network.

Helpers only open outgoing connections: they broadcast that they are free and connect to
whoever answers. Only the coordinator listens, so only it needs a firewall rule.
"""

import http.client
import json
import secrets
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import _group

CHUNK = 1 << 20
HANDSHAKE_TTL = 60.0


class ProtocolError(RuntimeError):
    """The other side refused (wrong group or code, another zeit version) or broke the protocol."""


# ---------------------------------------------------------------------------- server side

class Handshakes:
    """The server's half of the handshake. ``key`` is the group's secret or, while pairing,
    the key derived from the invitation code (whose ``salt`` goes to the helper)."""

    def __init__(self, key: bytes, fingerprint: str, salt: Optional[bytes] = None):
        self.key, self.fingerprint, self.salt = key, fingerprint, salt
        self._pending: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()

    def hello(self, body: dict) -> dict:
        theirs = str(body.get("nonce", ""))
        if len(theirs) != 64:
            raise ProtocolError("bad hello")
        ours = secrets.token_hex(32)
        now = time.monotonic()
        with self._lock:
            self._pending = {k: v for k, v in self._pending.items() if now - v[1] < HANDSHAKE_TTL}
            self._pending[ours] = (theirs, now)
        reply = {"nonce": ours, "proof": _group.proof(self.key, "server", theirs, ours, self.fingerprint),
                 "protocol": _group.PROTOCOL}
        if self.salt is not None:
            reply["salt"] = self.salt.hex()
        return reply

    def auth(self, body: dict) -> bool:
        ours = str(body.get("nonce", ""))
        with self._lock:
            entry = self._pending.pop(ours, None)      # single use
        if entry is None or time.monotonic() - entry[1] > HANDSHAKE_TTL:
            return False
        expected = _group.proof(self.key, "helper", entry[0], ours, self.fingerprint)
        return _group.same(expected, body.get("proof", ""))


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 300

    def log_message(self, *args):   # quiet
        pass

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def _dispatch(self, method: str) -> None:
        try:
            self.server.app.handle(self, method, self.path)
        except ProtocolError as e:
            self.send_json(400, {"error": str(e)})
        except (ConnectionError, ssl.SSLError, socket.timeout):
            self.close_connection = True
        except Exception as e:   # never take the server down
            try:
                self.send_json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                self.close_connection = True

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def read_json(self) -> dict:
        try:
            data = json.loads(self.read_body() or b"{}")
        except ValueError:
            raise ProtocolError("bad JSON")
        if not isinstance(data, dict):
            raise ProtocolError("bad JSON")
        return data

    def send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_stream(self, size: int, chunks) -> int:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        sent = 0
        for chunk in chunks:
            self.wfile.write(chunk)
            sent += len(chunk)
        return sent


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server take a port that is in use.
    allow_reuse_address = sys.platform != "win32"


class Server:
    """A TLS HTTP server in a background thread; ``app.handle(handler, method, path)``
    answers every request."""

    def __init__(self, app, port: int, cert: str, key: str, host: str = ""):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        try:
            self._httpd = _Server((host, port), _Handler)
        except OSError as e:
            raise OSError(f"port {port} is in use (another zeit coordinator or invitation?): {e}") from e
        # The TLS handshake happens in the request's thread, not in accept().
        self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True,
                                                 do_handshake_on_connect=False)
        self._httpd.app = app
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="zeit-tiles-server", daemon=True)

    def start(self) -> "Server":
        self._thread.start()
        return self

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# ---------------------------------------------------------------------------- client side

class Session:
    """A helper's connection to a coordinator (or an invitation), pinned to the certificate
    seen in the handshake. One per thread: ``clone`` shares the authentication."""

    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.fingerprint: Optional[str] = None
        self.token: Optional[str] = None
        self._conn: Optional[http.client.HTTPSConnection] = None

    def clone(self) -> "Session":
        other = Session(self.host, self.port, self.timeout)
        other.fingerprint, other.token = self.fingerprint, self.token
        return other

    def _connection(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE    # trust comes from the proofs and the pin
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=context,
                                               blocksize=CHUNK)
            conn.connect()
            seen = _fingerprint(conn)
            if self.fingerprint is None:
                self.fingerprint = seen
            elif seen != self.fingerprint:
                conn.close()
                raise ProtocolError("the coordinator's certificate changed")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _request(self, method: str, path: str, body=None, headers=None) -> http.client.HTTPResponse:
        headers = dict(headers or {})
        if self.token:
            headers["X-Zeit-Session"] = self.token
        retry = body is None or isinstance(body, (bytes, str))
        for attempt in range(2):
            try:
                conn = self._connection()
                conn.request(method, path, body=body, headers=headers)
                return conn.getresponse()
            except (http.client.HTTPException, ConnectionError, ssl.SSLError, socket.timeout, OSError):
                self.close()
                if attempt or not retry:
                    raise
        raise AssertionError("unreachable")

    def call(self, method: str, path: str, data: Optional[dict] = None, body=None,
             headers: Optional[dict] = None) -> Tuple[int, dict]:
        if data is not None:
            body = json.dumps(data).encode()
            headers = {**(headers or {}), "Content-Type": "application/json"}
        response = self._request(method, path, body=body, headers=headers)
        raw = response.read()
        try:
            reply = json.loads(raw) if raw else {}
        except ValueError:
            reply = {"error": raw[:200].decode("utf-8", "replace")}
        return response.status, reply

    def download(self, path: str, dest, progress: Optional[Callable[[int], None]] = None) -> int:
        response = self._request("GET", path)
        if response.status != 200:
            raw = response.read()
            raise ProtocolError(f"GET {path}: HTTP {response.status} {raw[:200]!r}")
        size = int(response.headers.get("Content-Length") or -1)
        got = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress is not None:
                    progress(got)
        if size >= 0 and got != size:
            self.close()
            raise ProtocolError(f"GET {path}: {got} of {size} bytes")
        return got

    def handshake(self, key_for: Callable[[Optional[bytes]], bytes], info: dict, mode: str = "session") -> dict:
        """Prove knowledge of the key both ways; ``key_for(salt)`` gives the key (the salt
        is only sent while pairing). Returns the coordinator's welcome."""
        ours = secrets.token_hex(32)
        status, hello = self.call("POST", "/hello", {"nonce": ours, "mode": mode})
        if status != 200:
            raise ProtocolError(hello.get("error") or f"hello: HTTP {status}")
        if hello.get("protocol") != _group.PROTOCOL:
            raise ProtocolError(f"the other computer speaks protocol {hello.get('protocol')}, this one "
                                f"{_group.PROTOCOL}: update zeit on both")
        salt = bytes.fromhex(hello["salt"]) if hello.get("salt") else None
        key = key_for(salt)
        theirs = str(hello.get("nonce", ""))
        expected = _group.proof(key, "server", ours, theirs, self.fingerprint)
        if not _group.same(expected, hello.get("proof", "")):
            raise ProtocolError("the other computer is not in this group" if mode == "session"
                                else "wrong invitation code")
        status, welcome = self.call("POST", "/auth", {
            "nonce": theirs, "proof": _group.proof(key, "helper", ours, theirs, self.fingerprint), "info": info})
        if status != 200:
            raise ProtocolError(welcome.get("error") or f"auth: HTTP {status}")
        self.token = welcome.get("token")
        return welcome


def _fingerprint(conn: http.client.HTTPSConnection) -> str:
    import hashlib

    der = conn.sock.getpeercert(binary_form=True)
    if not der:
        raise ProtocolError("no certificate")
    return hashlib.sha256(der).hexdigest()


# ---------------------------------------------------------------------------- discovery

def _broadcast_targets() -> List[str]:
    targets = ["255.255.255.255"]
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                targets.append(ip.rsplit(".", 1)[0] + ".255")    # the /24 of every interface
    except OSError:
        pass
    return list(dict.fromkeys(targets))


def find(message: dict, accept: Callable[[dict], bool], *, timeout: Optional[float] = None,
         interval: float = 2.0, stop: Optional[threading.Event] = None,
         port: int = _group.UDP_PORT, targets: Optional[List[str]] = None) -> Optional[Tuple[str, dict]]:
    """Broadcast ``message`` every ``interval`` seconds until a reply that ``accept``s
    arrives: (the replier's address, the reply). None on timeout or stop."""
    payload = json.dumps({"zeit": _group.PROTOCOL, **message}).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", 0))
    sock.settimeout(0.25)
    start, last = time.monotonic(), -interval
    try:
        while not (stop is not None and stop.is_set()):
            now = time.monotonic()
            if timeout is not None and now - start > timeout:
                return None
            if now - last >= interval:
                for target in targets or _broadcast_targets():
                    try:
                        sock.sendto(payload, (target, port))
                    except OSError:
                        pass
                last = now
            try:
                data, addr = sock.recvfrom(4096)
            except (socket.timeout, ConnectionResetError):
                continue
            except OSError:
                continue
            try:
                reply = json.loads(data)
            except ValueError:
                continue
            if isinstance(reply, dict) and reply.get("zeit") == _group.PROTOCOL and accept(reply):
                return addr[0], reply
    finally:
        sock.close()
    return None


class Responder:
    """Listens to the broadcasts on the UDP port; ``answer(message, address)`` returns the
    reply (sent back to the sender) or None."""

    def __init__(self, answer: Callable[[dict, Tuple[str, int]], Optional[dict]], port: int = _group.UDP_PORT,
                 host: str = ""):
        self._answer = answer
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="zeit-tiles-discovery", daemon=True)

    def start(self) -> "Responder":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(4096)
            except (socket.timeout, ConnectionResetError):
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            try:
                message = json.loads(data)
            except ValueError:
                continue
            if not isinstance(message, dict) or message.get("zeit") != _group.PROTOCOL:
                continue
            reply = self._answer(message, addr)
            if reply is not None:
                try:
                    self._sock.sendto(json.dumps({"zeit": _group.PROTOCOL, **reply}).encode(), addr)
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2)
