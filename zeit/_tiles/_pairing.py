"""Bringing a computer into the group: an invitation with a short code on a member, the
code typed on the newcomer. Also listing the helpers that are waiting for work."""

import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import _group, _net

MAX_WRONG = 10    # wrong codes before the invitation closes


class Invitation:
    """Open while it runs: hands the group's secret to computers that know the code."""

    def __init__(self, *, minutes: float = 10.0, port: int = _group.TCP_PORT, discovery: bool = True,
                 code: Optional[str] = None, out: Optional[Callable[[str], None]] = None, host: str = ""):
        self.secret = _group.ensure_secret()
        self.code = code or _group.new_code()
        self.deadline = time.monotonic() + minutes * 60
        self.out = out or (lambda text: print(text, flush=True))
        self.joined: List[str] = []
        self.closed = threading.Event()
        self._wrong = 0
        self._lock = threading.Lock()
        self._tmp = tempfile.TemporaryDirectory(prefix="zeit-invite-")
        salt = secrets.token_bytes(16)
        cert, key, fingerprint = _group.certificate(Path(self._tmp.name))
        self._handshakes = _net.Handshakes(_group.code_key(self.code, salt), fingerprint, salt)
        self._server = _net.Server(self, port, cert, key, host=host).start()
        self.port = self._server.port
        self._responder = None
        if discovery:
            try:
                self._responder = _net.Responder(self._answer).start()
            except OSError as e:
                self.out(f"Discovery is off (UDP port {_group.UDP_PORT}: {e}); join with --connect.")

    def _open(self) -> bool:
        return not self.closed.is_set() and time.monotonic() < self.deadline

    def _answer(self, message: dict, addr) -> Optional[dict]:
        if message.get("kind") == "join" and self._open():
            return {"kind": "invite", "port": self.port, "name": _group.machine_name()}
        return None

    def handle(self, handler, method: str, path: str) -> None:
        route = path.split("?", 1)[0].rstrip("/")
        if not self._open():
            handler.read_body()
            handler.send_json(410, {"error": "the invitation is closed: ask for a new code"})
            return
        if method == "POST" and route == "/hello":
            handler.send_json(200, self._handshakes.hello(handler.read_json()))
        elif method == "POST" and route == "/auth":
            body = handler.read_json()
            if not self._handshakes.auth(body):
                with self._lock:
                    self._wrong += 1
                    if self._wrong >= MAX_WRONG:
                        self.closed.set()
                        self.out("Too many wrong codes: the invitation is closed.")
                handler.send_json(403, {"error": "wrong invitation code"})
                return
            info = body.get("info") if isinstance(body.get("info"), dict) else {}
            name = "".join(c for c in str(info.get("name") or "a computer") if c.isprintable())[:64]
            with self._lock:
                self.joined.append(name)
            self.out(f"+ {name} joined the group")
            handler.send_json(200, {"secret": self.secret.hex(), "name": _group.machine_name()})
        else:
            handler.send_json(404, {"error": f"no {method} {route}"})

    def wait(self, stop: Optional[threading.Event] = None) -> List[str]:
        try:
            while self._open() and not (stop is not None and stop.is_set()):
                time.sleep(0.5)
        finally:
            self.close()
        return list(self.joined)

    def close(self) -> None:
        self.closed.set()
        if self._responder is not None:
            self._responder.close()
            self._responder = None
        if self._server is not None:
            self._server.close()
            self._server = None
        self._tmp.cleanup()


def join(code: str, *, connect: Optional[str] = None, timeout: float = 120.0,
         out: Optional[Callable[[str], None]] = None, stop: Optional[threading.Event] = None) -> bytes:
    """The group's secret, from the invitation that knows ``code``."""
    from ._worker import _target, info, default_cores

    out = out or (lambda text: print(text, flush=True))
    _group.normalize_code(code)          # a typo fails here, not after the search
    if connect:
        host, port = _target(connect)
    else:
        out("Looking for the invitation on the local network...")
        found = _net.find({"kind": "join"}, lambda r: r.get("kind") == "invite", timeout=timeout, stop=stop)
        if found is None:
            raise RuntimeError("no invitation found on the local network. Is `zeit helpers invite` running on the "
                               "other computer? If it is, join with: zeit worker --join <code> --connect <its address>")
        host, reply = found
        port = int(reply.get("port") or _group.TCP_PORT)
    session = _net.Session(host, port)
    try:
        welcome = session.handshake(lambda salt: _group.code_key(code, salt or b""), info(default_cores()),
                                    mode="pair")
    finally:
        session.close()
    out(f"Paired with {welcome.get('name', host)} ({host}).")
    return bytes.fromhex(welcome["secret"])


def list_helpers(seconds: float = 5.0) -> List[Dict]:
    """The helpers of this computer's group that are waiting for work (busy ones are not
    broadcasting)."""
    secret = _group.load_secret()
    if secret is None:
        raise RuntimeError("this computer is not in a group yet: run `zeit helpers invite`")
    gid = _group.group_id(secret)
    seen: Dict[str, Dict] = {}

    def answer(message, addr):
        if message.get("kind") == "helper" and message.get("group") == gid:
            seen[f"{addr[0]}|{message.get('name')}"] = {"name": message.get("name"), "address": addr[0],
                                                        "cores": message.get("cores"),
                                                        "version": message.get("version")}
        return None

    responder = _net.Responder(answer).start()
    try:
        time.sleep(seconds)
    finally:
        responder.close()
    return sorted(seen.values(), key=lambda h: str(h["name"]))
