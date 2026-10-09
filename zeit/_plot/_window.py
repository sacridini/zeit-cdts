"""The standalone viewer: the same Session and viewer, carried over local HTTP to a window.

Outside a notebook, ``show_window`` serves the viewer from a small HTTP server on
127.0.0.1 and opens it in a native window (pywebview, when installed) or the browser.

Routes, all under a random token (``http://127.0.0.1:<port>/<token>/``), so other local
users and web pages cannot read the data:

- ``GET /``: the page, which imports ``viewer.js`` and mounts the viewer full-window;
- ``GET /viewer.js``, ``GET /viewer.css``: the viewer itself;
- ``POST /request`` (``{"request": {...}}``): one Session request. The reply is binary: a
  uint32 (little-endian) length, a UTF-8 JSON header (``{"content": ...}`` or
  ``{"error": ...}``, plus ``"sizes"`` of the buffers), then the buffers back to back;
- ``POST /ping`` (every 2 s while the page is open) and ``POST /close`` (a beacon when it is
  closed): how a blocking ``show_window`` knows when to return.
"""

import atexit
import importlib.util
import json
import math
import pathlib
import secrets
import struct
import subprocess
import sys
import threading
import time
import weakref
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Set, Union

from ._session import Session
from ._widget import handle_safely

HERE = pathlib.Path(__file__).parent

PING_TIMEOUT = 10.0      # a visible page that stops pinging for this long is gone (crashed, killed)
HIDDEN_TIMEOUT = 180.0   # browsers throttle the timers of hidden tabs to about one a minute
GRACE = 2.0              # no page for this long: closed (a reload comes back sooner)

_OPEN_WINDOWS: "weakref.WeakSet[Window]" = weakref.WeakSet()


def is_interactive() -> bool:
    """True in an interactive interpreter (``python -i``, the REPL, IPython, Jupyter)."""
    if sys.flags.interactive or hasattr(sys, "ps1"):
        return True
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def has_pywebview() -> bool:
    return importlib.util.find_spec("webview") is not None


# ---------------------------------------------------------------------- presence
class Presence:
    """Which pages are open, from their pings and close beacons."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pages: Dict[str, Any] = {}   # page id -> (last ping, hidden)
        self.left: Set[str] = set()       # closed pages (a late ping must not bring them back)
        self.seen = False
        self.empty_since: Optional[float] = None

    def ping(self, page: str, hidden: bool = False) -> None:
        with self.lock:
            if page in self.left:
                return
            self.pages[page] = (time.monotonic(), bool(hidden))
            self.seen = True

    def leave(self, page: str) -> None:
        with self.lock:
            self.pages.pop(page, None)
            self.left.add(page)

    def gone(self) -> bool:
        """True once a page was seen and none has been open for GRACE seconds."""
        now = time.monotonic()
        with self.lock:
            if not self.seen:
                return False
            for page, (last, hidden) in list(self.pages.items()):
                if now - last > (HIDDEN_TIMEOUT if hidden else PING_TIMEOUT):
                    del self.pages[page]
            if self.pages:
                self.empty_since = None
                return False
            if self.empty_since is None:
                self.empty_since = now
            return now - self.empty_since >= GRACE


# ---------------------------------------------------------------------- HTTP
def _jsonable(value: Any) -> Any:
    """JSON-safe copy: numpy scalars and arrays as Python values, NaN and inf as null."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):   # numpy scalar or array
        return _jsonable(value.tolist())
    return value


def encode_reply(reply: Dict[str, Any], buffers) -> list:
    """The binary framing of a reply: [length + JSON header, buffer, buffer...]."""
    views = [memoryview(b).cast("B") if not isinstance(b, (bytes, bytearray)) else b for b in buffers]
    header = dict(_jsonable(reply), sizes=[len(v) for v in views])
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return [struct.pack("<I", len(head)) + head, *views]


def decode_reply(data: bytes):
    """(header, buffers) from encode_reply's bytes (the browser does the same in JS)."""
    (n,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + n].decode("utf-8"))
    at, buffers = 4 + n, []
    for size in header.pop("sizes"):
        buffers.append(data[at:at + size])
        at += size
    return header, buffers


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "zeit"
    timeout = 60   # idle keep-alive connections are dropped

    @property
    def window(self) -> "Window":
        return self.server.window   # type: ignore[attr-defined]

    def log_message(self, format, *args):   # noqa: A002 - quiet
        pass

    def _route(self) -> Optional[str]:
        """The path after the token, or None (and a 403/404 sent) when it is not ours."""
        if self.window.closed:
            self._send(503, b"closed", "text/plain; charset=utf-8", close=True)
            return None
        path = self.path.split("?", 1)[0]
        prefix = f"/{self.window.token}"
        if path == prefix:
            self.send_response(301)
            self.send_header("Location", prefix + "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if not path.startswith(prefix + "/"):
            self._send(403, b"forbidden", "text/plain; charset=utf-8")
            return None
        return path[len(prefix):]

    def _send(self, code: int, body, content_type: str, *, close: bool = False) -> None:
        parts = body if isinstance(body, list) else [body]
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(sum(len(p) for p in parts)))
        self.send_header("Cache-Control", "no-store")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        for p in parts:
            self.wfile.write(p)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self):  # noqa: N802
        route = self._route()
        if route is None:
            return
        files = self.window.files()
        if route in ("/", "/index.html"):
            self._send(200, self.window.page().encode("utf-8"), "text/html; charset=utf-8")
        elif route in files:
            body, content_type = files[route]
            self._send(200, body, content_type)
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):  # noqa: N802
        route = self._route()
        body = self._body()
        if route is None:
            return
        try:
            message = json.loads(body.decode("utf-8")) if body else {}
        except ValueError:
            self._send(400, b"bad JSON", "text/plain; charset=utf-8")
            return
        if route == "/request":
            with self.window.lock:   # a Session is not thread-safe
                reply, buffers = handle_safely(self.window.session, message.get("request") or {})
            self._send(200, encode_reply(reply, buffers), "application/octet-stream")
        elif route == "/ping":
            self.window.presence.ping(str(message.get("page")), bool(message.get("hidden")))
            self._send(200, b"{}", "application/json")
        elif route == "/close":
            self.window.presence.leave(str(message.get("page")))
            self._send(200, b"{}", "application/json")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return   # the page went away mid-request
        super().handle_error(request, client_address)


# ---------------------------------------------------------------------- the page
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="viewer.css">
<style>
  :root { color-scheme: light dark; --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de;
          --accent: #1976d2; --warn-bg: #fff4e5; --warn-fg: #8a4b00; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #16181c; --fg: #e6e8eb; --muted: #9aa1aa; --line: #3a3f47; --accent: #5aa9f0;
            --warn-bg: #3b2a12; --warn-fg: #ffcf8a; }
  }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); }
  body { overflow: hidden; }
  #app { box-sizing: border-box; height: 100%; padding: 12px 16px; }
  body .zv-root { --zv-fg: var(--fg); --zv-muted: var(--muted); --zv-bg: var(--bg); --zv-line: var(--line);
                  --zv-accent: var(--accent); }
  .zv-root:focus-visible { outline: none; }
  #status { position: fixed; left: 16px; right: 16px; bottom: 12px; display: none; padding: 8px 12px;
            border-radius: 6px; background: var(--warn-bg); color: var(--warn-fg);
            font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
</style>
</head>
<body>
<main id="app"></main>
<div id="status" role="status"></div>
<script type="module">
import { mount } from "./viewer.js";

const app = document.getElementById("app");
const status = document.getElementById("status");
const say = (text) => { status.textContent = text; status.style.display = text ? "block" : "none"; };
const options = __OPTIONS__;

// ---- transport: one POST per request, binary reply (see zeit/_plot/_window.py)
const transport = {
  async request(req) {
    let r;
    try {
      r = await fetch("request", { method: "POST", headers: { "Content-Type": "application/json" },
                                   body: JSON.stringify({ request: req }) });
    } catch (err) {
      say("Lost the connection to Python: the session has ended.");
      throw err;
    }
    if (!r.ok) throw new Error(`zeit: HTTP ${r.status}`);
    const data = await r.arrayBuffer();
    const n = new DataView(data).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, n)));
    if (header.error) {
      if (header.trace) console.error(header.trace);
      throw new Error(header.error);
    }
    let at = 4 + n;
    const buffers = header.sizes.map((size) => { const b = new Uint8Array(data, at, size); at += size; return b; });
    return { content: header.content, buffers };
  },
};

// ---- presence: Python's blocking show_window returns when the page is closed
const newId = () => (crypto.randomUUID && crypto.randomUUID()) || String(Math.random()).slice(2);
let page = newId(), gone = false;
const ping = () => gone ? null : fetch("ping", { method: "POST",
                                                  body: JSON.stringify({ page, hidden: document.hidden }) })
  .then((r) => { if (r.ok) say(""); else say("The Python session has ended."); })
  .catch(() => say("Lost the connection to Python: the session has ended."));
ping();
setInterval(ping, 2000);
document.addEventListener("visibilitychange", ping);
addEventListener("pagehide", () => { gone = true; navigator.sendBeacon("close", JSON.stringify({ page })); });
// back from the back/forward cache: a new page as far as Python is concerned
addEventListener("pageshow", (e) => { if (e.persisted) { page = newId(); gone = false; ping(); } });
addEventListener("unhandledrejection", (e) => say(String((e.reason && e.reason.message) || e.reason)));

// ---- the viewer, filling the window
const viewer = mount(app, transport, { fps: options.fps });
if (options.height) {
  app.style.setProperty("--zv-height", `${options.height}px`);
  document.body.style.overflow = "auto";
} else {
  let last = 0;
  const fill = () => {
    const stage = viewer.stage.getBoundingClientRect().height;
    const rest = viewer.root.getBoundingClientRect().height - stage;   // title, slider, legend...
    const pad = parseFloat(getComputedStyle(app).paddingTop) + parseFloat(getComputedStyle(app).paddingBottom);
    const height = Math.max(160, Math.floor(innerHeight - pad - rest));
    if (Math.abs(height - last) >= 1) { last = height; app.style.setProperty("--zv-height", `${height}px`); }
  };
  fill();
  addEventListener("resize", fill);
  new ResizeObserver(fill).observe(viewer.root);
}
viewer.root.focus();
window.zeitViewer = viewer;
</script>
</body>
</html>
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------- window
_NATIVE = """
import sys, webview
url, title, width, height = sys.argv[1:5]
webview.create_window(title, url, width=int(width), height=int(height))
webview.start()
"""


class Window:
    """A viewer served on 127.0.0.1 and shown in a native window or the browser.

    ``url`` is the page's address (with its access token); ``close()`` stops the server (and
    the native window); ``wait()`` blocks until the page is closed, then stops the server.
    """

    def __init__(self, session: Session, *, height: Optional[int] = None, fps: int = 8, port: int = 0,
                 title: Optional[str] = None, kind: str = "viewer"):
        if kind not in ("viewer", "interpret"):
            raise ValueError(f"kind must be 'viewer' or 'interpret', got {kind!r}")
        self.kind = kind
        self.session = session
        self.height = height
        self.fps = fps
        self.title = title or session.title or session.frames.name or "zeit.plot"
        self.token = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        self.presence = Presence()
        self.closed = False
        self.process: Optional[subprocess.Popen] = None
        self._server = _Server(("127.0.0.1", port), _Handler)
        self._server.window = self   # type: ignore[attr-defined]
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/{self.token}/"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2},
                                        name=f"zeit.plot window {self.port}", daemon=True)
        self._thread.start()
        _OPEN_WINDOWS.add(self)

    def page(self) -> str:
        options = {"fps": int(self.fps), "height": int(self.height) if self.height else None}
        page = PAGE if self.kind == "viewer" else PAGE.replace(
            'import { mount } from "./viewer.js";', 'import { mountInterpret as mount } from "./interpret.js";').replace(
            '<link rel="stylesheet" href="viewer.css">',
            '<link rel="stylesheet" href="viewer.css"><link rel="stylesheet" href="interpret.css">')
        return page.replace("__TITLE__", _escape(str(self.title))).replace("__OPTIONS__", json.dumps(options))

    def files(self) -> Dict[str, Any]:
        """The files the page loads: route -> (bytes, content type)."""
        js, css = "text/javascript; charset=utf-8", "text/css; charset=utf-8"
        out = {"/viewer.js": ((HERE / "viewer.js").read_bytes(), js),
               "/viewer.css": ((HERE / "viewer.css").read_bytes(), css)}
        if self.kind == "interpret":
            from ._widget import interpret_bundle
            out["/interpret.js"] = (interpret_bundle().encode("utf-8"), js)
            out["/interpret.css"] = ((HERE / "interpret.css").read_bytes(), css)
        return out

    # ------------------------------------------------------------------ showing
    def open(self, how: Union[bool, str] = True) -> None:
        """Show the page: ``"native"`` (pywebview), ``"browser"``, or True for native if available."""
        if how == "native" or (how is True and has_pywebview()):
            if self._open_native():
                return
            if how == "native":
                raise ImportError("a native window needs pywebview: pip install pywebview")
        webbrowser.open(self.url)

    def _open_native(self) -> bool:
        """pywebview in a child process: its GUI loop wants a main thread of its own."""
        if not has_pywebview():
            return False
        width, height = 1100, (self.height + 190) if self.height else 780
        try:
            self.process = subprocess.Popen([sys.executable, "-c", _NATIVE, self.url, str(self.title), str(width),
                                             str(height)])
        except OSError:
            return False
        threading.Thread(target=self._watch_native, args=(self.process,), daemon=True).start()
        return True

    def _watch_native(self, process: subprocess.Popen) -> None:
        """If pywebview fails before showing the page, fall back to the browser."""
        code = process.wait()
        if code != 0 and not self.presence.seen and not self.closed:
            webbrowser.open(self.url)

    # ------------------------------------------------------------------ lifetime
    def wait(self) -> None:
        """Block until the page is closed (or Ctrl+C), then stop the server."""
        try:
            while not self.closed:
                time.sleep(0.2)
                if self.presence.gone():
                    break
                process = self.process   # the native window was closed (not a failed start)
                if process is not None and process.poll() is not None and (
                        process.returncode == 0 or self.presence.seen):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Stop the server and close the native window, if any."""
        if self.closed:
            return
        self.closed = True
        self._server.shutdown()
        self._server.server_close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        _OPEN_WINDOWS.discard(self)

    def __enter__(self) -> "Window":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else self.url
        return f"<zeit.plot window {self.title!r}: {state}>"

    def _repr_html_(self) -> str:
        if self.closed:
            return f"<code>zeit.plot window {_escape(str(self.title))} (closed)</code>"
        return f'<a href="{self.url}" target="_blank">zeit.plot window: {_escape(str(self.title))}</a>'


@atexit.register
def _close_all() -> None:
    for window in list(_OPEN_WINDOWS):
        window.close()


def show_window(session: Session, *, height: Optional[int] = None, fps: int = 8, block: Optional[bool] = None,
                open: Union[bool, str] = True, port: int = 0, title: Optional[str] = None,  # noqa: A002
                kind: str = "viewer") -> Window:
    """Show a Session's viewer in its own window, outside a notebook.

    Parameters
    ----------
    session
        What to show.
    height
        Map height in pixels; by default the map fills the window.
    fps
        Playback speed.
    block
        Wait until the window is closed, like ``plt.show()``. By default: yes when running a
        script, no in an interactive interpreter (where the window stays up until
        ``Window.close()`` or the interpreter exits).
    open
        Show the page: True for a native window (with pywebview installed) or else the
        browser; ``"native"``, ``"browser"``; False to only serve it (see ``Window.url``).
    port
        Port on 127.0.0.1 (0: any free port).
    title
        Window title (default: the plot's title or variable name).
    kind
        ``"viewer"`` (``zeit.plot``) or ``"interpret"`` (``zeit.interpret``).

    Returns
    -------
    Window
        The served page: ``url``, ``close()``, ``wait()``.
    """
    window = Window(session, height=height, fps=fps, port=port, title=title, kind=kind)
    if open:
        window.open(open)
    if block is None:
        block = not is_interactive()
    if block:
        window.wait()
    return window
