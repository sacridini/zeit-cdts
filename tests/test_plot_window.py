"""zeit.plot's standalone window: the local HTTP transport between a Session and the page."""
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from zeit._plot._session import Session
from zeit._plot._window import Presence, decode_reply, encode_reply, show_window


def _cube(n=5, h=40, w=60):
    data = np.arange(n * h * w, dtype=np.float32).reshape(n, h, w) % 997
    return xr.DataArray(data, dims=("time", "y", "x"), name="ndvi",
                        coords={"time": pd.date_range("2000-01-01", periods=n, freq="YS"),
                                "y": (np.arange(h) + 0.5)[::-1], "x": np.arange(w) + 0.5}).rio.write_crs(32721)


@pytest.fixture
def window():
    win = show_window(Session(_cube(), max_size=30), open=False, block=False)
    yield win
    win.close()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.headers["Content-Type"] == "application/octet-stream" or r.headers["Content-Type"].startswith(
            "application/json")
        return r.read()


def _request(win, request):
    return decode_reply(_post(win.url + "request", {"request": request}))


def test_meta_and_frames_match_the_session(window):
    reference = Session(_cube(), max_size=30)
    for request in ({"type": "meta"}, {"type": "frames", "start": 1, "count": 3},
                    {"type": "detail", "index": 2, "x0": 10, "y0": 4, "x1": 30, "y1": 20, "max_px": 100}):
        header, buffers = _request(window, request)
        content, expected = reference.handle(request)
        assert header["content"] == json.loads(json.dumps(content))
        assert buffers == [bytes(b) for b in expected]


def test_errors_reach_the_page(window):
    header, buffers = _request(window, {"type": "nope"})
    assert "unknown request" in header["error"] and "content" not in header and buffers == []


def test_token_is_required(window):
    base = window.url.rsplit("/", 2)[0]   # http://127.0.0.1:port
    for url in (base + "/", base + "/viewer.js", base + "/wrong-token/viewer.js"):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(url, timeout=10)
        assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base + "/request", {"request": {"type": "meta"}})
    assert err.value.code == 403


def test_page_and_viewer_files(window):
    with urllib.request.urlopen(window.url, timeout=10) as r:
        assert r.headers["Content-Type"] == "text/html; charset=utf-8"
        html = r.read().decode()
    assert 'import { mount } from "./viewer.js"' in html and "<title>ndvi</title>" in html
    with urllib.request.urlopen(window.url + "viewer.js", timeout=10) as r:
        assert r.headers["Content-Type"].startswith("text/javascript")
        assert b"export function mount" in r.read()
    with urllib.request.urlopen(window.url + "viewer.css", timeout=10) as r:
        assert r.headers["Content-Type"].startswith("text/css") and b".zv-stage" in r.read()
    assert window.url.startswith("http://127.0.0.1:") and window.url in repr(window)


def test_close_stops_the_server():
    win = show_window(Session(_cube()), open=False, block=False)
    url = win.url
    urllib.request.urlopen(url, timeout=10).close()
    win.close()
    assert "closed" in repr(win)
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url, timeout=2)
    win.close()   # twice is fine


def test_framing_roundtrip():
    reply = {"content": {"a": np.float32(1.5), "b": [np.int64(2), float("nan")], "c": np.arange(3)}}
    parts = encode_reply(reply, [b"xy", np.arange(3, dtype=np.uint8)])
    header, buffers = decode_reply(b"".join(bytes(p) for p in parts))
    assert header == {"content": {"a": 1.5, "b": [2, None], "c": [0, 1, 2]}}
    assert buffers == [b"xy", b"\x00\x01\x02"]


def test_presence(monkeypatch):
    import zeit._plot._window as w

    clock = [100.0]
    monkeypatch.setattr(w.time, "monotonic", lambda: clock[0])
    p = Presence()
    assert not p.gone()                       # never seen: not closed
    p.ping("a")
    assert not p.gone()
    p.leave("a")
    assert not p.gone()                       # a reload comes back within the grace period
    p.ping("b")
    clock[0] += 5
    assert not p.gone()
    clock[0] += w.PING_TIMEOUT                # stopped pinging: crashed or killed
    assert not p.gone()
    clock[0] += w.GRACE
    assert p.gone()
    p.ping("c", hidden=True)                  # hidden tabs ping rarely
    clock[0] += 60
    assert not p.gone()
    p.leave("c")
    p.ping("c", hidden=True)                  # a ping sent just before the close beacon, arriving after it
    assert "c" not in p.pages


def test_wait_returns_when_the_page_closes(monkeypatch):
    import threading

    import zeit._plot._window as w

    monkeypatch.setattr(w, "GRACE", 0.1)
    win = show_window(Session(_cube()), open=False, block=False)
    _post(win.url + "ping", {"page": "p1", "hidden": False})
    threading.Timer(0.3, lambda: _post(win.url + "close", {"page": "p1"})).start()
    done = threading.Event()
    thread = threading.Thread(target=lambda: (win.wait(), done.set()), daemon=True)
    thread.start()
    assert done.wait(10) and win.closed


# ---------------------------------------------------------------------- browser
def test_end_to_end_in_edge():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=True, args=["--ignore-gpu-blocklist"])
        except Exception as err:  # noqa: BLE001
            pytest.skip(f"Microsoft Edge is not available: {err}")
        win = show_window(Session(_cube()), open=False, block=False)
        errors = []
        try:
            page = browser.new_page(viewport={"width": 900, "height": 640})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(win.url)
            page.wait_for_selector(".zv-root")
            page.wait_for_function("document.querySelector('.zv-label').textContent === '2000'")
            page.wait_for_function("window.zeitViewer.frames.filter(Boolean).length === 5")
            box = page.locator(".zv-stage").bounding_box()
            assert box["height"] > 400   # the map fills the window
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.wait_for_function("document.querySelector('.zv-readout').textContent.includes('≈')")
            page.keyboard.press("ArrowRight")
            page.wait_for_function("document.querySelector('.zv-label').textContent === '2001'")
            assert win.presence.seen and len(win.presence.pages) == 1
            page.close()
            for _ in range(50):   # the close beacon
                if not win.presence.pages:
                    break
                time.sleep(0.1)
            assert not win.presence.pages
        finally:
            browser.close()
            win.close()
        assert not errors


def test_plot_routes_outside_a_notebook(monkeypatch):
    """zeit.plot outside a notebook: a window when a screen is available, a figure otherwise."""
    import matplotlib
    matplotlib.use("Agg")
    import zeit
    from zeit._plot import _window
    import zeit._plot as zplot

    cube = _cube()
    original = zplot.can_open_window
    assert type(zeit.plot(cube)).__name__ == "Figure"                     # under pytest: static
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(zplot, "can_open_window", lambda: True)
    opened = []
    monkeypatch.setattr(_window.Window, "open", lambda self, how=True: opened.append(how))
    window = zeit.plot(cube, block=False)
    try:
        assert type(window).__name__ == "Window" and opened == [True]
        assert window.url.startswith("http://127.0.0.1:")
    finally:
        window.close()
    assert type(zeit.plot(cube, save=None, static=True)).__name__ == "Figure"
    monkeypatch.setenv("ZEIT_PLOT", "static")
    assert original() is False
