"""zeit.plot's interactive viewer: the Session that serves the browser, and the widget."""
import zlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._plot._session import Session

anywidget = pytest.importorskip("anywidget")


def _cube(n=5, h=40, w=60, y_up=False):
    data = np.arange(n * h * w, dtype=np.float32).reshape(n, h, w) % 997
    y = np.arange(h) + 0.5
    return xr.DataArray(data, dims=("time", "y", "x"), name="ndvi",
                        coords={"time": pd.date_range("2000-01-01", periods=n, freq="YS"),
                                "y": y if y_up else y[::-1], "x": np.arange(w) + 0.5}).rio.write_crs(32721)


def _codes(session, buffers):
    meta = session._meta({})[0]
    raw = [zlib.decompress(b) if meta["compressed"] else b for b in buffers]
    return [np.frombuffer(r, np.uint8) for r in raw]


def test_meta():
    session = Session(_cube(), max_size=30)
    meta, (lut,) = session.handle({"type": "meta"})
    assert meta["n"] == 5 and meta["labels"][0] == "2000"
    assert meta["step"] == 2 and (meta["width"], meta["height"]) == (30, 20)
    assert (meta["full_width"], meta["full_height"]) == (60, 40)
    assert meta["extent"] == [0.0, 60.0, 0.0, 40.0] and meta["crs"] == "EPSG:32721"
    assert meta["style"]["kind"] == "continuous" and len(lut) == 1024
    assert meta["frame_bytes"] == 600 and meta["variables"] == []


def test_frames_match_the_style_encoding():
    cube = _cube()
    session = Session(cube, max_size=30)
    content, buffers = session.handle({"type": "frames", "start": 1, "count": 3})
    assert content == {"start": 1, "count": 3} and len(buffers) == 3
    codes = _codes(session, buffers)
    expected = session.style.encode(cube.values[2, ::2, ::2], session.nodata).ravel()
    np.testing.assert_array_equal(codes[1], expected)
    content, buffers = session.handle({"type": "frames", "start": 4, "count": 10})   # clipped at the end
    assert content["count"] == 1 and len(buffers) == 1


def test_rows_go_top_first_when_y_increases():
    down = Session(_cube(), max_size=None or 100, compress=False)
    up = Session(_cube(y_up=True), max_size=100, compress=False)
    a = np.frombuffer(down.handle({"type": "frames", "start": 0, "count": 1})[1][0], np.uint8).reshape(40, 60)
    b = np.frombuffer(up.handle({"type": "frames", "start": 0, "count": 1})[1][0], np.uint8).reshape(40, 60)
    np.testing.assert_array_equal(a, b[::-1])   # same values; the y-up cube is sent flipped (north up)
    assert up._meta({})[0]["extent"] == [0.0, 60.0, 0.0, 40.0]


def test_detail_window():
    cube = _cube()
    session = Session(cube, max_size=30, compress=False)
    content, (buf,) = session.handle({"type": "detail", "index": 2, "x0": 10, "y0": 4, "x1": 30, "y1": 20,
                                      "max_px": 100})
    assert content["step"] == 1 and (content["width"], content["height"]) == (20, 16)
    codes = np.frombuffer(buf, np.uint8).reshape(16, 20)
    np.testing.assert_array_equal(codes, session.style.encode(cube.values[2, 4:20, 10:30], session.nodata))
    # a window that would come out coarser than the preview: nothing to add
    big = Session(_cube(h=400, w=600), max_size=300)
    assert big.handle({"type": "detail", "index": 0, "x0": 0, "y0": 0, "x1": 600, "y1": 400,
                       "max_px": 10})[0] == {"empty": True}


def test_select_variable_and_errors():
    events = xr.Dataset({"yod": (("y", "x"), np.array([[0, 1990], [2005, 2020]], dtype=np.uint16)),
                         "magnitude": (("y", "x"), np.array([[np.nan, 1.0], [2.0, 3.0]]))},
                        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]})
    session = Session(events)
    meta, _ = session.handle({"type": "meta"})
    assert meta["variables"] == ["yod", "magnitude"] and meta["var"] == "yod"
    assert meta["style"]["kind"] == "years"
    meta, _ = session.handle({"type": "select", "var": "magnitude"})
    assert meta["var"] == "magnitude" and meta["style"]["kind"] == "continuous"
    with pytest.raises(ValueError, match="unknown request"):
        session.handle({"type": "nope"})
    with pytest.raises(ValueError, match="maps"):
        Session(pd.Series([1.0, 2.0]))


def test_widget_routes_requests_and_errors():
    from zeit._plot._widget import make_widget

    widget = make_widget(Session(_cube(), max_size=30), height=300, fps=12)
    assert (widget.height, widget.fps) == (300, 12)
    assert "zv-root" in widget._css_text
    sent = []
    widget.send = lambda content, buffers=None: sent.append((content, buffers))
    widget._on_request(widget, {"type": "request", "id": 7, "request": {"type": "frames", "start": 0, "count": 2}}, [])
    assert sent[-1][0]["id"] == 7 and sent[-1][0]["content"]["count"] == 2 and len(sent[-1][1]) == 2
    widget._on_request(widget, {"type": "request", "id": 8, "request": {"type": "bogus"}}, [])
    assert sent[-1][0]["id"] == 8 and "unknown request" in sent[-1][0]["error"]
    widget._on_request(widget, {"type": "something else"}, [])
    assert len(sent) == 2


def test_plot_returns_a_viewer_or_a_figure(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from zeit._plot import _widget

    cube = _cube()
    assert type(zeit.plot(cube)).__name__ == "Figure"                  # outside a notebook, under tests
    monkeypatch.setattr(_widget, "in_notebook", lambda: True)
    assert type(zeit.plot(cube, static=False)).__name__ == "Viewer"
    assert type(zeit.plot(cube)).__name__ == "Viewer"                  # inside one
    assert type(zeit.plot(cube[:, 2, 3])).__name__ == "Figure"         # a pixel's series is a figure
