"""zeit.plot's embedding mode: the vectors travel to the browser, which draws the
principal components, the similarity to the pixel under the cursor and the change between
years on the GPU."""
import json
import math
import zlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from zeit._embedding_tools import pca_fit
from zeit._plot import _embeddings
from zeit._plot._session import Session

NB = 64


def _cube(n=4, h=30, w=40, nb=NB, y_up=False, seed=0):
    """Two kinds of embeddings (left and right halves), a patch that switches in the third
    year, and a pixel without an embedding."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0, 1, nb), rng.normal(0, 1, nb)
    data = np.empty((n, nb, h, w), dtype=np.float32)
    for t in range(n):
        kind = np.where(np.arange(w) < w // 2, 0, 1)[None, :].repeat(h, 0)
        if t >= 2:
            kind[2:6, 2:6] = 1
        base = np.where(kind[None] == 0, a[:, None, None], b[:, None, None])
        data[t] = base + rng.normal(0, 0.1, (nb, h, w))
    data[:, :, 0, w - 1] = np.nan
    y = 8900000 - 5 - 10 * np.arange(h)
    if y_up:
        y, data = y[::-1], data[:, :, ::-1].copy()
    da = xr.DataArray(data, dims=("time", "band", "y", "x"),
                      coords={"time": pd.DatetimeIndex([pd.Timestamp(2017 + t, 1, 1) for t in range(n)]),
                              "band": [f"A{i:02d}" for i in range(nb)], "y": y, "x": 500000 + 5 + 10 * np.arange(w)})
    da = da.rio.write_crs(32720)
    da.attrs.update({"embedding_source": "tessera", "embedding_version": "1.1", "embedding_variant": "dclimate"})
    return da


def _years(session, start, count):
    content, buffers = session.handle({"type": "frames", "start": start, "count": count})
    meta = session.handle({"type": "meta"})[0]
    shape = (meta["embedding"]["layers"], meta["height"], meta["width"], 4)
    return [np.frombuffer(zlib.decompress(b), np.int8).reshape(shape) for b in buffers]


def _vector(q, row, col, nb=NB):
    return q[:, row, col, :].reshape(-1)[:nb]


def test_embedding_mode_and_meta():
    cube = _cube()
    s = Session(cube)
    meta, buffers = s.handle({"type": "meta"})
    e = meta["embedding"]
    assert s.emb is not None and e["dims"] == NB and e["layers"] == NB // 4 and len(e["scale"]) == NB
    assert e["source"] == "TESSERA (1.1, dclimate)" and e["is_time"] and len(e["x"]) == 4
    assert np.array(e["pca"]["weights"]).shape == (6, NB) and e["pca"]["explained"][0] > 0.5
    assert len(buffers) == 3 and all(len(b) == 1024 for b in buffers)   # the map's LUT, similarity, change
    assert meta["frame_bytes"] == meta["width"] * meta["height"] * NB and meta["rgb"]
    json.dumps(meta)
    # one dimension, no colour, or not embeddings: the usual viewer
    for other in (Session(cube, band="A03"), Session(cube, rgb=False)):
        assert other.emb is None and other.handle({"type": "meta"})[0]["embedding"] is None
    plain = cube.isel(band=0, drop=True).rename("ndvi")
    plain.attrs = {}
    assert Session(plain).handle({"type": "meta"})[0]["embedding"] is None


def test_vectors_round_trip_layout_and_missing_pixels():
    cube = _cube()
    s = Session(cube)
    q = _years(s, 1, 2)
    assert len(q) == 2 and q[0].dtype == np.int8
    scale = s.emb.scale
    values = cube.values[1]
    for row, col in [(0, 0), (5, 7), (29, 39), (12, 30)]:
        np.testing.assert_allclose(_vector(q[0], row, col) * scale, values[:, row, col], atol=float(scale.max()) * 0.51)
    # layer k holds dimensions 4k..4k+3
    k, j = 3, 2
    assert q[0][k, 5, 7, j] == np.int8(np.rint(values[4 * k + j, 5, 7] / scale[4 * k + j]))
    # no embedding: 0 everywhere, -128 in the first dimension
    assert q[0][0, 0, 39, 0] == -128 and (_vector(q[0], 0, 39)[1:] == 0).all()
    back = s.emb.dequantize(q[0])
    assert np.isnan(back[39]).all() and np.isfinite(back[40]).all()


def test_rows_go_top_first():
    down, up = Session(_cube()), Session(_cube(y_up=True))
    assert up.flip and not down.flip
    np.testing.assert_array_equal(_years(down, 0, 1)[0], _years(up, 0, 1)[0])


def test_coarser_cells_keep_a_year_within_budget(monkeypatch):
    cube = _cube(h=60, w=80)
    monkeypatch.setattr(_embeddings, "YEAR_BYTES", 60 * 80 * NB / 3.5)
    s = Session(cube)
    meta = s.handle({"type": "meta"})[0]
    assert s.step == 2 and (meta["width"], meta["height"]) == (40, 30)
    content, buffers = s.handle({"type": "frames", "start": 0, "count": 1})
    assert len(zlib.decompress(buffers[0])) == meta["frame_bytes"] <= _embeddings.YEAR_BYTES


def test_detail_carries_the_year_compared_with():
    cube = _cube(h=60, w=80)
    s = Session(cube, max_size=20)
    assert s.step == 4
    content, buffers = s.handle({"type": "detail", "index": 3, "other": 1, "x0": 8, "y0": 4, "x1": 30, "y1": 20,
                                 "max_px": 400})
    assert content["step"] == 1 and content["other"] == 1 and len(buffers) == 2
    h, w = content["height"], content["width"]
    for buffer, year in zip(buffers, (3, 1)):
        q = np.frombuffer(zlib.decompress(buffer), np.int8).reshape(NB // 4, h, w, 4)
        expected = s.emb.quantize(cube.values[year][:, content["y0"]:content["y1"], content["x0"]:content["x1"]])
        np.testing.assert_array_equal(q, expected)
    content, buffers = s.handle({"type": "detail", "index": 3, "x0": 8, "y0": 4, "x1": 30, "y1": 20, "max_px": 400})
    assert content["other"] is None and len(buffers) == 1


def test_components_of_the_visible_area():
    cube = _cube()
    s = Session(cube)
    whole = s.handle({"type": "pca", "window": None})[0]
    assert np.allclose(whole["weights"], s.emb.fit["weights"])
    # nothing sent yet: read from the cube; then from the years the browser has
    window = [10, 10, 20, 30]   # in the left half only: one kind, its components are the noise
    direct = s.handle({"type": "pca", "window": window})[0]
    x = cube.isel(x=slice(10, 20), y=slice(10, 30)).transpose("time", "y", "x", "band").values.reshape(-1, NB)
    expected = pca_fit(x)
    assert abs(np.dot(direct["weights"][0], expected["weights"][0])) > 0.9
    assert direct["explained"][0] < 0.2 and whole["explained"][0] > 0.5
    _years(s, 0, 4)
    cached = s.handle({"type": "pca", "window": window})[0]
    assert abs(cached["explained"][0] - expected["explained"][0]) < 0.02
    with pytest.raises(ValueError, match="no pixel"):
        s.handle({"type": "pca", "window": [50, 50, 60, 60]})
    plain = cube.isel(band=0, drop=True)
    plain.attrs = {}
    with pytest.raises(ValueError, match="embeddings"):
        Session(plain).handle({"type": "pca", "window": None})


def test_static_plots_keep_the_components():
    pytest.importorskip("matplotlib")
    import zeit

    assert zeit.plot(_cube(), static=True, time=["2017"]) is not None


def test_bundles_declare_each_name_once():
    """viewer.js, embeddings.js and interpret.js become one module: a name declared twice at
    the top level is a syntax error that only a browser would show."""
    import re

    from zeit._plot._widget import interpret_bundle, viewer_bundle

    for bundle in (viewer_bundle(), interpret_bundle()):
        names = re.findall(r"^(?:export )?(?:async )?(?:function|class|const|let) (\w+)", bundle, flags=re.M)
        assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})
        assert bundle.count("export default {") == 1


# ---------------------------------------------------------------------- browser
def test_views_in_edge():
    sync_api = pytest.importorskip("playwright.sync_api")
    from zeit._plot._window import show_window

    cube = _cube(h=30, w=40)
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=True, args=["--ignore-gpu-blocklist"])
        except Exception as err:  # noqa: BLE001
            pytest.skip(f"Microsoft Edge is not available: {err}")
        win = show_window(Session(cube), open=False, block=False)
        errors = []
        try:
            page = browser.new_page(viewport={"width": 1000, "height": 700})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(win.url)
            page.wait_for_selector(".zv-emb")
            page.wait_for_function("window.zeitViewer.frames.filter(Boolean).length === 4")
            box = page.locator(".zv-stage .zv-above").bounding_box()
            # hover a pixel of the left half: components first
            px, py = box["width"] * 0.3, box["height"] * 0.5
            page.mouse.move(box["x"] + px, box["y"] + py)
            page.wait_for_function("document.querySelector('.zv-readout').textContent.includes('PC1')")
            # similarity to the pixel under the cursor: 1 there, the shader's value equals numpy's
            page.select_option(".zv-emb select", "similarity")
            page.mouse.move(box["x"] + px + 1, box["y"] + py)
            page.wait_for_function("document.querySelector('.zv-readout').textContent.includes('similarity')")
            cx, cy = page.evaluate("window.zeitViewer.emb.cursor")   # the reference: the cell under the mouse
            col, row = int(cx), int(cy)
            ref = cube.values[0, :, row, col]
            cos = lambda a, b: float(np.dot(a, b) / math.sqrt(np.dot(a, a) * np.dot(b, b)))   # noqa: E731
            for c, r in [(col, row), (35, 20), (3, 3), (10, 25)]:
                gpu = page.evaluate("([c, r]) => window.zeitViewer.emb.probe(c + 0.5, r + 0.5)", [c, r])
                cpu = page.evaluate("([c, r]) => window.zeitViewer.emb.value(c + 0.5, r + 0.5)", [c, r])
                expected = cos(cube.values[0, :, r, c], ref)
                assert abs(gpu - expected) < 5e-3 and abs(cpu - expected) < 5e-3 and abs(gpu - cpu) < 1e-3
            assert page.evaluate("window.zeitViewer.emb.probe(39.5, 0.5)") is None   # no embedding there
            # pin it, page to another year with the reference's year kept: the switched patch drifts away
            page.mouse.click(box["x"] + px + 1, box["y"] + py)
            assert page.evaluate("window.zeitViewer.emb.pinned !== null")
            page.check(".zv-emb input[type=checkbox]")
            page.focus(".zv-root")   # not the panel's controls
            page.keyboard.press("End")
            page.wait_for_function("document.querySelector('.zv-label').textContent === '2020'")
            gpu = page.evaluate("window.zeitViewer.emb.probe(3.5, 3.5)")
            assert abs(gpu - cos(cube.values[3, :, 3, 3], ref)) < 5e-3 and gpu < 0.5
            assert page.evaluate("window.zeitViewer.pixel.series.length") == 2   # cursor and pin
            page.focus(".zv-root")   # not the panel's controls
            page.keyboard.press("Escape")
            assert page.evaluate("window.zeitViewer.emb.pinned") is None
            # change from the previous year: high only on the patch, in its year
            page.select_option(".zv-emb select", "change")
            page.focus(".zv-root")   # not the panel's controls
            page.keyboard.press("ArrowLeft")
            page.wait_for_function("document.querySelector('.zv-label').textContent === '2019'")
            for c, r in [(3, 3), (30, 20)]:
                gpu = page.evaluate("([c, r]) => window.zeitViewer.emb.probe(c + 0.5, r + 0.5)", [c, r])
                assert abs(gpu - (1 - cos(cube.values[2, :, r, c], cube.values[1, :, r, c]))) < 5e-3
            assert page.evaluate("window.zeitViewer.emb.probe(3.5, 3.5)") > 0.5
            page.mouse.move(box["x"] + px, box["y"] + py)
            page.wait_for_function("document.querySelector('.zv-readout').textContent.includes('change')")
            # components of the visible area
            page.select_option(".zv-emb select", "components")
            page.select_option(".zv-emb label select >> nth=1", "visible")
            page.wait_for_function("window.zeitViewer.emb.pcaWindow !== null && window.zeitViewer.emb.pca !== "
                                   "window.zeitViewer.emb.e.pca")
            assert page.locator(".zv-emb-profile svg path").count() >= 1
        finally:
            browser.close()
            win.close()
        assert not errors, errors
