"""zeit.interpret: the labelling session (without a browser), saving and resuming, and the page."""
import json
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._plot._interpret import InterpretSession

YEARS = pd.date_range("2000-07-01", periods=12, freq="12MS")


@pytest.fixture(scope="module")
def scene():
    """Four bands, 12 years; two blocks are cleared in 2006 and 2009."""
    rng = np.random.default_rng(0)
    ny, nx = 60, 80
    data = np.empty((len(YEARS), 4, ny, nx), dtype=np.float32)
    cleared = np.zeros((ny, nx), dtype=int)
    cleared[10:35, 20:45] = 2006
    cleared[40:55, 50:75] = 2009
    for t, y in enumerate(YEARS):
        img = np.array([0.03, 0.06, 0.04, 0.35])[:, None, None] + rng.normal(0, 0.01, (4, ny, nx))
        gone = (cleared > 0) & (y.year >= cleared)
        img[:, gone] = np.array([0.08, 0.12, 0.14, 0.22])[:, None] + rng.normal(0, 0.01, (4, int(gone.sum())))
        data[t] = img
    coords = {"time": YEARS, "band": ["blue", "green", "red", "nir"], "y": 9000 - 15 - 30 * np.arange(ny),
              "x": 500000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data, dims=("time", "band", "y", "x"), coords=coords).rio.write_crs(32722)
    nir, red = cube.sel(band="nir"), cube.sel(band="red")
    ndvi = ((nir - red) / (nir + red) * 10000).rename("ndvi")
    loss = zeit.extract_events(zeit.landtrendr(ndvi), min_magnitude=1000)
    points = zeit.stratified_sample(loss, n=24, min_per_stratum=12)
    return cube, ndvi, loss, points, cleared


def _session(scene, tmp_path=None, **kwargs):
    cube, ndvi, loss, points, _ = scene
    options = dict(map=loss, series=ndvi, fit=loss, open=False, block=False)
    if tmp_path is not None:
        options["save"] = tmp_path / "ref.gpkg"
    options.update(kwargs)
    s = zeit.interpret(cube, kwargs.pop("samples", points), **{k: v for k, v in options.items() if k != "samples"})
    return s


def _truth(scene, s):
    """The right label of each point: change where its block was cleared, with the year."""
    _, _, _, points, cleared = scene
    years = cleared[points.row, points.col]
    return ["change" if y else "no change" for y in years], [f"{y}-07-01" if y else None for y in years]


def test_points_and_a_blind_point(scene, tmp_path):
    s = _session(scene, tmp_path)
    try:
        content, _ = s.session.handle({"type": "points"})
        assert content["classes"] == ["no change", "change"] and content["blind"]
        assert len(content["points"]) == 24 and content["progress"] == {"done": 0, "skipped": 0, "total": 24}
        assert "stratum" not in content["points"][0] and not content["review_ready"]
        assert content["has_map"] and content["chips"]
        change = int(np.flatnonzero(s.session.points.stratum_name == "change")[0])
        point, buffers = s.session.handle({"type": "point", "id": change})
        assert point["map"] == "hidden until labelled" and point["series"]["overlays"] == [] and point["fit_hidden"]
        series = point["series"]
        assert len(series["x"]) == 12 and series["series"][0]["name"] == "ndvi"
        # one chip per year, RGBA
        chips = point["chips"]
        assert chips["labels"] == [str(y) for y in range(2000, 2012)] and chips["width"] == chips["height"] == 33
        assert len(buffers[0]) == 12 * 33 * 33 * 4
        # the point's cell: its series is the cube's at the point
        row, col = int(s.session.points.row[change]), int(s.session.points.col[change])
        np.testing.assert_allclose([v for v in series["series"][0]["y"]],
                                   scene[1].isel(y=row, x=col).values, rtol=1e-6)
        # once labelled, the map and the fit show
        s.session.handle({"type": "label", "id": change, "ref": "change", "ref_date": "2006-07-01",
                          "confidence": "medium", "note": "clear-cut"})
        point, _ = s.session.handle({"type": "point", "id": change})
        assert point["map"].startswith("change (last year before:")
        assert any(o["kind"] == "span" for o in point["series"]["overlays"])
        assert point["state"]["ref"] == "change" and point["state"]["ref_date"] == "2006-07-01"
    finally:
        s.close()


def test_labels_are_saved_and_resumed(scene, tmp_path):
    s = _session(scene, tmp_path)
    try:
        h = s.session.handle
        h({"type": "label", "id": 0, "ref": "no change", "confidence": "high"})
        h({"type": "label", "id": 1, "status": "skipped", "note": "cloudy"})
        with pytest.raises(ValueError, match="choose a class"):
            h({"type": "label", "id": 2, "ref": "water"})
        with pytest.raises(ValueError, match="confidence"):
            h({"type": "label", "id": 2, "ref": "change", "confidence": "sure"})
        saved = gpd.read_file(tmp_path / "ref.gpkg")
        assert saved.status.tolist()[:3] == ["done", "skipped", "todo"]
        assert saved.ref.tolist()[0] == "no change" and saved.note.tolist()[1] == "cloudy"
        assert saved.interpreter[0] is not None and saved.labelled_at[0] is not None
        assert saved.crs == scene[3].crs and len(saved) == 24
        assert s.progress == {"done": 1, "skipped": 1, "total": 24}
    finally:
        s.close()
    # the same file: the session goes on from there, without the points
    again = _session(scene, tmp_path, samples=None)
    try:
        content, _ = again.session.handle({"type": "points"})
        assert content["progress"] == {"done": 1, "skipped": 1, "total": 24}
        assert content["points"][0]["ref"] == "no change"
    finally:
        again.close()


def test_review_and_accuracy(scene, tmp_path):
    s = _session(scene, tmp_path)
    try:
        h = s.session.handle
        refs, dates = _truth(scene, s)
        for i, (ref, date) in enumerate(zip(refs, dates)):
            content, _ = h({"type": "review"})
            assert not content["available"] and "label every point" in content["reason"]
            h({"type": "label", "id": i, "ref": ref, "ref_date": date})
        content, _ = h({"type": "review"})
        assert content["available"] and content["classes"] == ["no change", "change"]
        counts = np.array(content["counts"])
        assert counts.sum() == 24
        # each cell lists its points
        for key, ids in content["cells"].items():
            a, b = (int(k) for k in key.split(","))
            assert len(ids) == counts[a, b]
        # the same numbers as zeit.accuracy
        acc = s.accuracy()
        direct = zeit.accuracy(scene[2], s.samples)
        assert content["accuracy"]["overall"][0] == pytest.approx(acc.overall.estimate)
        assert acc.overall.estimate == pytest.approx(direct.overall.estimate)
        assert zeit.accuracy(scene[2], s).overall.estimate == pytest.approx(direct.overall.estimate)
        # dates of change too
        assert s.accuracy(date_tolerance=1).date.n >= 1
    finally:
        s.close()


def test_not_blind_and_options(scene):
    cube, ndvi, loss, points, _ = scene
    s = _session(scene, blind=False, rgb=["nir", "red", "green"], chips=None, series=["nir", "red"])
    try:
        content, _ = s.session.handle({"type": "points"})
        assert content["review_ready"] and "stratum" in content["points"][0] and not content["chips"]
        point, buffers = s.session.handle({"type": "point", "id": 0})
        assert point["map"] != "hidden until labelled" and buffers == [] and "chips" not in point
        assert [line["name"] for line in point["series"]["series"]] == ["nir", "red"]
        assert s.session.frames.rgb
    finally:
        s.close()
    with pytest.raises(ValueError, match="rgb= takes three bands"):
        zeit.interpret(cube, points, classes=["a"], rgb=["nir", "red"], open=False, block=False)
    with pytest.raises(ValueError, match="choose the image"):
        zeit.interpret(cube.sel(band=["blue", "nir"]), points, classes=["a"], open=False, block=False)
    with pytest.raises(ValueError, match="reference classes"):
        zeit.interpret(ndvi, points, open=False, block=False)
    with pytest.raises(ValueError, match="give the points"):
        zeit.interpret(ndvi, classes=["a"], open=False, block=False)
    # chips by date
    s = _session(scene, chips="date", band="nir", series=None)
    try:
        point, buffers = s.session.handle({"type": "point", "id": 0})
        assert len(point["chips"]["labels"]) == 12 and point["series"]["series"][0]["name"] != "red"
    finally:
        s.close()


def test_notebook_widget(scene):
    pytest.importorskip("anywidget")
    from zeit._plot._widget import interpret_bundle, make_interpret_widget

    bundle = interpret_bundle()
    assert bundle.count("export default {") == 1 and "mountInterpret" in bundle
    s = _session(scene)
    try:
        widget = make_interpret_widget(s.session)
        assert "mountInterpret" in widget._esm and ".zi-panel" in widget._css_text
    finally:
        s.close()


def test_labelling_in_edge(scene, tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=True, args=["--ignore-gpu-blocklist"])
        except Exception as err:  # noqa: BLE001
            pytest.skip(f"Microsoft Edge is not available: {err}")
        s = _session(scene, tmp_path)
        errors = []
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 860})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(s.view.url)
            page.wait_for_selector(".zi-item")
            page.wait_for_function("document.querySelectorAll('.zi-chip').length === 12")
            # the first point is in the middle of the map
            box = page.locator(".zv-stage").bounding_box()
            pin = page.evaluate("(() => { const v = window.zeitViewer; return [(v.pin[0] - v.view.ox) * v.view.s, "
                                "(v.pin[1] - v.view.oy) * v.view.s]; })()")
            assert abs(pin[0] - box["width"] / 2) < 2 and abs(pin[1] - box["height"] / 2) < 2
            # three points by the keyboard: a class, a date from the image shown, Enter
            page.keyboard.press("2")
            page.keyboard.press("End")              # the last image
            page.keyboard.press("d")
            page.wait_for_function("document.querySelector('.zi-date').textContent === '2011-07-01'")
            page.keyboard.press("l")
            page.keyboard.press("Enter")
            page.wait_for_function("document.querySelector('.zi-progress').textContent.startsWith('1 of 24')")
            page.keyboard.press("1")
            page.keyboard.press("Enter")
            page.wait_for_function("document.querySelector('.zi-progress').textContent.startsWith('2 of 24')")
            page.keyboard.press("s")
            page.wait_for_function("document.querySelector('.zi-progress').textContent.includes('1 skipped')")
            # a click on a chip shows its image
            page.locator(".zi-chip").nth(3).click()
            page.wait_for_function("document.querySelector('.zv-label').textContent === '2003-07-01'")
            # the review waits for every point
            page.locator(".zi-tab", has_text="Review").click()
            page.wait_for_function("document.querySelector('.zi-review').textContent.includes('label every point')")
        finally:
            browser.close()
            s.close()
        saved = gpd.read_file(tmp_path / "ref.gpkg")
        first, second, third = (saved.iloc[i] for i in range(3))
        assert (first.ref, first.ref_date, first.confidence) == ("change", "2011-07-01", "low")
        assert (second.ref, second.status) == ("no change", "done") and third.status == "skipped"
        assert not errors
