"""zeit.unmix, the NDFI and zeit.coded: forest degradation from spectral mixture analysis."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.optimize import nnls as scipy_nnls

import zeit
from zeit._core import coded as coded_core
from zeit._core import sma
from zeit._sma import endmember_table, ndfi_of

E = endmember_table("souza2005")   # gv, shade, npv, soil, cloud x blue..swir2, reflectance


# ---------------------------------------------------------------------- a scene
def _fractions(kind, t, rng):
    """(gv, shade, npv, soil, cloud) of a kind of pixel at decimal year t."""
    season = 0.03 * np.sin(2 * np.pi * t)
    forest = np.array([0.80 + season, 0.12, 0.05 - season / 3, 0.03, 0.0])
    pasture = np.array([0.30 + 2 * season, 0.05, 0.35, 0.30, 0.0])
    if kind == "forest":
        f = forest
    elif kind == "pasture":
        f = pasture
    elif kind == "logged":   # logged in mid-2005; the canopy closes again over two years
        k = 0.0 if t < 2005.5 else max(0.0, 1 - (t - 2005.5) / 2.0)
        f = forest + k * np.array([-0.35, 0.05, 0.22, 0.08, 0.0])
    else:                    # cleared in mid-2007, pasture afterwards
        f = forest if t < 2007.5 else pasture
    f = np.clip(f + rng.normal(0, 0.01, 5) * np.array([1, 1, 1, 1, 0]), 0, None)
    return f / f.sum()


KINDS = ["forest", "logged", "cleared", "pasture"]   # one per row


@pytest.fixture(scope="module")
def scene():
    """16-day Landsat-like reflectance (x 10000) mixed from known fractions, 2000-2012, with a
    third of the dates missing and some clouds the mask missed."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", "2012-12-31", freq="16D")
    t = dates.year + (dates.dayofyear - 1) / 365.25
    ny, nx = 4, 5
    data = np.full((len(dates), 6, ny, nx), np.nan, dtype=np.float32)
    for r in range(ny):
        for c in range(nx):
            for i, tt in enumerate(t):
                u = rng.random()
                if u < 0.3:
                    continue
                f = _fractions(KINDS[r], tt, rng)
                if u < 0.33:
                    f = 0.5 * f + 0.5 * np.array([0, 0, 0, 0, 1.0])
                data[i, :, r, c] = (f @ E.to_numpy()) * 10000 + rng.normal(0, 20, 6)
    coords = {"time": dates, "band": ["blue", "green", "red", "nir", "swir1", "swir2"],
              "y": 9000 - 15 - 30 * np.arange(ny), "x": 500000 + 15 + 30 * np.arange(nx)}
    return xr.DataArray(data, dims=("time", "band", "y", "x"), coords=coords).rio.write_crs(32722)


# ---------------------------------------------------------------------- unmixing
def test_nnls_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(500):
        m, n = int(rng.integers(3, 9)), int(rng.integers(2, 7))
        A, b = rng.normal(size=(m, n)), rng.normal(size=m)
        got = np.array(sma.nnls(A.ravel().tolist(), b.tolist(), m, n))
        np.testing.assert_allclose(got, scipy_nnls(A, b)[0], atol=1e-9)


def test_unmixing_recovers_the_fractions():
    rng = np.random.default_rng(2)
    f = rng.dirichlet(np.ones(5), size=500)
    out = sma.unmix_batch(f @ E.to_numpy(), E.to_numpy())
    np.testing.assert_allclose(out[:, :5], f, atol=1e-9)
    assert out[:, 5].max() < 1e-9
    # the fully constrained solution is scipy's NNLS of the system with the sum-to-one row
    y = f[0] @ E.to_numpy() + rng.normal(0, 0.01, 6)
    A = np.vstack([E.to_numpy().T, 1000 * np.ones(5)])
    np.testing.assert_allclose(sma.unmix_pixel(y.tolist(), E.to_numpy().ravel().tolist(), 5)[:5],
                               scipy_nnls(A, np.append(y, 1000))[0], atol=1e-9)
    # without constraints: plain least squares, which may go negative
    free = sma.unmix_batch((f[:3] @ E.to_numpy()) + 0.05, E.to_numpy(), sum_to_one=False, nonneg=False)
    np.testing.assert_allclose(free[:, :5], np.linalg.lstsq(E.to_numpy().T, ((f[:3] @ E.to_numpy()) + 0.05).T,
                                                            rcond=None)[0].T, atol=1e-9)


def test_unmix(scene):
    fr = zeit.unmix(scene)
    assert list(fr.data_vars) == ["gv", "shade", "npv", "soil", "cloud", "rmse", "ndfi"]
    assert fr.gv.dims == ("time", "y", "x") and fr.rio.crs.to_epsg() == 32722 and fr.attrs["scale"] == 10000
    valid = fr.gv.notnull()
    total = sum(fr[v] for v in ["gv", "shade", "npv", "soil", "cloud"])
    np.testing.assert_allclose(total.values[valid.values], 1, atol=1e-5)
    assert float(fr.rmse.max()) < 0.01
    # forest near 1, pasture below 0 (Souza et al. 2005)
    assert float(fr.ndfi.isel(y=0).median()) > 0.75 and float(fr.ndfi.isel(y=3).median()) < 0
    # the clouds the mask missed: masked by their fraction
    masked = zeit.unmix(scene, cloud_threshold=0.05)
    assert int(masked.ndfi.notnull().sum()) < int(fr.ndfi.notnull().sum())
    assert float(masked.cloud.max()) <= 0.05
    # lazy is the same; reflectance 0-1 too; integers are x 10000
    xr.testing.assert_allclose(zeit.unmix(scene.chunk({"time": 40, "x": 2})).compute(), fr)
    np.testing.assert_allclose(zeit.unmix(scene / 10000).gv.values, fr.gv.values, atol=1e-6, equal_nan=True)
    assert zeit.unmix(scene.fillna(0).astype(np.int16)).attrs["scale"] == 10000


def test_custom_endmembers(scene):
    table = E.loc[["gv", "soil", "shade"]].copy()
    table.columns = ["blue", "green", "red", "nir", "swir1", "swir2"]
    fr = zeit.unmix(scene, table)
    assert list(fr.data_vars) == ["gv", "soil", "shade", "rmse"]    # no NDFI without npv
    renamed = scene.assign_coords(band=["B2", "B3", "B4", "B5", "B6", "B7"])
    by_bands = zeit.unmix(renamed, table, bands=["B2", "B3", "B4", "B5", "B6", "B7"])
    np.testing.assert_allclose(by_bands.gv.values, fr.gv.values, equal_nan=True)
    with pytest.raises(ValueError, match="no band"):
        zeit.unmix(renamed, table)
    with pytest.raises(ValueError, match="band dimension"):
        zeit.unmix(scene.isel(band=0))


def test_ndfi():
    pure = {"gv": np.array([1.0, 0.0, 0.0, 0.5]), "npv": np.array([0.0, 1.0, 0.0, 0.0]),
            "soil": np.array([0.0, 0.0, 1.0, 0.0]), "shade": np.array([0.0, 0.0, 0.0, 0.5])}
    np.testing.assert_allclose(ndfi_of(pure), [1, -1, -1, 1])


# ---------------------------------------------------------------------- the engine
def _fit(t, y):
    X = np.column_stack([np.ones_like(t), np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, float(np.sqrt(np.mean((y - X @ beta) ** 2)))


def _reference(t, y, train_start, train_years, consec, thresh, min_years, min_obs):
    """CODED's monitoring written out in Python, the oracle of the C++ engine."""
    ok = np.isfinite(y)
    win = ok & (t >= train_start) & (t < train_start + train_years)
    if win.sum() < max(min_obs, 4):
        return None, []
    beta, rmse = _fit(t[win], y[win])
    model = (beta, rmse)
    events, run, total, first, before, previous = [], 0, 0.0, None, None, None
    i = int(np.searchsorted(t, train_start + train_years))
    previous = int(np.flatnonzero(ok[:i])[-1]) if ok[:i].any() else None
    while i < len(t):
        if not ok[i]:
            i += 1
            continue
        b, r = model
        res = y[i] - (b[0] + b[1] * np.sin(2 * np.pi * t[i]) + b[2] * np.cos(2 * np.pi * t[i]))
        if res / r < -thresh:
            if run == 0:
                first, before = i, previous
            run += 1
            total += res
        else:
            run, total = 0, 0.0
        previous = i
        if run == consec:
            post = ok & (t >= t[first]) & (t < t[first] + min_years)
            events.append((t[first], t[before] if before is not None else np.nan, total / run))
            if post.sum() < max(min_obs, 4):
                break
            model = _fit(t[post], y[post])
            resume = t[first] + min_years
            later = np.flatnonzero(t >= resume)
            if not len(later):
                break
            previous = int(np.flatnonzero(post)[-1])
            i, run, total = int(later[0]), 0, 0.0
            continue
        i += 1
    return (beta, rmse), events


def test_engine_matches_the_reference():
    rng = np.random.default_rng(4)
    p = coded_core.Params()
    p.train_start, p.train_years, p.consec, p.thresh, p.min_years, p.min_obs, p.max_events = \
        2000.0, 3.0, 3, 3.0, 2.0, 6, 5
    t = np.sort(2000 + rng.random(260) * 12)
    checked = 0
    for k in range(200):
        y = 0.85 + 0.03 * np.sin(2 * np.pi * t) + rng.normal(0, 0.02, len(t))
        for _ in range(rng.integers(0, 3)):   # drops, some recovering
            at = 2003.5 + rng.random() * 7
            depth = rng.uniform(0.1, 0.8)
            length = rng.uniform(0.3, 5)
            y = y - depth * ((t >= at) & (t < at + length))
        y[rng.random(len(t)) < 0.25] = np.nan
        out = np.array(coded_core.coded_pixel(t.tolist(), y.tolist(), [], 0, p))
        model, events = _reference(t, y, 2000.0, 3.0, 3, 3.0, 2.0, 6)
        np.testing.assert_allclose(out[:3], model[0], rtol=1e-9, atol=1e-12)
        assert out[3] == pytest.approx(model[1], rel=1e-9)
        assert int(out[5]) == len(events)
        for e, (when, before, change) in enumerate(events):
            ev = out[6 + e * 7: 6 + (e + 1) * 7]
            assert ev[0] == when and (ev[1] == before or (np.isnan(ev[1]) and np.isnan(before)))
            assert ev[2] == pytest.approx(change, rel=1e-9)
            checked += 1
    assert checked > 50


# ---------------------------------------------------------------------- zeit.coded
def test_coded(scene):
    result = zeit.coded(scene, start=2003)
    assert result.attrs["algorithm"] == "CODED" and result.rio.crs.to_epsg() == 32722
    assert result.strata.attrs["flag_meanings"] == "forest non_forest degradation deforestation disturbance"
    # rows: forest, logged (degradation), cleared (deforestation), pasture
    assert result.strata.values.tolist() == [[1] * 5, [3] * 5, [4] * 5, [2] * 5]
    assert result.forest.values.tolist() == [[1] * 5, [1] * 5, [1] * 5, [0] * 5]
    first = pd.DatetimeIndex(result.t_change.isel(event=0).values[1])
    assert ((first >= "2005-06-15") & (first <= "2005-10-01")).all()
    first = pd.DatetimeIndex(result.t_change.isel(event=0).values[2])
    assert ((first >= "2007-06-15") & (first <= "2007-10-01")).all()
    assert (result.ndfi_change.isel(event=0).values[1:3] < -0.15).all()
    assert (result.n_events.values[[0, 3]] == 0).all()
    # lazy is the same
    xr.testing.assert_identical(zeit.coded(scene.chunk({"x": 2}), start=2003).compute(), result)
    # the fractions of zeit.unmix work as input
    fr = zeit.unmix(scene, cloud_threshold=0.05)
    xr.testing.assert_identical(zeit.coded(fr, start=2003), result)


def test_coded_with_training_points(scene):
    import geopandas as gpd

    xs, ys = np.meshgrid(scene.x.values, scene.y.values)
    labels = np.array(KINDS)[:, None].repeat(scene.sizes["x"], 1)
    points = gpd.GeoDataFrame({"label": np.where(labels == "pasture", "pasture", "forest").ravel()},
                              geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=32722).to_crs(4326)
    result = zeit.coded(scene, start=2003, training=points)
    assert result.strata.values.tolist() == [[1] * 5, [3] * 5, [4] * 5, [2] * 5]
    assert "random forest" in result.attrs["forest"]
    with pytest.raises(ValueError, match="forest_label"):
        zeit.coded(scene, start=2003, training=points.assign(label=np.where(points.label == "forest", "mata", "pasto")))


def test_coded_and_the_rest_of_zeit(scene):
    result = zeit.coded(scene, start=2003)
    events = zeit.extract_events(result)
    assert events.attrs["algorithm"] == "CODED"
    assert events.yod.values[1].tolist() == [2005] * 5 and events.yod.values[2].tolist() == [2007] * 5
    assert (events.yod.values[[0, 3]] == 0).all() and (events.dsnr.values[1:3] > 3).all()
    # the strata go straight into a sample design
    design = zeit.sampling_design(result.strata, n=40, min_per_stratum=5)
    assert design.name.tolist() == ["forest", "non_forest", "degradation", "deforestation"]
    # the viewer marks the changes
    from zeit._plot._data import Frames
    from zeit._plot._fit import overlays, pixel_series
    fr = zeit.unmix(scene)
    s = pixel_series(Frames(fr.ndfi), 0, 1)
    marks = [o["label"] for o in overlays(result, s, shape=(4, 5)) if o["kind"] == "vline"]
    assert marks == ["CODED degradation"]
    # compute_indices has the NDFI too
    nd = zeit.compute_indices(scene / 10000, ["NDFI"])
    np.testing.assert_allclose(nd.sel(band="NDFI").values, fr.ndfi.values, atol=1e-5, equal_nan=True)
    with pytest.raises(ValueError, match="direction"):
        zeit.coded(scene, direction="gain")


def test_cli(scene, tmp_path):
    import subprocess
    import sys

    import rasterio

    path = zeit.save_raster(scene, tmp_path / "landsat.tif")
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, "-m", "zeit.cli", "coded", str(path), str(out), "--start", "2003"],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    with rasterio.open(out / "coded.tif") as src:
        names = list(src.descriptions)
        strata = src.read(names.index("strata") + 1)
    assert strata.tolist() == zeit.coded(scene, start=2003).strata.values.tolist()


def test_ndfi_on_earth_engine_builds_the_unmixing():
    """Without an Earth Engine account: a stand-in image records what _compute_indices asks."""
    pytest.importorskip("ee")
    from zeit.gee.main import _compute_indices

    calls = []

    class Image:
        def __init__(self, name="img"):
            self.name = name

        def __getattr__(self, method):
            def call(*args, **kwargs):
                calls.append((self.name, method, args))
                return Image(f"{self.name}.{method}")
            return call

    _compute_indices(Image(), ["NDFI"])
    unmix = [c for c in calls if c[1] == "unmix"]
    assert len(unmix) == 1
    endmembers, sum_to_one, nonneg = unmix[0][2]
    assert sum_to_one is True and nonneg is True
    np.testing.assert_allclose(endmembers, E.to_numpy())    # gv, shade, npv, soil, cloud in reflectance
    selected = [c for c in calls if c[1] == "select" and c[0].endswith("add")][0][2][0]
    assert selected == ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    expression = [c for c in calls if c[1] == "expression"][0][2]
    assert "GV / (1 - SHADE)" in expression[0] and set(expression[1]) == {"GV", "SHADE", "NPV", "SOIL"}
    assert ("img.addBands", "rename", ("NDFI",)) not in calls and any(c[1] == "rename" and c[2] == ("NDFI",)
                                                                       for c in calls)
