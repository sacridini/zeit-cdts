"""The short scripts: zeit.qa_to_fmask, NaN bands and ``scale=`` in ccdc/sccd, one pixel in coded/tmask, and
load_raster without rioxarray's "Scales differ across bands"."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit._ccdc_api import _to_10000

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = np.load(os.path.join(HERE, "data", "ccdc_matlab_parity_inputs.npz"))
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


# ---- zeit.qa_to_fmask ---------------------------------------------------------------------------------------

def test_qa_to_fmask_landsat_c2_bits():
    codes = np.array([0, 1, 2, 8, 10, 16, 32, 128, 16 | 32, 8 | 16, 1 | 8, 128 | 32, 21824, 22280, 30048, 1 << 14])
    want = [0, 255, 4, 4, 4, 2, 3, 1, 2, 4, 255, 3, 0, 4, 3, 0]
    out = zeit.qa_to_fmask(codes, "landsat_c2")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, want)


def test_qa_to_fmask_other_rules():
    np.testing.assert_array_equal(zeit.qa_to_fmask(np.array([0, 1, 2, 3, 4, 5, 100, 255]), "fmask"),
                                  [0, 1, 2, 3, 4, 255, 255, 255])
    scl = np.arange(0, 13)
    np.testing.assert_array_equal(zeit.qa_to_fmask(scl, "scl"), [255, 255, 2, 2, 0, 0, 1, 4, 4, 4, 4, 3, 255])
    hls = np.array([0, 2, 4, 6, 8, 16, 32, 8 | 16, 255, 2 | 32])
    np.testing.assert_array_equal(zeit.qa_to_fmask(hls, "hls_fmask"), [0, 4, 4, 4, 2, 3, 1, 2, 255, 4])
    np.testing.assert_array_equal(zeit.qa_to_fmask(np.array([0, 1, 7]), "nonzero"), [4, 0, 0])


@pytest.mark.parametrize("rule", ["fmask", "landsat_c2", "scl", "hls_fmask", "nonzero"])
def test_qa_to_fmask_agrees_with_qa_mask(rule):
    codes = np.arange(0, 256)
    np.testing.assert_array_equal(zeit.qa_to_fmask(codes, rule) <= 1, zeit.qa_mask(codes, rule))


def test_qa_to_fmask_nan_negative_nodata_dask_and_errors():
    q = np.array([np.nan, -1.0, 0.0, 1.9, 4.0, 255.0])
    np.testing.assert_array_equal(zeit.qa_to_fmask(q, "fmask"), [255, 255, 0, 1, 4, 255])
    np.testing.assert_array_equal(zeit.qa_to_fmask(q, "Landsat-C2"), [255, 255, 0, 255, 0, 255])
    da = xr.DataArray(np.array([[0, 1, 0]], dtype="uint8"), dims=("y", "x")).rio.write_nodata(0)
    out = zeit.qa_to_fmask(da, "landsat_c2")
    assert out.dims == ("y", "x") and out.name == "fmask" and list(out.values[0]) == [255, 255, 255]
    lazy = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("y", "x")).chunk({"y": 2})
    out = zeit.qa_to_fmask(lazy, "scl")
    assert out.chunks is not None
    np.testing.assert_array_equal(out.values, zeit.qa_to_fmask(lazy.values, "scl"))
    with pytest.raises(ValueError, match="unknown QA rule"):
        zeit.qa_to_fmask(q, "cloudy")
    with pytest.raises(TypeError):
        zeit.qa_to_fmask(np.array(["a"]), "fmask")


# ---- NaN bands and scale= in ccdc and sccd ------------------------------------------------------------------

@pytest.fixture(scope="module")
def pixel():
    """The first parity pixel: (time, band) in reflectance x 10000 with 6 bands, its Fmask and its dates."""
    dates = INPUTS["two_breaks__dates"]
    values = INPUTS["two_breaks__bands"].astype(float)[:6].T      # (time, band)
    qa = INPUTS["two_breaks__qa"].astype(int)
    times = pd.DatetimeIndex([pd.Timestamp.fromordinal(int(d)) for d in dates])
    return xr.DataArray(values, dims=("time", "band"), coords={"time": times, "band": BANDS}), qa


@pytest.mark.parametrize("name,code", [("ccdc", 4), ("sccd", 255)])
def test_nan_band_is_no_observation(pixel, name, code):
    series, qa = pixel
    function = getattr(zeit, name)
    gappy = series.copy()
    gappy[[10, 11, 40, 90], 2] = np.nan       # one band missing on four dates
    gappy[[25, 26], :] = np.nan               # every band missing on two
    by_hand = gappy.fillna(0)
    fmask = qa.copy()
    for i in (10, 11, 40, 90, 25, 26):
        fmask[i] = code
    want = function(by_hand, qa=fmask)
    got = function(gappy, qa=qa)
    assert int(got.n_segments) >= 1
    for variable in ("t_start", "t_end", "t_break", "n_segments", "rmse", "coefs"):
        xr.testing.assert_identical(got[variable], want[variable])
    # and what NaN did before: the dates it leaves out are not a clear observation
    clean = function(series, qa=qa)
    assert not np.array_equal(clean.coefs.values, got.coefs.values, equal_nan=True)


def test_nan_band_in_a_cube_and_lazily(pixel):
    series, qa = pixel
    cube = xr.concat([series, series], dim="x").assign_coords(x=[1.0, 2.0]).expand_dims(y=[1.0])
    cube = cube.transpose("time", "band", "y", "x").copy()
    cube[[10, 40], 1, 0, 1] = np.nan
    fmask = xr.DataArray(np.repeat(qa[:, None, None], 2, axis=2), dims=("time", "y", "x"))
    got = zeit.ccdc(cube, qa=fmask)
    lazy = zeit.ccdc(cube.chunk({"x": 1}), qa=fmask.chunk({"x": 1})).compute()
    xr.testing.assert_allclose(got, lazy)
    first = zeit.ccdc(series, qa=qa)
    xr.testing.assert_identical(got.coefs.isel(y=0, x=0, drop=True), first.coefs)
    assert not np.array_equal(got.coefs.values[..., 0, 1], got.coefs.values[..., 0, 0], equal_nan=True)


@pytest.mark.parametrize("name", ["ccdc", "sccd"])
def test_scale_units(pixel, name):
    series, qa = pixel
    function = getattr(zeit, name)
    reference = function(series, qa=qa)
    for scale in (1, "auto"):
        got = function(series / 10000.0, qa=qa, scale=scale)
        assert (got.t_break.values == reference.t_break.values).all() or (
            pd.isna(got.t_break.values) == pd.isna(reference.t_break.values)).all()
        assert int(got.n_segments) == int(reference.n_segments)
        np.testing.assert_array_equal(got.t_start.values, reference.t_start.values)
        np.testing.assert_allclose(got.coefs.values, reference.coefs.values, rtol=1e-6, atol=1e-3, equal_nan=True)
    for scale in (None, 10000, "auto"):  # reflectance x 10000 stays as it is
        got = function(series, qa=qa, scale=scale)
        xr.testing.assert_identical(got.coefs, reference.coefs)
    with pytest.raises(ValueError, match="scale"):
        function(series, qa=qa, scale="reflectance")


def test_scale_leaves_the_thermal_band(pixel):
    series, _ = pixel
    cube = series.expand_dims(y=[1.0], x=[1.0], axis=(2, 3)).assign_coords(band=BANDS)
    assert _to_10000(cube, None) is None and _to_10000(cube, 10000) is None
    factor = _to_10000(cube, 1, thermal=5)
    assert factor.shape == (1, 6, 1, 1) and list(factor.ravel()) == [10000.0] * 5 + [1.0]


# ---- one pixel in coded and tmask ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def forest_cut():
    """A (time, band) series of reflectance x 10000 of a forest cleared in 2005, a 16-day cycle."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", "2010-12-31", freq="16D")
    forest = np.array([500, 900, 400, 6100, 3000, 1000.0])
    soil = np.array([2000, 3000, 3400, 5800, 6000, 5800.0])
    cleared = np.asarray(dates >= "2005-07-01")[:, None]
    values = np.where(cleared, soil, forest) + rng.normal(0, 25, (len(dates), 6))
    values[rng.random(len(dates)) < 0.2] = np.nan
    return xr.DataArray(values, dims=("time", "band"), coords={"time": dates, "band": BANDS})


def test_coded_takes_one_pixel(forest_cut):
    one = zeit.coded(forest_cut, start=2003)
    cube = forest_cut.expand_dims(y=[1.0], x=[1.0], axis=(2, 3))
    full = zeit.coded(cube, start=2003)
    assert "y" not in one.dims and "x" not in one.dims
    assert one.t_change.dims == ("event",) and one.strata.dims == ()
    for variable in ("t_change", "t_before", "n_events", "n_train", "type", "strata", "ndfi_mean", "rmse"):
        xr.testing.assert_identical(one[variable], full[variable].isel(y=0, x=0, drop=True))
    assert int(one.n_events) >= 1 and int(one.strata) == 4
    first = pd.Timestamp(one.t_change.values[0])
    assert pd.Timestamp("2005-07-01") <= first <= pd.Timestamp("2005-12-31")
    # a (band, time) series is the same pixel
    xr.testing.assert_identical(zeit.coded(forest_cut.transpose("band", "time"), start=2003).t_change, one.t_change)


def test_tmask_and_unmix_take_one_pixel(forest_cut):
    clear = zeit.tmask(forest_cut)
    assert clear.dims == ("time",)
    full = zeit.tmask(forest_cut.expand_dims(y=[1.0], x=[1.0], axis=(2, 3)))
    np.testing.assert_array_equal(clear.values, full.isel(y=0, x=0).values)
    fractions = zeit.unmix(forest_cut)
    assert fractions.ndfi.dims == ("time",)


def test_other_functions_with_a_band_series(pixel):
    series, qa = pixel
    assert zeit.ccdc(series, qa=qa).coefs.dims == ("segment", "band", "coef")
    assert zeit.sccd(series, qa=qa).coefs.dims == ("segment", "band", "coef")


# ---- load_raster and rioxarray's "Scales differ across bands" -----------------------------------------------

def _stack(folder, n=3):
    transform = from_origin(500000.0, 8000000.0, 30.0, 30.0)
    rng = np.random.default_rng(3)
    for k in range(n):
        data = rng.integers(100, 8000, (3, 6, 7)).astype("int16")
        data[2] = rng.choice([0, 21824, 22280], size=(6, 7))
        with rasterio.open(folder / f"ls_2020-0{k + 1}-10.tif", "w", driver="GTiff", height=6, width=7, count=3,
                           dtype="int16", crs="EPSG:32722", transform=transform) as dst:
            dst.write(data)
            dst.scales = [2.75e-05, 2.75e-05, 1.0]
            dst.offsets = [-0.2, -0.2, 0.0]
    return str(folder)


def _differ(records):
    return [str(w.message) for w in records if "differ across bands" in str(w.message)]


def test_no_scales_differ_warning_when_zeit_reads_the_scales(tmp_path):
    src = _stack(tmp_path)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        cube = zeit.load_raster(src, bands={"blue": 1, "green": 2}, scale=True, masked=True, pixel=(3, 2))
        qa = zeit.load_raster(src, band=3, masked=True, pixel=(3, 2))
        series = zeit.load_raster(src, band=1, scale=True, masked=True)
    assert _differ(records) == []
    assert cube.dims == ("time", "band") and qa.dims == ("time",) and series.dims == ("time", "y", "x")
    np.testing.assert_allclose(series.values[0, 0, 0], np.float32(series.values[0, 0, 0]))


def test_other_warnings_and_unselected_bands_still_warn(tmp_path):
    src = _stack(tmp_path, n=1)
    path = os.path.join(src, "ls_2020-01-10.tif")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        zeit.load_raster(path)                  # all the bands, unscaled: the warning is true
    assert _differ(records)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        zeit.load_raster(path, band=1, scale=True)
        warnings.warn("something else", UserWarning)
    assert _differ(records) == [] and any("something else" in str(w.message) for w in records)
