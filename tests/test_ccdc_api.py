"""zeit.ccdc: one function for every input, and predict_synthetic_image on its result."""
import os

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit._ccdc import predict, run_ccdc

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = np.load(os.path.join(HERE, "data", "ccdc_matlab_parity_inputs.npz"))
NAMES = ["two_breaks", "unflagged_clouds", "mostly_snow_cat54"]  # pixels on the same date grid
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]
OPTIONS = dict(detection_bands=["green", "red", "nir", "swir1", "swir2"], tmask_bands=["green", "swir1"],
               thermal_band="thermal")
TRANSFORM = from_origin(-63.0, -10.0, 30, 30)


@pytest.fixture(scope="module")
def pixels():
    dates = INPUTS[NAMES[0] + "__dates"]
    values = np.stack([INPUTS[n + "__bands"].astype(float) for n in NAMES])   # (x, band, time)
    qa = np.stack([INPUTS[n + "__qa"].astype(int) for n in NAMES])             # (x, time)
    return dates, values, qa


@pytest.fixture(scope="module")
def cube(pixels):
    """(time, band, y=2, x=3): the three parity pixels on two rows."""
    dates, values, qa = pixels
    data = np.repeat(values.transpose(2, 1, 0)[:, :, None, :], 2, axis=2)
    times = pd.DatetimeIndex([pd.Timestamp.fromordinal(int(d)) for d in dates])
    da_ = xr.DataArray(data, dims=("time", "band", "y", "x"),
                       coords={"time": times, "band": BANDS, "y": [-10.0 - 15, -10.0 - 45],
                               "x": [-63.0 + 15, -63.0 + 45, -63.0 + 75]}).rio.write_crs(32721)
    qa_cube = xr.DataArray(np.repeat(qa.T[:, None, :], 2, axis=1), dims=("time", "y", "x"),
                           coords={"time": times})
    return da_, qa_cube


def _expected(pixels, x):
    dates, values, qa = pixels
    return run_ccdc(dates, values[x], qa[x], detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4], thermal_band=6)


def test_exported_and_old_entry_points_gone():
    assert callable(zeit.ccdc)
    for old in ("run_ccdc_array", "run_ccdc_image"):
        assert not hasattr(zeit, old)
    assert "ccdc" in zeit.__all__


def test_matches_the_single_pixel_engine(cube, pixels):
    data, qa = cube
    seg = zeit.ccdc(data, qa=qa, **OPTIONS)
    assert seg.coefs.dims == ("segment", "band", "coef", "y", "x")
    assert seg.coef.values.tolist() == ["a0", "c1", "a1", "b1", "a2", "b2", "a3", "b3"]
    assert seg.band.values.tolist() == BANDS
    assert seg.rio.crs.to_epsg() == 32721
    for row in range(2):
        for x in range(3):
            want = _expected(pixels, x)
            assert int(seg.n_segments.values[row, x]) == len(want)
            for i, s in enumerate(want):
                got = seg.isel(segment=i, y=row, x=x)
                assert pd.Timestamp(got.t_start.values).toordinal() == s["t_start"]
                assert pd.Timestamp(got.t_end.values).toordinal() == s["t_end"]
                if s["t_break"]:
                    assert pd.Timestamp(got.t_break.values).toordinal() == s["t_break"]
                else:
                    assert np.isnat(got.t_break.values)
                np.testing.assert_allclose(got.coefs.values, np.asarray(s["coefs"]), rtol=1e-5, atol=1e-3)
                np.testing.assert_allclose(got.rmse.values, s["rmse"], rtol=1e-5)
            beyond = seg.isel(y=row, x=x, segment=slice(len(want), None))
            assert np.isnat(beyond.t_start.values).all() and np.isnan(beyond.coefs.values).all()


def test_inputs_qa_band_numpy_dask_and_pixel(cube, pixels):
    data, qa = cube
    reference = zeit.ccdc(data, qa=qa, **OPTIONS)

    # the QA as a band of the cube
    with_qa = xr.concat([data, qa.expand_dims(band=["fmask"], axis=1)], dim="band")
    xr.testing.assert_allclose(zeit.ccdc(with_qa, qa="fmask", **OPTIONS), reference)

    # numpy + dates
    dates = pd.DatetimeIndex(data.time.values)
    seg = zeit.ccdc(data.values, dates=dates, qa=qa.values, detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4],
                    thermal_band=6)
    np.testing.assert_allclose(seg.coefs.values, reference.coefs.values, equal_nan=True)

    # dask: lazy, same result
    lazy = zeit.ccdc(data.chunk({"y": 1, "x": 2}), qa=qa.chunk({"y": 1, "x": 2}), **OPTIONS)
    assert isinstance(lazy.coefs.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), reference)

    # one pixel: a DataFrame indexed by date (QA as a column) and a (time, band) DataArray
    frame = pd.DataFrame(data.isel(y=0, x=0).values, index=dates, columns=BANDS)
    frame["fmask"] = qa.isel(y=0, x=0).values
    one = zeit.ccdc(frame, qa="fmask", **OPTIONS)
    assert one.coefs.dims == ("segment", "band", "coef")
    np.testing.assert_allclose(one.coefs.values, reference.coefs.isel(y=0, x=0).values, equal_nan=True)
    two = zeit.ccdc(data.isel(y=0, x=0), qa=qa.isel(y=0, x=0).values, **OPTIONS)
    np.testing.assert_allclose(two.coefs.values, one.coefs.values, equal_nan=True)


def test_raster_files(cube, tmp_path):
    data, qa = cube
    reference = zeit.ccdc(data, qa=qa, **OPTIONS)
    with_qa = xr.concat([data, qa.expand_dims(band=["fmask"], axis=1)], dim="band").astype(np.int16)
    path = zeit.save_raster(with_qa, tmp_path / "landsat.tif")  # bands named date_band
    seg = zeit.ccdc(str(path), qa="fmask", **OPTIONS)
    np.testing.assert_allclose(seg.coefs.values, reference.coefs.values, rtol=1e-4, atol=1e-2, equal_nan=True)
    assert seg.rio.crs.to_epsg() == 32721

    # interleaved by date, without descriptions: dates + bands
    plain = tmp_path / "plain.tif"
    stack = with_qa.values.reshape(-1, 2, 3)
    with rasterio.open(plain, "w", driver="GTiff", height=2, width=3, count=stack.shape[0], dtype="int16",
                       crs="EPSG:32721", transform=TRANSFORM) as dst:
        dst.write(stack)
    seg = zeit.ccdc(str(plain), dates=data.time.values, bands=BANDS + ["fmask"], qa="fmask", **OPTIONS)
    np.testing.assert_allclose(seg.coefs.values, reference.coefs.values, rtol=1e-4, atol=1e-2, equal_nan=True)


def test_missing_observations(cube, pixels):
    data, qa = cube
    gappy = data.copy()
    gappy[5, :, 0, 0] = np.nan                       # no observation on one date
    seg = zeit.ccdc(gappy, qa=qa, **OPTIONS)
    dates, values, qa_px = pixels
    v = values[0].copy()
    q = qa_px[0].copy()
    v[:, 5] = 0.0
    q[5] = 255
    want = run_ccdc(dates, v, q, detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4], thermal_band=6)
    assert int(seg.n_segments.values[0, 0]) == len(want)
    np.testing.assert_allclose(seg.coefs.values[0, :, :, 0, 0], np.asarray(want[0]["coefs"]), rtol=1e-5, atol=1e-3)

    zeros = data.astype(np.int16).copy()
    zeros[5, :, 0, 0] = 0                            # integer stack: 0 is no observation
    seg0 = zeit.ccdc(zeros, qa=qa, **OPTIONS)
    assert int(seg0.n_segments.values[0, 0]) == len(want)


def test_predict_synthetic_image_from_segments(cube, pixels):
    data, qa = cube
    seg = zeit.ccdc(data, qa=qa, **OPTIONS)
    image = zeit.predict_synthetic_image(seg, "2005-07-01")
    assert image.dims == ("band", "y", "x") and image.rio.crs.to_epsg() == 32721
    day = pd.Timestamp("2005-07-01").toordinal()
    for x in range(3):
        want = _expected(pixels, x)
        model = next((s for s in want if s["t_start"] <= day <= s["t_end"]), None)
        if model is None:
            model = [s for s in want if s["t_start"] <= day][-1] if any(s["t_start"] <= day for s in want) else want[0]
        for b in range(len(BANDS)):
            assert image.values[b, 0, x] == pytest.approx(predict(model["coefs"][b], day), rel=1e-4, abs=0.5)
    # a lazy result predicts lazily, with the same values
    lazy = zeit.ccdc(data.chunk({"y": 1, "x": 2}), qa=qa.chunk({"y": 1, "x": 2}), **OPTIONS)
    lazy_image = zeit.predict_synthetic_image(lazy, "2005-07-01")
    assert isinstance(lazy_image.data, da.Array)
    np.testing.assert_allclose(lazy_image.values, image.values, rtol=1e-6, equal_nan=True)
    # the old numpy call still works
    old = zeit.predict_synthetic_image(np.zeros((1, 12, 1, 1), dtype=np.float32), target_julian_day=5, num_bands=1)
    assert old.shape == (1, 1, 1)


def test_save_segments(cube, tmp_path):
    data, qa = cube
    seg = zeit.ccdc(data, qa=qa, **OPTIONS)
    folder = zeit.save_raster(seg, tmp_path / "ccdc")
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["coefs.tif", "n_segments.tif", "rmse.tif", "t_break.tif", "t_end.tif", "t_start.tif"])
    with rasterio.open(folder / "coefs.tif") as src:
        assert src.count == 6 * 7 * 8
        assert src.descriptions[0] == "1_blue_a0" and src.descriptions[9] == "1_green_c1"
    with rasterio.open(folder / "t_start.tif") as src:   # dates as decimal years
        first = src.read(1)
        assert first[0, 0] == pytest.approx(2000.04, abs=0.01)
        assert np.isnan(src.read(6)).all()


def test_errors(cube):
    data, qa = cube
    with pytest.raises(ValueError, match="qa="):
        zeit.ccdc(data, qa="fmask")
    with pytest.raises(ValueError, match="detection_bands"):
        zeit.ccdc(data, detection_bands=["swir3"])
    with pytest.raises(ValueError, match="dates="):
        zeit.ccdc(data.values)


def test_cli_ccdc(cube, tmp_path):
    import subprocess
    import sys

    data, qa = cube
    with_qa = xr.concat([data, qa.expand_dims(band=["fmask"], axis=1)], dim="band").astype(np.int16)
    path = zeit.save_raster(with_qa, tmp_path / "landsat.tif")
    reference = zeit.ccdc(str(path), qa="fmask", max_segments=3)
    out = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-m", "zeit.cli", "ccdc", str(path), str(out), "--qa-band", "7",
                             "--max-segments", "3", "--chunk-size", "1"], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"ccdc_{k}.tif" for k in ("t_start", "t_end", "t_break", "n_segments", "rmse", "coefs"))
    with rasterio.open(out / "ccdc_coefs.tif") as src:
        np.testing.assert_allclose(src.read().reshape(reference.coefs.shape), reference.coefs.values,
                                   rtol=1e-5, equal_nan=True)
