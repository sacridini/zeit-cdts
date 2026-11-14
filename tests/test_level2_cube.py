"""zeit.load_level2_series and zeit.level2_series: the scenes of run_series as one lazy reflectance cube, with
stand-ins for the downloads and for level2 (no network, no real scene)."""

import json
import zlib
from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr

import zeit
from zeit import _level2, _level2_series, _s2fetch
from zeit._level2 import BOA_BANDS
from zeit._level2_series import JOURNAL, MARKER, run_series

N = 8
FILL = -9999


def name(date, n="0509", tile="T23LKC", platform="S2B"):
    d = pd.Timestamp(date).strftime("%Y%m%d")
    return f"{platform}_MSIL1C_{d}T131239_N{n}_R138_{tile}_{d}T145717.SAFE"


def scene_ds(product, shift=0.0):
    """A fake level2 result: 10 bands with a seasonal signal, fills, and QAI with chosen FORCE bits."""
    base = Path(product).name
    seed = zlib.crc32(base.encode())
    rng = np.random.default_rng(seed)
    when = pd.Timestamp(base.split("_")[2][:8])
    season = 0.5 + 0.5 * np.sin(2 * np.pi * (when.dayofyear - 80) / 365.25)
    factor = np.linspace(0.6, 1.4, len(BOA_BANDS))[:, None, None]
    boa = (500 + 3500 * season * factor + rng.normal(0, 40, (len(BOA_BANDS), N, N))).astype("int16")
    boa[:, 0, 0] = FILL            # no data in every band
    boa[3, 1, 3] = FILL            # one band only
    qai = np.zeros((N, N), "int16")
    qai[1, 1] = 4                  # cloud state 2 (opaque)
    qai[2, 2] = 2                  # cloud state 1 (buffer)
    qai[3, 3] = 8                  # shadow
    qai[3, 5] = 16                 # snow
    qai[5, 5] = 32                 # water: does not mask
    qai[6, 6] = 1                  # bit 0, no data
    qai[7, seed % N] = 6           # cloud state 3, somewhere that changes by scene
    qai[4, 4] = 64                 # aerosol flag: does not mask
    x = 500000 + shift + 10 * (np.arange(N) + 0.5)
    y = 8000000 - 10 * (np.arange(N) + 0.5)
    coords = {"y": y, "x": x}
    ds = xr.Dataset({
        "boa": xr.DataArray(boa, dims=("band", "y", "x"), coords={"band": list(BOA_BANDS), **coords}),
        "qai": xr.DataArray(qai, dims=("y", "x"), coords=coords),
        "dst": xr.DataArray(rng.integers(0, 5, (N, N)).astype("int16"), dims=("y", "x"), coords=coords),
    }).rio.write_crs("EPSG:32723")
    ds["boa"].attrs["_FillValue"] = FILL
    water = np.nan if when.month == 3 else 1.0 + when.dayofyear / 100
    ds.attrs.update(sensor="MSI", aod_green=np.float64(0.05 + when.dayofyear / 1000), water_vapor_cm=np.float64(water),
                    fmask_cloud_pct=np.float64(when.month), fmask_version="4.7", n_targets=np.int64(7))
    return ds


class Stub:
    def __init__(self):
        self.calls, self.fail, self.shift, self.find = [], set(), set(), None

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


@pytest.fixture
def stub(monkeypatch):
    s = Stub()

    def find_l1c(**kw):
        s.calls.append(("find_l1c", kw))
        return s.find

    def download_l1c(product, dest=None, n_jobs=4):
        s.calls.append(("download_l1c", product))
        root = Path(dest) / product
        root.mkdir(parents=True, exist_ok=True)
        (root / "MTD_MSIL1C.xml").write_text("x")
        return root

    def download_dem(area, dest=None, n_jobs=4):
        s.calls.append(("download_dem",))
        Path(dest).mkdir(parents=True, exist_ok=True)
        path = Path(dest) / "dem.tif"
        path.write_bytes(b"dem")
        return path

    def level2(safe, dem=None, **kw):
        s.calls.append(("level2", Path(safe).name))
        if Path(safe).name in s.fail:
            raise RuntimeError("boom")
        return scene_ds(safe, shift=10.0 if Path(safe).name in s.shift else 0.0)

    monkeypatch.setattr(_s2fetch, "find_l1c", find_l1c)
    monkeypatch.setattr(_s2fetch, "download_l1c", download_l1c)
    monkeypatch.setattr(_s2fetch, "download_dem", download_dem)
    monkeypatch.setattr(_level2, "level2", level2)
    return s


DATES = ["2022-03-14", "2022-01-05", "2022-02-09", "2022-04-20", "2022-05-12"]
NAMES = [name(d) for d in DATES]


def make(stub, tmp_path, names=NAMES):
    out = tmp_path / "out"
    run_series(out_dir=out, products=names, verbose=False)
    return out


def folder(out, product):
    return out / product[:-5]


def hand(out, names=NAMES, mask=True):
    """The cube by hand: each scene's tifs read with load_raster, the fill and the QAI applied with numpy."""
    layers = []
    for product in sorted(names, key=lambda n: n.split("_")[2]):
        boa = zeit.load_raster(folder(out, product) / "boa.tif").values
        qai = zeit.load_raster(folder(out, product) / "qai.tif").values
        keep = (boa != FILL)
        if mask:
            keep = keep & zeit.qa_mask(qai, "force")[None]
        layers.append(np.where(keep, boa, np.nan).astype("float32"))
    return np.stack(layers)


def test_cube_equals_the_one_built_by_hand(stub, tmp_path):
    out = make(stub, tmp_path)
    cube = zeit.load_level2_series(out)
    assert cube.dims == ("time", "band", "y", "x") and cube.dtype == np.float32 and cube.name == "boa"
    want = hand(out)
    np.testing.assert_array_equal(cube.values, want)
    assert np.isnan(want).any() and np.isfinite(want).any()
    # the QAI screening took cloud 1-3, shadow, snow and bit 0, not water or the aerosol flag
    first = cube.isel(time=0).values            # 2022-01-05
    for r, c in [(1, 1), (2, 2), (3, 3), (3, 5), (6, 6)]:
        assert np.isnan(first[:, r, c]).all()
    for r, c in [(5, 5), (4, 4)]:
        assert np.isfinite(first[:, r, c]).all()
    assert np.isnan(first[:, 0, 0]).all() and np.isnan(first[3, 1, 3]) and np.isfinite(first[2, 1, 3])
    # mask=None: only the fill
    raw = zeit.load_level2_series(out, mask=None)
    np.testing.assert_array_equal(raw.values, hand(out, mask=False))
    assert np.isfinite(raw.isel(time=0).values[:, 1, 1]).all()


def test_time_coordinates_and_attrs(stub, tmp_path):
    out = make(stub, tmp_path)
    cube = zeit.load_level2_series(out)
    assert cube.time.dtype == np.dtype("datetime64[ns]")
    assert list(cube.time.values) == sorted(pd.Timestamp(d + " 13:12:39").to_datetime64() for d in DATES)
    assert list(cube.band.values) == list(BOA_BANDS)
    order = sorted(NAMES, key=lambda n: n.split("_")[2])
    assert list(cube["product"].values) == order
    assert set(cube["platform"].values) == {"S2B"} and set(cube["baseline"].values) == {"05.09"}
    assert set(cube["sensor"].values) == {"MSI"}
    assert cube["aod_green"].dims == ("time",) and np.isfinite(cube["aod_green"].values).all()
    assert cube["fmask_cloud_pct"].values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert "fmask_version" not in cube.coords and "n_targets" not in cube.coords
    wv = cube["water_vapor_cm"].values
    assert np.isnan(wv[2]) and np.isfinite(np.delete(wv, 2)).all()      # March's scene stored NaN
    assert cube.attrs["tile"] == "23LKC" and cube.attrs["source"] == "zeit.level2"
    assert cube.rio.crs is not None and cube.rio.crs.to_epsg() == 32723
    assert cube.rio.transform().a == 10 and cube.rio.transform().c == 500000
    assert cube.rio.transform().f == 8000000


def test_save_roundtrip_has_ten_named_bands(stub, tmp_path):
    out = make(stub, tmp_path)
    with rasterio.open(folder(out, NAMES[0]) / "boa.tif") as src:
        assert src.count == 10 and list(src.descriptions) == list(BOA_BANDS)
    cube = zeit.load_level2_series(out).isel(time=slice(0, 2))
    written = cube.zeit.save(tmp_path / "cube.tif")
    with rasterio.open(written) as src:
        assert src.count == 20
    assert zeit.load_level2_series(out, products=NAMES[:1]).sizes["time"] == 1


def test_lazy(stub, tmp_path):
    out = make(stub, tmp_path)
    cube = zeit.load_level2_series(out, chunks={"y": 4, "x": 4})
    assert isinstance(cube.data, dask.array.Array) and cube.chunks is not None
    assert cube.chunksizes["y"] == (4, 4) and cube.chunksizes["time"] == (1,) * 5
    one = cube.isel(time=1).values
    np.testing.assert_array_equal(one, hand(out)[1])
    assert isinstance(zeit.load_level2_series(out).data, dask.array.Array)


def test_failed_scenes_products_and_errors(stub, tmp_path):
    stub.fail = {NAMES[0]}
    with pytest.warns(UserWarning, match="failed"):
        out = make(stub, tmp_path)
    assert not folder(out, NAMES[0]).exists()
    ok = NAMES[1:]
    cube = zeit.load_level2_series(out)
    assert cube.sizes["time"] == 4 and name("2022-03-14") not in set(cube["product"].values)
    part = zeit.load_level2_series(out, products=[NAMES[1], NAMES[3][:-5]])
    assert list(part["product"].values) == [NAMES[1], NAMES[3]]
    with pytest.raises(KeyError, match="S2B_MSIL1C_20220314"):
        zeit.load_level2_series(out, products=[NAMES[0]])
    with pytest.raises(ValueError, match="mask"):
        zeit.load_level2_series(out, mask="fmask")
    assert len(ok) == 4


def test_no_scene(tmp_path):
    with pytest.raises(ValueError, match="no level2 scene"):
        zeit.load_level2_series(tmp_path)
    (tmp_path / "dem").mkdir()
    (tmp_path / ".safe").mkdir()
    (tmp_path / "S2B_nomarker").mkdir()
    with pytest.raises(ValueError, match="no level2 scene"):
        zeit.load_level2_series(tmp_path)


def test_other_folders_and_unreadable_markers_are_ignored(stub, tmp_path):
    out = make(stub, tmp_path)
    (out / "dem").mkdir(exist_ok=True)
    (out / ".part-x").mkdir()
    (out / ".part-x" / MARKER).write_text("{}")
    (out / "stray").mkdir()
    (out / "broken").mkdir()
    (out / "broken" / MARKER).write_text("not json")
    assert zeit.load_level2_series(out).sizes["time"] == 5


def test_grid_mismatch_lists_the_odd_one(stub, tmp_path):
    stub.shift = {NAMES[2]}
    out = make(stub, tmp_path)
    with pytest.raises(ValueError, match=NAMES[2][:-5]) as err:
        zeit.load_level2_series(out)
    assert NAMES[0][:-5] not in str(err.value)
    # like= puts them on one grid
    like = xr.DataArray(np.zeros((N, N), "float32"), dims=("y", "x"),
                        coords={"y": 8000000 - 10 * (np.arange(N) + 0.5), "x": 500000 + 10 * (np.arange(N) + 0.5)}
                        ).rio.write_crs("EPSG:32723")
    cube = zeit.load_level2_series(out, like=like)
    assert cube.sizes["time"] == 5
    odd = cube.sel(time=cube["product"] == NAMES[2])
    # the scene shifted 10 m east does not cover the first column of this grid (NaN); the rest is its file's
    assert np.isnan(odd.values[..., 0]).all()
    own = zeit.load_raster(folder(out, NAMES[2]) / "boa.tif").values
    got = odd.values[0]
    for r, c in [(2, 1), (5, 6), (7, 3)]:
        assert got[0, r, c] == own[0, r, c - 1]


def coarse_like():
    # 20 m cells whose centres are the centres of the 10 m pixels (2r+1, 2c+1)
    return xr.DataArray(np.zeros((4, 4), "float32"), dims=("y", "x"),
                        coords={"y": 8000000 - 15 - 20 * np.arange(4), "x": 500015 + 20 * np.arange(4)}
                        ).rio.write_crs("EPSG:32723")


def test_like_a_coarser_grid(stub, tmp_path):
    out = make(stub, tmp_path)
    like = coarse_like()
    for resampling in ("auto", "nearest", "bilinear"):
        cube = zeit.load_level2_series(out, like=like, resampling=resampling)
        assert cube.shape == (5, 10, 4, 4) and cube.rio.transform().a == 20
        np.testing.assert_array_equal(cube.x.values, like.x.values)
        for t, product in enumerate(sorted(NAMES, key=lambda n: n.split("_")[2])):
            boa = zeit.load_raster(folder(out, product) / "boa.tif", like=like,
                                   resampling="nearest" if resampling == "auto" else resampling).values
            qai = zeit.load_raster(folder(out, product) / "qai.tif", like=like, resampling="nearest").values
            keep = (boa != FILL) & zeit.qa_mask(qai, "force")[None]
            np.testing.assert_array_equal(cube.values[t], np.where(keep, boa, np.nan).astype("float32"))
    # the QAI took the nearest 10 m pixel (2r+1, 2c+1) even with a bilinear boa: its codes are not interpolated
    cube = zeit.load_level2_series(out, like=like, resampling="bilinear")
    first = cube.isel(time=0).values            # the scene of 2022-01-05
    assert np.isnan(first[:, 0, 0]).all()       # cloud at (1, 1)
    assert np.isnan(first[:, 1, 1]).all()       # shadow at (3, 3)
    assert np.isfinite(first[:, 2, 2]).all()    # water at (5, 5) does not mask
    assert np.isnan(first[:, 1, 2]).all()       # snow at (3, 5)
    assert np.isnan(first[:, 3, 3]).all()       # bit 0 at (7, 7)
    nearest = zeit.load_level2_series(out, like=like, resampling="nearest")
    src = zeit.load_raster(folder(out, NAMES[1]) / "boa.tif").values
    assert nearest.isel(time=0).values[2, 2, 2] == src[2, 5, 5]


def test_lazy_with_like(stub, tmp_path):
    out = make(stub, tmp_path)
    cube = zeit.load_level2_series(out, like=coarse_like())
    assert isinstance(cube.data, dask.array.Array) and cube.dtype == np.float32


@pytest.fixture(scope="module")
def long_series(tmp_path_factory):
    out = tmp_path_factory.mktemp("long") / "out"
    dates = pd.date_range("2021-01-03", periods=36, freq="20D")
    names = [name(d) for d in dates]
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(_level2, "level2", lambda safe, dem=None, **kw: scene_ds(safe))
        mp.setattr(_s2fetch, "download_l1c", lambda product, dest=None, n_jobs=4: _fake_safe(dest, product))
        mp.setattr(_s2fetch, "download_dem", lambda area, dest=None, n_jobs=4: _fake_dem(dest))
        run_series(out_dir=out, products=names, verbose=False)
    finally:
        mp.undo()
    return out, names


def _fake_safe(dest, product):
    root = Path(dest) / product
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fake_dem(dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    path = Path(dest) / "dem.tif"
    path.write_bytes(b"dem")
    return path


def by_hand(cube):
    return xr.DataArray(cube.values, dims=cube.dims,
                        coords={d: cube[d].values for d in cube.dims}).rio.write_crs(cube.rio.crs)


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_the_cube_goes_straight_into_the_analysis_functions(long_series):
    out, names = long_series
    cube = zeit.load_level2_series(out, chunks={"y": 4, "x": 4})
    assert cube.sizes["time"] == 36 and len(np.unique(cube.time.dt.year.values)) == 2
    ref = by_hand(cube)
    metrics = zeit.temporal_metrics(cube, ["median", "mean", "count"], band="NIR", by="quarter")
    same(metrics, zeit.temporal_metrics(ref, ["median", "mean", "count"], band="NIR", by="quarter"))
    assert metrics.shape[-3:] != () and np.isfinite(metrics.values).any()
    regular = zeit.regularize_time_series(cube.sel(band=["RED", "NIR"]), freq="60D", method="median")
    same(regular, zeit.regularize_time_series(ref.sel(band=["RED", "NIR"]), freq="60D", method="median"))
    assert regular.sizes["time"] > 10
    bands = ["GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
    seg = zeit.ccdc(cube.sel(band=bands), detection_bands=["GREEN", "NIR", "SWIR1"], n_jobs=2)
    want = zeit.ccdc(ref.sel(band=bands), detection_bands=["GREEN", "NIR", "SWIR1"], n_jobs=2)
    assert list(seg.data_vars) == list(want.data_vars)
    for key in seg.data_vars:
        same(seg[key], want[key])


def frame(names):
    return pd.DataFrame({"product": names, "datetime": [pd.Timestamp(n.split("_")[2]) for n in names],
                         "tile": "23LKC", "cloud_cover": 5.0, "baseline": "05.09", "platform": "sentinel-2b"})


def test_level2_series_end_to_end_and_resume(stub, tmp_path):
    stub.find = frame(NAMES)
    stub.fail = {NAMES[3]}
    with pytest.warns(UserWarning, match="1 of 5 scenes failed"):
        cube = zeit.level2_series("23LKC", "2022-01-01", "2022-06-01", tmp_path, verbose=False)
    assert cube.sizes["time"] == 4 and cube.attrs["failed"] == [NAMES[3]]
    assert cube.attrs["journal"] == str(tmp_path / JOURNAL) and (tmp_path / JOURNAL).is_file()
    assert NAMES[3] not in set(cube["product"].values)
    assert isinstance(cube.data, dask.array.Array)
    np.testing.assert_array_equal(cube.values, hand(tmp_path, [n for n in NAMES if n != NAMES[3]]))
    # second call: nothing is processed again, but the failed scene is retried (now it works)
    stub.calls.clear()
    stub.fail = set()
    again = zeit.level2_series("23LKC", "2022-01-01", "2022-06-01", tmp_path, verbose=False)
    assert [c[1] for c in stub.of("level2")] == [NAMES[3]]
    assert again.sizes["time"] == 5 and again.attrs["failed"] == []
    # third: all skipped, level2 never called, same cube
    stub.calls.clear()
    third = zeit.level2_series("23LKC", "2022-01-01", "2022-06-01", tmp_path, verbose=False)
    assert stub.of("level2") == [] and stub.of("download_l1c") == []
    xr.testing.assert_identical(third.drop_vars([]), again)
    np.testing.assert_array_equal(third.values, again.values)
    np.testing.assert_array_equal(third.values, zeit.load_level2_series(tmp_path).values)


def test_level2_series_mask_like_and_no_success(stub, tmp_path):
    cube = zeit.level2_series(out_dir=tmp_path / "a", products=NAMES[:2], mask=None, like=coarse_like(),
                              resampling="nearest", verbose=False)
    assert cube.shape == (2, 10, 4, 4) and cube.attrs["failed"] == []
    stub.fail = set(NAMES[:2])
    with pytest.warns(UserWarning, match="failed"):
        with pytest.raises(RuntimeError, match=JOURNAL):
            zeit.level2_series(out_dir=tmp_path / "b", products=NAMES[:2], verbose=False)
    with pytest.raises(ValueError, match="mask"):
        zeit.level2_series(out_dir=tmp_path / "c", products=NAMES[:1], mask="x", verbose=False)
    assert not (tmp_path / "c").exists()


def test_marker_has_what_the_cube_reads(stub, tmp_path):
    out = make(stub, tmp_path)
    marker = json.loads((folder(out, NAMES[0]) / MARKER).read_text())
    assert marker["datetime"].startswith("2022-03-14 13:12:39") and np.isnan(float(marker["attrs"]["water_vapor_cm"]))
    assert _level2_series.MARKER == MARKER
