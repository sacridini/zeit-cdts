"""Landsat compositing fixes: the QA_PIXEL mask, the platform and SLC-off filter, the native grid, ``n_clear``
and ``attrs["scenes"]``, the read-error warning. No network: STAC items and stackstac are stand-ins."""
import json
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.errors import RasterioIOError
from stackstac.nodata_reader import NodataReader
from stackstac.rio_reader import AutoParallelRioReader

import zeit.cube as cube_mod
from zeit.cube import (
    READ_ERRORS, ReadErrorWarning, _ZeitReader, _WatchedDataset, _clear_mask, _composite_block,
    _filter_platforms, _native_lattice, _qa_states, _snap_to_lattice, build_annual_composites,
    build_time_series, BAP_CLEAR, BAP_CLOUD, BAP_NO_DATA,
)

FILL, DILATED, CIRRUS, CLOUD, SHADOW, SNOW, CLEAR, WATER = (1 << k for k in range(8))


# ------------------------------------------------------------------------------------------ the QA mask
def test_landsat_qa_pixel_bits():
    cases = {
        "clear": (CLEAR, True),
        "clear land L8 (21824)": (21824, True),
        "clear + snow": (CLEAR | SNOW, True),
        "clear + water": (CLEAR | WATER, True),
        "clear + shadow": (CLEAR | SHADOW, False),     # the clear bit alone keeps the shadow
        "clear + dilated cloud": (CLEAR | DILATED, False),
        "clear + cirrus": (CLEAR | CIRRUS, False),
        "clear + cloud": (CLEAR | CLOUD, False),
        "clear + fill": (CLEAR | FILL, False),
        "fill": (FILL, False),
        "no clear bit": (0, False),
        "snow without the clear bit": (SNOW, False),
    }
    qa = np.array([v for v, _ in cases.values()], dtype="uint16")
    assert _clear_mask(qa, "qa_pixel").tolist() == [ok for _, ok in cases.values()], list(cases)
    assert _clear_mask(qa, "QA_PIXEL").tolist() == [ok for _, ok in cases.values()]


def test_other_masks_unchanged():
    assert _clear_mask(np.array([4, 9, 3, 11], dtype="uint16"), "SCL").tolist() == [True, False, False, True]
    fmask = np.array([0, 2, 8, 255], dtype="uint16")
    assert _clear_mask(fmask, "Fmask").tolist() == [True, False, False, False]


def test_bap_states_with_the_stricter_mask():
    qa = np.array([CLEAR, CLEAR | SHADOW, CLOUD, FILL, 0], dtype="uint16")
    assert _qa_states(qa, "qa_pixel").tolist() == [BAP_CLEAR, BAP_CLOUD, BAP_CLOUD, BAP_NO_DATA, BAP_NO_DATA]


def test_composite_ignores_a_shadow_that_the_clear_bit_keeps():
    raw = np.zeros((3, 2, 1, 1), dtype="uint16")
    raw[:, 0, 0, 0] = [1000, 1000, 9000]           # the third scene is a shadow with the clear bit set
    raw[:, 1, 0, 0] = [CLEAR, CLEAR, CLEAR | SHADOW]
    one = np.ones((3, 1), dtype="float32")
    got = _composite_block(raw, [0], 1, "qa_pixel", one, 0 * one, "median")
    assert got[0, 0, 0] == 1000


# ---------------------------------------------------------------------------------- platforms and SLC-off
def _scene(item_id, platform, date, code="EPSG:32621", transform=(30.0, 0.0, 434685.0, 0.0, -30.0, -1322685.0)):
    props = {"platform": platform, "datetime": f"{date}T13:51:00Z", "proj:code": code, "proj:transform": list(transform)}
    return SimpleNamespace(id=item_id, properties=props, datetime=pd.Timestamp(props["datetime"]).to_pydatetime(),
                           assets={})


ITEMS = [
    _scene("l5", "landsat-5", "2010-07-02"),
    _scene("l7-ok", "landsat-7", "2002-07-02"),
    _scene("l7-off", "landsat-7", "2010-07-09"),
    _scene("l7-day", "landsat-7", "2003-05-31"),
    _scene("l8", "landsat-8", "2019-07-24"),
    _scene("l9", "landsat-9", "2022-07-24"),
]


def _ids(items):
    return [i.id for i in items]


def test_slc_off_dropped_by_default():
    assert _ids(_filter_platforms(ITEMS, None, False)) == ["l5", "l7-ok", "l8", "l9"]
    assert _ids(_filter_platforms(ITEMS, None, True)) == _ids(ITEMS)


def test_platform_filter():
    assert _ids(_filter_platforms(ITEMS, ["landsat-8", "landsat-9"], False)) == ["l8", "l9"]
    assert _ids(_filter_platforms(ITEMS, ["LANDSAT_7"], True)) == ["l7-ok", "l7-off", "l7-day"]
    assert _ids(_filter_platforms(ITEMS, ["landsat-7"], False)) == ["l7-ok"]
    assert _ids(_filter_platforms(ITEMS, "landsat-5", False)) == ["l5"]
    assert _filter_platforms([SimpleNamespace(id="x", properties={})], ["landsat-8"], False) == []


class _Stop(Exception):
    pass


def _catalog(items):
    class Search:
        def item_collection(self):
            return list(items)

    class Catalog:
        def search(self, **kw):
            return Search()

    return Catalog()


def _stack_kwargs(monkeypatch, items, **kw):
    """The keyword arguments `build_time_series` gives stackstac for these items."""
    got = {}

    def stack(its, **k):
        got["items"] = list(its)
        got.update(k)
        raise _Stop

    monkeypatch.setattr(cube_mod.Client, "open", staticmethod(lambda url: _catalog(items)))
    monkeypatch.setattr(cube_mod.stackstac, "stack", stack)
    with pytest.raises(_Stop):
        build_time_series(collection="landsat-c2-l2", bands=["red"], **kw)
    return got


def test_build_time_series_filters_before_reading(monkeypatch):
    got = _stack_kwargs(monkeypatch, ITEMS, bbox=[-55.8, -12.6, -55.7, -12.5], platforms=["landsat-7", "landsat-8"])
    assert _ids(got["items"]) == ["l7-ok", "l8"]
    got = _stack_kwargs(monkeypatch, ITEMS, bbox=[-55.8, -12.6, -55.7, -12.5], slc_off=True)
    assert _ids(got["items"]) == _ids(ITEMS)
    with pytest.raises(ValueError, match="No images found"):
        _stack_kwargs(monkeypatch, ITEMS[2:3], bbox=[-55.8, -12.6, -55.7, -12.5])


# ----------------------------------------------------------------------------------------- the native grid
def test_native_lattice_and_snap():
    assert _native_lattice(ITEMS, 32621, 30) == (15.0, 15.0)
    # the southern twin of the zone: the false northing of 10,000,000 m (= 10 mod 30) moves y only
    assert _native_lattice(ITEMS, 32721, 30) == (15.0, 25.0)
    assert _native_lattice(ITEMS, 4326, 30) is None
    assert _native_lattice(ITEMS, 32622, 30) is None      # another zone
    assert _native_lattice(ITEMS, 32621, 10) is None      # another resolution
    assert _native_lattice([SimpleNamespace(id="x", properties={}, assets={})], 32621, 30) is None
    assert _snap_to_lattice((434700.0, 100.0, 434800.0, 200.0), (15.0, 15.0), 30) == (434685.0, 75.0, 434805.0, 225.0)
    # a bound exactly on an edge stays
    assert _snap_to_lattice((434685.0, 75.0, 434805.0, 225.0), (15.0, 15.0), 30) == (434685.0, 75.0, 434805.0, 225.0)


def test_build_time_series_native_grid(monkeypatch):
    bbox = [-55.8, -12.6, -55.7, -12.5]
    got = _stack_kwargs(monkeypatch, ITEMS, bbox=bbox, resolution=30, epsg=32621)
    x0, y0, x1, y1 = got["bounds"]
    assert x0 % 30 == 15 and x1 % 30 == 15 and y0 % 30 == 15 and y1 % 30 == 15
    assert got["snap_bounds"] is False and "bounds_latlon" not in got
    # the bbox is inside (the bounds are only widened, by less than a pixel each side)
    from pyproj import Transformer
    mx, my = Transformer.from_crs(4326, 32621, always_xy=True).transform(-55.7, -12.5)
    assert x0 <= Transformer.from_crs(4326, 32621, always_xy=True).transform(-55.8, -12.6)[0] + 1e-6
    assert x1 >= mx - 1e-6 and y1 >= my - 1e-6 and (x1 - x0) % 30 == 0 and (y1 - y0) % 30 == 0

    # the southern twin: x on the lattice, y at 25 mod 30
    got = _stack_kwargs(monkeypatch, ITEMS, bbox=bbox, resolution=30, epsg=32721)
    x0, y0, x1, y1 = got["bounds"]
    assert x0 % 30 == 15 and y0 % 30 == 25 and y1 % 30 == 25


def test_build_time_series_keeps_the_old_grid_otherwise(monkeypatch):
    bbox = [-55.8, -12.6, -55.7, -12.5]
    for kw in (dict(resolution=30, epsg=32621, align="resolution"),   # asked for
               dict(resolution=30, epsg=4326),                        # not the items' CRS
               dict(resolution=10, epsg=32621),                       # not the native resolution
               dict(epsg=32621)):                                     # no resolution
        got = _stack_kwargs(monkeypatch, ITEMS, bbox=bbox, **kw)
        assert got["bounds_latlon"] == bbox and "bounds" not in got and "snap_bounds" not in got, kw
    with pytest.raises(ValueError, match="align"):
        build_time_series(bbox=bbox, align="nearest")


# ------------------------------------------------------------------------------------ n_clear and scenes
def _qa_cube(qa, values=(1000, 2000)):
    """(time, band, y, x) uint16 cube [b1, b2, qa_pixel] with the given QA planes (time, y, x)."""
    t, ny, nx = qa.shape
    raw = np.zeros((t, 3, ny, nx), dtype="uint16")
    raw[:, 0], raw[:, 1], raw[:, 2] = values[0], values[1], qa
    return xr.DataArray(
        da.from_array(raw, chunks=(1, 1, 3, 3)), dims=("time", "band", "y", "x"),
        coords={"time": pd.date_range("2019-06-05", periods=t, freq="10D"),
                "id": ("time", [f"LC08_scene{i}" for i in range(t)]), "band": ["b1", "b2", "qa_pixel"],
                "y": np.arange(ny) * -30.0 + 100, "x": np.arange(nx) * 30.0,
                "scale": (("time", "band"), np.full((t, 3), 1e-4, dtype="float32")),
                "offset": (("time", "band"), np.zeros((t, 3), dtype="float32")), "epsg": 32621},
        attrs={"crs": "epsg:32621", "resolution": 30},
    )


def _annual(monkeypatch, qa_by_year, **kw):
    def fake(**k):
        year = int(k["start_date"][:4])
        if year not in qa_by_year:
            raise ValueError("No images found for the given criteria.")
        return _qa_cube(qa_by_year[year])

    monkeypatch.setattr(cube_mod, "build_time_series", fake)
    return build_annual_composites(collection="landsat-c2-l2", bbox=[0, 0, 1, 1], start_year=2019, end_year=2020,
                                   bands=["b1", "b2"], **kw)


def test_n_clear_and_scenes(monkeypatch):
    qa = np.full((4, 5, 6), CLEAR, dtype="uint16")
    qa[0, 0, 0] = CLEAR | SHADOW      # a shadow: not counted
    qa[1, 0, 0] = CLOUD
    qa[2, 0, 0] = FILL
    qa[:, 1, 1] = 0                   # no clear observation
    qa[1, 2, 2] = CLEAR | SNOW        # snow counts
    for method in ("median", "medoid", "bap"):
        out = _annual(monkeypatch, {2019: qa}, method=method, clear_count=True)
        assert out.dims == ("time", "band", "y", "x") and out.shape == (2, 2, 5, 6)
        n = out.n_clear
        assert n.dims == ("time", "y", "x") and n.dtype == np.int16, method
        got = n.values
        ref = ((qa & 0b11111) == 0) & ((qa & CLEAR) > 0)
        np.testing.assert_array_equal(got[0], ref.sum(0))
        assert got[0, 0, 0] == 1 and got[0, 1, 1] == 0 and got[0, 2, 2] == 4
        assert (got[1] == 0).all()               # 2020 had no scenes
        assert np.isnan(out.values[0, :, 1, 1]).all() and np.isfinite(out.values[0, :, 0, 0]).all()
        assert json.loads(out.attrs["scenes"]) == {"2019": ["LC08_scene0", "LC08_scene1", "LC08_scene2",
                                                            "LC08_scene3"], "2020": []}
    # the composite and the count computed together are one read of the same graph
    assert out.n_clear.data.dask is not None


def test_n_clear_is_opt_in(monkeypatch):
    out = _annual(monkeypatch, {2019: np.full((2, 5, 6), CLEAR, dtype="uint16")})
    assert "n_clear" not in out.coords and out.shape == (2, 2, 5, 6)
    assert json.loads(out.attrs["scenes"])["2019"] == ["LC08_scene0", "LC08_scene1"]


def test_n_clear_and_scenes_survive_saving(monkeypatch, tmp_path):
    import zeit

    out = _annual(monkeypatch, {2019: np.full((2, 5, 6), CLEAR, dtype="uint16"),
                                2020: np.full((2, 5, 6), CLEAR, dtype="uint16")}, clear_count=True)
    zeit.save_raster(out.isel(band=0), tmp_path / "c.tif")
    ds = out.compute().to_dataset(name="composite")
    ds.to_netcdf(tmp_path / "c.nc")
    back = xr.open_dataset(tmp_path / "c.nc")
    assert json.loads(back["composite"].attrs["scenes"])["2020"] == ["LC08_scene0", "LC08_scene1"]
    assert (back["n_clear"].values == 2).all()
    back.close()


# ------------------------------------------------------------------------------------------ the warning
class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def read(self, window=None, **kw):
        raise self.exc

    def close(self):
        pass


def test_tolerated_read_error_warns_and_becomes_nodata():
    from rasterio.windows import Window

    READ_ERRORS.clear()
    url = "https://x.blob.core.windows.net/landsat/LC08_L2SP_227079_20220630_02_T1_SR_B2.TIF?sig=secret"
    ds = _WatchedDataset(_Raises(RasterioIOError("x.TIF: TIFFReadEncodedTile() failed. IReadBlock failed at X offset 1")),
                         url, cube_mod.ERRORS_AS_NODATA)
    with pytest.warns(ReadErrorWarning, match=r"LC08_L2SP_227079_20220630_02_T1_SR_B2\.TIF") as rec:
        block = ds.read(window=Window(0, 0, 4, 3), out_dtype=np.dtype("uint16"), masked=True)
    assert "secret" not in str(rec[0].message)
    assert block.shape == (1, 3, 4) and block.mask.all()
    assert list(READ_ERRORS) == [url.split("?")[0]]
    # one warning per file, however many blocks fail
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", cube_mod.ReadErrorWarning)
        ds.read(window=Window(4, 0, 4, 3), out_dtype=np.dtype("uint16"), masked=True)


def test_access_errors_still_raise():
    ds = _WatchedDataset(_Raises(RasterioIOError("AccessDenied: Requester Pays")), "u.tif", cube_mod.ERRORS_AS_NODATA)
    with pytest.raises(RasterioIOError):
        ds.read(window=None, out_dtype=np.dtype("uint16"), masked=True)


def test_unopenable_file_warns(monkeypatch):
    READ_ERRORS.clear()
    monkeypatch.setattr(AutoParallelRioReader, "_open",
                        lambda self: NodataReader(dtype=self.dtype, fill_value=self.fill_value))
    reader = _ZeitReader(url="https://x/LE07_L2SP_227079_20220630_02_T1_SR_B1.TIF", spec=None, resampling=None,
                         dtype=np.dtype("uint16"), fill_value=0, scale_offset=(1, 0))
    with pytest.warns(ReadErrorWarning, match="LE07_L2SP_227079_20220630_02_T1_SR_B1.TIF.*could not be opened"):
        assert isinstance(reader._open(), NodataReader)


def test_build_time_series_uses_the_reporting_reader(monkeypatch):
    got = _stack_kwargs(monkeypatch, ITEMS, bbox=[0, 0, 1, 1])
    assert got["reader"] is _ZeitReader and got["errors_as_nodata"] is cube_mod.ERRORS_AS_NODATA
