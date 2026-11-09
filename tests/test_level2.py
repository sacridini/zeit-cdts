"""The reading side of the atmospheric correction of a Sentinel-2 scene (zeit._level2): the .SAFE directory, the bands put on the
10 m grid by nearest neighbour as FORCE's read_level1 does, the geography of the coarse corners and the DEM on the grid of the
scene. The correction itself (zeit_core::l2scene) is tested in Rust and compared with FORCE's force-l2ps on real scenes by
benchmarks/vs_originals/force_l2ps_real.py."""
import os
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from zeit import _level2, _s2fetch

CRS = "EPSG:32723"


def write(path, data, res, ulx=199980.0, uly=8300020.0, crs=CRS, nodata=None):
    h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype=data.dtype, crs=crs,
                       transform=from_origin(ulx, uly, res, res), nodata=nodata) as d:
        d.write(data, 1)


def test_a_directory_that_is_not_a_safe_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a Sentinel-2 L1C .SAFE"):
        _level2.Safe(tmp_path)
    (tmp_path / "MTD_MSIL1C.xml").write_text("<x/>")
    (tmp_path / "GRANULE" / "L1C_T23LKC_x").mkdir(parents=True)
    with pytest.raises(ValueError, match="no MTD_TL.xml"):
        _level2.Safe(tmp_path)


def test_the_bands_are_read_in_their_window_and_replicated_to_the_finest_grid(tmp_path):
    # a 12 x 12 scene of 10 m: the 20 m bands have 6 x 6 cells, the 60 m bands 2 x 2; the window skips a column and a row
    rng = np.random.default_rng(1)
    bands = {}
    files = []
    for name in _level2.BAND_NAMES:
        res = 60.0 if name in ("01", "09", "10") else 10.0 if name in ("02", "03", "04", "08") else 20.0
        n = int(round(130 * 10 / res))  # 130 x 130 pixels of 10 m in the tile
        a = rng.integers(1, 60000, (n, n), dtype=np.uint16)
        bands[name] = (a, res)
        p = tmp_path / f"T23LKC_20230728T132239_B{name}.jp2"
        write(p, a, res)
        files.append(str(p))
    safe = types.SimpleNamespace(bands=files)
    # the image: 120 x 120 pixels at (10, 10) of the 10 m tile; the windows in pixels of each resolution
    desc = {"nrows": 120, "ncols": 120, "windows": [(10.0, (10, 10, 120, 120)), (20.0, (5, 5, 60, 60)), (60.0, (1, 1, 20, 20))]}
    out = _level2.read_bands(safe, desc)
    assert out.shape == (13, 120, 120) and out.dtype == np.uint16
    for b, name in enumerate(_level2.BAND_NAMES):
        a, res = bands[name]
        k = int(res // 10)
        win = {10.0: (10, 10, 120, 120), 20.0: (5, 5, 60, 60), 60.0: (1, 1, 20, 20)}[res]
        part = a[win[1]:win[1] + win[3], win[0]:win[0] + win[2]]
        assert np.array_equal(out[b], np.repeat(np.repeat(part, k, axis=0), k, axis=1)), name


def test_the_geography_of_the_corners_is_the_projection_of_the_grid_points(tmp_path):
    p = tmp_path / "B02.jp2"
    write(p, np.ones((10, 10), np.uint16), 10.0)
    safe = types.SimpleNamespace(bands=[str(p)])
    desc = {"geo_grid": (199980.0, 8300020.0, 5000.0, 3, 3)}
    lon, lat = _level2.geo_tables(safe, desc)
    assert lon.shape == lat.shape == (9,)
    # UTM 23 S: the central meridian is 45 W; the corners go east and south, the latitude falls with the row
    assert -48 < lon[0] < -47 and lon[1] > lon[0] and lat[3] < lat[0]
    from pyproj import Transformer

    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    x, y = t.transform(199980.0 + 5000.0, 8300020.0 - 10000.0)
    assert abs(lon[2 * 3 + 1] - x) < 1e-7 and abs(lat[2 * 3 + 1] - y) < 1e-7


def test_the_dem_is_warped_to_the_grid_with_the_no_data_value(tmp_path):
    p = tmp_path / "B02.jp2"
    write(p, np.ones((100, 100), np.uint16), 10.0)
    # a DEM in geographic coordinates over the scene (about 47.8 W, 15.4 S): a ramp in longitude, with a hole of no data
    dem = np.tile(np.linspace(800, 900, 60, dtype=np.float32), (60, 1))
    dem[:, 30:40] = -32767  # about 47.805 to 47.800 W: inside the scene
    d = tmp_path / "dem.tif"
    with rasterio.open(d, "w", driver="GTiff", height=60, width=60, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(-47.82, -15.34, 0.0005, 0.0005), nodata=-32767) as f:
        f.write(dem, 1)
    safe = types.SimpleNamespace(bands=[str(p)])
    out = _level2.dem_on_grid(str(d), safe)
    assert out.shape == (100, 100) and out.dtype == np.float32
    valid = out != _level2.DEM_NODATA
    assert valid.any() and (~valid).any() and (out[valid] >= 800).all() and (out[valid] <= 900).all()
    # east is higher: the ramp rises with the longitude
    row = out[50]
    ok = row != _level2.DEM_NODATA
    assert row[ok][-1] > row[ok][0]


# --- the Fmask step (fmask_flags) with a recording stand-in for zeit._core.fmask_run ------------------------------

NR = NC = 12


def fake_run_core(calls, nr=NR, nc=NC):
    rows, cols = nr // 2, nc // 2

    def dn_to_toa20(r, c, *args):
        calls["toa"] = (r, c, args)
        names = ("blue", "green", "red", "nir", "swir1", "swir2", "cirrus", "b7", "b8")
        out = {k: np.full(r * c, 100.0 * (i + 1), np.float32) for i, k in enumerate(names)}
        out.update({k: np.zeros(r * c, np.uint8) for k in ("satu_blue", "satu_green", "satu_red")})
        return out

    def run_fmask(r, c, *args, **kw):
        calls["run"] = (r, c, args, kw)
        z = np.zeros(r * c, np.uint8)
        mask = np.ones(r * c, np.uint8)
        pcloud = z.copy()
        pcloud[: r * c // 4] = 1
        return dict(sum_clr=1234, few_clear=False, match_skipped=False, clr_pct=75.0, mask=mask, pcloud=pcloud, pshadow=z, psnow=z, water=z)

    def to_qai(r, c, factor, resolution, *args):
        calls["qai"] = (r, c, factor, resolution, args)
        return np.zeros((r * factor) * (c * factor), np.int16)

    def slope_aspect_deg(d, r, c, res):
        calls["slope"] = (d, r, c, res)
        return np.zeros_like(d), np.zeros_like(d)

    fr = types.SimpleNamespace(dn_to_toa20=dn_to_toa20, run_fmask=run_fmask, to_qai=to_qai, slope_aspect_deg=slope_aspect_deg)
    return fr


def scene_for_fmask(monkeypatch, calls, baseline="05.09", radd=None):
    monkeypatch.setattr(_level2, "_core", types.SimpleNamespace(fmask_run=fake_run_core(calls)))
    rows, cols = NR // 2, NC // 2
    monkeypatch.setattr(_level2, "_mean_sun_angle", lambda p: (43.5, 38.5))
    monkeypatch.setattr(_level2, "_view_angles_20m", lambda *a, **k: (np.full((rows, cols), 500.0, np.float32), np.full((rows, cols), 9000.0, np.float32), 0.25))
    safe = types.SimpleNamespace(tile="MTD_TL.xml")
    radd = [-1000.0] * 13 if radd is None else radd
    desc = dict(nrows=NR, ncols=NC, baseline=baseline, dn_scale=10000.0, radd=radd)
    dn = (np.arange(13)[:, None, None] * 1000 + np.arange(NR * NC).reshape(1, NR, NC)).astype(np.uint16)
    return safe, desc, dn


def test_the_fmask_gets_the_bands_decimated_to_their_resolution_and_the_route_of_the_baseline(monkeypatch):
    calls = {}
    radd = [-float(b) for b in range(13)]
    safe, desc, dn = scene_for_fmask(monkeypatch, calls, "05.09", radd)
    _level2.fmask_flags(safe, desc, dn)
    r, c, args = calls["toa"]
    assert (r, c) == (NR // 2, NC // 2)
    bands = args[:9]
    for a, b, k in zip(bands, (1, 2, 3, 8, 11, 12, 10, 6, 7), (1, 1, 1, 2, 2, 2, 6, 2, 1)):
        assert np.array_equal(a, dn[b, ::k, ::k].ravel()), b
    assert [a.size for a in bands] == [4 * 36, 4 * 36, 4 * 36, 36, 36, 36, 4, 36, 4 * 36]
    # baseline 04.00 and later: the offsets in the order B02 B03 B04 B8A B11 B12 B10 B07 B08, and the quantification
    assert args[9] is False
    assert np.array_equal(args[10], [-1.0, -2.0, -3.0, -8.0, -11.0, -12.0, -10.0, -6.0, -7.0]) and args[11] == 10000.0
    # an older baseline: the legacy route, no offsets
    calls.clear()
    safe, desc, dn = scene_for_fmask(monkeypatch, calls, "03.01", [0.0] * 13)
    _level2.fmask_flags(safe, desc, dn)
    assert len(calls["toa"][2]) == 10
    assert calls["toa"][2][9] is False
    with pytest.raises(ValueError, match="multiples of 6"):
        d = dict(desc, nrows=10, ncols=10)
        _level2.fmask_flags(safe, d, dn[:, :10, :10])


def test_the_flags_have_the_scene_shape_and_the_fmask_gets_the_angles_dem_and_buffer(monkeypatch):
    calls = {}
    safe, desc, dn = scene_for_fmask(monkeypatch, calls)
    dem = np.full((NR // 2, NC // 2), 800.0, np.float32)
    dem[0, 0] = _level2.DEM_NODATA
    flags, info = _level2.fmask_flags(safe, desc, dn, dem=dem, gswo=np.full((NR // 2, NC // 2), 7, np.uint8))
    assert flags.shape == (NR, NC) and flags.dtype == np.int16
    r, c, args, kw = calls["run"]
    assert (r, c) == (6, 6)
    assert args[10].dtype == np.float32 and np.all(args[10] == 500.0) and np.all(args[11] == 9000.0)   # vza, vaa (1/100 degree)
    assert (args[12], args[13]) == (43.5, 38.5)                                                      # the mean sun angle
    assert np.isnan(kw["dem"][0]) and kw["dem"][1] == 800.0 and kw["gswo"].dtype == np.uint8
    assert calls["slope"][3] == 20.0 and set(kw) >= {"dem", "slope", "aspect", "gswo", "b7", "b8"}
    assert calls["qai"][2:4] == (2, 20.0) and calls["qai"][4][-2:] == (300.0, 8)
    assert info["sum_clr"] == 1234 and info["cloud"] == 0.25 and info["shadow"] == 0 and info["route"] == "baseline04"
    assert info["dem"] and info["gswo"] and info["view_missing"] == 0.25
    # without DEM and GSWO the Fmask gets none
    calls.clear()
    _level2.fmask_flags(safe, desc, dn)
    assert not {"dem", "slope", "aspect", "gswo"} & set(calls["run"][3])


def level2_stub(monkeypatch, nfmask):
    stub = types.SimpleNamespace(l2scene=types.SimpleNamespace(run=lambda *a, **k: {"boa": np.zeros(10 * NR * NC, np.int16), "qai": np.zeros(NR * NC, np.int16)}))
    monkeypatch.setattr(_level2, "_core", stub)

    def fmask_flags(*a, **k):
        nfmask.append(k)
        return np.zeros((NR, NC), np.int16), {"sum_clr": 1}

    monkeypatch.setattr(_level2, "fmask_flags", fmask_flags)
    safe = object.__new__(_level2.Safe)
    safe.product = types.SimpleNamespace(read_bytes=lambda: b"")
    safe.tile = types.SimpleNamespace(read_bytes=lambda: b"")
    safe.granule = types.SimpleNamespace(name="g")
    safe.describe = lambda: dict(nrows=NR, ncols=NC)
    return safe, dict(bands=np.zeros((13, NR, NC), np.uint16), geography=(np.zeros(1), np.zeros(1)))


def test_the_fmask_runs_by_default_and_not_with_flags_or_fmask_false(monkeypatch):
    calls = []
    safe, kw = level2_stub(monkeypatch, calls)
    out = _level2._level2_arrays(safe, **kw, gswo="g.tif", fmask_compat=True)
    assert len(calls) == 1 and calls[0]["gswo"] == "g.tif" and calls[0]["compat"] is True and out["fmask"] == {"sum_clr": 1}
    out = _level2._level2_arrays(safe, **kw, flags=np.zeros((NR, NC), np.int16))
    assert len(calls) == 1 and "fmask" not in out
    out = _level2._level2_arrays(safe, **kw, fmask=False)
    assert len(calls) == 1 and "fmask" not in out


def fake_arrays(monkeypatch, calls, fmask_info=None):
    shape = (NR, NC)
    desc = dict(nrows=NR, ncols=NC, res=10.0, ulx=199980.0, uly=8300020.0, epsg=32723, date=(2023, 7, 28, 13, 22, 39),
                sensor="S2B", baseline="05.09")

    def arrays(safe, **kw):
        calls.append((safe, kw))
        out = {"boa": np.full((10,) + shape, 123, np.int16), "desc": desc, "aod_scene": np.arange(13) / 100.0, "aodmap": True,
               "n_targets": [5, 6, 7], "ms": 2500.0}
        for i, k in enumerate(("qai", "dst", "vzn", "hot", "aod", "wvp")):
            out[k] = np.full(shape, i, np.int16)
        out["wvp"][:] = 2000
        out["wvp"][0, 0] = -9999
        if fmask_info is not None:
            out["fmask"] = fmask_info
        return out

    monkeypatch.setattr(_level2, "_level2_arrays", arrays)
    # "scene.SAFE" is not a directory: level2 would download it; the name goes through as it is
    monkeypatch.setattr(_s2fetch, "download_l1c", lambda product, *a, **k: product)


def test_level2_without_a_dem_for_the_topography_says_what_to_do(monkeypatch):
    calls = []
    fake_arrays(monkeypatch, calls)
    with pytest.raises(ValueError, match="topography=False"):
        _level2.level2("scene.SAFE")
    assert not calls
    _level2.level2("scene.SAFE", topography=False)
    assert len(calls) == 1


def test_level2_maps_the_public_parameters_to_the_engine(monkeypatch):
    calls = []
    fake_arrays(monkeypatch, calls)
    flags = np.zeros((NR, NC), np.int16)
    _level2.level2("scene.SAFE", "dem.tif", brdf=False, topography=False, aod=False, adjacency=False, multiple_scattering=False,
                   erase_cloud=True, flags=flags, fmask=False, gswo="g.tif", fmask_compat=True, dem_resampling="nearest", n_jobs=3)
    safe, kw = calls[0]
    assert safe == "scene.SAFE"
    assert kw == dict(dem="dem.tif", flags=flags, dobrdf=False, dotopo=False, doaod=False, doenv=False, domulti=False,
                      erase_cloud=True, dem_resampling="nearest", n_jobs=3, fmask=False, gswo="g.tif", fmask_compat=True)
    calls.clear()
    _level2.level2("scene.SAFE", "dem.tif")
    kw = calls[0][1]
    assert (kw["dobrdf"], kw["dotopo"], kw["doaod"], kw["doenv"], kw["domulti"], kw["erase_cloud"]) == (True,) * 5 + (False,)
    assert kw["fmask"] is True and kw["n_jobs"] == 8 and kw["dem_resampling"] == "bilinear"


def test_level2_dataset_has_the_grid_the_crs_and_the_attrs(monkeypatch):
    calls = []
    fake_arrays(monkeypatch, calls, fmask_info=dict(sum_clr=np.int64(9), few_clear=False, cloud=0.25, grid=(6, 6), baseline="05.09",
                                                    dem=True, missing=None))
    ds = _level2.level2("scene.SAFE", "dem.tif")
    assert ds.boa.dims == ("band", "y", "x") and ds.boa.shape == (10, NR, NC) and ds.boa.dtype == np.int16
    assert list(ds.band.values) == ["BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2"]
    for k in ("qai", "dst", "vzn", "hot", "aod", "wvp"):
        assert ds[k].dims == ("y", "x") and ds[k].dtype == np.int16
    assert np.allclose(ds.x.values, 199980.0 + 10.0 * (np.arange(NC) + 0.5))
    assert np.allclose(ds.y.values, 8300020.0 - 10.0 * (np.arange(NR) + 0.5))
    assert ds.rio.crs.to_epsg() == 32723
    t = ds.rio.transform()
    assert tuple(t)[:6] == (10.0, 0.0, 199980.0, 0.0, -10.0, 8300020.0)
    assert ds.boa.rio.nodata == -9999
    a = ds.attrs
    assert a["datetime"] == "2023-07-28T13:22:39Z" and a["sensor"] == "S2B" and a["baseline"] == "05.09"
    assert a["aod_green"] == 0.02 and a["aod_map"] is True and a["n_targets"] == [5, 6, 7] and a["core_seconds"] == 2.5
    assert a["water_vapor_cm"] == 2.0
    assert a["fmask_sum_clr"] == 9 and a["fmask_few_clear"] is False and a["fmask_grid"] == [6, 6] and "fmask_missing" not in a
    assert "Frantz 2019" in a["provenance"] and "Qiu, Zhu and Woodcock 2019" in a["provenance"]
    assert hasattr(ds.zeit, "save")


def test_level2_without_the_fmask_has_no_fmask_attrs(monkeypatch):
    calls = []
    fake_arrays(monkeypatch, calls)
    ds = _level2.level2("scene.SAFE", topography=False, fmask=False)
    assert not [k for k in ds.attrs if k.startswith("fmask")]


def test_level2_a_local_path_that_does_not_exist_is_not_downloaded(monkeypatch):
    called = []
    monkeypatch.setattr(_s2fetch, "download_l1c", lambda *a, **k: called.append(a))
    with pytest.raises(ValueError, match="not a Sentinel-2 L1C"):
        _level2.level2(os.path.join("no", "such", "folder", "S2B_MSIL1C_x.SAFE"), topography=False)
    assert not called
