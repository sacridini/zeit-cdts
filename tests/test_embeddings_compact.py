"""save_raster(compact=True): int8 copies of embedding cubes, read back as float32 by load_raster.
Synthetic cubes only, no network."""
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

import zeit
from zeit import _alphaearth, _compact

NB = 8
SIZE = 24
YEARS = [2018, 2019, 2020]


def _cube(data, source="tessera", dask=False, years=YEARS):
    n = data.shape[0]
    da = xr.DataArray(data.astype(np.float32), dims=("time", "band", "y", "x"),
                      coords={"time": pd.DatetimeIndex([pd.Timestamp(y, 1, 1) for y in years[:n]]),
                              "band": [f"A{i:02d}" for i in range(data.shape[1])],
                              "y": 8900000 - 5 - 10 * np.arange(data.shape[2]),
                              "x": 500000 + 5 + 10 * np.arange(data.shape[3])}, name="embeddings")
    da = da.rio.write_crs(32720)
    da.attrs.update({"embedding_source": source, "embedding_version": "1.1", "embedding_variant": "dclimate"})
    return da.chunk({"time": 1, "band": 4, "y": 16, "x": 16}) if dask else da


def _varied(seed=0, n=3, nb=NB):
    """Pixels of very different magnitudes, an all-zero one, a tiny one."""
    rng = np.random.default_rng(seed)
    mag = 10.0 ** rng.uniform(-4, 3, size=(n, 1, SIZE, SIZE))
    data = rng.normal(size=(n, nb, SIZE, SIZE)) * mag
    data[0, :, 0, 0] = 0.0
    data[-1 if n < 2 else 1, :, 0, 1] = 1e-9
    data[-1, :, 0, 2] = rng.normal(size=nb) * 1e6
    return data.astype(np.float32)


def test_aef_round_trip_is_exact_for_every_code(tmp_path):
    codes = np.arange(-127, 128, dtype=np.int8)
    block = np.zeros((1, 64, SIZE, SIZE), np.int8)
    block[0, :, :, :] = codes[np.arange(64 * SIZE * SIZE).reshape(64, SIZE, SIZE) % 255]
    block[0, :, 3, 3] = codes[:64]
    block[0, :, 4, 4] = codes[-64:]
    x = _alphaearth.dequantise(block)
    cube = _cube(x, source="alphaearth", dask=True)
    assert np.isin(codes, block).all()
    path = zeit.save_raster(cube, tmp_path / "aef.tif", compact=True)
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "int8" and src.count == 64 and src.nodata == -128
        assert src.tags()["ZEIT_EMBEDDING_ENCODING"] == "aef-int8"
        assert json.loads(src.tags()["ZEIT_EMBEDDING"])["embedding_source"] == "alphaearth"
        np.testing.assert_array_equal(src.read(), block[0])
    back = zeit.load_raster(path)
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back.values, x)


def test_aef_pixel_with_a_nan_dimension_is_nodata_in_all():
    x = _alphaearth.dequantise(np.random.default_rng(1).integers(-127, 128, (2, 6, 4, 4)).astype(np.int8))
    x[0, 2, 1, 1] = np.nan
    enc = _compact.encode_aef(x)
    assert (enc[0, :, 1, 1] == -128).all() and (enc[1, :, 1, 1] != -128).all()
    out = _compact.decode_aef(enc)
    assert np.isnan(out[0, :, 1, 1]).all() and not np.isnan(out[1]).any()


def test_scaled_error_bound_and_scale_band(tmp_path):
    data = _varied()
    data[1, :, 5, 5] = np.nan                  # a pixel-date without an embedding
    data[2, 3, 6, 6] = np.nan                  # one dimension is enough
    cube = _cube(data, dask=True)
    path = zeit.save_raster(cube, tmp_path / "t.tif", compact=True)
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "int8" and src.count == len(YEARS) * (NB + 1) and src.nodata == -128
        assert src.tags()["ZEIT_EMBEDDING_ENCODING"] == "scaled-int8"
        assert "ZEIT_EMBEDDING" in src.tags()
        assert src.descriptions[NB] == "2018_scale" and src.descriptions[NB + 1] == "2019_A00"
        raw = src.read()
    assert path.stat().st_size < 0.4 * zeit.save_raster(cube, tmp_path / "f.tif").stat().st_size
    back = zeit.load_raster(path)
    assert back.dims == cube.dims and back.dtype == np.float32
    assert list(back.band.values) == list(cube.band.values) and "scale" not in back.band.values
    assert (~np.isfinite(back.values) == np.isnan(data).any(axis=1, keepdims=True)).all()
    # the scale codes, from the file
    codes = raw.reshape(len(YEARS), NB + 1, SIZE, SIZE)[:, NB].astype(np.float64)
    s = 2.0 ** (codes / 8)
    ok = np.isfinite(back.values).all(axis=1)
    err = np.abs(back.values - data).max(axis=1)
    assert (err[ok] <= s[ok] / 2 * (1 + 1e-6) + 1e-30).all()
    m = np.abs(data).max(axis=1)
    sel = ok & (m > 0) & (codes > -127)
    assert (s[sel] < 2 ** (1 / 8) * m[sel] / 127 * (1 + 1e-9)).all()
    assert (np.abs(data).max(axis=1)[ok] / s[ok] <= 127.0 + 1e-6).all()
    assert (back.values[0, :, 0, 0] == 0).all() and codes[0, 0, 0] == -127


def test_scaled_codes_never_overflow_the_int8_range():
    rng = np.random.default_rng(3)
    # magnitudes on the edges of the powers of 2**(1/8): m / s lands near 127
    for e in range(-120, 120):
        base = 127 * 2.0 ** (e / 8)
        x = np.zeros((1, 3, 1, 3))
        x[0, 0, 0, :] = [base, np.nextafter(base, 0), np.nextafter(base, np.inf)]
        x[0, 1:, 0, :] = rng.normal(size=(2, 3)) * base / 3
        enc = _compact.encode_scaled(x.astype(np.float32))
        assert enc.min() >= -127 or (enc == -128).all()
        assert (np.abs(enc[:, :3].astype(int)) <= 127).all()


def test_too_large_values_raise(tmp_path):
    data = _varied()
    data[0, 0, 2, 2] = 1e8
    with pytest.raises(ValueError, match="too large"):
        zeit.save_raster(_cube(data), tmp_path / "big.tif", compact=True)
    data[0, 0, 2, 2] = np.inf
    with pytest.raises(ValueError, match="too large"):
        zeit.save_raster(_cube(data, dask=True), tmp_path / "inf.tif", compact=True)


def test_loads_as_the_float32_save_does(tmp_path):
    cube = _cube(_varied(seed=2))
    plain = zeit.load_raster(zeit.save_raster(cube, tmp_path / "f.tif"))
    compact = zeit.load_raster(zeit.save_raster(cube, tmp_path / "c.tif", compact=True))
    assert compact.dims == plain.dims and compact.dtype == plain.dtype
    assert compact.sizes == plain.sizes
    for name in ("time", "band", "y", "x"):
        np.testing.assert_array_equal(compact[name].values, plain[name].values)
    assert {k: v for k, v in compact.attrs.items() if k != "source"} == \
        {k: v for k, v in plain.attrs.items() if k != "source"}
    assert compact.attrs["source"].endswith("c.tif") and "ZEIT_EMBEDDING_ENCODING" not in compact.attrs
    assert compact.rio.crs == plain.rio.crs and compact.rio.transform() == plain.rio.transform()
    m = np.abs(plain.values).max(axis=1, keepdims=True)
    np.testing.assert_allclose(compact.values, plain.values, rtol=0, atol=float(m.max()) * 0.0043)
    assert (np.abs(compact.values - plain.values) <= m * 2 ** (1 / 8) / 127 / 2 * (1 + 1e-5) + 1e-4).all()
    # lazy, in the chunks asked
    lazy = zeit.load_raster(tmp_path / "c.tif", chunks={"y": 8, "x": 8})
    assert lazy.chunks is not None and lazy.chunks[-1][0] == 8
    np.testing.assert_array_equal(lazy.values, compact.values)
    # band= counts the cube's dimension bands, as in the float32 file
    two = zeit.load_raster(tmp_path / "c.tif", band=[1, 2])
    ref = zeit.load_raster(tmp_path / "f.tif", band=[1, 2])
    assert two.sizes == ref.sizes and list(two.band.values) == list(ref.band.values)


def test_a_single_date_and_a_selected_year(tmp_path):
    cube = _cube(_varied(seed=4))
    year = cube.isel(time=[1])
    back = zeit.load_raster(zeit.save_raster(year, tmp_path / "one.tif", compact=True))
    ref = zeit.load_raster(zeit.save_raster(year, tmp_path / "one32.tif"))
    assert back.dims == ref.dims and back.sizes == ref.sizes
    flat = cube.isel(time=1)          # (band, y, x) with a scalar date
    back3 = zeit.load_raster(zeit.save_raster(flat, tmp_path / "flat.tif", compact=True))
    ref3 = zeit.load_raster(zeit.save_raster(flat, tmp_path / "flat32.tif"))
    assert back3.dims == ref3.dims == ("band", "y", "x")
    assert list(back3.band.values) == list(flat.band.values)
    bound = np.abs(flat.values).max(axis=0, keepdims=True) * 2 ** (1 / 8) / 127 / 2 * (1 + 1e-5) + 1e-4
    assert (np.abs(back3.values - flat.values) <= bound).all()
    assert zeit.save_raster(year, tmp_path / "cog.tif", compact=True, driver="COG").exists()
    assert zeit.load_raster(tmp_path / "cog.tif").sizes == ref.sizes


def test_compact_is_lazy(tmp_path):
    import dask.array as da

    shapes = []

    def spy(block):
        if block.size:      # dask's own probe of the output type is empty
            shapes.append(block.shape)
        return block

    cube = _cube(_varied(seed=5), dask=True)
    spied = cube.copy(data=cube.data.map_blocks(spy, dtype=np.float32))
    enc, _ = _compact.encode_cube(spied)
    assert isinstance(enc.data, da.Array) and enc.dtype == np.int8 and enc.sizes["band"] == NB + 1
    assert enc.data.chunks[1] == (NB + 1,) and not shapes    # one chunk of bands; nothing computed yet
    path = zeit.save_raster(spied, tmp_path / "lazy.tif", compact=True)
    assert shapes and max(s[0] * s[2] * s[3] for s in shapes) < spied.sizes["time"] * SIZE * SIZE   # block by block
    assert path.exists()


def test_classify_accepts_the_compact_cube(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    rng = np.random.default_rng(6)
    a, b = rng.normal(size=NB), rng.normal(size=NB)
    data = np.empty((2, NB, SIZE, SIZE), np.float32)
    data[:, :, :, :SIZE // 2] = a[None, :, None, None]
    data[:, :, :, SIZE // 2:] = b[None, :, None, None]
    data += rng.normal(0, 0.05, data.shape).astype(np.float32)
    cube = _cube(data, years=[2020, 2021])
    year = cube.isel(time=-1)
    pts = gpd.GeoDataFrame({"class": ["a", "a", "b", "b"]},
                           geometry=[Point(500015, 8899985), Point(500025, 8899905), Point(500235, 8899985),
                                     Point(500225, 8899905)], crs=32720)
    rf = zeit.train_classifier(year, pts)
    back = zeit.load_raster(zeit.save_raster(cube, tmp_path / "c.tif", compact=True)).isel(time=-1)
    classes = zeit.classify(back, rf)
    left, right = classes.label.values[:, :SIZE // 2], classes.label.values[:, SIZE // 2:]
    assert (left == 1).all() and (right == 2).all()
    sim = zeit.similarity(back, a)
    assert float(sim.isel(x=0, y=0)) > 0.99


def test_errors(tmp_path):
    cube = _cube(_varied())
    plain = xr.DataArray(np.ones((2, 5, 5), np.float32), dims=("band", "y", "x"))
    with pytest.raises(ValueError, match="embedding"):
        zeit.save_raster(plain, tmp_path / "p.tif", compact=True,
                         crs="EPSG:32720", transform=rasterio.transform.from_origin(0, 50, 10, 10))
    with pytest.raises(ValueError, match="dtype"):
        zeit.save_raster(cube, tmp_path / "d.tif", compact=True, dtype="float32")
    with pytest.raises(ValueError, match="nodata"):
        zeit.save_raster(cube, tmp_path / "d.tif", compact=True, nodata=0)
    with pytest.raises(ValueError, match="GeoTIFF"):
        zeit.save_raster(cube, tmp_path / "d.nc", compact=True)
    with pytest.raises(ValueError, match="GeoTIFF"):
        zeit.save_raster(cube, tmp_path / "d.zarr", compact=True)
    with pytest.raises(ValueError, match="one cube"):
        zeit.save_raster({"a": cube.isel(time=0, band=0)}, tmp_path / "dd", compact=True)
    path = zeit.save_raster(cube, tmp_path / "ok.tif", compact=True)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(ZEIT_EMBEDDING_ENCODING="mystery-int4")
    with pytest.raises(ValueError, match="mystery-int4"):
        zeit.load_raster(path)


def test_cli_accepts_compact(monkeypatch, tmp_path):
    from zeit import cli

    parser = cli.build_parser()
    args = parser.parse_args(["embeddings", str(tmp_path), "--source", "tessera", "--bbox", "-63", "-10", "-62.9",
                              "-9.9", "--compact"])
    assert args.compact is True
    args = parser.parse_args(["fetch", "embeddings", str(tmp_path), "--source", "alphaearth", "--region", "r.gpkg"])
    assert args.compact is False
    args = parser.parse_args(["fetch", "embeddings", str(tmp_path), "--source", "alphaearth", "--region", "r.gpkg",
                              "--compact"])
    assert args.compact is True
    saved = {}
    import zeit._save as save_module
    import zeit._embeddings as emb_module

    cube = _cube(_varied(n=1))
    monkeypatch.setattr(emb_module, "load_embeddings", lambda *a, **k: cube)
    real = save_module.save_raster
    monkeypatch.setattr(save_module, "save_raster", lambda *a, **k: saved.update(k) or real(*a, **k))
    cli.run_embeddings_cli(parser.parse_args(["embeddings", str(tmp_path), "--source", "tessera", "--bbox", "-63", "-10",
                                              "-62.9", "-9.9", "--compact"]))
    assert saved["compact"] is True
    with rasterio.open(tmp_path / "tessera.tif") as src:
        assert src.dtypes[0] == "int8"
