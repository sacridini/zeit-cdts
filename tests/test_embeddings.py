"""zeit.load_embeddings, zeit.similarity, zeit.embedding_change: the yearly embeddings of
TESSERA and AlphaEarth as zeit cubes. Without a network: a TESSERA Zarr store and a copy of the
AlphaEarth COGs with the published layouts are made in fixtures."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.transform import Affine

import zeit
from zeit import _alphaearth, _embeddings, _tessera

zarr = pytest.importorskip("zarr")

YEARS = [2018, 2019, 2020]
NB = 8           # dimensions of the fake TESSERA store (the real one has 128)
SIZE = 80        # cells a side of each zone's grid


def _utm(lon, lat, epsg):
    return Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)


def _zone_arrays(rng):
    emb = rng.integers(-127, 128, size=(len(YEARS), NB, SIZE, SIZE)).astype(np.int8)
    scales = rng.uniform(0.01, 0.05, size=(len(YEARS), SIZE, SIZE)).astype(np.float32)
    scales[:, 0, 0] = np.nan       # no embedding (water, or a gap)
    scales[1, 0, 1] = np.inf       # land not done yet
    return emb, scales


def _zone_group(root, name, epsg, x0, y0, width, height=SIZE, array="embeddings", nb=NB):
    """An empty zone grid of ``height`` rows and ``width`` columns from (x0, y0), as sparse as
    the published one: unwritten chunks read as no embedding (a NaN scale)."""
    g = root.create_group(name)
    g.attrs.update({"proj:code": f"EPSG:{epsg}", "spatial:transform": [10, 0, x0, 0, -10, y0]})
    g.create_array("time", data=np.array(YEARS, dtype=np.int32))
    g.create_array(array, shape=(len(YEARS), nb, height, width), dtype="int8", chunks=(1, nb, 32, 32), fill_value=0)
    g.create_array("scales", shape=(len(YEARS), height, width), dtype="float32", chunks=(1, 32, 32),
                   fill_value=np.nan)
    return g


def _write_block(group, row, col, emb, scales, array="embeddings"):
    h, w = emb.shape[-2:]
    group[array][:, :, row:row + h, col:col + w] = emb
    group["scales"][:, row:row + h, col:col + w] = scales


@pytest.fixture(scope="module")
def tessera(tmp_path_factory):
    """A store like the published v1.1: zone 20 (in the northern CRS, with negative northings
    south of the equator) with a block of embeddings around (-62.5, -10) and one at the zone's
    eastern edge, and zone 21 with a block just across -60 degrees."""
    path = tmp_path_factory.mktemp("tessera") / "store.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"geoemb:dimensions": NB, "geoemb:model": "https://geotessera.org/model/1.1",
                       "geoemb:build_version": "0.10.3",
                       "geotessera:source_store": "s3://tessera-embeddings/v1.1/dclimate.icechunk"})
    rng = np.random.default_rng(1)
    x, y = _utm(-62.5, -10.0, 32620)
    x0, y0 = 10 * round(x / 10) - 400, 10 * round(y / 10) + 400
    xe, ye = _utm(-60.0005, -10.0, 32620)
    edge = int(round((xe - x0) / 10)) - SIZE           # the eastern block ends just west of -60
    down = int(round((y0 - ye) / 10)) - SIZE // 2      # where -10 degrees is there (the grid converges)
    assert down > SIZE
    g20 = _zone_group(root, "utm20", 32620, x0, y0, edge + SIZE, height=down + SIZE)
    emb, scales = _zone_arrays(rng)
    _write_block(g20, 0, 0, emb, scales)
    emb_e, scales_e = _zone_arrays(rng)
    _write_block(g20, down, edge, emb_e, scales_e)
    x21, y21 = _utm(-60.002, -10.0, 32621)
    x21, y21 = 10 * round(x21 / 10) - 300, 10 * round(y21 / 10) + 400
    g21 = _zone_group(root, "utm21", 32621, x21, y21, SIZE)
    emb21, scales21 = _zone_arrays(rng)
    _write_block(g21, 0, 0, emb21, scales21)
    zones = {20: (x0, y0, emb, scales), "20e": (x0 + 10 * edge, y0 - 10 * down, emb_e, scales_e),
             21: (x21, y21, emb21, scales21)}
    return path, zones


def _bbox_of(x0, y0, epsg, r0, r1, c0, c1):
    """Longitude/latitude bounds strictly inside cells r0..r1, c0..c1 of a grid."""
    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon0, lat1 = t.transform(x0 + 10 * c0 + 3, y0 - 10 * r0 - 3)
    lon1, lat0 = t.transform(x0 + 10 * c1 - 3, y0 - 10 * r1 + 3)
    return (lon0, lat0, lon1, lat1)


def _truth(emb, scales):
    s = np.where(np.isfinite(scales), scales, np.nan)
    return emb.astype(np.float32) * s[:, None]


# ---------------------------------------------------------------------- TESSERA
def test_tessera_reads_and_dequantises(tessera):
    path, zones = tessera
    x0, y0, emb, scales = zones[20]
    bbox = _bbox_of(x0, y0, 32620, 0, 20, 0, 30)
    cube = zeit.load_embeddings(bbox, source="tessera", store=path, chunks=None)
    assert cube.dims == ("time", "band", "y", "x")
    assert list(cube.band.values) == [f"A{i:02d}" for i in range(NB)]
    assert list(cube.time.dt.year.values) == YEARS
    assert cube.dtype == np.float32 and cube.name == "embeddings"
    # south of the equator: the southern zone's CRS, the same cells 10,000 km up in northing
    assert cube.rio.crs.to_epsg() == 32720
    r0 = int(round((y0 - (cube.y.values[0] - 1e7) - 5) / 10))
    c0 = int(round((cube.x.values[0] - x0 - 5) / 10))
    assert (r0, c0) == (0, 0)
    expected = _truth(emb, scales)[:, :, :cube.sizes["y"], :cube.sizes["x"]]
    np.testing.assert_allclose(cube.values, expected, rtol=1e-6)
    assert np.isnan(cube.values[:, :, 0, 0]).all()            # NaN scale
    assert np.isnan(cube.values[1, :, 0, 1]).all() and np.isfinite(cube.values[0, :, 0, 1]).all()   # +inf
    t = cube.rio.transform()
    assert (t.a, t.e, t.c, t.f) == (10, -10, x0, y0 + 1e7)
    a = cube.attrs
    assert a["embedding_source"] == "tessera" and a["embedding_version"] == "1.1"
    assert a["embedding_variant"] == "dclimate" and a["embedding_dimensions"] == NB


def test_tessera_lazy_years_and_errors(tessera):
    path, zones = tessera
    x0, y0, emb, scales = zones[20]
    bbox = _bbox_of(x0, y0, 32620, 5, 70, 5, 70)
    lazy = zeit.load_embeddings(bbox, source="tessera", store=path, years=[2020, 2018])
    assert lazy.chunks is not None
    assert list(lazy.time.dt.year.values) == [2018, 2020]
    eager = zeit.load_embeddings(bbox, source="tessera", store=path, years=[2018, 2020], chunks=None)
    np.testing.assert_array_equal(lazy.values, eager.values)
    np.testing.assert_allclose(eager.values, _truth(emb, scales)[[0, 2], :, 5:70, 5:70], rtol=1e-6)
    with pytest.raises(ValueError, match="years"):
        zeit.load_embeddings(bbox, source="tessera", store=path, years=2016)
    with pytest.raises(ValueError, match="depths"):
        zeit.load_embeddings(bbox, source="tessera", store=path, depth=4)
    with pytest.raises(ValueError, match="longitude and latitude"):
        zeit.load_embeddings((500000, 8900000, 501000, 8901000), source="tessera", store=path)
    with pytest.raises(ValueError, match="cover"):
        zeit.load_embeddings((-62.0, -9.0, -61.9, -8.9), source="tessera", store=path)
    with pytest.raises(ValueError, match="backend"):
        zeit.load_embeddings(bbox, source="tessera", store=path, backend="gee")
    with pytest.raises(ValueError, match="source"):
        zeit.load_embeddings(bbox, source="clay")


def test_tessera_polygons_points_and_aligned_chunks(tessera):
    import geopandas as gpd
    from shapely.geometry import Point, box

    path, zones = tessera
    x0, y0, emb, scales = zones[20]
    poly = gpd.GeoDataFrame(geometry=[box(x0 + 100, y0 - 500, x0 + 400, y0 - 100)], crs=32620)
    cube = zeit.load_embeddings(poly, source="tessera", store=path, years=2018, chunks=None)
    assert cube.sizes["y"] == 40 and cube.sizes["x"] == 30
    assert np.isfinite(cube.values).all()
    pts = gpd.GeoDataFrame(geometry=[Point(x0 + 205, y0 - 205), Point(x0 + 315, y0 - 115)], crs=32620)
    around = zeit.load_embeddings(pts, source="tessera", store=path, years=2018, chunks=None)
    assert around.x.min() < x0 + 205 < around.x.max() and around.y.min() - 1e7 < y0 - 205 < around.y.max() - 1e7
    assert _embeddings.aligned_chunks(500, 1100, 512) == (12, 512, 512, 64)
    assert _embeddings.aligned_chunks(0, 100, 512) == (100,)


def test_tessera_across_zones(tessera):
    path, zones = tessera
    bbox = (-60.0035, -10.0015, -59.9985, -9.9985)
    with pytest.raises(ValueError, match="UTM zones"):
        zeit.load_embeddings(bbox, source="tessera", store=path, years=2018)
    cube = zeit.load_embeddings(bbox, source="tessera", store=path, years=2018, crs="EPSG:32721", chunks=None)
    assert cube.rio.crs.to_epsg() == 32721 and cube.rio.transform().a == 10
    x21, y21, emb21, scales21 = zones[21]
    values = cube.values[0]
    assert np.isfinite(values).all(axis=0).mean() > 0.4
    # zone 21's cells are taken as they are (the same grid: nearest is an exact copy)
    native = _truth(emb21, scales21)[0]
    col = int(round((cube.x.values[-1] - 5 - x21) / 10))
    row = int(round((y21 + 1e7 - cube.y.values[0] - 5) / 10))
    np.testing.assert_allclose(values[:, 0, -1], native[:, row, col], rtol=1e-6)
    # and zone 20's, warped from its CRS, at the western edge (away from the corners, which the
    # target grid, turned by the convergence of the two zones, takes from zone 21)
    xw, yw, emb_e, scales_e = zones["20e"]
    west = _truth(emb_e, scales_e)[0]
    edge = values[0, 5:-5, 0]
    assert np.isfinite(edge).all() and np.isin(edge, west[0]).all()


def test_tessera_like_averages_onto_coarser_cells(tessera):
    path, zones = tessera
    x0, y0, emb, scales = zones[20]
    y0s = y0 + 1e7
    like = xr.DataArray(np.zeros((10, 12), np.float32), dims=("y", "x"),
                        coords={"y": y0s - 60 - 15 - 30 * np.arange(10), "x": x0 + 60 + 15 + 30 * np.arange(12)})
    like = like.rio.write_crs(32720)
    cube = zeit.load_embeddings(source="tessera", like=like, store=path, years=2019, chunks=None)
    assert cube.sizes["y"] == 10 and cube.sizes["x"] == 12
    truth = _truth(emb, scales)[1]
    block = truth[:, 6 + 3 * 4:6 + 3 * 5, 6 + 3 * 7:6 + 3 * 8]
    np.testing.assert_allclose(cube.values[0, :, 4, 7], np.nanmean(block, axis=(1, 2)), rtol=1e-5, atol=1e-6)
    nearest = zeit.load_embeddings(source="tessera", like=like, store=path, years=2019, chunks=None,
                                   resampling="nearest")
    np.testing.assert_allclose(nearest.values[0, :, 4, 7], truth[:, 6 + 3 * 4 + 1, 6 + 3 * 7 + 1], rtol=1e-6)
    with pytest.raises(ValueError, match="not both"):
        zeit.load_embeddings(source="tessera", like=like, crs=32720, store=path)


def test_tessera_matryoshka_depth(tmp_path):
    path = tmp_path / "v2.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"geoemb:dimensions": NB, "geoemb:model": "https://geotessera.org/model/2.0",
                       "geoemb:depths": [{"dimensions": 4, "array": "embeddings_d4"}]})
    rng = np.random.default_rng(3)
    emb, scales = _zone_arrays(rng)
    x, y = _utm(-62.5, 10.0, 32620)
    x0, y0 = 10 * round(x / 10), 10 * round(y / 10)
    group = _zone_group(root, "utm20", 32620, x0, y0, SIZE)
    group.create_array("embeddings_d4", shape=(len(YEARS), 4, SIZE, SIZE), dtype="int8", chunks=(1, 4, 32, 32),
                       fill_value=0)
    _write_block(group, 0, 0, emb, scales)
    _write_block(group, 0, 0, emb[:, :4], scales, array="embeddings_d4")
    bbox = _bbox_of(x0, y0, 32620, 2, 10, 2, 10)
    full = zeit.load_embeddings(bbox, source="tessera", store=path, years=2018, chunks=None)
    prefix = zeit.load_embeddings(bbox, source="tessera", store=path, years=2018, depth=4, chunks=None)
    assert full.sizes["band"] == NB and prefix.sizes["band"] == 4
    assert full.rio.crs.to_epsg() == 32620      # north of the equator: the store's CRS
    np.testing.assert_allclose(prefix.values, full.values[:, :4])
    assert prefix.attrs["embedding_dimensions"] == 4 and prefix.attrs["embedding_version"] == "2.0"
    with pytest.raises(ValueError, match=r"depths \[4, 8\]"):
        zeit.load_embeddings(bbox, source="tessera", store=path, depth=6)


def test_tessera_through_geotessera(tessera, monkeypatch):
    """Without store=: geotessera names the published store and opens it."""
    from zarr.storage import LocalStore

    path, zones = tessera
    asked = {}
    registry = types.ModuleType("geotessera.registry")

    def zarr_store_url(version, variant=None):
        asked["url"] = (version, variant)
        return str(path)

    registry.zarr_store_url = zarr_store_url
    registry.dataset_for_location = lambda location: types.SimpleNamespace(version="1.1", variant="dclimate")
    store = types.ModuleType("geotessera.store")
    store.zarr_store = lambda location: LocalStore(location, read_only=True)
    package = types.ModuleType("geotessera")
    monkeypatch.setitem(sys.modules, "geotessera", package)
    monkeypatch.setitem(sys.modules, "geotessera.registry", registry)
    monkeypatch.setitem(sys.modules, "geotessera.store", store)
    x0, y0, _, _ = zones[20]
    cube = zeit.load_embeddings(_bbox_of(x0, y0, 32620, 0, 5, 0, 5), source="tessera", years=2018)
    assert asked["url"] == ("v1.1", None)
    assert (cube.attrs["embedding_version"], cube.attrs["embedding_variant"]) == ("1.1", "dclimate")
    monkeypatch.setitem(sys.modules, "geotessera", None)
    with pytest.raises(ImportError, match="pip install geotessera"):
        zeit.load_embeddings(_bbox_of(x0, y0, 32620, 0, 5, 0, 5), source="tessera", years=2018)


# ---------------------------------------------------------------------- AlphaEarth
def _aef_values(rng, shape):
    v = rng.integers(-127, 128, size=shape).astype(np.int8)
    v[:, 0, 0] = -128
    return v


@pytest.fixture(scope="module")
def alphaearth(tmp_path_factory):
    """A copy of the AlphaEarth COGs: two bottom-up files side by side in zone 20S, two years,
    and the index."""
    root = tmp_path_factory.mktemp("aef")
    rng = np.random.default_rng(2)
    n = 48
    x0, top = 500000.0, 8890000.0
    rows, truth = [], {}
    for year in (2018, 2024):
        full = _aef_values(rng, (64, n, 2 * n))      # top-down, as on the map
        full[:, 0, n] = -128
        truth[year] = full
        for k in range(2):
            west = x0 + k * n * 10
            rel = f"{year}/20S/x{year}{k}-0000000000-{k * n:010d}.tiff"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            part = full[:, :, k * n:(k + 1) * n]
            south = top - n * 10
            with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=64, dtype="int8", nodata=-128,
                               crs="EPSG:32720", transform=Affine(10, 0, west, 0, 10, south)) as dst:
                dst.write(part[:, ::-1, :])           # bottom-up: the first row is the southern one
                for b in range(64):
                    dst.set_band_description(b + 1, f"A{b:02d}")
            to_ll = Transformer.from_crs(32720, 4326, always_xy=True)
            lons, lats = to_ll.transform([west, west + n * 10], [south, top])
            rows.append({"path": f"s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/{rel}", "year": year,
                         "utm_zone": "20S", "crs": "EPSG:32720", "utm_west": west, "utm_south": south,
                         "utm_east": west + n * 10, "utm_north": top, "wgs84_west": min(lons),
                         "wgs84_south": min(lats), "wgs84_east": max(lons), "wgs84_north": max(lats)})
    pd.DataFrame(rows).to_parquet(root / "aef_index.parquet")
    return root, truth, (x0, top, n)


def _dequantise(v):
    out = np.sign(v.astype(np.float32)) * (v.astype(np.float32) / 127.5) ** 2
    return np.where(v == -128, np.nan, out)


def test_alphaearth_reads_bottom_up_files(alphaearth):
    root, truth, (x0, top, n) = alphaearth
    t = Transformer.from_crs(32720, 4326, always_xy=True)
    w, s = t.transform(x0 + 3, top - n * 10 + 3)
    e, nn = t.transform(x0 + 2 * n * 10 - 3, top - 3)
    cube = zeit.load_embeddings((w, s, e, nn), source="alphaearth", store=root, chunks=None)
    assert cube.dims == ("time", "band", "y", "x") and cube.sizes["band"] == 64
    assert list(cube.time.dt.year.values) == [2018, 2024]
    assert cube.rio.crs.to_epsg() == 32720
    t2 = cube.rio.transform()
    assert (t2.c, t2.f) == (x0, top)
    np.testing.assert_allclose(cube.values[0], _dequantise(truth[2018]), rtol=1e-6)
    np.testing.assert_allclose(cube.values[1], _dequantise(truth[2024]), rtol=1e-6)
    assert np.isnan(cube.values[0, :, 0, 0]).all() and np.isnan(cube.values[0, :, 0, n]).all()
    a = cube.attrs
    assert a["embedding_source"] == "alphaearth" and a["embedding_license"] == "CC-BY-4.0"
    assert "Google DeepMind" in a["embedding_attribution"]
    lazy = zeit.load_embeddings((w, s, e, nn), source="alphaearth", store=root, years=2024)
    assert lazy.chunks is not None and lazy.chunks[1] == (16, 16, 16, 16)
    np.testing.assert_array_equal(lazy.values, cube.values[1:])
    with pytest.raises(ValueError, match="years"):
        zeit.load_embeddings((w, s, e, nn), source="alphaearth", store=root, years=2020)
    with pytest.raises(ValueError, match="TESSERA"):
        zeit.load_embeddings((w, s, e, nn), source="alphaearth", store=root, depth=16)


def test_alphaearth_index_is_downloaded_once(alphaearth, tmp_path, monkeypatch):
    root, _, _ = alphaearth
    calls = []

    def fake(url, path):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / "aef_index.parquet").read_bytes())

    monkeypatch.setattr(_alphaearth, "_download", fake)
    first = _alphaearth._index("https://example.org/aef", tmp_path)
    again = _alphaearth._index("https://example.org/aef", tmp_path)
    assert len(calls) == 1 and calls[0] == "https://example.org/aef/aef_index.parquet"
    assert len(first) == len(again) == 4
    assert first.rel.iloc[0].startswith("2018/20S/")
    assert _alphaearth._location("https://example.org/aef", "2018/20S/a.tiff") == \
        "/vsicurl/https://example.org/aef/2018/20S/a.tiff"


def test_alphaearth_from_earth_engine(tmp_path, monkeypatch):
    """backend='gee': a year at a time, downloaded once (a stand-in for Earth Engine)."""
    calls = []

    def fake(year, bbox, crs, path):
        calls.append((year, crs))
        x, y = _utm((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 32720)
        data = np.full((64, 4, 5), year - 2000, dtype=np.float32) / 100
        data[:, 0, 0] = -np.inf
        with rasterio.open(path, "w", driver="GTiff", width=5, height=4, count=64, dtype="float32", nodata=-np.inf,
                           crs=crs, transform=Affine(10, 0, x, 0, -10, y)) as dst:
            dst.write(data)
        return path

    monkeypatch.setattr(_alphaearth, "_gee_download", fake)
    bbox = (-62.95, -9.95, -62.949, -9.949)
    cube = zeit.load_embeddings(bbox, source="alphaearth", backend="gee", years=[2018, 2019], cache_dir=tmp_path,
                                chunks=None)
    assert calls == [(2018, "EPSG:32720"), (2019, "EPSG:32720")]
    assert cube.dims == ("time", "band", "y", "x") and cube.sizes["band"] == 64
    assert np.isnan(cube.values[:, :, 0, 0]).all()
    np.testing.assert_allclose(cube.values[:, 0, 1, 1], [0.18, 0.19])
    assert cube.attrs["embedding_backend"] == "gee"
    zeit.load_embeddings(bbox, source="alphaearth", backend="gee", years=[2018, 2019], cache_dir=tmp_path)
    assert len(calls) == 2          # already downloaded
    with pytest.raises(ValueError, match="years"):
        zeit.load_embeddings(bbox, source="alphaearth", backend="gee", cache_dir=tmp_path)


# ---------------------------------------------------------------------- the rest of zeit
def _scene(source="tessera", n=6, size=12, change_at=None):
    """Embeddings of a synthetic scene: two kinds of land (left and right halves), each year the
    same vectors plus a little noise; the pixels of ``change_at`` (year, rows, cols) switch kind."""
    rng = np.random.default_rng(5)
    years = list(range(2017, 2017 + n))
    a, b = rng.normal(size=NB), rng.normal(size=NB)
    data = np.empty((n, NB, size, size), np.float32)
    for i, year in enumerate(years):
        for r in range(size):
            for c in range(size):
                kind = a if c < size // 2 else b
                if change_at and year >= change_at[0] and r in change_at[1] and c in change_at[2]:
                    kind = b if c < size // 2 else a
                data[i, :, r, c] = kind + rng.normal(0, 0.05, NB)
    da = xr.DataArray(data, dims=("time", "band", "y", "x"),
                      coords={"time": pd.DatetimeIndex([pd.Timestamp(y, 1, 1) for y in years]),
                              "band": [f"A{i:02d}" for i in range(NB)],
                              "y": 8900000 - 5 - 10 * np.arange(size), "x": 500000 + 5 + 10 * np.arange(size)})
    da = da.rio.write_crs(32720)
    da.attrs.update({"embedding_source": source, "embedding_version": "1.1", "embedding_variant": "dclimate"})
    return da, a, b


def test_save_and_load_keep_what_the_embeddings_are(tmp_path):
    cube, _, _ = _scene()
    path = zeit.save_raster(cube.isel(time=[0, 1]), tmp_path / "emb.tif")
    with rasterio.open(path) as src:
        assert json.loads(src.tags()["ZEIT_EMBEDDING"])["embedding_source"] == "tessera"
    back = zeit.load_raster(path)
    assert back.dims == ("time", "band", "y", "x")
    assert list(back.band.values) == list(cube.band.values)
    assert _embeddings.embedding_meta(back) == _embeddings.embedding_meta(cube)
    np.testing.assert_allclose(back.values, cube.isel(time=[0, 1]).values)


def test_classifier_remembers_the_embeddings():
    import geopandas as gpd
    from shapely.geometry import Point

    cube, _, _ = _scene()
    year = cube.isel(time=-1)
    pts = gpd.GeoDataFrame({"class": ["a", "a", "b", "b"]},
                           geometry=[Point(500015, 8899985), Point(500025, 8899905), Point(500105, 8899985),
                                     Point(500095, 8899905)], crs=32720)
    rf = zeit.train_classifier(year, pts)
    assert rf.zeit_embedding_["embedding_source"] == "tessera"
    classes = zeit.classify(year, rf)
    left, right = classes.label.values[:, :6], classes.label.values[:, 6:]
    assert (left == 1).all() and (right == 2).all()
    other = year.copy()
    other.attrs.update(embedding_source="alphaearth", embedding_version="1", embedding_variant="annual")
    with pytest.raises(ValueError, match="trained on TESSERA"):
        zeit.classify(other, rf)
    other.attrs.update(embedding_source="tessera", embedding_version="1.1", embedding_variant="cambridge")
    with pytest.raises(ValueError, match="different spaces"):
        zeit.classify(other, rf)
    plain = year.copy()
    plain.attrs = {}
    zeit.classify(plain, rf)       # attributes lost on the way: nothing to check against


def test_samples_for_zeit_ai_remember_the_embeddings():
    import geopandas as gpd
    from shapely.geometry import Point

    cube, _, _ = _scene()
    pts = gpd.GeoDataFrame({"class": ["a", "b"]}, geometry=[Point(500015, 8899985), Point(500105, 8899985)],
                           crs=32720)
    s = zeit.ai.samples(cube, pts, split=0)
    assert s.meta["embedding"]["embedding_source"] == "tessera"


def test_algorithms_that_need_a_physical_series_refuse_embeddings():
    cube, _, _ = _scene()
    for run in (zeit.ccdc, zeit.landtrendr, zeit.bfast_lite, zeit.phenology, zeit.regularize_time_series):
        with pytest.raises(ValueError, match="embeddings"):
            run(cube)
    with pytest.raises(ValueError, match="embeddings"):
        zeit.compute_indices(cube, ["NDVI"])


def test_similarity():
    import geopandas as gpd
    from shapely.geometry import Point, box

    cube, a, b = _scene(change_at=(2020, range(0, 3), range(0, 3)))
    sim = zeit.similarity(cube, a)
    assert sim.dims == ("time", "y", "x") and sim.dtype == np.float32
    v = cube.values[2, :, 5, 5]
    np.testing.assert_allclose(sim.values[2, 5, 5], v @ a / np.linalg.norm(v) / np.linalg.norm(a), rtol=1e-5)
    assert sim.values[0, 0, 0] > 0.99 and sim.values[-1, 0, 0] < sim.values[0, 0, 0]   # switched kind in 2020
    samples = gpd.GeoDataFrame({"kind": ["left", "right"]},
                               geometry=[box(500000, 8899880, 500060, 8900000), Point(500105, 8899985)], crs=32720)
    by = zeit.similarity(cube, samples, by="kind", year=2017)
    assert by.dims == ("class", "time", "y", "x") and list(by["class"].values) == ["left", "right"]
    assert (by.sel({"class": "left"}).values[0, :, :6] > 0.99).all()
    assert (by.sel({"class": "right"}).values[0, :, 6:] > 0.99).all()
    dist = zeit.similarity(cube, a, metric="euclidean")
    np.testing.assert_allclose(dist.values[2, 5, 5], np.linalg.norm(v - a), rtol=1e-5)
    assert by.attrs["embedding_source"] == "tessera"
    with pytest.raises(ValueError, match="one per band"):
        zeit.similarity(cube, [1.0, 2.0])
    with pytest.raises(ValueError, match="year"):
        zeit.similarity(cube, samples, year=2030)


def test_embedding_change_and_its_events():
    rows, cols = range(2, 5), range(1, 4)
    cube, _, _ = _scene(change_at=(2020, rows, cols))
    change = zeit.embedding_change(cube)
    assert change.distance.dims == ("time", "y", "x")
    assert list(change.time.dt.year.values) == [2018, 2019, 2020, 2021, 2022]
    d = change.distance.values
    assert d[2, 3, 2] > 1.0 and d[:, 0, 0].max() < 0.1    # the switch in 2020; a stable pixel
    events = zeit.extract_events(change, min_magnitude=0.5)
    yod = events.yod.values
    assert (yod[2:5, 1:4] == 2019).all() and (yod > 0).sum() == 9
    assert (events.date.values[3, 2] == np.datetime64("2020-01-01"))
    assert events.attrs["algorithm"] == "embedding_change"
    assert np.isfinite(events.dsnr.values[3, 2]) and events.dsnr.values[3, 2] > 10
    with pytest.raises(ValueError, match="direction"):
        zeit.extract_events(change, event_type="loss")
    first = zeit.embedding_change(cube, baseline="first")
    assert first.distance.values[-1, 3, 2] > 1.0 and first.attrs["baseline"] == "first"
    with pytest.raises(ValueError, match="previous"):
        zeit.extract_events(first)
    fixed = zeit.embedding_change(cube, baseline=2019, metric="euclidean")
    assert 2019 not in fixed.time.dt.year.values
    lazy = zeit.embedding_change(cube.chunk({"y": 6, "x": 6}))
    np.testing.assert_allclose(lazy.distance.values, d)
    agree = zeit.agreement(events, zeit.extract_events(lazy, min_magnitude=0.5), tolerance=1)
    assert (agree.n_agree.values[2:5, 1:4] == 2).all()
    with pytest.raises(ValueError, match="two years"):
        zeit.embedding_change(cube.isel(time=[0]))


def test_plot_shows_principal_components():
    from zeit._plot._data import prepare
    from zeit._embedding_tools import pca_rgb

    cube, _, _ = _scene(change_at=(2020, range(0, 3), range(0, 3)))
    frames, _ = prepare(cube)
    assert frames.rgb and frames.n == cube.sizes["time"]
    rgb = pca_rgb(cube)
    assert list(rgb.band.values) == ["red", "green", "blue"] and float(rgb.min()) >= 0 and float(rgb.max()) <= 1
    # the same embedding has the same colour in every year (here the first component tells the
    # two kinds apart; the others are noise): a stable pixel keeps it, one that switched does not
    assert abs(rgb.values[0, 0, 8, 8] - rgb.values[-1, 0, 8, 8]) < 0.1
    assert abs(rgb.values[0, 0, 0, 0] - rgb.values[-1, 0, 0, 0]) > 0.5
    one, _ = prepare(cube, band="A03")
    assert not one.rgb
    pytest.importorskip("matplotlib")
    fig = zeit.plot(cube, static=True, time=["2017", "2022"])
    assert fig is not None


def test_cli_downloads_embeddings(tessera, tmp_path, monkeypatch):
    from zeit import cli

    path, zones = tessera
    x0, y0, _, _ = zones[20]
    w, s, e, n = _bbox_of(x0, y0, 32620, 0, 10, 0, 10)
    monkeypatch.setattr(sys, "argv", ["zeit", "embeddings", str(tmp_path), "--source", "tessera", "--bbox", str(w),
                                      str(s), str(e), str(n), "--years", "2018-2019", "--store", str(path)])
    cli.main()
    back = zeit.load_raster(tmp_path / "tessera.tif")
    assert back.dims == ("time", "band", "y", "x") and list(back.time.dt.year.values) == [2018, 2019]
    assert back.attrs["embedding_source"] == "tessera"
