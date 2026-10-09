"""zeit.plot's basemap and vector overlays."""
import io

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._plot import _overlay
from zeit._plot._data import prepare
from zeit._plot._session import Session


def _cube(crs=4326):
    if crs == 4326:
        x = -63.3 + (np.arange(40) + 0.5) * 0.001
        y = -10.1 - (np.arange(30) + 0.5) * 0.001
    else:   # UTM 20S, the same area
        x = 465000 + (np.arange(40) + 0.5) * 30.0
        y = 8880000 - (np.arange(30) + 0.5) * 30.0
    data = np.random.default_rng(0).uniform(0.2, 0.9, (3, 30, 40)).astype(np.float32)
    return xr.DataArray(data, dims=("time", "y", "x"), name="ndvi",
                        coords={"time": pd.date_range("2000", periods=3, freq="YS"), "y": y, "x": x}).rio.write_crs(crs)


def _interp(g, x, y):
    fx = (x - g["x0"]) / (g["x1"] - g["x0"]) * (g["nx"] - 1)
    fy = (y - g["y0"]) / (g["y1"] - g["y0"]) * (g["ny"] - 1)
    i, j = min(max(int(np.floor(fx)), 0), g["nx"] - 2), min(max(int(np.floor(fy)), 0), g["ny"] - 2)
    tx, ty, k = fx - i, fy - j, j * g["nx"] + i
    a = lambda arr: (arr[k] * (1 - tx) + arr[k + 1] * tx) * (1 - ty) + (arr[k + g["nx"]] * (1 - tx) + arr[k + g["nx"] + 1] * tx) * ty
    return a(g["a"]), a(g["b"])


def test_providers():
    assert "arcgisonline" in _overlay.basemap_provider("satellite")["url"]
    assert _overlay.basemap_provider(True)["url"] == _overlay.basemap_provider("satellite")["url"]
    assert _overlay.basemap_provider("https://t/{z}/{x}/{y}.png")["url"] == "https://t/{z}/{x}/{y}.png"
    assert _overlay.basemap_provider(None) is None
    xyz = pytest.importorskip("xyzservices")
    assert "{z}" in _overlay.basemap_provider("Esri.WorldTopoMap")["url"]
    with pytest.raises(ValueError, match="unknown basemap"):
        _overlay.basemap_provider("Nope.Nothing")


@pytest.mark.parametrize("crs", [4326, 32720])
def test_geo_grids_round_trip(crs):
    from rasterio.warp import transform

    cube = _cube(crs)
    frames = prepare(cube)[0]
    grids = _overlay.geo_grids(frames, cube.rio.crs)
    # cell (10, 5) -> lon/lat through the forward grid == the exact transform
    left, top = float(cube.x[0]) - (float(cube.x[1]) - float(cube.x[0])) / 2, float(cube.y[0]) + abs(float(cube.y[1] - cube.y[0])) / 2
    dx, dy = float(cube.x[1] - cube.x[0]), abs(float(cube.y[1] - cube.y[0]))
    lon, lat = _interp(grids["forward"], 10.0, 5.0)
    (elon,), (elat,) = transform(cube.rio.crs, "EPSG:4326", [left + 10 * dx], [top - 5 * dy])
    assert lon == pytest.approx(elon, abs=1e-6) and lat == pytest.approx(elat, abs=1e-6)
    # and back through the inverse grid
    cx, cy = _interp(grids["inverse"], lon, lat)
    assert cx == pytest.approx(10.0, abs=0.05) and cy == pytest.approx(5.0, abs=0.05)
    assert _overlay.geo_grids(frames, None) is None


def test_vector_paths():
    from shapely.geometry import LineString, Point, Polygon
    import geopandas as gpd

    cube = _cube()
    frames = prepare(cube)[0]
    square = Polygon([(-63.29, -10.11), (-63.28, -10.11), (-63.28, -10.12), (-63.29, -10.12)])
    gdf = gpd.GeoDataFrame(geometry=[square, LineString([(-63.3, -10.1), (-63.26, -10.13)]), Point(-63.29, -10.11)],
                           crs=4326)
    content, buffer = _overlay.vector_paths(frames, gdf, cube.rio.crs, step=1)
    xy = np.frombuffer(buffer, np.float32).reshape(-1, 2)
    assert content["count"] == 3
    (o, n, closed), (o2, n2, closed2), (o3, n3, _) = content["paths"]
    assert closed == 1 and closed2 == 0 and n3 == 1
    np.testing.assert_allclose(xy[o3], [10.0, 10.0], atol=1e-3)            # (-63.29, -10.11) -> cell (10, 10)
    np.testing.assert_allclose(xy[o2], [0.0, 0.0], atol=1e-3)              # the top-left corner
    utm = gdf.to_crs(32720)                                                 # reprojected to the data's CRS
    content2, buffer2 = _overlay.vector_paths(frames, utm, cube.rio.crs, step=1)
    np.testing.assert_allclose(np.frombuffer(buffer2, np.float32).reshape(-1, 2)[o3], [10.0, 10.0], atol=0.05)


def test_session_meta_and_vectors():
    from shapely.geometry import box
    cube = _cube()
    session = Session(cube, basemap="satellite", vector=[box(-63.29, -10.12, -63.28, -10.11)], opacity=0.7)
    meta, _ = session.handle({"type": "meta"})
    assert meta["basemap"]["max_zoom"] == 19 and meta["opacity"] == 0.7 and meta["has_vectors"]
    assert set(meta["geo"]) == {"forward", "inverse"}
    content, (buffer,) = session.handle({"type": "vectors"})
    assert content["count"] == 1 and len(buffer) == 5 * 2 * 4
    plain = Session(cube.drop_vars("spatial_ref"), basemap="osm")
    assert plain.handle({"type": "meta"})[0]["basemap"] is None              # no CRS: no basemap


def test_static_basemap_with_fake_tiles(monkeypatch, tmp_path):
    from PIL import Image

    def fake_fetch(url):
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), (10, 120, 200)).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(_overlay, "_fetch", fake_fetch)
    cube = _cube()
    image, extent, attribution = _overlay.static_basemap(prepare(cube)[0], cube.rio.crs, "satellite", width_px=200)
    assert image.shape[1] == 200 and (image[image.shape[0] // 2, 100] == [10, 120, 200]).all()
    assert "Esri" in attribution
    fig = zeit.plot(cube.isel(time=0), basemap="satellite", vector=[], static=True, save=str(tmp_path / "b.png"))
    assert len(fig.axes[0].images) == 2                                      # basemap + data
    assert fig.axes[0].images[1].get_alpha() == 0.8


def test_offline_basemap_is_skipped(monkeypatch):
    monkeypatch.setattr(_overlay, "_fetch", lambda url: None)
    cube = _cube()
    assert _overlay.static_basemap(prepare(cube)[0], cube.rio.crs, "osm") is None
    fig = zeit.plot(cube.isel(time=0), basemap="osm", static=True)
    assert len(fig.axes[0].images) == 1
