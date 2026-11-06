"""zeit.sentinel2_angles: the MTD_TL.xml angle grids, fused over detectors and interpolated to a cube's pixels."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

import zeit

DATA = Path(__file__).parent / "data"
ULX, ULY = 300000.0, 8000000.0
N = 23
STEP = 5000.0
I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")     # row (south), column (east) of the nodes


# The synthetic tile: every field is linear in the node's (row i, column j), so a bilinear interpolation reproduces
# it exactly; detector 1 reaches columns 0-12, detector 2 columns 10-22 (they overlap on 10-12), NaN elsewhere.
def sun_zenith(i, j):
    return 30.0 + 0.1 * j + 0.05 * i


def sun_azimuth(i, j):
    return 100.0 + 0.2 * i - 0.3 * j


def view_zenith(i, j, band=5):
    return 5.0 + 0.1 * j + 0.02 * i + 0.5 * (band != 5)


def view_azimuth(i, j, band=5):
    return (358.0 + 0.5 * j + 0.1 * i + 3.0 * (band != 5)) % 360.0


def _values(grid):
    rows = "".join(f"<VALUES>{' '.join('NaN' if not np.isfinite(v) else repr(float(v)) for v in r)}</VALUES>"
                   for r in grid)
    return f"<COL_STEP unit=\"m\">5000</COL_STEP><ROW_STEP unit=\"m\">5000</ROW_STEP><Values_List>{rows}</Values_List>"


def make_xml(sensing="2023-07-23T13:36:41.935298Z", bands=(5, 3), cover=((0, 12), (10, 22)), offset2=(0.5, 2.0),
             epsg=32723, ulx=ULX, uly=ULY):
    """An MTD_TL.xml with the synthetic fields. Detector 2's zenith is +offset2[0] and its azimuth +offset2[1]
    degrees from detector 1's where both are valid."""
    sun = (f"<Sun_Angles_Grid><Zenith>{_values(sun_zenith(I, J))}</Zenith>"
           f"<Azimuth>{_values(sun_azimuth(I, J))}</Azimuth></Sun_Angles_Grid>")
    views = ""
    for b in bands:
        for d, (lo, hi) in enumerate(cover, start=1):
            ok = (J >= lo) & (J <= hi)
            z = np.where(ok, view_zenith(I, J, b) + (offset2[0] if d == 2 else 0.0), np.nan)
            a = np.where(ok, (view_azimuth(I, J, b) + (offset2[1] if d == 2 else 0.0)) % 360.0, np.nan)
            views += (f"<Viewing_Incidence_Angles_Grids bandId=\"{b}\" detectorId=\"{d}\"><Zenith>{_values(z)}"
                      f"</Zenith><Azimuth>{_values(a)}</Azimuth></Viewing_Incidence_Angles_Grids>")
    geo = "".join(f"<Geoposition resolution=\"{r}\"><ULX>{ulx:.0f}</ULX><ULY>{uly:.0f}</ULY><XDIM>{r}</XDIM>"
                  f"<YDIM>-{r}</YDIM></Geoposition>" for r in (10, 20, 60))
    return (f"<?xml version=\"1.0\"?><n1:Level-2A_Tile_ID xmlns:n1=\"https://psd-14.sentinel2.eo.esa.int/PSD/x.xsd\">"
            f"<n1:General_Info><SENSING_TIME>{sensing}</SENSING_TIME></n1:General_Info>"
            f"<n1:Geometric_Info><Tile_Geocoding><HORIZONTAL_CS_CODE>EPSG:{epsg}</HORIZONTAL_CS_CODE>{geo}"
            f"</Tile_Geocoding><Tile_Angles>{sun}<Mean_Sun_Angle><ZENITH_ANGLE>1</ZENITH_ANGLE></Mean_Sun_Angle>"
            f"{views}</Tile_Angles></n1:Geometric_Info></n1:Level-2A_Tile_ID>")


def grid_cube(xs, ys, times=("2023-07-23",), bands=("B04", "B06"), crs=32723, dtype=np.float32):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    data = np.full((len(times), len(bands), ys.size, xs.size), 0.2, dtype=dtype)
    return xr.DataArray(data, dims=("time", "band", "y", "x"), name="reflectance",
                        coords={"time": pd.to_datetime(list(times)), "band": list(bands), "y": ys, "x": xs}
                        ).rio.write_crs(crs)


def circ(a, b):
    """The angular distance in degrees."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


@pytest.fixture()
def xml_path(tmp_path):
    p = tmp_path / "MTD_TL.xml"
    p.write_text(make_xml(), encoding="utf-8")
    return p


def test_interpolation_at_known_points(xml_path):
    # pixel centres in the tile's metres: nodes and the middle of cells, south-up image (y descending)
    u = np.array([0.0, 0.3, 4.5, 8.25, 9.0])               # single detector (columns 0-9: only detector 1)
    v = np.array([0.0, 1.5, 7.2, 21.9])
    x, y = ULX + u * STEP, ULY - v * STEP
    angles = zeit.sentinel2_angles(xml_path, grid_cube(x, y))
    assert set(angles.data_vars) == {"SZA", "SAA", "VZA", "VAA"}
    assert angles.SZA.dims == ("time", "y", "x") and angles.SZA.dtype == np.float32
    uu, vv = np.meshgrid(u, v)
    np.testing.assert_allclose(angles.SZA.values[0], sun_zenith(vv, uu), atol=2e-5)
    np.testing.assert_allclose(angles.SAA.values[0], sun_azimuth(vv, uu), atol=2e-5)
    np.testing.assert_allclose(angles.VZA.values[0], view_zenith(vv, uu), atol=2e-5)
    # azimuths wrap (358 + 0.5 j crosses 360 at j = 4): their sin/cos interpolation follows the line
    assert circ(angles.VAA.values[0], view_azimuth(vv, uu)).max() < 2e-3
    assert angles.VAA.values.min() >= 0 and angles.VAA.values.max() < 360
    # both ways of ordering y give the same angles
    flipped = zeit.sentinel2_angles(xml_path, grid_cube(x, y[::-1]))
    np.testing.assert_array_equal(flipped.VZA.values[0], angles.VZA.values[0][::-1])


def test_detectors_are_fused_by_the_mean_per_cell(xml_path):
    # at the nodes themselves: detector 1 alone (j < 10), both (10-12), detector 2 alone (j > 12)
    j = np.arange(0, 23)
    x, y = ULX + j * STEP, np.array([ULY - 4 * STEP])
    a = zeit.sentinel2_angles(xml_path, grid_cube(x, y))
    want_z = view_zenith(4, j) + np.where((j >= 10) & (j <= 12), 0.25, np.where(j > 12, 0.5, 0.0))
    np.testing.assert_allclose(a.VZA.values[0, 0], want_z, atol=2e-5)
    # azimuth: circular mean of a and a + 2 is a + 1 on the overlap, a + 2 where only detector 2 sees
    want_a = view_azimuth(4, j) + np.where((j >= 10) & (j <= 12), 1.0, np.where(j > 12, 2.0, 0.0))
    assert circ(a.VAA.values[0, 0], want_a % 360).max() < 2e-3
    # between nodes of the overlap the field is the bilinear of the fused nodes
    xm = np.array([ULX + 10.5 * STEP, ULX + 11.9 * STEP, ULX + 9.5 * STEP])
    b = zeit.sentinel2_angles(xml_path, grid_cube(xm, y))
    np.testing.assert_allclose(b.VZA.values[0, 0], [view_zenith(4, 10.5) + 0.25, view_zenith(4, 11.9) + 0.25,
                                                    view_zenith(4, 9.5) + 0.125], atol=2e-5)


def test_azimuth_wraps_through_north():
    # at the node (row 0, column 2) detector 1's azimuth is 359 and detector 2's 361 = 1: the circular mean is 0
    # (an arithmetic mean would be 180)
    text = make_xml(offset2=(0.0, 2.0), cover=((0, 22), (0, 22)))
    a = zeit.sentinel2_angles(text, grid_cube([ULX + 2 * STEP], [ULY]))
    assert circ(a.VAA.values[0, 0, 0], 0.0) < 1e-3
    # and between nodes, 359.5 + 0.5 * (u - 2) crosses north halfway to the next node
    a = zeit.sentinel2_angles(make_xml(), grid_cube([ULX + 3 * STEP], [ULY]))
    assert circ(a.VAA.values[0, 0, 0], 359.5) < 2e-3


def test_nan_where_no_detector_covers_and_flat_to_the_valid_corner():
    text = make_xml(cover=((0, 12), (30, 31)))                  # only detector 1, columns 0-12
    ev = grid_cube(ULX + np.array([11.0, 12.0, 12.5, 12.999, 13.2, 20.0]) * STEP, [ULY - 5 * STEP])
    a = zeit.sentinel2_angles(text, ev)
    vza = a.VZA.values[0, 0]
    assert np.allclose(vza[:2], view_zenith(5, np.array([11.0, 12.0])), atol=2e-5)
    # between the last valid node (12) and the invalid one (13) the valid corner's value is kept (no extrapolation)
    assert np.allclose(vza[2:4], view_zenith(5, 12), atol=2e-5)
    assert np.isnan(vza[4:]).all()                                  # the cell 13-14 has no valid corner
    assert np.isnan(a.VAA.values[0, 0, 4:]).all()
    # the sun's grid is complete: never NaN inside the grid
    assert np.isfinite(a.SZA.values).all() and np.isfinite(a.SAA.values).all()


def test_outside_the_grid_is_nan(xml_path):
    xs = ULX + np.array([-100.0, 0.0, 110000.0, 110100.0])
    ys = np.array([ULY - 10 * STEP, ULY + 100.0, ULY - 22 * STEP - 50.0])
    a = zeit.sentinel2_angles(xml_path, grid_cube(xs, ys))
    sza = a.SZA.values[0]
    assert np.isnan(sza[:, 0]).all() and np.isnan(sza[:, 3]).all()      # west of and beyond the last column
    assert np.isnan(sza[1]).all() and np.isnan(sza[2]).all()            # north of the first row, south of the last
    assert np.isfinite(sza[0, 1]) and np.isfinite(sza[0, 2])


def test_aligned_to_the_cube_for_several_dates_and_lazily(tmp_path):
    p1, p2 = tmp_path / "a.xml", tmp_path / "b.xml"
    p1.write_text(make_xml("2023-07-23T13:36:41Z"), encoding="utf-8")
    p2.write_text(make_xml("2023-07-28T13:36:41Z", offset2=(0.7, 2.0)), encoding="utf-8")
    xs = ULX + 2500 + np.arange(60) * 1700.0
    ys = ULY - 2500 - np.arange(50) * 700.0
    cube = grid_cube(xs, ys, times=("2023-07-23", "2023-07-28"))
    a = zeit.sentinel2_angles([p1, str(p2)], cube)
    assert a.SZA.dims == ("time", "y", "x") and a.SZA.shape == (2, 50, 60)
    np.testing.assert_array_equal(a.x.values, cube.x.values)
    np.testing.assert_array_equal(a.y.values, cube.y.values)
    np.testing.assert_array_equal(a.time.values, cube.time.values)
    assert a.rio.crs == cube.rio.crs
    assert a.attrs["sources"]
    # the second date differs from the first where detector 2's offset reaches (east of column 10)
    east = a.x.values > ULX + 13 * STEP
    assert not np.allclose(a.VZA.values[0][:, east], a.VZA.values[1][:, east])
    # lazy with dask: same values, stays lazy, the cube's chunks
    lazy = zeit.sentinel2_angles([p1, p2], cube.chunk({"time": 1, "y": 20, "x": 25}))
    assert lazy.VZA.chunks is not None and lazy.VZA.chunks[1] == (20, 20, 10) and lazy.VZA.chunks[2] == (25, 25, 10)
    for k in a:
        np.testing.assert_array_equal(lazy[k].values, a[k].values)
    forced = zeit.sentinel2_angles([p1, p2], cube, chunks={"y": 30, "x": 30})
    assert forced.SZA.chunks is not None
    np.testing.assert_array_equal(forced.VAA.values, a.VAA.values)
    # a threaded run equals the sequential one
    one = zeit.sentinel2_angles([p1, p2], cube, n_jobs=1)
    np.testing.assert_array_equal(one.VZA.values, a.VZA.values)
    # a cube without time: a single source
    flat = zeit.sentinel2_angles(p1, cube.isel(time=0, drop=True))
    assert flat.VZA.dims == ("y", "x")
    np.testing.assert_array_equal(flat.VZA.values, a.VZA.values[0])


def test_strips_of_rows_equal_one_block(xml_path, monkeypatch):
    import zeit._s2angles as s2

    cube = grid_cube(ULX + 100 + np.arange(40) * 2500.0, ULY - 100 - np.arange(37) * 2800.0)
    whole = zeit.sentinel2_angles(xml_path, cube, n_jobs=1)
    monkeypatch.setattr(s2, "_STRIP", 7)
    strips = zeit.sentinel2_angles(xml_path, cube, n_jobs=3)
    for k in whole:
        np.testing.assert_array_equal(strips[k].values, whole[k].values)


def test_cube_in_another_crs_is_projected_to_the_tile(xml_path):
    # a cube in UTM zone 22S, the neighbouring zone: pixel centres are projected to the tile's zone 23S
    to22 = Transformer.from_crs(32723, 32722, always_xy=True)
    to23 = Transformer.from_crs(32722, 32723, always_xy=True)
    x23, y23 = ULX + 3 * STEP, ULY - 4 * STEP
    x22, y22 = to22.transform(x23, y23)
    xs = x22 + np.arange(8) * 900.0
    ys = y22 - np.arange(6) * 1100.0
    cube = grid_cube(xs, ys, crs=32722)
    a = zeit.sentinel2_angles(xml_path, cube)
    gx, gy = np.meshgrid(xs, ys)
    tx, ty = to23.transform(gx, gy)
    u, v = (tx - ULX) / STEP, (ULY - ty) / STEP
    np.testing.assert_allclose(a.SZA.values[0], sun_zenith(v, u), atol=1e-4)
    np.testing.assert_allclose(a.VZA.values[0], view_zenith(v, u), atol=1e-4)
    assert circ(a.VAA.values[0], view_azimuth(v, u)).max() < 2e-3
    assert a.rio.crs == cube.rio.crs
    # geographic coordinates too
    to4326 = Transformer.from_crs(32723, 4326, always_xy=True)
    lon, lat = to4326.transform(x23, y23)
    cube = grid_cube(lon + np.arange(5) * 0.004, lat - np.arange(4) * 0.004, crs=4326)
    a = zeit.sentinel2_angles(xml_path, cube)
    gx, gy = np.meshgrid(cube.x.values, cube.y.values)
    tx, ty = Transformer.from_crs(4326, 32723, always_xy=True).transform(gx, gy)
    np.testing.assert_allclose(a.SZA.values[0], sun_zenith((ULY - ty) / STEP, (tx - ULX) / STEP), atol=1e-4)


def test_view_band_choices_and_each(xml_path):
    xs, ys = ULX + 2 * STEP + np.arange(5) * 300.0, ULY - 3 * STEP - np.arange(4) * 300.0
    cube = grid_cube(xs, ys, bands=("B04", "B06", "B09"))
    b06 = zeit.sentinel2_angles(xml_path, cube)
    by_name = zeit.sentinel2_angles(xml_path, cube, view_band="rededge2")
    np.testing.assert_array_equal(by_name.VZA.values, b06.VZA.values)
    red = zeit.sentinel2_angles(xml_path, cube, view_band="B04")
    uu, vv = np.meshgrid((xs - ULX) / STEP, (ULY - ys) / STEP)
    np.testing.assert_allclose(red.VZA.values[0], view_zenith(vv, uu, band=3), atol=2e-5)
    each = zeit.sentinel2_angles(xml_path, cube.sel(band=["B04", "B06"]), view_band="each")
    assert each.VZA.dims == ("time", "band", "y", "x") and list(each.band.values) == ["B04", "B06"]
    assert each.SZA.dims == ("time", "y", "x")
    np.testing.assert_array_equal(each.VZA.values[0, 0], red.VZA.values[0])
    np.testing.assert_array_equal(each.VAA.values[0, 1], b06.VAA.values[0])
    # a band the tile has no view angles for
    with pytest.raises(ValueError, match="no view angles for band id 9"):
        zeit.sentinel2_angles(xml_path, cube, view_band="each")
    with pytest.raises(ValueError, match="not a Sentinel-2 band"):
        zeit.sentinel2_angles(xml_path, cube, view_band="VV")
    with pytest.raises(ValueError, match="band dimension"):
        zeit.sentinel2_angles(xml_path, cube.isel(band=0, drop=True), view_band="each")


def test_sources_as_stac_items_and_errors(tmp_path):
    p = tmp_path / "granule_metadata.xml"
    p.write_text(make_xml(), encoding="utf-8")
    cube = grid_cube(ULX + np.arange(3) * 1000.0 + 500, ULY - np.arange(3) * 1000.0 - 500)
    ref = zeit.sentinel2_angles(p, cube)
    item = {"type": "Feature", "id": "S2A_X", "assets": {"granule_metadata": {"href": str(p)}}}
    np.testing.assert_array_equal(zeit.sentinel2_angles(item, cube).VZA.values, ref.VZA.values)
    pc = {"type": "Feature", "id": "S2A_X", "assets": {"granule-metadata": {"href": str(p)}}}
    np.testing.assert_array_equal(zeit.sentinel2_angles([pc], cube).VZA.values, ref.VZA.values)
    pystac = pytest.importorskip("pystac")
    it = pystac.Item("S2A_X", None, None, pd.Timestamp("2023-07-23").to_pydatetime(), {})
    it.add_asset("granule_metadata", pystac.Asset(href=str(p)))
    np.testing.assert_array_equal(zeit.sentinel2_angles(it, cube).VZA.values, ref.VZA.values)
    np.testing.assert_array_equal(zeit.sentinel2_angles(pystac.ItemCollection([it]), cube).VZA.values, ref.VZA.values)
    # the text itself
    np.testing.assert_array_equal(zeit.sentinel2_angles(make_xml(), cube).VZA.values, ref.VZA.values)
    with pytest.raises(ValueError, match="granule_metadata or granule-metadata"):
        zeit.sentinel2_angles({"assets": {"B04": {"href": str(p)}}}, cube)
    with pytest.raises(ValueError, match="2 sources for the 1 date"):
        zeit.sentinel2_angles([p, p], cube)
    with pytest.raises(ValueError, match="sensed on 2023-07-23"):
        zeit.sentinel2_angles(p, grid_cube([ULX + 500], [ULY - 500], times=("2023-09-01",)))
    with pytest.raises(ValueError, match="no CRS"):
        zeit.sentinel2_angles(p, cube.drop_vars("spatial_ref").assign_attrs())
    with pytest.raises(ValueError, match="Tile_Geocoding"):
        zeit.sentinel2_angles("<a><b/></a>", cube)
    with pytest.raises(ValueError, match="DataArray"):
        zeit.sentinel2_angles(p, "cube")


def test_mosaic_of_two_tiles_takes_the_first_with_angles(tmp_path):
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text(make_xml(cover=((0, 12), (30, 31))), encoding="utf-8")                # reaches column 12 only
    b.write_text(make_xml(cover=((0, 22), (30, 31)), offset2=(0, 0)), encoding="utf-8")
    cube = grid_cube(ULX + np.array([6.0, 16.0]) * STEP, [ULY - 5 * STEP])
    one = zeit.sentinel2_angles([[a]], cube)
    assert np.isfinite(one.VZA.values[0, 0, 0]) and np.isnan(one.VZA.values[0, 0, 1])
    both = zeit.sentinel2_angles([[a, b]], cube)
    assert np.isfinite(both.VZA.values[0, 0]).all()
    assert both.VZA.values[0, 0, 0] == one.VZA.values[0, 0, 0]
    np.testing.assert_allclose(both.VZA.values[0, 0, 1], view_zenith(5, 16), atol=2e-5)


def test_end_to_end_with_nbar(xml_path):
    # the region of single-detector columns 0-9, so the fields are the linear ones: nbar through the reader equals
    # nbar through angles written from the formulas
    xs = ULX + 300 + np.arange(12) * 2000.0                       # u in 0.06 - 4.4
    ys = ULY - 300 - np.arange(9) * 2200.0
    cube = grid_cube(xs, ys, bands=("B04", "B06", "B08"))
    cube[:] = np.random.default_rng(3).uniform(0.05, 0.4, cube.shape).astype(np.float32)
    got = zeit.nbar(cube, zeit.sentinel2_angles(xml_path, cube))
    uu, vv = np.meshgrid((xs - ULX) / STEP, (ULY - ys) / STEP)
    dims = ("time", "y", "x")
    manual = xr.Dataset({k: xr.DataArray(v[None].astype(np.float32), dims=dims,
                                         coords={d: cube[d] for d in dims})
                         for k, v in dict(SZA=sun_zenith(vv, uu), SAA=sun_azimuth(vv, uu),
                                          VZA=view_zenith(vv, uu), VAA=view_azimuth(vv, uu)).items()})
    want = zeit.nbar(cube, manual)
    np.testing.assert_allclose(got.values, want.values, rtol=2e-5)
    assert (got.values != cube.values).any()
    # view_band="each": every band with its own view angles (B04 here has +0.5 degrees of zenith, +3 of azimuth)
    each = zeit.sentinel2_angles(xml_path, cube.sel(band=["B04", "B06"]), view_band="each")
    got = zeit.nbar(cube.sel(band=["B04", "B06"]), each)
    red = zeit.nbar(cube.sel(band=["B04"]), manual.assign(VZA=manual.VZA + 0.5, VAA=(manual.VAA + 3.0) % 360.0))
    np.testing.assert_allclose(got.sel(band="B04").values, red.sel(band="B04").values, rtol=2e-5)
    np.testing.assert_allclose(got.sel(band="B06").values, want.sel(band="B06").values, rtol=2e-5)
    assert not np.allclose(got.sel(band="B04").values, want.sel(band="B04").values, rtol=1e-6)


# --- the real tile metadata ------------------------------------------------------------------------------------

def _mean_angles(path):
    root = ET.parse(path).getroot()
    loc = lambda e: e.tag.rsplit("}", 1)[-1]
    out = {}
    for e in root.iter():
        if loc(e) == "Mean_Sun_Angle":
            out["sun"] = (float(e[0].text), float(e[1].text))
        elif loc(e) == "Mean_Viewing_Incidence_Angle" and e.get("bandId") == "5":
            out["view"] = (float(e[0].text), float(e[1].text))
    return out


def _tile_cube(step=60, times=("2023-07-23",), bands=("B04", "B06", "B8A")):
    n = 10980 // (step // 10)
    xs = 199980 + step / 2 + np.arange(n) * float(step)
    ys = 8300020 - step / 2 - np.arange(n) * float(step)
    return grid_cube(xs, ys, times=times, bands=bands, dtype=np.float32)


def test_real_tile_matches_the_mean_angles_of_the_xml():
    """The tile 23LKC of 2023-07-23 (Earth Search's granule_metadata, six detectors, overlapping swaths)."""
    path = DATA / "s2_23LKC_20230723_MTD_TL.xml"
    means = _mean_angles(path)
    a = zeit.sentinel2_angles(path, _tile_cube())
    assert a.SZA.shape == (1, 1830, 1830)
    assert abs(float(a.SZA.mean()) - means["sun"][0]) < 0.02                  # 44.778
    assert circ(float(a.SAA.mean()), means["sun"][1]) < 0.02                  # 37.590
    assert abs(float(a.VZA.mean()) - means["view"][0]) < 0.05                 # B06: 6.046
    vaa = np.radians(a.VAA.values)
    mean_az = np.degrees(np.arctan2(np.sin(vaa).mean(), np.cos(vaa).mean())) % 360
    assert circ(mean_az, means["view"][1]) < 1.0                              # 100.2
    # a whole tile is covered by detectors: no NaN, and the view zenith runs from the swath's centre (about 2 degrees)
    # to its edge (11) in the cross-track direction of the 290 km swath
    assert np.isfinite(a.VZA.values).all() and np.isfinite(a.VAA.values).all()
    assert 1.5 < float(a.VZA.min()) < 3 and 10.5 < float(a.VZA.max()) < 11.5
    # the detectors are fused: the angle varies smoothly across their overlaps (no step of more than 0.01 degrees per
    # 60 m pixel, where one detector's edge to the next differs by up to 1.5 degrees)
    assert np.abs(np.diff(a.VZA.values[0], axis=1)).max() < 0.01
    assert np.abs(np.diff(a.VZA.values[0], axis=0)).max() < 0.01
    # the grid's rows run south and its columns east: the sun is further from the zenith to the south
    assert a.SZA.values[0, -1, 0] > a.SZA.values[0, 0, 0]
    assert a.SZA.values[0, 0, -1] < a.SZA.values[0, 0, 0]


def test_real_tile_with_a_partial_swath_is_nan_where_no_detector_covers():
    """23LKC of 2023-07-26: the swath's edge, one detector with 61 of 529 grid cells valid."""
    path = DATA / "s2_23LKC_20230726_partial_MTD_TL.xml"
    means = _mean_angles(path)
    a = zeit.sentinel2_angles(path, _tile_cube(times=("2023-07-26",)))
    assert np.isfinite(a.SZA.values).all()                                   # the sun grid is complete
    valid = np.isfinite(a.VZA.values)
    assert 0.02 < valid.mean() < 0.2                                         # a strip along the west edge
    assert (valid == np.isfinite(a.VAA.values)).all()
    assert abs(float(np.nanmean(a.VZA.values)) - means["view"][0]) < 0.3
    # nbar of such a cube: NaN where there is no view angle
    cube = _tile_cube(times=("2023-07-26",))
    cube[:] = 0.2
    out = zeit.nbar(cube, a)
    assert np.isnan(out.values[0, 0][~valid[0]]).all() and np.isfinite(out.values[0, 0][valid[0]]).all()


def test_real_tile_each_band_and_dask_pipeline():
    path = DATA / "s2_23LKC_20230723_MTD_TL.xml"
    cube = _tile_cube(step=120, bands=("B02", "B08", "B8A", "B12"))
    each = zeit.sentinel2_angles(path, cube.chunk({"y": 256, "x": 256}), view_band="each")
    assert each.VZA.dims == ("time", "band", "y", "x") and each.sizes["band"] == 4
    z = each.VZA.compute()
    # the bands look at the tile from slightly different angles (the detectors are offset along track)
    d = np.abs(z.sel(band="B12").values - z.sel(band="B02").values)
    assert 0.01 < d.mean() < 1.0
    cube[:] = np.random.default_rng(0).uniform(0.05, 0.4, cube.shape).astype(np.float32)
    out = zeit.nbar(cube.chunk({"y": 256, "x": 256}), each)
    assert out.chunks is not None
    ratio = out.values / cube.values
    assert 0.9 < np.nanmin(ratio) and np.nanmax(ratio) < 1.1 and np.abs(ratio - 1).max() > 0.005
