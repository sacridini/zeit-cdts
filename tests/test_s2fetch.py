"""Finding and fetching the inputs of zeit.level2 (zeit._s2fetch): the STAC search, the download of a .SAFE from the bucket, the
Copernicus GLO-30 mosaic. No network: the HTTP layer (``_send``), the STAC client and the DEM tile search are replaced."""
import base64
import hashlib
import threading
import time
import types

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_origin

from zeit import _level2, _s2fetch

NAME = "S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149.SAFE"
PREFIX = f"tiles/23/L/KC/{NAME}/"
GRANULE = "GRANULE/L1C_T23LKC_A033383_20230728T132238"
BANDS = ("01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12")


def md5(data):
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def bucket_files():
    """The objects of a product in the bucket: what level2 reads, and a lot that it does not."""
    files = {
        "MTD_MSIL1C.xml": b"<product/>",
        f"{GRANULE}/MTD_TL.xml": b"<tile/>",
        f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_TCI.jp2": b"tci" * 50,
        "manifest.safe": b"manifest",
        "INSPIRE.xml": b"inspire",
        f"{GRANULE}/QI_DATA/MSK_CLOUDS_B00.gml": b"gml",
        f"{GRANULE}/AUX_DATA/AUX_ECMWFT": b"ecmwf",
        "DATASTRIP/DS_X/MTD_DS.xml": b"ds",
    }
    for b in BANDS:
        files[f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_B{b}.jp2"] = (b"band" + b.encode()) * 100
    return {PREFIX + k: v for k, v in files.items()}


class Response:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self.payload, self.content = status, payload, content

    def json(self):
        return self.payload

    def iter_content(self, n):
        for i in range(0, len(self.content), 100):
            yield self.content[i:i + 100]

    def close(self):
        pass


class FakeBucket:
    """Replaces ``_send``: the paginated listing and the downloads of a dict of objects."""

    def __init__(self, files, page=4, corrupt=()):
        self.files, self.page, self.corrupt = dict(files), page, set(corrupt)
        self.requests = []
        self.downloads = []
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.delay = 0.0

    def __call__(self, url, params=None, stream=False):
        with self.lock:
            self.requests.append((url, dict(params or {})))
        if url == _s2fetch._LIST_URL:
            names = sorted(n for n in self.files if n.startswith(params["prefix"]))
            names += [n + "_$folder$" for n in names[:2]]                # the placeholders the bucket has
            names.sort()
            start = int(params.get("pageToken", 0))
            chunk = names[start:start + self.page]
            items = [dict(name=n, size=str(len(self.files.get(n, b"123456"))),
                          md5Hash=md5(self.files.get(n, b"123456"))) for n in chunk]
            payload = {"items": items}
            if start + self.page < len(names):
                payload["nextPageToken"] = str(start + self.page)
            return Response(payload=payload)
        name = url[len(_s2fetch._FILE_URL):]
        with self.lock:
            self.downloads.append(name)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            data = self.files[name]
            if name in self.corrupt:
                data = b"X" * len(data)
            return Response(content=data)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def bucket(monkeypatch):
    b = FakeBucket(bucket_files())
    monkeypatch.setattr(_s2fetch, "_send", b)
    monkeypatch.setattr(_s2fetch, "_sleep", lambda s: None)
    return b


def on_disk(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != _s2fetch._DONE)


def test_download_l1c_takes_every_page_skips_folders_and_gets_only_what_level2_reads(bucket, tmp_path):
    path = _s2fetch.download_l1c(NAME, tmp_path)
    assert path == tmp_path / NAME
    want = sorted(["MTD_MSIL1C.xml", f"{GRANULE}/MTD_TL.xml"]
                  + [f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_B{b}.jp2" for b in BANDS])
    assert on_disk(path) == want
    pages = [p for u, p in bucket.requests if u == _s2fetch._LIST_URL]
    assert len(pages) > 3 and pages[0]["prefix"] == PREFIX and "pageToken" in pages[1]
    assert all("$folder$" not in n for n in bucket.downloads)
    assert (path / "MTD_MSIL1C.xml").read_bytes() == b"<product/>"
    # what the reader of level2 needs is there
    safe = _level2.Safe(path)
    assert safe.granule.name == "L1C_T23LKC_A033383_20230728T132238" and len(safe.bands) == 13


def test_download_l1c_into_the_cache_by_default(bucket, tmp_path, monkeypatch):
    monkeypatch.setenv("ZEIT_CACHE_DIR", str(tmp_path / "cache"))
    path = _s2fetch.download_l1c(NAME)
    assert path == tmp_path / "cache" / "l1c" / NAME and (path / "MTD_MSIL1C.xml").is_file()


def test_a_complete_product_is_not_asked_for_again_and_an_interrupted_one_resumes(bucket, tmp_path):
    root = tmp_path / NAME
    good = f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_B02.jp2"
    bad = f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_B03.jp2"
    (root / good).parent.mkdir(parents=True)
    (root / good).write_bytes(bucket.files[PREFIX + good])              # there with the right MD5
    (root / bad).write_bytes(b"truncated")                              # there with the wrong one
    (root / (bad + ".part")).write_bytes(b"half")
    _s2fetch.download_l1c(NAME, tmp_path)
    assert PREFIX + good not in bucket.downloads and PREFIX + bad in bucket.downloads
    assert len(bucket.downloads) == 14
    assert (root / bad).read_bytes() == bucket.files[PREFIX + bad] and not (root / (bad + ".part")).exists()
    n = len(bucket.requests)
    assert _s2fetch.download_l1c(NAME, tmp_path) == root
    assert len(bucket.requests) == n                                     # complete: not even the listing

    # no marker (an older or interrupted run), every file right: listed, not downloaded
    (root / _s2fetch._DONE).unlink()
    bucket.downloads.clear()
    _s2fetch.download_l1c(NAME, tmp_path)
    assert bucket.downloads == []


def test_a_wrong_md5_is_a_clear_error_and_leaves_no_file(bucket, tmp_path):
    victim = PREFIX + f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_B04.jp2"
    bucket.corrupt.add(victim)
    with pytest.raises(RuntimeError, match="MD5"):
        _s2fetch.download_l1c(NAME, tmp_path)
    root = tmp_path / NAME
    assert not (root / _s2fetch._DONE).exists()
    assert not (root / victim[len(PREFIX):]).exists() and not (root / (victim[len(PREFIX):] + ".part")).exists()


def test_a_product_that_the_bucket_does_not_have_is_a_value_error_naming_it(bucket, tmp_path):
    other = "S2A_MSIL1C_20200101T132239_N0500_R038_T23LKC_20200101T150149.SAFE"
    with pytest.raises(ValueError, match=other):
        _s2fetch.download_l1c(other, tmp_path)
    with pytest.raises(ValueError, match="not the name of a Sentinel-2 L1C product"):
        _s2fetch.download_l1c("something.SAFE", tmp_path)
    assert not (tmp_path / other).exists()


def test_a_product_missing_a_band_in_the_bucket_is_refused(bucket, tmp_path):
    del bucket.files[PREFIX + f"{GRANULE}/IMG_DATA/T23LKC_20230728T132239_B8A.jp2"]
    with pytest.raises(ValueError, match="13 band files"):
        _s2fetch.download_l1c(NAME, tmp_path)


def test_transient_errors_are_asked_again_with_a_wait(bucket, tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(_s2fetch, "_sleep", waits.append)
    real = bucket.__call__
    state = {"n": 0}

    def flaky(url, params=None, stream=False):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.ConnectionError("reset")
        if state["n"] == 3:
            return Response(status=503)
        return real(url, params, stream)

    monkeypatch.setattr(_s2fetch, "_send", flaky)
    path = _s2fetch.download_l1c(NAME, tmp_path)
    assert len(on_disk(path)) == 15 and len(waits) == 2 and min(waits) > 0


def test_a_server_that_never_answers_is_an_error_after_a_few_tries(bucket, tmp_path, monkeypatch):
    def down(url, params=None, stream=False):
        return Response(status=503)

    monkeypatch.setattr(_s2fetch, "_send", down)
    with pytest.raises(RuntimeError, match="failed 5 times"):
        _s2fetch.download_l1c(NAME, tmp_path)

    def gone(url, params=None, stream=False):
        return Response(status=403)

    monkeypatch.setattr(_s2fetch, "_send", gone)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        _s2fetch.download_l1c(NAME, tmp_path)


def test_n_jobs_caps_the_simultaneous_downloads_and_never_passes_eight(bucket, tmp_path):
    bucket.delay = 0.02
    _s2fetch.download_l1c(NAME, tmp_path, n_jobs=2)
    assert 1 <= bucket.peak <= 2
    assert _s2fetch._n_jobs(64) == 8 and _s2fetch._n_jobs(3) == 3
    with pytest.raises(ValueError, match="n_jobs"):
        _s2fetch.download_l1c(NAME, tmp_path, n_jobs=0)


def test_download_l1c_takes_a_row_or_a_table_of_find_l1c(bucket, tmp_path):
    import pandas as pd

    other = NAME.replace("20230728", "20230729")
    bucket.files.update({k.replace("20230728", "20230729"): v for k, v in list(bucket.files.items())})
    df = pd.DataFrame({"product": [NAME, other]})
    paths = _s2fetch.download_l1c(df, tmp_path)
    assert paths == [tmp_path / NAME, tmp_path / other]
    assert _s2fetch.download_l1c(df.iloc[0], tmp_path) == tmp_path / NAME


# --- find_l1c ---------------------------------------------------------------------------------------------------------------


def fake_stac(monkeypatch, items):
    seen = {}

    class Search:
        def items(self):
            return iter(items)

    class Client:
        def search(self, **kw):
            seen.update(kw)
            return Search()

    monkeypatch.setattr(_s2fetch, "_stac_client", lambda url: seen.setdefault("url", url) and Client())
    return seen


def item(i, day, cloud, tile="MGRS-23LKC", uri=True):
    import datetime

    props = {"eo:cloud_cover": cloud, "s2:processing_baseline": "05.09", "grid:code": tile, "platform": "sentinel-2b"}
    if uri:
        props["s2:product_uri"] = f"S2B_MSIL1C_202307{day:02d}T132239_N0509_R038_T23LKC_202307{day:02d}T150149.SAFE"
    return types.SimpleNamespace(id=i, datetime=datetime.datetime(2023, 7, day, 13, 26, 49, tzinfo=datetime.timezone.utc), properties=props)


def test_find_l1c_asks_the_tile_and_the_dates_and_sorts_by_date(monkeypatch):
    seen = fake_stac(monkeypatch, [item("b", 28, 17.3), item("a", 3, 4.0)])
    df = _s2fetch.find_l1c(tile="t23lkc", start="2023-07-01", end="2023-07-28", max_cloud=50, limit=7)
    assert list(df.columns) == ["product", "datetime", "tile", "cloud_cover", "baseline", "platform"]
    assert list(df["product"]) == [item("", 3, 0).properties["s2:product_uri"], item("", 28, 0).properties["s2:product_uri"]]
    assert df["datetime"].is_monotonic_increasing and df["tile"].iloc[0] == "23LKC" and df["baseline"].iloc[0] == "05.09"
    assert seen["url"] == "https://earth-search.aws.element84.com/v1" and seen["collections"] == ["sentinel-2-l1c"]
    assert seen["query"] == {"grid:code": {"eq": "MGRS-23LKC"}, "eo:cloud_cover": {"lte": 50.0}}
    assert seen["datetime"] == "2023-07-01T00:00:00Z/2023-07-28T23:59:59Z"      # the last day is whole
    assert seen["max_items"] == 7 and "bbox" not in seen


def test_find_l1c_by_box_and_the_argument_errors(monkeypatch):
    seen = fake_stac(monkeypatch, [])
    df = _s2fetch.find_l1c(bbox=(-47.5, -16.5, -47.0, -16.0))
    assert len(df) == 0 and list(df.columns)[0] == "product"
    assert seen["bbox"] == [-47.5, -16.5, -47.0, -16.0] and "query" not in seen and "datetime" not in seen
    with pytest.raises(ValueError, match="exactly one"):
        _s2fetch.find_l1c()
    with pytest.raises(ValueError, match="exactly one"):
        _s2fetch.find_l1c(tile="23LKC", bbox=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="MGRS"):
        _s2fetch.find_l1c(tile="nonsense")


def test_find_l1c_adds_the_safe_suffix_when_the_item_lacks_it(monkeypatch):
    it = item("a", 3, 1.0)
    it.properties["s2:product_uri"] = it.properties["s2:product_uri"][:-5]
    fake_stac(monkeypatch, [it])
    assert _s2fetch.find_l1c(tile="23LKC")["product"].iloc[0].endswith(".SAFE")


# --- download_dem -----------------------------------------------------------------------------------------------------------


def f(lon, lat):
    return np.float32(lat * 100.0 + lon)


def write_tile(path, west, north, ncols, nrows=36, size=1.0):
    """A synthetic tile of ``size`` degrees: the value of a cell is a function of the centre of the cell."""
    xres, yres = size / ncols, size / nrows
    lon = west + xres * (np.arange(ncols) + 0.5)
    lat = north - yres * (np.arange(nrows) + 0.5)
    data = (lat[:, None] * 100.0 + lon[None, :]).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(west, north, xres, yres)) as d:
        d.write(data, 1)
    return str(path)


def with_tiles(monkeypatch, paths):
    asked = []
    monkeypatch.setattr(_s2fetch, "_dem_sources", lambda bounds: asked.append(bounds) or list(paths))
    return asked


def test_two_adjacent_tiles_make_one_mosaic_in_the_geometry_of_the_source(tmp_path, monkeypatch):
    a = write_tile(tmp_path / "a.tif", -48.0, -16.0, 36)
    b = write_tile(tmp_path / "b.tif", -47.0, -16.0, 36)
    asked = with_tiles(monkeypatch, [a, b])
    out = _s2fetch.download_dem((-47.2, -16.8, -46.8, -16.6), tmp_path / "dem", n_jobs=2)
    assert out.parent == tmp_path / "dem" and out.suffix == ".tif"
    w, s, e, n = asked[0]
    assert w < -47.2 and s < -16.8 and e > -46.8 and n > -16.6 and w == pytest.approx(-47.2 - 6 / 3600)
    with rasterio.open(out) as d:
        assert d.crs.to_epsg() == 4326 and d.nodata == -32767 and d.dtypes[0] == "float32"
        assert d.bounds.left <= -47.2 - 6 / 3600 and d.bounds.right >= -46.8 + 6 / 3600
        got = d.read(1)
        lon = d.transform.c + d.transform.a * (np.arange(d.width) + 0.5)
        lat = d.transform.f + d.transform.e * (np.arange(d.height) + 0.5)
    assert d.width > 10 and np.all(got != -32767)
    # the seam at -47 is not a seam: every cell has the value of the source at its centre
    np.testing.assert_array_equal(got, (lat[:, None] * 100.0 + lon[None, :]).astype(np.float32))
    assert (lon < -47).any() and (lon > -47).any()
    assert _s2fetch.download_dem((-47.2, -16.8, -46.8, -16.6), tmp_path / "dem") == out       # reused


def test_a_missing_tile_leaves_no_data_and_no_tile_at_all_is_an_error(tmp_path, monkeypatch):
    with_tiles(monkeypatch, [write_tile(tmp_path / "a.tif", -48.0, -16.0, 36)])
    out = _s2fetch.download_dem((-48.2, -16.8, -47.8, -16.6), tmp_path / "o.tif")
    assert out == tmp_path / "o.tif"
    with rasterio.open(out) as d:
        got = d.read(1)
        lon = d.transform.c + d.transform.a * (np.arange(d.width) + 0.5)
    assert (got[:, lon < -48] == -32767).all() and (got[:, lon > -48] != -32767).all()
    with_tiles(monkeypatch, [])
    with pytest.raises(ValueError, match="topography=False"):
        _s2fetch.download_dem((10.0, -50.0, 10.2, -49.8), tmp_path / "x")
    # a tile that the search lists but whose window is empty is no tile either
    with_tiles(monkeypatch, [write_tile(tmp_path / "far.tif", 100.0, 10.0, 36)])
    with pytest.raises(ValueError, match="topography=False"):
        _s2fetch.download_dem((10.0, -50.0, 10.2, -49.8), tmp_path / "x")


def test_tiles_of_different_resolution_beyond_50_degrees_use_the_finest_cell(tmp_path, monkeypatch):
    south = write_tile(tmp_path / "s.tif", 10.0, 50.0, 36)          # 49 to 50 degrees: 36 cells a degree
    north = write_tile(tmp_path / "n.tif", 10.0, 51.0, 24)          # 50 to 51: 24 cells a degree in longitude
    with_tiles(monkeypatch, [north, south])
    out = _s2fetch.download_dem((10.2, 49.8, 10.7, 50.2), tmp_path)
    with rasterio.open(out) as d:
        assert d.transform.a == pytest.approx(1 / 36) and -d.transform.e == pytest.approx(1 / 36)
        got = d.read(1)
        lon = d.transform.c + d.transform.a * (np.arange(d.width) + 0.5)
        lat = d.transform.f + d.transform.e * (np.arange(d.height) + 0.5)
    # the windows end at a cell of their own tile, so the edge of the mosaic may have gaps; the box and its margin has none
    inside = ((lat >= 49.8) & (lat <= 50.2))[:, None] & ((lon >= 10.2) & (lon <= 10.7))[None, :]
    assert inside.sum() > 100 and np.all(got[inside] != -32767)
    for r in range(d.height):
        for c in range(d.width):
            if got[r, c] == -32767:
                continue
            if lat[r] < 50:
                want = f(lon[c], lat[r])
            else:                                                   # the value of the coarse cell that holds the centre
                lon_c = (np.floor((lon[c] - 10.0) * 24 + 1e-6) + 0.5) / 24 + 10.0       # a centre on the edge of a coarse cell (every other one at 1.5) goes to the east
                want = f(lon_c, lat[r])
            assert got[r, c] == pytest.approx(want, abs=1e-3)
    assert (lat > 50).any() and (lat < 50).any()


def test_the_antimeridian_and_bad_areas_are_refused(tmp_path):
    with pytest.raises(ValueError, match="antimeridian"):
        _s2fetch.download_dem((179.5, -17.0, -179.5, -16.0), tmp_path)
    with pytest.raises(ValueError, match="antimeridian"):
        _s2fetch.download_dem((179.5, -17.0, 180.5, -16.0), tmp_path)
    with pytest.raises(ValueError, match="invalid area"):
        _s2fetch.download_dem((10.0, 5.0, 11.0, 4.0), tmp_path)
    with pytest.raises(ValueError, match="directory"):
        _s2fetch.download_dem(str(tmp_path / "missing.SAFE"), tmp_path)


def test_the_href_of_the_tile_is_read_anonymously():
    s3 = "s3://copernicus-dem-30m/Copernicus_DSM_COG_10_S17_00_W048_00_DEM/Copernicus_DSM_COG_10_S17_00_W048_00_DEM.tif"
    assert _s2fetch._anonymous(s3) == ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S17_00_W048_00_DEM/"
                                       "Copernicus_DSM_COG_10_S17_00_W048_00_DEM.tif")
    assert _s2fetch._anonymous("https://x/y.tif") == "https://x/y.tif"


def test_dem_sources_ask_the_collection_with_the_box(monkeypatch):
    asset = types.SimpleNamespace(href="s3://copernicus-dem-30m/T/T.tif")
    seen = fake_stac(monkeypatch, [types.SimpleNamespace(assets={"data": asset})])
    assert _s2fetch._dem_sources((1.0, 2.0, 3.0, 4.0)) == ["https://copernicus-dem-30m.s3.amazonaws.com/T/T.tif"]
    assert seen["collections"] == ["cop-dem-glo-30"] and seen["bbox"] == [1.0, 2.0, 3.0, 4.0]


def write_safe(root):
    """A minimal .SAFE: the footprint is a 400 x 300 pixel grid of 10 m in UTM 23S, near 47 W and 16 S."""
    (root / "GRANULE" / "L1C_T23LKC_x" / "IMG_DATA").mkdir(parents=True)
    (root / "MTD_MSIL1C.xml").write_text("<x/>")
    (root / "GRANULE" / "L1C_T23LKC_x" / "MTD_TL.xml").write_text("<x/>")
    for b in BANDS:
        with rasterio.open(root / "GRANULE" / "L1C_T23LKC_x" / "IMG_DATA" / f"T23LKC_20230728T132239_B{b}.tif", "w", driver="GTiff",
                           height=30, width=40, count=1, dtype="uint16", crs="EPSG:32723",
                           transform=from_origin(199980.0, 8200020.0, 100.0, 100.0)) as d:
            d.write(np.zeros((30, 40), np.uint16), 1)
    return root


def test_the_area_of_a_safe_is_its_footprint_with_a_margin(tmp_path, monkeypatch):
    from rasterio.warp import transform_bounds

    safe = write_safe(tmp_path / NAME)
    asked = with_tiles(monkeypatch, [])
    with pytest.raises(ValueError, match="topography=False"):
        _s2fetch.download_dem(safe, tmp_path / "dem")
    w, s, e, n = asked[0]
    # the corners of the grid, each edge densified: no corner of the footprint is outside the box
    exact = transform_bounds("EPSG:32723", "EPSG:4326", 199980.0, 8200020.0 - 3000.0, 199980.0 + 4000.0, 8200020.0, densify_pts=101)
    assert w == pytest.approx(exact[0] - 6 / 3600) and s == pytest.approx(exact[1] - 6 / 3600)
    assert e == pytest.approx(exact[2] + 6 / 3600) and n == pytest.approx(exact[3] + 6 / 3600)
    # the name of the output comes from the product
    tile = write_tile(tmp_path / "t.tif", -48.0, -16.0, 36)
    with_tiles(monkeypatch, [tile])
    out = _s2fetch.download_dem(str(safe), tmp_path / "dem")
    assert out.name == "cop_glo30_" + NAME[:-5] + ".tif"


# --- level2 -----------------------------------------------------------------------------------------------------------------


def fake_engine(monkeypatch, calls):
    desc = dict(nrows=6, ncols=6, res=10.0, ulx=199980.0, uly=8300020.0, epsg=32723, date=(2023, 7, 28, 13, 22, 39),
                sensor="S2B", baseline="05.09")

    def arrays(safe, **kw):
        calls.append((safe, kw))
        out = {"boa": np.zeros((10, 6, 6), np.int16), "desc": desc, "aod_scene": np.arange(13) / 100.0, "aodmap": True,
               "n_targets": [1, 2, 3], "ms": 10.0}
        for k in ("qai", "dst", "vzn", "hot", "aod", "wvp"):
            out[k] = np.zeros((6, 6), np.int16)
        return out

    monkeypatch.setattr(_level2, "_level2_arrays", arrays)


def test_level2_takes_the_name_of_a_product_and_downloads_it(monkeypatch, tmp_path):
    calls, got = [], []
    fake_engine(monkeypatch, calls)
    monkeypatch.setattr(_s2fetch, "download_l1c", lambda p, *a, **k: got.append(p) or tmp_path / p)
    _level2.level2(NAME, topography=False)
    assert got == [NAME] and calls[0][0] == tmp_path / NAME
    got.clear()
    _level2.level2("S2B_MSIL1C_whatever", topography=False)                 # starts with S2
    assert got == ["S2B_MSIL1C_whatever"]
    got.clear()
    _level2.level2(str(tmp_path), topography=False)                          # an existing directory is a directory
    _level2.level2(tmp_path, topography=False)
    assert got == [] and calls[-1][0] == tmp_path


def test_level2_with_glo30_makes_the_dem_of_the_scene(monkeypatch, tmp_path):
    calls, asked = [], []
    fake_engine(monkeypatch, calls)
    monkeypatch.setattr(_s2fetch, "download_l1c", lambda p, *a, **k: tmp_path / p)
    monkeypatch.setattr(_s2fetch, "download_dem", lambda area, *a, **k: asked.append(area) or tmp_path / "dem.tif")
    _level2.level2(NAME, "glo30")
    assert asked == [tmp_path / NAME] and calls[0][1]["dem"] == tmp_path / "dem.tif"
    asked.clear()
    _level2.level2(tmp_path, "GLO30")
    assert asked == [tmp_path]
    asked.clear()
    _level2.level2(tmp_path, "dem.tif")                                      # any other string is a raster
    assert asked == [] and calls[-1][1]["dem"] == "dem.tif"
