"""zeit._level2_series.run_series: the orchestration of zeit.level2 over a series of scenes, with stand-ins for
the downloads and for level2 (no network, no real scene)."""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr

import zeit  # noqa: F401  (registers the .zeit accessor)
from zeit import _level2, _level2_series, _s2fetch
from zeit._level2_series import MARKER, JOURNAL, parse_name, run_series


def name(day, n="0509", generated=None, tile="T23LKC", platform="S2B", orbit="R138"):
    generated = generated or f"202307{day:02d}T145717"
    return f"{platform}_MSIL1C_202307{day:02d}T131239_N{n}_{orbit}_{tile}_{generated}.SAFE"


def scene_ds(product):
    seed = zlib.crc32(Path(product).name.encode())
    rng = np.random.default_rng(seed)
    x = 500000 + 10 * (np.arange(8) + 0.5)
    y = 8000000 - 10 * (np.arange(8) + 0.5)
    coords = {"y": y, "x": x}
    boa = xr.DataArray(rng.integers(0, 10000, (2, 8, 8)).astype("int16"), dims=("band", "y", "x"),
                       coords={"band": ["RED", "NIR"], **coords})
    qai = xr.DataArray(rng.integers(0, 50, (8, 8)).astype("int16"), dims=("y", "x"), coords=coords)
    dst = xr.DataArray(rng.integers(0, 5, (8, 8)).astype("int16"), dims=("y", "x"), coords=coords)
    ds = xr.Dataset({"boa": boa, "qai": qai, "dst": dst}).rio.write_crs("EPSG:32723")
    ds.attrs.update(aod_green=np.float64(0.12), sensor="MSI", n_targets=np.int64(7), arr=np.array([1.5, 2.5]))
    return ds


@pytest.fixture
def stub(monkeypatch):
    calls = []
    state = {"fail": set(), "find": None, "level2_kw": []}

    def find_l1c(**kw):
        calls.append(("find_l1c", kw))
        return state["find"]

    def download_l1c(product, dest=None, n_jobs=4):
        calls.append(("download_l1c", product, n_jobs))
        root = Path(dest) / product
        root.mkdir(parents=True, exist_ok=True)
        (root / "MTD_MSIL1C.xml").write_text("x")
        return root

    def download_dem(area, dest=None, n_jobs=4):
        calls.append(("download_dem", Path(area).name))
        Path(dest).mkdir(parents=True, exist_ok=True)
        path = Path(dest) / "dem.tif"
        path.write_bytes(b"dem")
        return path

    def level2(safe, dem=None, *, brdf=True, topography=True, aod=True, erase_cloud=False, n_jobs=8):
        calls.append(("level2", Path(safe).name, None if dem is None else str(dem), n_jobs,
                      dict(brdf=brdf, erase_cloud=erase_cloud)))
        if Path(safe).name in state["fail"]:
            raise RuntimeError("boom")
        return scene_ds(safe)

    monkeypatch.setattr(_s2fetch, "find_l1c", find_l1c)
    monkeypatch.setattr(_s2fetch, "download_l1c", download_l1c)
    monkeypatch.setattr(_s2fetch, "download_dem", download_dem)
    monkeypatch.setattr(_level2, "level2", level2)
    stub = type("Stub", (), {})()
    stub.calls, stub.state = calls, state
    stub.of = lambda kind: [c for c in calls if c[0] == kind]
    return stub


def frame(names):
    return pd.DataFrame({"product": names,
                         "datetime": [pd.Timestamp(parse_name(n)["datetime"]) for n in names],
                         "tile": "23LKC", "cloud_cover": 5.0, "baseline": "05.09", "platform": "sentinel-2b"})


def read_tifs(folder):
    out = {}
    for tif in sorted(Path(folder).glob("*.tif")):
        with rasterio.open(tif) as src:
            out[tif.name] = (src.read(), src.crs, src.transform)
    return out


NAMES = [name(15), name(5), name(25)]


def test_parser():
    info = parse_name("C:/data/S2B_MSIL1C_20230715T131239_N0509_R138_T23LKC_20230715T145717.SAFE")
    assert info["product"] == "S2B_MSIL1C_20230715T131239_N0509_R138_T23LKC_20230715T145717.SAFE"
    assert (info["platform"], info["baseline"], info["n"], info["orbit"], info["tile"]) == ("S2B", "05.09", 509, "138", "23LKC")
    assert info["datetime"] == pd.Timestamp("2023-07-15 13:12:39", tz="UTC")
    assert info["generated"] == "20230715T145717"
    assert parse_name(info["product"][:-5])["product"] == info["product"]
    with pytest.raises(ValueError, match="not the name"):
        parse_name("not_a_product.SAFE")


def test_steps_in_date_order_dem_once_and_options_forwarded(stub, tmp_path):
    stub.state["find"] = frame(NAMES)
    table = run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, level2_kw={"brdf": False}, n_jobs=6, verbose=False)
    assert list(table["status"]) == ["done"] * 3
    assert list(table["product"]) == sorted(NAMES)
    kinds = [c[0] for c in stub.calls]
    assert kinds == ["find_l1c", "download_l1c", "download_dem", "level2", "download_l1c", "level2", "download_l1c", "level2"]
    assert [c[1] for c in stub.of("download_l1c")] == sorted(NAMES)
    assert stub.of("find_l1c")[0][1]["tile"] == "23LKC"
    assert len(stub.of("download_dem")) == 1
    dem = str(tmp_path / "dem" / "dem.tif")
    for c in stub.of("level2"):
        assert c[2] == dem and c[3] == 6 and c[4]["brdf"] is False
    assert all(c[2] == 4 for c in stub.of("download_l1c"))
    journal = json.loads((tmp_path / JOURNAL).read_text())
    assert journal["dem"] == dem and len(journal["done"]) == 3 and journal["failed"] == {}
    # the downloaded .SAFE are deleted
    assert not any(p.name.endswith(".SAFE") for p in (tmp_path / ".safe").iterdir())


def test_files_equal_level2_then_save(stub, tmp_path):
    stub.state["find"] = frame(NAMES[:1])
    table = run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path / "out", verbose=False)
    folder = Path(table["path"][0])
    assert folder == tmp_path / "out" / NAMES[0][:-5]
    scene_ds(NAMES[0]).zeit.save(tmp_path / "hand")
    got, want = read_tifs(folder), read_tifs(tmp_path / "hand")
    assert set(got) == set(want) == {"boa.tif", "qai.tif", "dst.tif"}
    for key in want:
        np.testing.assert_array_equal(got[key][0], want[key][0])
        assert got[key][1] == want[key][1] and got[key][2] == want[key][2]
    marker = json.loads((folder / MARKER).read_text())
    assert marker["attrs"]["aod_green"] == 0.12 and marker["attrs"]["n_targets"] == 7
    assert marker["attrs"]["arr"] == [1.5, 2.5] and marker["attrs"]["sensor"] == "MSI"
    assert marker["product"] == NAMES[0] and marker["tile"] == "23LKC" and marker["baseline"] == "05.09"
    assert marker["zeit_version"] and marker["params_digest"]


def test_resume_after_a_failure(stub, tmp_path):
    stub.state["find"] = frame(NAMES)
    middle = sorted(NAMES)[1]
    stub.state["fail"] = {middle}
    with pytest.warns(UserWarning, match="1 of 3 scenes failed"):
        table = run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, verbose=False)
    assert list(table["status"]) == ["done", "failed", "done"]
    assert "boom" in table["error"][1] and table["path"][1] is None
    assert not list(tmp_path.glob(".part-*"))
    journal = json.loads((tmp_path / JOURNAL).read_text())
    assert "boom" in journal["failed"][middle[:-5]]["error"] and journal["failed"][middle[:-5]]["at"]
    stub.calls.clear()
    stub.state["fail"] = set()
    table = run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, verbose=False)
    assert list(table["status"]) == ["skipped", "done", "skipped"]
    assert [c[1] for c in stub.of("level2")] == [middle]
    assert stub.of("download_dem") == []
    assert json.loads((tmp_path / JOURNAL).read_text())["failed"] == {}


def test_leftover_folder_without_marker_is_redone(stub, tmp_path):
    stub.state["find"] = frame(NAMES[:1])
    leftover = tmp_path / NAMES[0][:-5]
    leftover.mkdir()
    (leftover / "junk.tif").write_text("half")
    table = run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, verbose=False)
    assert table["status"][0] == "done"
    assert not (leftover / "junk.tif").exists() and (leftover / MARKER).is_file()


def test_other_options_raise_before_any_work(stub, tmp_path):
    stub.state["find"] = frame(NAMES)
    run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, verbose=False)
    stub.calls.clear()
    with pytest.raises(FileExistsError, match="another out_dir"):
        run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, level2_kw={"brdf": False}, verbose=False)
    assert stub.of("level2") == [] and stub.of("download_l1c") == []


def test_duplicates_keep_the_newest_processing(stub, tmp_path):
    old, new = name(5, "0500"), name(5, "0509")
    same_a, same_b = name(15, "0509", "20230715T145717"), name(15, "0509", "20230716T090000")
    stub.state["find"] = frame([old, new, same_a, same_b])
    table = run_series("23LKC", "2023-07-01", "2023-07-31", tmp_path, verbose=False)
    status = dict(zip(table["product"], table["status"]))
    assert status == {old: "duplicate", new: "done", same_a: "duplicate", same_b: "done"}
    assert [c[1] for c in stub.of("level2")] == [new, same_b]
    assert table.loc[table["status"] == "duplicate", "path"].isna().all()


def test_products_as_names_dataframe_and_local_safe(stub, tmp_path):
    # names: no tile/start/end needed
    table = run_series(out_dir=tmp_path / "a", products=NAMES, verbose=False)
    assert list(table["status"]) == ["done"] * 3 and stub.of("find_l1c") == []
    assert list(table["baseline"]) == ["05.09"] * 3 and list(table["tile"]) == ["23LKC"] * 3
    # a DataFrame
    table = run_series(out_dir=tmp_path / "b", products=frame(NAMES[:2]), verbose=False)
    assert list(table["status"]) == ["done"] * 2 and list(table["platform"]) == ["sentinel-2b"] * 2
    # a local .SAFE is read in place and kept; downloaded ones are deleted unless keep_safe
    local = tmp_path / "mine" / NAMES[0]
    local.mkdir(parents=True)
    stub.calls.clear()
    table = run_series(out_dir=tmp_path / "c", products=[local, NAMES[1]], keep_safe=False, verbose=False)
    assert list(table["status"]) == ["done"] * 2
    assert [c[1] for c in stub.of("download_l1c")] == [NAMES[1]]
    assert local.is_dir()
    assert not (tmp_path / "c" / ".safe" / NAMES[1]).exists()
    run_series(out_dir=tmp_path / "d", products=[NAMES[0]], keep_safe=True, safe_dir=tmp_path / "keep", verbose=False)
    assert (tmp_path / "keep" / NAMES[0]).is_dir()


def test_own_dem_path_and_no_topography(stub, tmp_path):
    dem = tmp_path / "my.tif"
    dem.write_bytes(b"x")
    run_series(out_dir=tmp_path / "a", products=NAMES[:1], dem=dem, verbose=False)
    assert stub.of("download_dem") == [] and stub.of("level2")[0][2] == str(dem)
    run_series(out_dir=tmp_path / "b", products=NAMES[:1], dem=None, level2_kw={"topography": False}, verbose=False)
    assert stub.of("level2")[1][2] is None


def test_up_front_errors(stub, tmp_path):
    with pytest.raises(ValueError, match="topography"):
        run_series(out_dir=tmp_path, products=NAMES, dem=None)
    with pytest.raises(TypeError, match="nonsense"):
        run_series(out_dir=tmp_path, products=NAMES, level2_kw={"nonsense": 1})
    with pytest.raises(TypeError, match="n_jobs"):
        run_series(out_dir=tmp_path, products=NAMES, level2_kw={"n_jobs": 1})
    with pytest.raises(ValueError, match="several tiles"):
        run_series(out_dir=tmp_path, products=[name(5), name(6, tile="T23LLC")])
    with pytest.raises(ValueError, match="tile, start and end"):
        run_series("23LKC", out_dir=tmp_path)
    with pytest.raises(ValueError, match="not the name"):
        run_series(out_dir=tmp_path, products=["nope.SAFE"])
    assert stub.calls == []


def test_ctrl_c_propagates_and_leaves_no_part(stub, tmp_path, monkeypatch):
    def interrupt(safe, dem=None, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(_level2, "level2", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_series(out_dir=tmp_path, products=NAMES, level2_kw={}, verbose=False)
    assert not list(tmp_path.glob(".part-*"))
    assert (tmp_path / JOURNAL).is_file()


def test_disk_warning_after_the_first_scene(stub, tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12, 10))
    with pytest.warns(UserWarning, match="GiB are free"):
        run_series(out_dir=tmp_path, products=NAMES, verbose=False)


def test_verbose_lines(stub, tmp_path, capsys):
    stub.state["fail"] = {sorted(NAMES)[1]}
    with pytest.warns(UserWarning):
        run_series(out_dir=tmp_path, products=NAMES)
    stub.state["fail"] = set()
    run_series(out_dir=tmp_path, products=NAMES)
    out = capsys.readouterr().out
    assert "[1/3]" in out and "done in" in out and "failed: RuntimeError: boom" in out and "skipped" in out
