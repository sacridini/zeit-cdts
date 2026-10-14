"""Many tiles on this computer and on helpers (zeit.run_tiles, zeit worker): the options,
the plan and its resumption, the handshake and the pairing, and whole runs between a
coordinator and a helper on this machine."""

import io
import os
import tarfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit._tiles import _coordinator, _group, _net, _pairing, _tasks, _worker


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEIT_CONFIG_DIR", str(tmp_path / "config"))


def write_tile(path, seed, h, w):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=12, freq="YS")
    trend = np.linspace(0.2, 0.8, 12)[:, None, None] * (1 + seed)
    values = (trend + rng.normal(0, 0.05, (12, h, w))).astype("float32")
    tr = from_origin(500000.0 + 1000 * seed, 9000000.0, 30.0, 30.0)
    da = xr.DataArray(values, dims=("time", "y", "x"),
                      coords={"time": dates, "y": tr.f - 30.0 * (np.arange(h) + 0.5),
                              "x": tr.c + 30.0 * (np.arange(w) + 0.5)}).rio.write_crs("EPSG:32722")
    zeit.save_raster(da, path)


@pytest.fixture
def tiles(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    for i, (h, w) in enumerate([(10, 12), (14, 9)]):
        write_tile(folder / f"tile_{i}.tif", i, h, w)
    return folder


def read(path):
    with rasterio.open(path) as src:
        return src.read()


# ---------------------------------------------------------------------------- options

def test_options_become_the_commands_flags():
    args, shared = _tasks.forward_args("landtrendr", {"max_segments": 4, "allow_fast_recovery": True,
                                                      "event_type": "loss", "start_year": None})
    assert args == ["--max-segments", "4", "--allow-fast-recovery"]     # defaults left out
    assert shared == []
    args, _ = _tasks.forward_args("bandpass-adjust", {"sensor": ["sentinel-2a", "sentinel-2b"]})
    assert args == ["--sensor", "sentinel-2a", "sentinel-2b"]


def test_bad_options_fail_before_any_tile(tmp_path):
    with pytest.raises(TypeError, match="no option bogus"):
        _tasks.option_values("landtrendr", {"bogus": 1})
    with pytest.raises(ValueError, match="one raster"):
        _tasks.forward_args("phenology", {"weights": "w.tif"})
    with pytest.raises(ValueError, match="monitor-start-time"):
        _tasks.forward_args("bfast-monitor", {})
    with pytest.raises(ValueError, match="does not run per tile"):
        _tasks.command_name("embeddings")


def test_shared_files_travel_apart(tmp_path):
    patterns = tmp_path / "patterns.csv"
    patterns.write_text("pattern,date,ndvi\nforest,2020-01-01,0.8\n")
    args, shared = _tasks.forward_args("twdtw", {"patterns": str(patterns), "steepness": 0.2})
    assert args == ["--steepness", "0.2"]
    assert shared == [{"option": "--patterns", "path": str(patterns.resolve())}]


def test_a_coordinator_cannot_make_a_helper_start_a_run_or_read_files():
    for args in (["--tiles"], ["--helpers"], ["--no-local"], ["--local-cores", "2"]):
        with pytest.raises(ValueError, match="tile run"):
            _tasks.check_args("mann-kendall", args, received=True)
    with pytest.raises(ValueError, match="name files"):
        _tasks.check_args("twdtw", ["--patterns", "/etc/passwd"], received=True)
    with pytest.raises(ValueError, match="does not run per tile"):
        _tasks.check_args("embeddings", [], received=True)
    _tasks.check_args("mann-kendall", ["--alpha", "0.1"], received=True)


def test_shared_paths_are_mapped():
    maps = _worker.parse_maps(["\\\\nas\\share=/mnt/nas"])
    found = _worker.map_path("\\\\nas\\share\\tiles\\a.tif", maps)
    assert found[0] == os.path.join("/mnt/nas", "tiles", "a.tif")
    assert found[-1] == "\\\\nas\\share\\tiles\\a.tif"
    assert _worker.map_path("\\\\nas\\shared\\a.tif", maps) == ["\\\\nas\\shared\\a.tif"]   # not a prefix


def test_an_archive_cannot_write_outside_its_folder(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        data = b"x"
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    with pytest.raises(ValueError, match="refusing"):
        _tasks.unpack(buffer, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


# ---------------------------------------------------------------------------- this computer

def test_a_folder_of_tiles_runs_here_and_resumes(tiles, tmp_path):
    out = tmp_path / "out"
    result = zeit.run_tiles("mann_kendall", tiles, out, local_cores=2, verbose=False)
    assert result.ok and sorted(result.done) == ["tile_0", "tile_1"]
    for stem, shape in (("tile_0", (10, 12)), ("tile_1", (14, 9))):
        maps = read(out / stem / "mann_kendall.tif")
        assert maps.shape[1:] == shape and np.isfinite(maps).any()
        assert (out / stem / _tasks.MARKER).exists()
    assert not (out / _tasks.PARTIAL).exists()

    again = zeit.run_tiles("mann_kendall", tiles, out, verbose=False)
    assert again.done == [] and sorted(again.skipped) == ["tile_0", "tile_1"]
    with pytest.raises(FileExistsError, match="other options"):
        zeit.run_tiles("mann_kendall", tiles, out, alpha=0.1, verbose=False)


def test_a_broken_tile_is_reported_and_the_others_finish(tiles, tmp_path):
    (tiles / "broken.tif").write_bytes(b"not a raster")
    result = zeit.run_tiles("mann_kendall", tiles, tmp_path / "out", local_cores=1, verbose=False)
    assert sorted(result.done) == ["tile_0", "tile_1"]
    assert list(result.failed) == ["broken"] and not result.ok


def test_the_cli_runs_tiles(tiles, tmp_path):
    from zeit import cli

    with pytest.raises(SystemExit) as exit_:
        cli.main(["mann-kendall", str(tiles), str(tmp_path / "out"), "--tiles", "--jobs", "1"])
    assert exit_.value.code == 0
    assert (tmp_path / "out" / "tile_1" / "mann_kendall.tif").exists()
    with pytest.raises(SystemExit, match="needs --helpers"):
        cli.main(["mann-kendall", str(tiles), str(tmp_path / "out2"), "--no-local"])


# ---------------------------------------------------------------------------- helpers

def run_with_helper(tiles, out, cache, **coordinator):
    _group.ensure_secret()
    plan = _tasks.make_plan("mann-kendall", tiles, out, {})
    return run_plan_with_helper(plan, cache, **coordinator)


def run_plan_with_helper(plan, cache, **coordinator):
    lines = []
    coord = _coordinator.Coordinator(plan, local=False, helpers=True, port=0, discovery=False, host="127.0.0.1",
                                     out=lines.append, **coordinator).start()
    helper_lines = []
    helper = threading.Thread(target=_worker.run_worker, kwargs=dict(
        cores=1, connect=f"127.0.0.1:{coord.port}", once=True, cache_dir=str(cache), out=helper_lines.append))
    helper.start()
    result = coord.wait()
    helper.join(timeout=60)
    assert not helper.is_alive()
    remote = [m for m in coord._machines if not m.local]
    return result, remote, lines, helper_lines


@pytest.mark.parametrize("seen_by_helper", [False, True], ids=["over the network", "from shared storage"])
def test_a_helper_processes_every_tile(tiles, tmp_path, monkeypatch, seen_by_helper):
    if not seen_by_helper:   # the helper does not see the coordinator's folder: the tiles are sent
        monkeypatch.setattr(_tasks, "network_path", lambda path: "/nowhere/" + path.name)
    reference = zeit.run_tiles("mann_kendall", tiles, tmp_path / "here", verbose=False)
    result, remote, lines, helper_lines = run_with_helper(tiles, tmp_path / "out", tmp_path / "cache")
    assert result.ok and sorted(result.done) == ["tile_0", "tile_1"]
    assert len(remote) == 1 and remote[0].done == 2 and remote[0].name in result.machines
    assert remote[0].via == {"shared" if seen_by_helper else "network": 2}
    assert remote[0].placed == (2 if seen_by_helper else 0)     # results written straight into the folder
    for stem in reference.done:
        np.testing.assert_array_equal(read(tmp_path / "out" / stem / "mann_kendall.tif"),
                                      read(tmp_path / "here" / stem / "mann_kendall.tif"))
    assert any("joined" in line for line in lines)
    assert len(list((tmp_path / "cache").glob("*.tif"))) == 2


def test_the_cache_spares_sending_a_tile_again(tiles, tmp_path, monkeypatch):
    monkeypatch.setattr(_tasks, "network_path", lambda path: "/nowhere/" + path.name)
    run_with_helper(tiles, tmp_path / "a", tmp_path / "cache")
    _, remote, _, _ = run_with_helper(tiles, tmp_path / "b", tmp_path / "cache")
    assert remote[0].via == {"cache": 2}


def test_another_group_or_version_is_refused(tiles, tmp_path):
    secret = _group.ensure_secret()
    plan = _tasks.make_plan("mann-kendall", tiles, tmp_path / "out", {})
    coord = _coordinator.Coordinator(plan, local=False, helpers=True, port=0, discovery=False, host="127.0.0.1",
                                     out=lambda text: None).start()
    try:
        stranger = _net.Session("127.0.0.1", coord.port)
        with pytest.raises(_net.ProtocolError, match="not in this group"):
            stranger.handshake(lambda salt: b"x" * 32, _worker.info(1))
        stranger.close()
        old = _net.Session("127.0.0.1", coord.port)
        with pytest.raises(_net.ProtocolError, match="pip install zeit-cdts"):
            old.handshake(lambda salt: secret, {**_worker.info(1), "version": "0.0.1"})
        old.close()
        edited = _net.Session("127.0.0.1", coord.port)
        with pytest.raises(_net.ProtocolError, match="other code"):
            edited.handshake(lambda salt: secret, {**_worker.info(1), "code": "0" * 16})
        edited.close()
        status, reply = _net.Session("127.0.0.1", coord.port).call("POST", "/next", {})
        assert status == 401
    finally:
        coord.close()


def test_pairing_hands_over_the_groups_secret(tmp_path):
    lines = []
    invitation = _pairing.Invitation(minutes=1, port=0, discovery=False, host="127.0.0.1", out=lines.append)
    try:
        with pytest.raises(_net.ProtocolError, match="wrong invitation code"):
            _pairing.join(_group.new_code(), connect=f"127.0.0.1:{invitation.port}", out=lambda text: None)
        secret = _pairing.join(invitation.code.lower(), connect=f"127.0.0.1:{invitation.port}",
                               out=lambda text: None)
    finally:
        invitation.close()
    assert secret == _group.load_secret()       # the inviter's own group
    assert len(invitation.joined) == 1 and any("joined the group" in line for line in lines)
    with pytest.raises(ValueError, match="not an invitation code"):
        _group.normalize_code("ABC")


def test_a_straggler_is_also_given_to_a_faster_machine(tmp_path):
    tile = _tasks.Tile(0, tmp_path / "a.tif", "a", 10 ** 9, 0)
    plan = _tasks.Plan("mann-kendall", [], [], tmp_path, [tile])
    coord = _coordinator.Coordinator(plan, local=False, helpers=True, out=lambda text: None)
    slow = _coordinator.Machine("slow", False, 2, compute_rate=1e6, fetch_rate=1e8)      # ~1000 s
    fast = _coordinator.Machine("fast", False, 16, compute_rate=1e8, fetch_rate=1e8)     # ~20 s
    coord._machines += [slow, fast]
    kind, first = coord._next(slow)
    assert kind == "task"
    kind, second = coord._next(fast)
    assert kind == "task" and second.task is first.task
    assert coord._next(slow)[0] == "wait"              # never a third copy
    coord._lose(first)
    assert first.task.state == "running"               # the fast copy still runs


def test_helpers_find_a_coordinator_of_their_group():
    gid = _group.group_id(_group.ensure_secret())
    asked = []

    def answer(message, addr):
        asked.append(message)
        if message.get("kind") == "helper" and message.get("group") == gid:
            return {"kind": "coordinator", "group": gid, "port": 1234}
        return None

    responder = _net.Responder(answer, port=0, host="127.0.0.1").start()
    port = responder._sock.getsockname()[1]
    try:
        found = _net.find({"kind": "helper", "group": gid}, lambda r: r.get("kind") == "coordinator",
                          timeout=10, port=port, targets=["127.0.0.1"])
        assert found == ("127.0.0.1", {"zeit": _group.PROTOCOL, "kind": "coordinator", "group": gid, "port": 1234})
        other = _net.find({"kind": "helper", "group": "another group"}, lambda r: True, timeout=1,
                          port=port, targets=["127.0.0.1"])
        assert other is None
    finally:
        responder.close()
    assert asked and all(m["zeit"] == _group.PROTOCOL for m in asked)


def test_a_broken_transfer_is_not_a_failure_of_the_tile(tmp_path):
    tile = _tasks.Tile(0, tmp_path / "a.tif", "a", 10 ** 6, 0)
    plan = _tasks.Plan("mann-kendall", [], [], tmp_path, [tile])
    coord = _coordinator.Coordinator(plan, local=False, helpers=True, out=lambda text: None)
    helper = _coordinator.Machine("helper", False, 4)
    coord._machines.append(helper)
    for _ in range(_coordinator.MAX_TRANSFER_ERRORS):
        kind, attempt = coord._next(helper)
        coord._lose(attempt, transfer_error="connection reset")
        assert attempt.task.state == "pending" and attempt.task.failures == 0
    kind, attempt = coord._next(helper)
    coord._lose(attempt, transfer_error="connection reset")     # one too many: now it counts
    assert attempt.task.failures == 1


def test_the_last_tile_stays_here_when_the_helper_is_still_busy(tmp_path):
    tiles = [_tasks.Tile(i, tmp_path / f"{i}.tif", str(i), 10 ** 9, 0) for i in range(2)]
    plan = _tasks.Plan("mann-kendall", [], [], tmp_path, tiles)
    coord = _coordinator.Coordinator(plan, local=True, helpers=True, out=lambda text: None)
    coord.local_machine.compute_rate = 1e7                                 # 100 s a tile
    helper = _coordinator.Machine("helper", False, 4, compute_rate=2e7, fetch_rate=1e9)   # 51 s a tile
    coord._machines.append(helper)
    assert coord._next(helper)[0] == "task"            # the first tile: the helper is busy for ~51 s
    assert coord._next(helper)[0] == "wait"            # ~102 s there, 100 s here: it stays here
    assert coord._next(coord.local_machine)[0] == "task"


# ---------------------------------------------------------------------------- transfers and recipes

def test_an_uncompressed_tile_travels_compressed(tmp_path, monkeypatch):
    monkeypatch.setattr(_tasks, "network_path", lambda path: "/nowhere/" + path.name)
    folder = tmp_path / "raw"
    folder.mkdir()
    write_tile(tmp_path / "packed.tif", 3, 40, 30)
    cube = zeit.load_raster(tmp_path / "packed.tif")
    zeit.save_raster(cube, folder / "raw.tif", compress=None)
    assert not _tasks.is_compressed(folder / "raw.tif")
    reference = zeit.run_tiles("mann_kendall", folder, tmp_path / "here", verbose=False)
    result, remote, lines, _ = run_with_helper(folder, tmp_path / "out", tmp_path / "cache")
    assert result.ok and remote[0].via == {"network": 1}
    assert any("compressed for the network" in line for line in lines)
    np.testing.assert_array_equal(read(tmp_path / "out" / "raw" / "mann_kendall.tif"),
                                  read(tmp_path / "here" / "raw" / "mann_kendall.tif"))
    received = next((tmp_path / "cache").glob("*.tif"))
    assert _tasks.is_compressed(received) and received.stat().st_size < (folder / "raw.tif").stat().st_size
    xr.testing.assert_equal(zeit.load_raster(received).rename(None),         # values, dates, grid
                            zeit.load_raster(folder / "raw.tif").rename(None))


@pytest.fixture
def web(tiles):
    """The tiles on a web server."""
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(tiles)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_tiles_from_urls_are_fetched_by_whoever_processes_them(tiles, web, tmp_path):
    sources = zeit.tile_urls([f"{web}/tile_0.tif", f"{web}/tile_1.tif"])
    assert [name for name, _ in sources] == ["tile_0", "tile_1"]
    reference = zeit.run_tiles("mann_kendall", tiles, tmp_path / "files", verbose=False)

    here = zeit.run_tiles("mann_kendall", sources, tmp_path / "here", verbose=False)
    assert sorted(here.done) == ["tile_0", "tile_1"]
    _group.ensure_secret()
    plan = _tasks.make_plan("mann-kendall", sources, tmp_path / "out", {})
    result, remote, lines, _ = run_plan_with_helper(plan, tmp_path / "cache")
    assert result.ok and remote[0].via == {"built": 2}
    assert any("shares its internet link" in line for line in lines)        # a helper on this network
    for stem in reference.done:
        expected = read(tmp_path / "files" / stem / "mann_kendall.tif")
        np.testing.assert_array_equal(read(tmp_path / "here" / stem / "mann_kendall.tif"), expected)
        np.testing.assert_array_equal(read(tmp_path / "out" / stem / "mann_kendall.tif"), expected)

    again = zeit.run_tiles("mann_kendall", sources, tmp_path / "out", verbose=False)
    assert sorted(again.skipped) == ["tile_0", "tile_1"]
    other = zeit.tile_urls({"tile_0": f"{web}/tile_1.tif"})                 # another source, same name
    with pytest.raises(FileExistsError):
        zeit.run_tiles("mann_kendall", other, tmp_path / "out", verbose=False)


def test_a_tile_that_does_not_build_is_reported(web, tmp_path):
    sources = zeit.tile_urls({"missing": f"{web}/missing.tif"})
    result = zeit.run_tiles("mann_kendall", sources, tmp_path / "out", verbose=False)
    assert list(result.failed) == ["missing"] and "could not build" in result.failed["missing"]


def test_recipes_are_data_a_helper_checks():
    from zeit._tiles import _sources

    for bad in ({"kind": "raster", "url": "file:///etc/passwd"}, {"kind": "raster", "url": "C:/secret.tif"},
                {"kind": "python", "code": "print(1)"},
                {"kind": "raster", "url": "https://example.org/a.tif", "extra": 1}):
        with pytest.raises(ValueError):
            _sources.check_recipe(bad)
    stac = zeit.stac_tiles(["23KPQ"], bands=["nir08", "swir22"], index="NBR", start_year=2000, end_year=2002)
    (name, recipe), = list(stac)
    assert name == "23KPQ" and recipe["kind"] == "stac_annual" and len(recipe["bbox"]) == 4
    assert recipe["dtype"] == "int16" and recipe["scale"] == 10000.0
    assert recipe["options"]["epsg"] == 32723                      # 23KPQ: its own UTM zone, south
    with pytest.raises(ValueError):
        _sources.check_recipe({**recipe, "options": {**recipe["options"], "access_token": "x"}})
    with pytest.raises(TypeError, match="no option"):
        zeit.stac_tiles(["23KPQ"], bands=["nir08"], vector_path="x.gpkg")
    with pytest.raises(ValueError, match="bands"):
        zeit.stac_tiles(["23KPQ"])
    gee = zeit.gee_tiles(["217/076", (-43.5, -23.0, -43.0, -22.5)], start_year=1990, end_year=2000)
    assert [name for name, _ in gee] == ["217_076", "tile_1"]
    assert dict(gee)["217_076"]["roi"] == "217/076"


def test_an_index_is_written_compact():
    from zeit._tiles import _sources

    nbr = xr.DataArray(np.array([[0.51234, -0.2], [np.nan, 1.0]], dtype="float32"), dims=("y", "x"))
    data, nodata = _sources._compact(nbr, 10000.0, "int16")
    assert data.dtype == np.int16 and nodata == -32768
    assert data.values.tolist() == [[5123, -2000], [-32768, 10000]]
