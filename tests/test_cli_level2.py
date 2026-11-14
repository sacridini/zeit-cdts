"""``zeit level2``: one scene or a series, with zeit.level2 and run_series replaced by recorders."""

import sys

import pandas as pd
import pytest

import zeit
from zeit import _level2_series, cli


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zeit", *map(str, argv)])
    cli.main()


class Saved:
    """What level2 returns: a Dataset whose .zeit.save is recorded."""

    def __init__(self, calls):
        self.calls = calls

    @property
    def zeit(self):
        return self

    def save(self, path):
        self.calls.append(("save", str(path)))


@pytest.fixture
def scene_calls(monkeypatch):
    calls = []

    def fake(safe, dem=None, **kw):
        calls.append(("level2", safe, dem, kw))
        return Saved(calls)

    monkeypatch.setattr(zeit, "level2", fake)
    return calls


def table(*statuses):
    return pd.DataFrame({"product": [f"P{i}.SAFE" for i in range(len(statuses))], "status": list(statuses)})


@pytest.fixture
def series_calls(monkeypatch):
    calls = []
    result = {"table": table("done", "done")}

    def fake(tile, start, end, out_dir, **kw):
        calls.append((tile, start, end, str(out_dir), kw))
        return result["table"]

    monkeypatch.setattr(_level2_series, "run_series", fake)
    return calls, result


def test_scene_defaults(monkeypatch, tmp_path, scene_calls):
    run(monkeypatch, "level2", "S2B_X.SAFE", "--out", tmp_path / "o")
    assert scene_calls[0] == ("level2", "S2B_X.SAFE", "glo30", {"n_jobs": 8, "brdf": True, "topography": True})
    assert scene_calls[1] == ("save", str(tmp_path / "o"))


def test_scene_options(monkeypatch, tmp_path, scene_calls):
    run(monkeypatch, "level2", "scene.SAFE", "--out", tmp_path, "--dem", "dem.tif", "--merge", "improphe", "--no-brdf",
        "--no-topography", "--threads", 3)
    _, safe, dem, kw = scene_calls[0]
    assert (safe, dem) == ("scene.SAFE", "dem.tif")
    assert kw == {"n_jobs": 3, "brdf": False, "topography": False, "merge": "improphe"}


def test_scene_without_dem(monkeypatch, tmp_path, scene_calls):
    run(monkeypatch, "level2", "scene.SAFE", "--out", tmp_path, "--dem", "none", "--no-topography")
    assert scene_calls[0][2] is None


def test_scene_error_is_printed(monkeypatch, tmp_path, capsys):
    def fail(safe, **kw):
        raise ValueError("topography=True needs a DEM")

    monkeypatch.setattr(zeit, "level2", fail)
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, "level2", "scene.SAFE", "--out", tmp_path, "--dem", "none")
    assert e.value.code == 1
    assert "Error: topography=True needs a DEM" in capsys.readouterr().out


def test_series(monkeypatch, tmp_path, series_calls, capsys):
    calls, _ = series_calls
    run(monkeypatch, "level2", "--tile", "23LKC", "--start", "2023-07-01", "--end", "2023-07-31", "--out", tmp_path,
        "--max-cloud", 80, "--merge", "regression", "--keep-safe", "--no-brdf", "--threads", 4)
    assert calls == [("23LKC", "2023-07-01", "2023-07-31", str(tmp_path),
                      dict(max_cloud=80.0, dem="glo30", level2_kw={"brdf": False, "topography": True, "merge": "regression"},
                           keep_safe=True, n_jobs=4))]
    out = capsys.readouterr().out
    assert "2 done, 0 skipped, 0 failed, 0 duplicate" in out
    assert "level2_series.json" in out


def test_series_defaults_and_no_dem(monkeypatch, tmp_path, series_calls):
    calls, _ = series_calls
    run(monkeypatch, "level2", "--tile", "23LKC", "--start", "2023-07-01", "--end", "2023-07-31", "--out", tmp_path,
        "--dem", "none", "--no-topography")
    kw = calls[0][4]
    assert kw["dem"] is None and kw["max_cloud"] == 100.0 and kw["keep_safe"] is False and kw["n_jobs"] == 8


def test_series_failed_scene_exits_nonzero(monkeypatch, tmp_path, series_calls, capsys):
    calls, result = series_calls
    result["table"] = table("done", "failed", "duplicate")
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, "level2", "--tile", "23LKC", "--start", "2023-07-01", "--end", "2023-07-31", "--out", tmp_path)
    assert e.value.code == 1
    assert "failed: P1.SAFE" in capsys.readouterr().out


def test_series_nothing_done_exits_nonzero(monkeypatch, tmp_path, series_calls):
    calls, result = series_calls
    result["table"] = table()
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, "level2", "--tile", "23LKC", "--start", "2023-07-01", "--end", "2023-07-31", "--out", tmp_path)
    assert e.value.code == 1


def test_series_skipped_only_is_ok(monkeypatch, tmp_path, series_calls):
    calls, result = series_calls
    result["table"] = table("skipped", "skipped")
    run(monkeypatch, "level2", "--tile", "23LKC", "--start", "2023-07-01", "--end", "2023-07-31", "--out", tmp_path)


@pytest.mark.parametrize("argv", [
    [],                                                                       # neither a scene nor a tile
    ["scene.SAFE", "--tile", "23LKC", "--start", "2023-07-01", "--end", "2023-07-31"],   # both
    ["--tile", "23LKC", "--end", "2023-07-31"],                               # no --start
    ["--tile", "23LKC", "--start", "2023-07-01"],                             # no --end
    ["scene.SAFE", "--start", "2023-07-01"],                                  # series option on a scene
    ["scene.SAFE", "--keep-safe"],
])
def test_wrong_combinations(monkeypatch, tmp_path, scene_calls, series_calls, argv):
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, "level2", *argv, "--out", tmp_path)
    assert e.value.code not in (0, None)
    assert not scene_calls and not series_calls[0]


def test_out_is_required(monkeypatch, scene_calls):
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, "level2", "scene.SAFE")
    assert e.value.code == 2
