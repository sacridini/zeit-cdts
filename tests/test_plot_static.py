"""zeit.plot: colour inference, frames, and static matplotlib figures."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._plot import infer_style
from zeit._plot._data import Frames, decimal_years, prepare
from zeit._plot._static import resolve_frames


def _cube(n=6, h=20, w=30, dtype=np.float32, name="ndvi"):
    rng = np.random.default_rng(0)
    data = rng.uniform(0.2, 0.9, (n, h, w)).astype(dtype)
    return xr.DataArray(data, dims=("time", "y", "x"), name=name,
                        coords={"time": pd.date_range("2000-01-01", periods=n, freq="YS"),
                                "y": np.arange(h)[::-1] + 0.5, "x": np.arange(w) + 0.5}).rio.write_crs(32721)


# ---------------------------------------------------------------- style inference

def test_continuous_and_index_palette():
    style = infer_style(np.random.default_rng(0).uniform(0.2, 0.9, 1000), name="ndvi")
    assert style.kind == "continuous" and style.cmap == "RdYlGn"
    assert 0.2 <= style.vmin < style.vmax <= 0.9
    assert infer_style(np.linspace(0, 10, 100), name="LT_Stack_NBR_site").cmap == "RdYlGn"
    assert infer_style(np.linspace(0, 10, 100), name="temperature").cmap == "viridis"


def test_diverging_categorical_years_rgb():
    assert infer_style(np.linspace(-5, 4, 100)).kind == "diverging"
    div = infer_style(np.linspace(-5, 4, 100))
    assert div.vmin == -div.vmax
    assert infer_style(np.linspace(-0.1, 10, 100)).kind == "continuous"      # barely below 0

    cat = infer_style(np.array([0, 1, 2, 2, 1], dtype=np.uint8))
    assert cat.kind == "categorical" and [c[0] for c in cat.classes] == [0, 1, 2]
    assert infer_style(np.array([True, False])).classes[1][1] == "True"
    named = infer_style(np.array([1, 2]), classes={1: "forest", 2: ("water", "#0000ff")})
    assert named.classes[1] == (2.0, "water", "#0000ff")

    years = infer_style(np.array([1990, 2001, 2015, 2020], dtype=np.uint16), name="yod")
    assert years.kind == "years" and years.vmin == 1990 and [t for _, t in years.ticks()][0] == "1990"
    assert infer_style(np.array([1990, 1995], dtype=np.uint16)).kind == "years"   # integers that look like years

    rgb = infer_style(np.random.default_rng(1).uniform(0, 3000, (50, 3)), rgb=True)
    assert rgb.kind == "rgb"
    with pytest.raises(ValueError, match="kind"):
        infer_style(np.ones(3), kind="rainbow")


def test_encode_lut_and_nodata():
    style = infer_style(np.linspace(0, 100, 101), kind="continuous", vmin=0, vmax=100)
    codes = style.encode(np.array([0.0, 50.0, 100.0, np.nan, -9999.0]), nodata=-9999)
    assert codes.tolist() == [1, 128, 255, 0, 0]
    assert style.decode(255) == pytest.approx(100) and style.decode(0) is None
    lut = style.lut()
    assert lut.shape == (256, 4) and lut[0, 3] == 0 and (lut[1:, 3] == 255).all()
    cat = infer_style(np.array([3, 7]), kind="categorical")
    assert cat.encode(np.array([7, 3, 5])).tolist() == [2, 1, 0]   # unknown value -> transparent


# ---------------------------------------------------------------- frames

def test_frames_labels_extent_and_decimation():
    frames, dataset = prepare(_cube())
    assert isinstance(frames, Frames) and dataset is None
    assert frames.n == 6 and frames.labels[:2] == ["2000", "2001"]
    assert frames.extent() == (0.0, 30.0, 0.0, 20.0) and frames.y_down()
    assert frames.frame(1, step=2).shape == (10, 15)
    assert frames.step_for(10) == 3
    assert frames.sample().ndim == 3


def test_frames_from_results_and_other_inputs(tmp_path):
    lt = xr.Dataset({"vertex_year": (("vertex", "y", "x"), np.zeros((3, 4, 5), np.int16))},
                    coords={"vertex": [1, 2, 3]})
    frames, ds = prepare(lt)
    assert frames.labels == ["1", "2", "3"] and ds is lt
    coefs = xr.DataArray(np.zeros((2, 2, 3, 4, 5)), dims=("segment", "band", "coef", "y", "x"),
                         coords={"segment": [1, 2], "band": ["red", "nir"], "coef": ["a0", "c1", "a1"]})
    frames, _ = prepare(coefs)
    assert frames.n == 12 and frames.labels[0] == "1_red_a0" and frames.labels[-1] == "2_nir_a1"
    rgb = xr.DataArray(np.ones((2, 3, 4, 5)), dims=("time", "band", "y", "x"),
                       coords={"band": ["blue", "green", "red"]})
    frames, _ = prepare(rgb)
    assert frames.rgb and frames.frame(0).shape == (4, 5, 3) and frames.n == 2
    series, _ = prepare(pd.Series([1.0, 2.0], index=pd.date_range("2000", periods=2, freq="YS")))
    assert series.ndim == 1
    path = zeit.save_raster(_cube(), tmp_path / "ndvi.tif")
    frames, _ = prepare(str(path))
    assert frames.n == 6 and frames.da.chunks is not None   # read lazily
    dates = xr.DataArray(np.array([["2020-07-02", "NaT"]], dtype="datetime64[ns]"), dims=("y", "x"))
    assert np.isnan(prepare(dates)[0].frame(0)[0, 1])
    assert decimal_years(np.array(["2020-07-02"], dtype="datetime64[ns]"))[0] == pytest.approx(2020.5)


def test_resolve_frames():
    frames, _ = prepare(_cube(n=20))
    assert resolve_frames(frames, None, default_all=True) == list(np.unique(np.linspace(0, 19, 12).astype(int)))
    assert resolve_frames(frames, "2005", True) == [5]
    assert resolve_frames(frames, 2010, True) == [10]          # a year, not an index
    assert resolve_frames(frames, [0, -1], True) == [0, 19]
    assert resolve_frames(frames, "2003-06-01", True) == [3]   # nearest date
    assert len(resolve_frames(frames, "all", True)) == 20
    with pytest.raises(ValueError, match="no frame"):
        resolve_frames(prepare(_cube().rename(time="band"))[0], "nope", True)


# ---------------------------------------------------------------- static figures

def test_static_figures(tmp_path):
    cube = _cube()
    fig = zeit.plot(cube, static=True, save=str(tmp_path / "grid.png"))
    assert len([a for a in fig.axes if a.get_visible()]) == 6 + 1      # 6 maps + colorbar
    assert (tmp_path / "grid.png").stat().st_size > 1000
    fig = zeit.plot(cube, time="2003", static=True)
    assert fig.axes[0].get_title() == "2003"
    fig = zeit.plot(cube.isel(time=0), static=True, title="first")
    assert fig._suptitle.get_text() == "first"

    classes = xr.DataArray(np.array([[0, 1], [2, 1]], dtype=np.uint8), dims=("y", "x"), name="landcover")
    fig = zeit.plot(classes, classes={0: "bare", 1: "pasture", 2: "forest"}, static=True)
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["bare", "pasture", "forest"]

    events = xr.Dataset({"yod": (("y", "x"), np.array([[0, 1990], [2005, 2020]], dtype=np.uint16)),
                         "magnitude": (("y", "x"), np.array([[np.nan, 1.0], [2.0, 3.0]]))})
    fig = zeit.plot(events, static=True)                              # first variable: yod, as years
    ticks = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
    assert "1990" in ticks
    fig = zeit.plot(events, var="magnitude", static=True)
    assert fig._suptitle.get_text() == "magnitude"

    series = cube[:, 3, 4]
    fig = zeit.plot(series, static=True)
    assert len(fig.axes[0].lines) == 1

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    assert zeit.plot(cube, time=0, ax=ax, static=True) is fig
    with pytest.raises(ValueError, match="one map"):
        zeit.plot(cube, time=[0, 1], ax=ax, static=True)
    plt.close("all")


def test_nodata_auto_for_integer_maps():
    from zeit._plot import style_for
    yod = xr.DataArray(np.array([[0, 1990], [2005, 2020]], dtype=np.uint16), dims=("y", "x"), name="yod")
    style, nodata = style_for(prepare(yod)[0])
    assert style.kind == "years" and nodata == 0 and style.vmin == 1990
    cat = xr.DataArray(np.array([[0, 1], [2, 1]], dtype=np.uint8), dims=("y", "x"))
    style, nodata = style_for(prepare(cat)[0])
    assert style.kind == "categorical" and nodata is None             # 0 is a class here


def test_regressions_orientation_and_year_maps_with_zeros():
    # data without coordinates: row 0 is drawn at the top, as in the viewer
    a = np.zeros((10, 10)); a[:3] = 1
    ax = zeit.plot(a, static=True).axes[0]
    assert ax.yaxis_inverted()
    # georeferenced north-up data: the y axis grows upwards
    assert not zeit.plot(_cube().isel(time=0), static=True).axes[0].yaxis_inverted()
    # a year map with 0 for "no event", whatever its name: years, 0 transparent
    from zeit._plot import style_for
    loss = xr.DataArray(np.array([[0, 2000, 2005], [2010, 0, 2020]], dtype=np.uint16), dims=("y", "x"), name="loss")
    style, nodata = style_for(prepare(loss)[0])
    assert style.kind == "years" and nodata == 0 and style.vmin == 2000
    import matplotlib.pyplot as plt
    plt.close("all")
