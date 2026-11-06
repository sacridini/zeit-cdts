"""zeit.improphe: FORCE's CF- and L2-ImproPhe (Frantz et al. 2016), ported from FORCE 3.11."""
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor
from zeit import _core

THR = np.array([0.025, 0.05, 0.1, 0.2, 0.4], dtype=np.float32).astype(np.float64)


def rescale(w, mn, mx):
    if w == 0 or mn == mx:
        return 1.0
    return 1.0 + np.exp(25.0 * ((w - mn) / (mx - mn)) - 7.5)


def naive_pixel(hr, hr_tex, mr, mr_tex, h, i, j, mink):
    """spec 1.6-1.8 of FORCE's improphe(), one pixel, literally: hr (npc, ny, nx), mr (nb, ny, nx) float32."""
    npc, ny, nx = hr.shape
    nb = mr.shape[0]
    if np.isnan(mr[:, i, j]).any():
        return np.full(nb, np.nan)
    keep = []   # (S, class, ni, nj), in FORCE's scan order
    sn = [0] * 5
    for ii in range(-h, h + 1):
        for jj in range(-h, h + 1):
            ni, nj = i + ii, j + jj
            if ni < 0 or nj < 0 or ni >= ny or nj >= nx:
                continue
            if np.float64(np.float32(np.sqrt(ii * ii + jj * jj))) > h:
                continue
            if np.isnan(hr[0, ni, nj]) or np.isnan(mr[:, ni, nj]).any():
                continue
            diff = np.abs(hr[:, i, j] - hr[:, ni, nj])            # float32 differences
            s = float(np.sum(diff.astype(np.float64))) / float(np.float32(npc))
            sc = next((k for k in range(5) if s < THR[k]), None)
            if sc is None:
                continue
            keep.append((s, sc, ni, nj))
            for k in range(sc, 5):
                sn[k] += 1
    if not keep:
        return mr[:, i, j].astype(np.float64)
    sc = next((k for k in range(5) if sn[k] >= mink), 4)
    use = [k for k in keep if k[1] <= sc]

    def positive_range(values):
        pos = [v for v in values if v > 0]
        return (min(pos), max(pos)) if pos else (np.inf, -np.inf)

    s_rng = positive_range([k[0] for k in use])
    t_rng = positive_range([float(hr_tex[k[2], k[3]]) for k in use])
    out = np.empty(nb)
    for b in range(nb):
        u_rng = positive_range([float(mr_tex[b, k[2], k[3]]) for k in use])
        num = den = 0.0
        for s, _, ni, nj in use:
            ss = rescale(s, *s_rng)
            tt = rescale(float(hr_tex[ni, nj]), *t_rng)
            uu = rescale(float(mr_tex[b, ni, nj]), *u_rng)
            w = 1.0 / (ss * tt * uu)
            num += w * float(mr[b, ni, nj])
            den += w
        out[b] = num / den
    return out


def blobs(ny, nx, seeds, rng_seed=0):
    """Voronoi classes of the given seed pixels."""
    ii, jj = np.mgrid[:ny, :nx]
    d = np.stack([(ii - a) ** 2 + (jj - b) ** 2 for a, b in seeds])
    return d.argmin(0)


def fine_cube(ny=30, nx=30, years=(2020, 2021), seed=0, nb=3, res=10.0, crs=32633, empty_years=()):
    """Fine reflectance (time, band, y, x): four classes with their own seasonal profile, 7 scenes a year, and the
    class map."""
    rng = np.random.default_rng(seed)
    cls = blobs(ny, nx, [(7, 7), (8, 22), (22, 9), (23, 23)])
    base = np.array([[1000, 2000, 3000], [3000, 1500, 800], [500, 4000, 2500], [5000, 900, 3500]], dtype=float)[:, :nb]
    amp = np.array([300.0, 900.0, 600.0, 1200.0])
    dates, scenes = [], []
    for y in years:
        if y in empty_years:
            continue
        for doy in (20, 70, 120, 170, 220, 270, 320):
            date = pd.Timestamp(f"{y}-01-01") + pd.to_timedelta(doy - 1, unit="D")
            dates.append(date)
            season = np.sin(doy / 365 * 2 * np.pi)
            x = base[cls] + (amp[cls] * season)[..., None] * np.array([1.0, 0.5, -0.5])[:nb]
            x = x.transpose(2, 0, 1) + rng.normal(0, 20, (nb, ny, nx))
            scenes.append(x.astype(np.float32))
    arr = np.stack(scenes)
    da = xr.DataArray(arr, dims=("time", "band", "y", "x"),
                      coords={"time": dates, "band": [f"b{i}" for i in range(nb)],
                              "y": 5000000.0 - res * (np.arange(ny) + 0.5), "x": 500000.0 + res * (np.arange(nx) + 0.5)},
                      name="reflectance")
    return da.rio.write_crs(f"EPSG:{crs}"), cls


def coarse_field(cls, years, factor=3, res=10.0, crs=32633, by_year_scale=0.1):
    """The truth (a value per class and year) averaged over blocks of factor x factor pixels, on the coarse grid."""
    ny, nx = cls.shape
    value = np.array([10.0, 40.0, 25.0, 70.0])
    truth = np.stack([value[cls] * (1 + by_year_scale * (y - years[0])) for y in years])
    cy, cx = ny // factor, nx // factor
    coarse = truth[:, :cy * factor, :cx * factor].reshape(len(years), cy, factor, cx, factor).mean((2, 4))
    da = xr.DataArray(coarse.astype(np.float32), dims=("year", "y", "x"),
                      coords={"year": list(years), "y": 5000000.0 - res * factor * (np.arange(cy) + 0.5),
                              "x": 500000.0 + res * factor * (np.arange(cx) + 0.5)}, name="lsp")
    return da.rio.write_crs(f"EPSG:{crs}"), truth


# --------------------------------------------------------------------------------------------------------------
# the core

@pytest.mark.parametrize("h", [5, 8])
def test_kernel_is_the_literal_numpy_port(h):
    rng = np.random.default_rng(1)
    ny = nx = 40
    cls = blobs(ny, nx, [(8, 8), (10, 30), (30, 12), (31, 31)])
    proto = np.array([[-1.0, 0.2, 0.0], [1.0, 0.3, 0.1], [0.1, -1.0, 0.4], [0.0, 0.9, -1.2]])
    hr = (proto[cls].transpose(2, 0, 1) + 0.04 * rng.normal(size=(3, ny, nx))).astype(np.float32)
    hr[:, 9, 9] = np.nan
    hr_tex = rng.uniform(0, 0.5, (ny, nx)).astype(np.float32)
    mr = (np.array([10.0, 40.0, 25.0, 70.0])[cls][None] * np.array([1.0, 1.5])[:, None, None]
          + rng.normal(0, 2, (2, ny, nx))).astype(np.float32)
    mr[0, 3, 4] = np.nan
    ksd = 2
    mr_tex = np.stack([_core.improphe.focal_sd(mr, ksd, b, b + 1) for b in range(2)])
    got = _core.improphe.predict(hr, hr_tex, mr, h, ksd)
    mink = _core.improphe.min_neighbors(h)
    for i in range(0, ny, 3):
        for j in range(0, nx, 3):
            want = naive_pixel(hr, hr_tex, mr, mr_tex, h, i, j, mink)
            np.testing.assert_allclose(got[:, i, j], want, rtol=1e-6, equal_nan=True, err_msg=f"({i},{j})")
    # the pixels with a missing band and with no fine data
    assert np.isnan(got[:, 3, 4]).all()
    np.testing.assert_array_equal(got[:, 9, 9], mr[:, 9, 9])


def test_focal_sd_against_numpy():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 15, 17)).astype(np.float32)
    x[0, 4, 5] = np.nan
    got = _core.improphe.focal_sd(x, 3)
    for i in range(15):
        for j in range(17):
            sd = 0.0
            for b in range(2):
                w = x[b, max(i - 3, 0):i + 4, max(j - 3, 0):j + 4].astype(np.float64).ravel()
                w = w[~np.isnan(w)]
                if len(w) > 1:
                    sd = max(sd, w.std(ddof=1))
            assert abs(got[i, j] - sd) < 1e-5 * max(sd, 1), (i, j)


def test_gsl_eigen_is_an_eigendecomposition():
    rng = np.random.default_rng(3)
    for n in (3, 12, 40):
        b = rng.normal(size=(n, n))
        a = b @ b.T
        values, vectors = _core.improphe.gsl_eigen(a)
        np.testing.assert_allclose(a @ vectors, vectors * values, atol=1e-10 * values[0])
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(n), atol=1e-12)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(a)[::-1], rtol=1e-9)


def seasonal_numpy(x, doy, year, dwin, ywin, force):
    """average_season of FORCE, literally (the windows, the empty-window rule, truncation) on x (t, b, y, x)."""
    nt, nb, ny, nx = x.shape
    nwin = len(dwin) - 1

    def window(d, yr):
        for w in range(nwin):
            lo, hi = dwin[w], dwin[w + 1]
            any_year = ywin is None
            y0 = 0 if ywin is None else ywin
            if lo < hi:
                upper = (d <= hi) if (not force and w == nwin - 1) else (d < hi)
                if d >= lo and upper and (yr == y0 or any_year):
                    return w
            if lo > hi:
                if (d >= lo and d <= 365 and (yr == y0 - 1 or any_year)) or (d < hi and (yr == y0 or any_year)):
                    return w
                if (d >= lo and d <= 365 and (yr == y0 or any_year)) or (d < hi and (yr == y0 + 1 or any_year)):
                    return w
        return None

    out = np.full((nwin * nb, ny, nx), np.nan)
    sums = np.zeros((nwin + 1, nb, ny, nx))
    k = np.zeros((nwin + 1, ny, nx))
    for t in range(nt):
        w = window(doy[t], year[t])
        if w is None:
            continue
        ok = np.isfinite(x[t]).all(0)
        v = np.where(ok, x[t], 0.0)
        sums[w] += v
        sums[nwin] += v
        k[w] += ok
        k[nwin] += ok
    for w in range(nwin):
        for b in range(nb):
            src = np.where(k[w] > 0, sums[w, b] / np.maximum(k[w], 1), sums[nwin, b] / np.maximum(k[nwin], 1))
            src = np.trunc(src) if force else src
            out[w * nb + b] = np.where(k[nwin] > 0, src, np.nan)
    return out


@pytest.mark.parametrize("force", [False, True])
def test_season_means_against_numpy(force):
    rng = np.random.default_rng(4)
    nt, nb, ny, nx = 9, 2, 5, 6
    x = rng.uniform(100, 4000, (nt, nb, ny, nx)).astype(np.float32)
    x[2, 0, 1, 1] = np.nan
    x[:, :, 3, 3] = np.nan
    doy = [10, 100, 100, 200, 365, 364, 330, 5, 90]
    year = [2020, 2020, 2020, 2020, 2020, 2020, 2019, 2021, 2020]
    for dwin, ywin in (([1, 91, 181, 271, 365], 2020), ([1, 91, 181, 271, 365], None), ([330, 60, 330], 2020)):
        got, n = _core.improphe.season_means(x, doy, year, dwin, ywin, force)
        want = seasonal_numpy(x, doy, year, dwin, ywin, force)
        np.testing.assert_allclose(got, want, rtol=1e-6, equal_nan=True)
        assert n == ny * nx - 1


def test_day_365_and_truncation_differ_between_the_modes():
    x = np.full((2, 1, 1, 1), 1000.5, dtype=np.float32)
    x[1] = 3000.9
    doy, year = [100, 365], [2020, 2020]
    dwin = [1, 91, 181, 271, 365]
    paper, n1 = _core.improphe.season_means(x, doy, year, dwin, 2020, False)
    force, n2 = _core.improphe.season_means(x, doy, year, dwin, 2020, True)
    # paper: day 365 is in the last window, the means are floats
    assert paper[3, 0, 0] == pytest.approx(3000.9, rel=1e-6) and paper[1, 0, 0] == 1000.5
    # FORCE: day 365 is in no window, so the last window has no scene and takes the mean of the scene counted,
    # truncated towards zero
    assert force[3, 0, 0] == 1000.0 and force[1, 0, 0] == 1000.0
    # and a pixel whose only scene is on day 365 has no data in FORCE's mode
    _, n = _core.improphe.season_means(x[1:], doy[1:], year[1:], dwin, 2020, True)
    assert n == 0 and n1 == n2 == 1


def test_prepare_matches_a_numpy_pipeline():
    fine, _ = fine_cube(ny=20, nx=22, seed=5)
    x = fine.transpose("time", "band", "y", "x").values.astype(np.float32)
    times = pd.DatetimeIndex(fine.time.values)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    doy = (cum[times.month - 1] + times.day).tolist()
    year = times.year.tolist()
    dwin = [1, 91, 181, 271, 365]
    for force in (False, True):
        pcs, tex, n_valid, eigenvalues = _core.improphe.prepare(x, doy, year, dwin, 2020, 5, 2, 0.975, force)
        seas = seasonal_numpy(x, doy, year, dwin, 2020, force)
        nb, ny, nx = seas.shape
        flat = seas.reshape(nb, -1)
        centred = flat - flat.mean(1, keepdims=True)
        cov = centred @ centred.T / (flat.shape[1] - 1)
        if force:
            values, vectors = _core.improphe.gsl_eigen(cov)
            rows = vectors                                  # FORCE: the row b of the eigenvector matrix
        else:
            values, vectors = np.linalg.eigh(cov)
            values, vectors = values[::-1], vectors[:, ::-1]
            rows = vectors.T
        np.testing.assert_allclose(eigenvalues, values, rtol=1e-6, atol=1e-6 * values[0])
        total = np.cumsum(values) / values.sum()
        npc = int(np.argmax(total > 0.975)) + 1
        assert pcs.shape[0] == npc
        want = (rows[:npc] @ centred).astype(np.float64)
        want = (want - want.mean(1, keepdims=True)) / want.std(1, ddof=1, keepdims=True)
        sign = np.sign((pcs.reshape(npc, -1) * want).sum(1, keepdims=True))     # a sign is the solver's choice
        np.testing.assert_allclose(pcs.reshape(npc, -1) * sign, want, atol=2e-4)
        # texture: the largest standard deviation over the components in the 5 x 5 window
        want_tex = _core.improphe.focal_sd(pcs, 2)
        np.testing.assert_allclose(tex, want_tex, rtol=1e-6)
        assert n_valid == ny * nx


# --------------------------------------------------------------------------------------------------------------
# the API

def test_cf_recovers_a_field_better_than_nearest_neighbour_upsampling():
    years = (2020, 2021)
    fine, cls = fine_cube(seed=6)
    coarse, truth = coarse_field(cls, years)
    out = zeit.improphe(coarse, fine, by="year", kernel_size=60, texture_size=20, n_jobs=2)
    assert out.dims == ("year", "y", "x") and out.dtype == np.float32
    assert list(out.year.values) == list(years)
    np.testing.assert_array_equal(out.y.values, fine.y.values)
    np.testing.assert_array_equal(out.x.values, fine.x.values)
    assert out.rio.crs == fine.rio.crs
    # the reproject path is nearest neighbour: compare with the 3 x 3 replication of the coarse field
    nn = np.kron(coarse.values, np.ones((1, 3, 3), dtype=np.float32))
    mask = np.isfinite(out.values)
    assert mask.all()
    rmse = lambda a: float(np.sqrt(np.mean((a - truth[:, :30, :30]) ** 2)))
    assert rmse(out.values) < 0.7 * rmse(nn), (rmse(out.values), rmse(nn))
    info = out.attrs["improphe"]
    assert info.startswith("CF-ImproPhe; radius 6 px, texture radius 2 px, min_neighbors 0;")
    assert re.search(r"components \(0.975 of the variance\) 2020: \d+, 2021: \d+;", info) and info.endswith("force=False")


def test_cf_with_fields_and_the_accessor():
    years = (2020, 2021)
    fine, cls = fine_cube(seed=7)
    coarse, _ = coarse_field(cls, years)
    two = xr.concat([coarse, coarse * 2.0], dim=pd.Index(["a", "b"], name="field")).transpose("year", "field", "y", "x")
    out = zeit.improphe(two, fine, kernel_size=60, texture_size=20, n_jobs=2)       # by="auto": a year dimension
    assert out.dims == ("year", "field", "y", "x") and list(out.field.values) == ["a", "b"]
    # field b is twice field a (the weights do not depend on the scale)
    np.testing.assert_allclose(out.sel(field="b").values, 2 * out.sel(field="a").values, rtol=2e-4)
    again = fine.zeit.improphe(two, kernel_size=60, texture_size=20, n_jobs=1)
    np.testing.assert_allclose(again.values, out.values, rtol=1e-6)


def test_coarse_on_the_grid_of_fine_and_other_year_layouts():
    years = (2020, 2021)
    fine, cls = fine_cube(seed=8)
    # the field already on the fine grid, as `time` with one date per year, and as `band` named by the years
    truth = np.stack([np.array([10.0, 40.0, 25.0, 70.0])[cls] * (1 + 0.1 * k) for k in range(2)]).astype(np.float32)
    base = xr.DataArray(truth, dims=("year", "y", "x"), coords={"year": list(years), "y": fine.y, "x": fine.x}, name="m")
    a = zeit.improphe(base, fine, by="year", kernel_size=60, texture_size=20, n_jobs=2)
    as_time = base.rename(year="time").assign_coords(time=pd.to_datetime([f"{y}-07-01" for y in years]))
    b = zeit.improphe(as_time, fine, by="year", kernel_size=60, texture_size=20, n_jobs=2)
    as_band = base.rename(year="band")
    c = zeit.improphe(as_band, fine, kernel_size=60, texture_size=20, n_jobs=2)         # auto: band of years
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.values, c.values)
    assert list(b.year.values) == list(years)
    # a field that is exactly a function of the classes is reproduced
    np.testing.assert_allclose(a.values, truth, rtol=1e-4)
    with pytest.raises(ValueError, match="one date per year"):
        zeit.improphe(as_time, fine, kernel_size=60, texture_size=20)


def test_nan_holes_pixels_without_fine_data_and_a_year_without_scenes():
    years = (2019, 2020, 2021)
    fine, cls = fine_cube(seed=9, years=(2020, 2021))             # no scene in 2019
    fine.values[:, :, 12, 13] = np.nan                            # a fine pixel with no data at all
    coarse, truth = coarse_field(cls, years)
    coarse.values[1, 4, 4] = np.nan                               # a hole in 2020
    out = zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, n_jobs=2)
    nn = np.kron(coarse.values, np.ones((1, 3, 3), dtype=np.float32))
    # 2019: no fine data, the coarse values are copied (replicated onto the fine grid)
    np.testing.assert_array_equal(out.sel(year=2019).values, nn[0])
    assert re.search(r"2019: 0, 2020: [1-9]", out.attrs["improphe"])
    # the hole: NaN on its 3 x 3 pixels of 2020 only
    hole = np.isnan(out.sel(year=2020).values)
    assert hole.sum() == 9 and hole[12:15, 12:15].all()
    assert np.isfinite(out.sel(year=2021).values).all()
    # a fine pixel with no data takes the coarse value
    assert out.sel(year=2021).values[12, 13] == nn[2, 12, 13]


def test_l2_scenes_shape_and_prediction():
    fine, cls = fine_cube(seed=10, nb=3)
    # three "Landsat" scenes of 2 bands on the 3x coarser grid, each the class means scaled differently
    coarse_ref, _ = coarse_field(cls, (2020,))
    dates = pd.to_datetime(["2020-03-01", "2020-06-01", "2021-03-01"])
    layers = np.stack([np.stack([coarse_ref.values[0] * s, coarse_ref.values[0] * (s + 0.5)]) for s in (1.0, 2.0, 3.0)])
    coarse = xr.DataArray(layers, dims=("time", "band", "y", "x"),
                          coords={"time": dates, "band": ["red", "nir"], "y": coarse_ref.y, "x": coarse_ref.x})
    coarse = coarse.rio.write_crs("EPSG:32633")
    out = zeit.improphe(coarse, fine, by=None, kernel_size=60, texture_size=20, n_jobs=2)       # several dates a year: auto
    assert out.dims == ("time", "band", "y", "x") and out.shape == (3, 2, 30, 30)
    assert list(out.band.values) == ["red", "nir"] and (out.time.values == coarse.time.values).all()
    info = out.attrs["improphe"]
    assert re.match(r"L2-ImproPhe; .*components \(0.975 of the variance\) [1-9]\d*;", info)
    auto = zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, n_jobs=2)
    np.testing.assert_array_equal(auto.values, out.values)
    # the scenes are scaled copies (1, 2, 3 and +0.5 for the second band): the weights do not depend on the scale
    np.testing.assert_allclose(out.values[1, 0], 2 * out.values[0, 0], rtol=2e-4)
    np.testing.assert_allclose(out.values[1, 1], out.values[0, 1] * 2.5 / 1.5, rtol=2e-4)
    # sharper than the replicated coarse field
    cls_value = np.array([10.0, 40.0, 25.0, 70.0])[cls]
    nn = np.kron(coarse.values[0, 0], np.ones((3, 3), dtype=np.float32))
    e_out = np.sqrt(np.mean((out.values[0, 0] - cls_value) ** 2))
    e_nn = np.sqrt(np.mean((nn - cls_value) ** 2))
    assert e_out < 0.7 * e_nn, (e_out, e_nn)


def test_force_mode_runs_and_differs_as_documented():
    years = (2020, 2021)
    fine, cls = fine_cube(seed=11)
    coarse, _ = coarse_field(cls, years)
    a = zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, n_jobs=2)
    b = zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, n_jobs=2, force=True)
    assert b.attrs["improphe"].endswith("force=True")
    assert np.isfinite(a.values).all() and np.isfinite(b.values).all()
    # both land near the class values (the classes are far apart in the components)
    cls_value = np.array([10.0, 40.0, 25.0, 70.0])[cls]
    for r in (a, b):
        assert abs(r.values[0] - cls_value).mean() < 5.0


def test_input_checks(recwarn):
    fine, cls = fine_cube(seed=12)
    coarse, _ = coarse_field(cls, (2020, 2021))
    with pytest.raises(ValueError, match="at least 1 pixel"):
        zeit.improphe(coarse, fine, kernel_size=5)
    with pytest.warns(UserWarning, match="multiple of the pixel size"):
        zeit.improphe(coarse, fine, kernel_size=65, texture_size=20, n_jobs=1)
    with pytest.raises(ValueError, match="pc_variance"):
        zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, pc_variance=1.5)
    with pytest.raises(ValueError, match="seasons"):
        zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, seasons=(1,))
    with pytest.raises(ValueError, match="time"):
        zeit.improphe(coarse, fine.drop_vars("time").assign_coords(time=np.arange(fine.sizes["time"])),
                      kernel_size=60, texture_size=20)
    other = coarse.rio.reproject("EPSG:32632")
    with pytest.raises(ValueError, match="different CRS"):
        zeit.improphe(other, fine, kernel_size=60, texture_size=20)
    with pytest.raises(ValueError, match="by must be"):
        zeit.improphe(coarse, fine, by="month", kernel_size=60, texture_size=20)


def test_dask_fine_is_loaded():
    fine, cls = fine_cube(seed=13)
    coarse, _ = coarse_field(cls, (2020, 2021))
    a = zeit.improphe(coarse, fine, kernel_size=60, texture_size=20, n_jobs=2)
    b = zeit.improphe(coarse.chunk({"y": 5}), fine.chunk({"time": 4}), kernel_size=60, texture_size=20, n_jobs=2)
    np.testing.assert_array_equal(a.values, b.values)
