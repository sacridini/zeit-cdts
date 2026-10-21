"""zeit.landtrendr(..., ftv=[...]): other bands fitted to the segmentation's vertices.

The single-pixel engine against the original LandTrendr IDL code
(apply_fitted_trajectory_v1.pro + ftv_v1.pro, LLR-LandTrendr), run under GDL on the
cases below: vertices on observed years, desawtooth, a vertex year the band has no
observation for (moved back, and forward when the year before is a vertex already),
missing first/last years (flat vertices, and the vertex dropped past the count), two
vertices only. The original works in single precision and stores the fitted series in
an integer array; its vertex values are floats.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit import _core

YEARS = list(range(1990, 2016))

# Reference outputs of the ORIGINAL IDL code, run under GDL on exactly these inputs
# (None: a year the band has no observation for).
FTV_IDL_CASES = [
    dict(name='all_years_vertices_observed', values=[820.51, 841.51, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, 443.5, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, 697.71, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, 747.79], vertices=[1990, 1999, 2000, 2008, 2015], spike=1.0,
         idl_vertices=[1990, 1999, 2000, 2008, 2015], idl_vertvals=[831.1131591796875, 820.9727783203125, 443.5, 692.6541137695312, 744.494873046875],
         idl_yfit=[831, 829, 828, 827, 826, 825, 824, 823, 822, 820, 443, 474, 505, 536, 568, 599, 630, 661, 692, 700, 707, 714, 722, 729, 737, 744]),
    dict(name='desawtooth', values=[820.51, 841.51, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, 443.5, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, 697.71, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, 747.79], vertices=[1990, 1999, 2000, 2008, 2015], spike=0.9,
         idl_vertices=[1990, 1999, 2000, 2008, 2015], idl_vertvals=[831.1131591796875, 820.9727783203125, 443.5, 692.6541137695312, 743.8677368164062],
         idl_yfit=[831, 829, 828, 827, 826, 825, 824, 823, 822, 820, 443, 474, 505, 536, 568, 599, 630, 661, 692, 699, 707, 714, 721, 729, 736, 743]),
    dict(name='vertex_year_cloudy_moves_back', values=[820.51, 841.51, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, 443.5, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, None, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, 747.79], vertices=[1990, 1999, 2000, 2008, 2015], spike=1.0,
         idl_vertices=[1990, 1999, 2000, 2007, 2015], idl_vertvals=[831.1131591796875, 820.9727783203125, 443.5, 659.4874877929688, 754.79541015625],
         idl_yfit=[831, 829, 828, 827, 826, 825, 824, 823, 822, 820, 443, 474, 505, 536, 566, 597, 628, 659, 671, 683, 695, 707, 719, 730, 742, 754]),
    dict(name='vertex_year_cloudy_moves_forward', values=[820.51, 841.51, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, None, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, 697.71, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, 747.79], vertices=[1990, 1999, 2000, 2008, 2015], spike=1.0,
         idl_vertices=[1990, 1999, 2001, 2008, 2015], idl_vertvals=[831.1131591796875, 820.9727783203125, 453.54998779296875, 702.801025390625, 740.4360961914062],
         idl_yfit=[831, 829, 828, 827, 826, 825, 824, 823, 822, 820, 637, 453, 489, 524, 560, 595, 631, 667, 702, 708, 713, 718, 724, 729, 735, 740]),
    dict(name='first_year_missing', values=[None, 841.51, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, 443.5, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, 697.71, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, 747.79], vertices=[1990, 1999, 2000, 2008, 2015], spike=1.0,
         idl_vertices=[1990, 1999, 2000, 2008, 2015], idl_vertvals=[834.6990966796875, 818.616455078125, 443.5, 692.6541137695312, 744.494873046875],
         idl_yfit=[834, 832, 831, 829, 827, 825, 823, 822, 820, 818, 443, 474, 505, 536, 568, 599, 630, 661, 692, 700, 707, 714, 722, 729, 737, 744]),
    dict(name='last_year_missing', values=[820.51, 841.51, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, 443.5, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, 697.71, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, None], vertices=[1990, 1999, 2000, 2008, 2015], spike=1.0,
         idl_vertices=[1990, 1999, 2000, 2008, 2015], idl_vertvals=[831.1131591796875, 820.9727783203125, 443.5, 692.6541137695312, 735.5681762695312],
         idl_yfit=[831, 829, 828, 827, 826, 825, 824, 823, 822, 820, 443, 474, 505, 536, 568, 599, 630, 661, 692, 698, 704, 711, 717, 723, 729, 735]),
    dict(name='both_ends_missing', values=[None, None, 840.59, 815.68, 819.97, 817.64, 835.21, 826.94, 840.09, 802.29, 443.5, 453.55, 500.21, 522.95, 554.31, 601.95, 642.37, 661.96, 697.71, 718.86, 704.09, 703.0, 740.21, 732.8, 722.62, None], vertices=[1990, 1999, 2000, 2008, 2015], spike=0.9,
         idl_vertices=[1990, 1999, 2000, 2008, 2015], idl_vertvals=[829.2833251953125, 820.3191528320312, 443.5, 692.6541137695312, 734.7412109375],
         idl_yfit=[829, 828, 827, 826, 825, 824, 823, 822, 821, 820, 443, 474, 505, 536, 568, 599, 630, 661, 692, 698, 704, 710, 716, 722, 728, 734]),
    dict(name='two_vertices', values=[281.3, 276.27, 288.3, 373.47, 431.89, None, 414.26, 483.4, 520.69, 504.0, 561.79, 605.72, 579.72, None, 649.91, 669.9, 727.95, 656.62, 705.54, 722.47, 710.64, 809.06, 849.11, 822.45, 931.43, 932.88], vertices=[1990, 2015], spike=0.9,
         idl_vertices=[1990, 2015], idl_vertvals=[287.0135192871094, 903.28369140625],
         idl_yfit=[287, 311, 336, 360, 385, 410, 434, 459, 484, 508, 533, 558, 582, 607, 632, 656, 681, 706, 730, 755, 780, 804, 829, 853, 878, 903]),
    dict(name='two_disturbances', values=[5125.48, 5113.67, 5257.8, None, 5256.12, 5287.22, 4846.46, 2169.41, 2449.92, 3056.3, 3212.19, 3696.35, 4168.57, 4324.75, 4653.05, None, 4698.5, 4528.97, 4883.91, 4821.04, 1464.39, 2106.39, None, 3298.24, 3105.28, 4076.63], vertices=[1990, 1996, 1997, 2003, 2009, 2010, 2015], spike=0.9,
         idl_vertices=[1990, 1996, 1997, 2003, 2009, 2010, 2015], idl_vertvals=[5200.46728515625, 5095.11669921875, 2169.409912109375, 4425.5576171875, 4882.49755859375, 1464.39013671875, 3990.69482421875],
         idl_yfit=[5200, 5182, 5165, 5147, 5130, 5112, 5095, 2169, 2545, 2921, 3297, 3673, 4049, 4425, 4501, 4577, 4654, 4730, 4806, 4882, 1464, 1969, 2474, 2980, 3485, 3990]),
    dict(name='two_disturbances_ends', values=[None, 5113.67, 5257.8, 4833.6, 5256.12, 5287.22, 4846.46, 2169.41, 2449.92, 3056.3, 3212.19, 3696.35, 4168.57, 4324.75, 4653.05, 4545.67, 4698.5, 4528.97, 4883.91, 4821.04, None, 2106.39, 2711.95, 3298.24, 3105.28, None], vertices=[1990, 1996, 1997, 2003, 2009, 2010, 2015], spike=1.0,
         idl_vertices=[1990, 1996, 1997, 2003, 2009, 2011, 2015], idl_vertvals=[5158.09326171875, 5040.19775390625, 2169.409912109375, 4425.5576171875, 4845.21826171875, 2106.389892578125, 3389.08935546875],
         idl_yfit=[5158, 5138, 5118, 5099, 5079, 5059, 5040, 2169, 2545, 2921, 3297, 3673, 4049, 4425, 4495, 4565, 4635, 4705, 4775, 4845, 3475, 2106, 2427, 2747, 3068, 3389]),
]


@pytest.mark.parametrize("case", FTV_IDL_CASES, ids=lambda c: c["name"])
def test_fit_to_vertices_matches_the_original(case):
    values = [np.nan if v is None else v for v in case["values"]]
    fit = np.asarray(_core.landtrendr.fit_to_vertices(YEARS, values, case["vertices"], case["spike"]))
    at = [YEARS.index(y) for y in case["idl_vertices"]]
    np.testing.assert_allclose(fit[at], case["idl_vertvals"], rtol=2e-6, atol=1e-3)  # float32 in the original
    assert (np.trunc(fit) == np.asarray(case["idl_yfit"])).all()  # its yfit is an integer array


def test_without_observations_or_vertices_the_fit_is_nan():
    f = _core.landtrendr.fit_to_vertices
    assert np.isnan(f(YEARS, [np.nan] * len(YEARS), [1990, 2015])).all()
    assert np.isnan(f(YEARS, list(range(len(YEARS))), [])).all()


# --- the API ---------------------------------------------------------------------


def stack(bands=("ndvi", "nbr", "tcw"), rows=4, cols=5, seed=0) -> xr.DataArray:
    """A (time, band, y, x) annual cube: a disturbance in 2000 in every band, recovering."""
    rng = np.random.default_rng(seed)
    t = len(YEARS)
    base = np.interp(YEARS, [1990, 1999, 2000, 2015], [8000, 8100, 3000, 7500])
    values = np.empty((t, len(bands), rows, cols), dtype=np.float32)
    for b, scale in enumerate((1.0, 0.8, -0.3)[:len(bands)]):
        values[:, b] = (base * scale)[:, None, None] + rng.normal(0, 150, (t, rows, cols))
    values[5, 1, 0, 0] = np.nan  # a cloudy year of one band only
    tr = from_origin(500000, 9000000, 30, 30)
    da = xr.DataArray(values, dims=("time", "band", "y", "x"),
                      coords={"time": pd.to_datetime([f"{y}-01-01" for y in YEARS]), "band": list(bands),
                              "y": tr.f - 30 * (np.arange(rows) + 0.5), "x": tr.c + 30 * (np.arange(cols) + 0.5)})
    return da.rio.write_crs("EPSG:32722").rio.write_transform(tr)


def test_ftv_bands_are_fitted_to_the_segmentation_vertices():
    cube = stack()
    lt = zeit.landtrendr(cube, band="ndvi", ftv=["nbr", "tcw"])
    for name in ("nbr", "tcw"):
        assert lt[f"ftv_{name}"].dims == ("time", "y", "x") and lt[f"ftv_{name}"].shape == (len(YEARS), 4, 5)
        assert lt[f"vertex_value_{name}"].dims == ("vertex", "y", "x")
    assert (lt.time.values == cube.time.values).all() and lt.rio.crs == cube.rio.crs
    for r, c in [(0, 0), (2, 3)]:
        n = int(lt.n_vertices[r, c])
        vertex_years = [int(v) for v in lt.vertex_year[:n, r, c]]
        series = cube.sel(band="nbr").values[:, r, c].astype(float)
        expected = np.asarray(_core.landtrendr.fit_to_vertices(YEARS, list(series), vertex_years, 0.9))
        np.testing.assert_allclose(lt.ftv_nbr.values[:, r, c], expected, rtol=1e-6)  # float32 output
        at = [YEARS.index(y) for y in vertex_years]
        np.testing.assert_allclose(lt.vertex_value_nbr.values[:n, r, c], expected[at], rtol=1e-6)
        assert np.isnan(lt.vertex_value_nbr.values[n:, r, c]).all()
    # the segmentation itself is unchanged by ftv=
    alone = zeit.landtrendr(cube, band="ndvi")
    for name in ("vertex_year", "vertex_value", "n_vertices", "rmse"):
        np.testing.assert_array_equal(lt[name].values, alone[name].values)


def test_ftv_lazy_and_from_a_dataset_give_the_same():
    cube = stack()
    eager = zeit.landtrendr(cube, band="ndvi", ftv=["nbr"], fitted=True)
    lazy = zeit.landtrendr(cube.chunk({"y": 2, "x": 3}), band="ndvi", ftv=["nbr"], fitted=True)
    assert lazy.ftv_nbr.chunks is not None
    np.testing.assert_array_equal(lazy.ftv_nbr.values, eager.ftv_nbr.values)
    np.testing.assert_array_equal(lazy.fitted.values, eager.fitted.values)
    ds = cube.to_dataset("band")
    from_ds = zeit.landtrendr(ds, band="ndvi", ftv=["nbr"])
    np.testing.assert_array_equal(from_ds.ftv_nbr.values, eager.ftv_nbr.values)


def test_ftv_works_with_events_and_saving(tmp_path):
    lt = zeit.landtrendr(stack(), band="ndvi", ftv=["nbr"])
    events = zeit.extract_events(lt)
    assert "yod" in events
    out = zeit.save_raster(lt, tmp_path / "lt")
    assert (out / "ftv_nbr.tif").exists() and (out / "vertex_value_nbr.tif").exists()


def test_ftv_errors():
    cube = stack()
    with pytest.raises(ValueError, match="no bands"):
        zeit.landtrendr(cube, band="ndvi", ftv=["swir"])
    with pytest.raises(ValueError, match="several bands"):
        zeit.landtrendr(cube.sel(band="ndvi"), ftv=["nbr"])
    with pytest.raises(ValueError, match="several bands"):
        zeit.landtrendr([1.0, 2.0, 3.0], years=[2000, 2001, 2002], ftv=["nbr"])
    with pytest.raises(ValueError, match="twice"):
        zeit.landtrendr(cube, band="ndvi", ftv=["nbr", "nbr"])
