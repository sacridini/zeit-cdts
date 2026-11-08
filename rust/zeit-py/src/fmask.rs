//! zeit._core.fmask: a minimal binding of the MATLAB primitives of the Fmask 4.7 port (`zeit_core::fmask::mlab`), for
//! the comparison with GNU Octave (`benchmarks/vs_originals/fmask4.py`). No public API: the detection comes later.
//! Images are 1-D arrays of `rows * cols` in row-major order.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use zeit_core::fmask::commission;
use zeit_core::fmask::detect::{self, Toa20};
use zeit_core::fmask::mlab::{self, Strel};

fn check(len: usize, rows: usize, cols: usize) -> PyResult<()> {
    if len != rows * cols {
        return Err(PyValueError::new_err("the array does not have rows * cols values"));
    }
    Ok(())
}

fn sl<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err("arrays must be contiguous"))
}

fn strel_of(rows: usize, cols: usize, mask: &[u8]) -> PyResult<Strel> {
    if rows % 2 == 0 || cols % 2 == 0 || mask.len() != rows * cols {
        return Err(PyValueError::new_err("the element must be odd-sized"));
    }
    Ok(Strel::new(rows, cols, mask.iter().map(|&b| b != 0).collect()))
}

#[pyfunction]
fn round_half_away(x: f64) -> f64 {
    mlab::round_half_away(x)
}

#[pyfunction]
fn prctile(py: Python<'_>, values: PyReadonlyArray1<'_, f32>, p: f64) -> PyResult<f32> {
    let v = sl(&values)?;
    Ok(py.detach(|| mlab::prctile(v, p)))
}

#[pyfunction]
fn resize_box_half<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, f32>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    if rows % 2 != 0 || cols % 2 != 0 {
        return Err(PyValueError::new_err("rows and cols must be even"));
    }
    Ok(PyArray1::from_vec(py, py.detach(|| mlab::resize_box_half(a, rows, cols))))
}

#[pyfunction]
fn resize_replicate3<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, f32>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    Ok(PyArray1::from_vec(py, py.detach(|| mlab::resize_replicate3(a, rows, cols))))
}

#[pyfunction]
fn stdfilt<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, f32>, rows: usize, cols: usize, k: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    if k % 2 == 0 {
        return Err(PyValueError::new_err("k must be odd"));
    }
    Ok(PyArray1::from_vec(py, py.detach(|| mlab::stdfilt(a, rows, cols, k))))
}

/// The mask (row-major, 0/1) of `strel('disk', r)`, its side.
#[pyfunction]
fn strel_disk<'py>(py: Python<'py>, r: usize) -> (usize, Bound<'py, PyArray1<u8>>) {
    let s = mlab::strel_disk(r);
    (s.rows, PyArray1::from_vec(py, s.mask.iter().map(|&b| b as u8).collect()))
}

/// Erosion or dilation, binary (`img` u8) or grayscale (`img` f32), by the element `mask` (`se_rows x se_cols`)
/// or, with `square > 0`, by the square of that side (the fast two-pass one).
#[pyfunction]
#[pyo3(signature = (img, rows, cols, erode, mask = None, se_rows = 0, se_cols = 0, square = 0))]
#[allow(clippy::too_many_arguments)]
fn morph_bin<'py>(
    py: Python<'py>,
    img: PyReadonlyArray1<'py, u8>,
    rows: usize,
    cols: usize,
    erode: bool,
    mask: Option<PyReadonlyArray1<'py, u8>>,
    se_rows: usize,
    se_cols: usize,
    square: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    let r = if square > 0 {
        if square % 2 == 0 {
            return Err(PyValueError::new_err("square must be odd"));
        }
        py.detach(|| if erode { mlab::erode_bin_square(a, rows, cols, square) } else { mlab::dilate_bin_square(a, rows, cols, square) })
    } else {
        let m = mask.ok_or_else(|| PyValueError::new_err("mask or square"))?;
        let se = strel_of(se_rows, se_cols, sl(&m)?)?;
        py.detach(|| if erode { mlab::erode_bin(a, rows, cols, &se) } else { mlab::dilate_bin(a, rows, cols, &se) })
    };
    Ok(PyArray1::from_vec(py, r))
}

#[pyfunction]
#[pyo3(signature = (img, rows, cols, erode, mask = None, se_rows = 0, se_cols = 0, square = 0))]
#[allow(clippy::too_many_arguments)]
fn morph_gray<'py>(
    py: Python<'py>,
    img: PyReadonlyArray1<'py, f32>,
    rows: usize,
    cols: usize,
    erode: bool,
    mask: Option<PyReadonlyArray1<'py, u8>>,
    se_rows: usize,
    se_cols: usize,
    square: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    let r = if square > 0 {
        if square % 2 == 0 {
            return Err(PyValueError::new_err("square must be odd"));
        }
        py.detach(|| if erode { mlab::erode_gray_square(a, rows, cols, square) } else { mlab::dilate_gray_square(a, rows, cols, square) })
    } else {
        let m = mask.ok_or_else(|| PyValueError::new_err("mask or square"))?;
        let se = strel_of(se_rows, se_cols, sl(&m)?)?;
        py.detach(|| if erode { mlab::erode_gray(a, rows, cols, &se) } else { mlab::dilate_gray(a, rows, cols, &se) })
    };
    Ok(PyArray1::from_vec(py, r))
}

#[pyfunction]
fn bwlabel<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, u8>, rows: usize, cols: usize) -> PyResult<(Bound<'py, PyArray1<i32>>, usize)> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    let (l, n) = py.detach(|| mlab::bwlabel(a, rows, cols));
    Ok((PyArray1::from_vec(py, l), n))
}

#[pyfunction]
fn bwareaopen<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, u8>, rows: usize, cols: usize, min: usize) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    Ok(PyArray1::from_vec(py, py.detach(|| mlab::bwareaopen(a, rows, cols, min))))
}

/// `(area, centroid [n, 2] (x, y), pixel_list [sum(area), 2] (x, y) region after region, all 1-based)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn regionprops<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray1<'py, i32>,
    rows: usize,
    cols: usize,
    n: usize,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u32>>)> {
    let a = sl(&labels)?;
    check(a.len(), rows, cols)?;
    let regs = py.detach(|| mlab::regionprops(a, rows, cols, n));
    let area: Vec<u64> = regs.iter().map(|r| r.area as u64).collect();
    let cen: Vec<f64> = regs.iter().flat_map(|r| [r.centroid.0, r.centroid.1]).collect();
    let pix: Vec<u32> = regs.iter().flat_map(|r| r.pixel_list.iter().flat_map(|&(x, y)| [x, y])).collect();
    Ok((PyArray1::from_vec(py, area), PyArray1::from_vec(py, cen), PyArray1::from_vec(py, pix)))
}

#[pyfunction]
fn imfill_gray<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, f32>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    Ok(PyArray1::from_vec(py, py.detach(|| mlab::imfill_gray(a, rows, cols))))
}

#[pyfunction]
fn imfilter3<'py>(py: Python<'py>, img: PyReadonlyArray1<'py, f32>, rows: usize, cols: usize, t: [f32; 9]) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a = sl(&img)?;
    check(a.len(), rows, cols)?;
    Ok(PyArray1::from_vec(py, py.detach(|| mlab::imfilter3(a, rows, cols, &t))))
}

// ---- fmask::detect: one function per .m of the potential pixels (Sentinel-2) ----

type A1<'py, T> = PyReadonlyArray1<'py, T>;

#[pyfunction]
fn parameters() -> Vec<f64> {
    let p = detect::FmaskParameters::default();
    vec![p.cloud_buffer, p.cloud_shadow_buffer, p.snow_buffer, p.thin_weight, p.cloud_probability_threshold, p.pfpc_layer_extension_radius, p.pfpc_erosion_radius, p.output_resolution, p.shadow_water]
}

#[pyfunction]
fn ndvi<'py>(py: Python<'py>, red: A1<'py, f32>, nir: A1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (r, n) = (sl(&red)?, sl(&nir)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::ndvi(r, n))))
}

#[pyfunction]
fn ndsi<'py>(py: Python<'py>, green: A1<'py, f32>, swir: A1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (g, s) = (sl(&green)?, sl(&swir)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::ndsi(g, s))))
}

#[pyfunction]
fn ndbi<'py>(py: Python<'py>, nir: A1<'py, f32>, swir: A1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (n, s) = (sl(&nir)?, sl(&swir)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::ndbi(n, s))))
}

#[pyfunction]
fn saturate<'py>(py: Python<'py>, blue: A1<'py, u8>, green: A1<'py, u8>, red: A1<'py, u8>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (b, g, r) = (sl(&blue)?, sl(&green)?, sl(&red)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::saturate(b, g, r))))
}

#[pyfunction]
fn detect_snow<'py>(py: Python<'py>, green: A1<'py, f32>, nir: A1<'py, f32>, ndsi: A1<'py, f32>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (g, n, d) = (sl(&green)?, sl(&nir)?, sl(&ndsi)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::detect_snow(g, n, d))))
}

#[pyfunction]
#[pyo3(signature = (mask, nir, ndvi, psnow, gswater = None))]
fn detect_water<'py>(
    py: Python<'py>,
    mask: A1<'py, u8>,
    nir: A1<'py, f32>,
    ndvi: A1<'py, f32>,
    psnow: A1<'py, u8>,
    gswater: Option<A1<'py, u8>>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>)> {
    let (m, n, v, p) = (sl(&mask)?, sl(&nir)?, sl(&ndvi)?, sl(&psnow)?);
    let g = match &gswater {
        Some(a) => Some(sl(a)?),
        None => None,
    };
    let (w, wa) = py.detach(|| detect::detect_water(m, n, v, p, g));
    Ok((PyArray1::from_vec(py, w), PyArray1::from_vec(py, wa)))
}

#[pyfunction]
#[pyo3(signature = (mask, idplcd, cirrus, dem = None))]
fn normaliza_cirrus_dem<'py>(
    py: Python<'py>,
    mask: A1<'py, u8>,
    idplcd: A1<'py, u8>,
    cirrus: A1<'py, f32>,
    dem: Option<A1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (m, i, c) = (sl(&mask)?, sl(&idplcd)?, sl(&cirrus)?);
    let d = match &dem {
        Some(a) => Some(sl(a)?),
        None => None,
    };
    Ok(PyArray1::from_vec(py, py.detach(|| detect::normaliza_cirrus_dem(m, i, c, d))))
}

#[pyfunction]
#[pyo3(signature = (mask, blue, green, red, nir, swir1, swir2, cirrus, ndvi, ndsi, satu_bv, dem = None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn detect_potential_pixels<'py>(
    py: Python<'py>,
    mask: A1<'py, u8>,
    blue: A1<'py, f32>,
    green: A1<'py, f32>,
    red: A1<'py, f32>,
    nir: A1<'py, f32>,
    swir1: A1<'py, f32>,
    swir2: A1<'py, f32>,
    cirrus: A1<'py, f32>,
    ndvi: A1<'py, f32>,
    ndsi: A1<'py, f32>,
    satu_bv: A1<'py, u8>,
    dem: Option<A1<'py, f32>>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let toa = Toa20 { blue: sl(&blue)?, green: sl(&green)?, red: sl(&red)?, nir: sl(&nir)?, swir1: sl(&swir1)?, swir2: sl(&swir2)?, cirrus: sl(&cirrus)? };
    let (m, v, s, sb) = (sl(&mask)?, sl(&ndvi)?, sl(&ndsi)?, sl(&satu_bv)?);
    let d = match &dem {
        Some(a) => Some(sl(a)?),
        None => None,
    };
    let r = py.detach(|| detect::detect_potential_pixels(m, &toa, d, v, s, sb));
    Ok((PyArray1::from_vec(py, r.idplcd), PyArray1::from_vec(py, r.cirrus_normal), PyArray1::from_vec(py, r.whiteness), PyArray1::from_vec(py, r.hot)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn detect_abs_snow<'py>(
    py: Python<'py>,
    green: A1<'py, f32>,
    satu_green: A1<'py, u8>,
    ndsi: A1<'py, f32>,
    psnow: A1<'py, u8>,
    rows: usize,
    cols: usize,
    resolution: f64,
) -> PyResult<Option<Bound<'py, PyArray1<u8>>>> {
    let (g, sg, d, p) = (sl(&green)?, sl(&satu_green)?, sl(&ndsi)?, sl(&psnow)?);
    check(g.len(), rows, cols)?;
    let r = py.detach(|| detect::detect_abs_snow(g, sg, d, p, rows, cols, resolution));
    Ok(r.map(|v| PyArray1::from_vec(py, v)))
}

#[pyfunction]
fn prob_thin<'py>(py: Python<'py>, cirrus: A1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = sl(&cirrus)?;
    Ok(PyArray1::from_vec(py, py.detach(|| detect::prob_thin(c))))
}

#[pyfunction]
fn probw_brightness<'py>(py: Python<'py>, swir: A1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let c = sl(&swir)?;
    Ok(PyArray1::from_vec(py, py.detach(|| detect::probw_brightness(c))))
}

#[pyfunction]
fn probl_brightness<'py>(py: Python<'py>, hot: A1<'py, f32>, idlnd: A1<'py, u8>, l_pt: f64, h_pt: f64) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (h, i) = (sl(&hot)?, sl(&idlnd)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::probl_brightness(h, i, l_pt, h_pt))))
}

#[pyfunction]
fn probl_spectral_variability<'py>(
    py: Python<'py>,
    ndvi: A1<'py, f32>,
    ndsi: A1<'py, f32>,
    ndbi: A1<'py, f32>,
    whiteness: A1<'py, f32>,
    satu_green: A1<'py, u8>,
    satu_red: A1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (a, b, c, d, e, f) = (sl(&ndvi)?, sl(&ndsi)?, sl(&ndbi)?, sl(&whiteness)?, sl(&satu_green)?, sl(&satu_red)?);
    Ok(PyArray1::from_vec(py, py.detach(|| detect::probl_spectral_variability(a, b, c, d, e, f))))
}

/// `(sum_clr, cloud, idused or None, t_templ, t_temph)`.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn detect_potential_cloud<'py>(
    py: Python<'py>,
    mask: A1<'py, u8>,
    water: A1<'py, u8>,
    swir1: A1<'py, f32>,
    cirrus: A1<'py, f64>,
    satu_green: A1<'py, u8>,
    satu_red: A1<'py, u8>,
    ndvi: A1<'py, f32>,
    ndsi: A1<'py, f32>,
    ndbi: A1<'py, f32>,
    idplcd: A1<'py, u8>,
    whiteness: A1<'py, f32>,
    hot: A1<'py, f32>,
    wpt: f64,
    cldprob: f64,
) -> PyResult<(usize, Bound<'py, PyArray1<u8>>, Option<Bound<'py, PyArray1<u8>>>, f64, f64)> {
    let (m, w, s1, c, sg, sr) = (sl(&mask)?, sl(&water)?, sl(&swir1)?, sl(&cirrus)?, sl(&satu_green)?, sl(&satu_red)?);
    let (nv, ns, nb, ip, wh, ho) = (sl(&ndvi)?, sl(&ndsi)?, sl(&ndbi)?, sl(&idplcd)?, sl(&whiteness)?, sl(&hot)?);
    let r = py.detach(|| detect::detect_potential_cloud(m, w, s1, c, sg, sr, nv, ns, nb, ip, wh, ho, wpt, cldprob));
    Ok((r.sum_clr, PyArray1::from_vec(py, r.cloud), r.idused.map(|v| PyArray1::from_vec(py, v)), r.t_templ, r.t_temph))
}

// ---- the internals of DetectPotentialCloud / DetectAbsSnow, and fmask::commission ----

/// `(sum_clr, cloud, idused or None, probs or None)`, probs = `(lprob_final, wprob_final, clr_max, wclr_max)` (single).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn detect_potential_cloud_probs<'py>(
    py: Python<'py>,
    mask: A1<'py, u8>,
    water: A1<'py, u8>,
    swir1: A1<'py, f32>,
    cirrus: A1<'py, f64>,
    satu_green: A1<'py, u8>,
    satu_red: A1<'py, u8>,
    ndvi: A1<'py, f32>,
    ndsi: A1<'py, f32>,
    ndbi: A1<'py, f32>,
    idplcd: A1<'py, u8>,
    whiteness: A1<'py, f32>,
    hot: A1<'py, f32>,
    wpt: f64,
    cldprob: f64,
) -> PyResult<(usize, Bound<'py, PyArray1<u8>>, Option<Bound<'py, PyArray1<u8>>>, Option<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, f32, f32)>)> {
    let (m, w, s1, c, sg, sr) = (sl(&mask)?, sl(&water)?, sl(&swir1)?, sl(&cirrus)?, sl(&satu_green)?, sl(&satu_red)?);
    let (nv, ns, nb, ip, wh, ho) = (sl(&ndvi)?, sl(&ndsi)?, sl(&ndbi)?, sl(&idplcd)?, sl(&whiteness)?, sl(&hot)?);
    let (r, p) = py.detach(|| detect::detect_potential_cloud_probs(m, w, s1, c, sg, sr, nv, ns, nb, ip, wh, ho, wpt, cldprob));
    let probs = p.map(|p| (PyArray1::from_vec(py, p.lprob_final), PyArray1::from_vec(py, p.wprob_final), p.clr_max, p.wclr_max));
    Ok((r.sum_clr, PyArray1::from_vec(py, r.cloud), r.idused.map(|v| PyArray1::from_vec(py, v)), probs))
}

/// `(abs_snow, scsi)` or None.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn detect_abs_snow_scsi<'py>(
    py: Python<'py>,
    green: A1<'py, f32>,
    satu_green: A1<'py, u8>,
    ndsi: A1<'py, f32>,
    psnow: A1<'py, u8>,
    rows: usize,
    cols: usize,
    resolution: f64,
) -> PyResult<Option<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<f64>>)>> {
    let (g, sg, d, p) = (sl(&green)?, sl(&satu_green)?, sl(&ndsi)?, sl(&psnow)?);
    check(g.len(), rows, cols)?;
    let r = py.detach(|| detect::detect_abs_snow_scsi(g, sg, d, p, rows, cols, resolution));
    Ok(r.map(|(a, s)| (PyArray1::from_vec(py, a), PyArray1::from_vec(py, s))))
}

#[pyfunction]
fn cdi<'py>(py: Python<'py>, band7: A1<'py, f32>, band8: A1<'py, f32>, band8a: A1<'py, f32>, rows: usize, cols: usize) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let (a, b, c) = (sl(&band7)?, sl(&band8)?, sl(&band8a)?);
    check(c.len(), rows, cols)?;
    Ok(py.detach(|| commission::cdi(a, b, c, rows, cols)).map(|v| PyArray1::from_vec(py, v)))
}

#[pyfunction]
fn enhance_line<'py>(py: Python<'py>, band: A1<'py, f32>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let b = sl(&band)?;
    check(b.len(), rows, cols)?;
    Ok(PyArray1::from_vec(py, py.detach(|| commission::enhance_line(b, rows, cols))))
}

/// DetectPotentialFalsePositivePixels; with `enhanced` the `ndbi` is already the output of EnhanceLine.
#[pyfunction]
#[pyo3(signature = (mask, psnow, ndbi, ndvi, water, rows, cols, resolution, slope = None, cdi = None, enhanced = false))]
#[allow(clippy::too_many_arguments)]
fn detect_potential_false_positive_pixels<'py>(
    py: Python<'py>,
    mask: A1<'py, u8>,
    psnow: A1<'py, u8>,
    ndbi: A1<'py, f32>,
    ndvi: A1<'py, f32>,
    water: A1<'py, u8>,
    rows: usize,
    cols: usize,
    resolution: f64,
    slope: Option<A1<'py, f32>>,
    cdi: Option<A1<'py, f64>>,
    enhanced: bool,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (m, p, nb, nv, w) = (sl(&mask)?, sl(&psnow)?, sl(&ndbi)?, sl(&ndvi)?, sl(&water)?);
    check(m.len(), rows, cols)?;
    let s = match &slope {
        Some(a) => Some(sl(a)?),
        None => None,
    };
    let c = match &cdi {
        Some(a) => Some(sl(a)?),
        None => None,
    };
    let r = py.detach(|| {
        if enhanced {
            commission::pfpl_from_enhanced(m, p, s, nb, nv, c, w, rows, cols, resolution)
        } else {
            commission::detect_potential_false_positive_pixels(m, p, s, nb, nv, c, w, rows, cols, resolution)
        }
    });
    Ok(PyArray1::from_vec(py, r))
}

#[pyfunction]
#[pyo3(signature = (pcloud, pfpl, water, erdpix, rows, cols, cdi = None))]
fn erode_commissons<'py>(
    py: Python<'py>,
    pcloud: A1<'py, u8>,
    pfpl: A1<'py, u8>,
    water: A1<'py, u8>,
    erdpix: usize,
    rows: usize,
    cols: usize,
    cdi: Option<A1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (p, f, w) = (sl(&pcloud)?, sl(&pfpl)?, sl(&water)?);
    check(p.len(), rows, cols)?;
    let c = match &cdi {
        Some(a) => Some(sl(a)?),
        None => None,
    };
    Ok(PyArray1::from_vec(py, py.detach(|| commission::erode_commissons(p, f, w, c, erdpix, rows, cols))))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "fmask")?;
    m.add("__doc__", "The MATLAB primitives of the Fmask 4.7 port (for the comparison with GNU Octave)")?;
    m.add_function(wrap_pyfunction!(round_half_away, &m)?)?;
    m.add_function(wrap_pyfunction!(prctile, &m)?)?;
    m.add_function(wrap_pyfunction!(resize_box_half, &m)?)?;
    m.add_function(wrap_pyfunction!(resize_replicate3, &m)?)?;
    m.add_function(wrap_pyfunction!(stdfilt, &m)?)?;
    m.add_function(wrap_pyfunction!(strel_disk, &m)?)?;
    m.add_function(wrap_pyfunction!(morph_bin, &m)?)?;
    m.add_function(wrap_pyfunction!(morph_gray, &m)?)?;
    m.add_function(wrap_pyfunction!(bwlabel, &m)?)?;
    m.add_function(wrap_pyfunction!(bwareaopen, &m)?)?;
    m.add_function(wrap_pyfunction!(regionprops, &m)?)?;
    m.add_function(wrap_pyfunction!(imfill_gray, &m)?)?;
    m.add_function(wrap_pyfunction!(imfilter3, &m)?)?;
    m.add_function(wrap_pyfunction!(parameters, &m)?)?;
    m.add_function(wrap_pyfunction!(ndvi, &m)?)?;
    m.add_function(wrap_pyfunction!(ndsi, &m)?)?;
    m.add_function(wrap_pyfunction!(ndbi, &m)?)?;
    m.add_function(wrap_pyfunction!(saturate, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_snow, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_water, &m)?)?;
    m.add_function(wrap_pyfunction!(normaliza_cirrus_dem, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_potential_pixels, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_abs_snow, &m)?)?;
    m.add_function(wrap_pyfunction!(prob_thin, &m)?)?;
    m.add_function(wrap_pyfunction!(probw_brightness, &m)?)?;
    m.add_function(wrap_pyfunction!(probl_brightness, &m)?)?;
    m.add_function(wrap_pyfunction!(probl_spectral_variability, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_potential_cloud, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_potential_cloud_probs, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_abs_snow_scsi, &m)?)?;
    m.add_function(wrap_pyfunction!(cdi, &m)?)?;
    m.add_function(wrap_pyfunction!(enhance_line, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_potential_false_positive_pixels, &m)?)?;
    m.add_function(wrap_pyfunction!(erode_commissons, &m)?)?;
    Ok(m)
}
