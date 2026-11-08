//! zeit._core.fmask_shadow: a minimal binding of the cloud shadow step of the Fmask 4.7 port
//! (`zeit_core::fmask::shadow`), for the comparison with GNU Octave (`benchmarks/vs_originals/fmask4_shadow.py`). No
//! public API. Images are 1-D arrays of `rows * cols` in row-major order.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use zeit_core::fmask::shadow::{self, MatchInput};

fn check(len: usize, rows: usize, cols: usize) -> PyResult<()> {
    if len != rows * cols {
        return Err(PyValueError::new_err("the array does not have rows * cols values"));
    }
    Ok(())
}

fn sl<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err("arrays must be contiguous"))
}

#[pyfunction]
fn colon_f64<'py>(py: Python<'py>, base: f64, inc: f64, limit: f64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, shadow::colon_f64(base, inc, limit))
}

#[pyfunction]
fn colon_f32<'py>(py: Python<'py>, base: f32, inc: f32, limit: f32) -> Bound<'py, PyArray1<f32>> {
    PyArray1::from_vec(py, shadow::colon_f32(base, inc, limit))
}

/// `fmask_randperm(n, k)`, 0-based.
#[pyfunction]
fn randperm<'py>(py: Python<'py>, n: usize, k: usize) -> Bound<'py, PyArray1<u64>> {
    PyArray1::from_vec(py, py.detach(|| shadow::fmask_randperm(n, k)).into_iter().map(|v| v as u64).collect())
}

#[pyfunction]
fn polyfit1(x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    let (a, b) = (sl(&x)?, sl(&y)?);
    if a.len() != b.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    Ok(shadow::polyfit1(a, b))
}

#[pyfunction]
fn stratied_sample<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f32>, dem_b: f32, dem_t: f32, total_sample: f64, ele_strata: f64) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let d = sl(&data)?;
    let r = py.detach(|| shadow::stratied_sample_handler(d, dem_b, dem_t, total_sample, ele_strata));
    Ok(PyArray1::from_vec(py, r.into_iter().map(|v| v as u64).collect()))
}

/// `DetectPotentialCloudShadow`; returns the mask and `[c_nir, c_swir, applied_nir, applied_swir, n_samples]` (empty
/// without the topographic correction).
#[pyfunction]
#[pyo3(signature = (rows, cols, nir, swir, clear_land, observation, zen, azi, slope = None, aspect = None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn detect_potential_cloud_shadow<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    nir: PyReadonlyArray1<'py, f32>,
    swir: PyReadonlyArray1<'py, f32>,
    clear_land: PyReadonlyArray1<'py, u8>,
    observation: PyReadonlyArray1<'py, u8>,
    zen: f64,
    azi: f64,
    slope: Option<PyReadonlyArray1<'py, f32>>,
    aspect: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<f64>>)> {
    let (n, s, c, o) = (sl(&nir)?, sl(&swir)?, sl(&clear_land)?, sl(&observation)?);
    for a in [n.len(), s.len(), c.len(), o.len()] {
        check(a, rows, cols)?;
    }
    let topo = match (&slope, &aspect) {
        (Some(a), Some(b)) => {
            let (a, b) = (sl(a)?, sl(b)?);
            check(a.len(), rows, cols)?;
            check(b.len(), rows, cols)?;
            Some((a, b))
        }
        _ => None,
    };
    let (m, info) = py.detach(|| shadow::detect_potential_cloud_shadow(rows, cols, n, s, c, o, topo, zen, azi));
    let info = match info {
        Some(i) => vec![i.c_nir, i.c_swir, i.applied_nir as u8 as f64, i.applied_swir as u8 as f64, i.n_samples as f64],
        None => Vec::new(),
    };
    Ok((PyArray1::from_vec(py, m), PyArray1::from_vec(py, info)))
}

/// `ProjectDEM2Plane`; returns the size of the plane and its two recorders (row-major).
#[pyfunction]
#[pyo3(signature = (rows, cols, dem, dem_base, elev_rad, tazi_rad, azi_deg, dim_expd, resolution))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn project_dem_to_plane<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    dem: Option<PyReadonlyArray1<'py, f32>>,
    dem_base: f64,
    elev_rad: f64,
    tazi_rad: f64,
    azi_deg: f64,
    dim_expd: usize,
    resolution: f64,
) -> PyResult<((usize, usize), Bound<'py, PyArray1<u16>>, Bound<'py, PyArray1<u16>>)> {
    let d = match &dem {
        Some(a) => {
            let s = sl(a)?;
            check(s.len(), rows, cols)?;
            Some(s)
        }
        None => None,
    };
    let p = py.detach(|| shadow::project_dem_to_plane(rows, cols, d, dem_base, elev_rad, tazi_rad, azi_deg, dim_expd, resolution));
    Ok(((p.rows, p.cols), PyArray1::from_vec(py, p.row), PyArray1::from_vec(py, p.col)))
}

#[pyfunction]
fn real_position<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    h: f32,
    vzaxy: f32,
    vaaxy: f32,
    resolution: f64,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let (x, y) = (sl(&x)?, sl(&y)?);
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let (mut xn, mut yn) = (Vec::with_capacity(x.len()), Vec::with_capacity(x.len()));
    for k in 0..x.len() {
        let (a, b) = shadow::get_real_cloud_position_s2(x[k], y[k], h, vzaxy, vaaxy, resolution);
        xn.push(a);
        yn.push(b);
    }
    Ok((PyArray1::from_vec(py, xn), PyArray1::from_vec(py, yn)))
}

/// `MatchCloudShadow`; returns `(skipped, num_clouds, similar_num, cloud, shadow, events)`, `events` flat with five values
/// per match: `cloud_type_cur, cloud_type, record_base_h, record_thresh, record_base_h_near`.
#[pyfunction]
#[pyo3(signature = (rows, cols, mask, plcim, plsim, water_all, vza, vaa, zen, azi, ptm, is_shadow_water = false, dem = None, resolution = 20.0, num_near = 14))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn match_cloud_shadow<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    mask: PyReadonlyArray1<'py, u8>,
    plcim: PyReadonlyArray1<'py, u8>,
    plsim: PyReadonlyArray1<'py, u8>,
    water_all: PyReadonlyArray1<'py, u8>,
    vza: PyReadonlyArray1<'py, f32>,
    vaa: PyReadonlyArray1<'py, f32>,
    zen: f64,
    azi: f64,
    ptm: f64,
    is_shadow_water: bool,
    dem: Option<PyReadonlyArray1<'py, f32>>,
    resolution: f64,
    num_near: usize,
) -> PyResult<(bool, usize, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<f64>>)> {
    let (m, c, s, w, va, vz) = (sl(&mask)?, sl(&plcim)?, sl(&plsim)?, sl(&water_all)?, sl(&vaa)?, sl(&vza)?);
    for a in [m.len(), c.len(), s.len(), w.len(), va.len(), vz.len()] {
        check(a, rows, cols)?;
    }
    let d = match &dem {
        Some(a) => {
            let s = sl(a)?;
            check(s.len(), rows, cols)?;
            Some(s)
        }
        None => None,
    };
    let o = py.detach(|| {
        shadow::match_cloud_shadow(&MatchInput {
            rows,
            cols,
            mask: m,
            plcim: c,
            plsim: s,
            is_shadow_water,
            water_all: w,
            dem: d,
            sun_zenith_deg: zen,
            sun_azimuth_deg: azi,
            resolution,
            ptm,
            num_near,
            vza: vz,
            vaa: va,
        })
    });
    let ev: Vec<f64> = o
        .events
        .iter()
        .flat_map(|e| [e.cloud_type_cur as f64, e.cloud_type as f64, e.record_base_h, e.record_thresh, e.record_base_h_near])
        .collect();
    Ok((o.skipped, o.num_clouds, PyArray1::from_vec(py, o.similar_num), PyArray1::from_vec(py, o.cloud_matched), PyArray1::from_vec(py, o.shadow_matched), PyArray1::from_vec(py, ev)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn buffer_masks<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    pcloud: PyReadonlyArray1<'py, u8>,
    cdpix: usize,
    pshadow: PyReadonlyArray1<'py, u8>,
    csdpix: usize,
    psnow: PyReadonlyArray1<'py, u8>,
    sdpix: usize,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>)> {
    let (a, b, c) = (sl(&pcloud)?, sl(&pshadow)?, sl(&psnow)?);
    for l in [a.len(), b.len(), c.len()] {
        check(l, rows, cols)?;
    }
    let (x, y, z) = py.detach(|| shadow::buffer_masks(rows, cols, a, cdpix, b, csdpix, c, sdpix));
    Ok((PyArray1::from_vec(py, x), PyArray1::from_vec(py, y), PyArray1::from_vec(py, z)))
}

/// `knn_cityblock`: the `k` nearest of the points `cents` (flat x, y pairs) to `(qx, qy)`; returns the 0-based indices and
/// the distances.
#[pyfunction]
fn knn_cityblock<'py>(py: Python<'py>, cents: PyReadonlyArray1<'py, f64>, qx: f64, qy: f64, k: usize) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f64>>)> {
    let c = sl(&cents)?;
    if c.len() % 2 != 0 {
        return Err(PyValueError::new_err("cents must hold x, y pairs"));
    }
    let pts: Vec<(f64, f64)> = c.chunks(2).map(|p| (p[0], p[1])).collect();
    let r = shadow::knn_cityblock(&pts, (qx, qy), k);
    Ok((PyArray1::from_vec(py, r.iter().map(|v| v.0 as u64).collect()), PyArray1::from_vec(py, r.iter().map(|v| v.1).collect())))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "fmask_shadow")?;
    m.add("__doc__", "The cloud shadow step of the Fmask 4.7 port (for the comparison with GNU Octave)")?;
    m.add_function(wrap_pyfunction!(colon_f64, &m)?)?;
    m.add_function(wrap_pyfunction!(colon_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(randperm, &m)?)?;
    m.add_function(wrap_pyfunction!(polyfit1, &m)?)?;
    m.add_function(wrap_pyfunction!(stratied_sample, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_potential_cloud_shadow, &m)?)?;
    m.add_function(wrap_pyfunction!(project_dem_to_plane, &m)?)?;
    m.add_function(wrap_pyfunction!(real_position, &m)?)?;
    m.add_function(wrap_pyfunction!(match_cloud_shadow, &m)?)?;
    m.add_function(wrap_pyfunction!(buffer_masks, &m)?)?;
    m.add_function(wrap_pyfunction!(knn_cityblock, &m)?)?;
    Ok(m)
}
