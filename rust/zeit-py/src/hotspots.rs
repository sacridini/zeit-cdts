//! zeit._core.hotspots: the Python module of zeit-core's emerging hot spot analysis.

use numpy::{AllowTypeChange, PyArray1, PyArray3, PyArrayLike1, PyArrayLike3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::hotspots::{self, Params};
use zeit_core::mann_kendall::Method;

use crate::arrays::c_order;

/// Getis-Ord Gi* z-scores of a cube (time, y, x), NaN missing -> (time, y, x).
#[pyfunction]
#[pyo3(signature = (values, radius = 1, circle = false, lag = 1, cube_reference = false, n_jobs = -1))]
fn gi_star<'py>(
    py: Python<'py>,
    values: PyArrayLike3<'py, f64, AllowTypeChange>,
    radius: usize,
    circle: bool,
    lag: usize,
    cube_reference: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = values.as_array();
    let (nt, ny, nx) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py.detach(|| hotspots::gi_star(v, nt, ny, nx, radius, circle, lag, cube_reference, n_jobs));
    PyArray1::from_vec(py, out).reshape([nt, ny, nx])
}

/// The emerging hot spot analysis of a cube (time, y, x) -> dict of gi and p
/// (time, y, x), category (int8, y x), trend_z, trend_p, tau, n_hot, n_cold
/// (y, x). method: 0 Mann-Kendall, 1 Hamed-Rao, 2 Yue-Wang.
#[pyfunction]
#[pyo3(signature = (values, radius = 1, circle = false, lag = 1, cube_reference = false, alpha = 0.05, method = 1, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn emerging<'py>(
    py: Python<'py>,
    values: PyArrayLike3<'py, f64, AllowTypeChange>,
    radius: usize,
    circle: bool,
    lag: usize,
    cube_reference: bool,
    alpha: f64,
    method: i32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err(format!("alpha must be in (0, 1), got {alpha}")));
    }
    if !(0..=2).contains(&method) {
        return Err(PyValueError::new_err(format!("unknown trend method {method}")));
    }
    let view = values.as_array();
    let (nt, ny, nx) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let p = Params {
        radius,
        circle,
        lag,
        cube_reference,
        alpha,
        method: Method::from_code(method),
    };
    let e = py.detach(|| hotspots::emerging(v, nt, ny, nx, &p, n_jobs));
    let d = PyDict::new(py);
    d.set_item("gi", PyArray1::from_vec(py, e.gi).reshape([nt, ny, nx])?)?;
    d.set_item("p", PyArray1::from_vec(py, e.p).reshape([nt, ny, nx])?)?;
    d.set_item("category", PyArray1::from_vec(py, e.category).reshape([ny, nx])?)?;
    d.set_item("trend_z", PyArray1::from_vec(py, e.trend_z).reshape([ny, nx])?)?;
    d.set_item("trend_p", PyArray1::from_vec(py, e.trend_p).reshape([ny, nx])?)?;
    d.set_item("tau", PyArray1::from_vec(py, e.tau).reshape([ny, nx])?)?;
    d.set_item("n_hot", PyArray1::from_vec(py, e.n_hot).reshape([ny, nx])?)?;
    d.set_item("n_cold", PyArray1::from_vec(py, e.n_cold).reshape([ny, nx])?)?;
    Ok(d)
}

/// The pattern of each Gi* series (pixels, time) given its trend's sign and
/// significance -> int8 codes.
#[pyfunction]
#[pyo3(signature = (gi, trend, significant, alpha = 0.05))]
fn classify<'py>(
    py: Python<'py>,
    gi: numpy::PyArrayLike2<'py, f64, AllowTypeChange>,
    trend: PyArrayLike1<'py, i32, AllowTypeChange>,
    significant: PyArrayLike1<'py, bool, AllowTypeChange>,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let view = gi.as_array();
    let (n, nt) = view.dim();
    let (trend, sig) = (trend.as_array(), significant.as_array());
    if trend.len() != n || sig.len() != n {
        return Err(PyValueError::new_err("one trend and one significance per series"));
    }
    let v = c_order(&view);
    let out: Vec<i8> = (0..n)
        .map(|i| hotspots::classify(&v[i * nt..(i + 1) * nt], alpha, trend[i], sig[i]))
        .collect();
    Ok(PyArray1::from_vec(py, out))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "hotspots")?;
    m.add(
        "__doc__",
        "Emerging hot spot analysis: Getis-Ord Gi* and Mann-Kendall (ESRI's patterns)",
    )?;
    m.add_function(wrap_pyfunction!(gi_star, &m)?)?;
    m.add_function(wrap_pyfunction!(emerging, &m)?)?;
    m.add_function(wrap_pyfunction!(classify, &m)?)?;
    Ok(m)
}
