//! zeit._rust.smooth: the bindings of zeit._core.smooth.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::whittaker as wh;

use crate::arrays::c_order;

/// Whittaker smoother of one series at positions x (divided differences; NaN weighs 0).
#[pyfunction]
fn whittaker(py: Python<'_>, x: Vec<f64>, y: Vec<f64>, w: Vec<f64>, lmbda: f64) -> PyResult<Vec<f64>> {
    if x.len() != y.len() || w.len() != y.len() {
        return Err(PyValueError::new_err("whittaker: x, y and w must have the same length"));
    }
    py.detach(|| wh::whittaker(&x, &y, &w, lmbda))
        .map_err(PyValueError::new_err)
}

/// Whittaker smoother of (pixels, time) values at positions x; weights
/// (pixels, time) or empty.
#[pyfunction]
#[pyo3(signature = (values, x, weights, lmbda, n_jobs = -1))]
fn whittaker_batch<'py>(
    py: Python<'py>,
    values: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    x: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    weights: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    lmbda: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (v, x, w) = (values.as_array(), x.as_array(), weights.as_array());
    if v.ndim() != 2 || x.ndim() != 1 {
        return Err(PyValueError::new_err("values (pixels, time), x (time,)"));
    }
    let (pixels, t) = (v.shape()[0], v.shape()[1]);
    if x.len() != t {
        return Err(PyValueError::new_err("one x per time step"));
    }
    let has_w = !w.is_empty();
    if has_w && (w.ndim() != 2 || w.shape() != v.shape()) {
        return Err(PyValueError::new_err("weights must be (pixels, time), as values"));
    }
    let (v, x, w) = (c_order(&v), c_order(&x), c_order(&w));
    let (v, x, w): (&[f64], &[f64], &[f64]) = (&v, &x, &w);
    let out = py
        .detach(|| wh::batch(v, pixels, x, has_w.then_some(w), lmbda, n_jobs))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([pixels, t])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "smooth")?;
    m.add("__doc__", "Smoothers of time series")?;
    m.add_function(wrap_pyfunction!(whittaker, &m)?)?;
    m.add_function(wrap_pyfunction!(whittaker_batch, &m)?)?;
    Ok(m)
}
