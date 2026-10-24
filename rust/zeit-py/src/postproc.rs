//! zeit._core.postproc: the Python module of zeit-core's post-processing of classifications.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArray3, PyArrayLike1, PyArrayLike2, PyArrayLike3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::postproc;

use crate::arrays::c_order;

/// sits' Bayesian smoothing of probabilities (class, y, x), with one
/// smoothness per class -> the smoothed probabilities (class, y, x).
#[pyfunction]
#[pyo3(signature = (probs, window_size, smoothness, neigh_fraction, n_jobs = -1))]
fn smooth_probs<'py>(
    py: Python<'py>,
    probs: PyArrayLike3<'py, f64, AllowTypeChange>,
    window_size: usize,
    smoothness: PyArrayLike1<'py, f64, AllowTypeChange>,
    neigh_fraction: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = probs.as_array();
    let (nc, ny, nx) = view.dim();
    let smoothness: Vec<f64> = smoothness.as_array().to_vec();
    if smoothness.len() != nc {
        return Err(PyValueError::new_err(format!(
            "{} smoothness values for {nc} classes",
            smoothness.len()
        )));
    }
    if window_size < 3 || window_size.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "window_size must be odd and at least 3, got {window_size}"
        )));
    }
    if !(neigh_fraction > 0.0 && neigh_fraction <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "neigh_fraction must be in (0, 1], got {neigh_fraction}"
        )));
    }
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out =
        py.detach(|| postproc::smooth_probs(values, nc, ny, nx, window_size, &smoothness, neigh_fraction, n_jobs));
    PyArray1::from_vec(py, out).reshape([nc, ny, nx])
}

/// Majority filter of a class map (y, x), NaN missing.
#[pyfunction]
#[pyo3(signature = (classes, window_size = 3, iterations = 1, n_jobs = -1))]
fn majority<'py>(
    py: Python<'py>,
    classes: PyArrayLike2<'py, f64, AllowTypeChange>,
    window_size: usize,
    iterations: usize,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = classes.as_array();
    let (ny, nx) = view.dim();
    if window_size < 1 || window_size.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "window_size must be odd, got {window_size}"
        )));
    }
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| postproc::majority(values, ny, nx, window_size, iterations, n_jobs));
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// Minimum mapping unit: the patches of a class map (y, x) smaller than
/// min_size merged into the neighbouring patch of the longest border.
#[pyfunction]
#[pyo3(signature = (classes, min_size, eight = true))]
fn sieve<'py>(
    py: Python<'py>,
    classes: PyArrayLike2<'py, f64, AllowTypeChange>,
    min_size: usize,
    eight: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = classes.as_array();
    let (ny, nx) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| postproc::sieve(values, ny, nx, min_size, eight));
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// The short excursions of class series (pixels, dates) removed: A B A -> A A A.
#[pyfunction]
#[pyo3(signature = (classes, max_length = 1, n_jobs = -1))]
fn desawtooth<'py>(
    py: Python<'py>,
    classes: PyArrayLike2<'py, f64, AllowTypeChange>,
    max_length: usize,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = classes.as_array();
    let (n, t) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| postproc::desawtooth(values, n, t, max_length, n_jobs));
    PyArray1::from_vec(py, out).reshape([n, t])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "postproc")?;
    m.add(
        "__doc__",
        "Post-processing of classifications: sits' Bayesian smoothing, majority, sieve, desawtooth",
    )?;
    m.add_function(wrap_pyfunction!(smooth_probs, &m)?)?;
    m.add_function(wrap_pyfunction!(majority, &m)?)?;
    m.add_function(wrap_pyfunction!(sieve, &m)?)?;
    m.add_function(wrap_pyfunction!(desawtooth, &m)?)?;
    Ok(m)
}
