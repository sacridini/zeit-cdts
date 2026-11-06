//! zeit._core.tsa: the Python module (statistics of series and their folds,
//! standardization).

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArray3, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::tsa::stats::{self, Metric, Standardize};

use crate::arrays::c_order;

fn metrics_of(names: &[String]) -> PyResult<Vec<Metric>> {
    if names.is_empty() {
        return Err(PyValueError::new_err("provide at least one metric"));
    }
    Metric::parse_all(names).map_err(PyValueError::new_err)
}

/// The metrics of every series of values (pixels, time), over the whole time
/// axis: (pixels, metrics). `metrics`: min, max, range, iqr, mean, std, skew,
/// kurtosis, count, median, pNN. NaN marks a missing observation.
#[pyfunction]
#[pyo3(signature = (values, metrics, ddof = 1.0, n_jobs = -1))]
fn stm_batch<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    metrics: Vec<String>,
    ddof: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let metrics = metrics_of(&metrics)?;
    let view = values.as_array();
    let (n_pixels, n_time) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py.detach(|| stats::stm_batch(v, n_pixels, n_time, &metrics, ddof, n_jobs));
    PyArray1::from_vec(py, out).reshape([n_pixels, metrics.len()])
}

/// The metrics of every series of values (pixels, time) for each group of time
/// steps: (pixels, metrics, groups). group[t] in 0..n_groups, or -1 to leave
/// step t out.
#[pyfunction]
#[pyo3(signature = (values, group, n_groups, metrics, ddof = 1.0, n_jobs = -1))]
fn fold_batch<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    group: PyArrayLike1<'py, i32, AllowTypeChange>,
    n_groups: usize,
    metrics: Vec<String>,
    ddof: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let metrics = metrics_of(&metrics)?;
    let view = values.as_array();
    let (n_pixels, n_time) = view.dim();
    let group: Vec<i32> = group.as_array().iter().copied().collect();
    if group.len() != n_time {
        return Err(PyValueError::new_err("one group per time step"));
    }
    if group.iter().any(|&g| g < -1 || g >= n_groups as i32) {
        return Err(PyValueError::new_err("group must be -1 or in 0..n_groups"));
    }
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py.detach(|| stats::fold_batch(v, n_pixels, n_time, &group, n_groups, &metrics, ddof, n_jobs));
    PyArray1::from_vec(py, out).reshape([n_pixels, metrics.len(), n_groups])
}

/// Every series of values (pixels, time) centered ("center") or normalized
/// ("normalize") along time; fewer than 2 valid values stay as they are.
#[pyfunction]
#[pyo3(signature = (values, method = "normalize", ddof = 1.0, n_jobs = -1))]
fn standardize_batch<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    method: &str,
    ddof: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let method = Standardize::parse(method).map_err(PyValueError::new_err)?;
    let view = values.as_array();
    let (n_pixels, n_time) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py.detach(|| stats::standardize_batch(v, n_pixels, n_time, method, ddof, n_jobs));
    PyArray1::from_vec(py, out).reshape([n_pixels, n_time])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "tsa")?;
    m.add("__doc__", "Statistics of time series and their folds, standardization")?;
    m.add_function(wrap_pyfunction!(stm_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fold_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(standardize_batch, &m)?)?;
    Ok(m)
}
