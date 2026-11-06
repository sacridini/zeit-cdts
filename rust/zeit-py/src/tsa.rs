//! zeit._core.tsa: the Python module (statistics of series and their folds,
//! standardization, interpolation,
//! linear trend and greatest change).

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArray3, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::tsa::interp::{self, Method};
use zeit_core::tsa::stats::{self, Metric, Standardize};
use zeit_core::tsa::trend::{self, Tail, CAT_LEN, TREND_LEN};

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

/// Every series of values (pixels, n_in) at the times `t_out` (days): (pixels,
/// n_out). `method`: "linear", "moving" (`window`), "rbf" (`sigma`, `cutoff`)
/// or "harmonic" (`modes`, `trend`, the fit range `fit_lo`..`fit_hi`, `period`).
/// `t_in` increases; NaN marks a missing observation.
#[pyfunction]
#[pyo3(signature = (values, t_in, t_out, method = "rbf", sigma = vec![8.0, 16.0, 32.0], cutoff = 0.95, window = 16.0,
                    modes = 3, trend = true, fit_lo = f64::NEG_INFINITY, fit_hi = f64::INFINITY, period = 365.25,
                    n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn interp_batch<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    t_in: PyArrayLike1<'py, f64, AllowTypeChange>,
    t_out: PyArrayLike1<'py, f64, AllowTypeChange>,
    method: &str,
    sigma: Vec<f64>,
    cutoff: f64,
    window: f64,
    modes: usize,
    trend: bool,
    fit_lo: f64,
    fit_hi: f64,
    period: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let method = match method.trim().to_ascii_lowercase().as_str() {
        "linear" => Method::Linear,
        "moving" => Method::Moving { window },
        "rbf" => Method::Rbf { sigma, cutoff },
        "harmonic" => Method::Harmonic {
            modes,
            trend,
            fit: (fit_lo, fit_hi),
            period,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}: use \"linear\", \"moving\", \"rbf\" or \"harmonic\""
            )))
        }
    };
    let view = values.as_array();
    let (n_pixels, n_in) = view.dim();
    let t_in: Vec<f64> = t_in.as_array().iter().copied().collect();
    let t_out: Vec<f64> = t_out.as_array().iter().copied().collect();
    let n_out = t_out.len();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py
        .detach(|| interp::interp_batch(v, n_pixels, n_in, &t_in, &t_out, &method, n_jobs))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, n_out])
}

/// The linear trend of every series of values (pixels, steps; NaN marks a
/// missing step), x being the step: (pixels, 12) with mean, offset, slope,
/// pct_gain, abs_gain, rsq (0-1), sig (-1, 0, 1), rmse, mae, maxe, num, len.
#[pyfunction]
#[pyo3(signature = (values, conf = 0.95, tail = "two", n_jobs = -1))]
fn trend_batch<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    conf: f64,
    tail: &str,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tail = Tail::parse(tail).map_err(PyValueError::new_err)?;
    let view = values.as_array();
    let (n_pixels, n_time) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py
        .detach(|| trend::trend_batch(v, n_pixels, n_time, conf, tail, n_jobs))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, TREND_LEN])
}

/// The greatest change of every series of values (pixels, steps) and the trend
/// of the whole series, before the change and after it: (pixels, 39) with
/// change, loss, change_index, then the 12 values of the trend for the total,
/// before and after parts.
#[pyfunction]
#[pyo3(signature = (values, penalty = false, conf = 0.95, tail = "two", n_jobs = -1))]
fn cat_batch<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    penalty: bool,
    conf: f64,
    tail: &str,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tail = Tail::parse(tail).map_err(PyValueError::new_err)?;
    let view = values.as_array();
    let (n_pixels, n_time) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let out = py
        .detach(|| trend::cat_batch(v, n_pixels, n_time, penalty, conf, tail, n_jobs))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, CAT_LEN])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "tsa")?;
    m.add("__doc__", "Statistics of time series and their folds, standardization, interpolation")?;
    m.add_function(wrap_pyfunction!(stm_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fold_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(standardize_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(interp_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(trend_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(cat_batch, &m)?)?;
    Ok(m)
}
