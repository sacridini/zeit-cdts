//! zeit._core.tsa: the Python module (statistics of series and their folds,
//! standardization, interpolation,
//! linear trend and greatest change, clear observations, polar phenology metrics).

use numpy::{
    AllowTypeChange, PyArray1, PyArray2, PyArray3, PyArrayLike1, PyArrayLike2, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::tsa::cso;
use zeit_core::tsa::interp::{self, Method};
use zeit_core::tsa::polar;
use zeit_core::tsa::stats::{self, Metric, Standardize};
use zeit_core::tsa::trend::{self, Tail, CAT_LEN, TREND_LEN};

use crate::arrays::c_order;

/// The (pixels, time) cube of the functions that read a whole one: a float32
/// array is read as it is (the work is in float64 either way, but the cube is
/// not converted), any other dtype is converted to float64.
#[derive(FromPyObject)]
enum Cube<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyArrayLike2<'py, f64, AllowTypeChange>),
}

impl Cube<'_> {
    fn dim(&self) -> (usize, usize) {
        match self {
            Cube::F32(a) => (a.shape()[0], a.shape()[1]),
            Cube::F64(a) => a.as_array().dim(),
        }
    }
}

/// Runs `$body` with `$v` the cube's values as a C-ordered slice (f32 or f64).
macro_rules! with_cube {
    ($cube:expr, $v:ident => $body:expr) => {
        match &$cube {
            Cube::F32(a) => {
                let view = a.as_array();
                let c = c_order(&view);
                let $v: &[f32] = &c;
                $body
            }
            Cube::F64(a) => {
                let view = a.as_array();
                let c = c_order(&view);
                let $v: &[f64] = &c;
                $body
            }
        }
    };
}

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
    values: Cube<'py>,
    metrics: Vec<String>,
    ddof: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let metrics = metrics_of(&metrics)?;
    let (n_pixels, n_time) = values.dim();
    let out = with_cube!(values, v => py.detach(|| stats::stm_batch(v, n_pixels, n_time, &metrics, ddof, n_jobs)));
    PyArray1::from_vec(py, out).reshape([n_pixels, metrics.len()])
}

/// The metrics of every series of values (pixels, time) for each group of time
/// steps: (pixels, metrics, groups). group[t] in 0..n_groups, or -1 to leave
/// step t out.
#[pyfunction]
#[pyo3(signature = (values, group, n_groups, metrics, ddof = 1.0, n_jobs = -1))]
fn fold_batch<'py>(
    py: Python<'py>,
    values: Cube<'py>,
    group: PyArrayLike1<'py, i32, AllowTypeChange>,
    n_groups: usize,
    metrics: Vec<String>,
    ddof: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let metrics = metrics_of(&metrics)?;
    let (n_pixels, n_time) = values.dim();
    let group: Vec<i32> = group.as_array().iter().copied().collect();
    if group.len() != n_time {
        return Err(PyValueError::new_err("one group per time step"));
    }
    if group.iter().any(|&g| g < -1 || g >= n_groups as i32) {
        return Err(PyValueError::new_err("group must be -1 or in 0..n_groups"));
    }
    let out = with_cube!(values, v => py.detach(|| {
        stats::fold_batch(v, n_pixels, n_time, &group, n_groups, &metrics, ddof, n_jobs)
    }));
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
    values: Cube<'py>,
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
    let (n_pixels, n_in) = values.dim();
    let t_in: Vec<f64> = t_in.as_array().iter().copied().collect();
    let t_out: Vec<f64> = t_out.as_array().iter().copied().collect();
    let n_out = t_out.len();
    let out = with_cube!(values, v => py
        .detach(|| interp::interp_batch(v, n_pixels, n_in, &t_in, &t_out, &method, n_jobs)))
    .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, n_out])
}

/// How the robust fit of every series of values (pixels, n_in) ends under
/// `interp_batch`'s harmonic method: the iterations of the reweighted least
/// squares (1..=100), 65535 when it ran 100 without converging, 0 when not fit.
#[pyfunction]
#[pyo3(signature = (values, t_in, modes = 3, trend = true, fit_lo = f64::NEG_INFINITY, fit_hi = f64::INFINITY,
                    period = 365.25, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn harmonic_iterations<'py>(
    py: Python<'py>,
    values: Cube<'py>,
    t_in: PyArrayLike1<'py, f64, AllowTypeChange>,
    modes: usize,
    trend: bool,
    fit_lo: f64,
    fit_hi: f64,
    period: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let (n_pixels, n_in) = values.dim();
    let t_in: Vec<f64> = t_in.as_array().iter().copied().collect();
    let out = with_cube!(values, v => py.detach(|| {
        interp::harmonic_iterations(v, n_pixels, n_in, &t_in, modes, trend, (fit_lo, fit_hi), period, n_jobs)
    }))
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, out))
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

/// The clear observations of every series of values (pixels, time; NaN marks a
/// missing observation) in each window [start, end) of `windows` (an array
/// (windows, 2) in the unit of `t`, days): (pixels, metrics, windows). `t`
/// increases. `count` is the valid observations, the other metrics are those of
/// the intervals between them.
#[pyfunction]
#[pyo3(signature = (values, t, windows, metrics, n_jobs = -1))]
fn cso_batch<'py>(
    py: Python<'py>,
    values: Cube<'py>,
    t: PyArrayLike1<'py, f64, AllowTypeChange>,
    windows: PyArrayLike2<'py, f64, AllowTypeChange>,
    metrics: Vec<String>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let metrics = metrics_of(&metrics)?;
    let (n_pixels, n_time) = values.dim();
    let t: Vec<f64> = t.as_array().iter().copied().collect();
    let w = windows.as_array();
    if w.ncols() != 2 {
        return Err(PyValueError::new_err("windows must be (n, 2): the start and the end of each"));
    }
    let windows: Vec<(f64, f64)> = w.rows().into_iter().map(|r| (r[0], r[1])).collect();
    let n_windows = windows.len();
    let out = with_cube!(values, v => py
        .detach(|| cso::cso_batch(v, n_pixels, n_time, &t, &windows, &metrics, n_jobs)))
    .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, metrics.len(), n_windows])
}

/// The polar phenology metrics of every series of values (pixels, steps; NaN marks a missing
/// step) on the days `ce` (a calendar of 365 days counted from the first year: `year * 365 +
/// doy`, the first year 0): (pixels, metrics, years). `step` is the nominal interval in days;
/// `start`, `mid` and `end` the fractions of a season's accumulated value that mark its start,
/// middle and end. `metrics` are FORCE's names (DEM, DSS, VSS...). A band without a season is NaN.
#[pyfunction]
#[pyo3(signature = (values, ce, step, metrics, start = 0.2, mid = 0.5, end = 0.8, adaptive = true, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn polar_batch<'py>(
    py: Python<'py>,
    values: Cube<'py>,
    ce: PyArrayLike1<'py, i64, AllowTypeChange>,
    step: i32,
    metrics: Vec<String>,
    start: f64,
    mid: f64,
    end: f64,
    adaptive: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    if metrics.is_empty() {
        return Err(PyValueError::new_err("provide at least one metric"));
    }
    let metrics = polar::parse_all(&metrics).map_err(PyValueError::new_err)?;
    let (n_pixels, n_time) = values.dim();
    let ce: Vec<i64> = ce.as_array().iter().copied().collect();
    let (out, ny) = with_cube!(values, v => py.detach(|| {
        polar::polar_batch(v, n_pixels, n_time, &ce, step, start, mid, end, adaptive, &metrics, n_jobs)
    }))
    .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, metrics.len(), ny])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "tsa")?;
    m.add("__doc__", "Statistics of time series and their folds, standardization, interpolation")?;
    m.add_function(wrap_pyfunction!(stm_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fold_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(standardize_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(interp_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_iterations, &m)?)?;
    m.add_function(wrap_pyfunction!(trend_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(cat_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(cso_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(polar_batch, &m)?)?;
    Ok(m)
}
