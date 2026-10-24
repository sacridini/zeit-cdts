//! zeit._core.ewmacd: the Python module of zeit-core's EWMACD.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::ewmacd::{self, Params};

use crate::arrays::c_order;

#[allow(clippy::too_many_arguments)]
fn params(
    train_start: i32,
    train_end: i32,
    harmonics: usize,
    xbar_train: f64,
    xbar_monitor: f64,
    lmbda: f64,
    sigmas: f64,
    rounding: bool,
    persistence: usize,
    low: f64,
) -> PyResult<Params> {
    if !(lmbda > 0.0 && lmbda <= 1.0) {
        return Err(PyValueError::new_err(format!("lmbda must be in (0, 1], got {lmbda}")));
    }
    if harmonics == 0 {
        return Err(PyValueError::new_err("harmonics must be at least 1"));
    }
    if train_end <= train_start {
        return Err(PyValueError::new_err(format!(
            "train_end ({train_end}) must come after train_start ({train_start})"
        )));
    }
    Ok(Params {
        train_start,
        train_end,
        harmonics,
        xbar_train,
        xbar_monitor,
        lambda: lmbda,
        sigmas,
        rounding,
        persistence: persistence.max(1),
        low,
    })
}

fn dates(years: &[i32], doys: &[f64], times: &[f64], n: usize) -> PyResult<()> {
    if years.len() != n || doys.len() != n || times.len() != n {
        return Err(PyValueError::new_err(format!(
            "{} years, {} days and {} times for {n} dates",
            years.len(),
            doys.len(),
            times.len()
        )));
    }
    if times.windows(2).any(|w| w[1] < w[0]) {
        return Err(PyValueError::new_err("the dates must be in order"));
    }
    Ok(())
}

/// EWMACD of every pixel of values_array (pixels, dates), the dates by their
/// calendar years, days of the year and times (decimal years) -> flags
/// (pixels, dates) and maps (NAMES then the coefficients, pixels).
#[pyfunction]
#[pyo3(signature = (values_array, years, doys, times, train_start, train_end, harmonics = 2, xbar_train = 1.5,
                    xbar_monitor = 20.0, lmbda = 0.3, sigmas = 3.0, rounding = true, persistence = 3,
                    low = f64::NEG_INFINITY, n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ewmacd_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    years: PyArrayLike1<'py, i32, AllowTypeChange>,
    doys: PyArrayLike1<'py, f64, AllowTypeChange>,
    times: PyArrayLike1<'py, f64, AllowTypeChange>,
    train_start: i32,
    train_end: i32,
    harmonics: usize,
    xbar_train: f64,
    xbar_monitor: f64,
    lmbda: f64,
    sigmas: f64,
    rounding: bool,
    persistence: usize,
    low: f64,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let p = params(
        train_start,
        train_end,
        harmonics,
        xbar_train,
        xbar_monitor,
        lmbda,
        sigmas,
        rounding,
        persistence,
        low,
    )?;
    let view = values_array.as_array();
    let (n_pixels, n_dates) = view.dim();
    let years: Vec<i32> = years.as_array().to_vec();
    let doys: Vec<f64> = doys.as_array().to_vec();
    let times: Vec<f64> = times.as_array().to_vec();
    dates(&years, &doys, &times, n_dates)?;
    let values = c_order(&view);
    let values: &[f64] = &values;
    let (flags, maps) = py.detach(|| ewmacd::batch(values, n_pixels, &years, &doys, &times, &p, n_jobs));
    let rows = ewmacd::NAMES.len() + p.n_coefficients();
    Ok((
        PyArray1::from_vec(py, flags).reshape([n_pixels, n_dates])?,
        PyArray1::from_vec(py, maps).reshape([rows, n_pixels])?,
    ))
}

/// EWMACD of one series, everything its chart shows: flags, ewma, limit,
/// fitted (per date), charted (per date), beta, sigma, rmse, r2 and the
/// events (break_time, severity, recovery_time, gain_time, state).
#[pyfunction]
#[pyo3(signature = (values, years, doys, times, train_start, train_end, harmonics = 2, xbar_train = 1.5,
                    xbar_monitor = 20.0, lmbda = 0.3, sigmas = 3.0, rounding = true, persistence = 3,
                    low = f64::NEG_INFINITY))]
#[allow(clippy::too_many_arguments)]
fn ewmacd_pixel<'py>(
    py: Python<'py>,
    values: PyArrayLike1<'py, f64, AllowTypeChange>,
    years: PyArrayLike1<'py, i32, AllowTypeChange>,
    doys: PyArrayLike1<'py, f64, AllowTypeChange>,
    times: PyArrayLike1<'py, f64, AllowTypeChange>,
    train_start: i32,
    train_end: i32,
    harmonics: usize,
    xbar_train: f64,
    xbar_monitor: f64,
    lmbda: f64,
    sigmas: f64,
    rounding: bool,
    persistence: usize,
    low: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = params(
        train_start,
        train_end,
        harmonics,
        xbar_train,
        xbar_monitor,
        lmbda,
        sigmas,
        rounding,
        persistence,
        low,
    )?;
    let values: Vec<f64> = values.as_array().to_vec();
    let years: Vec<i32> = years.as_array().to_vec();
    let doys: Vec<f64> = doys.as_array().to_vec();
    let times: Vec<f64> = times.as_array().to_vec();
    dates(&years, &doys, &times, values.len())?;
    let e = ewmacd::ewmacd(&years, &doys, &times, &values, &p);
    let d = PyDict::new(py);
    d.set_item("flags", PyArray1::from_vec(py, e.flags))?;
    d.set_item("ewma", PyArray1::from_vec(py, e.ewma))?;
    d.set_item("limit", PyArray1::from_vec(py, e.limit))?;
    d.set_item("fitted", PyArray1::from_vec(py, e.fitted))?;
    d.set_item("charted", PyArray1::from_vec(py, e.charted))?;
    d.set_item("beta", PyArray1::from_vec(py, e.beta))?;
    d.set_item("sigma", e.sigma)?;
    d.set_item("rmse", e.rmse)?;
    d.set_item("r2", e.r2)?;
    let ev = e.events;
    d.set_item("break_time", ev.break_time)?;
    d.set_item("severity", ev.severity)?;
    d.set_item("recovery_time", ev.recovery_time)?;
    d.set_item("gain_time", ev.gain_time)?;
    d.set_item("state", ev.state)?;
    Ok(d)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "ewmacd")?;
    m.add(
        "__doc__",
        "EWMACD: monitoring with EWMA control charts (Brooks et al. 2014)",
    )?;
    m.add("NAMES", ewmacd::NAMES.to_vec())?;
    m.add_function(wrap_pyfunction!(ewmacd_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(ewmacd_pixel, &m)?)?;
    Ok(m)
}
