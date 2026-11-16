//! zeit._core.anomaly: the Python module of zeit-core's anomalies (the seasonal
//! cycle and trend taken out of every pixel's series) and monthly means.

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::anomaly::{self, AnomalySettings, Seasonal};

use crate::arrays::c_order;

fn settings(
    method: &str,
    min_per_month: usize,
    order: usize,
    base: Option<(i32, i32)>,
    detrend: bool,
    standardize: bool,
) -> PyResult<AnomalySettings> {
    let seasonal = match method {
        "mean" => Seasonal::Mean,
        "month" | "monthly" => Seasonal::Monthly { min_per_month },
        "harmonic" => Seasonal::Harmonic { order },
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'mean', 'month' or 'harmonic', got {other:?}"
            )))
        }
    };
    if let Some((a, b)) = base {
        if a > b {
            return Err(PyValueError::new_err(format!("base years must be (first, last), got ({a}, {b})")));
        }
    }
    if matches!(seasonal, Seasonal::Harmonic { order: 0 }) {
        return Err(PyValueError::new_err("the harmonic order must be at least 1"));
    }
    Ok(AnomalySettings {
        seasonal,
        base,
        detrend,
        standardize,
    })
}

/// The anomalies of a cube (dates, pixels), NaN missing, with the day count
/// since 1970-01-01 of each date -> (dates, pixels) float64.
#[pyfunction]
#[pyo3(signature = (values, days, method = "month", min_per_month = 2, order = 2, base = None, detrend = false,
    standardize = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn anomalies<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    days: PyArrayLike1<'py, i64, AllowTypeChange>,
    method: &str,
    min_per_month: usize,
    order: usize,
    base: Option<(i32, i32)>,
    detrend: bool,
    standardize: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let s = settings(method, min_per_month, order, base, detrend, standardize)?;
    let view = values.as_array();
    let (t, n) = view.dim();
    let dv = days.as_array();
    if dv.len() != t {
        return Err(PyValueError::new_err(format!("{} days for {t} dates", dv.len())));
    }
    let days: Vec<i64> = dv.iter().copied().collect();
    let mut data: Vec<f64> = c_order(&view).into_owned();
    py.detach(|| anomaly::anomalies_cube(&mut data, t, n, &days, &s, n_jobs));
    PyArray1::from_vec(py, data).reshape([t, n])
}

/// The monthly means of a cube (dates, pixels) -> (months as year * 12 + month - 1,
/// means (months, pixels)); only the months present, ascending.
#[pyfunction]
fn monthly_means<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    days: PyArrayLike1<'py, i64, AllowTypeChange>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, numpy::PyArray2<f64>>)> {
    let view = values.as_array();
    let (t, n) = view.dim();
    let dv = days.as_array();
    if dv.len() != t {
        return Err(PyValueError::new_err(format!("{} days for {t} dates", dv.len())));
    }
    let days: Vec<i64> = dv.iter().copied().collect();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let (months, means) = py.detach(|| anomaly::monthly_means(v, t, n, &days));
    let nm = months.len();
    Ok((PyArray1::from_vec(py, months), PyArray1::from_vec(py, means).reshape([nm, n])?))
}

/// What the anomalies remove (the seasonal part and trend), at every date, of
/// one series -> (dates,) float64, NaN where there is no fit.
#[pyfunction]
#[pyo3(signature = (values, days, method = "month", min_per_month = 2, order = 2, base = None, detrend = false))]
#[allow(clippy::too_many_arguments)]
fn baseline<'py>(
    py: Python<'py>,
    values: PyArrayLike1<'py, f64, AllowTypeChange>,
    days: PyArrayLike1<'py, i64, AllowTypeChange>,
    method: &str,
    min_per_month: usize,
    order: usize,
    base: Option<(i32, i32)>,
    detrend: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = settings(method, min_per_month, order, base, detrend, false)?;
    let v: Vec<f64> = values.as_array().iter().copied().collect();
    let d: Vec<i64> = days.as_array().iter().copied().collect();
    if v.len() != d.len() {
        return Err(PyValueError::new_err(format!("{} days for {} values", d.len(), v.len())));
    }
    Ok(PyArray1::from_vec(py, anomaly::baseline(&v, &d, &s)))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "anomaly")?;
    m.add("__doc__", "Anomalies of time series (mean, monthly climatology, harmonic fit) and monthly means")?;
    m.add_function(wrap_pyfunction!(anomalies, &m)?)?;
    m.add_function(wrap_pyfunction!(monthly_means, &m)?)?;
    m.add_function(wrap_pyfunction!(baseline, &m)?)?;
    Ok(m)
}
