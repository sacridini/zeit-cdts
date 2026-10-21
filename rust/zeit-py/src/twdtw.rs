//! zeit._rust.twdtw: the bindings of zeit._core.twdtw.

use numpy::{AllowTypeChange, IxDyn, PyArray1, PyArrayDyn, PyArrayLike1, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::twdtw::{self as tw, Params, Series, TimeWeight};

use crate::arrays::c_order;

#[pyclass(name = "TWDTWParams", module = "zeit._rust.twdtw", get_all, set_all)]
#[derive(Clone)]
pub struct TWDTWParams {
    alpha: f64,
    beta: f64,
    gamma: f64,
    max_time_warp: i32,
    subsequence_matching: bool,
    cycle_length: f64,
}

#[pymethods]
impl TWDTWParams {
    #[new]
    fn new() -> Self {
        let p = Params::default();
        TWDTWParams {
            alpha: p.alpha,
            beta: p.beta,
            gamma: p.gamma,
            max_time_warp: p.max_time_warp,
            subsequence_matching: p.subsequence_matching,
            cycle_length: p.cycle_length,
        }
    }
}

impl TWDTWParams {
    fn core(&self) -> Params {
        Params {
            alpha: self.alpha,
            beta: self.beta,
            gamma: self.gamma,
            max_time_warp: self.max_time_warp,
            subsequence_matching: self.subsequence_matching,
            cycle_length: self.cycle_length,
        }
    }
}

#[pyclass(name = "TWDTWResult", module = "zeit._rust.twdtw", get_all, set_all)]
pub struct TWDTWResult {
    distance: f64,
    path: Vec<(i32, i32)>,
}

#[pymethods]
impl TWDTWResult {
    #[new]
    fn new() -> Self {
        TWDTWResult {
            distance: 0.0,
            path: Vec::new(),
        }
    }
}

fn check_series(what: &str, values: &[f64], dates: &[i32], num_bands: usize) -> PyResult<()> {
    if values.len() != dates.len() * num_bands {
        return Err(PyValueError::new_err(format!(
            "{what}: {} values for {} dates of {num_bands} band(s)",
            values.len(),
            dates.len()
        )));
    }
    Ok(())
}

/// Run TWDTW on a single time series against a pattern.
#[pyfunction]
#[pyo3(signature = (ts_values, ts_dates, pattern_values, pattern_dates, num_bands = 1, params = None,
                    abort_threshold = f64::INFINITY, return_path = false))]
#[allow(clippy::too_many_arguments)]
fn fit_twdtw(
    py: Python<'_>,
    ts_values: Vec<f64>,
    ts_dates: Vec<i32>,
    pattern_values: Vec<f64>,
    pattern_dates: Vec<i32>,
    num_bands: i32,
    params: Option<PyRef<'_, TWDTWParams>>,
    abort_threshold: f64,
    return_path: bool,
) -> PyResult<TWDTWResult> {
    let num_bands = usize::try_from(num_bands)
        .ok()
        .filter(|&b| b > 0)
        .ok_or_else(|| PyValueError::new_err("num_bands must be at least 1"))?;
    check_series("ts", &ts_values, &ts_dates, num_bands)?;
    check_series("pattern", &pattern_values, &pattern_dates, num_bands)?;
    let p = params.map_or_else(Params::default, |p| p.core());
    let (distance, path) = py.detach(|| {
        let ts = Series {
            values: &ts_values,
            dates: &ts_dates,
            num_bands,
        };
        let pattern = Series {
            values: &pattern_values,
            dates: &pattern_dates,
            num_bands,
        };
        let weight = TimeWeight::new(&p);
        if return_path {
            tw::distance_and_path(ts, pattern, &p, &weight, abort_threshold)
        } else {
            (
                tw::distance(ts, pattern, &p, &weight, abort_threshold, &mut Default::default()),
                Vec::new(),
            )
        }
    });
    Ok(TWDTWResult { distance, path })
}

/// Run TWDTW on a batch of pixels: values (y, x, time) or (y, x, time, bands)
/// against one pattern -> (y, x) distances.
#[pyfunction]
#[pyo3(signature = (values_array, dates_array, pattern_values_array, pattern_dates_array, params,
                    abort_threshold = f64::INFINITY, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_twdtw_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    dates_array: PyArrayLike1<'py, i32, AllowTypeChange>,
    pattern_values_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    pattern_dates_array: PyArrayLike1<'py, i32, AllowTypeChange>,
    params: PyRef<'py, TWDTWParams>,
    abort_threshold: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let view = values_array.as_array();
    let shape = view.shape().to_vec();
    if shape.len() != 3 && shape.len() != 4 {
        return Err(PyValueError::new_err(
            "values_array must be (y, x, time) or (y, x, time, bands)",
        ));
    }
    let (ny, nx, nt) = (shape[0], shape[1], shape[2]);
    let num_bands = if shape.len() == 4 { shape[3] } else { 1 };
    let dates = dates_array.as_array();
    let dates = dates
        .as_slice()
        .map(<[i32]>::to_vec)
        .unwrap_or_else(|| dates.iter().copied().collect());
    if dates.len() != nt {
        return Err(PyValueError::new_err(format!(
            "{} dates for a time axis of {nt}",
            dates.len()
        )));
    }
    let pat_view = pattern_values_array.as_array();
    let pat_values = c_order(&pat_view);
    let pat_dates = pattern_dates_array.as_array();
    let pat_dates = pat_dates
        .as_slice()
        .map(<[i32]>::to_vec)
        .unwrap_or_else(|| pat_dates.iter().copied().collect());
    check_series("pattern", &pat_values, &pat_dates, num_bands)?;
    let values = c_order(&view);
    let values: &[f64] = &values;
    let pat_values: &[f64] = &pat_values;
    let p = params.core();
    let out = py.detach(|| {
        let pattern = Series {
            values: pat_values,
            dates: &pat_dates,
            num_bands,
        };
        tw::batch(values, ny * nx, &dates, num_bands, pattern, &p, abort_threshold, n_jobs)
    });
    PyArray1::from_vec(py, out).reshape(IxDyn(&[ny, nx]))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "twdtw")?;
    m.add("__doc__", "TWDTW algorithms")?;
    m.add_class::<TWDTWParams>()?;
    m.add_class::<TWDTWResult>()?;
    m.add_function(wrap_pyfunction!(fit_twdtw, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_twdtw_batch, &m)?)?;
    Ok(m)
}
