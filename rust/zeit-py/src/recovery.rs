//! zeit._core.recovery: the Python module of zeit-core's recovery metrics.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::recovery::{self, Basis, Params};

use crate::arrays::c_order;

/// The metrics after each pixel's disturbance: values_array (pixels, years)
/// with one value per year of `years`, the disturbance of each pixel from
/// `starts` (the last year before it; 0: none) to `troughs` (the year it
/// bottoms out) -> (9, pixels): pre_val, dist_val, magnitude, post_val, ari,
/// rri, r80p, yryr, y2r; NaN where they cannot be measured.
#[pyfunction]
#[pyo3(signature = (values_array, years, starts, troughs, target = 0.8, horizon = 5, pre_years = 2,
                    basis = "pre_value", loss = true, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn recovery_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    years: PyArrayLike1<'py, i32, AllowTypeChange>,
    starts: PyArrayLike1<'py, i32, AllowTypeChange>,
    troughs: PyArrayLike1<'py, i32, AllowTypeChange>,
    target: f64,
    horizon: i32,
    pre_years: i32,
    basis: &str,
    loss: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_years) = view.dim();
    let years: Vec<i32> = years.as_array().to_vec();
    let starts: Vec<i32> = starts.as_array().to_vec();
    let troughs: Vec<i32> = troughs.as_array().to_vec();
    if years.len() != n_years {
        return Err(PyValueError::new_err(format!(
            "{} years for {n_years} values per pixel",
            years.len()
        )));
    }
    if years.windows(2).any(|w| w[1] <= w[0]) {
        return Err(PyValueError::new_err("the years must increase, each once"));
    }
    if starts.len() != n_pixels || troughs.len() != n_pixels {
        return Err(PyValueError::new_err("one start and one trough per pixel"));
    }
    let basis = Basis::parse(basis)
        .ok_or_else(|| PyValueError::new_err(format!("basis must be 'pre_value' or 'magnitude', got {basis:?}")))?;
    let p = Params {
        target,
        horizon,
        pre_years,
        basis,
        loss,
    };
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| recovery::batch(values, n_pixels, &years, &starts, &troughs, &p, n_jobs));
    PyArray1::from_vec(py, out).reshape([recovery::N_METRICS, n_pixels])
}

/// The fitted trajectory of LandTrendr's vertices at each of `years`:
/// vertex_years and vertex_values (pixels, max_vertices), counts (pixels,)
/// -> (pixels, years), NaN outside the vertices.
#[pyfunction]
#[pyo3(signature = (vertex_years, vertex_values, counts, years, n_jobs = -1))]
fn trajectory_batch<'py>(
    py: Python<'py>,
    vertex_years: PyArrayLike2<'py, f64, AllowTypeChange>,
    vertex_values: PyArrayLike2<'py, f64, AllowTypeChange>,
    counts: PyArrayLike1<'py, i32, AllowTypeChange>,
    years: PyArrayLike1<'py, i32, AllowTypeChange>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let vy = vertex_years.as_array();
    let vv = vertex_values.as_array();
    let (n_pixels, max_v) = vy.dim();
    if vv.dim() != (n_pixels, max_v) {
        return Err(PyValueError::new_err("vertex_years and vertex_values differ in shape"));
    }
    let counts: Vec<i32> = counts.as_array().to_vec();
    if counts.len() != n_pixels {
        return Err(PyValueError::new_err("one count per pixel"));
    }
    let years: Vec<i32> = years.as_array().to_vec();
    let (vy, vv) = (c_order(&vy), c_order(&vv));
    let (vy, vv): (&[f64], &[f64]) = (&vy, &vv);
    let t = years.len();
    let out = py.detach(|| recovery::trajectory_batch(vy, vv, &counts, max_v, &years, n_jobs));
    PyArray1::from_vec(py, out).reshape([n_pixels, t])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "recovery")?;
    m.add("__doc__", "Spectral recovery metrics after a disturbance")?;
    m.add("NAMES", recovery::NAMES.to_vec())?;
    m.add_function(wrap_pyfunction!(recovery_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(trajectory_batch, &m)?)?;
    Ok(m)
}
