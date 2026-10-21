//! zeit._core.tmask: the Python module.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::robustfit as rf;
use zeit_core::tmask;

use crate::arrays::c_order;

/// Tmask of one pixel's observations: 1 clear, 0 cloud or shadow (a list of
/// ints, as pybind11 gave it).
#[pyfunction]
#[pyo3(signature = (days, green, swir, scale = 10000.0, cloud_threshold = 0.04, shadow_threshold = 0.04))]
fn tmask_pixel(
    days: Vec<f64>,
    green: Vec<f64>,
    swir: Vec<f64>,
    scale: f64,
    cloud_threshold: f64,
    shadow_threshold: f64,
) -> PyResult<Vec<i64>> {
    tmask::tmask_pixel(&days, &green, &swir, scale, cloud_threshold, shadow_threshold)
        .map(|c| c.into_iter().map(i64::from).collect())
        .map_err(PyValueError::new_err)
}

/// Tmask of (pixels, time) green and SWIR: (pixels, time) uint8, 1 clear.
#[pyfunction]
#[pyo3(signature = (green, swir, days, scale = 10000.0, min_observations = 5, cloud_threshold = 0.04,
                    shadow_threshold = 0.04, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn tmask_batch<'py>(
    py: Python<'py>,
    green: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    swir: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    days: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    scale: f64,
    min_observations: i64,
    cloud_threshold: f64,
    shadow_threshold: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (g, s, d) = (green.as_array(), swir.as_array(), days.as_array());
    if g.ndim() != 2 || s.ndim() != 2 || d.ndim() != 1 {
        return Err(PyValueError::new_err("green and swir (pixels, time), days (time,)"));
    }
    let (pixels, t) = (g.shape()[0], g.shape()[1]);
    if s.shape() != g.shape() || d.len() != t {
        return Err(PyValueError::new_err("green, swir and days do not match"));
    }
    let (g, s, d) = (c_order(&g), c_order(&s), c_order(&d));
    let (g, s, d): (&[f64], &[f64], &[f64]) = (&g, &s, &d);
    let out = py.detach(|| {
        tmask::tmask_batch(
            g,
            s,
            d,
            pixels,
            scale,
            min_observations,
            cloud_threshold,
            shadow_threshold,
            n_jobs,
        )
    });
    PyArray1::from_vec(py, out).reshape([pixels, t])
}

/// MATLAB robustfit (bisquare, at most 5 iterations) as CCDC runs it:
/// [intercept, coefficients].
#[pyfunction]
fn robustfit(x: Vec<f64>, n_columns: i64, y: Vec<f64>) -> PyResult<Vec<f64>> {
    let n_columns = usize::try_from(n_columns).map_err(|_| PyValueError::new_err("n_columns < 0"))?;
    if x.len() != y.len() * n_columns {
        return Err(PyValueError::new_err(
            "robustfit: x must be n x n_columns, y of length n",
        ));
    }
    Ok(rf::robustfit_bisquare(&x, n_columns, &y))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "tmask")?;
    m.add("__doc__", "Tmask cloud and shadow screening of time series")?;
    m.add_function(wrap_pyfunction!(tmask_pixel, &m)?)?;
    m.add_function(wrap_pyfunction!(tmask_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(robustfit, &m)?)?;
    Ok(m)
}
