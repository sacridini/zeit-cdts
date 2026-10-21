//! zeit._rust.landtrendr: the bindings of zeit._core.landtrendr.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArray3, PyArrayLike1, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use zeit_core::landtrendr as lt;

use crate::arrays::c_order;

#[pyclass(name = "LandTrendrParams", module = "zeit._rust.landtrendr", get_all, set_all)]
#[derive(Clone)]
pub struct LandTrendrParams {
    max_segments: i32,
    pval_threshold: f64,
    prevent_fast_recovery: bool,
    recovery_threshold: f64,
    spike_threshold: f64,
    best_model_proportion: f64,
    vertex_count_overshoot: i32,
    min_observations_needed: i32,
    modifier: f64,
}

#[pymethods]
impl LandTrendrParams {
    #[new]
    fn new() -> Self {
        let p = lt::Params::default();
        LandTrendrParams {
            max_segments: p.max_segments,
            pval_threshold: p.pval_threshold,
            prevent_fast_recovery: p.prevent_fast_recovery,
            recovery_threshold: p.recovery_threshold,
            spike_threshold: p.spike_threshold,
            best_model_proportion: p.best_model_proportion,
            vertex_count_overshoot: p.vertex_count_overshoot,
            min_observations_needed: p.min_observations_needed,
            modifier: p.modifier,
        }
    }
}

impl LandTrendrParams {
    fn core(&self) -> lt::Params {
        lt::Params {
            max_segments: self.max_segments,
            pval_threshold: self.pval_threshold,
            prevent_fast_recovery: self.prevent_fast_recovery,
            recovery_threshold: self.recovery_threshold,
            spike_threshold: self.spike_threshold,
            best_model_proportion: self.best_model_proportion,
            vertex_count_overshoot: self.vertex_count_overshoot,
            min_observations_needed: self.min_observations_needed,
            modifier: self.modifier,
        }
    }
}

#[pyclass(name = "Vertex", module = "zeit._rust.landtrendr", get_all, set_all)]
#[derive(Clone)]
pub struct Vertex {
    year: i32,
    value: f64,
}

#[pymethods]
impl Vertex {
    #[new]
    fn new(year: i32, value: f64) -> Self {
        Vertex { year, value }
    }
}

/// Run LandTrendr on a single pixel time series.
#[pyfunction]
fn fit_trajectory(
    py: Python<'_>,
    years: Vec<i32>,
    values: Vec<f64>,
    params: PyRef<'_, LandTrendrParams>,
) -> Vec<Vertex> {
    let p = params.core();
    let r = py.detach(|| lt::fit_trajectory(&years, &values, &p));
    r.vertices
        .into_iter()
        .map(|v| Vertex {
            year: v.year,
            value: v.value,
        })
        .collect()
}

/// Run LandTrendr on a batch of pixels (3D array: [Y, X, Time]): (vertices
/// [pixels, max_vertices, 2], counts [pixels] int32, rmse [pixels]).
#[pyfunction]
#[pyo3(signature = (values_array, years_array, params, no_data_value = -9999.0, n_jobs = -1))]
fn fit_trajectory_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    years_array: PyArrayLike1<'py, i32, AllowTypeChange>,
    params: PyRef<'py, LandTrendrParams>,
    no_data_value: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let view = values_array.as_array();
    if view.ndim() != 3 {
        return Err(PyValueError::new_err("values_array must be 3D [Y, X, Time]"));
    }
    let (pixels, times) = (view.shape()[0] * view.shape()[1], view.shape()[2]);
    let years: Vec<i32> = years_array.as_array().iter().copied().collect();
    if years.len() < times {
        return Err(PyValueError::new_err("years_array must have one year per time step"));
    }
    let years = &years[..times];
    let values = c_order(&view);
    let values: &[f64] = &values;
    let p = params.core();
    let max_vertices = (p.max_segments as i64 + 1).max(0) as usize;
    let (vertices, counts, rmse) =
        py.detach(|| lt::fit_trajectory_batch(values, pixels, years, &p, no_data_value, n_jobs));
    let vertices: Bound<'py, PyArray3<f64>> = PyArray1::from_vec(py, vertices).reshape([pixels, max_vertices, 2])?;
    let counts = PyArray1::from_vec(py, counts);
    let rmse = PyArray1::from_vec(py, rmse);
    PyTuple::new(py, [vertices.into_any(), counts.into_any(), rmse.into_any()])
}

/// Fitted-to-vertices of one pixel's series of another band (ftv_v1): the
/// fitted value of every year.
#[pyfunction]
#[pyo3(signature = (years, values, vertex_years, spike_threshold = 0.9))]
fn fit_to_vertices(
    years: Vec<i32>,
    values: Vec<f64>,
    vertex_years: Vec<i32>,
    spike_threshold: f64,
) -> PyResult<Vec<f64>> {
    if values.len() != years.len() {
        return Err(PyValueError::new_err("years and values must have the same length"));
    }
    Ok(lt::fit_to_vertices(&years, &values, &vertex_years, spike_threshold))
}

/// fit_to_vertices on (pixels, time) values with (pixels, vertices) vertex
/// years and their counts.
#[pyfunction]
#[pyo3(signature = (values, years, vertex_years, counts, spike_threshold = 0.9, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_to_vertices_batch<'py>(
    py: Python<'py>,
    values: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    years: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    vertex_years: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    counts: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    spike_threshold: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (v, y, vy, c) = (
        values.as_array(),
        years.as_array(),
        vertex_years.as_array(),
        counts.as_array(),
    );
    if v.ndim() != 2 || vy.ndim() != 2 || y.ndim() != 1 || c.ndim() != 1 {
        return Err(PyValueError::new_err(
            "values (pixels, time), years (time,), vertex_years (pixels, vertices), counts (pixels,)",
        ));
    }
    let (pixels, times, max_vertices) = (v.shape()[0], v.shape()[1], vy.shape()[1]);
    if y.len() != times || vy.shape()[0] != pixels || c.len() != pixels {
        return Err(PyValueError::new_err(
            "values, years, vertex_years and counts do not match",
        ));
    }
    let v = c_order(&v);
    let v: &[f64] = &v;
    let y: Vec<i32> = y.iter().copied().collect();
    let vy: Vec<i32> = vy.iter().copied().collect();
    let c: Vec<i32> = c.iter().copied().collect();
    let out = py.detach(|| lt::fit_to_vertices_batch(v, &y, &vy, &c, max_vertices, spike_threshold, n_jobs));
    PyArray1::from_vec(py, out).reshape([pixels, times])
}

/// Remove spikes from a time series.
#[pyfunction]
#[pyo3(signature = (vals, stopat = 0.9))]
fn desawtooth(vals: Vec<f64>, stopat: f64) -> Vec<f64> {
    lt::desawtooth(&vals, stopat)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "landtrendr")?;
    m.add("__doc__", "LandTrendr algorithms")?;
    m.add_class::<LandTrendrParams>()?;
    m.add_class::<Vertex>()?;
    m.add_function(wrap_pyfunction!(fit_trajectory, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_trajectory_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_to_vertices, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_to_vertices_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(desawtooth, &m)?)?;
    Ok(m)
}
