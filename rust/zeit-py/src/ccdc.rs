//! zeit._core.ccdc: the Python module.

use numpy::{AllowTypeChange, PyArray1, PyArray3, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use zeit_core::ccdc;

use crate::arrays::c_order;

#[pyclass(name = "CCDCParams", module = "zeit._core.ccdc", get_all, set_all)]
#[derive(Clone)]
pub struct CCDCParams {
    min_obs: i32,
    conseq_anom: i32,
    chi2_prob_threshold: f64,
    tmax_cg_prob_threshold: f64,
    detection_bands: Vec<i32>,
    num_c: i32,
    tmask_bands: Vec<i32>,
    thermal_band: i32,
    valid_min: f64,
    valid_max: f64,
    thermal_min: f64,
    thermal_max: f64,
}

#[pymethods]
impl CCDCParams {
    #[new]
    fn new() -> Self {
        let p = ccdc::Params::default();
        CCDCParams {
            min_obs: p.min_obs,
            conseq_anom: p.conseq_anom,
            chi2_prob_threshold: p.chi2_prob_threshold,
            tmax_cg_prob_threshold: p.tmax_cg_prob_threshold,
            detection_bands: p.detection_bands,
            num_c: p.num_c,
            tmask_bands: p.tmask_bands,
            thermal_band: p.thermal_band,
            valid_min: p.valid_min,
            valid_max: p.valid_max,
            thermal_min: p.thermal_min,
            thermal_max: p.thermal_max,
        }
    }
}

impl CCDCParams {
    fn core(&self) -> ccdc::Params {
        ccdc::Params {
            min_obs: self.min_obs,
            conseq_anom: self.conseq_anom,
            chi2_prob_threshold: self.chi2_prob_threshold,
            tmax_cg_prob_threshold: self.tmax_cg_prob_threshold,
            num_c: self.num_c,
            detection_bands: self.detection_bands.clone(),
            tmask_bands: self.tmask_bands.clone(),
            thermal_band: self.thermal_band,
            valid_min: self.valid_min,
            valid_max: self.valid_max,
            thermal_min: self.thermal_min,
            thermal_max: self.thermal_max,
        }
    }
}

#[pyclass(name = "CCDCSegment", module = "zeit._core.ccdc", get_all, set_all)]
#[derive(Clone, Default)]
pub struct CCDCSegment {
    t_start: i32,
    t_end: i32,
    t_break: i32,
    coefs: Vec<Vec<f64>>,
    rmse: Vec<f64>,
    magnitude: Vec<f64>,
    change_prob: f64,
    category: i32,
    num_obs: i32,
}

#[pymethods]
impl CCDCSegment {
    #[new]
    fn new() -> Self {
        CCDCSegment::default()
    }
}

/// Run CCDC on a single pixel time series.
#[pyfunction]
#[pyo3(signature = (dates, values, qa, params = None))]
fn fit_ccdc(
    py: Python<'_>,
    dates: Vec<i32>,
    values: Vec<Vec<f64>>,
    qa: Vec<i32>,
    params: Option<PyRef<'_, CCDCParams>>,
) -> PyResult<Vec<CCDCSegment>> {
    let p = params.map_or_else(ccdc::Params::default, |p| p.core());
    let segs = py
        .detach(|| ccdc::fit_ccdc(&dates, &values, &qa, &p))
        .map_err(PyValueError::new_err)?;
    Ok(segs
        .into_iter()
        .map(|s| CCDCSegment {
            t_start: s.t_start,
            t_end: s.t_end,
            t_break: s.t_break,
            coefs: s.coefs,
            rmse: s.rmse,
            magnitude: s.magnitude,
            change_prob: s.change_prob,
            category: s.category,
            num_obs: s.num_obs,
        })
        .collect())
}

/// Run CCDC on a batch of pixels: values [Y, X, Bands, Time], qa [Y, X, Time],
/// dates [Time]; (segments [Y*X, max_segments, 3 + Bands * 9 or 1], counts
/// [Y*X] int32).
#[pyfunction]
#[pyo3(signature = (values_array, qa_array, dates_array, params, max_segments = 6, return_coefs = true, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_ccdc_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    qa_array: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    dates_array: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    params: PyRef<'py, CCDCParams>,
    max_segments: i32,
    return_coefs: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let view = values_array.as_array();
    if view.ndim() != 4 {
        return Err(PyValueError::new_err("values_array must be 4D [Y, X, Bands, Time]"));
    }
    let sh = view.shape();
    let (pixels, nb, times) = (sh[0] * sh[1], sh[2], sh[3]);
    if max_segments < 0 {
        return Err(PyValueError::new_err("max_segments must not be negative"));
    }
    let qa: Vec<i32> = qa_array.as_array().iter().copied().collect();
    if qa.len() != pixels * times {
        return Err(PyValueError::new_err("qa_array must be [Y, X, Time]"));
    }
    let dates: Vec<i32> = dates_array.as_array().iter().copied().collect();
    if dates.len() < times {
        return Err(PyValueError::new_err("dates_array must have one date per time step"));
    }
    let dates = &dates[..times];
    let values = c_order(&view);
    let values: &[f64] = &values;
    let p = params.core();
    let max_segments = max_segments as usize;
    let (segments, counts) = py
        .detach(|| ccdc::fit_ccdc_batch(values, &qa, dates, pixels, nb, &p, max_segments, return_coefs, n_jobs))
        .map_err(PyValueError::new_err)?;
    let pps = if return_coefs { 3 + nb * 9 } else { 1 };
    let segments: Bound<'py, PyArray3<f64>> = PyArray1::from_vec(py, segments).reshape([pixels, max_segments, pps])?;
    let counts = PyArray1::from_vec(py, counts);
    PyTuple::new(py, [segments.into_any(), counts.into_any()])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "ccdc")?;
    m.add("__doc__", "CCDC algorithms")?;
    m.add_class::<CCDCParams>()?;
    m.add_class::<CCDCSegment>()?;
    m.add_function(wrap_pyfunction!(fit_ccdc, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_ccdc_batch, &m)?)?;
    Ok(m)
}
