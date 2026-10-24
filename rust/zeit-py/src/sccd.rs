//! zeit._core.sccd: the Python module of zeit-core's S-CCD.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::sccd::{self, Params};

use crate::arrays::c_order;

#[allow(clippy::too_many_arguments)]
fn params(
    conse: i32,
    prob_threshold: f64,
    outlier_prob: f64,
    max_angle: f64,
    detection_bands: Option<Vec<i32>>,
    tmask_bands: Option<Vec<i32>>,
    valid_min: f64,
    valid_max: f64,
) -> PyResult<Params> {
    if conse < 1 {
        return Err(PyValueError::new_err(format!("conse must be at least 1, got {conse}")));
    }
    for (name, p) in [("prob_threshold", prob_threshold), ("outlier_prob", outlier_prob)] {
        if !(p > 0.0 && p < 1.0) {
            return Err(PyValueError::new_err(format!("{name} must be in (0, 1), got {p}")));
        }
    }
    Ok(Params {
        conse,
        prob_threshold,
        outlier_prob,
        max_angle,
        detection_bands: detection_bands.unwrap_or_default(),
        tmask_bands: tmask_bands.unwrap_or_default(),
        valid_min,
        valid_max,
    })
}

/// S-CCD of one pixel: values (bands, dates), dates (Python ordinal days), qa (Fmask codes) ->
/// a dict of the segments (t_start, t_end, t_break, change_prob, num_obs, coefs (segments,
/// bands, 6), rmse, magnitude), the near-real-time state (mode, n_anomalies, anomaly_start,
/// anomaly_statistic, anomaly_magnitude, last_statistic, last_date, n_queue), the trace per
/// date (status, statistic, predicted, trend, annual, semiannual (dates, bands)) and the
/// change threshold.
#[pyfunction]
#[pyo3(signature = (dates, values, qa, conse = 6, prob_threshold = 0.99, outlier_prob = 0.99999, max_angle = 30.0,
                    detection_bands = None, tmask_bands = None, valid_min = 0.0, valid_max = 10000.0))]
#[allow(clippy::too_many_arguments)]
fn sccd_pixel<'py>(
    py: Python<'py>,
    dates: PyArrayLike1<'py, i32, AllowTypeChange>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    qa: PyArrayLike1<'py, i32, AllowTypeChange>,
    conse: i32,
    prob_threshold: f64,
    outlier_prob: f64,
    max_angle: f64,
    detection_bands: Option<Vec<i32>>,
    tmask_bands: Option<Vec<i32>>,
    valid_min: f64,
    valid_max: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = params(
        conse,
        prob_threshold,
        outlier_prob,
        max_angle,
        detection_bands,
        tmask_bands,
        valid_min,
        valid_max,
    )?;
    let dates: Vec<i32> = dates.as_array().to_vec();
    let qa: Vec<i32> = qa.as_array().to_vec();
    let view = values.as_array();
    let bands: Vec<Vec<f64>> = view.outer_iter().map(|r| r.to_vec()).collect();
    let r = py
        .detach(|| sccd::sccd(&dates, &bands, &qa, &p))
        .map_err(PyValueError::new_err)?;
    let nb = bands.len();
    let nt = dates.len();
    let ns = r.segments.len();
    let d = PyDict::new(py);
    let seg = |f: &dyn Fn(&sccd::Segment) -> f64| PyArray1::from_vec(py, r.segments.iter().map(f).collect());
    d.set_item("t_start", seg(&|s| s.t_start as f64))?;
    d.set_item("t_end", seg(&|s| s.t_end as f64))?;
    d.set_item("t_break", seg(&|s| s.t_break as f64))?;
    d.set_item("change_prob", seg(&|s| s.change_prob))?;
    d.set_item("num_obs", seg(&|s| s.num_obs as f64))?;
    let per_band = |f: &dyn Fn(&sccd::Segment) -> &Vec<f64>| -> PyResult<Bound<'py, PyArray2<f64>>> {
        let v: Vec<f64> = r.segments.iter().flat_map(|s| f(s).iter().copied()).collect();
        PyArray1::from_vec(py, v).reshape([ns, nb])
    };
    d.set_item("rmse", per_band(&|s| &s.rmse)?)?;
    d.set_item("magnitude", per_band(&|s| &s.magnitude)?)?;
    let coefs: Vec<f64> = r
        .segments
        .iter()
        .flat_map(|s| s.coefs.iter().flat_map(|c| c.iter().copied()))
        .collect();
    d.set_item("coefs", PyArray1::from_vec(py, coefs).reshape([ns, nb, sccd::NUM_C])?)?;
    let m = &r.monitor;
    d.set_item("mode", m.mode)?;
    d.set_item("n_anomalies", m.n_anomalies)?;
    d.set_item("anomaly_start", m.anomaly_start)?;
    d.set_item("anomaly_statistic", m.anomaly_statistic)?;
    d.set_item("anomaly_magnitude", PyArray1::from_vec(py, m.anomaly_magnitude.clone()))?;
    d.set_item("last_statistic", m.last_statistic)?;
    d.set_item("last_date", m.last_date)?;
    d.set_item("n_queue", m.n_queue)?;
    let t = r.trace;
    d.set_item("status", PyArray1::from_vec(py, t.status))?;
    d.set_item("statistic", PyArray1::from_vec(py, t.statistic))?;
    for (name, v) in [
        ("predicted", t.predicted),
        ("trend", t.trend),
        ("annual", t.annual),
        ("semiannual", t.semiannual),
    ] {
        d.set_item(name, PyArray1::from_vec(py, v).reshape([nt, nb])?)?;
    }
    d.set_item("threshold", r.threshold)?;
    Ok(d)
}

/// S-CCD of a batch of pixels: values [Y, X, Bands, Time], qa [Y, X, Time], dates [Time] ->
/// [Y*X, max_segments * segment_width + 1 + monitor_width] (see zeit_core::sccd::sccd_batch).
#[pyfunction]
#[pyo3(signature = (values_array, qa_array, dates_array, conse = 6, prob_threshold = 0.99, outlier_prob = 0.99999,
                    max_angle = 30.0, detection_bands = None, tmask_bands = None, valid_min = 0.0,
                    valid_max = 10000.0, max_segments = 6, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn sccd_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    qa_array: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    dates_array: PyArrayLike1<'py, i32, AllowTypeChange>,
    conse: i32,
    prob_threshold: f64,
    outlier_prob: f64,
    max_angle: f64,
    detection_bands: Option<Vec<i32>>,
    tmask_bands: Option<Vec<i32>>,
    valid_min: f64,
    valid_max: f64,
    max_segments: i32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = params(
        conse,
        prob_threshold,
        outlier_prob,
        max_angle,
        detection_bands,
        tmask_bands,
        valid_min,
        valid_max,
    )?;
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
    let dates: Vec<i32> = dates_array.as_array().to_vec();
    if dates.len() != times {
        return Err(PyValueError::new_err("dates_array must have one date per time step"));
    }
    let values = c_order(&view);
    let values: &[f64] = &values;
    let ms = max_segments as usize;
    let out = py
        .detach(|| sccd::sccd_batch(values, &qa, &dates, pixels, nb, &p, ms, n_jobs))
        .map_err(PyValueError::new_err)?;
    let width = ms * sccd::segment_width(nb) + 1 + sccd::monitor_width(nb);
    PyArray1::from_vec(py, out).reshape([pixels, width])
}

/// The change threshold, chi2inv(prob_threshold, detection bands), with `n_bands` bands.
#[pyfunction]
#[pyo3(signature = (n_bands, prob_threshold = 0.99, detection_bands = None))]
fn threshold(n_bands: usize, prob_threshold: f64, detection_bands: Option<Vec<i32>>) -> PyResult<f64> {
    let p = params(6, prob_threshold, 0.99999, 30.0, detection_bands, None, 0.0, 10000.0)?;
    sccd::threshold(&p, n_bands).map_err(PyValueError::new_err)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "sccd")?;
    m.add(
        "__doc__",
        "S-CCD: stochastic continuous change detection (Ye et al. 2021)",
    )?;
    m.add("NUM_C", sccd::NUM_C)?;
    m.add_function(wrap_pyfunction!(sccd_pixel, &m)?)?;
    m.add_function(wrap_pyfunction!(sccd_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(threshold, &m)?)?;
    Ok(m)
}
