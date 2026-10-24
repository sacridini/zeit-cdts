//! zeit._core.beast: the Python module of zeit-core's BEAST.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::beast::{self, Config, Layout, Options, PrecType, TimeAxis};

use crate::arrays::c_order;

#[allow(clippy::too_many_arguments)]
fn options(
    season: &str,
    period: f64,
    scp: (i32, i32),
    sorder: (i32, i32),
    tcp: (i32, i32),
    torder: (i32, i32),
    sseg_min: Option<i32>,
    tseg_min: Option<i32>,
    samples: u32,
    burnin: u32,
    chains: u32,
    thin: u32,
    seed: u64,
    prec_value: f64,
    prec_type: &str,
    ci: bool,
    max_missing_rate: f64,
) -> PyResult<Options> {
    let season = match season {
        "harmonic" => true,
        "none" => false,
        s => {
            return Err(PyValueError::new_err(format!(
                "season must be 'harmonic' or 'none', got {s:?}"
            )))
        }
    };
    let prec_type = match prec_type {
        "componentwise" => PrecType::ComponentWise,
        "uniform" => PrecType::Uniform,
        "constant" => PrecType::Const,
        s => {
            return Err(PyValueError::new_err(format!(
                "prec_type must be 'componentwise', 'uniform' or 'constant', got {s:?}"
            )))
        }
    };
    Ok(Options {
        season,
        period: period as f32,
        start: 1.0,
        dt: 1.0,
        scp,
        sorder,
        tcp,
        torder,
        sseg_min,
        tseg_min,
        samples,
        burnin,
        chains,
        thin,
        seed,
        prec_value: prec_value as f32,
        prec_type,
        ci,
        max_missing_rate: max_missing_rate as f32,
    })
}

fn time_axis(
    n_raw: usize,
    times: Option<Vec<f64>>,
    start: f64,
    dt: f64,
    period: f32,
    decimal_years: bool,
) -> PyResult<TimeAxis> {
    match times {
        Some(t) => {
            if t.len() != n_raw {
                return Err(PyValueError::new_err(format!("{} times for {n_raw} values", t.len())));
            }
            TimeAxis::irregular(&t, start as f32, dt as f32, period, decimal_years).map_err(PyValueError::new_err)
        }
        None => Ok(TimeAxis::regular(
            n_raw,
            if start.is_nan() { 1.0 } else { start as f32 },
            if dt.is_nan() { 1.0 } else { dt as f32 },
        )),
    }
}

/// BEAST of every series of values_array (pixels, dates): on the regular
/// grid of `start` and `dt` (NaN: chosen from the data), the series
/// aggregated to it first when `times` (numeric, one per date) are given
/// (`decimal_years`: times that are dates, as Rbeast chooses their step).
/// -> the outputs (pixels, width), laid out as `beast_layout` says.
#[pyfunction]
#[pyo3(signature = (values_array, times = None, start = f64::NAN, dt = f64::NAN, season = "harmonic", period = f64::NAN,
                    scp = (0, 10), sorder = (0, 5), tcp = (0, 10), torder = (0, 1), sseg_min = None, tseg_min = None,
                    samples = 8000, burnin = 200, chains = 3, thin = 5, seed = 42, prec_value = 1.5,
                    prec_type = "componentwise", ci = true, max_missing_rate = 0.75, decimal_years = false,
                    n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn beast_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    times: Option<PyArrayLike1<'py, f64, AllowTypeChange>>,
    start: f64,
    dt: f64,
    season: &str,
    period: f64,
    scp: (i32, i32),
    sorder: (i32, i32),
    tcp: (i32, i32),
    torder: (i32, i32),
    sseg_min: Option<i32>,
    tseg_min: Option<i32>,
    samples: u32,
    burnin: u32,
    chains: u32,
    thin: u32,
    seed: u64,
    prec_value: f64,
    prec_type: &str,
    ci: bool,
    max_missing_rate: f64,
    decimal_years: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let opts = options(
        season,
        period,
        scp,
        sorder,
        tcp,
        torder,
        sseg_min,
        tseg_min,
        samples,
        burnin,
        chains,
        thin,
        seed,
        prec_value,
        prec_type,
        ci,
        max_missing_rate,
    )?;
    let view = values_array.as_array();
    let (n_pixels, n_raw) = view.dim();
    let axis = time_axis(
        n_raw,
        times.map(|t| t.as_array().to_vec()),
        start,
        dt,
        opts.period,
        decimal_years,
    )?;
    let values = c_order(&view);
    let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let (layout, out) = py
        .detach(|| beast::batch(&values, n_raw, &axis, &opts, n_jobs))
        .map_err(PyValueError::new_err)?;
    PyArray1::from_vec(py, out).reshape([n_pixels, layout.width])
}

/// The outputs of `beast_batch` for series of `n_raw` values: a list of
/// (name, offset, shape) and the regular grid's (n, start, dt).
#[pyfunction]
#[pyo3(signature = (n_raw, times = None, start = f64::NAN, dt = f64::NAN, season = "harmonic", period = f64::NAN,
                    scp = (0, 10), sorder = (0, 5), tcp = (0, 10), torder = (0, 1), sseg_min = None, tseg_min = None,
                    samples = 8000, burnin = 200, chains = 3, thin = 5, seed = 42, prec_value = 1.5,
                    prec_type = "componentwise", ci = true, max_missing_rate = 0.75, decimal_years = false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn beast_layout(
    n_raw: usize,
    times: Option<PyArrayLike1<'_, f64, AllowTypeChange>>,
    start: f64,
    dt: f64,
    season: &str,
    period: f64,
    scp: (i32, i32),
    sorder: (i32, i32),
    tcp: (i32, i32),
    torder: (i32, i32),
    sseg_min: Option<i32>,
    tseg_min: Option<i32>,
    samples: u32,
    burnin: u32,
    chains: u32,
    thin: u32,
    seed: u64,
    prec_value: f64,
    prec_type: &str,
    ci: bool,
    max_missing_rate: f64,
    decimal_years: bool,
) -> PyResult<(Vec<(String, usize, Vec<usize>)>, (usize, f64, f64))> {
    let mut opts = options(
        season,
        period,
        scp,
        sorder,
        tcp,
        torder,
        sseg_min,
        tseg_min,
        samples,
        burnin,
        chains,
        thin,
        seed,
        prec_value,
        prec_type,
        ci,
        max_missing_rate,
    )?;
    let axis = time_axis(
        n_raw,
        times.map(|t| t.as_array().to_vec()),
        start,
        dt,
        opts.period,
        decimal_years,
    )?;
    opts.start = axis.start;
    opts.dt = axis.dt;
    let cfg = Config::new(&opts, axis.n).map_err(PyValueError::new_err)?;
    let layout = Layout::new(&cfg);
    Ok((layout.fields, (axis.n, axis.start as f64, axis.dt as f64)))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "beast")?;
    m.add(
        "__doc__",
        "BEAST: Bayesian estimator of abrupt change, seasonal change and trend (Zhao et al. 2019)",
    )?;
    m.add_function(wrap_pyfunction!(beast_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(beast_layout, &m)?)?;
    Ok(m)
}
