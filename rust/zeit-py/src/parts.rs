//! zeit._core.parts: the Python module of remotePARTS's methods (zeit-core's
//! ar_reml, spatial and parts).

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::ar_reml::{self, Gaps};
use zeit_core::gls::Nugget;
use zeit_core::parts;
use zeit_core::spatial::{self, Correlation};

use crate::arrays::c_order;

fn gaps(name: &str) -> PyResult<Gaps> {
    match name {
        "exact" => Ok(Gaps::Exact),
        "ignore" => Ok(Gaps::Ignore),
        _ => Err(PyValueError::new_err(format!(
            "gaps must be 'exact' or 'ignore', got {name:?}"
        ))),
    }
}

fn correlation(kind: &str, range: f64, shape: f64) -> PyResult<Correlation> {
    match kind {
        "exponential" => Ok(Correlation::Exponential { range }),
        "exponential_power" => Ok(Correlation::ExponentialPower { range, shape }),
        _ => Err(PyValueError::new_err(format!(
            "correlation must be 'exponential' or 'exponential_power', got {kind:?}"
        ))),
    }
}

/// A design (time, q) in C order as zeit-core takes it: by column.
fn by_column(design: &PyArrayLike2<'_, f64, AllowTypeChange>) -> (Vec<f64>, usize, usize) {
    let view = design.as_array();
    let (rows, cols) = view.dim();
    let mut out = vec![0.0; rows * cols];
    for ((r, c), v) in view.indexed_iter() {
        out[c * rows + r] = *v;
    }
    (out, rows, cols)
}

/// The AR(1) REML fit of every pixel's series on a design shared by the
/// pixels: values_array (pixels, time), design (time, q) -> (4 q + 3, pixels):
/// the coefficients, their standard errors, t and p (q each), rho, the
/// residual variance and the log-likelihood; NaN where fewer than min_valid
/// (at least q + 2) valid observations.
#[pyfunction]
#[pyo3(signature = (values_array, design, gaps = "exact", min_valid = 0, n_jobs = -1))]
fn fit_ar_batch<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    design: PyArrayLike2<'py, f64, AllowTypeChange>,
    gaps: &str,
    min_valid: usize,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_time) = view.dim();
    let (x, rows, q) = by_column(&design);
    if rows != n_time {
        return Err(PyValueError::new_err(format!(
            "the design has {rows} rows for {n_time} dates"
        )));
    }
    let g = self::gaps(gaps)?;
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| ar_reml::batch(values, n_pixels, n_time, &x, q, g, min_valid, n_jobs));
    PyArray1::from_vec(py, out).reshape([ar_reml::n_metrics(q), n_pixels])
}

/// The residuals y - X beta of each pixel's AR(1) fit: (pixels, time), NaN
/// where y is missing or the pixel has no fit.
#[pyfunction]
#[pyo3(signature = (values_array, design, gaps = "exact"))]
fn ar_residuals<'py>(
    py: Python<'py>,
    values_array: PyArrayLike2<'py, f64, AllowTypeChange>,
    design: PyArrayLike2<'py, f64, AllowTypeChange>,
    gaps: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let view = values_array.as_array();
    let (n_pixels, n_time) = view.dim();
    let (x, rows, q) = by_column(&design);
    if rows != n_time {
        return Err(PyValueError::new_err(format!(
            "the design has {rows} rows for {n_time} dates"
        )));
    }
    let g = self::gaps(gaps)?;
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| {
        let mut out = vec![f64::NAN; n_pixels * n_time];
        for (p, row) in out.chunks_mut(n_time).enumerate() {
            if let Some(f) = ar_reml::fit(&values[p * n_time..(p + 1) * n_time], &x, q, g) {
                row.copy_from_slice(&f.residuals);
            }
        }
        out
    });
    PyArray1::from_vec(py, out).reshape([n_pixels, n_time])
}

/// The correlation function of distance fitted to the correlations between
/// the residual series resids (points, time) of the points (x, y)
/// (remotePARTS's fitCor on its sample): a dict of range, shape (None for
/// "exponential"), max_distance, sse and pairs.
#[pyfunction]
#[pyo3(signature = (resids, x, y, correlation = "exponential", range = 0.1, shape = 1.0, scaled = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_correlation<'py>(
    py: Python<'py>,
    resids: PyArrayLike2<'py, f64, AllowTypeChange>,
    x: Vec<f64>,
    y: Vec<f64>,
    correlation: &str,
    range: f64,
    shape: f64,
    scaled: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let view = resids.as_array();
    let (n, n_time) = view.dim();
    if x.len() != n || y.len() != n {
        return Err(PyValueError::new_err(format!(
            "{n} series for {} x and {} y",
            x.len(),
            y.len()
        )));
    }
    let start = self::correlation(correlation, range, shape)?;
    let values = c_order(&view);
    let values: &[f64] = &values;
    let fit = py
        .detach(|| spatial::fit_correlation(values, n_time, &x, &y, start, scaled, n_jobs))
        .ok_or_else(|| PyValueError::new_err("the correlation function did not converge"))?;
    let d = PyDict::new(py);
    match fit.cor {
        Correlation::Exponential { range } => {
            d.set_item("range", range)?;
            d.set_item("shape", py.None())?;
        }
        Correlation::ExponentialPower { range, shape } => {
            d.set_item("range", range)?;
            d.set_item("shape", shape)?;
        }
    }
    d.set_item("max_distance", fit.max_distance)?;
    d.set_item("sse", fit.sse)?;
    d.set_item("pairs", fit.pairs)?;
    Ok(d)
}

/// npart partitions of partsize pixels out of npix, without replacement
/// (remotePARTS's sample_partitions, seeded): (npart, partsize) indices.
#[pyfunction]
#[pyo3(signature = (npix, partsize, npart, seed = 0))]
fn sample_partitions<'py>(
    py: Python<'py>,
    npix: usize,
    partsize: usize,
    npart: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    if partsize * npart > npix {
        return Err(PyValueError::new_err("npart * partsize may not be greater than npix"));
    }
    let idx: Vec<i64> = parts::sample_partitions(npix, partsize, npart, seed)
        .into_iter()
        .map(|i| i as i64)
        .collect();
    PyArray1::from_vec(py, idx).reshape([npart, partsize])
}

/// PARTS (remotePARTS's fitGLS_partition) on the pixels at (x, y) with their
/// response, the design (pixels, p) and the null design (pixels, p0), in the
/// partitions (npart, partsize); nugget None for the maximum-likelihood one.
/// A dict of numpy arrays and numbers (see zeit.parts).
#[pyfunction]
#[pyo3(signature = (x, y, response, design, design0, partitions, correlation = "exponential", range = 0.1,
                    shape = 1.0, scaled = false, nugget = None, ncross = 6, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit_parts<'py>(
    py: Python<'py>,
    x: PyArrayLike1<'py, f64, AllowTypeChange>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    response: PyArrayLike1<'py, f64, AllowTypeChange>,
    design: PyArrayLike2<'py, f64, AllowTypeChange>,
    design0: PyArrayLike2<'py, f64, AllowTypeChange>,
    partitions: PyArrayLike2<'py, i64, AllowTypeChange>,
    correlation: &str,
    range: f64,
    shape: f64,
    scaled: bool,
    nugget: Option<f64>,
    ncross: usize,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let (xv, yv, rv) = (x.as_array(), y.as_array(), response.as_array());
    let (x, y, response) = (c_order(&xv), c_order(&yv), c_order(&rv));
    let npix = x.len();
    let (design, rows, p) = by_column(&design);
    let (design0, rows0, p0) = by_column(&design0);
    if y.len() != npix || response.len() != npix || rows != npix || rows0 != npix {
        return Err(PyValueError::new_err(
            "x, y, response and the designs must have a row per pixel",
        ));
    }
    if p == 0 || p0 > p {
        return Err(PyValueError::new_err(
            "the design needs a column, and the null design no more than it",
        ));
    }
    let pv = partitions.as_array();
    let (npart, partsize) = pv.dim();
    let mut idx = Vec::with_capacity(npart * partsize);
    for &i in pv.iter() {
        if i < 0 || i as usize >= npix {
            return Err(PyValueError::new_err(format!(
                "partition index {i} outside the {npix} pixels"
            )));
        }
        idx.push(i as usize);
    }
    if partsize <= p + 1 {
        return Err(PyValueError::new_err(
            "the partitions need more pixels than the design has columns",
        ));
    }
    let opts = parts::Options {
        cor: self::correlation(correlation, range, shape)?,
        scaled,
        nugget: nugget.map_or(Nugget::Ml, Nugget::Fixed),
        ncross,
        n_jobs,
    };
    let data = parts::Data {
        x: &x,
        y: &y,
        response: &response,
        design: &design,
        p,
        design0: &design0,
        p0,
    };
    let fit = py
        .detach(|| parts::fit(&data, &idx, partsize, &opts))
        .ok_or_else(|| PyValueError::new_err("a partition's covariance or design is singular"))?;

    let d = PyDict::new(py);
    let vec = |v: Vec<f64>| PyArray1::from_vec(py, v);
    let mat = |v: Vec<f64>, r: usize, c: usize| PyArray1::from_vec(py, v).reshape([r, c]);
    // p x p by column -> (p, p) by row
    let square = |v: &[f64]| (0..p * p).map(|e| v[(e % p) * p + e / p]).collect::<Vec<_>>();
    d.set_item("coefficients", vec(fit.t_test.coefficients.clone()))?;
    d.set_item("se", vec(fit.t_test.se.clone()))?;
    d.set_item("t", vec(fit.t_test.t.clone()))?;
    d.set_item("p", vec(fit.t_test.p.clone()))?;
    d.set_item("covar", mat(square(&fit.t_test.covar), p, p)?)?;
    d.set_item("rcoefficients", mat(square(&fit.rcoefficients), p, p)?)?;
    d.set_item("rssr", fit.rssr)?;
    d.set_item("rsse", fit.rsse)?;
    d.set_item("f", fit.f)?;
    d.set_item("df1", fit.df1)?;
    d.set_item("df2", fit.df2)?;
    d.set_item("p_chisqr", fit.p_chisqr)?;
    d.set_item("rank_deficient", fit.rank_deficient)?;
    let per = |f: &dyn Fn(&zeit_core::gls::GlsFit) -> Vec<f64>, cols: usize| -> PyResult<_> {
        mat(fit.parts.iter().flat_map(f).collect(), npart, cols)
    };
    d.set_item("part_coefficients", per(&|g| g.coefficients.clone(), p)?)?;
    d.set_item("part_se", per(&|g| g.se.clone(), p)?)?;
    let one = |f: &dyn Fn(&zeit_core::gls::GlsFit) -> f64| vec(fit.parts.iter().map(f).collect());
    d.set_item("part_nugget", one(&|g| g.nugget))?;
    d.set_item("part_loglik", one(&|g| g.loglik))?;
    d.set_item("part_f", one(&|g| g.null.as_ref().map_or(f64::NAN, |n| n.f)))?;
    d.set_item("part_p_f", one(&|g| g.null.as_ref().map_or(f64::NAN, |n| n.p)))?;
    let pairs: Vec<f64> = fit.cross.iter().flat_map(|c| [c.i as f64, c.j as f64]).collect();
    d.set_item("cross_pairs", mat(pairs, fit.cross.len(), 2)?)?;
    d.set_item("cross_rssr", vec(fit.cross.iter().map(|c| c.rssr).collect()))?;
    d.set_item("cross_rsse", vec(fit.cross.iter().map(|c| c.rsse).collect()))?;
    d.set_item("npart", npart)?;
    d.set_item("partsize", partsize)?;
    Ok(d)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "parts")?;
    m.add(
        "__doc__",
        "remotePARTS's methods: AR(1) trends, spatial correlation, partitioned GLS",
    )?;
    m.add_function(wrap_pyfunction!(fit_ar_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(ar_residuals, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(sample_partitions, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_parts, &m)?)?;
    Ok(m)
}
