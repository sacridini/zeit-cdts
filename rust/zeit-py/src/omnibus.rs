//! zeit._core.omnibus: the Python module of zeit-core's omnibus change test for Sentinel-1 series.

use numpy::{Element, PyArray3, PyArrayMethods, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::omnibus::{self, Approx, OmnibusOptions};
use zeit_core::sample64::Sample64;

/// The series `(time, band, y, x)`, C-contiguous: int16, uint16, int32, float32 or float64.
#[derive(FromPyObject)]
enum Cube<'py> {
    I16(PyReadonlyArray4<'py, i16>),
    U16(PyReadonlyArray4<'py, u16>),
    I32(PyReadonlyArray4<'py, i32>),
    F32(PyReadonlyArray4<'py, f32>),
    F64(PyReadonlyArray4<'py, f64>),
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run<'py, T: Sample64 + Element>(
    py: Python<'py>,
    x: &PyReadonlyArray4<'py, T>,
    bands: &[usize],
    o: &OmnibusOptions,
    per_date: bool,
    missing: &[f64],
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Option<Bound<'py, PyArray3<f32>>>)> {
    let [k, nb, ny, nx] = *x.shape() else { unreachable!() };
    if bands.iter().any(|&b| b >= nb) {
        return Err(PyValueError::new_err("bands names a band the series does not have"));
    }
    let xa = x.as_array();
    let xs = xa
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("the series must be C-contiguous"))?;
    let npix = ny * nx;
    // SAFETY: the core writes every value before anything reads it.
    let out = unsafe { PyArray3::<f32>::new(py, [omnibus::PLANES.len(), ny, nx], false) };
    let per = per_date.then(|| unsafe { PyArray3::<f32>::new(py, [k, ny, nx], false) });
    {
        let mut w = out.readwrite();
        let o_all = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let mut wp = per.as_ref().map(|p| p.readwrite());
        let pd = match wp.as_mut() {
            Some(w) => Some(
                w.as_slice_mut()
                    .map_err(|_| PyValueError::new_err("output not contiguous"))?,
            ),
            None => None,
        };
        py.detach(|| omnibus::omnibus_batch(xs, nb, npix, k, bands, o, missing, o_all, pd, n_jobs));
    }
    Ok((out, per))
}

/// The omnibus change planes `(8, y, x)` of a series `(time, band, y, x)` of linear power, in the order of
/// `zeit_core::omnibus::PLANES` (dates as indices of the series), and with `per_date` the direction at each date
/// `(time, y, x)`. `bands` are the positions of VV (and VH); `approximation` is `"improved"` or `"chi2"`.
#[pyfunction]
#[pyo3(signature = (x, bands, enl = 4.4, alpha = 0.01, approximation = "improved", per_date = false,
    missing = Vec::new(), n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn apply<'py>(
    py: Python<'py>,
    x: Cube<'py>,
    bands: Vec<usize>,
    enl: f64,
    alpha: f64,
    approximation: &str,
    per_date: bool,
    missing: Vec<f64>,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Option<Bound<'py, PyArray3<f32>>>)> {
    if bands.is_empty() || bands.len() > 2 {
        return Err(PyValueError::new_err("bands must be one (VV) or two (VV, VH) bands"));
    }
    if !(enl > 0.0) || !(alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err("enl must be positive and alpha between 0 and 1"));
    }
    let approx = match approximation {
        "improved" => Approx::Improved,
        "chi2" => Approx::Chi2,
        _ => return Err(PyValueError::new_err("approximation must be 'improved' or 'chi2'")),
    };
    let o = OmnibusOptions {
        enl,
        alpha,
        approx,
        p: bands.len(),
    };
    match &x {
        Cube::I16(a) => run(py, a, &bands, &o, per_date, &missing, n_jobs),
        Cube::U16(a) => run(py, a, &bands, &o, per_date, &missing, n_jobs),
        Cube::I32(a) => run(py, a, &bands, &o, per_date, &missing, n_jobs),
        Cube::F32(a) => run(py, a, &bands, &o, per_date, &missing, n_jobs),
        Cube::F64(a) => run(py, a, &bands, &o, per_date, &missing, n_jobs),
    }
}

/// `(f, rho, omega2)` of the omnibus test of `k` dates (`kind = "q"`) or `(g, rho_j, omega2_j)` of the test of date
/// `k` against those before it (`kind = "r"`), for `p` bands.
#[pyfunction]
fn params(kind: &str, p: usize, enl: f64, k: usize) -> PyResult<(f64, f64, f64)> {
    match kind {
        "q" if k >= 2 => Ok(omnibus::q_params(p, enl, k)),
        "r" if k >= 2 => Ok(omnibus::rj_params(p, enl, k)),
        _ => Err(PyValueError::new_err("kind must be 'q' or 'r' and k at least 2")),
    }
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "omnibus")?;
    m.add("__doc__", "The omnibus change test for Sentinel-1 series")?;
    m.add_function(wrap_pyfunction!(apply, &m)?)?;
    m.add_function(wrap_pyfunction!(params, &m)?)?;
    Ok(m)
}
