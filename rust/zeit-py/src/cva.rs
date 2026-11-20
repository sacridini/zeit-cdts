//! zeit._core.cva: the Python module of zeit-core's change vector analysis.

use numpy::{
    AllowTypeChange, Element, PyArray1, PyArray3, PyArrayLike1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::cva::{self, CvaOptions, Direction};
use zeit_core::sample64::Sample64;

/// A date's bands `(band, y, x)`, C-contiguous: int16, uint16, int32, float32 or float64.
#[derive(FromPyObject)]
enum Cube<'py> {
    I16(PyReadonlyArray3<'py, i16>),
    U16(PyReadonlyArray3<'py, u16>),
    I32(PyReadonlyArray3<'py, i32>),
    F32(PyReadonlyArray3<'py, f32>),
    F64(PyReadonlyArray3<'py, f64>),
}

fn sample_run<T: Sample64 + Element>(
    py: Python<'_>,
    a: &PyReadonlyArray3<'_, T>,
    b: &PyReadonlyArray3<'_, T>,
    bands: &[usize],
    scale_std: bool,
    stride: usize,
    missing: &[f64],
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let [nb, ny, nx] = *a.shape() else { unreachable!() };
    if b.shape() != a.shape() {
        return Err(PyValueError::new_err("the two dates must have the same shape"));
    }
    if bands.iter().any(|&k| k >= nb) {
        return Err(PyValueError::new_err("bands names a band the dates do not have"));
    }
    let (aa, ba) = (a.as_array(), b.as_array());
    let xs = aa
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("x1 must be C-contiguous"))?;
    let ys = ba
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("x2 must be C-contiguous"))?;
    Ok(py.detach(|| cva::cva_sample(xs, ys, nb, nx, ny, bands, scale_std, stride, missing)))
}

/// The magnitudes of the valid pixels of the lattice sample and the scale of each band.
#[pyfunction]
#[pyo3(signature = (x1, x2, bands, scale_std = false, stride = 1, missing = Vec::new()))]
fn sample<'py>(
    py: Python<'py>,
    x1: Cube<'py>,
    x2: Cube<'py>,
    bands: Vec<usize>,
    scale_std: bool,
    stride: usize,
    missing: Vec<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (rho, scale) = match (&x1, &x2) {
        (Cube::I16(a), Cube::I16(b)) => sample_run(py, a, b, &bands, scale_std, stride, &missing)?,
        (Cube::U16(a), Cube::U16(b)) => sample_run(py, a, b, &bands, scale_std, stride, &missing)?,
        (Cube::I32(a), Cube::I32(b)) => sample_run(py, a, b, &bands, scale_std, stride, &missing)?,
        (Cube::F32(a), Cube::F32(b)) => sample_run(py, a, b, &bands, scale_std, stride, &missing)?,
        (Cube::F64(a), Cube::F64(b)) => sample_run(py, a, b, &bands, scale_std, stride, &missing)?,
        _ => return Err(PyValueError::new_err("the two dates must have the same dtype")),
    };
    Ok((PyArray1::from_vec(py, rho), PyArray1::from_vec(py, scale)))
}

/// The 99.9th percentile and the like: the element `floor(q (N - 1))` of the sorted values.
#[pyfunction]
fn percentile(values: PyArrayLike1<'_, f64, AllowTypeChange>, q: f64) -> PyResult<f64> {
    let v: Vec<f64> = values.as_array().iter().copied().collect();
    if v.is_empty() {
        return Err(PyValueError::new_err("no values"));
    }
    Ok(cva::percentile(&v, q))
}

/// Otsu's threshold of the values, its between-class variance and the bin width.
#[pyfunction]
#[pyo3(signature = (values, nbins = 1024, upper = None))]
fn otsu(values: PyArrayLike1<'_, f64, AllowTypeChange>, nbins: usize, upper: Option<f64>) -> PyResult<(f64, f64, f64)> {
    let v: Vec<f64> = values.as_array().iter().copied().collect();
    if v.is_empty() || nbins < 2 {
        return Err(PyValueError::new_err("Otsu needs values and at least 2 bins"));
    }
    let up = upper.unwrap_or_else(|| cva::percentile(&v, 0.999));
    let (t, sb) = cva::otsu(&v, nbins, up);
    Ok((t, sb, up / nbins as f64))
}

/// The EM threshold of the values: the threshold and a dict with the fit.
#[pyfunction]
#[pyo3(signature = (values, init = 0.2, max_iter = 500, tol = 1e-9))]
fn em<'py>(
    py: Python<'py>,
    values: PyArrayLike1<'py, f64, AllowTypeChange>,
    init: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, Bound<'py, PyDict>)> {
    let v: Vec<f64> = values.as_array().iter().copied().collect();
    let (t, f) = py
        .detach(|| cva::em_threshold(&v, init, max_iter, tol))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("p", f.p.to_vec())?;
    d.set_item("mu", f.mu.to_vec())?;
    d.set_item("sigma", f.sigma.to_vec())?;
    d.set_item("iterations", f.iterations)?;
    d.set_item("converged", f.converged)?;
    Ok((t, d))
}

fn apply_run<'py, T: Sample64 + Element>(
    py: Python<'py>,
    a: &PyReadonlyArray3<'py, T>,
    b: &PyReadonlyArray3<'py, T>,
    o: &CvaOptions,
    missing: &[f64],
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let [nb, ny, nx] = *a.shape() else { unreachable!() };
    if b.shape() != a.shape() {
        return Err(PyValueError::new_err("the two dates must have the same shape"));
    }
    let (aa, ba) = (a.as_array(), b.as_array());
    let xs = aa
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("x1 must be C-contiguous"))?;
    let ys = ba
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("x2 must be C-contiguous"))?;
    let npix = ny * nx;
    // SAFETY: the core writes every value before anything reads it.
    let out = unsafe { PyArray3::<f32>::new(py, [4, ny, nx], false) };
    {
        let mut w = out.readwrite();
        let o_all = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let (m, rest) = o_all.split_at_mut(npix);
        let (d, rest) = rest.split_at_mut(npix);
        let (c, s) = rest.split_at_mut(npix);
        py.detach(|| cva::cva_batch(xs, ys, nb, npix, o, missing, m, d, c, s, n_jobs));
    }
    Ok(out)
}

/// The magnitude, direction, change and sector planes `(4, y, x)` of two dates `(band, y, x)`. `direction` is
/// `"c2va"`, `"polar"` (`polar_bands` the two positions in `bands`, `n_sectors`) or `"sign"` (`roles` the positions
/// of red, NIR and SWIR1 in `bands`).
#[pyfunction]
#[pyo3(signature = (x1, x2, bands, scale, direction, threshold, polar_bands = None, n_sectors = 4, roles = None,
    edges = None, missing = Vec::new(), n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn apply<'py>(
    py: Python<'py>,
    x1: Cube<'py>,
    x2: Cube<'py>,
    bands: Vec<usize>,
    scale: Vec<f64>,
    direction: &str,
    threshold: f64,
    polar_bands: Option<(usize, usize)>,
    n_sectors: usize,
    roles: Option<(usize, usize, usize)>,
    edges: Option<Vec<f64>>,
    missing: Vec<f64>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if scale.len() != bands.len() || bands.is_empty() || bands.len() > 64 {
        return Err(PyValueError::new_err(
            "scale must have one value per band (1 to 64 bands)",
        ));
    }
    let dir = match direction {
        "c2va" => Direction::C2va,
        "polar" => {
            let (b1, b2) = polar_bands.ok_or_else(|| PyValueError::new_err("polar needs polar_bands"))?;
            if b1 >= bands.len() || b2 >= bands.len() || n_sectors == 0 {
                return Err(PyValueError::new_err("polar_bands out of the bands"));
            }
            Direction::Polar { b1, b2, n_sectors }
        }
        "sign" => {
            let (red, nir, swir1) =
                roles.ok_or_else(|| PyValueError::new_err("sign needs the roles red, nir and swir1"))?;
            if red.max(nir).max(swir1) >= bands.len() {
                return Err(PyValueError::new_err("roles out of the bands"));
            }
            Direction::Sign { red, nir, swir1 }
        }
        _ => return Err(PyValueError::new_err("direction must be 'c2va', 'polar' or 'sign'")),
    };
    if let Some(e) = &edges {
        if e.len() < 2 {
            return Err(PyValueError::new_err("edges needs at least two values"));
        }
    }
    let o = CvaOptions {
        bands: &bands,
        scale: &scale,
        direction: dir,
        threshold,
        c2va_edges: edges.as_deref(),
    };
    match (&x1, &x2) {
        (Cube::I16(a), Cube::I16(b)) => apply_run(py, a, b, &o, &missing, n_jobs),
        (Cube::U16(a), Cube::U16(b)) => apply_run(py, a, b, &o, &missing, n_jobs),
        (Cube::I32(a), Cube::I32(b)) => apply_run(py, a, b, &o, &missing, n_jobs),
        (Cube::F32(a), Cube::F32(b)) => apply_run(py, a, b, &o, &missing, n_jobs),
        (Cube::F64(a), Cube::F64(b)) => apply_run(py, a, b, &o, &missing, n_jobs),
        _ => Err(PyValueError::new_err("the two dates must have the same dtype")),
    }
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "cva")?;
    m.add("__doc__", "Change vector analysis: magnitude, direction and thresholds")?;
    m.add_function(wrap_pyfunction!(sample, &m)?)?;
    m.add_function(wrap_pyfunction!(percentile, &m)?)?;
    m.add_function(wrap_pyfunction!(otsu, &m)?)?;
    m.add_function(wrap_pyfunction!(em, &m)?)?;
    m.add_function(wrap_pyfunction!(apply, &m)?)?;
    Ok(m)
}
