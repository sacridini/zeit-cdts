//! zeit._core.imad: the Python module of zeit-core's iMAD (iteratively re-weighted MAD) and the radiometric
//! normalization from its invariant pixels.

use numpy::{
    AllowTypeChange, Element, PyArray1, PyArray3, PyArrayLike1, PyArrayLike2, PyArrayMethods, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::imad::{self, ImadOptions, Penalty, Transform};
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

fn penalty(name: &str) -> PyResult<Penalty> {
    match name {
        "ridge" => Ok(Penalty::Ridge),
        "curvature" => Ok(Penalty::Curvature),
        _ => Err(PyValueError::new_err(format!(
            "penalty must be 'ridge' or 'curvature', got {name:?}"
        ))),
    }
}

fn transform(n: usize, mean: &[f64], a: &[f64], b: &[f64], sd: &[f64]) -> PyResult<Transform> {
    if mean.len() != 2 * n || a.len() != n * n || b.len() != n * n || sd.len() != n {
        return Err(PyValueError::new_err(
            "the transform's arrays do not match the number of bands",
        ));
    }
    Ok(Transform {
        n,
        mean: mean.to_vec(),
        a: a.to_vec(),
        b: b.to_vec(),
        sd: sd.to_vec(),
    })
}

fn vec1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, v)
}

fn mat<'py>(py: Python<'py>, v: Vec<f64>, r: usize, c: usize) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    Ok(PyArray1::from_vec(py, v).reshape([r, c])?)
}

/// The iterations on the valid pixels `z` `(pixels, 2 bands)`: a dict with the transformation (`mean`, `a`, `b`,
/// `sd`), the correlations of every iteration (`rho_history`, `max_delta`), `iterations`, `converged`, `cov`,
/// `loadings_x`, `loadings_y`, `n_pixels` and `warnings`.
#[pyfunction]
#[pyo3(signature = (z, n, max_iter = 50, tol = 1e-6, lam = 0.0, penalty_name = "ridge", weighted_sd = true, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn fit<'py>(
    py: Python<'py>,
    z: PyArrayLike2<'py, f64, AllowTypeChange>,
    n: usize,
    max_iter: usize,
    tol: f64,
    lam: f64,
    penalty_name: &str,
    weighted_sd: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = ImadOptions {
        max_iter,
        tol,
        lam,
        penalty: penalty(penalty_name)?,
        weighted_sd,
    };
    let za = z.as_array();
    if za.shape()[1] != 2 * n {
        return Err(PyValueError::new_err(format!("z must have {} columns", 2 * n)));
    }
    let zs: Vec<f64> = za.iter().copied().collect();
    let f = py
        .detach(|| imad::imad_fit(&zs, n, &opts, n_jobs))
        .map_err(PyValueError::new_err)?;
    let (lx, ly) = imad::loadings(&f);
    let d = PyDict::new(py);
    let k = f.rho_history.len();
    d.set_item("rho_history", mat(py, f.rho_history.concat(), k, n)?)?;
    d.set_item("max_delta", vec1(py, f.max_delta.clone()))?;
    d.set_item("rho", vec1(py, f.cca.rho.clone()))?;
    d.set_item("iterations", f.iterations)?;
    d.set_item("converged", f.converged)?;
    d.set_item("mean", vec1(py, f.transform.mean.clone()))?;
    d.set_item("a", mat(py, f.transform.a.clone(), n, n)?)?;
    d.set_item("b", mat(py, f.transform.b.clone(), n, n)?)?;
    d.set_item("sd", vec1(py, f.transform.sd.clone()))?;
    d.set_item("cov", mat(py, f.cov.clone(), 2 * n, 2 * n)?)?;
    d.set_item("loadings_x", mat(py, lx, n, n)?)?;
    d.set_item("loadings_y", mat(py, ly, n, n)?)?;
    d.set_item("n_pixels", f.n_pixels)?;
    d.set_item("warnings", f.warnings.clone())?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn run<'py, T: Sample64 + Element>(
    py: Python<'py>,
    x: &PyReadonlyArray3<'py, T>,
    y: &PyReadonlyArray3<'py, T>,
    t: &Transform,
    missing: &[f64],
    canonical: bool,
    n_jobs: i32,
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Option<Bound<'py, PyArray3<f32>>>,
)> {
    let [n, ny, nx] = *x.shape() else { unreachable!() };
    if y.shape() != x.shape() {
        return Err(PyValueError::new_err("the two dates must have the same shape"));
    }
    if t.n != n {
        return Err(PyValueError::new_err("the transform is for another number of bands"));
    }
    let npix = ny * nx;
    let (xa, ya) = (x.as_array(), y.as_array());
    let xs = xa
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = ya
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("y must be C-contiguous"))?;
    // SAFETY: the core writes every value before anything reads it.
    let mad = unsafe { PyArray3::<f32>::new(py, [n, ny, nx], false) };
    let chi2 = unsafe { PyArray3::<f32>::new(py, [1, ny, nx], false) };
    let pnc = unsafe { PyArray3::<f32>::new(py, [1, ny, nx], false) };
    let cv = canonical.then(|| unsafe { PyArray3::<f32>::new(py, [2 * n, ny, nx], false) });
    {
        let (mut wm, mut wc, mut wp) = (mad.readwrite(), chi2.readwrite(), pnc.readwrite());
        let mut wcv = cv.as_ref().map(|c| c.readwrite());
        let om = wm
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let oc = wc
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let op = wp
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let ocv = match wcv.as_mut() {
            Some(w) => Some(
                w.as_slice_mut()
                    .map_err(|_| PyValueError::new_err("output not contiguous"))?,
            ),
            None => None,
        };
        py.detach(|| imad::imad_apply(xs, ys, n, npix, t, missing, om, oc, op, ocv, n_jobs));
    }
    Ok((mad, chi2, pnc, cv))
}

/// The MAD variates `(bands, y, x)`, the chi-square and the probability of no change `(1, y, x)` and, with
/// `canonical`, the canonical variates `(2 bands, y, x)` (U then V) of two dates `x`, `y` `(band, y, x)`
/// (C-contiguous, the same dtype) under the transformation of `fit`.
#[pyfunction]
#[pyo3(signature = (x, y, mean, a, b, sd, missing = Vec::new(), canonical = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn apply<'py>(
    py: Python<'py>,
    x: Cube<'py>,
    y: Cube<'py>,
    mean: PyArrayLike1<'py, f64, AllowTypeChange>,
    a: PyArrayLike2<'py, f64, AllowTypeChange>,
    b: PyArrayLike2<'py, f64, AllowTypeChange>,
    sd: PyArrayLike1<'py, f64, AllowTypeChange>,
    missing: Vec<f64>,
    canonical: bool,
    n_jobs: i32,
) -> PyResult<(
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Option<Bound<'py, PyArray3<f32>>>,
)> {
    let n = sd.as_array().len();
    let t = transform(
        n,
        &mean.as_array().iter().copied().collect::<Vec<_>>(),
        &a.as_array().iter().copied().collect::<Vec<_>>(),
        &b.as_array().iter().copied().collect::<Vec<_>>(),
        &sd.as_array().iter().copied().collect::<Vec<_>>(),
    )?;
    match (&x, &y) {
        (Cube::I16(p), Cube::I16(q)) => run(py, p, q, &t, &missing, canonical, n_jobs),
        (Cube::U16(p), Cube::U16(q)) => run(py, p, q, &t, &missing, canonical, n_jobs),
        (Cube::I32(p), Cube::I32(q)) => run(py, p, q, &t, &missing, canonical, n_jobs),
        (Cube::F32(p), Cube::F32(q)) => run(py, p, q, &t, &missing, canonical, n_jobs),
        (Cube::F64(p), Cube::F64(q)) => run(py, p, q, &t, &missing, canonical, n_jobs),
        _ => Err(PyValueError::new_err("the two dates must have the same dtype")),
    }
}

/// The normalization fit from the sample `z` `(pixels, 2 bands)`: the invariant pixels (probability of no change
/// above `threshold`), every `holdout_every`-th kept for the tests (0: none); a list with a dict for each band and
/// the number of invariant pixels.
#[pyfunction]
#[pyo3(signature = (z, mean, a, b, sd, threshold = 0.95, holdout_every = 3, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn radcal_fit<'py>(
    py: Python<'py>,
    z: PyArrayLike2<'py, f64, AllowTypeChange>,
    mean: PyArrayLike1<'py, f64, AllowTypeChange>,
    a: PyArrayLike2<'py, f64, AllowTypeChange>,
    b: PyArrayLike2<'py, f64, AllowTypeChange>,
    sd: PyArrayLike1<'py, f64, AllowTypeChange>,
    threshold: f64,
    holdout_every: usize,
    n_jobs: i32,
) -> PyResult<(Vec<Bound<'py, PyDict>>, usize)> {
    let n = sd.as_array().len();
    let t = transform(
        n,
        &mean.as_array().iter().copied().collect::<Vec<_>>(),
        &a.as_array().iter().copied().collect::<Vec<_>>(),
        &b.as_array().iter().copied().collect::<Vec<_>>(),
        &sd.as_array().iter().copied().collect::<Vec<_>>(),
    )?;
    let zs: Vec<f64> = z.as_array().iter().copied().collect();
    let (bands, invariant) = py
        .detach(|| imad::radcal_fit(&zs, n, &t, threshold, holdout_every, n_jobs))
        .map_err(PyValueError::new_err)?;
    let mut rows = Vec::with_capacity(n);
    for bf in &bands {
        let d = PyDict::new(py);
        d.set_item("a", bf.fit.a)?;
        d.set_item("se_a", bf.fit.se_a)?;
        d.set_item("b", bf.fit.b)?;
        d.set_item("se_b", bf.fit.se_b)?;
        d.set_item("r", bf.fit.r)?;
        d.set_item("rmse", bf.fit.rmse)?;
        d.set_item("n_train", bf.n_train)?;
        d.set_item("n_test", bf.n_test)?;
        match &bf.test {
            Some(h) => {
                d.set_item("mean_target", h.mean_target)?;
                d.set_item("mean_ref", h.mean_ref)?;
                d.set_item("mean_norm", h.mean_norm)?;
                d.set_item("t", h.t)?;
                d.set_item("p_t", h.p_t)?;
                d.set_item("var_target", h.var_target)?;
                d.set_item("var_ref", h.var_ref)?;
                d.set_item("var_norm", h.var_norm)?;
                d.set_item("F", h.f)?;
                d.set_item("p_F", h.p_f)?;
            }
            None => {
                for k in [
                    "mean_target",
                    "mean_ref",
                    "mean_norm",
                    "t",
                    "p_t",
                    "var_target",
                    "var_ref",
                    "var_norm",
                    "F",
                    "p_F",
                ] {
                    d.set_item(k, f64::NAN)?;
                }
            }
        }
        rows.push(d);
    }
    Ok((rows, invariant))
}

/// The orthogonal regression of `v` on `u`: `(a, b, se_a, se_b, r, rmse)`.
#[pyfunction]
fn orthogonal_regression(
    u: PyArrayLike1<'_, f64, AllowTypeChange>,
    v: PyArrayLike1<'_, f64, AllowTypeChange>,
) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    let (u, v): (Vec<f64>, Vec<f64>) = (
        u.as_array().iter().copied().collect(),
        v.as_array().iter().copied().collect(),
    );
    let f = imad::orthogonal_regression(&u, &v).map_err(PyValueError::new_err)?;
    Ok((f.a, f.b, f.se_a, f.se_b, f.r, f.rmse))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "imad")?;
    m.add(
        "__doc__",
        "iMAD, the iteratively re-weighted MAD, and the radiometric normalization from its invariant pixels",
    )?;
    m.add_function(wrap_pyfunction!(fit, &m)?)?;
    m.add_function(wrap_pyfunction!(apply, &m)?)?;
    m.add_function(wrap_pyfunction!(radcal_fit, &m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_regression, &m)?)?;
    Ok(m)
}
