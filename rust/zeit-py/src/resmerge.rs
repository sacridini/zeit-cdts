//! zeit._core.resmerge: the Python module (Sentinel-2 resolution merge, FORCE's three methods).

use numpy::{PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::resmerge::{self, Scene, NT};
use zeit_core::starfm::{self, StarfmParams};

fn slice<'a, D: numpy::ndarray::Dimension, T: numpy::Element>(
    a: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

/// The six 20 m bands `targets` `(6, y, x)` float32 predicted at the pixels that are neither `off` nor `bad`, from
/// `green`, `red` and `bnir` `(y, x)` float32, all on the 10 m grid; the values divided by `denom` are reflectance.
/// `method` is "regression", "improphe" or "starfm"; `force` is FORCE's arithmetic (truncated integers, the regression in
/// place on one thread). `(6, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (green, red, bnir, targets, off, bad, method, denom = 10000.0, force = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn resmerge_batch<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f32>,
    red: PyReadonlyArray2<'py, f32>,
    bnir: PyReadonlyArray2<'py, f32>,
    targets: PyReadonlyArray3<'py, f32>,
    off: PyReadonlyArray2<'py, bool>,
    bad: PyReadonlyArray2<'py, bool>,
    method: &str,
    denom: f64,
    force: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let [ny, nx] = *green.shape() else { unreachable!() };
    for (shape, what) in [
        (red.shape(), "red"),
        (bnir.shape(), "bnir"),
        (off.shape(), "off"),
        (bad.shape(), "bad"),
    ] {
        if shape != [ny, nx] {
            return Err(PyValueError::new_err(format!(
                "{what} must be (y, x) = ({ny}, {nx}), got {shape:?}"
            )));
        }
    }
    if targets.shape() != [NT, ny, nx] {
        return Err(PyValueError::new_err(format!(
            "targets must be (6, y, x) = (6, {ny}, {nx}), got {:?}",
            targets.shape()
        )));
    }
    if !(denom > 0.0) || !denom.is_finite() {
        return Err(PyValueError::new_err("denom must be positive"));
    }
    let run: fn(&Scene, f64, bool, i32) -> Vec<f32> = match method {
        "regression" => resmerge::regression,
        "improphe" => resmerge::improphe,
        "starfm" => starfm::resolution_merge_3,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'regression', 'improphe' or 'starfm', got {other:?}"
            )))
        }
    };
    let sc = Scene {
        nx,
        ny,
        green: slice(&green, "green")?,
        red: slice(&red, "red")?,
        bnir: slice(&bnir, "bnir")?,
        targets: slice(&targets, "targets")?,
        off: slice(&off, "off")?,
        bad: slice(&bad, "bad")?,
    };
    let out = py.detach(|| run(&sc, denom, force, n_jobs));
    let arr = PyArray3::<f32>::zeros(py, [NT, ny, nx], false);
    {
        let mut w = arr.readwrite();
        w.as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?
            .copy_from_slice(&out);
    }
    Ok(arr)
}

/// STARFM (FORCE's `starfm`) for any number of pairs and prediction dates, band by band. `fine` is `(pairs, bands, y,
/// x)` float32, `coarse` `(pairs + targets, bands, y, x)`: the pairs' coarse images, then the ones at the dates to
/// predict (reflectance, 0-1: the thresholds are). A pixel with a NaN in any pair or in the target's coarse image
/// (of that band) is off: not used and not predicted (NaN). `(targets, bands, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (fine, coarse, radius, nclass = 40.0, unc_coarse = 0.002, unc_fine = 0.002, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn starfm_batch<'py>(
    py: Python<'py>,
    fine: PyReadonlyArray4<'py, f32>,
    coarse: PyReadonlyArray4<'py, f32>,
    radius: usize,
    nclass: f32,
    unc_coarse: f32,
    unc_fine: f32,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let [np, nb, ny, nx] = *fine.shape() else { unreachable!() };
    let [nc, cb, cy, cx] = *coarse.shape() else { unreachable!() };
    if np < 1 || nc <= np || cb != nb || cy != ny || cx != nx {
        return Err(PyValueError::new_err(format!(
            "fine is (pairs, bands, y, x) = {:?} and coarse (pairs + targets, bands, y, x) with at least one more image, got {:?}",
            fine.shape(),
            coarse.shape()
        )));
    }
    if !(nclass > 0.0) || !nclass.is_finite() || !(unc_coarse >= 0.0) || !(unc_fine >= 0.0) {
        return Err(PyValueError::new_err("nclass must be positive and the uncertainties not negative"));
    }
    let nt = nc - np;
    let npix = ny * nx;
    let fs = slice(&fine, "fine")?;
    let cs = slice(&coarse, "coarse")?;
    let par = StarfmParams { radius, nclass, unc_coarse, unc_fine };
    let out = py.detach(|| {
        let mut out = vec![f32::NAN; nt * nb * npix];
        let mut off = vec![false; npix];
        for t in 0..nt {
            for b in 0..nb {
                fn img(a: &[f32], k: usize, nb: usize, b: usize, npix: usize) -> &[f32] {
                    &a[(k * nb + b) * npix..(k * nb + b + 1) * npix]
                }
                let target = img(cs, np + t, nb, b, npix);
                let fines: Vec<&[f32]> = (0..np).map(|k| img(fs, k, nb, b, npix)).collect();
                let mut coarses: Vec<&[f32]> = (0..np).map(|k| img(cs, k, nb, b, npix)).collect();
                coarses.push(target);
                for p in 0..npix {
                    off[p] = !target[p].is_finite()
                        || fines.iter().any(|a| !a[p].is_finite())
                        || coarses[..np].iter().any(|a| !a[p].is_finite());
                }
                let pred = starfm::starfm(nx, ny, &off, &coarses, &fines, &par, n_jobs);
                let dst = &mut out[(t * nb + b) * npix..(t * nb + b + 1) * npix];
                for p in 0..npix {
                    if !off[p] {
                        dst[p] = pred[p];
                    }
                }
            }
        }
        out
    });
    let arr = PyArray4::<f32>::zeros(py, [nt, nb, ny, nx], false);
    {
        let mut w = arr.readwrite();
        w.as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?
            .copy_from_slice(&out);
    }
    Ok(arr)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "resmerge")?;
    m.add(
        "__doc__",
        "Sentinel-2 resolution merge (FORCE's regression, ImproPhe and STARFM methods)",
    )?;
    m.add_function(wrap_pyfunction!(resmerge_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(starfm_batch, &m)?)?;
    Ok(m)
}
