//! zeit._core.improphe: the Python module (ImproPhe, FORCE's CF- and L2-ImproPhe).

use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::improphe::{self, Params};

fn slice<'a, D: numpy::ndarray::Dimension, T: numpy::Element>(
    a: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

fn check_windows(dwin: &[i32]) -> PyResult<()> {
    if dwin.len() < 2 {
        return Err(PyValueError::new_err("seasons needs at least two limits"));
    }
    Ok(())
}

/// The seasonal means of `x` `(time, band, y, x)` float32 (NaN: nodata): `(window * band, y, x)` float32, and the
/// number of pixels with any scene.
#[pyfunction]
#[pyo3(signature = (x, doy, year, dwin, ywin = None, force = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn season_means<'py>(
    py: Python<'py>,
    x: PyReadonlyArray4<'py, f32>,
    doy: Vec<i32>,
    year: Vec<i32>,
    dwin: Vec<i32>,
    ywin: Option<i32>,
    force: bool,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray3<f32>>, usize)> {
    check_windows(&dwin)?;
    let [nt, nb, ny, nx] = *x.shape() else { unreachable!() };
    if doy.len() != nt || year.len() != nt {
        return Err(PyValueError::new_err("doy and year must have one value per time step"));
    }
    let xs = slice(&x, "x")?;
    let nwin = dwin.len() - 1;
    // SAFETY: the core writes every value.
    let out = unsafe { PyArray3::<f32>::new(py, [nwin * nb, ny, nx], false) };
    let n = {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        py.detach(|| improphe::average_season(xs, nt, nb, ny * nx, &doy, &year, &dwin, ywin, force, o, n_jobs))
    };
    Ok((out, n))
}

/// The fine data of the year `ywin` (`None`: all) as components and texture: `(components (npc, y, x), texture
/// (y, x), number of pixels with all the seasonal bands, all the eigenvalues)`, or `None` when there are no
/// components (no scene, or fewer than two valid pixels).
#[pyfunction]
#[pyo3(signature = (x, doy, year, dwin, ywin, h, ksd, min_var = 0.975, force = false, n_jobs = -1, force_pca = None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn prepare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray4<'py, f32>,
    doy: Vec<i32>,
    year: Vec<i32>,
    dwin: Vec<i32>,
    ywin: Option<i32>,
    h: usize,
    ksd: usize,
    min_var: f32,
    force: bool,
    n_jobs: i32,
    force_pca: Option<bool>,
) -> PyResult<
    Option<(
        Bound<'py, PyArray3<f32>>,
        Bound<'py, numpy::PyArray2<f32>>,
        usize,
        Bound<'py, PyArray1<f64>>,
    )>,
> {
    check_windows(&dwin)?;
    let [nt, nb, ny, nx] = *x.shape() else { unreachable!() };
    if doy.len() != nt || year.len() != nt {
        return Err(PyValueError::new_err("doy and year must have one value per time step"));
    }
    let xs = slice(&x, "x")?;
    let params = Params {
        dwin,
        h,
        ksd,
        min_var,
        force,
        force_pca: force_pca.unwrap_or(force),
    };
    let prep = py.detach(|| improphe::prepare_fine(xs, nt, nb, ny, nx, &doy, &year, ywin, &params, n_jobs));
    let Some(p) = prep else { return Ok(None) };
    let pcs = PyArray1::from_vec(py, p.pcs).reshape([p.npc, ny, nx])?;
    let tex = PyArray1::from_vec(py, p.tex).reshape([ny, nx])?;
    let ev = PyArray1::from_vec(py, p.eigenvalues);
    Ok(Some((pcs, tex, p.n_valid, ev)))
}

/// The prediction of the coarse bands `mr` `(band, y, x)` float32 (NaN: nodata) from the components `pcs`
/// `(npc, y, x)` and the texture `tex` `(y, x)` of the fine data: `(band, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (pcs, tex, mr, h, ksd, n_jobs = -1))]
fn predict<'py>(
    py: Python<'py>,
    pcs: PyReadonlyArray3<'py, f32>,
    tex: numpy::PyReadonlyArray2<'py, f32>,
    mr: PyReadonlyArray3<'py, f32>,
    h: usize,
    ksd: usize,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let [npc, ny, nx] = *pcs.shape() else { unreachable!() };
    let [nb, my, mx] = *mr.shape() else { unreachable!() };
    if [my, mx] != [ny, nx] || tex.shape() != [ny, nx] {
        return Err(PyValueError::new_err("pcs, tex and mr must share the (y, x) grid"));
    }
    if npc == 0 || nb == 0 {
        return Err(PyValueError::new_err("pcs and mr need at least one band"));
    }
    if h < 1 {
        return Err(PyValueError::new_err("the kernel radius must be at least one pixel"));
    }
    let (pcs_s, tex_s, mr_s) = (slice(&pcs, "pcs")?, slice(&tex, "tex")?, slice(&mr, "mr")?);
    // SAFETY: the core writes every value.
    let out = unsafe { PyArray3::<f32>::new(py, [nb, ny, nx], false) };
    {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        py.detach(|| {
            let params = Params {
                dwin: vec![1, 365],
                h,
                ksd,
                min_var: 1.0,
                force: false,
                force_pca: false,
            };
            improphe::predict_planes(pcs_s, npc, tex_s, mr_s, nb, ny, nx, &params, o, n_jobs)
        });
    }
    Ok(out)
}

/// `focal_sd` of the planes of `data` `(band, y, x)` float32 over `bands`: `(y, x)` float32.
#[pyfunction]
#[pyo3(signature = (data, k, start = 0, stop = None, n_jobs = -1))]
fn focal_sd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    k: usize,
    start: usize,
    stop: Option<usize>,
    n_jobs: i32,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let [nb, ny, nx] = *data.shape() else { unreachable!() };
    let stop = stop.unwrap_or(nb);
    if start > stop || stop > nb {
        return Err(PyValueError::new_err("bands out of range"));
    }
    let ds = slice(&data, "data")?;
    let mut out = vec![0f32; ny * nx];
    py.detach(|| improphe::focal_sd(ds, ny, nx, k, start..stop, &mut out, n_jobs));
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// FORCE's `mink` of a kernel radius `h` in pixels.
#[pyfunction]
fn min_neighbors(h: usize) -> usize {
    improphe::min_neighbors(h)
}

/// `gsl_eigen_symmv` + `gsl_eigen_symmv_sort(VAL_DESC)` of a symmetric `(n, n)` float64 matrix (its lower
/// triangle): `(eigenvalues, eigenvectors as columns)`.
#[pyfunction]
fn gsl_eigen<'py>(
    py: Python<'py>,
    a: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, numpy::PyArray2<f64>>)> {
    let [n, m] = *a.shape() else { unreachable!() };
    if n != m || n == 0 {
        return Err(PyValueError::new_err("a must be square"));
    }
    let mut work = slice(&a, "a")?.to_vec();
    let (e, v) = zeit_core::gsl_symmv::eigen_symmv_desc(&mut work, n);
    Ok((PyArray1::from_vec(py, e), PyArray1::from_vec(py, v).reshape([n, n])?))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "improphe")?;
    m.add("__doc__", "ImproPhe (FORCE's CF- and L2-ImproPhe)")?;
    m.add_function(wrap_pyfunction!(season_means, &m)?)?;
    m.add_function(wrap_pyfunction!(prepare, &m)?)?;
    m.add_function(wrap_pyfunction!(predict, &m)?)?;
    m.add_function(wrap_pyfunction!(focal_sd, &m)?)?;
    m.add_function(wrap_pyfunction!(gsl_eigen, &m)?)?;
    m.add_function(wrap_pyfunction!(min_neighbors, &m)?)?;
    Ok(m)
}
