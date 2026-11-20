//! zeit._core.mnf: the Python module of zeit-core's Minimum Noise Fraction (statistics in blocks, the fit, the
//! forward and inverse transforms).

use numpy::{
    Element, PyArray1, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::mnf::{self, Dropped, MnfError, MnfStats, MnfTransform, Noise, Scale, StatsOptions};

fn err(e: MnfError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn noise_of(name: &str, given: Option<PyReadonlyArray2<'_, f64>>) -> PyResult<Noise> {
    Ok(match name {
        "diff_h" => Noise::DiffH,
        "diff_v" => Noise::DiffV,
        "diff_hv" => Noise::DiffHv,
        "diff_diag" => Noise::DiffDiag,
        "diff_antidiag" => Noise::DiffAntidiag,
        "given" => {
            let g = given.ok_or_else(|| PyValueError::new_err("noise='given' needs noise_cov"))?;
            Noise::Given(g.as_array().iter().copied().collect())
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "noise= {other:?}: use 'diff_hv', 'diff_h', 'diff_v', 'diff_diag', 'diff_antidiag' or 'given'"
            )))
        }
    })
}

fn scale_of(name: &str) -> PyResult<Scale> {
    match name {
        "noise" => Ok(Scale::Noise),
        "unit" => Ok(Scale::Unit),
        other => Err(PyValueError::new_err(format!(
            "scale= {other:?}: use 'noise' or 'unit'"
        ))),
    }
}

fn contiguous<'a, T: Element, D: numpy::ndarray::Dimension>(
    a: &'a numpy::ndarray::ArrayView<'a, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    a.to_slice()
        .ok_or_else(|| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

/// The cube `(time, band, y, x)`, C-contiguous: int16, uint16, int32, float32 or float64.
#[derive(FromPyObject)]
enum Cube<'py> {
    I16(PyReadonlyArray4<'py, i16>),
    U16(PyReadonlyArray4<'py, u16>),
    I32(PyReadonlyArray4<'py, i32>),
    F32(PyReadonlyArray4<'py, f32>),
    F64(PyReadonlyArray4<'py, f64>),
}

macro_rules! with_cube {
    ($cube:expr, |$x:ident, $shape:ident| $body:expr) => {
        match &$cube {
            Cube::I16(a) => {
                let $shape = a.shape().to_vec();
                let v = a.as_array();
                let $x = contiguous(&v, "the cube")?;
                $body
            }
            Cube::U16(a) => {
                let $shape = a.shape().to_vec();
                let v = a.as_array();
                let $x = contiguous(&v, "the cube")?;
                $body
            }
            Cube::I32(a) => {
                let $shape = a.shape().to_vec();
                let v = a.as_array();
                let $x = contiguous(&v, "the cube")?;
                $body
            }
            Cube::F32(a) => {
                let $shape = a.shape().to_vec();
                let v = a.as_array();
                let $x = contiguous(&v, "the cube")?;
                $body
            }
            Cube::F64(a) => {
                let $shape = a.shape().to_vec();
                let v = a.as_array();
                let $x = contiguous(&v, "the cube")?;
                $body
            }
        }
    };
}

fn mask_slice<'a>(
    m: &'a Option<PyReadonlyArray2<'_, u8>>,
    shape: (usize, usize),
    what: &str,
) -> PyResult<Option<Vec<u8>>> {
    match m {
        None => Ok(None),
        Some(a) => {
            let s = a.shape();
            if (s[0], s[1]) != shape {
                return Err(PyValueError::new_err(format!(
                    "{what} must be (y, x) = {shape:?}, got {s:?}"
                )));
            }
            Ok(Some(a.as_array().iter().copied().collect()))
        }
    }
}

/// Partial statistics (Chan's update): blocks of rows added in order, merged, then fitted.
#[pyclass(name = "Stats", module = "zeit._core.mnf")]
struct Stats {
    inner: MnfStats,
}

#[pymethods]
impl Stats {
    #[new]
    #[pyo3(signature = (nb, noise = "diff_hv", noise_cov = None))]
    fn new(nb: usize, noise: &str, noise_cov: Option<PyReadonlyArray2<'_, f64>>) -> PyResult<Self> {
        let noise = noise_of(noise, noise_cov)?;
        Ok(Stats {
            inner: MnfStats::new(nb, &noise),
        })
    }

    #[getter]
    fn n(&self) -> u64 {
        self.inner.n
    }

    #[getter]
    fn nb(&self) -> usize {
        self.inner.nb
    }

    #[getter]
    fn band_valid(&self) -> Vec<u64> {
        self.inner.band_valid.clone()
    }

    #[getter]
    fn pairs(&self) -> Vec<u64> {
        self.inner.pairs.iter().map(|d| d.m).collect()
    }

    /// Rows `row0..row0 + rows` of date `t`, `x` `(band, rows, nx)` float64, and the row below (`halo`
    /// `(band, nx)`) unless it is the last of the image of `ny` rows.
    #[pyo3(signature = (x, halo, t, row0, ny, mask = None, noise_region = None, missing = Vec::new(),
                        sample_fraction = 1.0, seed = 0, date0 = 0, active = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_block(
        &mut self,
        x: PyReadonlyArray3<'_, f64>,
        halo: Option<PyReadonlyArray2<'_, f64>>,
        t: usize,
        row0: usize,
        ny: usize,
        mask: Option<PyReadonlyArray2<'_, u8>>,
        noise_region: Option<PyReadonlyArray2<'_, u8>>,
        missing: Vec<f64>,
        sample_fraction: f64,
        seed: u64,
        date0: usize,
        active: Option<Vec<u8>>,
    ) -> PyResult<()> {
        let [nb, rows, nx] = *x.shape() else { unreachable!() };
        if nb != self.inner.nb {
            return Err(PyValueError::new_err("the block has another number of bands"));
        }
        let xv = x.as_array();
        let xs = contiguous(&xv, "the block")?;
        let hv = halo.as_ref().map(|h| h.as_array());
        let hs = match &hv {
            Some(v) => Some(contiguous(v, "the halo")?),
            None => None,
        };
        let mask = mask_slice(&mask, (ny, nx), "mask")?;
        let region = mask_slice(&noise_region, (ny, nx), "noise_region")?;
        let mut opts = StatsOptions::new(Noise::DiffH);
        opts.mask = mask.as_deref();
        opts.noise_region = region.as_deref();
        opts.missing = &missing;
        opts.sample_fraction = sample_fraction;
        opts.seed = seed;
        opts.date0 = date0;
        opts.active = active.as_deref();
        self.inner.add_block(xs, hs, t, row0, rows, ny, nx, &opts);
        Ok(())
    }

    fn merge(&mut self, other: PyRef<'_, Stats>) -> PyResult<()> {
        if other.inner.nb != self.inner.nb || other.inner.pairs.len() != self.inner.pairs.len() {
            return Err(PyValueError::new_err("the statistics are of another transform"));
        }
        self.inner.merge(&other.inner);
        Ok(())
    }

    fn fit(&self) -> PyResult<Transform> {
        Ok(Transform {
            inner: mnf::mnf_fit(&self.inner).map_err(err)?,
        })
    }
}

/// A fitted transform.
#[pyclass(name = "Transform", module = "zeit._core.mnf")]
struct Transform {
    inner: MnfTransform,
}

#[pymethods]
impl Transform {
    /// From the matrices of a fitted transform: `forward` `(m, nb)`, `inverse` `(nb, m)`, the bands left out
    /// (`empty`: no valid value; the others of `dropped` are constant).
    #[staticmethod]
    fn from_parts(
        nb: usize,
        kept: Vec<usize>,
        empty: Vec<usize>,
        mean: PyReadonlyArray1<'_, f64>,
        forward: PyReadonlyArray2<'_, f64>,
        inverse: PyReadonlyArray2<'_, f64>,
        eigenvalues: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let m = kept.len();
        let (fs, is) = (forward.shape(), inverse.shape());
        if mean.len() != nb || fs != [m, nb] || is != [nb, m] || eigenvalues.len() != m {
            return Err(PyValueError::new_err("the matrices of the transform do not agree"));
        }
        let dropped = (0..nb)
            .filter(|j| !kept.contains(j))
            .map(|j| {
                (
                    j,
                    if empty.contains(&j) {
                        Dropped::Empty
                    } else {
                        Dropped::Constant
                    },
                )
            })
            .collect();
        Ok(Transform {
            inner: MnfTransform {
                nb,
                m,
                kept,
                dropped,
                mean: mean.as_array().iter().copied().collect(),
                cov: Vec::new(),
                noise_cov: Vec::new(),
                eigenvalues: eigenvalues.as_array().iter().copied().collect(),
                forward: forward.as_array().iter().copied().collect(),
                inverse: inverse.as_array().iter().copied().collect(),
                correlation: Vec::new(),
                n_pixels: 0,
                n_pairs: Vec::new(),
                noise_floored: 0,
                floored: vec![false; m],
            },
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = &self.inner;
        let (nb, m) = (t.nb, t.m);
        let d = PyDict::new(py);
        d.set_item("kept", t.kept.clone())?;
        d.set_item(
            "dropped",
            t.dropped
                .iter()
                .map(|&(j, r)| (j, if r == Dropped::Empty { "empty" } else { "constant" }))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("mean", PyArray1::from_vec(py, t.mean.clone()))?;
        d.set_item("cov", PyArray1::from_vec(py, t.cov.clone()).reshape([nb, nb])?)?;
        d.set_item(
            "noise_cov",
            PyArray1::from_vec(py, t.noise_cov.clone()).reshape([nb, nb])?,
        )?;
        d.set_item("eigenvalues", PyArray1::from_vec(py, t.eigenvalues.clone()))?;
        d.set_item("forward", PyArray1::from_vec(py, t.forward.clone()).reshape([m, nb])?)?;
        d.set_item("inverse", PyArray1::from_vec(py, t.inverse.clone()).reshape([nb, m])?)?;
        d.set_item(
            "correlation",
            PyArray1::from_vec(py, t.correlation.clone()).reshape([nb, m])?,
        )?;
        d.set_item("n_pixels", t.n_pixels)?;
        d.set_item("n_pairs", t.n_pairs.clone())?;
        d.set_item("noise_floored", t.noise_floored)?;
        d.set_item("floored", t.floored.clone())?;
        Ok(d)
    }
}

/// The statistics of a cube `(time, band, y, x)` (C-contiguous) in blocks of rows on `n_jobs` threads.
#[pyfunction]
#[pyo3(signature = (x, noise = "diff_hv", noise_cov = None, mask = None, noise_region = None, missing = Vec::new(),
                    sample_fraction = 1.0, seed = 0, date0 = 0, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn stats_batch(
    py: Python<'_>,
    x: Cube<'_>,
    noise: &str,
    noise_cov: Option<PyReadonlyArray2<'_, f64>>,
    mask: Option<PyReadonlyArray2<'_, u8>>,
    noise_region: Option<PyReadonlyArray2<'_, u8>>,
    missing: Vec<f64>,
    sample_fraction: f64,
    seed: u64,
    date0: usize,
    n_jobs: i32,
) -> PyResult<Stats> {
    let noise = noise_of(noise, noise_cov)?;
    with_cube!(x, |xs, shape| {
        let (nt, nb, ny, nx) = (shape[0], shape[1], shape[2], shape[3]);
        let mask = mask_slice(&mask, (ny, nx), "mask")?;
        let region = mask_slice(&noise_region, (ny, nx), "noise_region")?;
        let mut opts = StatsOptions::new(noise);
        opts.mask = mask.as_deref();
        opts.noise_region = region.as_deref();
        opts.missing = &missing;
        opts.sample_fraction = sample_fraction;
        opts.seed = seed;
        opts.date0 = date0;
        let inner = py
            .detach(|| mnf::mnf_stats_batch(xs, nt, nb, ny, nx, &opts, n_jobs))
            .map_err(err)?;
        Ok(Stats { inner })
    })
}

/// The transform of given covariances.
#[pyfunction]
fn fit_cov(
    mean: PyReadonlyArray1<'_, f64>,
    cov: PyReadonlyArray2<'_, f64>,
    noise_cov: PyReadonlyArray2<'_, f64>,
) -> PyResult<Transform> {
    let nb = mean.len();
    let m: Vec<f64> = mean.as_array().iter().copied().collect();
    let c: Vec<f64> = cov.as_array().iter().copied().collect();
    let n: Vec<f64> = noise_cov.as_array().iter().copied().collect();
    if cov.shape() != [nb, nb] {
        return Err(err(MnfError::BadInput(format!("cov must be {nb} x {nb}"))));
    }
    if noise_cov.shape() != [nb, nb] {
        return Err(err(MnfError::BadNoiseCov(format!("it must be {nb} x {nb}"))));
    }
    Ok(Transform {
        inner: mnf::mnf_fit_cov(&m, &c, &n, nb).map_err(err)?,
    })
}

fn check_cube(shape: &[usize], tr: &Transform) -> PyResult<()> {
    if shape[1] != tr.inner.nb {
        return Err(PyValueError::new_err(format!(
            "the cube has {} bands, the transform {}",
            shape[1], tr.inner.nb
        )));
    }
    Ok(())
}

/// The components of the cube `x` `(time, band, y, x)`: `(time, k_out, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (x, tr, k_out, scale = "noise", missing = Vec::new(), mask = None, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn forward<'py>(
    py: Python<'py>,
    x: Cube<'py>,
    tr: PyRef<'py, Transform>,
    k_out: usize,
    scale: &str,
    missing: Vec<f64>,
    mask: Option<PyReadonlyArray2<'py, u8>>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let scale = scale_of(scale)?;
    let trc = &tr.inner;
    if k_out > tr.inner.m {
        return Err(PyValueError::new_err(format!(
            "n_components={k_out}: the transform has {} components",
            tr.inner.m
        )));
    }
    with_cube!(x, |xs, shape| {
        check_cube(&shape, &tr)?;
        let (nt, nb, ny, nx) = (shape[0], shape[1], shape[2], shape[3]);
        let mask = mask_slice(&mask, (ny, nx), "mask")?;
        // SAFETY: the core writes every value before anything reads it.
        let out = unsafe { PyArray4::<f32>::new(py, [nt, k_out, ny, nx], false) };
        {
            let mut w = out.readwrite();
            let o = w
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("output not contiguous"))?;
            py.detach(|| {
                mnf::mnf_forward_batch(xs, nb, ny * nx, trc, k_out, scale, &missing, mask.as_deref(), o, n_jobs)
            });
        }
        Ok(out)
    })
}

/// The bands from the components `y` `(time, k_in, y, x)` float32, the first `k` of them: `(time, band, y, x)`.
#[pyfunction]
#[pyo3(signature = (y, tr, k, scale = "noise", n_jobs = -1))]
fn inverse<'py>(
    py: Python<'py>,
    y: PyReadonlyArray4<'py, f32>,
    tr: PyRef<'py, Transform>,
    k: usize,
    scale: &str,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let scale = scale_of(scale)?;
    let trc = &tr.inner;
    let [nt, k_in, ny, nx] = *y.shape() else { unreachable!() };
    if k > k_in || k > tr.inner.m {
        return Err(PyValueError::new_err(format!(
            "k={k}: the components given are {k_in} and the transform has {}",
            tr.inner.m
        )));
    }
    let yv = y.as_array();
    let ys = contiguous(&yv, "the components")?;
    let nb = tr.inner.nb;
    // SAFETY: the core writes every value before anything reads it.
    let out = unsafe { PyArray4::<f32>::new(py, [nt, nb, ny, nx], false) };
    {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        py.detach(|| mnf::mnf_inverse_batch(ys, k_in, ny * nx, trc, k, scale, o, n_jobs));
    }
    Ok(out)
}

/// The cube `x` `(time, band, y, x)` rebuilt from its first `k` components: `(time, band, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (x, tr, k, missing = Vec::new(), mask = None, n_jobs = -1))]
fn denoise<'py>(
    py: Python<'py>,
    x: Cube<'py>,
    tr: PyRef<'py, Transform>,
    k: usize,
    missing: Vec<f64>,
    mask: Option<PyReadonlyArray2<'py, u8>>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let trc = &tr.inner;
    if k > tr.inner.m {
        return Err(PyValueError::new_err(format!(
            "k={k}: the transform has {} components",
            tr.inner.m
        )));
    }
    with_cube!(x, |xs, shape| {
        check_cube(&shape, &tr)?;
        let (nt, nb, ny, nx) = (shape[0], shape[1], shape[2], shape[3]);
        let mask = mask_slice(&mask, (ny, nx), "mask")?;
        // SAFETY: the core writes every value before anything reads it.
        let out = unsafe { PyArray4::<f32>::new(py, [nt, nb, ny, nx], false) };
        {
            let mut w = out.readwrite();
            let o = w
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("output not contiguous"))?;
            py.detach(|| mnf::mnf_denoise_batch(xs, nb, ny * nx, trc, k, &missing, mask.as_deref(), o, n_jobs));
        }
        Ok(out)
    })
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "mnf")?;
    m.add("__doc__", "Minimum Noise Fraction transform")?;
    m.add_class::<Stats>()?;
    m.add_class::<Transform>()?;
    m.add_function(wrap_pyfunction!(stats_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_cov, &m)?)?;
    m.add_function(wrap_pyfunction!(forward, &m)?)?;
    m.add_function(wrap_pyfunction!(inverse, &m)?)?;
    m.add_function(wrap_pyfunction!(denoise, &m)?)?;
    Ok(m)
}
