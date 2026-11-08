//! zeit._core.fmask_run: a minimal binding of the end-to-end Fmask 4.7 step (`zeit_core::fmask::run`: the DN to TOA
//! math, the chain of `autoFmask.m` for Sentinel-2, the QAI flags), for the comparison with GNU Octave
//! (`benchmarks/vs_originals/fmask4_run.py`). No public API. Images are 1-D arrays of `rows * cols` in row-major order.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use zeit_core::fmask::detect::{FmaskParameters, Toa20};
use zeit_core::fmask::run::{self, DnBands, Radiometry, RunConfig, RunInput};

fn sl<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err("arrays must be contiguous"))
}

fn opt<'a, T: numpy::Element>(a: &'a Option<PyReadonlyArray1<'_, T>>) -> PyResult<Option<&'a [T]>> {
    match a {
        Some(a) => Ok(Some(sl(a)?)),
        None => Ok(None),
    }
}

/// The DN to the TOA bands of the 20 m grid; returns a dict of the arrays (`blue`, `green`, `red`, `nir`, `swir1`,
/// `swir2`, `cirrus`, `b7`, `b8` single; `satu_blue`, `satu_green`, `satu_red` uint8). `offsets` (9 values in the order
/// B02, B03, B04, B8A, B11, B12, B10, B07, B08) and `quantification` select baseline 04.00; without them, the old route.
#[pyfunction]
#[pyo3(signature = (rows, cols, b02, b03, b04, b8a, b11, b12, b10, b07, b08, compat, offsets = None, quantification = 10000.0))]
#[allow(clippy::too_many_arguments)]
fn dn_to_toa20<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    b02: PyReadonlyArray1<'py, u16>,
    b03: PyReadonlyArray1<'py, u16>,
    b04: PyReadonlyArray1<'py, u16>,
    b8a: PyReadonlyArray1<'py, u16>,
    b11: PyReadonlyArray1<'py, u16>,
    b12: PyReadonlyArray1<'py, u16>,
    b10: PyReadonlyArray1<'py, u16>,
    b07: PyReadonlyArray1<'py, u16>,
    b08: PyReadonlyArray1<'py, u16>,
    compat: bool,
    offsets: Option<PyReadonlyArray1<'py, f64>>,
    quantification: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let n = rows * cols;
    if rows % 3 != 0 || cols % 3 != 0 {
        return Err(PyValueError::new_err("rows and cols must be multiples of 3"));
    }
    let lens = [(&b02, 4 * n), (&b03, 4 * n), (&b04, 4 * n), (&b8a, n), (&b11, n), (&b12, n), (&b10, n / 9), (&b07, n), (&b08, 4 * n)];
    for (a, l) in lens {
        if a.len()? != l {
            return Err(PyValueError::new_err("a band does not have the size of its resolution"));
        }
    }
    let radio = match &offsets {
        Some(o) => {
            let o = sl(o)?;
            if o.len() != 9 {
                return Err(PyValueError::new_err("offsets must have 9 values"));
            }
            let mut a = [0f64; 9];
            a.copy_from_slice(o);
            Radiometry::Baseline04 { offsets: a, quantification }
        }
        None => Radiometry::Legacy,
    };
    let (a, b, c, d, e, f, g, h, i) = (sl(&b02)?, sl(&b03)?, sl(&b04)?, sl(&b8a)?, sl(&b11)?, sl(&b12)?, sl(&b10)?, sl(&b07)?, sl(&b08)?);
    let t = py.detach(|| run::dn_to_toa20(&DnBands { b02: a, b03: b, b04: c, b8a: d, b11: e, b12: f, b10: g, b07: h, b08: i }, rows, cols, radio, compat));
    let out = PyDict::new(py);
    out.set_item("blue", PyArray1::from_vec(py, t.blue))?;
    out.set_item("green", PyArray1::from_vec(py, t.green))?;
    out.set_item("red", PyArray1::from_vec(py, t.red))?;
    out.set_item("nir", PyArray1::from_vec(py, t.nir))?;
    out.set_item("swir1", PyArray1::from_vec(py, t.swir1))?;
    out.set_item("swir2", PyArray1::from_vec(py, t.swir2))?;
    out.set_item("cirrus", PyArray1::from_vec(py, t.cirrus))?;
    out.set_item("b7", PyArray1::from_vec(py, t.b7))?;
    out.set_item("b8", PyArray1::from_vec(py, t.b8))?;
    out.set_item("satu_blue", PyArray1::from_vec(py, t.satu_blue))?;
    out.set_item("satu_green", PyArray1::from_vec(py, t.satu_green))?;
    out.set_item("satu_red", PyArray1::from_vec(py, t.satu_red))?;
    Ok(out)
}

/// Slope and aspect (single degrees) of a north-up DEM.
#[pyfunction]
fn slope_aspect_deg<'py>(py: Python<'py>, dem: PyReadonlyArray1<'py, f32>, rows: usize, cols: usize, resolution: f64) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let d = sl(&dem)?;
    if d.len() != rows * cols {
        return Err(PyValueError::new_err("dem does not have rows * cols values"));
    }
    let (s, a) = py.detach(|| run::slope_aspect_deg(d, rows, cols, resolution, 8));
    Ok((PyArray1::from_vec(py, s), PyArray1::from_vec(py, a)))
}

/// `autoFmask.m` from the TOA bands on. `params` is `[cloud, shadow, snow buffers (pixels), thin weight, cloud
/// probability, PFPC extension radius, PFPC erosion radius, output resolution, shadow over water]` (the order of
/// `FmaskParameters.m`); returns a dict with `cs_final`, `pcloud`, `pshadow`, `psnow`, `water`, `mask`, the counts and, with
/// `keep_steps`, the arrays of the steps.
#[pyfunction]
#[pyo3(signature = (rows, cols, blue, green, red, nir, swir1, swir2, cirrus, satu_blue, satu_green, satu_red, vza, vaa, zen, azi, compat, keep_steps, b7 = None, b8 = None, dem = None, slope = None, aspect = None, gswo = None, params = None))]
#[allow(clippy::too_many_arguments)]
fn run_fmask<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    blue: PyReadonlyArray1<'py, f32>,
    green: PyReadonlyArray1<'py, f32>,
    red: PyReadonlyArray1<'py, f32>,
    nir: PyReadonlyArray1<'py, f32>,
    swir1: PyReadonlyArray1<'py, f32>,
    swir2: PyReadonlyArray1<'py, f32>,
    cirrus: PyReadonlyArray1<'py, f32>,
    satu_blue: PyReadonlyArray1<'py, u8>,
    satu_green: PyReadonlyArray1<'py, u8>,
    satu_red: PyReadonlyArray1<'py, u8>,
    vza: PyReadonlyArray1<'py, f32>,
    vaa: PyReadonlyArray1<'py, f32>,
    zen: f64,
    azi: f64,
    compat: bool,
    keep_steps: bool,
    b7: Option<PyReadonlyArray1<'py, f32>>,
    b8: Option<PyReadonlyArray1<'py, f32>>,
    dem: Option<PyReadonlyArray1<'py, f32>>,
    slope: Option<PyReadonlyArray1<'py, f32>>,
    aspect: Option<PyReadonlyArray1<'py, f32>>,
    gswo: Option<PyReadonlyArray1<'py, u8>>,
    params: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let n = rows * cols;
    let toa = Toa20 { blue: sl(&blue)?, green: sl(&green)?, red: sl(&red)?, nir: sl(&nir)?, swir1: sl(&swir1)?, swir2: sl(&swir2)?, cirrus: sl(&cirrus)? };
    for l in [toa.blue.len(), toa.green.len(), toa.red.len(), toa.nir.len(), toa.swir1.len(), toa.swir2.len(), toa.cirrus.len(), sl(&satu_blue)?.len(), sl(&satu_green)?.len(), sl(&satu_red)?.len(), sl(&vza)?.len(), sl(&vaa)?.len()] {
        if l != n {
            return Err(PyValueError::new_err("an array does not have rows * cols values"));
        }
    }
    for l in [opt(&dem)?.map(|a| a.len()), opt(&slope)?.map(|a| a.len()), opt(&aspect)?.map(|a| a.len()), opt(&gswo)?.map(|a| a.len())].into_iter().flatten() {
        if l != n {
            return Err(PyValueError::new_err("an array does not have rows * cols values"));
        }
    }
    let mut prm = FmaskParameters::default();
    if let Some(p) = &params {
        let p = sl(p)?;
        if p.len() != 9 {
            return Err(PyValueError::new_err("params must have 9 values"));
        }
        prm = FmaskParameters {
            cloud_buffer: p[0],
            cloud_shadow_buffer: p[1],
            snow_buffer: p[2],
            thin_weight: p[3],
            cloud_probability_threshold: p[4],
            pfpc_layer_extension_radius: p[5],
            pfpc_erosion_radius: p[6],
            output_resolution: p[7],
            shadow_water: p[8],
        };
    }
    let inp = RunInput {
        rows,
        cols,
        toa,
        b7: opt(&b7)?.unwrap_or(&[]),
        b8: opt(&b8)?.unwrap_or(&[]),
        satu_blue: sl(&satu_blue)?,
        satu_green: sl(&satu_green)?,
        satu_red: sl(&satu_red)?,
        dem: opt(&dem)?,
        slope: opt(&slope)?,
        aspect: opt(&aspect)?,
        gswo: opt(&gswo)?,
        sun_zenith: zen,
        sun_azimuth: azi,
        vza: sl(&vza)?,
        vaa: sl(&vaa)?,
        resolution: 20.0,
    };
    let cfg = RunConfig { params: prm, compat, keep_steps };
    let r = py.detach(|| run::run(&inp, &cfg));
    let out = PyDict::new(py);
    out.set_item("sum_clr", r.sum_clr)?;
    out.set_item("few_clear", r.few_clear)?;
    out.set_item("match_skipped", r.match_skipped)?;
    out.set_item("clr_pct", r.clr_pct)?;
    out.set_item("cs_final", PyArray1::from_vec(py, r.cs_final))?;
    out.set_item("pcloud", PyArray1::from_vec(py, r.pcloud))?;
    out.set_item("pshadow", PyArray1::from_vec(py, r.pshadow))?;
    out.set_item("psnow", PyArray1::from_vec(py, r.psnow))?;
    out.set_item("water", PyArray1::from_vec(py, r.water))?;
    out.set_item("mask", PyArray1::from_vec(py, r.mask))?;
    if let Some(s) = r.steps {
        let s = *s;
        out.set_item("psnow0", PyArray1::from_vec(py, s.psnow0))?;
        out.set_item("water_all", PyArray1::from_vec(py, s.water_all))?;
        out.set_item("idplcd", PyArray1::from_vec(py, s.idplcd))?;
        out.set_item("cloud", PyArray1::from_vec(py, s.cloud))?;
        if let Some(u) = s.idused {
            out.set_item("idused", PyArray1::from_vec(py, u))?;
        }
        out.set_item("pfpl", PyArray1::from_vec(py, s.pfpl))?;
        out.set_item("pcloud_eroded", PyArray1::from_vec(py, s.pcloud_eroded))?;
        if let Some(v) = s.pshadow_potential {
            out.set_item("pshadow_potential", PyArray1::from_vec(py, v))?;
        }
        if let Some(v) = s.match_cloud {
            out.set_item("match_cloud", PyArray1::from_vec(py, v))?;
        }
        if let Some(v) = s.match_shadow {
            out.set_item("match_shadow", PyArray1::from_vec(py, v))?;
        }
        if let Some(t) = s.topo {
            out.set_item("topo", vec![t.c_nir, t.c_swir, t.applied_nir as u8 as f64, t.applied_swir as u8 as f64, t.n_samples as f64])?;
        }
    }
    Ok(out)
}

/// `to_qai` of the pieces of a result; returns the `i16` flags of the `(rows * factor) x (cols * factor)` grid.
#[pyfunction]
#[pyo3(signature = (rows, cols, factor, resolution, mask, pcloud, water, psnow, pshadow, cloud_buffer_m = 300.0, n_jobs = 8))]
#[allow(clippy::too_many_arguments)]
fn to_qai<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    factor: usize,
    resolution: f64,
    mask: PyReadonlyArray1<'py, u8>,
    pcloud: PyReadonlyArray1<'py, u8>,
    water: PyReadonlyArray1<'py, u8>,
    psnow: PyReadonlyArray1<'py, u8>,
    pshadow: PyReadonlyArray1<'py, u8>,
    cloud_buffer_m: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    let n = rows * cols;
    let (m, c, w, s, h) = (sl(&mask)?, sl(&pcloud)?, sl(&water)?, sl(&psnow)?, sl(&pshadow)?);
    if factor < 1 || [m.len(), c.len(), w.len(), s.len(), h.len()].iter().any(|&l| l != n) {
        return Err(PyValueError::new_err("factor must be at least 1 and the arrays must have rows * cols values"));
    }
    Ok(PyArray1::from_vec(py, py.detach(|| run::to_qai(rows, cols, factor, resolution, m, c, w, s, h, cloud_buffer_m, n_jobs))))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "fmask_run")?;
    m.add("__doc__", "The Fmask 4.7 chain for Sentinel-2 and its QAI output (for the comparison with GNU Octave)")?;
    m.add_function(wrap_pyfunction!(dn_to_toa20, &m)?)?;
    m.add_function(wrap_pyfunction!(slope_aspect_deg, &m)?)?;
    m.add_function(wrap_pyfunction!(run_fmask, &m)?)?;
    m.add_function(wrap_pyfunction!(to_qai, &m)?)?;
    Ok(m)
}
