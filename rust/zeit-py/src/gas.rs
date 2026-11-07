//! zeit._core.gas: a minimal binding of the gas functions of zeit-core (a port of FORCE's `gas-ll.c`: the water vapour
//! of Sentinel-2 and its transmittance map, the water vapour LUT files), for the comparison with FORCE. The public
//! function (`zeit.atmospheric_correction`) comes later.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::atc::{Atc, CoarseBrick, C_ZEN, N_ELEV};
use zeit_core::gas::{self, SceneGrid, Toa, WvLut, WvpSource, LUT_NM, LUT_NW};

fn err(e: gas::GasError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err("arrays must be contiguous"))
}

fn brick(nx: usize, ny: usize, res: f64, nodata: i32, data: &[f32], nb: usize) -> PyResult<CoarseBrick> {
    let gn = nx * ny;
    if data.len() != nb * gn {
        return Err(PyValueError::new_err("an atc array does not have bands x cells values"));
    }
    let mut b = CoarseBrick::new(nx, ny, res, nb, nodata);
    for (k, band) in b.bands.iter_mut().enumerate() {
        band.copy_from_slice(&data[k * gn..(k + 1) * gn]);
    }
    Ok(b)
}

/// The scene of FORCE's `water_vapor` and `gas_transmittance` (see `benchmarks/vs_originals/force_gas/main.inc` for
/// the layout of the arrays): returns `(lut, wvp, qai, tg, w_avg, n)`, with `w_avg` NaN if no block was retrieved.
#[pyfunction]
#[pyo3(signature = (nx, ny, res, gnx, gny, gres, nodata_view, b_nir, b_vapor, rsr_band, cosszen, cosvzen, sun_czen,
    view_czen, tso, tvo, rho_p, t, s, toa, qai, dem, n_jobs=1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn water_vapor_scene<'py>(
    py: Python<'py>,
    nx: usize,
    ny: usize,
    res: f64,
    gnx: usize,
    gny: usize,
    gres: f64,
    nodata_view: i32,
    b_nir: usize,
    b_vapor: usize,
    rsr_band: PyReadonlyArray1<'py, i32>,
    cosszen: PyReadonlyArray1<'py, f32>,
    cosvzen: PyReadonlyArray1<'py, f32>,
    sun_czen: PyReadonlyArray1<'py, f32>,
    view_czen: PyReadonlyArray1<'py, f32>,
    tso: PyReadonlyArray1<'py, f32>,
    tvo: PyReadonlyArray1<'py, f32>,
    rho_p: PyReadonlyArray1<'py, f32>,
    t: PyReadonlyArray1<'py, f32>,
    s: PyReadonlyArray1<'py, f32>,
    toa: PyReadonlyArray1<'py, i16>,
    qai: PyReadonlyArray1<'py, i16>,
    dem: PyReadonlyArray1<'py, u8>,
    n_jobs: i32,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<i16>>,
    f32,
    usize,
)> {
    let rsr: Vec<usize> = slice(&rsr_band)?.iter().map(|&b| b as usize).collect();
    let nb = rsr.len();
    if rsr.iter().any(|&b| b >= 56) {
        return Err(PyValueError::new_err("an RSR band must be in 0..56"));
    }
    let (gn, npix) = (gnx * gny, nx * ny);
    let two = |a: &PyReadonlyArray1<'_, f32>| -> PyResult<[f32; 2]> {
        let v = slice(a)?;
        <[f32; 2]>::try_from(v).map_err(|_| PyValueError::new_err("cosszen and cosvzen hold min and max"))
    };
    let mut atc = Atc::zeros(gnx, gny, gres, nb, nodata_view);
    atc.cosszen = two(&cosszen)?;
    atc.cosvzen = two(&cosvzen)?;
    atc.xy_sun.bands[C_ZEN].copy_from_slice(slice(&sun_czen)?);
    atc.xy_view.bands[C_ZEN].copy_from_slice(slice(&view_czen)?);
    atc.xy_tso = brick(gnx, gny, gres, nodata_view, slice(&tso)?, nb)?;
    atc.xy_tvo = brick(gnx, gny, gres, nodata_view, slice(&tvo)?, nb)?;
    let per_z = nb * gn;
    for (name, src, dst) in [("rho_p", &rho_p, 0), ("T", &t, 1), ("s", &s, 2)] {
        let v = slice(src)?;
        if v.len() != N_ELEV * per_z {
            return Err(PyValueError::new_err(format!("{name} needs 256 x bands x cells values")));
        }
        for z in 0..N_ELEV {
            let b = brick(gnx, gny, gres, nodata_view, &v[z * per_z..(z + 1) * per_z], nb)?;
            match dst {
                0 => atc.xyz_rho_p[z] = b,
                1 => atc.xyz_t[z] = b,
                _ => atc.xyz_s[z] = b,
            }
        }
    }
    let toa_v = slice(&toa)?;
    if toa_v.len() != nb * npix || b_nir >= nb || b_vapor >= nb {
        return Err(PyValueError::new_err("toa needs bands x pixels values"));
    }
    let mut domain = vec![String::new(); nb];
    domain[b_nir] = "NIR".into();
    domain[b_vapor] = "VAPOR".into();
    let grid = SceneGrid { nx, ny, res };
    let toa = Toa { grid, bands: toa_v.chunks(npix).map(|c| c.to_vec()).collect(), domain };
    let mut qai_v = slice(&qai)?.to_vec();
    let dem_v = slice(&dem)?;

    let out = py.detach(|| -> Result<_, gas::GasError> {
        let lut = WvLut::build(&rsr, &atc, n_jobs)?;
        let wv = gas::water_vapor(&rsr, &atc, &lut, &toa, &mut qai_v, dem_v)?;
        let mut tg = Vec::with_capacity(nb * npix);
        for b in 0..nb {
            tg.extend(gas::gas_transmittance(&atc, b, &wv.wvp, &grid, &qai_v, &lut)?);
        }
        Ok((lut, wv, tg))
    });
    let (lut, wv, tg) = out.map_err(err)?;
    debug_assert_eq!(lut.values().len(), nb * LUT_NW * LUT_NM);
    Ok((
        PyArray1::from_slice(py, lut.values()),
        PyArray1::from_vec(py, wv.wvp),
        PyArray1::from_vec(py, qai_v),
        PyArray1::from_vec(py, tg),
        wv.w_avg.unwrap_or(f32::NAN),
        wv.n,
    ))
}

/// `water_vapor_from_lut` for each scenario `(dir, lon, lat, year, month, day, wvp, strict)`; `dir` "NULL" is FORCE's
/// constant `wvp`. A failure is -1 (as FORCE returns it).
#[pyfunction]
fn water_vapor_from_lut(scenarios: Vec<(String, f64, f64, i32, i32, i32, f32, bool)>) -> Vec<f32> {
    scenarios
        .into_iter()
        .map(|(dir, lon, lat, y, m, d, wvp, strict)| {
            let src = WvpSource { dir: if dir == "NULL" { None } else { Some(dir) }, wvp, strict };
            gas::water_vapor_from_lut(&src, lon, lat, y, m, d).unwrap_or(-1.0)
        })
        .collect()
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "gas")?;
    m.add(
        "__doc__",
        "Water vapour and gas transmittance of FORCE's atmospheric correction (pieces, for the comparison with FORCE)",
    )?;
    m.add_function(wrap_pyfunction!(water_vapor_scene, &m)?)?;
    m.add_function(wrap_pyfunction!(water_vapor_from_lut, &m)?)?;
    Ok(m)
}
