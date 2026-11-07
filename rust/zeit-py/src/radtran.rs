//! zeit._core.radtran: a minimal binding of the radiative transfer pieces of zeit-core (a port of FORCE's
//! atmospheric correction), one vectorized call by name, for the comparison with FORCE's C code. The public
//! function (`zeit.atmospheric_correction`) comes later.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::radtran::{self as rt, Date};

/// (inputs, outputs) of each piece, in the order of its C arguments; integers (band, day, time) are given as
/// float32 holding whole numbers.
fn arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "molecular_optical_depth"
        | "mod_elev_factor"
        | "env_weight_aerosol"
        | "env_weight_molecular"
        | "phase_molecular"
        | "fresnel_reflection"
        | "wavelength"
        | "e0" => (1, 1),
        "aod_elev_factor" | "optical_depth" | "f_factor" | "c_factor_emp" => (2, 1),
        "mod_elev_scale" | "aod_elev_scale" | "sphere_albedo" | "c_factor_com" | "wvp_transmitt"
        | "ozone_transmitt" | "ozone_amount" => (3, 1),
        "env_weight" | "backscatter" | "gas_transmitt" => (4, 1),
        "scatt_transmitt" => (5, 7),
        "path_ref" => (11, 1),
        "phase_aerosol" => (7, 1),
        "illumin" => (6, 1),
        "localtime2gmt" => (4, 7), // year, doy, localtime, lon -> year, month, day, doy, hh, mm, ss
        "standard_sunzenith" => (5, 1), // year, doy, tz, lat, lon
        "sunpos" => (8, 2),        // lat, lon, year, doy, hh, mm, ss, tz -> zen, azi
        _ => return None,
    })
}

fn date(year: f32, doy: f32) -> Date {
    Date::from_year_doy(year as i32, doy as i32)
}

fn eval_one(name: &str, x: &[f32], y: &mut [f32]) {
    let i = |k: usize| x[k] as i32;
    let band = |k: usize| x[k] as usize;
    match name {
        "molecular_optical_depth" => y[0] = rt::molecular_optical_depth(x[0]),
        "aod_elev_factor" => y[0] = rt::aod_elev_factor(x[0], x[1]),
        "mod_elev_factor" => y[0] = rt::mod_elev_factor(x[0]),
        "mod_elev_scale" => y[0] = rt::mod_elev_scale(x[0], x[1], x[2]),
        "aod_elev_scale" => y[0] = rt::aod_elev_scale(x[0], x[1], x[2]),
        "optical_depth" => y[0] = rt::optical_depth(x[0], x[1]),
        "scatt_transmitt" => {
            let t = rt::scatt_transmitt(x[0], x[1], x[2], x[3], x[4]);
            y.copy_from_slice(&[t.total, t.ts, t.tv, t.tsd, t.tss, t.tvd, t.tvs]);
        }
        "path_ref" => y[0] = rt::path_ref(x[0] != 0.0, x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]),
        "sphere_albedo" => y[0] = rt::sphere_albedo(x[0], x[1], x[2]),
        "env_weight_aerosol" => y[0] = rt::env_weight_aerosol(x[0]),
        "env_weight_molecular" => y[0] = rt::env_weight_molecular(x[0]),
        "env_weight" => y[0] = rt::env_weight(x[0], x[1], x[2], x[3]),
        "backscatter" => y[0] = rt::backscatter(x[0], x[1], x[2], x[3]),
        "phase_molecular" => y[0] = rt::phase_molecular(x[0]),
        "phase_aerosol" => y[0] = rt::phase_aerosol(x[0], &[x[1], x[2], x[3], x[4], x[5], x[6]]),
        "wvp_transmitt" => y[0] = rt::wvp_transmitt(x[0], x[1], band(2)),
        "ozone_transmitt" => y[0] = rt::ozone_transmitt(x[0], x[1], band(2)),
        "gas_transmitt" => y[0] = rt::gas_transmitt(x[0], x[1], x[2], x[3]),
        "fresnel_reflection" => y[0] = rt::fresnel_reflection(x[0]),
        "illumin" => y[0] = rt::illumin(x[0], x[1], x[2], x[3], x[4], x[5]),
        "c_factor_emp" => y[0] = rt::c_factor_emp(x[0] as f64, x[1] as f64),
        "c_factor_com" => y[0] = rt::c_factor_com(x[0], x[1], x[2]),
        "f_factor" => y[0] = rt::f_factor(x[0], x[1]),
        "wavelength" => y[0] = rt::wavelength(band(0)),
        "e0" => y[0] = rt::e0(band(0)),
        "ozone_amount" => y[0] = rt::ozone_amount(x[0], x[1], i(2)),
        "localtime2gmt" => {
            let g = rt::localtime2gmt(&date(x[0], x[1]), x[2] as f64, x[3] as f64);
            for (o, v) in y.iter_mut().zip([g.year, g.month, g.day, g.doy, g.hh, g.mm, g.ss]) {
                *o = v as f32;
            }
        }
        "standard_sunzenith" => {
            let mut d = date(x[0], x[1]);
            d.tz = i(2);
            y[0] = rt::standard_sunzenith(&d, x[3] as f64, x[4] as f64) as f32;
        }
        "sunpos" => {
            let mut d = date(x[2], x[3]);
            (d.hh, d.mm, d.ss, d.tz) = (i(4), i(5), i(6), i(7));
            let (zen, azi) = rt::sunpos(x[0], x[1], &d);
            y[0] = zen;
            y[1] = azi;
        }
        _ => unreachable!(),
    }
}

/// Evaluate the piece `name` of FORCE's radiative transfer on equal-length float32 vectors, one per C argument (the
/// integers as whole floats, `multi` as 0 or 1); returns the list of float32 outputs, one vector per output.
#[pyfunction]
#[pyo3(signature = (name, *arrays))]
fn radtran_eval<'py>(
    py: Python<'py>,
    name: &str,
    arrays: Vec<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
    let (nin, nout) = arity(name).ok_or_else(|| PyValueError::new_err(format!("unknown radtran piece {name:?}")))?;
    if arrays.len() != nin {
        return Err(PyValueError::new_err(format!(
            "{name} takes {nin} arrays, got {}",
            arrays.len()
        )));
    }
    let cols = arrays
        .iter()
        .map(|a| {
            a.as_slice()
                .map_err(|_| PyValueError::new_err("arrays must be contiguous"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let n = cols[0].len();
    if cols.iter().any(|c| c.len() != n) {
        return Err(PyValueError::new_err("the arrays must have the same length"));
    }
    if matches!(name, "wvp_transmitt" | "ozone_transmitt" | "wavelength" | "e0") {
        let col = if name == "wavelength" || name == "e0" { 0 } else { 2 };
        if cols[col].iter().any(|&b| !(0.0..56.0).contains(&b)) {
            return Err(PyValueError::new_err("the band index must be in 0..56"));
        }
    }
    let out = py.detach(|| {
        let mut out = vec![vec![0f32; n]; nout];
        let mut x = vec![0f32; nin];
        let mut y = vec![0f32; nout];
        for p in 0..n {
            for (k, c) in cols.iter().enumerate() {
                x[k] = c[p];
            }
            eval_one(name, &x, &mut y);
            for (k, o) in out.iter_mut().enumerate() {
                o[p] = y[k];
            }
        }
        out
    });
    Ok(out.into_iter().map(|v| PyArray1::from_vec(py, v)).collect())
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "radtran")?;
    m.add(
        "__doc__",
        "Radiative transfer of FORCE's atmospheric correction (pieces, for the comparison with FORCE)",
    )?;
    m.add_function(wrap_pyfunction!(radtran_eval, &m)?)?;
    Ok(m)
}
