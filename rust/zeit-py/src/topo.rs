//! zeit._core.topo: a minimal binding of the topography functions of zeit-core (a port of FORCE's `topo-ll.c`:
//! `compile_topography`, `cfactor_topography`, `average_elevation_cell`), for the comparison with FORCE. The public
//! function (`zeit.atmospheric_correction`) comes later.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use zeit_core::atc::{Atc, AZI, C_ZEN, S_ZEN, ZEN};
use zeit_core::gas::{SceneGrid, Toa};
use zeit_core::topo::{self, TopoError};

fn get<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("the scene has no '{key}'")))
}

fn vec_of<'py, T: numpy::Element + Clone>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Vec<T>> {
    let a: PyReadonlyArray1<'py, T> = get(d, key)?.extract()?;
    Ok(a.as_slice().map_err(|_| PyValueError::new_err("arrays must be contiguous"))?.to_vec())
}

struct Out(Vec<u8>);

impl Out {
    fn i(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn d(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i16s(&mut self, v: &[i16]) {
        v.iter().for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
    }
    fn u16s(&mut self, v: &[u16]) {
        v.iter().for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
    }
}

fn err(e: TopoError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A scene dict (the layout of `benchmarks/vs_originals/force_topo/main.inc`, here keys of a dict: `nx`, `ny`, `res`,
/// `gnx`, `gny`, `gres`, `dem_nodata`, `flat`, `do_c`, `sun_zen`, `sun_azi`, `sun_czen`, `sun_szen`, `dem`, `qai`; with
/// `do_c`: `domain` (list of str), `toa` (bands x pixels), `rho_p`, `tss`, `tsd` (256 x coarse cells, the band of
/// SWIR2)): runs `compile_topography`, `cfactor_topography` and `average_elevation_cell` of every coarse cell in
/// that order and returns the bytes of the layout the C harness writes, with the times (ms).
#[pyfunction]
fn scene<'py>(py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<(Bound<'py, PyBytes>, f64, f64)> {
    let (nx, ny): (usize, usize) = (get(d, "nx")?.extract()?, get(d, "ny")?.extract()?);
    let res: f64 = get(d, "res")?.extract()?;
    let (gnx, gny): (usize, usize) = (get(d, "gnx")?.extract()?, get(d, "gny")?.extract()?);
    let gres: f64 = get(d, "gres")?.extract()?;
    let dem_nodata: i32 = get(d, "dem_nodata")?.extract()?;
    let flat: bool = get(d, "flat")?.extract()?;
    let do_c: bool = get(d, "do_c")?.extract()?;
    let (gn, npix) = (gnx * gny, nx * ny);
    let mut atc = Atc::zeros(gnx, gny, gres, 1, -9999);
    for (key, band) in [("sun_zen", ZEN), ("sun_azi", AZI), ("sun_czen", C_ZEN), ("sun_szen", S_ZEN)] {
        let v: Vec<f32> = vec_of(d, key)?;
        if v.len() != gn {
            return Err(PyValueError::new_err(format!("{key} needs one value per coarse cell")));
        }
        atc.xy_sun.bands[band] = v;
    }
    let dem: Vec<f32> = vec_of(d, "dem")?;
    let mut qai: Vec<i16> = vec_of(d, "qai")?;
    if dem.len() != npix || qai.len() != npix {
        return Err(PyValueError::new_err("dem and qai need nx * ny values"));
    }
    let toa = if do_c {
        let domain: Vec<String> = get(d, "domain")?.extract()?;
        let nb = domain.len();
        let toa_v: Vec<i16> = vec_of(d, "toa")?;
        if toa_v.len() != nb * npix {
            return Err(PyValueError::new_err("toa needs bands x pixels values"));
        }
        let sw2 = domain.iter().position(|s| s == "SWIR2");
        atc = Atc::zeros(gnx, gny, gres, nb, -9999);
        for (key, band) in [("sun_zen", ZEN), ("sun_azi", AZI), ("sun_czen", C_ZEN), ("sun_szen", S_ZEN)] {
            atc.xy_sun.bands[band] = vec_of(d, key)?;
        }
        if let Some(sw2) = sw2 {
            for (key, which) in [("rho_p", 0), ("tss", 1), ("tsd", 2)] {
                let v: Vec<f32> = vec_of(d, key)?;
                if v.len() != 256 * gn {
                    return Err(PyValueError::new_err(format!("{key} needs 256 x coarse cells values")));
                }
                let bricks = match which {
                    0 => &mut atc.xyz_rho_p,
                    1 => &mut atc.xyz_tss,
                    _ => &mut atc.xyz_tsd,
                };
                for (z, brick) in bricks.iter_mut().enumerate() {
                    brick.bands[sw2].copy_from_slice(&v[z * gn..(z + 1) * gn]);
                }
            }
        }
        Some(Toa { grid: SceneGrid { nx, ny, res }, bands: toa_v.chunks(npix).map(|c| c.to_vec()).collect(), domain })
    } else {
        None
    };

    let run = || -> Result<(Vec<u8>, f64, f64), TopoError> {
        let mut o = Out(Vec::new());
        let t0 = std::time::Instant::now();
        let top = topo::compile_topography(if flat { None } else { Some(&dem) }, dem_nodata, nx, ny, res, &mut atc, &mut qai);
        let t_compile = t0.elapsed().as_secs_f64() * 1e3;
        let top = match top {
            Ok(t) => Some(t),
            Err(TopoError::DemBounds { .. }) => None, // FORCE: FAILURE, atc.dem and the QAI are as they are
            Err(e) => return Err(e),
        };
        o.i(if top.is_some() { 0 } else { 1 });
        o.f(atc.dem.avg);
        o.f(atc.dem.min);
        o.f(atc.dem.max);
        o.f(atc.dem.step);
        o.i(atc.dem.cnum);
        o.i16s(&qai);
        let mut t_c = 0.0;
        if let Some(top) = top {
            o.0.extend_from_slice(&top.dem);
            o.u16s(&top.slope);
            o.u16s(&top.aspect);
            o.i16s(&top.ill);
            o.u16s(&top.sky);
            if let Some(toa) = &toa {
                let t1 = std::time::Instant::now();
                let c = topo::cfactor_topography(&atc, toa, &qai, &top.dem, &top.slope, &top.ill);
                t_c = t1.elapsed().as_secs_f64() * 1e3;
                o.i(c.is_err() as i32);
                o.u16s(&c.unwrap_or_else(|_| vec![0; npix]));
            }
            for g in 0..gn {
                topo::average_elevation_cell(g, &mut atc.xy_dem, &top.dem, nx, ny, res, &qai);
            }
            for g in 0..gn {
                o.i(atc.xy_dem.bands[0][g] as i32);
            }
        }
        o.d(t_compile);
        o.d(t_c);
        Ok((o.0, t_compile, t_c))
    };
    let (bytes, t_compile, t_c) = py.detach(run).map_err(err)?;
    Ok((PyBytes::new(py, &bytes), t_compile, t_c))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "topo")?;
    m.add("__doc__", "The topography of FORCE's atmospheric correction (pieces, for the comparison with FORCE)")?;
    m.add_function(wrap_pyfunction!(scene, &m)?)?;
    Ok(m)
}
