//! zeit._core.s2l1c: a minimal binding of the Sentinel-2 Level 1C functions of zeit-core (a port of FORCE's
//! `parse_metadata_sentinel2` of `meta-ll.c` and of `bounds_level1` and `convert_level1` of `read-ll.c`), for the
//! comparison with FORCE (`benchmarks/vs_originals/force_l1.py`). The public function comes later.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zeit_core::s2l1c::{self, DOMAINS, NB};

fn err(e: s2l1c::S2L1cError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A byte string of little endian values.
#[derive(Default)]
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
    fn s(&mut self, v: &str, n: usize) {
        let mut buf = vec![0u8; n];
        let k = v.len().min(n - 1);
        buf[..k].copy_from_slice(&v.as_bytes()[..k]);
        self.0.extend_from_slice(&buf);
    }
    fn fs(&mut self, v: &[f32]) {
        v.iter().for_each(|&x| self.f(x));
    }
    fn i16s(&mut self, v: &[i16]) {
        v.iter().for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
    }
}

/// `parse_metadata` for a Sentinel-2 granule (the product file and the granule file as bytes, the name of the granule
/// directory, `subset` = `doreproj || dotile`) and, if `dn` is given (`13 * nx * ny` values, band by band), `bounds_level1`
/// and `convert_level1` on it. Returns the dump in the layout of the C harness' output
/// (`benchmarks/vs_originals/force_l1/main.c`; the status is 0, a metadata error is raised).
#[pyfunction]
#[pyo3(signature = (product_xml, tile_xml, granule_name, subset, buffer_nodata, dn=None, nx=0, ny=0, n_jobs=1))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    product_xml: &Bound<'py, PyBytes>,
    tile_xml: &Bound<'py, PyBytes>,
    granule_name: &str,
    subset: bool,
    buffer_nodata: bool,
    dn: Option<PyReadonlyArray1<'py, u16>>,
    nx: usize,
    ny: usize,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let (p, t) = (product_xml.as_bytes(), tile_xml.as_bytes());
    let dn = dn
        .as_ref()
        .map(|a| a.as_slice().map_err(|_| PyValueError::new_err("dn must be contiguous")))
        .transpose()?;
    let bytes = py
        .detach(|| -> Result<Vec<u8>, s2l1c::S2L1cError> {
            let l1 = s2l1c::parse_metadata(p, t, granule_name, subset)?;
            let mut o = Out::default();
            o.i(0);
            o.i(NB as i32);
            for c in &l1.meta.cal {
                o.i(c.rsr_band as i32);
                o.f(c.radd);
                o.f(l1.product.dn_scale);
            }
            o.s(&l1.product.sensor, 8);
            o.i(s2l1c::DTYPE);
            o.i(s2l1c::SATURATION);
            o.i(1);
            o.s(&l1.refsys_id, 8);
            let d = &l1.product.date;
            for v in [d.year, d.month, d.day, d.doy, d.hh, d.mm, d.ss, d.tz] {
                o.i(v);
            }
            let t = &l1.tile;
            o.d(t.res);
            o.d(t.ulx);
            o.d(t.uly);
            o.i(t.ncols);
            o.i(t.nrows);
            o.i(t.s2.nx as i32);
            o.i(t.s2.ny as i32);
            for v in [t.left, t.right, t.top, t.bottom] {
                o.i(v);
            }
            o.f(t.s2.nodata);
            o.fs(&t.s2.vzen);
            o.fs(&t.s2.vazi);
            o.fs(&t.szen);
            o.fs(&t.sazi);
            let dnm = l1.dn();
            for b in 0..NB {
                o.f(dnm.wavelength[b]);
                o.s(DOMAINS[b], 16);
            }
            if let Some(dn) = dn {
                let nc = nx * ny;
                if dn.len() != NB * nc {
                    return Err(s2l1c::S2L1cError::Shape(format!(
                        "dn has {} values, not 13 x {nx} x {ny}",
                        dn.len()
                    )));
                }
                let bands: Vec<&[u16]> = dn.chunks(nc.max(1)).collect();
                let qai = s2l1c::bounds_level1(&bands, nx, ny, buffer_nodata)?;
                let radd: Vec<f32> = l1.meta.cal.iter().map(|c| c.radd).collect();
                let toa = s2l1c::convert_level1(&bands, &qai, &radd, l1.product.dn_scale, n_jobs)?;
                o.i(nx as i32);
                o.i(ny as i32);
                o.i16s(&qai);
                for b in &toa {
                    o.i16s(b);
                }
            }
            Ok(o.0)
        })
        .map_err(err)?;
    Ok(PyBytes::new(py, &bytes))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "s2l1c")?;
    m.add(
        "__doc__",
        "Sentinel-2 Level 1C for FORCE's atmospheric correction (metadata, initial QAI, TOA; for the comparison with FORCE)",
    )?;
    m.add_function(wrap_pyfunction!(run, &m)?)?;
    Ok(m)
}
