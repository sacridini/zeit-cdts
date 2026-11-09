//! zeit._core.l2scene: a minimal binding of the atmospheric correction of one Sentinel-2 L1C scene of zeit-core
//! (`zeit_core::l2scene`: `force-l2ps` from the digital numbers to the Level 2 products), for `zeit/_level2.py` and the
//! comparison with FORCE (`benchmarks/vs_originals/force_l2ps_real.py`). The public function comes later.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use zeit_core::atc::{AZI, ZEN};
use zeit_core::atmo::L2Params;
use zeit_core::gas::Toa;
use zeit_core::l2::L2Error;
use zeit_core::l2scene::{self, SceneParams, SunTable};
use zeit_core::s2l1c::{self, NB};
use zeit_core::sunview::GeoTable;

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{what} must be contiguous")))
}

/// What the scene's metadata says: the grid of the scene, the corners the correction needs the geography of, the date and
/// the window to read from a band of a given resolution. `subset` is false: the image is the whole tile.
#[pyfunction]
fn describe<'py>(
    py: Python<'py>,
    product_xml: &Bound<'py, PyBytes>,
    tile_xml: &Bound<'py, PyBytes>,
    granule_name: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let l1 = s2l1c::parse_metadata(product_xml.as_bytes(), tile_xml.as_bytes(), granule_name, false).map_err(err)?;
    let g = l2scene::geo_grid(&l1).map_err(err)?;
    let (t, p) = (&l1.tile, &l1.product);
    let d = PyDict::new(py);
    d.set_item("res", t.res)?;
    d.set_item("ncols", t.ncols)?;
    d.set_item("nrows", t.nrows)?;
    d.set_item("ulx", t.ulx)?;
    d.set_item("uly", t.uly)?;
    d.set_item("left", t.left)?;
    d.set_item("top", t.top)?;
    d.set_item("epsg", t.epsg)?;
    d.set_item("sensor", &p.sensor)?;
    d.set_item("baseline", &p.baseline)?;
    d.set_item("dn_scale", p.dn_scale)?;
    d.set_item("radd", p.radd.to_vec())?;
    let dt = &p.date;
    d.set_item("date", (dt.year, dt.month, dt.day, dt.hh, dt.mm, dt.ss))?;
    d.set_item("refsys_id", &l1.refsys_id)?;
    d.set_item("geo_grid", (g.ulx, g.uly, g.res, g.nx, g.ny))?;
    d.set_item("domains", s2l1c::DOMAINS.to_vec())?;
    d.set_item("wavelength", l1.dn().wavelength)?;
    // the read windows (xoff, yoff, nx, ny) for the three resolutions of Sentinel-2
    let w: Vec<(f64, (i32, i32, i32, i32))> = [10.0, 20.0, 60.0].iter().map(|&r| (r, l1.read_window(r))).collect();
    d.set_item("windows", w)?;
    d.set_item("rsr_band", l1.meta.cal.iter().map(|c| c.rsr_band).collect::<Vec<_>>())?;
    d.set_item("s2", (t.s2.nx, t.s2.ny, t.s2.nodata))?;
    d.set_item("vzen", PyArray1::from_vec(py, t.s2.vzen.clone()))?;
    d.set_item("vazi", PyArray1::from_vec(py, t.s2.vazi.clone()))?;
    Ok(d)
}

/// `bounds_level1` and `convert_level1` on the digital numbers (`13 * ncols * nrows` values, band by band): the initial QAI
/// and the TOA reflectance (`13 * ncols * nrows`), as the correction starts from them.
#[pyfunction]
fn prepare<'py>(
    py: Python<'py>,
    product_xml: &Bound<'py, PyBytes>,
    tile_xml: &Bound<'py, PyBytes>,
    granule_name: &str,
    dn: PyReadonlyArray1<'py, u16>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let l1 = s2l1c::parse_metadata(product_xml.as_bytes(), tile_xml.as_bytes(), granule_name, false).map_err(err)?;
    let nc = l1.tile.ncols as usize * l1.tile.nrows as usize;
    let dn = slice(&dn, "dn")?;
    if dn.len() != NB * nc {
        return Err(PyValueError::new_err(format!("dn has {} values, not 13 x {nc}", dn.len())));
    }
    let bands: Vec<&[u16]> = dn.chunks(nc.max(1)).collect();
    let (qai, toa) = py.detach(|| l1.bounds_and_convert(&bands, false, n_jobs)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("qai", PyArray1::from_vec(py, qai))?;
    d.set_item("toa", PyArray1::from_vec(py, toa.concat()))?;
    Ok(d)
}

/// `force-l2ps` for the scene: the digital numbers (`13 * ncols * nrows` values band by band), the flags (the cloud, shadow,
/// snow and water bits per pixel, or empty), the DEM in metres on the grid of the scene (or empty: flat), the longitude and
/// latitude of the `geo_grid` corners (row by row) and, optionally, the sun angles of the coarse cells in degrees.
/// `toa_hook(toa, qai) -> (toa, qai)` is called once between the cloud flags and the correction (FORCE's coregistration and
/// resolution merge) with the TOA (`13 * cells` int16, band by band, -9999 where off) and the QAI (`cells` int16). `params`
/// holds the switches of `L2Params`, `dem_nodata`, `buffer_nodata` and `n_jobs`.
#[pyfunction]
#[pyo3(signature = (product_xml, tile_xml, granule_name, dn, flags, dem, lon, lat, params, sun_zen=None, sun_azi=None, brdf=None, toa_hook=None))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    product_xml: &Bound<'py, PyBytes>,
    tile_xml: &Bound<'py, PyBytes>,
    granule_name: &str,
    dn: PyReadonlyArray1<'py, u16>,
    flags: PyReadonlyArray1<'py, i16>,
    dem: PyReadonlyArray1<'py, f32>,
    lon: PyReadonlyArray1<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    params: &Bound<'py, PyDict>,
    sun_zen: Option<PyReadonlyArray1<'py, f32>>,
    sun_azi: Option<PyReadonlyArray1<'py, f32>>,
    brdf: Option<PyReadonlyArray1<'py, f32>>,
    toa_hook: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let l1 = s2l1c::parse_metadata(product_xml.as_bytes(), tile_xml.as_bytes(), granule_name, false).map_err(err)?;
    let nc = l1.tile.ncols as usize * l1.tile.nrows as usize;
    let (dn, flags, dem) = (slice(&dn, "dn")?, slice(&flags, "flags")?, slice(&dem, "dem")?);
    if dn.len() != NB * nc {
        return Err(PyValueError::new_err(format!("dn has {} values, not 13 x {nc}", dn.len())));
    }
    let g = l2scene::geo_grid(&l1).map_err(err)?;
    let (lon, lat) = (slice(&lon, "lon")?, slice(&lat, "lat")?);
    if lon.len() != g.nx * g.ny || lat.len() != g.nx * g.ny {
        return Err(PyValueError::new_err(format!("the geography needs {} x {} corners", g.nx, g.ny)));
    }
    let table = GeoTable { ulx: g.ulx, uly: g.uly, res: g.res, nx: g.nx, ny: g.ny, lon: lon.to_vec(), lat: lat.to_vec() };
    let sun = match (&sun_zen, &sun_azi) {
        (Some(z), Some(a)) => Some(SunTable { zen: slice(z, "sun_zen")?.to_vec(), azi: slice(a, "sun_azi")?.to_vec() }),
        (None, None) => None,
        _ => return Err(PyValueError::new_err("sun_zen and sun_azi go together")),
    };
    // FORCE's BRDF factors, `13 * cells` values band by band
    let brdf_t: Option<Vec<Vec<f32>>> = match &brdf {
        Some(t) => {
            let t = slice(t, "brdf")?;
            let cells = t.len() / NB;
            if cells == 0 || t.len() != NB * cells {
                return Err(PyValueError::new_err(format!("brdf needs {NB} bands of one value per coarse cell")));
            }
            Some(t.chunks(cells).map(|c| c.to_vec()).collect())
        }
        None => None,
    };
    let b = |k: &str| -> PyResult<bool> { Ok(params.get_item(k)?.map(|v| v.extract::<bool>()).transpose()?.unwrap_or(false)) };
    let sp = SceneParams {
        l2: L2Params {
            domulti: b("domulti")?,
            doenv: b("doenv")?,
            doatmo: b("doatmo")?,
            dotopo: b("dotopo")?,
            dobrdf: b("dobrdf")?,
            erase_cloud: b("erase_cloud")?,
            doaod: b("doaod")?,
            odst: b("odst")?,
            ovzn: b("ovzn")?,
            ohot: b("ohot")?,
            oaod: b("oaod")?,
            owvp: b("owvp")?,
        },
        dem_nodata: params.get_item("dem_nodata")?.map(|v| v.extract()).transpose()?.unwrap_or(-32767),
        buffer_nodata: b("buffer_nodata")?,
        n_jobs: params.get_item("n_jobs")?.map(|v| v.extract()).transpose()?.unwrap_or(1),
    };
    let bands: Vec<&[u16]> = dn.chunks(nc.max(1)).collect();
    let t0 = std::time::Instant::now();
    // the hook: the callable gets the TOA (`13 * cells` int16, band by band) and the QAI (`cells` int16) as numpy arrays and
    // returns both; an exception it raises is kept and raised once the scene returns
    let hook_obj: Option<Py<PyAny>> = toa_hook.filter(|h| !h.is_none()).map(|h| h.unbind());
    let hook_err: std::sync::Mutex<Option<PyErr>> = std::sync::Mutex::new(None);
    let result = py.detach(|| {
        let mut hook = |toa: &mut Toa, q: &mut [i16]| -> Result<(), L2Error> {
            let h = hook_obj.as_ref().unwrap();
            let fail = |e: PyErr| {
                *hook_err.lock().unwrap() = Some(e);
                L2Error::Other("toa_hook raised an exception".into())
            };
            Python::attach(|py| -> PyResult<()> {
                let a = PyArray1::from_vec(py, toa.bands.concat());
                let b = PyArray1::from_vec(py, q.to_vec());
                let r = h.call1(py, (a, b))?;
                let (ra, rb): (PyReadonlyArray1<i16>, PyReadonlyArray1<i16>) = r.extract(py)?;
                let (ra, rb) = (slice(&ra, "toa_hook's toa")?, slice(&rb, "toa_hook's qai")?);
                if ra.len() != NB * nc || rb.len() != nc {
                    return Err(PyValueError::new_err(format!(
                        "toa_hook must return the TOA ({} values) and the QAI ({nc} values), got {} and {}",
                        NB * nc,
                        ra.len(),
                        rb.len()
                    )));
                }
                for (band, src) in toa.bands.iter_mut().zip(ra.chunks(nc.max(1))) {
                    band.copy_from_slice(src);
                }
                q.copy_from_slice(rb);
                Ok(())
            })
            .map_err(fail)
        };
        l2scene::level2_scene(
            &l1,
            &bands,
            if flags.is_empty() { None } else { Some(flags) },
            if dem.is_empty() { None } else { Some(dem) },
            &table,
            sun.as_ref(),
            brdf_t.as_deref(),
            &sp,
            if hook_obj.is_some() { Some(&mut hook) } else { None },
        )
    });
    if let Some(e) = hook_err.into_inner().unwrap() {
        return Err(e);
    }
    let out = result.map_err(err)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let (l2, atc, top, qai_in) = (out.level2, out.atc, out.top, out.qai_in);

    let d = PyDict::new(py);
    d.set_item("qai", PyArray1::from_vec(py, l2.qai))?;
    d.set_item("boa", PyArray1::from_vec(py, l2.boa.bands.concat()))?;
    d.set_item("boa_domain", l2.boa.domain)?;
    d.set_item("boa_wavelength", l2.boa.wavelength)?;
    for (k, v) in [("dst", l2.dst), ("vzn", l2.vzn), ("hot", l2.hot), ("aod", l2.aod), ("wvp", l2.wvp)] {
        if let Some(v) = v {
            d.set_item(k, PyArray1::from_vec(py, v))?;
        }
    }
    // the scene's values (what FORCE's log gives) and the coarse geometry
    d.set_item("aodmap", atc.aodmap)?;
    d.set_item("n_targets", l2.n_targets.to_vec())?;
    d.set_item("qai_in", PyArray1::from_vec(py, qai_in))?;
    d.set_item("top_dem", PyArray1::from_vec(py, top.dem))?;
    d.set_item("top_slope", PyArray1::from_vec(py, top.slope))?;
    d.set_item("top_aspect", PyArray1::from_vec(py, top.aspect))?;
    d.set_item("top_ill", PyArray1::from_vec(py, top.ill))?;
    d.set_item("top_sky", PyArray1::from_vec(py, top.sky))?;
    d.set_item("dem_stats", (atc.dem.avg, atc.dem.min, atc.dem.max, atc.dem.step))?;
    d.set_item("view_zen", PyArray1::from_vec(py, atc.xy_view.bands[ZEN].clone()))?;
    d.set_item("view_azi", PyArray1::from_vec(py, atc.xy_view.bands[AZI].clone()))?;
    d.set_item("xy_aod", PyArray1::from_vec(py, atc.xy_aod.bands.concat()))?;
    d.set_item("xy_brdf", PyArray1::from_vec(py, atc.xy_brdf.bands.concat()))?;
    d.set_item("xy_interp", PyArray1::from_vec(py, atc.xy_interp.bands[0].clone()))?;
    d.set_item("xy_dem", PyArray1::from_vec(py, atc.xy_dem.bands[0].clone()))?;
    d.set_item("coarse_shape", (atc.xy_sun.ny, atc.xy_sun.nx))?;
    d.set_item("aod_scene", atc.aod.clone())?;
    d.set_item("sun_zen", PyArray1::from_vec(py, atc.xy_sun.bands[ZEN].iter().map(|&r| (r as f64).to_degrees() as f32).collect()))?;
    d.set_item("sun_azi", PyArray1::from_vec(py, atc.xy_sun.bands[AZI].iter().map(|&r| (r as f64).to_degrees() as f32).collect()))?;
    d.set_item("ms", ms)?;
    Ok(d)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "l2scene")?;
    m.add("__doc__", "The atmospheric correction of one Sentinel-2 L1C scene (FORCE's force-l2ps), from the digital numbers")?;
    m.add_function(wrap_pyfunction!(describe, &m)?)?;
    m.add_function(wrap_pyfunction!(prepare, &m)?)?;
    m.add_function(wrap_pyfunction!(run, &m)?)?;
    Ok(m)
}
