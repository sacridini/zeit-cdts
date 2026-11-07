//! zeit._core.aod: a minimal binding of the dark target and AOD functions of zeit-core (a port of the first half of
//! FORCE's `aod-ll.c`: `aodfileread`, `extract_dark_target`, the spectral libraries, `aod_lib_to_target`,
//! `aod_from_target`, `aod_elev_dependency`), for the comparison with FORCE. The public function
//! (`zeit.atmospheric_correction`) comes later.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use zeit_core::aod::{self, DarkObj, DarkTargets, SpecLib, TargetType, Top};
use zeit_core::atc::{Atc, AZI, C_ZEN};
use zeit_core::gas::{SceneGrid, Toa};

fn err(e: aod::AodError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

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
    fn fs(&mut self, v: &[f32]) {
        v.iter().for_each(|&x| self.f(x));
    }
}

fn dump_objects(o: &mut Out, d: &[DarkObj]) {
    o.i(d.len() as i32);
    for t in d {
        for v in [t.k, t.valid as i32, t.imin, t.imax, t.jmin, t.jmax, t.i, t.j, t.e, t.f, t.g, t.r] {
            o.i(v);
        }
        o.fs(&[t.ms, t.mv, t.razi, t.cosi, t.sky, t.z, t.hr]);
        o.fs(&t.ttoa);
        o.fs(&t.etoa);
    }
}

fn dump_libmatch(
    o: &mut Out,
    atc: &Atc,
    res: f64,
    multi: bool,
    d: &[DarkObj],
    lib: &SpecLib,
    kind: TargetType,
) -> Result<(), aod::AodError> {
    let nb = atc.xy_aod.nb();
    o.i(d.iter().filter(|t| t.valid).count() as i32);
    for t in d.iter().filter(|t| t.valid) {
        let mut aodest = vec![vec![0.0f32; nb]; lib.n];
        aod::aod_lib_to_target(atc, res, multi, t, lib, kind, &mut aodest)?;
        for row in &aodest {
            o.fs(row);
        }
    }
    Ok(())
}

fn dump_fit(o: &mut Out, d: &[DarkObj], k: i32) {
    o.i(k);
    for t in d {
        o.i(t.valid as i32);
        o.i(t.lib_id);
        o.i(t.ang_fit);
        o.f(t.rsq);
        o.fs(&t.coef);
        o.fs(&t.aod);
        o.fs(&t.est);
    }
}

/// The scene of FORCE's dark target functions (see `benchmarks/vs_originals/force_aod/main.inc` for the layout of the
/// arrays, here the keys of a dict): runs `extract_dark_target` (water, shadow, vegetation), `aod_lib_to_target` on
/// the targets, `aod_from_target` and `aod_elev_dependency` in the order of `compile_aod`, and returns the dumps of
/// the stages as bytes in the layout the C harness writes, with the times (ms) of the extraction and the estimation.
#[pyfunction]
fn scene<'py>(py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<(Bound<'py, PyBytes>, f64, f64)> {
    let nb: usize = get(d, "nb")?.extract()?;
    let (nx, ny): (usize, usize) = (get(d, "nx")?.extract()?, get(d, "ny")?.extract()?);
    let res: f64 = get(d, "res")?.extract()?;
    let (gnx, gny): (usize, usize) = (get(d, "gnx")?.extract()?, get(d, "gny")?.extract()?);
    let gres: f64 = get(d, "gres")?.extract()?;
    let nodata: i32 = get(d, "nodata_view")?.extract()?;
    let multi: bool = get(d, "multi")?.extract()?;
    let domain: Vec<String> = get(d, "domain")?.extract()?;
    let rsr: Vec<usize> = vec_of::<i32>(d, "rsr_band")?.into_iter().map(|b| b as usize).collect();
    let (gn, npix) = (gnx * gny, nx * ny);
    if domain.len() != nb || rsr.len() != nb || rsr.iter().any(|&b| b >= 56) {
        return Err(PyValueError::new_err("domain and rsr_band need nb entries, rsr_band in 0..56"));
    }
    let mut atc = Atc::zeros(gnx, gny, gres, nb, nodata);
    atc.aod_bands = vec_of::<u8>(d, "aod_bands")?.into_iter().map(|v| v != 0).collect();
    atc.wvl = vec_of(d, "wvl")?;
    atc.lwvl = vec_of(d, "lwvl")?;
    atc.e0 = vec_of(d, "e0")?;
    let dem: Vec<f32> = vec_of(d, "dem")?;
    if dem.len() != 4 || atc.aod_bands.len() != nb || atc.wvl.len() != nb || atc.lwvl.len() != nb || atc.e0.len() != nb {
        return Err(PyValueError::new_err("per-band arrays need nb values, dem 4 (avg, min, max, step)"));
    }
    atc.dem = zeit_core::atc::DemStats { avg: dem[0], min: dem[1], max: dem[2], step: dem[3] };
    atc.sob = vec_of::<f32>(d, "sob")?[0];
    atc.xy_aod.domain = domain.clone();
    let one = |key: &str, dst: &mut Vec<f32>| -> PyResult<()> {
        let v: Vec<f32> = vec_of(d, key)?;
        if v.len() != gn {
            return Err(PyValueError::new_err(format!("{key} needs one value per coarse cell")));
        }
        *dst = v;
        Ok(())
    };
    one("sun_czen", &mut atc.xy_sun.bands[C_ZEN])?;
    one("sun_azi", &mut atc.xy_sun.bands[AZI])?;
    one("view_czen", &mut atc.xy_view.bands[C_ZEN])?;
    one("view_azi", &mut atc.xy_view.bands[AZI])?;
    one("pr", &mut atc.xy_pr.bands[0])?;
    one("pa", &mut atc.xy_pa.bands[0])?;
    one("hr", &mut atc.xy_hr.bands[0])?;
    one("fresnel", &mut atc.xy_fresnel.bands[0])?;
    one("dem_bin", &mut atc.xy_dem.bands[0])?;
    for (key, which) in [("tso", 0), ("tvo", 1), ("tg", 2), ("mod", 3), ("brdf", 4)] {
        let v: Vec<f32> = vec_of(d, key)?;
        if v.len() != nb * gn {
            return Err(PyValueError::new_err(format!("{key} needs bands x cells values")));
        }
        let brick = match which {
            0 => &mut atc.xy_tso,
            1 => &mut atc.xy_tvo,
            2 => &mut atc.xy_tg,
            3 => &mut atc.xy_mod,
            _ => &mut atc.xy_brdf,
        };
        for (b, band) in brick.bands.iter_mut().enumerate() {
            band.copy_from_slice(&v[b * gn..(b + 1) * gn]);
        }
    }
    let toa_v: Vec<i16> = vec_of(d, "toa")?;
    if toa_v.len() != nb * npix {
        return Err(PyValueError::new_err("toa needs bands x pixels values"));
    }
    let toa = Toa { grid: SceneGrid { nx, ny, res }, bands: toa_v.chunks(npix).map(|c| c.to_vec()).collect(), domain };
    let qai: Vec<i16> = vec_of(d, "qai")?;
    let top = Top { dem: vec_of(d, "top_dem")?, slope: vec_of(d, "slope")?, ill: vec_of(d, "ill")?, sky: vec_of(d, "sky")? };
    let green = atc.xy_aod.find_domain("GREEN").ok_or_else(|| PyValueError::new_err("no GREEN band"))?;
    let (blue, red, sw2) = (
        atc.xy_aod.find_domain("BLUE").ok_or_else(|| PyValueError::new_err("no BLUE band"))?,
        atc.xy_aod.find_domain("RED").ok_or_else(|| PyValueError::new_err("no RED band"))?,
        atc.xy_aod.find_domain("SWIR2").ok_or_else(|| PyValueError::new_err("no SWIR2 band"))?,
    );

    let run = || -> Result<(Vec<u8>, f64, f64), aod::AodError> {
        let mut o = Out(Vec::new());
        let t0 = std::time::Instant::now();
        let mut dark = DarkTargets {
            wat: aod::extract_dark_target(&atc, &toa, &qai, &top, TargetType::Water)?,
            shd: aod::extract_dark_target(&atc, &toa, &qai, &top, TargetType::Shadow)?,
            veg: aod::extract_dark_target(&atc, &toa, &qai, &top, TargetType::Vegetation)?,
            ..Default::default()
        };
        let t_extract = t0.elapsed().as_secs_f64() * 1e3;
        dump_objects(&mut o, &dark.wat);
        dump_objects(&mut o, &dark.veg);

        let wlib = aod::water_lib(nb, &rsr);
        dump_libmatch(&mut o, &atc, res, multi, &dark.wat, &wlib, TargetType::Water)?;
        let vlib = aod::veg_lib(nb, blue, green, red);
        {
            // what aod_from_target sets before the vegetation estimation
            let save = atc.aod_bands.clone();
            atc.aod_bands = vec![false; nb];
            for b in [blue, green, red, sw2] {
                atc.aod_bands[b] = true;
            }
            dump_libmatch(&mut o, &atc, res, multi, &dark.veg, &vlib, TargetType::Vegetation)?;
            atc.aod_bands = save;
        }

        let t1 = std::time::Instant::now();
        dark.nwat = aod::aod_from_target(&mut atc, &rsr, res, multi, &mut dark.wat, TargetType::Water)?;
        dark.nshd = aod::aod_from_target(&mut atc, &rsr, res, multi, &mut dark.shd, TargetType::Shadow)?;
        dark.nveg = aod::aod_from_target(&mut atc, &rsr, res, multi, &mut dark.veg, TargetType::Vegetation)?;
        let t_fit = t1.elapsed().as_secs_f64() * 1e3;
        dump_fit(&mut o, &dark.wat, dark.nwat);
        dump_fit(&mut o, &dark.veg, dark.nveg);

        aod::aod_elev_dependency(&mut atc, &mut dark, green)?;
        o.f(atc.hp);
        o.f(atc.ha);
        o.fs(&atc.xy_ha.bands[0]);
        for t in dark.wat.iter().chain(dark.veg.iter()) {
            o.f(t.ha);
        }
        Ok((o.0, t_extract, t_fit))
    };
    let (bytes, t_extract, t_fit) = py.detach(run).map_err(err)?;
    Ok((PyBytes::new(py, &bytes), t_extract, t_fit))
}

/// `aod_elev_dependency` on small synthetic cases `(dem_avg, dem_min, dem_max, dem_step, view[4], dem_bin[4],
/// targets[(z, aod_green)])`: returns, per case, the bits (u32) of `Hp`, `Ha`, the 4 cells of `xy_Ha` and the `Ha` of
/// each target.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn elev_cases(cases: Vec<(f32, f32, f32, f32, Vec<f32>, Vec<f32>, Vec<(f32, f32)>)>) -> PyResult<Vec<Vec<u32>>> {
    let mut out = Vec::new();
    for (avg, mn, mx, step, view, dembin, targets) in cases {
        if view.len() != 4 || dembin.len() != 4 {
            return Err(PyValueError::new_err("view and dem_bin need 4 values"));
        }
        let mut atc = Atc::zeros(2, 2, 1.0, 3, -9999);
        atc.dem = zeit_core::atc::DemStats { avg, min: mn, max: mx, step };
        atc.xy_view.bands[0].copy_from_slice(&view);
        atc.xy_dem.bands[0].copy_from_slice(&dembin);
        let n = targets.len() as i32;
        let wat: Vec<DarkObj> = targets
            .iter()
            .map(|&(z, a)| DarkObj { valid: true, z, aod: vec![0.0, a, 0.0], ..Default::default() })
            .collect();
        let mut dark = DarkTargets { wat, nwat: n, ..Default::default() };
        aod::aod_elev_dependency(&mut atc, &mut dark, 1).map_err(err)?;
        let mut v = vec![atc.hp.to_bits(), atc.ha.to_bits()];
        v.extend(atc.xy_ha.bands[0].iter().map(|x| x.to_bits()));
        v.extend(dark.wat.iter().map(|t| t.ha.to_bits()));
        out.push(v);
    }
    Ok(out)
}

/// `aodfileread` for each scenario `(dir, lon, lat, doy)` (`dir` "NULL": the global fallback): the bits (u32) of the
/// spectrum, or `None` where FORCE returns `NULL`.
#[pyfunction]
fn aodfile(wvl: Vec<f32>, lwvl: Vec<f32>, scenarios: Vec<(String, f64, f64, i32)>) -> Vec<Option<Vec<u32>>> {
    let nb = wvl.len();
    let mut atc = Atc::zeros(1, 1, 1.0, nb, -9999);
    atc.wvl = wvl;
    atc.lwvl = lwvl;
    scenarios
        .into_iter()
        .map(|(dir, lon, lat, doy)| {
            let dir = if dir == "NULL" { None } else { Some(dir) };
            aod::aodfileread(&atc, dir.as_deref(), lon, lat, doy).ok().map(|v| v.iter().map(|x| x.to_bits()).collect())
        })
        .collect()
}

/// The water and the vegetation spectral libraries (`water_lib`, `veg_lib`) for the bands `rsr_band`: `(water, veg)`,
/// each `n x nb` flat.
#[pyfunction]
fn libraries<'py>(
    py: Python<'py>,
    rsr_band: Vec<usize>,
    blue: usize,
    green: usize,
    red: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let nb = rsr_band.len();
    if rsr_band.iter().any(|&b| b >= 56) || blue >= nb || green >= nb || red >= nb {
        return Err(PyValueError::new_err("an RSR band must be in 0..56 and the bands in 0..nb"));
    }
    let flat = |l: SpecLib| PyArray1::from_vec(py, l.s.into_iter().flatten().collect::<Vec<f32>>());
    Ok((flat(aod::water_lib(nb, &rsr_band)), flat(aod::veg_lib(nb, blue, green, red))))
}

/// `dist_transform_` and `connectedcomponents_` of FORCE on a 0/1 image: `(distance, labels, n)`.
#[pyfunction]
fn image_functions<'py>(
    py: Python<'py>,
    image: PyReadonlyArray1<'py, u8>,
    nx: usize,
    ny: usize,
) -> PyResult<(Bound<'py, PyArray1<u16>>, Bound<'py, PyArray1<i32>>, i32)> {
    let img = image.as_slice().map_err(|_| PyValueError::new_err("arrays must be contiguous"))?;
    if img.len() != nx * ny || nx == 0 || ny == 0 {
        return Err(PyValueError::new_err("the image needs nx * ny values"));
    }
    let dist = aod::dist_transform(img, nx, ny);
    let (lab, n) = aod::connected_components(img, nx, ny);
    Ok((PyArray1::from_vec(py, dist), PyArray1::from_vec(py, lab), n))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "aod")?;
    m.add(
        "__doc__",
        "Dark targets and the AOD over each of them of FORCE's atmospheric correction (pieces, for the comparison with FORCE)",
    )?;
    m.add_function(wrap_pyfunction!(scene, &m)?)?;
    m.add_function(wrap_pyfunction!(elev_cases, &m)?)?;
    m.add_function(wrap_pyfunction!(aodfile, &m)?)?;
    m.add_function(wrap_pyfunction!(image_functions, &m)?)?;
    m.add_function(wrap_pyfunction!(libraries, &m)?)?;
    Ok(m)
}
