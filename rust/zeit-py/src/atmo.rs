//! zeit._core.atmo: a minimal binding of the atmosphere of the coarse grid of zeit-core (a port of FORCE's `atc-ll.c`,
//! `sunview-ll.c` and the functions of `atmo-ll.c` before the surface reflectance: `allocate_atc`, `sun_target_view`,
//! `atmo_angledep`, `atmo_elevdep`, `interpolation_weights`, `interpolate_coarse`, `apply_aoi`), for the comparison with
//! FORCE. The public function (`zeit.atmospheric_correction`) comes later.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use zeit_core::atc::{Atc, AtmoError, CoarseBrick, DemStats, Dn, ZEN};
use zeit_core::atmo::{self, L2Params};
use zeit_core::gas::{SceneGrid, Toa, WvpSource};
use zeit_core::l2::{radiometric_correction, L2Options};
use zeit_core::topo;
use zeit_core::meta::{Meta, S2Meta};
use zeit_core::radtran::Date;
use zeit_core::sunview::{self, GeoTable, Mission, SceneGeo};

fn err(e: AtmoError) -> PyErr {
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
    fn d(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn fs(&mut self, v: &[f32]) {
        v.iter().for_each(|&x| self.f(x));
    }
    fn i16s(&mut self, v: &[i16]) {
        v.iter().for_each(|x| self.0.extend_from_slice(&x.to_le_bytes()));
    }
    /// One brick: its band count, size, resolution, no-data and domain of every band.
    fn meta(&mut self, b: &CoarseBrick) {
        self.i(b.nb() as i32);
        self.i(b.nx as i32);
        self.i(b.ny as i32);
        self.d(b.res);
        for _ in 0..b.nb() {
            self.i(b.nodata);
        }
        for dom in &b.domain {
            let mut buf = [0u8; 16];
            buf[..dom.len().min(15)].copy_from_slice(&dom.as_bytes()[..dom.len().min(15)]);
            self.0.extend_from_slice(&buf);
        }
    }
    fn bands(&mut self, b: &CoarseBrick) {
        for band in &b.bands {
            self.fs(band);
        }
    }
    fn xyz(&mut self, z: &[CoarseBrick]) {
        for b in z {
            self.bands(b);
        }
    }
    /// A brick of the dump of `level2`: the number of bands, then the bands.
    fn bands_n(&mut self, b: &CoarseBrick) {
        self.i(b.nb() as i32);
        self.bands(b);
    }
    /// A family of bricks of the dump of `level2`: the number of bands, then z-major.
    fn xyz_n(&mut self, z: &[CoarseBrick]) {
        self.i(z[0].nb() as i32);
        self.xyz(z);
    }
    fn scalars(&mut self, atc: &Atc) {
        self.fs(&atc.cosszen);
        self.fs(&atc.cosvzen);
    }
    fn view(&mut self, atc: &Atc) {
        let v = &atc.view;
        self.fs(&[v.a, v.b, v.c, v.ab, v.geo_angle_nadir, v.h, v.h2]);
    }
}

fn dump_after_alloc(o: &mut Out, atc: &Atc) {
    let nb = atc.xy_mod.nb();
    o.i(atc.nx);
    o.i(atc.ny);
    o.i(atc.nc);
    o.i(atc.res);
    o.f(atc.nodata);
    o.scalars(atc);
    o.fs(&[atc.wvp, atc.cc, atc.hr, atc.ha, atc.hp, atc.km, atc.fr, atc.fa]);
    o.fs(&[atc.dem.avg, atc.dem.min, atc.dem.max, atc.dem.step]);
    o.i(atc.dem.cnum);
    o.view(atc);
    o.fs(&[atc.tthg.g1, atc.tthg.g2, atc.tthg.alpha]);
    o.fs(&atc.tthg.hg);
    o.f(atc.tthg.sob);
    o.i(atc.aodmap as i32);
    for b in 0..nb {
        o.fs(&[atc.wvl[b], atc.lwvl[b], atc.lwvl2[b], atc.e0[b], atc.od[b], atc.mod_[b], atc.aod[b]]);
        o.i(atc.aod_bands[b] as i32);
    }
    for b in [
        &atc.xy_mod, &atc.xy_aod, &atc.xy_pr, &atc.xy_pa, &atc.xy_hr, &atc.xy_ha, &atc.xy_tvw, &atc.xy_tsw, &atc.xy_tvo,
        &atc.xy_tso, &atc.xy_tg, &atc.xy_brdf, &atc.xy_fresnel, &atc.xy_dem, &atc.xy_interp, &atc.xy_view, &atc.xy_sun,
        &atc.xy_psi,
    ] {
        o.meta(b);
    }
    for z in [
        &atc.xyz_od, &atc.xyz_mod, &atc.xyz_aod, &atc.xyz_hr, &atc.xyz_ha, &atc.xyz_rho_p, &atc.xyz_ts, &atc.xyz_tsd,
        &atc.xyz_tss, &atc.xyz_tv, &atc.xyz_tvd, &atc.xyz_tvs, &atc.xyz_t, &atc.xyz_s, &atc.xyz_f, &atc.xyz_tg,
    ] {
        o.meta(&z[0]);
        o.meta(&z[z.len() - 1]);
    }
}

/// The scene of FORCE's atmosphere functions (see `benchmarks/vs_originals/force_atmo/main.inc` for the layout of the
/// arrays, here the keys of a dict): runs `allocate_atc`, `sun_target_view`, `atmo_angledep`, `atmo_elevdep`, the
/// interpolation weights of a list of pixels and `apply_aoi`, and returns the dumps of the stages as bytes in the
/// layout the C harness writes, with the times (ms) of the three stages.
#[pyfunction]
fn scene<'py>(py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<(Bound<'py, PyBytes>, f64, f64, f64)> {
    let mission = match get(d, "mission")?.extract::<i32>()? {
        0 => Mission::Landsat,
        1 => Mission::Sentinel2,
        m => return Err(PyValueError::new_err(format!("mission {m}: 0 is Landsat, 1 Sentinel-2"))),
    };
    let (nx, ny): (usize, usize) = (get(d, "nx")?.extract()?, get(d, "ny")?.extract()?);
    let res: f64 = get(d, "res")?.extract()?;
    let (ulx, uly): (f64, f64) = (get(d, "ulx")?.extract()?, get(d, "uly")?.extract()?);
    let domain: Vec<String> = get(d, "domain")?.extract()?;
    let wavelength: Vec<f32> = vec_of(d, "wavelength")?;
    let rsr: Vec<usize> = vec_of::<i32>(d, "rsr_band")?.into_iter().map(|b| b as usize).collect();
    let nb = domain.len();
    if wavelength.len() != nb || rsr.len() != nb || rsr.iter().any(|&b| b >= 56) {
        return Err(PyValueError::new_err("domain, wavelength and rsr_band need nb entries, rsr_band in 0..56"));
    }
    let i = |k: &str| -> PyResult<i32> { get(d, k)?.extract() };
    let mut date = Date::from_year_doy(i("year")?, i("doy")?);
    (date.hh, date.mm, date.ss, date.tz) = (i("hh")?, i("mm")?, i("ss")?, i("tz")?);
    let (lon0, x0, kgeo): (f64, f64, f64) = (get(d, "lon0")?.extract()?, get(d, "x0")?.extract()?, get(d, "kgeo")?.extract()?);
    let l2 = L2Params { domulti: i("domulti")? != 0, doenv: i("doenv")? != 0, ..Default::default() };
    let aodmap = i("aodmap")? != 0;
    let dem4: Vec<f32> = vec_of(d, "dem4")?;
    let cnum = i("cnum")?;
    let sc: Vec<f32> = vec_of(d, "wvp_hp_ha")?;
    let (mod_, aod): (Vec<f32>, Vec<f32>) = (vec_of(d, "mod")?, vec_of(d, "aod")?);
    let gn = i("gn")? as usize;
    let (xy_aod, xy_interp): (Vec<f32>, Vec<f32>) = (vec_of(d, "xy_aod")?, vec_of(d, "xy_interp")?);
    let mut qai: Vec<i16> = vec_of(d, "qai")?;
    let top_dem: Vec<u8> = vec_of(d, "top_dem")?;
    let npix = nx * ny;
    if dem4.len() != 4 || sc.len() != 3 || mod_.len() != nb || aod.len() != nb || xy_aod.len() != nb * gn || xy_interp.len() != gn
        || qai.len() != npix || top_dem.len() != npix
    {
        return Err(PyValueError::new_err("array sizes: dem4 4, wvp_hp_ha 3, mod and aod nb, xy_aod nb x gn, xy_interp gn, qai and top_dem nx x ny"));
    }
    let mut meta = Meta::from_rsr_bands(&rsr);
    if mission == Mission::Sentinel2 {
        meta.s2 = S2Meta {
            nodata: get(d, "s2nodata")?.extract()?,
            vzen: vec_of(d, "vzen")?,
            vazi: vec_of(d, "vazi")?,
            nx: get(d, "s2nx")?.extract()?,
            ny: get(d, "s2ny")?.extract()?,
        };
    }
    // the interpolation weights
    let w_ji: Vec<i32> = vec_of(d, "w_ji")?;
    let (w_res, w_full): (f64, f64) = (get(d, "w_res")?.extract()?, get(d, "w_full_res")?.extract()?);
    let (w_nf, w_ne): (i32, i32) = (get(d, "w_nf")?.extract()?, get(d, "w_ne")?.extract()?);
    let w_nodata: f32 = get(d, "w_nodata")?.extract()?;
    let coarse: Vec<f32> = vec_of(d, "coarse")?;
    if !w_ji.is_empty() {
        if coarse.len() != (w_nf * w_ne) as usize || w_ji.len() % 2 != 0 {
            return Err(PyValueError::new_err("coarse needs nf x ne values, w_ji (j, i) pairs"));
        }
        for p in w_ji.chunks(2) {
            let (e_, f_) = (p[1] as f64 * w_full / w_res, p[0] as f64 * w_full / w_res);
            if p[0] < 0 || p[1] < 0 || e_.floor() >= w_ne as f64 || f_.floor() >= w_nf as f64 {
                return Err(PyValueError::new_err("a pixel of w_ji is outside the coarse grid"));
            }
        }
    }
    let aoi: Vec<u8> = vec_of(d, "aoi")?;
    let n_jobs = d.get_item("n_jobs")?.map(|v| v.extract::<i32>()).transpose()?.unwrap_or(1);

    let qai_in = qai.clone();
    let run = || -> Result<(Vec<u8>, f64, f64, f64), AtmoError> {
        let mut o = Out(Vec::new());
        let dn = Dn { nx, ny, res, wavelength, domain };
        let mut atc = Atc::allocate(&dn, &meta)?;
        if atc.nc as usize != gn {
            return Err(AtmoError::Shape(format!("the grid has {} cells, the input {gn}", atc.nc)));
        }
        dump_after_alloc(&mut o, &atc);

        // what the stages before set
        atc.dem = DemStats { avg: dem4[0], min: dem4[1], max: dem4[2], step: dem4[3], cnum };
        (atc.wvp, atc.hp, atc.ha) = (sc[0], sc[1], sc[2]);
        atc.aodmap = aodmap;
        atc.mod_ = mod_.clone();
        atc.aod = aod.clone();
        for b in 0..nb {
            atc.xy_aod.bands[b].copy_from_slice(&xy_aod[b * gn..(b + 1) * gn]);
        }
        atc.xy_interp.bands[0].copy_from_slice(&xy_interp);

        let warp = move |x: f64, y: f64| (lon0 + (x - x0) * kgeo, y * kgeo);
        let geo = SceneGeo { ulx, uly, date, to_geo: &warp };
        let grid = SceneGrid { nx, ny, res };

        let t0 = std::time::Instant::now();
        sunview::sun_target_view(&mut atc, &meta, mission, &geo, &grid, &mut qai)?;
        let t_sun = t0.elapsed().as_secs_f64() * 1e3;
        o.i(0);
        o.bands(&atc.xy_sun);
        o.bands(&atc.xy_view);
        o.scalars(&atc);
        o.view(&atc);
        o.i16s(&qai);

        let t0 = std::time::Instant::now();
        atmo::atmo_angledep(&mut atc, &meta, &geo, &grid, &qai, &top_dem, None, n_jobs)?;
        let t_angle = t0.elapsed().as_secs_f64() * 1e3;
        o.i(0);
        for b in [
            &atc.xy_brdf, &atc.xy_psi, &atc.xy_pr, &atc.xy_pa, &atc.xy_tsw, &atc.xy_tvw, &atc.xy_tso, &atc.xy_tvo, &atc.xy_tg,
            &atc.xy_dem, &atc.xy_hr, &atc.xy_mod, &atc.xy_fresnel,
        ] {
            o.bands(b);
        }

        let t0 = std::time::Instant::now();
        atmo::atmo_elevdep(&mut atc, &l2, &grid, &mut qai, &top_dem, n_jobs)?;
        let t_elev = t0.elapsed().as_secs_f64() * 1e3;
        o.i(0);
        for z in [
            &atc.xyz_od, &atc.xyz_mod, &atc.xyz_aod, &atc.xyz_rho_p, &atc.xyz_ts, &atc.xyz_tsd, &atc.xyz_tss, &atc.xyz_tv,
            &atc.xyz_tvd, &atc.xyz_tvs, &atc.xyz_t, &atc.xyz_s, &atc.xyz_f, &atc.xyz_hr, &atc.xyz_ha,
        ] {
            o.xyz(z);
        }
        o.i16s(&qai);

        for p in w_ji.chunks(2) {
            let w = atmo::interpolation_weights(p[0], p[1], w_nf, w_ne, w_res, w_full, &coarse, w_nodata);
            for v in [w.gul, w.gur, w.gll, w.glr] {
                o.i(v);
            }
            o.fs(&[w.wul, w.wur, w.wll, w.wlr, w.wtop, w.wdown]);
            o.f(atmo::interpolate_coarse(&w, &coarse));
        }

        // apply_aoi on the QAI as it came in
        let mut qai0 = qai_in.clone();
        let ret = atmo::apply_aoi(&mut qai0, if aoi.is_empty() { None } else { Some(&aoi) })?;
        o.i(match ret {
            atmo::Aoi::Success => 0,
            atmo::Aoi::Cancel => 10,
        });
        o.i16s(&qai0);

        o.d(t_sun);
        o.d(t_angle);
        o.d(t_elev);
        Ok((o.0, t_sun, t_angle, t_elev))
    };
    let (bytes, t_sun, t_angle, t_elev) = py.detach(run).map_err(err)?;
    Ok((PyBytes::new(py, &bytes), t_sun, t_angle, t_elev))
}

/// The product of the dump of `l2`: its bands (domain, wavelength) and data.
fn product(o: &mut Out, domain: &[String], wavelength: &[f32], bands: &[&[i16]]) {
    o.i(bands.len() as i32);
    for (dom, w) in domain.iter().zip(wavelength) {
        let mut buf = [0u8; 16];
        buf[..dom.len().min(15)].copy_from_slice(&dom.as_bytes()[..dom.len().min(15)]);
        o.0.extend_from_slice(&buf);
        o.f(*w);
    }
    for b in bands {
        o.i16s(b);
    }
}

/// The whole Level 2 radiometric correction of FORCE (`allocate_atc`, `sun_target_view`, `compile_topography` and
/// `radiometric_correction`: the surface reflectance and the products) on a scene whose TOA reflectance and QAI are given
/// (see `benchmarks/vs_originals/force_l2/main.c` for the layout of the dict, which is the layout of the file the C harness
/// reads). Returns the dump (the layout of the C harness' output: the status, the time, the atc, the products) and the time
/// of `radiometric_correction` in ms. The BRDF factor is the one the harness computes (a function of the view zenith and the
/// band), as `brdf-ll.c` is not ported.
#[pyfunction]
fn level2<'py>(py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<(Bound<'py, PyBytes>, f64)> {
    let mission = match get(d, "mission")?.extract::<i32>()? {
        0 => Mission::Landsat,
        1 => Mission::Sentinel2,
        m => return Err(PyValueError::new_err(format!("mission {m}: 0 is Landsat, 1 Sentinel-2"))),
    };
    let (nx, ny): (usize, usize) = (get(d, "nx")?.extract()?, get(d, "ny")?.extract()?);
    let res: f64 = get(d, "res")?.extract()?;
    let (ulx, uly): (f64, f64) = (get(d, "ulx")?.extract()?, get(d, "uly")?.extract()?);
    let domain: Vec<String> = get(d, "domain")?.extract()?;
    let wavelength: Vec<f32> = vec_of(d, "wavelength")?;
    let rsr: Vec<usize> = vec_of::<i32>(d, "rsr_band")?.into_iter().map(|b| b as usize).collect();
    let nb = domain.len();
    if wavelength.len() != nb || rsr.len() != nb || rsr.iter().any(|&b| b >= 56) {
        return Err(PyValueError::new_err("domain, wavelength and rsr_band need nb entries, rsr_band in 0..56"));
    }
    let i = |k: &str| -> PyResult<i32> { get(d, k)?.extract() };
    let mut date = Date::from_year_doy(i("year")?, i("doy")?);
    (date.hh, date.mm, date.ss, date.tz) = (i("hh")?, i("mm")?, i("ss")?, i("tz")?);
    let (lon0, x0, kgeo): (f64, f64, f64) = (get(d, "lon0")?.extract()?, get(d, "x0")?.extract()?, get(d, "kgeo")?.extract()?);
    let par = L2Params {
        domulti: i("domulti")? != 0,
        doenv: i("doenv")? != 0,
        doatmo: i("doatmo")? != 0,
        dotopo: i("dotopo")? != 0,
        dobrdf: i("dobrdf")? != 0,
        erase_cloud: i("erase_cloud")? != 0,
        doaod: i("doaod")? != 0,
        odst: i("odst")? != 0,
        ovzn: i("ovzn")? != 0,
        ohot: i("ohot")? != 0,
        oaod: i("oaod")? != 0,
        owvp: i("owvp")? != 0,
    };
    let wvp_const: f32 = get(d, "wvp")?.extract()?;
    let dem_nodata = i("dem_nodata")?;
    let dem: Vec<f32> = vec_of(d, "dem")?;
    let gn = i("gn")? as usize;
    let qai: Vec<i16> = vec_of(d, "qai")?;
    let toa_flat: Vec<i16> = vec_of(d, "toa")?;
    let npix = nx * ny;
    if qai.len() != npix || toa_flat.len() != nb * npix || !(dem.is_empty() || dem.len() == npix) {
        return Err(PyValueError::new_err("array sizes: qai nx x ny, toa nb x nx x ny, dem empty or nx x ny"));
    }
    let mut meta = Meta::from_rsr_bands(&rsr);
    if mission == Mission::Sentinel2 {
        meta.s2 = S2Meta {
            nodata: get(d, "s2nodata")?.extract()?,
            vzen: vec_of(d, "vzen")?,
            vazi: vec_of(d, "vazi")?,
            nx: get(d, "s2nx")?.extract()?,
            ny: get(d, "s2ny")?.extract()?,
        };
    }
    let aoi: Vec<u8> = vec_of(d, "aoi")?;
    // a real scene: the longitude and latitude of the corners of the coarse cells by table (`geo_n` x `geo_n` corners of
    // `geo_res` metres) and, if given, the sun angles of the coarse cells in degrees, instead of the affine map and the sun
    // of the core
    let geo_n: usize = d.get_item("geo_n")?.map(|v| v.extract()).transpose()?.unwrap_or(0);
    let table = if geo_n > 0 {
        Some(GeoTable { ulx, uly, res: get(d, "geo_res")?.extract()?, nx: geo_n, ny: geo_n, lon: vec_of(d, "geo_lon")?, lat: vec_of(d, "geo_lat")? })
    } else {
        None
    };
    let sun_tab: Option<(Vec<f32>, Vec<f32>)> = if d.contains("sun_zen")? { Some((vec_of(d, "sun_zen")?, vec_of(d, "sun_azi")?)) } else { None };
    let n_jobs = d.get_item("n_jobs")?.map(|v| v.extract::<i32>()).transpose()?.unwrap_or(1);

    let run = || -> Result<(Vec<u8>, f64), String> {
        let e = |x: &dyn std::fmt::Display| x.to_string();
        let dn = Dn { nx, ny, res, wavelength, domain: domain.clone() };
        let mut atc = Atc::allocate(&dn, &meta).map_err(|x| e(&x))?;
        if atc.nc as usize != gn {
            return Err(format!("the grid has {} cells, the input {gn}", atc.nc));
        }
        let warp = |x: f64, y: f64| match &table {
            Some(t) => t.lookup(x, y),
            None => (lon0 + (x - x0) * kgeo, y * kgeo),
        };
        let geo = SceneGeo { ulx, uly, date, to_geo: &warp };
        let grid = SceneGrid { nx, ny, res };
        let mut qai = qai;
        match &sun_tab {
            None => sunview::sun_target_view(&mut atc, &meta, mission, &geo, &grid, &mut qai),
            Some((zen, azi)) => {
                sunview::sun_target_view_with(&mut atc, &meta, mission, &geo, &grid, &mut qai, &|g, _, _, _| (zen[g], azi[g]))
            }
        }
        .map_err(|x| e(&x))?;
        let mut top = topo::compile_topography(if dem.is_empty() { None } else { Some(&dem) }, dem_nodata, nx, ny, res, &mut atc, &mut qai)
            .map_err(|x| e(&x))?;

        let toa = Toa { grid, bands: toa_flat.chunks(npix).map(|c| c.to_vec()).collect(), domain: domain.clone() };
        let opts = L2Options { aod_dir: None, wvp: WvpSource { dir: None, wvp: wvp_const, strict: false } };
        let brdf = |atc: &Atc, g: usize, out: &mut [f32]| {
            let zen = atc.xy_view.bands[ZEN][g];
            for (b, o) in out.iter_mut().enumerate() {
                *o = 1.0f32 + 0.5f32 * zen * (1.0f32 + 0.1f32 * b as f32);
            }
        };
        let t0 = std::time::Instant::now();
        let res_l2 = radiometric_correction(
            &par,
            &opts,
            &meta,
            mission,
            &geo,
            &mut atc,
            &toa,
            &mut qai,
            if aoi.is_empty() { None } else { Some(&aoi) },
            &mut top,
            Some(&brdf),
            n_jobs,
        );
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut o = Out(Vec::new());
        let lv = match res_l2 {
            Ok(lv) => lv,
            Err(err) => {
                o.i(1);
                o.d(ms);
                let _ = err;
                return Ok((o.0, ms));
            }
        };
        o.i(0);
        o.d(ms);
        // the atc
        o.fs(&[atc.wvp, atc.hr, atc.ha, atc.hp]);
        o.i(atc.aodmap as i32);
        o.fs(&[atc.dem.avg, atc.dem.min, atc.dem.max, atc.dem.step]);
        o.i(atc.dem.cnum);
        o.scalars(&atc);
        for b in 0..nb {
            o.fs(&[atc.mod_[b], atc.aod[b], atc.od[b]]);
            o.i(atc.aod_bands[b] as i32);
        }
        for b in [
            &atc.xy_mod, &atc.xy_aod, &atc.xy_pr, &atc.xy_pa, &atc.xy_hr, &atc.xy_ha, &atc.xy_tvw, &atc.xy_tsw, &atc.xy_tvo,
            &atc.xy_tso, &atc.xy_tg, &atc.xy_brdf, &atc.xy_fresnel, &atc.xy_dem, &atc.xy_interp, &atc.xy_view, &atc.xy_sun,
            &atc.xy_psi,
        ] {
            o.bands_n(b);
        }
        for z in [
            &atc.xyz_od, &atc.xyz_mod, &atc.xyz_aod, &atc.xyz_rho_p, &atc.xyz_ts, &atc.xyz_tsd, &atc.xyz_tss, &atc.xyz_tv,
            &atc.xyz_tvd, &atc.xyz_tvs, &atc.xyz_t, &atc.xyz_s, &atc.xyz_f, &atc.xyz_hr, &atc.xyz_ha,
        ] {
            o.xyz_n(z);
        }
        // the products: QAI, BOA, then the optional ones in the order of compile_level2
        let nprod = 2 + [&lv.dst, &lv.vzn, &lv.hot, &lv.aod, &lv.wvp].iter().filter(|p| p.is_some()).count();
        o.i(nprod as i32);
        product(&mut o, &["QAI".to_string()], &[1.0], &[&lv.qai]);
        let bands: Vec<&[i16]> = lv.boa.bands.iter().map(|b| b.as_slice()).collect();
        product(&mut o, &lv.boa.domain, &lv.boa.wavelength, &bands);
        let green = atc.xy_aod.find_domain("GREEN").map_or(1.0, |g| atc.wvl[g]);
        for (name, wl, p) in [("DST", 1.0, &lv.dst), ("VZN", 1.0, &lv.vzn), ("HOT", 1.0, &lv.hot), ("AOD", green, &lv.aod), ("WVP", 1.0, &lv.wvp)] {
            if let Some(p) = p {
                product(&mut o, &[name.to_string()], &[wl], &[p]);
            }
        }
        // the gaseous transmittance map of every band (Sentinel-2, with the WVP product), from the final QAI
        match (&lv.wvp, mission == Mission::Sentinel2 && par.doatmo) {
            (Some(wvp), true) => {
                let lut = zeit_core::gas::WvLut::build(&rsr, &atc, n_jobs).map_err(|x| e(&x))?;
                o.i(nb as i32);
                for b in 0..nb {
                    let tg = zeit_core::gas::gas_transmittance(&atc, b, wvp, &grid, &lv.qai, &lut).map_err(|x| e(&x))?;
                    o.i16s(&tg);
                }
            }
            _ => o.i(0),
        }
        Ok((o.0, ms))
    };
    let (bytes, ms) = py.detach(run).map_err(PyValueError::new_err)?;
    Ok((PyBytes::new(py, &bytes), ms))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "atmo")?;
    m.add("__doc__", "The atmosphere of the coarse grid of FORCE's atmospheric correction, and the Level 2 correction (pieces, for the comparison with FORCE)")?;
    m.add_function(wrap_pyfunction!(scene, &m)?)?;
    m.add_function(wrap_pyfunction!(level2, &m)?)?;
    Ok(m)
}
