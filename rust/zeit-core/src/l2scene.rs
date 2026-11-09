//! The atmospheric correction of one Sentinel-2 Level 1C scene, from the digital numbers to the Level 2 products: what
//! FORCE 3.11.00's `force-l2ps` (`src/main/lower-level/force-l2ps.c`, Copyright (C) 2013-2022 David Frantz,
//! GPL-3.0-or-later, https://github.com/davidfrantz/force) does for one image, in its order, on the pieces of the core:
//!
//! 1. `allocate_atc` ([`Atc::allocate`]) from the metadata of [`crate::s2l1c`];
//! 2. `bounds_level1` ([`s2l1c::bounds_level1`]): the initial QAI (no-data, saturation);
//! 3. `sun_target_view` ([`sunview::sun_target_view_with`]): the sun and view angles of the coarse grid, the low sun
//!    flag and the cells without view angles;
//! 4. `convert_level1` ([`s2l1c::convert_level1`]): the TOA reflectance (after the QAI: the off pixels are no-data);
//! 5. `compile_topography` ([`topo::compile_topography`]): slope, aspect, illumination and the binned DEM;
//! 6. the cloud, shadow, snow and water flags (`detect_clouds` is not part of the core: they are an argument and are
//!    or-ed into the QAI);
//! 7. `radiometric_correction` ([`l2::radiometric_correction`]) with the BRDF factor of [`brdf_fn`].
//!
//! What the core does not do, and the caller gives:
//!
//! * **The reading of the rasters.** The 13 bands as `u16` on the grid of the finest band (the 20 and 60 m bands
//!   replicated by nearest neighbour, as `read_level1` does through GDAL), and the DEM on that grid, in metres, with the
//!   `DEM_NODATA` value where it has none (FORCE warps the DEM file to the grid of the scene with GDAL, bilinear by
//!   default; see [`crate::topo`]).
//! * **The projection.** The core has neither PROJ nor GDAL. The only map points the correction converts to longitude and
//!   latitude are the upper left corners of the cells of the coarse grid of `Atc` (5000 m): [`geo_grid`] says which, the
//!   caller transforms them and gives the result as a [`GeoTable`].
//! * **The cloud detection.** The cloud, shadow, snow and water bits of the QAI ([`FLAG_MASK`]) are given per pixel.
//! * **The output grid.** The products are on the grid of the scene; FORCE's reprojection to a datacube is not done.
//!
//! **The BRDF.** `brdf_factor` of FORCE's `brdf-ll.c` is not ported (and not read: its license is not verified). FORCE calls it
//! for every coarse cell whatever `DO_BRDF` says (the factor enters the AOD retrieval over water; `DO_BRDF` only decides whether
//! the surface reflectance is multiplied by it), and so does this. The factor of a coarse cell and band is Zeit's NBAR c-factor, written from Roy et al. (2016, 2017)
//! ([`crate::brdf`]): the ratio of the model at nadir with the standard sun zenith ([`radtran::standard_sunzenith`], the
//! sun zenith at the usual overpass time of the latitude of the cell) to the model at the sun and view angles of the cell.
//! The three red edge bands take the interpolation of Roy et al. (2017), the broad and narrow NIR the NIR parameters, and
//! the ultra blue, water vapour and cirrus bands no correction (factor 1; for the ultra blue that is what `force-l2ps`
//! outputs in every cell of the two scenes compared, taken from its results and not from its code). It is not FORCE's
//! code, so any difference between the two models shows in the result (with `dobrdf` in the surface reflectance, and
//! always in the AOD).
//!
//! **The sun.** [`SunTable`] replaces the sun position of the coarse cells by given angles: the comparison with FORCE,
//! whose `sun-cl.c` is not ported (Zeit's sun is [`crate::solpos`], written in a clean room), uses it to separate the
//! effect of the sun from the others.

use crate::atc::{Atc, Dn, AZI, ZEN};
use crate::atmo::{BrdfFn, L2Params};
use crate::brdf;
use crate::gas::{SceneGrid, Toa, WvpSource};
use crate::l2::{radiometric_correction, tick, L2Error, L2Options, Level2};
use crate::qai;
use crate::radtran::{self, Date};
use crate::s2l1c::{self, L1c, NB};
use crate::sunview::{self, GeoTable, Mission, SceneGeo};
use crate::topo;

/// The bits of the QAI that the cloud detection sets and the correction reads from it: the cloud state (2 bits), the
/// shadow, the snow and the water flag.
pub const FLAG_MASK: i16 = 0b11_1110;

/// The map points whose geographic coordinates the correction asks for, as a grid of corners: the corner of the column
/// `j`, row `i` is `(ulx + j * res, uly - i * res)`, for `j < nx`, `i < ny`. The grid is square (the larger side of the
/// coarse grid on both): `atmo_angledep` indexes the coarse cells with the numbers of columns and rows swapped, as FORCE
/// does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoGrid {
    pub ulx: f64,
    pub uly: f64,
    pub res: f64,
    pub nx: usize,
    pub ny: usize,
}

/// The corners [`level2_scene`] needs the longitude and latitude of, for this scene.
pub fn geo_grid(l1: &L1c) -> Result<GeoGrid, L2Error> {
    let atc = Atc::allocate(&l1.dn(), &l1.meta)?;
    let n = atc.xy_sun.nx.max(atc.xy_sun.ny);
    Ok(GeoGrid { ulx: l1.tile.ulx, uly: l1.tile.uly, res: atc.xy_sun.res, nx: n, ny: n })
}

/// The sun zenith and azimuth (degrees, as `sunpos` returns them) of every cell of the coarse grid, row by row, to use in
/// place of the sun position of the core.
#[derive(Clone, Debug)]
pub struct SunTable {
    pub zen: Vec<f32>,
    pub azi: Vec<f32>,
}

/// What [`level2_scene`] needs besides the data: the parameters of the correction (`par_ll_t`), the no-data value of the
/// DEM, the buffer of the no-data pixels (`BUFFER_NODATA`) and the threads.
#[derive(Clone, Copy, Debug)]
pub struct SceneParams {
    pub l2: L2Params,
    pub dem_nodata: i32,
    pub buffer_nodata: bool,
    pub n_jobs: i32,
}

impl SceneParams {
    /// The defaults of FORCE's parameter file: atmospheric, topographic and BRDF correction, adjacency effect, multiple
    /// scattering and AOD estimation on, the clouds not erased, every optional product on (DST, VZN, HOT, AOD, WVP),
    /// `DEM_NODATA = -32767`.
    pub fn force_defaults() -> Self {
        SceneParams {
            l2: L2Params {
                domulti: true,
                doenv: true,
                doatmo: true,
                dotopo: true,
                dobrdf: true,
                erase_cloud: false,
                doaod: true,
                odst: true,
                ovzn: true,
                ohot: true,
                oaod: true,
                owvp: true,
            },
            dem_nodata: -32767,
            buffer_nodata: false,
            n_jobs: 1,
        }
    }
}

/// The result: the Level 2 products, the atmosphere (the coarse grids), the topography (binned DEM, slope, aspect, illumination
/// and sky view; `c` is the C-factor of the radiometric correction) and the QAI the radiometric correction started from (the
/// initial QAI with the sun, topography and cloud flags).
#[derive(Clone, Debug)]
pub struct SceneOut {
    pub level2: Level2,
    pub atc: Atc,
    pub top: topo::Top,
    pub qai_in: Vec<i16>,
}

/// The parameters of the BRDF model of a band of the sensor (by its domain), or `None` where there is no correction.
fn band_model(domain: &str) -> Option<brdf::RossLi> {
    brdf::role(match domain {
        "BLUE" => "blue",
        "GREEN" => "green",
        "RED" => "red",
        "REDEDGE1" => "rededge1",
        "REDEDGE2" => "rededge2",
        "REDEDGE3" => "rededge3",
        "BROADNIR" => "nir",
        "NIR" => "nir08",
        "SWIR1" => "swir1",
        "SWIR2" => "swir2",
        _ => return None, // ULTRABLUE, VAPOR, CIRRUS
    })
}

/// The BRDF factor of the coarse cell `g` for every band: the NBAR c-factor with the standard sun zenith of the latitude
/// of the cell (see the module notes). `out` starts as 1 and keeps it for the bands without a model and for a cell
/// whose angles the model does not cover.
pub fn brdf_fn<'a>(domains: &'a [String], geo: &'a SceneGeo<'a>, date: &'a Date) -> impl Fn(&Atc, usize, &mut [f32]) + Sync + 'a {
    move |atc: &Atc, g: usize, out: &mut [f32]| {
        let (nx, res) = (atc.xy_sun.nx, atc.xy_sun.res);
        let (lon, lat) = geo.lonlat((g % nx) as i32, (g / nx) as i32, res);
        let norm = radtran::standard_sunzenith(date, lat, lon);
        let deg = |v: f32| (v as f64).to_degrees();
        let (sza, saa) = (deg(atc.xy_sun.bands[ZEN][g]), deg(atc.xy_sun.bands[AZI][g]));
        let (vza, vaa) = (deg(atc.xy_view.bands[ZEN][g]), deg(atc.xy_view.bands[AZI][g]));
        for (b, o) in out.iter_mut().enumerate() {
            if let Some(model) = domains.get(b).and_then(|d| band_model(d)) {
                let c = brdf::cfactor(model, sza, saa, vza, vaa, norm);
                if c.is_finite() {
                    *o = c as f32;
                }
            }
        }
    }
}

/// `force-l2ps` for one Sentinel-2 L1C scene, from the digital numbers.
///
/// * `l1`: the metadata ([`s2l1c::parse_metadata`] with `subset` false, the image being the whole tile), whose size is
///   that of the bands.
/// * `dn`: the 13 bands (`l1.tile.ncols * l1.tile.nrows` values each) on the grid of the scene.
/// * `flags`: the cloud, shadow, snow and water bits of the QAI per pixel (only [`FLAG_MASK`] is used), or `None`.
/// * `dem`: the elevation in metres on the grid of the scene (`sp.dem_nodata` where there is none), or `None` for the
///   flat DEM FORCE uses without `FILE_DEM`.
/// * `geo`: the geographic coordinates of the corners [`geo_grid`] lists.
/// * `sun`: the sun angles of the coarse cells (degrees) to use instead of the core's.
/// * `brdf`: the BRDF factor of every band and coarse cell (`brdf[b][g]`) to use instead of [`brdf_fn`]'s (FORCE's own
///   factors, for the comparison with FORCE).
#[allow(clippy::too_many_arguments)]
pub fn level2_scene(
    l1: &L1c,
    dn: &[&[u16]],
    flags: Option<&[i16]>,
    dem: Option<&[f32]>,
    geo: &GeoTable,
    sun: Option<&SunTable>,
    brdf: Option<&[Vec<f32>]>,
    sp: &SceneParams,
) -> Result<SceneOut, L2Error> {
    let (nx, ny) = (l1.tile.ncols as usize, l1.tile.nrows as usize);
    let nc = nx * ny;
    let res = l1.tile.res;
    let other = |e: s2l1c::S2L1cError| L2Error::Other(e.to_string());
    if dn.len() != NB || dn.iter().any(|b| b.len() != nc) {
        return Err(L2Error::Other(format!("expected {NB} bands of {nx} x {ny} values")));
    }
    if flags.is_some_and(|f| f.len() != nc) || dem.is_some_and(|d| d.len() != nc) {
        return Err(L2Error::Other("the flags and the DEM must have as many values as a band".into()));
    }
    let mut t = std::time::Instant::now();
    let grid = SceneGrid { nx, ny, res };
    let dnb: Dn = l1.dn();
    let mut atc = Atc::allocate(&dnb, &l1.meta)?;
    let date = l1.product.date;
    let to_geo = |x: f64, y: f64| geo.lookup(x, y);
    let scene_geo = SceneGeo { ulx: l1.tile.ulx, uly: l1.tile.uly, date, to_geo: &to_geo };

    // initialize the QAI: no-data and saturation
    let mut qai_v = s2l1c::bounds_level1(dn, nx, ny, sp.buffer_nodata).map_err(other)?;
    tick(&mut t, "allocate, bounds");

    // sun-target-view geometry
    match sun {
        None => sunview::sun_target_view(&mut atc, &l1.meta, Mission::Sentinel2, &scene_geo, &grid, &mut qai_v)?,
        Some(t) => {
            if t.zen.len() != atc.xy_sun.ncell() || t.azi.len() != atc.xy_sun.ncell() {
                return Err(L2Error::Other(format!("the sun table needs {} cells", atc.xy_sun.ncell())));
            }
            let f = |g: usize, _: f32, _: f32, _: &Date| (t.zen[g], t.azi[g]);
            sunview::sun_target_view_with(&mut atc, &l1.meta, Mission::Sentinel2, &scene_geo, &grid, &mut qai_v, &f)?;
        }
    }

    tick(&mut t, "sun_target_view");

    if brdf.is_some_and(|t| t.len() != dnb.wavelength.len() || t.iter().any(|b| b.len() != atc.xy_sun.ncell())) {
        return Err(L2Error::Other(format!("the BRDF table needs {} bands of {} cells", dnb.wavelength.len(), atc.xy_sun.ncell())));
    }

    // TOA reflectance
    let radd: Vec<f32> = l1.meta.cal.iter().map(|c| c.radd).collect();
    let bands = s2l1c::convert_level1(dn, &qai_v, &radd, l1.product.dn_scale, sp.n_jobs).map_err(other)?;
    let toa = Toa { grid, bands, domain: dnb.domain.clone() };

    tick(&mut t, "convert_level1");

    // topography
    let mut top = topo::compile_topography(dem, sp.dem_nodata, nx, ny, res, &mut atc, &mut qai_v)?;

    tick(&mut t, "compile_topography");

    // cloud, shadow, snow and water flags
    if let Some(f) = flags {
        for (q, &v) in qai_v.iter_mut().zip(f) {
            if !qai::off(*q) {
                *q |= v & FLAG_MASK;
            }
        }
    }

    let qai_in = qai_v.clone();

    // radiometric correction
    let opts = L2Options { aod_dir: None, wvp: WvpSource { dir: None, wvp: 0.0, strict: false } };
    let own = brdf_fn(&dnb.domain, &scene_geo, &date);
    let given = |_: &Atc, g: usize, out: &mut [f32]| {
        if let Some(t) = brdf {
            for (b, o) in out.iter_mut().enumerate() {
                *o = t[b][g];
            }
        }
    };
    // FORCE's `brdf_factor` runs for every coarse cell whatever `DO_BRDF` says: the factor is in the water AOD retrieval too;
    // `dobrdf` only decides whether the surface reflectance is multiplied by it
    let bf: Option<BrdfFn> = if brdf.is_some() { Some(&given) } else { Some(&own) };
    let level2 = radiometric_correction(
        &sp.l2,
        &opts,
        &l1.meta,
        Mission::Sentinel2,
        &scene_geo,
        &mut atc,
        &toa,
        &mut qai_v,
        None,
        &mut top,
        bf,
        sp.n_jobs,
    )?;
    tick(&mut t, "radiometric_correction");
    Ok(SceneOut { level2, atc, top, qai_in })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::{Cal, Meta, S2Meta};
    use crate::s2l1c::{ProductMeta, TileMeta};

    /// A 60 x 60 pixel scene of 10 m at 15 S (UTM 23S-like corner), baseline 05.09 offsets.
    fn l1(n: usize) -> L1c {
        let rsr_start = 43; // the last block of 13 response bands of the table (56 bands)
        let mut date = Date::from_year_doy(2023, 209);
        (date.hh, date.mm, date.ss) = (13, 22, 39);
        let product = ProductMeta {
            sensor: "SEN2B".into(),
            rsr_start,
            baseline: Some("05.09".into()),
            dn_scale: 10000.0,
            radd: [-1000.0; NB],
            date,
        };
        let cells = ((n as f64 * 10.0) / 5000.0).ceil() as usize;
        let tile = TileMeta {
            res: 10.0,
            tile_ncols: n as i32,
            tile_nrows: n as i32,
            tile_ulx: 199980.0,
            tile_uly: 8300020.0,
            ncols: n as i32,
            nrows: n as i32,
            ulx: 199980.0,
            uly: 8300020.0,
            left: 0,
            top: 0,
            right: n as i32,
            bottom: n as i32,
            s2: S2Meta { nodata: -9999.0, vzen: vec![5.0; cells * cells], vazi: vec![100.0; cells * cells], nx: cells, ny: cells },
            szen: vec![30.0; cells * cells],
            sazi: vec![40.0; cells * cells],
            epsg: Some(32723),
            tile_id: None,
        };
        let meta = Meta {
            cal: (0..NB).map(|b| Cal { rsr_band: rsr_start + b, radd: -1000.0 }).collect(),
            s2: tile.s2.clone(),
        };
        L1c { product, tile, meta, refsys_id: "T23LKC".into() }
    }

    /// The geography of the corner of the scene as an affine map: about 47 W, 16 S.
    fn table(l: &L1c) -> GeoTable {
        let g = geo_grid(l).unwrap();
        let mut t = GeoTable { ulx: g.ulx, uly: g.uly, res: g.res, nx: g.nx, ny: g.ny, lon: vec![], lat: vec![] };
        for i in 0..g.ny {
            for j in 0..g.nx {
                t.lon.push(-47.8 + (j as f64 * g.res) / 107000.0);
                t.lat.push(-15.36 - (i as f64 * g.res) / 111000.0);
            }
        }
        t
    }

    /// Digital numbers of a vegetated surface under a clear atmosphere: a gradient with a bright patch.
    fn dn(n: usize) -> Vec<Vec<u16>> {
        (0..NB)
            .map(|b| {
                (0..n * n)
                    .map(|p| {
                        let (i, j) = (p / n, p % n);
                        let base = 1200 + 130 * b + (i * 3 + j * 2) % 400;
                        (if (20..30).contains(&i) && (20..30).contains(&j) { base + 2500 } else { base }) as u16
                    })
                    .collect()
            })
            .collect()
    }

    fn dem(n: usize) -> Vec<f32> {
        (0..n * n).map(|p| 800.0 + ((p / n) as f32) * 2.0 + ((p % n) as f32).sin() * 3.0).collect()
    }

    fn run(n: usize, brdf: bool, jobs: i32, sun: Option<&SunTable>) -> SceneOut {
        let l = l1(n);
        let d = dn(n);
        let refs: Vec<&[u16]> = d.iter().map(|b| b.as_slice()).collect();
        let e = dem(n);
        let mut sp = SceneParams::force_defaults();
        sp.l2.dobrdf = brdf;
        sp.n_jobs = jobs;
        level2_scene(&l, &refs, None, Some(&e), &table(&l), sun, None, &sp).unwrap()
    }

    #[test]
    fn the_geo_grid_is_the_square_of_the_coarse_grid_corners() {
        let l = l1(1200);
        let g = geo_grid(&l).unwrap();
        assert_eq!((g.nx, g.ny, g.res, g.ulx, g.uly), (3, 3, 5000.0, 199980.0, 8300020.0));
    }

    #[test]
    fn a_scene_gives_every_product_and_the_threads_do_not_change_it() {
        let a = run(240, false, 1, None);
        let b = run(240, false, 4, None);
        assert_eq!(a.level2.boa.bands, b.level2.boa.bands);
        assert_eq!(a.level2.qai, b.level2.qai);
        let l2 = &a.level2;
        assert_eq!(l2.boa.bands.len(), 10);
        for p in [&l2.dst, &l2.vzn, &l2.hot, &l2.aod, &l2.wvp] {
            assert_eq!(p.as_ref().unwrap().len(), 240 * 240);
        }
        // a clear scene: reflectance of a few percent to a few tens, view zenith 5 degrees, a finite water vapour
        let red = &l2.boa.bands[2];
        let ok = red.iter().filter(|&&v| (100..8000).contains(&v)).count();
        assert!(ok > red.len() * 9 / 10, "{ok} of {}", red.len());
        assert!(l2.vzn.as_ref().unwrap().iter().all(|&v| (490..=510).contains(&v)));
        // the water vapour of a Sentinel-2 scene is the product (the scene's `atc.wvp` stays 0)
        let wvp = l2.wvp.as_ref().unwrap();
        let valid: Vec<i16> = wvp.iter().copied().filter(|&v| v != crate::l2::NODATA).collect();
        assert!(valid.len() > wvp.len() * 9 / 10 && valid.iter().all(|&v| (0..15000).contains(&v)), "{} valid", valid.len());
    }

    #[test]
    fn the_brdf_factor_changes_the_reflectance_a_little_and_the_sun_table_is_used() {
        let off = run(120, false, 1, None);
        let on = run(120, true, 1, None);
        let (a, b) = (&off.level2.boa.bands[3], &on.level2.boa.bands[3]);
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        assert!(diff > a.len() / 2, "the BRDF factor leaves {diff} of {} pixels equal", a.len());
        let rel = a.iter().zip(b).map(|(&x, &y)| ((x as f64 - y as f64) / x as f64).abs()).fold(0.0, f64::max);
        assert!(rel < 0.3, "relative change {rel}");
        // the sun table: the sun of the core given back changes nothing, another sun changes the result
        let n = off.atc.xy_sun.ncell();
        let zen: Vec<f32> = off.atc.xy_sun.bands[ZEN].iter().map(|&r| (r as f64).to_degrees() as f32).collect();
        let azi: Vec<f32> = off.atc.xy_sun.bands[AZI].iter().map(|&r| (r as f64).to_degrees() as f32).collect();
        assert_eq!(zen.len(), n);
        let same = run(120, false, 1, Some(&SunTable { zen: zen.clone(), azi: azi.clone() }));
        let close = same.level2.boa.bands[3].iter().zip(a).filter(|(x, y)| (**x as i32 - **y as i32).abs() > 1).count();
        assert_eq!(close, 0);
        let higher = run(120, false, 1, Some(&SunTable { zen: zen.iter().map(|z| z + 10.0).collect(), azi }));
        assert_ne!(higher.level2.boa.bands[3], *a);
    }

    #[test]
    fn the_flags_of_the_cloud_detection_reach_the_qai_but_not_the_off_pixels() {
        let n = 120;
        let l = l1(n);
        let mut d = dn(n);
        d[0][5] = 0; // pixel 5 is off
        let refs: Vec<&[u16]> = d.iter().map(|b| b.as_slice()).collect();
        let e = dem(n);
        let mut flags = vec![0i16; n * n];
        flags[5] = 0b11_1110;
        flags[6] = 1 << 3;
        flags[7] = 1 << 4 | 1 << 10; // snow; the other bits are not read
        let sp = SceneParams::force_defaults();
        let out = level2_scene(&l, &refs, Some(&flags), Some(&e), &table(&l), None, None, &sp).unwrap();
        let q = &out.level2.qai;
        assert_eq!(q[5], 1);
        assert!(qai::shadow(q[6]) && !qai::snow(q[6]));
        assert!(qai::snow(q[7]) && !qai::lowsun(q[7]));
        assert!(!qai::shadow(q[8]));
    }

    #[test]
    fn the_geo_table_answers_for_its_corners_only() {
        let l = l1(1200);
        let t = table(&l);
        let g = geo_grid(&l).unwrap();
        let (lon, lat) = t.lookup(g.ulx + 5000.0, g.uly - 10000.0);
        assert_eq!((lon, lat), (t.lon[2 * 3 + 1], t.lat[2 * 3 + 1]));
        assert!(t.lookup(g.ulx + 5001.0, g.uly).0.is_nan());
        assert!(t.lookup(g.ulx + 15000.0, g.uly).0.is_nan());
    }
}
