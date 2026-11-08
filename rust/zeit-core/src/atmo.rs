//! The atmosphere of every coarse grid cell, before the surface reflectance, and the interpolation of the coarse grid to
//! the pixels. A port of FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force), `src/modules/lower-level/atmo-ll.c`: `interpolation_weights` and
//! `interpolate_coarse` (the bilinear interpolation of a coarse grid to a pixel, that skips the cells without data),
//! `atmo_angledep` (what depends on the sun and view angles: the BRDF factor, the backscattering angle and the phase
//! functions, the gas transmittances, the average elevation of the cell, the molecular optical depth at that elevation,
//! the Fresnel reflection), `atmo_elevdep` (what depends on the elevation, for each of the 256 elevation bins: the
//! optical depths, the scattering transmittances, the path reflectance, the spherical albedo, the environmental weight;
//! and the high aerosol flag of the QAI) and `apply_aoi` (the area of interest). The radiative transfer formulas are in
//! [`crate::radtran`], the geometry in [`crate::sunview`], the structure they fill in [`crate::atc`].
//!
//! `surface_reflectance`, `background_reflectance`, `compile_l2_*` and the orchestration of the same file are in
//! [`crate::l2`].
//!
//! The arithmetic is the C's, as in [`crate::radtran`]: C `float` is `f32`, rounded where it is assigned; where C
//! promotes to `double` (`floor`, `0.5` literals, the elevation of the first bin) the value is computed in `f64`. Where
//! this differs from the C:
//!
//! * **The BRDF factor.** `atmo_angledep` calls `brdf_factor` of `brdf-ll.c`, which is not ported (it reads the BRDF
//!   coefficients of FORCE's own correction): the caller gives a function that fills the factor of one cell for every
//!   band (`brdf`, see [`BrdfFn`]), and with none the factor is 1 for every band, which is what `brdf_factor` starts from
//!   and leaves where it does not correct. Zeit's NBAR, written from Roy's papers ([`crate::brdf`]), is the replacement.
//! * **The loop order of `atmo_angledep` and `atmo_elevdep`.** FORCE names the number of columns of the coarse grid `ne`
//!   and the rows `nf` in both (swapped, as the names go in the rest of the file), loops `e < ne`, `f < nf` and takes the
//!   cell `g = e * nf + f`. Every cell is visited once, but for a grid with more columns than rows the cell `g` is not
//!   the one at the row `e` and column `f`: `atmo_angledep` takes the geographic coordinates (the ozone) of the cell
//!   `(column f, row e)` for it. This port does the same, so the ozone of the cells of a grid that is not square is
//!   FORCE's, whatever it is worth (for a square grid, and for everything but the ozone, no difference).
//! * **Threads.** FORCE runs the cell loops with OpenMP. Every cell writes only its own values and reads data that no
//!   thread writes in the same loop, so the result does not depend on the number of threads; the loops here are rayon
//!   loops over the cells (`n_jobs` as everywhere in zeit).
//! * **`apply_aoi`.** The C frees the AOI brick; here the caller keeps the slice. `CANCEL` is [`Aoi::Cancel`].
//! * **Errors.** Where the C prints a message and returns `FAILURE`, this returns an [`AtmoError`] (the sizes of the
//!   arrays, and a sensor without a `GREEN` band).
//! * **Omitted.** The `FORCE_DEBUG` and `FORCE_CLOCK` blocks (they only print or write files).

use crate::atc::{Atc, AtmoError, AZI, C_ZEN, N_ELEV, ZEN};
use crate::gas::{fequal, ji2p, SceneGrid};
use crate::meta::Meta;
use crate::parallel::{pool, threads};
use crate::qai;
use crate::radtran::{
    aod_elev_factor, aod_elev_scale, backscatter, env_weight, fresnel_reflection, gas_transmitt, mod_elev_factor,
    mod_elev_scale, optical_depth, ozone_amount, ozone_transmitt, path_ref, phase_aerosol, phase_molecular, scatt_transmitt,
    sphere_albedo, wvp_transmitt,
};
use crate::sunview::SceneGeo;
use crate::topo::average_elevation_value;
use rayon::prelude::*;

/// The parameters of the level 2 processing that the atmospheric functions read (`par_ll_t`): the multiple scattering
/// approximation of the path reflectance (`domulti`) and the environmental (adjacency) weighting (`doenv`); and, for
/// the surface reflectance and the products of [`crate::l2`], the atmospheric correction (`doatmo`), the topographic
/// correction (`dotopo`), the BRDF factor (`dobrdf`), the erasing of the clouds the cloud flag of the QAI marks as
/// unsure (`erase_cloud`), the AOD estimation (`doaod`) and which optional products are made (`odst`, `ovzn`, `ohot`,
/// `oaod`, `owvp`; the overview `oovv` is not ported). `Default` is all off.
#[derive(Clone, Copy, Debug, Default)]
pub struct L2Params {
    pub domulti: bool,
    pub doenv: bool,
    pub doatmo: bool,
    pub dotopo: bool,
    pub dobrdf: bool,
    pub erase_cloud: bool,
    pub doaod: bool,
    pub odst: bool,
    pub ovzn: bool,
    pub ohot: bool,
    pub oaod: bool,
    pub owvp: bool,
}

/// The weights of the interpolation of a coarse grid at a pixel (`iweights_t`): the four cells around the pixel (upper
/// left, upper right, lower left, lower right) and their weights, in each row (`wul`, `wur`; `wll`, `wlr`) and between
/// the rows (`wtop`, `wdown`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IWeights {
    pub gul: i32,
    pub gur: i32,
    pub gll: i32,
    pub glr: i32,
    pub wul: f32,
    pub wur: f32,
    pub wll: f32,
    pub wlr: f32,
    pub wtop: f32,
    pub wdown: f32,
}

/// `interpolation_weights`: the weights to interpolate the coarse atmospheric parameters `coarse` (a grid of `nf`
/// columns and `ne` rows of `res` metres, `nodata` where a cell has no value) at the pixel in column `j`, row `i` of an
/// image of `full_res` metres. The cells whose value is `nodata` get no weight (the others are renormalized, as far as
/// the grid goes: at the edge the cell itself is used twice).
pub fn interpolation_weights(j: i32, i: i32, nf: i32, ne: i32, res: f64, full_res: f64, coarse: &[f32], nodata: f32) -> IWeights {
    let mut w = IWeights::default();
    let (mut vtop, mut vdown) = (true, true);

    let e_ = (i as f64 * full_res / res) as f32;
    let e = (e_ as f64).floor() as i32;
    let f_ = (j as f64 * full_res / res) as f32;
    let f = (f_ as f64).floor() as i32;

    let (mut gleft, mut gright) = if ((f_ - f as f32) as f64) < 0.5 { (f - 1, f) } else { (f, f + 1) };
    let (mut gtop, mut gdown) = if ((e_ - e as f32) as f64) < 0.5 { (e - 1, e) } else { (e, e + 1) };

    if gleft < 0 {
        gleft = f;
    }
    if gtop < 0 {
        gtop = e;
    }
    if gright >= nf {
        gright = f;
    }
    if gdown >= ne {
        gdown = e;
    }

    let cleft = (gleft as f64 + 0.5) as f32;
    let cright = (gright as f64 + 0.5) as f32;
    let ctop = (gtop as f64 + 0.5) as f32;
    let cdown = (gdown as f64 + 0.5) as f32;

    w.gul = gtop * nf + gleft;
    w.gur = gtop * nf + gright;
    w.gll = gdown * nf + gleft;
    w.glr = gdown * nf + gright;

    let vul = !fequal(coarse[w.gul as usize], nodata);
    let vur = !fequal(coarse[w.gur as usize], nodata);
    let vll = !fequal(coarse[w.gll as usize], nodata);
    let vlr = !fequal(coarse[w.glr as usize], nodata);

    if gleft == gright {
        w.wul = 0.5;
        w.wur = 0.5;
    } else if vul && vur {
        w.wul = (cright - f_) / (cright - cleft);
        w.wur = (f_ - cleft) / (cright - cleft);
    } else if vul && !vur {
        w.wul = 1.0;
        w.wur = 0.0;
    } else if !vul && vur {
        w.wul = 0.0;
        w.wur = 1.0;
    } else {
        w.wul = 0.0;
        w.wur = 0.0;
        vtop = false;
    }

    if gleft == gright {
        w.wll = 0.5;
        w.wlr = 0.5;
    } else if vll && vlr {
        w.wll = (cright - f_) / (cright - cleft);
        w.wlr = (f_ - cleft) / (cright - cleft);
    } else if vll && !vlr {
        w.wll = 1.0;
        w.wlr = 0.0;
    } else if !vll && vlr {
        w.wll = 0.0;
        w.wlr = 1.0;
    } else {
        w.wll = 0.0;
        w.wlr = 0.0;
        vdown = false;
    }

    if gtop == gdown {
        w.wtop = 0.5;
        w.wdown = 0.5;
    } else if vtop && vdown {
        w.wtop = (cdown - e_) / (cdown - ctop);
        w.wdown = (e_ - ctop) / (cdown - ctop);
    } else if vtop {
        w.wtop = 1.0;
        w.wdown = 0.0;
    } else if vdown {
        w.wtop = 0.0;
        w.wdown = 1.0;
    } else {
        w.wtop = 0.0;
        w.wdown = 0.0;
    }
    w
}

/// `interpolate_coarse`: the value of the coarse grid `coarse` at the pixel of the weights `w`.
pub fn interpolate_coarse(w: &IWeights, coarse: &[f32]) -> f32 {
    let ul = coarse[w.gul as usize];
    let ur = coarse[w.gur as usize];
    let ll = coarse[w.gll as usize];
    let lr = coarse[w.glr as usize];
    w.wtop * (w.wul * ul + w.wur * ur) + w.wdown * (w.wll * ll + w.wlr * lr)
}

/// The BRDF correction factor of one cell (`brdf_factor` of `brdf-ll.c`, not ported): given the atmospheric structure
/// (the sun and view geometry of the cell are in `atc.xy_sun` and `atc.xy_view`), the cell `g` and a slice of one factor
/// per band that starts as 1, it writes the factors.
pub type BrdfFn<'a> = &'a (dyn Fn(&Atc, usize, &mut [f32]) + Sync);

/// A map over `n` items in order, on the pool of `n_jobs` threads (see [`crate::parallel`]).
fn map_cells<T: Send>(n: usize, n_jobs: i32, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    let t = threads(n_jobs);
    if t <= 1 {
        (0..n).map(f).collect()
    } else {
        pool(t).install(|| (0..n).into_par_iter().map(f).collect())
    }
}

/// What `atmo_angledep` gives for one coarse cell.
struct AngleCell {
    brdf: Vec<f32>,
    psi: f32,
    pr: f32,
    pa: f32,
    tsw: Vec<f32>,
    tvw: Vec<f32>,
    tso: Vec<f32>,
    tvo: Vec<f32>,
    tg: Vec<f32>,
    dem: f32,
    hr: f32,
    modv: Vec<f32>,
    fresnel: f32,
}

/// `atmo_angledep`: all the atmospheric parameters that are a function of the view and sun angles, for the cells of the
/// coarse grid that have a view angle (`atc.xy_view` band 0 not no-data). Needs `atc` after [`crate::sunview::sun_target_view`]
/// (the angles), with `wvp` (the water vapour) and `dem` (the elevation statistics) and the molecular optical depth
/// `mod_` of the scene set, `top_dem` the binned DEM of the image (`TOP->dem`) and `qai` on `grid`. `geo` gives the
/// geographic coordinates of the cells and the date (the ozone). `brdf` is the BRDF factor, see [`BrdfFn`]; with `None`
/// it is 1. Fills `xy_brdf`, `xy_psi`, `xy_Pr`, `xy_Pa`, `xy_Tsw`, `xy_Tvw`, `xy_Tso`, `xy_Tvo`, `xy_Tg`, `xy_dem`,
/// `xy_Hr`, `xy_mod` and `xy_fresnel`.
#[allow(clippy::too_many_arguments)]
pub fn atmo_angledep(
    atc: &mut Atc,
    meta: &Meta,
    geo: &SceneGeo,
    grid: &SceneGrid,
    qai: &[i16],
    top_dem: &[u8],
    brdf: Option<BrdfFn>,
    n_jobs: i32,
) -> Result<(), AtmoError> {
    // FORCE's names: ne the columns, nf the rows (see the module notes)
    let ne = atc.xy_mod.nx;
    let nf = atc.xy_mod.ny;
    let nb = atc.xy_mod.nb();
    let npix = grid.nx * grid.ny;
    if qai.len() != npix || top_dem.len() != npix {
        return Err(AtmoError::Shape(format!("the QAI has {} values and the DEM {} for a grid of {npix} pixels", qai.len(), top_dem.len())));
    }
    if meta.cal.len() < nb || atc.mod_.len() < nb || atc.xy_view.ncell() != ne * nf || nf == 0 {
        return Err(AtmoError::Shape("the bands of the metadata, the scene MOD and the grids must agree".into()));
    }

    let cells = {
        let atc: &Atc = atc;
        map_cells(ne * nf, n_jobs, |g| -> Option<AngleCell> {
            if atc.xy_view.is_nodata(0, g) {
                return None;
            }
            let (e, f) = (g / nf, g % nf);

            // bi-directional correction parameter
            let mut brdfv = vec![1.0f32; nb];
            if let Some(bf) = brdf {
                bf(atc, g, &mut brdfv);
            }

            // relative air mass
            let ms = atc.xy_sun.bands[C_ZEN][g];
            let mv = atc.xy_view.bands[C_ZEN][g];

            // cosine of backscattering angle
            let psi = backscatter(ms, mv, atc.xy_sun.bands[AZI][g], atc.xy_view.bands[AZI][g]);

            // phase functions for molecular and aerosol scattering
            let pr = phase_molecular(psi);
            let pa = phase_aerosol(psi, &atc.tthg.hg);

            // ozone amount (very rough approximation)
            let (lon, lat) = geo.lonlat(f as i32, e as i32, atc.xy_sun.res);
            let ozone = ozone_amount(lon as f32, lat as f32, geo.date.doy);

            let (mut tsw, mut tvw) = (vec![0.0f32; nb], vec![0.0f32; nb]);
            let (mut tso, mut tvo, mut tg) = (vec![0.0f32; nb], vec![0.0f32; nb], vec![0.0f32; nb]);
            for b in 0..nb {
                // gaseous transmittance
                let rsr = meta.cal[b].rsr_band;
                tsw[b] = wvp_transmitt(atc.wvp, ms, rsr);
                tvw[b] = wvp_transmitt(atc.wvp, mv, rsr);
                tso[b] = ozone_transmitt(ozone, ms, rsr);
                tvo[b] = ozone_transmitt(ozone, mv, rsr);
                tg[b] = gas_transmitt(tsw[b], tvw[b], tso[b], tvo[b]);
            }

            // average elevation
            let dem = average_elevation_value(g, atc.xy_dem.nx, atc.xy_dem.res, top_dem, grid.nx, grid.ny, grid.res, qai);
            let dem_bin = dem as i32;

            // correct rayleigh for elevation
            let hr = mod_elev_factor(dem_bin as f32 * atc.dem.step + atc.dem.min + atc.dem.step / 2.0);
            let modv = (0..nb).map(|b| mod_elev_scale(atc.mod_[b], 1.0, hr)).collect();

            // Fresnel reflection for AOD from water retrieval
            // --> assumed that the water body is flat -> incidence angle = szen
            let fresnel = fresnel_reflection(atc.xy_sun.bands[ZEN][g]);

            Some(AngleCell { brdf: brdfv, psi, pr, pa, tsw, tvw, tso, tvo, tg, dem, hr, modv, fresnel })
        })
    };

    for (g, c) in cells.into_iter().enumerate() {
        let Some(c) = c else { continue };
        for b in 0..nb {
            atc.xy_brdf.bands[b][g] = c.brdf[b];
            atc.xy_tsw.bands[b][g] = c.tsw[b];
            atc.xy_tvw.bands[b][g] = c.tvw[b];
            atc.xy_tso.bands[b][g] = c.tso[b];
            atc.xy_tvo.bands[b][g] = c.tvo[b];
            atc.xy_tg.bands[b][g] = c.tg[b];
            atc.xy_mod.bands[b][g] = c.modv[b];
        }
        atc.xy_psi.bands[0][g] = c.psi;
        atc.xy_pr.bands[0][g] = c.pr;
        atc.xy_pa.bands[0][g] = c.pa;
        atc.xy_dem.bands[0][g] = c.dem;
        atc.xy_hr.bands[0][g] = c.hr;
        atc.xy_fresnel.bands[0][g] = c.fresnel;
    }
    Ok(())
}

/// What `atmo_elevdep` gives for one coarse cell: per elevation bin the values of the families below, per band.
struct ElevCell {
    /// `[family][z][b]` flattened: mod, aod, od, T, Ts, Tv, tsd, tvd, tss, tvs, rho_p, s, F.
    fam: Vec<f32>,
}

const F_MOD: usize = 0;
const F_AOD: usize = 1;
const F_OD: usize = 2;
const F_T: usize = 3;
const F_TS: usize = 4;
const F_TV: usize = 5;
const F_TSD: usize = 6;
const F_TVD: usize = 7;
const F_TSS: usize = 8;
const F_TVS: usize = 9;
const F_RHO: usize = 10;
const F_S: usize = 11;
const F_F: usize = 12;
const N_FAM: usize = 13;

/// How many cells `atmo_elevdep` holds before it writes them into the structure (the memory of a cell is
/// `13 * 256 * nb` floats).
const ELEV_CHUNK: usize = 64;

/// `atmo_elevdep`: all the atmospheric parameters that are a function of the terrain elevation, for each of the 256
/// elevation bins from `atc.dem.min` (`step` metres a bin) and for the cells of the coarse grid that have a view angle:
/// the elevation factors, the optical depths, the scattering transmittances, the path reflectance, the spherical albedo
/// and the environmental weight. Then the aerosol flag of the QAI (`qai` on `grid`): 3 where there is no AOD map (the
/// fallback is used), else 2 where the AOD of the green band at the elevation of the pixel (`top_dem` is the binned DEM,
/// `TOP->dem`) is above 0.6, else 1 where the AOD of the cell was interpolated; the pixels that are off are left alone.
/// Needs `atc` after [`atmo_angledep`] (the phase functions, `xy_mod`) and the AOD (`aod`, `xy_aod`, `aodmap`, `ha`, `hp`,
/// `xy_interp`), with the domain `GREEN` in the bands of `xy_mod`.
pub fn atmo_elevdep(
    atc: &mut Atc,
    l2: &L2Params,
    grid: &SceneGrid,
    qai: &mut [i16],
    top_dem: &[u8],
    n_jobs: i32,
) -> Result<(), AtmoError> {
    let ne = atc.xy_mod.nx;
    let nf = atc.xy_mod.ny;
    let nb = atc.xy_mod.nb();
    let nc = grid.nx * grid.ny;
    let b_green = atc.xy_mod.find_domain("GREEN").ok_or(AtmoError::Domain("GREEN"))?;
    if qai.len() != nc || top_dem.len() != nc {
        return Err(AtmoError::Shape(format!("the QAI has {} values and the DEM {} for a grid of {nc} pixels", qai.len(), top_dem.len())));
    }
    if atc.mod_.len() < nb || atc.aod.len() < nb || atc.xy_view.ncell() != ne * nf || nf == 0 {
        return Err(AtmoError::Shape("the scene MOD and AOD and the grids must agree".into()));
    }

    // the elevation of every bin in km, as the C accumulates it: z0 is a double sum of a float and a double
    let z0 = (atc.dem.min as f64 + atc.dem.step as f64 / 2.0) as f32;
    let mut zmed = z0;
    let (mut hr_z, mut ha_z) = (vec![0.0f32; N_ELEV], vec![0.0f32; N_ELEV]);
    for z in 0..N_ELEV {
        // correct mod and aod for elevation
        hr_z[z] = mod_elev_factor(zmed);
        ha_z[z] = aod_elev_factor(zmed, atc.hp);
        zmed += atc.dem.step;
    }

    let ncell = ne * nf;
    for start in (0..ncell).step_by(ELEV_CHUNK) {
        let len = ELEV_CHUNK.min(ncell - start);
        let cells = {
            let atc: &Atc = atc;
            map_cells(len, n_jobs, |k| -> Option<ElevCell> {
                let g = start + k;
                if atc.xy_view.is_nodata(0, g) {
                    return None;
                }
                // phase functions
                let pr = atc.xy_pr.bands[0][g];
                let pa = atc.xy_pa.bands[0][g];
                // relative air mass
                let ms = atc.xy_sun.bands[C_ZEN][g];
                let mv = atc.xy_view.bands[C_ZEN][g];

                let mut fam = vec![0.0f32; N_FAM * N_ELEV * nb];
                // for every possible elevation (100m steps)
                for z in 0..N_ELEV {
                    let (hr, ha) = (hr_z[z], ha_z[z]);
                    // down/up-welling scattering transmittances
                    for b in 0..nb {
                        let at = |fam_k: usize| (fam_k * N_ELEV + z) * nb + b;
                        // correct mod and aod for elevation
                        let mod_ = mod_elev_scale(atc.mod_[b], 1.0, hr);
                        let aod = if atc.aodmap {
                            aod_elev_scale(atc.xy_aod.bands[b][g], atc.ha, ha)
                        } else {
                            aod_elev_scale(atc.aod[b], atc.ha, ha)
                        };
                        // optical depth
                        let od = optical_depth(aod, mod_);
                        // scattering transmittance
                        let t = scatt_transmitt(aod, mod_, od, ms, mv);
                        // path reflectance
                        let rho_p = path_ref(l2.domulti, atc.tthg.sob, aod, mod_, od, pa, pr, t.tsd, t.tvd, ms, mv);
                        // environmental weighting function
                        let f = if l2.doenv { env_weight(aod, mod_, atc.fa, atc.fr) } else { 1.0 };

                        fam[at(F_MOD)] = mod_;
                        fam[at(F_AOD)] = aod;
                        fam[at(F_OD)] = od;
                        fam[at(F_T)] = t.total;
                        fam[at(F_TS)] = t.ts;
                        fam[at(F_TV)] = t.tv;
                        fam[at(F_TSD)] = t.tsd;
                        fam[at(F_TVD)] = t.tvd;
                        fam[at(F_TSS)] = t.tss;
                        fam[at(F_TVS)] = t.tvs;
                        fam[at(F_RHO)] = rho_p;
                        fam[at(F_S)] = sphere_albedo(aod, mod_, od);
                        fam[at(F_F)] = f;
                    }
                }
                Some(ElevCell { fam })
            })
        };
        for (k, c) in cells.into_iter().enumerate() {
            let Some(c) = c else { continue };
            let g = start + k;
            for z in 0..N_ELEV {
                atc.xyz_hr[z].bands[0][g] = hr_z[z];
                atc.xyz_ha[z].bands[0][g] = ha_z[z];
                for b in 0..nb {
                    let at = |fam_k: usize| c.fam[(fam_k * N_ELEV + z) * nb + b];
                    atc.xyz_mod[z].bands[b][g] = at(F_MOD);
                    atc.xyz_aod[z].bands[b][g] = at(F_AOD);
                    atc.xyz_od[z].bands[b][g] = at(F_OD);
                    atc.xyz_t[z].bands[b][g] = at(F_T);
                    atc.xyz_ts[z].bands[b][g] = at(F_TS);
                    atc.xyz_tv[z].bands[b][g] = at(F_TV);
                    atc.xyz_tsd[z].bands[b][g] = at(F_TSD);
                    atc.xyz_tvd[z].bands[b][g] = at(F_TVD);
                    atc.xyz_tss[z].bands[b][g] = at(F_TSS);
                    atc.xyz_tvs[z].bands[b][g] = at(F_TVS);
                    atc.xyz_rho_p[z].bands[b][g] = at(F_RHO);
                    atc.xyz_s[z].bands[b][g] = at(F_S);
                    atc.xyz_f[z].bands[b][g] = at(F_F);
                }
            }
        }
    }

    // high aerosol flag
    let (gnx, gres) = (atc.xy_sun.nx, atc.xy_sun.res);
    for p in 0..nc {
        if qai::off(qai[p]) {
            continue;
        }
        let g = ji2p(grid.res, gres, gnx, p / grid.nx, p % grid.nx);
        let z = top_dem[p] as usize;
        if !atc.aodmap {
            qai::set_aerosol(&mut qai[p], 3);
        } else if atc.xyz_aod[z].bands[b_green][g] as f64 > 0.6 {
            qai::set_aerosol(&mut qai[p], 2);
        } else if atc.xy_interp.bands[0][g] != 0.0 {
            qai::set_aerosol(&mut qai[p], 1);
        }
    }
    Ok(())
}

/// What `apply_aoi` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aoi {
    /// There is no area of interest (`CANCEL`): nothing was changed.
    Cancel,
    /// The pixels outside the area were set off (`SUCCESS`).
    Success,
}

/// `apply_aoi`: sets off the pixels of the QAI where the area of interest `aoi` (one byte per pixel, 0 outside) is 0.
pub fn apply_aoi(qai: &mut [i16], aoi: Option<&[u8]>) -> Result<Aoi, AtmoError> {
    let Some(aoi) = aoi else { return Ok(Aoi::Cancel) };
    if aoi.len() != qai.len() {
        return Err(AtmoError::Shape(format!("the area of interest has {} values for {} pixels", aoi.len(), qai.len())));
    }
    for (q, &a) in qai.iter_mut().zip(aoi) {
        if a == 0 {
            qai::set_off(q, true);
        }
    }
    Ok(Aoi::Success)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atc::{Dn, DemStats, S_ZEN};
    use crate::radtran::Date;
    use crate::sunview::{sun_target_view, Mission};

    fn warp(x: f64, y: f64) -> (f64, f64) {
        (10.0 + (x - 500000.0) * 9.0e-6, y * 9.0e-6)
    }

    /// A Sentinel-2 like scene of `nx * ny` pixels of `res` metres with 4 bands, everything `atmo_angledep` and
    /// `atmo_elevdep` need set.
    struct Scene {
        atc: Atc,
        meta: Meta,
        grid: SceneGrid,
        qai: Vec<i16>,
        dem: Vec<u8>,
    }

    fn scene(nx: usize, ny: usize, res: f64, aodmap: bool) -> Scene {
        let domain: Vec<String> = ["BLUE", "GREEN", "RED", "NIR"].iter().map(|s| s.to_string()).collect();
        let dn = Dn { nx, ny, res, wavelength: vec![0.49, 0.56, 0.665, 0.842], domain };
        let mut meta = Meta::from_rsr_bands(&[1, 2, 3, 7]);
        let mut atc = Atc::allocate(&dn, &meta).unwrap();
        let (gnx, gny) = (atc.xy_sun.nx, atc.xy_sun.ny);
        meta.s2 = crate::meta::S2Meta {
            nodata: -9999.0,
            vzen: (0..gnx * gny).map(|k| 3.0 + 0.4 * k as f32).collect(),
            vazi: (0..gnx * gny).map(|k| 100.0 + 2.5 * (k % 7) as f32).collect(),
            nx: gnx,
            ny: gny,
        };
        meta.s2.vzen[3] = -9999.0;
        let grid = SceneGrid { nx, ny, res };
        let mut qai = vec![0i16; nx * ny];
        let geo = SceneGeo { ulx: 500000.0, uly: 5400000.0, date: Date { hh: 10, mm: 30, ..Date::from_year_doy(2021, 172) }, to_geo: &warp };
        sun_target_view(&mut atc, &meta, Mission::Sentinel2, &geo, &grid, &mut qai).unwrap();
        let dem: Vec<u8> = (0..nx * ny).map(|p| ((p % nx) / 4 + (p / nx) / 8) as u8 % 40).collect();
        atc.dem = DemStats { avg: 0.4, min: 0.2, max: 4.0, step: 0.1, cnum: 255 };
        atc.wvp = 2.5;
        atc.hp = 2.0;
        atc.ha = 0.8;
        atc.aodmap = aodmap;
        atc.mod_ = atc.wvl.iter().map(|&w| crate::radtran::molecular_optical_depth(w)).collect();
        atc.aod = vec![0.2, 0.15, 0.1, 0.05];
        for b in 0..4 {
            for g in 0..gnx * gny {
                atc.xy_aod.bands[b][g] = 0.1 + 0.02 * g as f32 + 0.01 * b as f32;
            }
        }
        Scene { atc, meta, grid, qai, dem }
    }

    #[test]
    fn allocation_has_the_initial_values_of_init_atc() {
        let s = scene(300, 250, 100.0, true);
        let atc = &s.atc;
        assert_eq!((atc.nx, atc.ny, atc.nc, atc.res), (6, 5, 30, 5000));
        assert_eq!(atc.xy_sun.nb(), 8);
        assert_eq!(atc.xyz_f.len(), 256);
        assert_eq!(atc.xyz_hr[255].nb(), 1);
        assert_eq!(atc.xy_mod.domain, ["BLUE", "GREEN", "RED", "NIR"]);
        assert_eq!(atc.xy_fresnel.domain, ["BLUE"]);
        assert_eq!(atc.xy_view.domain[7], "BLUE");
        // the Henyey-Greenstein terms, with the C's arithmetic on floats
        let t = &atc.tthg;
        assert_eq!((t.g1, t.g2, t.alpha), (0.801f32, 0.58f32, 0.948f32));
        assert!((t.sob - 3.0 * (0.948 * (0.801 + 0.58) - 0.58)).abs() < 1e-6);
        assert!((t.hg[0] - (1.0 - 0.801 * 0.801) * 0.948).abs() < 1e-6 && t.hg[4] == 1.602);
        // 100 m pixels: km = 0.1 + 0.05
        assert_eq!(atc.km, 0.15f32);
        assert!(atc.fr > 0.0 && atc.fa > 0.0 && atc.fr < 1.0 && atc.fa < 1.0);
        assert!((atc.lwvl[1] - 0.56f32.ln()).abs() < 1e-7 && atc.lwvl2[1] == atc.lwvl[1] * atc.lwvl[1]);
        // what later stages set is the nodata value
        let fresh = Atc::allocate(&Dn { nx: 300, ny: 250, res: 100.0, wavelength: vec![0.5], domain: vec!["GREEN".into()] }, &Meta::from_rsr_bands(&[1])).unwrap();
        assert_eq!((fresh.wvp, fresh.cc, fresh.hp, fresh.dem.cnum, fresh.view.ab, fresh.mod_[0]), (-9999.0, -9999.0, -9999.0, -9999, -9999.0, -9999.0));
        assert!(!fresh.aodmap && fresh.xy_view.bands.iter().all(|b| b.iter().all(|&v| v == 0.0)));
        assert!(Atc::allocate(&Dn { nx: 60000, ny: 60000, res: 30.0, wavelength: vec![0.5], domain: vec!["GREEN".into()] }, &Meta::from_rsr_bands(&[1])).is_err());
    }

    #[test]
    fn weights_at_the_centre_of_a_cell_and_between_cells() {
        // a grid of 4 x 3 cells of 5000 m, pixels of 100 m
        let grid: Vec<f32> = (0..12).map(|k| 10.0 * k as f32).collect();
        // the corner between the cells (0, 0), (0, 1), (1, 0) and (1, 1): 5000 m from the origin is 50 pixels of 100 m
        let w = interpolation_weights(50, 50, 4, 3, 5000.0, 100.0, &grid, -9999.0);
        assert_eq!((w.gul, w.gur, w.gll, w.glr), (0, 1, 4, 5));
        assert_eq!(interpolate_coarse(&w, &grid), 0.5 * (0.5 * 0.0 + 0.5 * 10.0) + 0.5 * (0.5 * 40.0 + 0.5 * 50.0));
        // the first pixel is at the edge of the grid: the cell itself is used twice
        let w = interpolation_weights(0, 0, 4, 3, 5000.0, 100.0, &grid, -9999.0);
        assert_eq!((w.gul, w.gur, w.gll, w.glr, w.wtop, w.wdown), (0, 0, 0, 0, 0.5, 0.5));
        assert_eq!(interpolate_coarse(&w, &grid), 0.0);
        // the right and bottom edge
        let w = interpolation_weights(199, 149, 4, 3, 5000.0, 100.0, &grid, -9999.0);
        assert_eq!(w.gur, w.glr);
    }

    #[test]
    fn cells_without_data_get_no_weight() {
        let mut grid: Vec<f32> = (0..12).map(|k| 10.0 * k as f32).collect();
        grid[5] = -9999.0;
        // between the cells 0, 1, 4, 5 (the pixel at 5000 m: the corner): the cell 5 has no data
        let w = interpolation_weights(50, 50, 4, 3, 5000.0, 100.0, &grid, -9999.0);
        assert_eq!((w.gul, w.gur, w.gll, w.glr), (0, 1, 4, 5));
        assert_eq!((w.wll, w.wlr), (1.0, 0.0));
        assert!((w.wul + w.wur - 1.0).abs() < 1e-6);
        let v = interpolate_coarse(&w, &grid);
        assert!(v > 0.0 && v < 50.0);
        // two cells without data in the lower row: the whole weight goes to the upper
        grid[4] = -9999.0;
        let w = interpolation_weights(50, 50, 4, 3, 5000.0, 100.0, &grid, -9999.0);
        assert_eq!((w.wtop, w.wdown, w.wll, w.wlr), (1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn angledep_matches_the_formulas_cell_by_cell() {
        let mut s = scene(300, 250, 100.0, true);
        let geo = SceneGeo { ulx: 500000.0, uly: 5400000.0, date: Date { hh: 10, mm: 30, ..Date::from_year_doy(2021, 172) }, to_geo: &warp };
        let brdf = |_: &Atc, g: usize, out: &mut [f32]| {
            for (b, o) in out.iter_mut().enumerate() {
                *o = 1.0 + 0.01 * (g + b) as f32;
            }
        };
        let (nf, ne) = (s.atc.xy_sun.nx, s.atc.xy_sun.ny);
        atmo_angledep(&mut s.atc, &s.meta, &geo, &s.grid, &s.qai, &s.dem, Some(&brdf), 1).unwrap();
        let atc = &s.atc;
        for g in 0..nf * ne {
            if g == 3 {
                // no view angle: the cell is not touched
                assert!(atc.xy_tg.bands.iter().all(|b| b[g] == 0.0) && atc.xy_psi.bands[0][g] == 0.0);
                continue;
            }
            let (ms, mv) = (atc.xy_sun.bands[C_ZEN][g], atc.xy_view.bands[C_ZEN][g]);
            let psi = backscatter(ms, mv, atc.xy_sun.bands[AZI][g], atc.xy_view.bands[AZI][g]);
            assert_eq!(atc.xy_psi.bands[0][g], psi);
            assert_eq!(atc.xy_pr.bands[0][g], phase_molecular(psi));
            assert_eq!(atc.xy_pa.bands[0][g], phase_aerosol(psi, &atc.tthg.hg));
            assert_eq!(atc.xy_fresnel.bands[0][g], fresnel_reflection(atc.xy_sun.bands[ZEN][g]));
            let dem = atc.xy_dem.bands[0][g] as i32;
            let hr = mod_elev_factor(dem as f32 * 0.1 + 0.2 + 0.1 / 2.0);
            assert_eq!(atc.xy_hr.bands[0][g], hr);
            for b in 0..4 {
                assert_eq!(atc.xy_brdf.bands[b][g], 1.0 + 0.01 * (g + b) as f32);
                assert_eq!(atc.xy_mod.bands[b][g], atc.mod_[b] / 1.0 * hr);
                assert_eq!(atc.xy_tsw.bands[b][g], wvp_transmitt(2.5, ms, s.meta.cal[b].rsr_band));
                let tg = gas_transmitt(atc.xy_tsw.bands[b][g], atc.xy_tvw.bands[b][g], atc.xy_tso.bands[b][g], atc.xy_tvo.bands[b][g]);
                assert_eq!(atc.xy_tg.bands[b][g], tg);
            }
        }
        // the elevation bin of a cell is the rounded mean of the 50 x 50 binned pixels (cell 0: columns 0..50, rows 0..50)
        let mean: f64 = (0..50).flat_map(|i| (0..50).map(move |j| (i, j))).map(|(i, j)| s.dem[i * 300 + j] as f64).sum::<f64>() / 2500.0;
        assert_eq!(atc.xy_dem.bands[0][0], mean.round() as f32);
        // without a BRDF function the factor is 1; and the threads do not matter
        let mut s2 = scene(300, 250, 100.0, true);
        atmo_angledep(&mut s2.atc, &s2.meta, &geo, &s2.grid, &s2.qai, &s2.dem, None, 3).unwrap();
        assert!(s2.atc.xy_brdf.bands.iter().all(|b| b.iter().all(|&v| v == 0.0 || v == 1.0)));
        let mut s3 = scene(300, 250, 100.0, true);
        atmo_angledep(&mut s3.atc, &s3.meta, &geo, &s3.grid, &s3.qai, &s3.dem, None, 1).unwrap();
        assert_eq!(s2.atc.xy_tg.bands, s3.atc.xy_tg.bands);
        assert_eq!(s2.atc.xy_dem.bands, s3.atc.xy_dem.bands);
    }

    #[test]
    fn elevdep_gives_every_elevation_bin_and_the_aerosol_flag() {
        for aodmap in [true, false] {
            let mut s = scene(300, 250, 100.0, aodmap);
            let geo = SceneGeo { ulx: 500000.0, uly: 5400000.0, date: Date { hh: 10, mm: 30, ..Date::from_year_doy(2021, 172) }, to_geo: &warp };
            atmo_angledep(&mut s.atc, &s.meta, &geo, &s.grid, &s.qai, &s.dem, None, 1).unwrap();
            s.atc.xy_interp.bands[0][1] = 1.0;
            let l2 = L2Params { domulti: true, doenv: true, ..Default::default() };
            atmo_elevdep(&mut s.atc, &l2, &s.grid, &mut s.qai, &s.dem, 2).unwrap();
            let atc = &s.atc;
            let g = 8;
            let (ms, mv) = (atc.xy_sun.bands[C_ZEN][g], atc.xy_view.bands[C_ZEN][g]);
            let (pr, pa) = (atc.xy_pr.bands[0][g], atc.xy_pa.bands[0][g]);
            let mut zmed = (0.2f64 + 0.1f64 / 2.0) as f32;
            for z in 0..256 {
                let (hr, ha) = (mod_elev_factor(zmed), aod_elev_factor(zmed, 2.0));
                assert_eq!((atc.xyz_hr[z].bands[0][g], atc.xyz_ha[z].bands[0][g]), (hr, ha));
                for b in 0..4 {
                    let mod_ = mod_elev_scale(atc.mod_[b], 1.0, hr);
                    let aod = aod_elev_scale(if aodmap { atc.xy_aod.bands[b][g] } else { atc.aod[b] }, 0.8, ha);
                    let od = optical_depth(aod, mod_);
                    let t = scatt_transmitt(aod, mod_, od, ms, mv);
                    assert_eq!(atc.xyz_aod[z].bands[b][g], aod);
                    assert_eq!(atc.xyz_od[z].bands[b][g], od);
                    assert_eq!(atc.xyz_t[z].bands[b][g], t.total);
                    assert_eq!(atc.xyz_tss[z].bands[b][g], t.tss);
                    assert_eq!(atc.xyz_rho_p[z].bands[b][g], path_ref(true, atc.tthg.sob, aod, mod_, od, pa, pr, t.tsd, t.tvd, ms, mv));
                    assert_eq!(atc.xyz_s[z].bands[b][g], sphere_albedo(aod, mod_, od));
                    assert_eq!(atc.xyz_f[z].bands[b][g], env_weight(aod, mod_, atc.fa, atc.fr));
                }
                zmed += 0.1;
            }
            // the cell without view angles keeps zeros
            assert_eq!(atc.xyz_t[100].bands[0][3], 0.0);
            // the aerosol flag of the pixels
            let counts: Vec<usize> = (0..4).map(|v| s.qai.iter().filter(|&&q| qai::aerosol(q) == v).count()).collect();
            if aodmap {
                assert!(counts[2] > 0, "high AOD pixels {counts:?}");
                assert_eq!(counts[3], 0);
                assert!(counts[0] + counts[1] + counts[2] == 300 * 250);
            } else {
                assert_eq!(counts[3], 300 * 250 - 50 * 50); // but the cell without view angles is off
            }
        }
        // a sensor without a green band
        let mut s = scene(300, 250, 100.0, true);
        s.atc.xy_mod.domain = vec!["BLUE".into(); 4];
        let l2 = L2Params::default();
        assert_eq!(atmo_elevdep(&mut s.atc, &l2, &s.grid, &mut s.qai, &s.dem, 1), Err(AtmoError::Domain("GREEN")));
    }

    #[test]
    fn the_area_of_interest_sets_pixels_off() {
        let mut qai = vec![0i16, 0b100, 1, 0];
        assert_eq!(apply_aoi(&mut qai, None), Ok(Aoi::Cancel));
        assert_eq!(qai, [0, 0b100, 1, 0]);
        assert_eq!(apply_aoi(&mut qai, Some(&[1, 0, 0, 255])), Ok(Aoi::Success));
        assert_eq!(qai, [0, 0b101, 1, 0]);
        assert!(apply_aoi(&mut qai, Some(&[1, 0])).is_err());
        let _ = S_ZEN;
    }
}
