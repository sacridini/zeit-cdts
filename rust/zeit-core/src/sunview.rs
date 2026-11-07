//! The sun and view geometry of the coarse grid of FORCE's atmospheric correction. A port of FORCE 3.11.00 (Copyright (C)
//! 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force),
//! `src/modules/lower-level/sunview-ll.c`: `sun_target_view` (the sun position and its cosine, sine and tangent for every
//! coarse cell, the view angles, the range of the cosines and the low sun flag), `viewgeo` (the nadir line of a Landsat
//! scene from the corners of the swath) and `view_angle` (the view angle of a coarse cell: Landsat's from the distance
//! to the nadir line, Sentinel-2's from the grids of the metadata). `localtime2gmt` and `standard_sunzenith` of the same
//! file are in [`crate::radtran`].
//!
//! The arithmetic is the C's, as in [`crate::radtran`]: C `float` is `f32`, rounded where it is assigned; where C
//! promotes to `double` (`acos`, `atan2`, `sqrt`, `cos`, `sin`, `tan`, `M_PI`, `_D2R_CONV_`) the value is computed in
//! `f64`. Where this differs from the C:
//!
//! * **The sun.** `sunpos` is [`crate::radtran::sunpos`], which wraps the solar position written in a clean room
//!   ([`crate::solpos`]): FORCE's `sun-cl.c` is not ported (its license is not verified), so the sun zenith and azimuth
//!   are not FORCE's bits. The comparison with FORCE gives the shim of `sunpos` in the harness the values of Zeit's.
//! * **The geography.** `get_brick_geo` warps the map coordinates of a cell of a brick to longitude and latitude with the
//!   projection of the brick (GDAL). Here a [`SceneGeo`] holds the map coordinates of the upper left corner of the image
//!   and the function that does the warp; the coordinates of a cell are those of its upper left corner, as FORCE's
//!   (`get_brick_x` and `get_brick_y` add no half pixel).
//! * **The QAI.** A slice of `i16` ([`crate::qai`]) on a [`SceneGrid`] instead of a brick.
//! * **Omitted.** The `FORCE_DEBUG` and `FORCE_CLOCK` blocks, the `ACIX2` branch of `viewgeo` (a build option of FORCE
//!   for a benchmark) and `pl2`, which only that branch reads.

use crate::atc::{Atc, AtmoError, AZI, C_AZI, C_ZEN, S_AZI, S_ZEN, T_AZI, T_ZEN, ZEN};
use crate::gas::{fequal, SceneGrid};
use crate::meta::Meta;
use crate::qai;
use crate::radtran::{sunpos, Date};
use std::f64::consts::PI;

/// `_D2R_CONV_` of `const-cl.h`.
const D2R: f64 = 0.01745329251994329547;

/// The missions of `enum { LANDSAT, SENTINEL2, ... }`: the view geometry is approximated from the swath for Landsat and
/// read from the metadata for Sentinel-2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mission {
    Landsat,
    Sentinel2,
}

/// The place and time of an image, as the bricks of the C carry them: the map coordinates of the upper left corner
/// (`geotran[_GT_ULX_]`, `geotran[_GT_ULY_]`, the same for the coarse grids as for the image), the date of the first
/// band (`get_brick_date(brick, 0)`: the day, and the time of day in GMT that the sun position reads) and the warp of
/// map coordinates to geographic ones (`warp_any_to_geo`, `(x, y) -> (lon, lat)` in degrees).
pub struct SceneGeo<'a> {
    pub ulx: f64,
    pub uly: f64,
    pub date: Date,
    pub to_geo: &'a (dyn Fn(f64, f64) -> (f64, f64) + Sync),
}

impl SceneGeo<'_> {
    /// `get_brick_geo(brick, j, i, &lon, &lat)` of a brick of `res` metres: the geographic coordinates of the upper left
    /// corner of the column `j`, row `i` (`get_brick_x`: `ulx + j * xres`; `get_brick_y`: `uly + i * yres`, `yres = -res`).
    #[inline]
    pub fn lonlat(&self, j: i32, i: i32, res: f64) -> (f64, f64) {
        let x = self.ulx + j as f64 * res;
        let y = self.uly + i as f64 * -res;
        (self.to_geo)(x, y)
    }
}

/// `sun_target_view`: the sun position of every cell of `atc.xy_sun`, the view angles of every cell of `atc.xy_view`
/// (and the nadir line, for Landsat), the range of both cosines, and the low sun flag of the QAI (`qai` on `grid`).
/// For Sentinel-2 the pixels under a cell without view angles are set off in the QAI.
pub fn sun_target_view(
    atc: &mut Atc,
    meta: &Meta,
    mission: Mission,
    geo: &SceneGeo,
    grid: &SceneGrid,
    qai: &mut [i16],
) -> Result<(), AtmoError> {
    let nf = atc.xy_sun.nx;
    let ne = atc.xy_sun.ny;
    let nc = grid.nx * grid.ny;
    if qai.len() != nc {
        return Err(AtmoError::Shape(format!("the QAI has {} values for a grid of {nc} pixels", qai.len())));
    }
    if mission == Mission::Sentinel2
        && (meta.s2.nx < nf || meta.s2.ny < ne || meta.s2.vzen.len() < nf * ne || meta.s2.vazi.len() < nf * ne)
    {
        return Err(AtmoError::Shape(format!(
            "the view angle grids are {} x {} cells, the coarse grid {nf} x {ne}",
            meta.s2.nx, meta.s2.ny
        )));
    }

    // average satellite height (m)
    atc.view.h = match mission {
        Mission::Landsat => 705000.0,
        Mission::Sentinel2 => 786000.0,
    };
    atc.view.h2 = atc.view.h * atc.view.h;

    // compute approx. view geometry for Landsat
    if mission == Mission::Landsat {
        viewgeo(grid, qai, atc, geo)?;
    }

    let res = atc.xy_sun.res;
    for e in 0..ne {
        for f in 0..nf {
            let g = e * nf + f;
            // geo coordinate
            let (lon, lat) = geo.lonlat(f as i32, e as i32, res);

            // calculate sun angles
            let (zen, azi) = sunpos(lat as f32, lon as f32, &geo.date);

            // degree to radians
            let zen = (zen as f64 * D2R) as f32;
            let azi = (azi as f64 * D2R) as f32;

            set_angles(&mut atc.xy_sun, g, zen, azi);

            // satellite view geometry
            view_angle(meta, mission, atc, grid, qai, f, e, g)?;
        }
    }

    // min/max of cos(szen) & cos(vzen)
    let (lo, hi) = atc.xy_sun.range(C_ZEN);
    atc.cosszen = [lo, hi];
    let (lo, hi) = atc.xy_view.range(C_ZEN);
    atc.cosvzen = [lo, hi];

    // low sun flag
    let (gnx, gres) = (atc.xy_sun.nx, atc.xy_sun.res);
    let xy_szen = atc.xy_sun.band(ZEN);
    for p in 0..nc {
        if qai::off(qai[p]) {
            continue;
        }
        let g = crate::gas::ji2p(grid.res, gres, gnx, p / grid.nx, p % grid.nx);
        if xy_szen[g] as f64 > 1.308997 {
            qai::set_lowsun(&mut qai[p], true);
        }
    }
    Ok(())
}

/// The eight bands of a sun or view brick of one cell: zenith, azimuth (radians) and their cosine, sine and tangent.
fn set_angles(brick: &mut crate::atc::CoarseBrick, g: usize, zen: f32, azi: f32) {
    let (zd, ad) = (zen as f64, azi as f64);
    brick.bands[ZEN][g] = zen;
    brick.bands[AZI][g] = azi;
    brick.bands[C_ZEN][g] = zd.cos() as f32;
    brick.bands[C_AZI][g] = ad.cos() as f32;
    brick.bands[S_ZEN][g] = zd.sin() as f32;
    brick.bands[S_AZI][g] = ad.sin() as f32;
    brick.bands[T_ZEN][g] = zd.tan() as f32;
    brick.bands[T_AZI][g] = ad.tan() as f32;
}

/// `viewgeo`: the approximate view geometry of a Landsat scene, the line through the middle of the top and the middle of
/// the bottom of the swath (the on pixels of the QAI): `atc.view` gets the nadir line `a x + b y + c = 0` in image
/// coordinates, its length `ab` and its angle in geographic coordinates.
pub fn viewgeo(grid: &SceneGrid, qai: &[i16], atc: &mut Atc, geo: &SceneGeo) -> Result<(), AtmoError> {
    let (nx, ny) = (grid.nx as i32, grid.ny as i32);
    if qai.len() != grid.nx * grid.ny {
        return Err(AtmoError::Shape(format!("the QAI has {} values for a grid of {} pixels", qai.len(), grid.nx * grid.ny)));
    }
    let (mut uly, mut lry, mut urx) = (-1i32, -1i32, -1i32);
    let mut llx = nx * ny;
    let (mut lrx, mut lly, mut ury, mut ulx) = (0i32, 0i32, 0i32, 0i32);

    // find corners of image
    let mut p = 0usize;
    for i in 0..ny {
        for j in 0..nx {
            let pp = p;
            p += 1;
            if qai::off(qai[pp]) {
                continue;
            }
            if uly == -1 {
                uly = i;
                ulx = j;
            }
            if j > urx {
                ury = i;
                urx = j;
            }
            if j <= llx {
                lly = i;
                llx = j;
            }
            if i >= lry {
                lry = i;
                lrx = j;
            }
        }
    }

    // midpoints of top and bottom
    let (ux, uy) = ((ulx + urx) / 2, (uly + ury) / 2);
    let (lx, ly) = ((llx + lrx) / 2, (lly + lry) / 2);

    // geo coordinate
    let res = grid.res;
    let (gux, guy) = geo.lonlat(ux, uy, res);
    let (glx, gly) = geo.lonlat(lx, ly, res);

    // difference from top to bottom
    let dx = gux - glx;
    let dy = gly - guy;

    // angle of nadir line, measured clockwise from north or south
    atc.view.geo_angle_nadir = (dy.atan2(dx) + PI / 2.0) as f32;

    // cartesian definition of nadir line: ax + by + c = 0
    atc.view.a = (uy - ly) as f32;
    atc.view.b = (lx - ux) as f32;
    atc.view.c = (ly * ux - lx * uy) as f32;

    // the perpendicular distance of a point to the nadir line is given by |ax+by+c|/sqrt(aa+bb)
    atc.view.ab = ((atc.view.a * atc.view.a + atc.view.b * atc.view.b) as f64).sqrt() as f32;
    Ok(())
}

/// `view_angle`: the view angles of the cell `g` (column `f`, row `e`) of the coarse grid. Landsat: from the distance
/// of the cell to the nadir line and the height of the sensor; the azimuth is perpendicular to the nadir line.
/// Sentinel-2: from the grids of the metadata, and the pixels under a cell with no data are set off in the QAI.
#[allow(clippy::too_many_arguments)]
pub fn view_angle(
    meta: &Meta,
    mission: Mission,
    atc: &mut Atc,
    grid: &SceneGrid,
    qai: &mut [i16],
    f: usize,
    e: usize,
    g: usize,
) -> Result<(), AtmoError> {
    let (nx, ny) = (grid.nx as i32, grid.ny as i32);
    let res = grid.res as f32;
    let cres = atc.xy_view.res as f32;
    let rres = res / cres;
    let (fi, ei) = (f as i32, e as i32);

    let (zen, azi) = match mission {
        Mission::Landsat => {
            // distance from nadir line in pixels
            let dist = (atc.view.a * fi as f32 / rres + atc.view.b * ei as f32 / rres + atc.view.c) / atc.view.ab;

            // view zenith
            let t = res * dist * res * dist;
            let mut zen = (atc.view.h as f64 / ((atc.view.h2 + t) as f64).sqrt()).acos() as f32;
            if zen.is_nan() {
                zen = 0.0;
            }

            // view azimuth
            let azi = if dist > 0.0 {
                // left of nadir, i.e. right looking
                (atc.view.geo_angle_nadir as f64 + PI / 2.0) as f32
            } else {
                // right of nadir, i.e. left looking
                let mut azi = (atc.view.geo_angle_nadir as f64 - PI / 2.0) as f32;
                while azi < 0.0 {
                    azi = (azi as f64 + 2.0 * PI) as f32;
                }
                azi
            };
            (zen, azi)
        }
        Mission::Sentinel2 => {
            let (vzen, vazi) = (meta.s2.vzen_at(e, f), meta.s2.vazi_at(e, f));
            if vzen == meta.s2.nodata || vazi == meta.s2.nodata {
                // set nodata for all pixels in this grid cell
                let mut i = (ei as f32 / rres) as i32;
                while (i as f32) < ((ei + 1) as f32 / rres) {
                    let mut j = (fi as f32 / rres) as i32;
                    while (j as f32) < ((fi + 1) as f32 / rres) {
                        if i <= ny - 1 && j <= nx - 1 {
                            let p = (i * nx + j) as usize;
                            qai::set_off(&mut qai[p], true);
                        }
                        j += 1;
                    }
                    i += 1;
                }
                (atc.nodata, atc.nodata)
            } else {
                ((vzen as f64 * D2R) as f32, (vazi as f64 * D2R) as f32)
            }
        }
    };

    if fequal(zen, atc.nodata) {
        let nd = atc.nodata;
        for b in 0..8 {
            atc.xy_view.bands[b][g] = nd;
        }
    } else {
        set_angles(&mut atc.xy_view, g, zen, azi);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atc::Dn;

    /// A geographic scene: map coordinates in degrees times 1e5 / 111 (a warp that is an affine map of the corner).
    fn warp(x: f64, y: f64) -> (f64, f64) {
        (10.0 + (x - 500000.0) * 9.0e-6, y * 9.0e-6)
    }

    fn scene(nx: usize, ny: usize, res: f64) -> (Atc, Meta, SceneGrid) {
        let dn = Dn { nx, ny, res, wavelength: vec![0.49, 0.56, 0.665], domain: vec!["BLUE".into(), "GREEN".into(), "RED".into()] };
        let meta = Meta::from_rsr_bands(&[1, 2, 3]);
        (Atc::allocate(&dn, &meta).unwrap(), meta, SceneGrid { nx, ny, res })
    }

    fn geo() -> SceneGeo<'static> {
        SceneGeo { ulx: 500000.0, uly: 5400000.0, date: Date { hh: 10, mm: 30, ..Date::from_year_doy(2021, 172) }, to_geo: &warp }
    }

    /// A Landsat swath: a parallelogram of on pixels leaning 12 degrees, the rest off.
    fn swath(nx: usize, ny: usize) -> Vec<i16> {
        let mut qai = vec![1i16; nx * ny];
        for i in 0..ny {
            let shift = (i as f64 * 0.2126) as usize;
            for j in 0..nx {
                if j >= shift && j < shift + nx * 3 / 4 {
                    qai[i * nx + j] = 0;
                }
            }
        }
        qai
    }

    #[test]
    fn landsat_view_angles_grow_from_the_nadir_line() {
        let (mut atc, meta, grid) = scene(400, 300, 400.0);
        let mut qai = swath(400, 300);
        sun_target_view(&mut atc, &meta, Mission::Landsat, &geo(), &grid, &mut qai).unwrap();
        assert_eq!((atc.view.h, atc.view.h2), (705000.0, 705000.0f32 * 705000.0));
        // the nadir line leans, the zenith is 0 on it and the farther the larger, up to about 15 degrees
        let (lo, hi) = (atc.cosvzen[0], atc.cosvzen[1]);
        assert!(hi > 0.9999 && lo < hi && lo > 0.95, "cosvzen {lo} {hi}");
        let zen = atc.xy_view.band(ZEN);
        assert!(zen.iter().all(|&z| (0.0..0.3).contains(&z)));
        // left and right looking azimuths differ by pi
        let azi = atc.xy_view.band(AZI);
        let (amin, amax) = azi.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!((amax - amin - std::f32::consts::PI).abs() < 1e-5, "{amin} {amax}");
        // each cell: cos and sin of the zenith
        for g in 0..atc.xy_view.ncell() {
            let z = zen[g] as f64;
            assert_eq!(atc.xy_view.bands[C_ZEN][g], z.cos() as f32);
            assert_eq!(atc.xy_view.bands[S_ZEN][g], z.sin() as f32);
        }
        // the sun of June 21 at 48 N is high: no low sun flag
        assert!(qai.iter().all(|&q| !qai::lowsun(q)));
    }

    #[test]
    fn low_sun_flag_and_range_of_the_cosines() {
        let (mut atc, meta, grid) = scene(120, 100, 400.0);
        let mut qai = swath(120, 100);
        let mut g = geo();
        g.date = Date { hh: 3, ..Date::from_year_doy(2021, 355) }; // before dawn at 10 E
        sun_target_view(&mut atc, &meta, Mission::Landsat, &g, &grid, &mut qai).unwrap();
        assert!(atc.xy_sun.band(ZEN).iter().all(|&z| z > 1.5));
        for (p, &q) in qai.iter().enumerate() {
            assert_eq!(qai::lowsun(q), !qai::off(q), "pixel {p}");
        }
        assert!(atc.cosszen[0] <= atc.cosszen[1] && atc.cosszen[1] < 0.0);
    }

    #[test]
    fn sentinel2_view_angles_from_the_grids_and_cells_without_data() {
        let (mut atc, mut meta, grid) = scene(300, 250, 100.0);
        let (nf, ne) = (atc.xy_sun.nx, atc.xy_sun.ny);
        assert_eq!((nf, ne), (6, 5));
        meta.s2 = crate::meta::S2Meta {
            nodata: -9999.0,
            vzen: (0..nf * ne).map(|k| 2.0 + 0.5 * k as f32).collect(),
            vazi: (0..nf * ne).map(|k| 100.0 + 1.5 * k as f32).collect(),
            nx: nf,
            ny: ne,
        };
        meta.s2.vzen[7] = -9999.0; // row 1, column 1
        let mut qai = vec![0i16; 300 * 250];
        sun_target_view(&mut atc, &meta, Mission::Sentinel2, &geo(), &grid, &mut qai).unwrap();
        assert_eq!(atc.view.h, 786000.0);
        for g in 0..nf * ne {
            if g == 7 {
                assert!(atc.xy_view.is_nodata(0, g) && atc.xy_view.is_nodata(7, g));
            } else {
                assert_eq!(atc.xy_view.bands[ZEN][g], (meta.s2.vzen[g] as f64 * D2R) as f32);
                assert_eq!(atc.xy_view.bands[AZI][g], (meta.s2.vazi[g] as f64 * D2R) as f32);
            }
        }
        // the pixels of the cell (res 100 m, 5000 m cell: 50 pixels) are off, and only they
        for i in 0..250 {
            for j in 0..300 {
                let inside = (50..100).contains(&i) && (50..100).contains(&j);
                assert_eq!(qai::off(qai[i * 300 + j]), inside, "({i}, {j})");
            }
        }
        // the cosine range skips the cell without data
        assert!(atc.cosvzen[0] > 0.95 && atc.cosvzen[1] <= 1.0);
    }

    #[test]
    fn grids_that_do_not_fit_are_errors() {
        let (mut atc, mut meta, grid) = scene(300, 250, 100.0);
        let mut qai = vec![0i16; 300 * 250];
        assert!(sun_target_view(&mut atc, &meta, Mission::Sentinel2, &geo(), &grid, &mut qai).is_err());
        meta.s2.nx = 6;
        let mut short = vec![0i16; 10];
        assert!(sun_target_view(&mut atc, &meta, Mission::Landsat, &geo(), &grid, &mut short).is_err());
    }
}
