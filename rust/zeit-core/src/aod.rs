//! The dark targets and the aerosol optical depth over each of them, and the AOD map: FORCE's `aod-ll.c`. A port of
//! FORCE 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring, Copyright (C)
//! 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force), `src/modules/lower-level/aod-ll.c`
//! lines 45-1370: `aodfileread`, `extract_dark_target`, `water_lib`, `land_lib`, `veg_lib`, `aod_lib_to_target`,
//! `aod_from_target`, `aod_elevation_minimizer` and `aod_elev_dependency`, with `aod_linear_fit` and
//! `aod_polynomial_fit` (lines 1371-1480, which `aod_from_target` calls), the segmentation and the distance
//! transform of `cross-level/imagefuns-cl.c` (`connectedcomponents_`: Chang, Chen & Lu 2004; `dist_transform_`:
//! Meijster, Roerdink & Hesselink 2006), and `get_cloud` of `quality-cl.c` ([`crate::qai`]). The minimizers are GSL
//! 2.7's ([`crate::gsl_brent`], [`crate::gsl_nmsimplex`]). The rest of the file (lines 1481-1837) is here too:
//! `aod_map`, `interpolate_aod_map` (with the inverse distance weighting of GDAL 3.9.3's `GDALGridCreate`,
//! [`crate::gdal_grid`]) and `compile_aod`.
//!
//! The arithmetic is the C's, as in [`crate::radtran`]: C `float` is `f32`, and every `float` variable is rounded
//! where it is assigned; where C promotes to `double` (`exp`, `log`, `pow`, `/ 10000.0`, a `double` literal) the
//! value is computed in `f64`. Where this differs from the C or from FORCE's own behaviour:
//!
//! * **Threads.** FORCE runs the pixel loops with OpenMP. Per-pixel loops (candidate targets, lapsing spectra) and
//!   the loops over objects of `extract_dark_target` do not depend on the order, but the scene-average environment
//!   reflectance is a sum of doubles per thread (`reduction(+: en)` and `esum_priv`), so its last bits depend on the
//!   number of threads, and the loop over objects of `aod_from_target` hands each thread ONE `aodest`, `aodlog` and
//!   `aodfit` array that the next object it processes finds as the last one left it (an entry that
//!   `aod_lib_to_target` does not write, or a band the fit does not set, keeps the value of the previous object). This
//!   port is FORCE with one thread (`OMP_NUM_THREADS=1`): sums in pixel order, the arrays carried from object to
//!   object in object order. It is deterministic and equals FORCE with one thread.
//! * **Errors.** Where the C prints a message and returns `FAILURE`/`NULL`, this returns an [`AodError`]. A `strtok`
//!   token missing in a line of the AOD file (a crash of FORCE: `atof(NULL)`) skips the line. A cell size of zero
//!   (the coarse grid finer than the image, an endless loop in FORCE) is an error. The elevation minimizer
//!   whose bracketing function is not finite (`GSL_EBADFUNC`; FORCE goes on with an uninitialized state) is "no
//!   minimum found", as for `GSL_EINVAL`.
//! * **Omitted.** `print_dark_object` (only compiled with `FORCE_DEBUG`, only prints).
//!
//! The brick machinery of FORCE is plain arrays here ([`Toa`], [`Top`], the QAI as `&[i16]`, [`crate::atc::Atc`]).

#![allow(clippy::needless_range_loop)] // the loops are the C's, index by index

use crate::atc::{Atc, AZI, C_ZEN};
use crate::gas::{atof, Toa};
use crate::gsl_brent::{self, Brent, IntervalTest};
use crate::gsl_nmsimplex::NmSimplex;
use crate::qai;
use crate::radtran::{
    aod_elev_factor, aod_elev_scale, env_weight, env_weight_aerosol, env_weight_molecular, mod_elev_factor, mod_elev_scale, optical_depth,
    path_ref, scatt_transmitt, sphere_albedo,
};
use crate::radtran_tables as tab;
use std::fmt;
use std::fs::File;
use std::io::Read;

/// `NPOW_10` of FORCE: the size of its string buffers (a file name, a line of the AOD file).
const NPOW_10: usize = 1024;

/// The type of a dark target (`_AOD_WAT_`, `_AOD_SHD_`, `_AOD_VEG_`). The shadow type is disabled in FORCE 3.11.00
/// ("until further notice"): no targets are extracted and no AOD is estimated for it; the branches that handle it in the
/// other functions are ported anyway.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetType {
    Water,
    Shadow,
    Vegetation,
}

/// What can go wrong in these functions (where FORCE prints a message and returns `FAILURE`).
#[derive(Debug, Clone, PartialEq)]
pub enum AodError {
    /// The brick has no band of this domain (`find_domain`).
    NoDomain(&'static str),
    /// The inputs do not have the sizes they claim, or a grid does not fit.
    Shape(String),
    /// `Buffer Overflow in assembling filename`.
    FileName,
    /// `Unable to open aod file!`
    OpenFile(String),
    /// `error in library/target AOD optimization` / `error in estimating AOD`.
    Estimation,
    /// A GSL minimizer failed where FORCE's error handler would abort.
    Gsl(&'static str),
}

impl fmt::Display for AodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AodError::NoDomain(d) => write!(f, "the brick has no band of domain {d}"),
            AodError::Shape(s) => write!(f, "{s}"),
            AodError::FileName => write!(f, "buffer overflow in assembling filename"),
            AodError::OpenFile(p) => write!(f, "unable to open aod file {p}"),
            AodError::Estimation => write!(f, "error in estimating AOD"),
            AodError::Gsl(s) => write!(f, "GSL: {s}"),
        }
    }
}

impl std::error::Error for AodError {}

// ---------------------------------------------------------------------------------------------------------------
// aodfileread
// ---------------------------------------------------------------------------------------------------------------

/// `fgets(buffer, NPOW_10, fp)` on a whole file: the pieces of at most 1023 bytes, each ending after a newline,
/// cut at a NUL (where the C string ends).
fn fgets_chunks(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let end = (pos + NPOW_10 - 1).min(data.len());
        let mut e = pos;
        while e < end {
            e += 1;
            if data[e - 1] == b'\n' {
                break;
            }
        }
        let chunk = &data[pos..e];
        pos = e;
        let cut = chunk.iter().position(|&c| c == 0).unwrap_or(chunk.len());
        out.push(&chunk[..cut]);
    }
    out
}

/// The AOD spectrum of a power law `exp(co0) * wvl ** (co1 + co2 * ln wvl)` (the global AERONET median, or a line of
/// the AOD file).
fn aod_spectrum(atc: &Atc, co: &[f32; 3]) -> Vec<f32> {
    (0..atc.xy_aod.nb())
        .map(|b| ((co[0] as f64).exp() * (atc.wvl[b] as f64).powf((co[1] + co[2] * atc.lwvl[b]) as f64)) as f32)
        .collect()
}

/// `aodfileread`: the AOD spectrum (per band of `atc.xy_aod`) from the global fallback (median AOD of AERONET), or,
/// if `dir` is given (`d_aod` is not `"NULL"`), from the line of `AOD_DDD.txt` (`DDD` the day of the year `doy`, from
/// the date of the image) closest to the scene centre `(center_lon, center_lat)` (geographic coordinates; FORCE
/// gets them from the projection of `xy_aod`). The file has one site per line, blank-separated `lon lat c0 c1 c2`:
/// the coefficients of `exp(c0) * wvl ** (c1 + c2 * ln wvl)`.
pub fn aodfileread(atc: &Atc, dir: Option<&str>, center_lon: f64, center_lat: f64, doy: i32) -> Result<Vec<f32>, AodError> {
    // global fallback: AOD derived from global AERONET data (median AOD)
    let mut co: [f32; 3] = [-2.8665, -1.4787, -0.0486];
    let Some(dir) = dir else {
        return Ok(aod_spectrum(atc, &co));
    };
    // daily LUT
    let fname = format!("{dir}/AOD_{doy:03}.txt");
    if fname.len() >= NPOW_10 {
        return Err(AodError::FileName);
    }
    let mut data = Vec::new();
    File::open(&fname).and_then(|mut f| f.read_to_end(&mut data)).map_err(|_| AodError::OpenFile(fname.clone()))?;

    let mut min_dist = i64::MAX as f32;
    for line in fgets_chunks(&data) {
        let mut tok = line.split(|&c| c == b' ').filter(|t| !t.is_empty());
        let t: Vec<&[u8]> = tok.by_ref().take(5).collect();
        if t.len() < 5 {
            continue;
        }
        let (site_x, site_y) = (atof(t[0]), atof(t[1]));
        let (diff_x, diff_y) = (center_lon - site_x, center_lat - site_y);
        let dist = (diff_x * diff_x + diff_y * diff_y).sqrt() as f32;
        if dist < min_dist {
            min_dist = dist;
            co = [atof(t[2]) as f32, atof(t[3]) as f32, atof(t[4]) as f32];
        }
    }
    Ok(aod_spectrum(atc, &co))
}

// ---------------------------------------------------------------------------------------------------------------
// the inputs and outputs of the dark target extraction
// ---------------------------------------------------------------------------------------------------------------

/// The topographic derivatives `extract_dark_target` reads (`top_t` of `topo-ll.h`: `dem`, `exp`, `ill`, `sky`), on
/// the grid of the TOA brick, one value per pixel. `compile_topography` is not ported yet: the caller provides them.
#[derive(Clone, Debug)]
pub struct Top {
    /// The binned elevation (`TOP->dem`, band 0): the elevation of a pixel is `dem * step + min + step / 2` with
    /// the `dem` statistics of the [`Atc`].
    pub dem: Vec<u8>,
    /// The slope (`TOP->exp`, band `ZEN`), in units of 0.01 degree... as FORCE scales it: `slope < 870` is 5 degrees.
    pub slope: Vec<u16>,
    /// The illumination angle (`TOP->ill`), cosine times 10000.
    pub ill: Vec<i16>,
    /// The sky view factor (`TOP->sky`), times 10000.
    pub sky: Vec<u16>,
}

/// A single dark target (`darkobj_t`).
#[derive(Clone, Debug, Default)]
pub struct DarkObj {
    /// Number of valid pixels.
    pub k: i32,
    /// Is the object valid?
    pub valid: bool,
    /// The bounding box.
    pub imin: i32,
    pub imax: i32,
    pub jmin: i32,
    pub jmax: i32,
    /// The centroid (row, column).
    pub i: i32,
    pub j: i32,
    /// Cosine of the sun and of the view zenith and relative azimuth.
    pub ms: f32,
    pub mv: f32,
    pub razi: f32,
    /// The coarse grid cell (row, column, index).
    pub e: i32,
    pub f: i32,
    pub g: i32,
    /// The target and the environment reflectance, per band.
    pub ttoa: Vec<f32>,
    pub etoa: Vec<f32>,
    /// The radius of the target (the largest distance to the shore).
    pub r: i32,
    /// The illumination angle (cosine), the sky view and the elevation of the target.
    pub cosi: f32,
    pub sky: f32,
    pub z: f32,
    /// The elevation correction factors of the AOD and the MOD.
    pub ha: f32,
    pub hr: f32,
    /// The fitted and the estimated AOD, per band.
    pub aod: Vec<f32>,
    pub est: Vec<f32>,
    /// R-squared of the best fit.
    pub rsq: f32,
    /// Which spectrum of the library was the best.
    pub lib_id: i32,
    /// The type of the fit: 0 curved (polynomial), 1 linear (Angstrom).
    pub ang_fit: i32,
    /// The coefficients of the best fit.
    pub coef: [f32; 3],
}

/// The container of the targets of an image (`dark_t`): `nwat`, `nshd`, `nveg` are the numbers of valid targets
/// (what `aod_from_target` returns), the lengths of the vectors the candidates (`kwat`, ...).
#[derive(Clone, Debug, Default)]
pub struct DarkTargets {
    pub wat: Vec<DarkObj>,
    pub shd: Vec<DarkObj>,
    pub veg: Vec<DarkObj>,
    pub nwat: i32,
    pub nshd: i32,
    pub nveg: i32,
}

// ---------------------------------------------------------------------------------------------------------------
// imagefuns-cl.c: the distance transform and the connected components
// ---------------------------------------------------------------------------------------------------------------

/// `dist_transform_`: the distance of each FALSE cell to the next TRUE one (Meijster et al. 2006), the Euclidean
/// distance truncated to an integer. `image` is 0/1, row-major, `nx` columns.
pub fn dist_transform(image: &[u8], nx: usize, ny: usize) -> Vec<u16> {
    let (nxi, nyi) = (nx as i64, ny as i64);
    let mut g = vec![0u16; nx * ny];
    let mut distance = vec![0u16; nx * ny];

    // first phase
    for x in 0..nx {
        // scan 1
        g[x] = if image[x] != 0 { 0 } else { (nx + ny) as u16 };
        for y in 1..ny {
            g[y * nx + x] = if image[y * nx + x] != 0 { 0 } else { (1 + g[(y - 1) * nx + x] as u32) as u16 };
        }
        // scan 2
        let mut y = nyi - 2;
        while y >= 0 {
            let yu = y as usize;
            if g[(yu + 1) * nx + x] < g[yu * nx + x] {
                g[yu * nx + x] = (1 + g[(yu + 1) * nx + x] as u32) as u16;
            }
            y -= 1;
        }
    }

    // second phase
    let dfun = |x: i64, i: i64, y: usize, g: &[u16]| -> i32 {
        let gi = g[nx * y + i as usize] as i32;
        ((x - i) * (x - i)) as i32 + gi.wrapping_mul(gi)
    };
    let sep = |i: i64, u: i64, y: usize, g: &[u16]| -> i64 {
        let gu = g[nx * y + u as usize] as i32;
        let gi = g[nx * y + i as usize] as i32;
        ((u * u - i * i) as i32 + gu.wrapping_mul(gu) - gi.wrapping_mul(gi)) as i64 / (2 * (u - i))
    };
    let mut s = vec![0i64; nx];
    let mut t = vec![0i64; nx];
    for y in 0..ny {
        let mut q: i64 = 0;
        s[0] = 0;
        t[0] = 0;
        // scan 3
        for u in 1..nxi {
            while q >= 0 && dfun(t[q as usize], s[q as usize], y, &g) > dfun(t[q as usize], u, y, &g) {
                q -= 1;
            }
            if q < 0 {
                q = 0;
                s[0] = u;
            } else {
                let w = 1 + sep(s[q as usize], u, y, &g);
                if w < nxi {
                    q += 1;
                    s[q as usize] = u;
                    t[q as usize] = w;
                }
            }
        }
        // scan 4
        let mut u = nxi - 1;
        while u >= 0 {
            distance[y * nx + u as usize] = (dfun(u, s[q as usize], y, &g) as f64).sqrt() as u16;
            if u == t[q as usize] {
                q -= 1;
            }
            u -= 1;
        }
    }
    distance
}

const NEIGHBOR: [[i64; 2]; 8] = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]];

/// `ccl_tracer`.
fn ccl_tracer(cy: &mut i64, cx: &mut i64, dir: &mut usize, image: &[u8], ccl: &mut [i32], nx: i64, ny: i64) {
    for _ in 0..7 {
        let y = *cy + NEIGHBOR[*dir][0];
        let x = *cx + NEIGHBOR[*dir][1];
        let inside = y >= 0 && y < ny && x >= 0 && x < nx;
        let tval = if inside { image[(y * nx + x) as usize] } else { 0 };
        if tval == 0 {
            if inside {
                ccl[(y * nx + x) as usize] = -1;
            }
            *dir = (*dir + 1) % 8;
        } else {
            *cy = y;
            *cx = x;
            break;
        }
    }
}

/// `ccl_contourtrace`.
#[allow(clippy::too_many_arguments)]
fn ccl_contourtrace(mut cy: i64, mut cx: i64, label: i32, mut dir: usize, image: &[u8], ccl: &mut [i32], nx: i64, ny: i64) {
    let mut stop = false;
    let mut search = true;
    let (sx, sy) = (cx, cy);

    ccl_tracer(&mut cy, &mut cx, &mut dir, image, ccl, nx, ny);

    if cx != sx || cy != sy {
        let (fx, fy) = (cx, cy);
        while search {
            dir = (dir + 6) % 8;
            ccl[(cy * nx + cx) as usize] = label;
            ccl_tracer(&mut cy, &mut cx, &mut dir, image, ccl, nx, ny);

            if cx == sx && cy == sy {
                stop = true;
            } else if stop {
                if cx == fx && cy == fy {
                    search = false;
                } else {
                    stop = false;
                }
            }
        }
    }
}

/// `connectedcomponents_`: label all 8-connected TRUE patches of `image` (Chang et al. 2004), a unique id (from 1,
/// in the order of the first pixel from the top left) per segment; returns the labels and the number of segments.
pub fn connected_components(image: &[u8], nx: usize, ny: usize) -> (Vec<i32>, i32) {
    let mut ccl = vec![0i32; nx * ny];
    let (nxi, nyi) = (nx as i64, ny as i64);
    let mut k = 0i32;
    let mut p = 0usize;
    for i in 0..nyi {
        let mut label = 0i32;
        for j in 0..nxi {
            if image[p] != 0 {
                if label != 0 {
                    // use the last label
                    ccl[p] = label;
                } else {
                    label = ccl[p];
                    if label == 0 {
                        k += 1;
                        label = k;
                        ccl_contourtrace(i, j, label, 0, image, &mut ccl, nxi, nyi); // external contour
                        ccl[p] = label;
                    }
                }
            } else if label != 0 {
                // the last pixel was labeled
                if ccl[p] == 0 {
                    ccl_contourtrace(i, j - 1, label, 1, image, &mut ccl, nxi, nyi); // internal contour
                }
                label = 0;
            }
            p += 1;
        }
    }
    // replace -1 with 0
    for v in ccl.iter_mut() {
        if *v < 0 {
            *v = 0;
        }
    }
    (ccl, k)
}

// ---------------------------------------------------------------------------------------------------------------
// extract_dark_target
// ---------------------------------------------------------------------------------------------------------------

/// `extract_dark_target`: candidate dark targets of one `kind` in a scene: pixels that pass the spectral tests of
/// the type, are cut into objects by the grid of the coarse cells (a target cannot run through the image), 8-connected
/// and tabulated: bounding box, size, radius (distance to the shore), mean illumination, sky view and elevation,
/// and the mean TOA reflectance of the target and its environment (3 radii around it, or the scene average if
/// fewer than 10 pixels). Objects under 10 pixels or radius 3 are marked not valid. Shadow targets are disabled in
/// FORCE: no targets. `toa` and `qai` are on the grid of the image (`toa.grid`), `top` too.
pub fn extract_dark_target(atc: &Atc, toa: &Toa, qai: &[i16], top: &Top, kind: TargetType) -> Result<Vec<DarkObj>, AodError> {
    // shadow type is disabled until further notice
    if kind == TargetType::Shadow {
        return Ok(Vec::new());
    }
    let nb = toa.bands.len();
    let (nx, ny) = (toa.grid.nx, toa.grid.ny);
    let nc = nx * ny;
    let cs = (atc.xy_aod.res / toa.grid.res).floor();
    if !(cs >= 1.0) {
        return Err(AodError::Shape("the coarse grid must be at least as coarse as the image".into()));
    }
    let cell_size = cs as usize;
    if qai.len() != nc || top.dem.len() != nc || top.slope.len() != nc || top.ill.len() != nc || top.sky.len() != nc {
        return Err(AodError::Shape("qai and the topography must have one value per pixel".into()));
    }
    if toa.bands.iter().any(|b| b.len() != nc) || atc.aod_bands.len() != nb || atc.wvl.len() != nb {
        return Err(AodError::Shape("toa and the atc must have nb bands of nx * ny pixels".into()));
    }
    let blue = toa.find_domain("BLUE").ok_or(AodError::NoDomain("BLUE"))?;
    toa.find_domain("GREEN").ok_or(AodError::NoDomain("GREEN"))?;
    let red = toa.find_domain("RED").ok_or(AodError::NoDomain("RED"))?;
    let nir = toa.find_domain("NIR").ok_or(AodError::NoDomain("NIR"))?;
    let sw2 = toa.find_domain("SWIR2").ok_or(AodError::NoDomain("SWIR2"))?;
    let toa_ = &toa.bands;
    let aod_bands = &atc.aod_bands;
    let (dem_step, dem_min) = (atc.dem.step, atc.dem.min);
    let elev = |p: usize| -> f32 { top.dem[p] as f32 * dem_step + dem_min + dem_step / 2.0 };

    // identify candidate targets
    let mut target = vec![0u8; nc];
    let mut invert = vec![0u8; nc];
    for p in 0..nc {
        if qai::off(qai[p]) {
            invert[p] = 1;
            continue;
        }
        let z = elev(p);
        let slp = top.slope[p];
        let (s2, r, b, n) = (toa_[sw2][p], toa_[red][p], toa_[blue][p], toa_[nir][p]);
        let hit = match kind {
            // water: swir2 < 2.5%, red < 20%, blue > nir, slope < 1 deg if z > 1 km, < 5 deg if z < 1 km
            TargetType::Water => s2 < 250 && r < 2000 && b > n && ((z < 1.0 && slp < 870) || (z > 1.0 && slp < 175)),
            // shadow: swir2 < 5%, red < 20%, blue > nir, illumination angle > 60 deg, slope > 5 deg
            TargetType::Shadow => s2 < 500 && r < 2000 && b > n && top.ill[p] < 5000 && slp > 870,
            // vegetation: swir2 < 5%, red < 20%, blue < nir, NDVI > 0.4, slope < 15 deg
            TargetType::Vegetation => {
                s2 < 500
                    && r < 2000
                    && b < n
                    && ((n as i32 - r as i32) as f32 / (n as i32 + r as i32) as f32) as f64 > 0.4
                    && slp < 2620
            }
        };
        if hit {
            target[p] = 1;
        } else {
            invert[p] = 1;
        }
    }

    // target segmentation: sub-divide the targets, so that a target cannot run through the whole image
    for i in (0..ny).step_by(cell_size) {
        for j in 0..nx {
            target[i * nx + j] = 0;
        }
    }
    for j in (0..nx).step_by(cell_size) {
        for i in 0..ny {
            target[i * nx + j] = 0;
        }
    }
    let (segment, num) = connected_components(&target, nx, ny);
    drop(target);
    // distance to the "shore"
    let distance = dist_transform(&invert, nx, ny);
    drop(invert);

    // find lapsing spectra
    let mut lapse = vec![false; nc];
    for p in 0..nc {
        if qai::off(qai[p]) || segment[p] < 1 {
            continue;
        }
        if (qai::shadow(qai[p]) || qai::cloud(qai[p]) > 0) && kind != TargetType::Shadow {
            continue;
        }
        if kind == TargetType::Vegetation && distance[p] < 2 {
            continue;
        }
        let mut valid = true;
        for b in 1..nb {
            if !aod_bands[b] {
                continue;
            }
            if kind == TargetType::Vegetation && b == sw2 {
                continue;
            }
            for bb in (0..b).rev() {
                if !aod_bands[bb] {
                    continue;
                }
                if toa_[b][p] > toa_[bb][p] {
                    valid = false;
                    break;
                }
            }
            if !valid {
                break;
            }
        }
        lapse[p] = valid;
    }

    // scene average: global environment
    let mut esum = vec![0.0f64; nb];
    let mut en = 0.0f64;
    for p in 0..nc {
        if qai::off(qai[p]) || qai::shadow(qai[p]) || qai::cloud(qai[p]) > 0 {
            continue;
        }
        for b in 0..nb {
            if !aod_bands[b] {
                continue;
            }
            esum[b] += toa_[b][p] as f64 / 10000.0; // rescale to prevent overflow
        }
        en += 1.0;
    }
    let etoa: Vec<f32> = if en > 0.0 { (0..nb).map(|b| (esum[b] / en) as f32).collect() } else { vec![0.0; nb] };

    // allocate and initialize the darkobject struct
    let num_u = num as usize;
    let mut dobj: Vec<DarkObj> = (0..num_u)
        .map(|_| DarkObj {
            imin: ny as i32,
            jmin: nx as i32,
            ttoa: vec![0.0; nb],
            etoa: vec![0.0; nb],
            aod: vec![0.0; nb],
            est: vec![0.0; nb],
            ..Default::default()
        })
        .collect();
    let mut zsum = vec![0.0f64; num_u];
    let mut csum = vec![0.0f64; num_u];
    let mut ssum = vec![0.0f64; num_u];

    // 1st pass, loop over the image: bounding box, object size, mean illumination angle, elevation
    for i in 0..ny {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) {
                continue;
            }
            let o = segment[p] - 1;
            if o < 0 {
                continue;
            }
            let o = o as usize;
            if (qai::shadow(qai[p]) || qai::cloud(qai[p]) > 0) && kind != TargetType::Shadow {
                continue;
            }
            if !lapse[p] {
                continue;
            }
            csum[o] += top.ill[p] as f64 / 10000.0; // rescale to prevent overflow
            ssum[o] += top.sky[p] as f64 / 10000.0;
            zsum[o] += elev(p) as f64;

            let d = &mut dobj[o];
            // object radius
            if distance[p] as i32 > d.r {
                d.r = distance[p] as i32;
            }
            // object location
            let (ii, jj) = (i as i32, j as i32);
            if ii > d.imax {
                d.imax = ii;
            }
            if jj > d.jmax {
                d.jmax = jj;
            }
            if ii < d.imin {
                d.imin = ii;
            }
            if jj < d.jmin {
                d.jmin = jj;
            }
            // pixel number
            d.k += 1;
        }
    }

    // 2nd pass, loop over the objects
    let mut tsum = vec![0.0f64; nb];
    let mut esum = vec![0.0f64; nb];
    for o in 0..num_u {
        // ignore small and narrow objects
        if dobj[o].k < 10 || dobj[o].r < 3 {
            dobj[o].valid = false;
            continue;
        }
        dobj[o].valid = true;

        let (mut en, mut tn) = (0.0f64, 0.0f64);
        tsum.iter_mut().for_each(|v| *v = 0.0);
        esum.iter_mut().for_each(|v| *v = 0.0);

        let r3 = 3 * dobj[o].r;
        // use 3x object radius for the environment reflectance
        for i in (dobj[o].imin - r3)..=(dobj[o].imax + r3) {
            for j in (dobj[o].jmin - r3)..=(dobj[o].jmax + r3) {
                if i < 0 || j < 0 || i >= ny as i32 || j >= nx as i32 {
                    continue;
                }
                let p = i as usize * nx + j as usize;
                if qai::off(qai[p]) {
                    continue;
                }
                if (qai::shadow(qai[p]) || qai::cloud(qai[p]) > 0) && kind != TargetType::Shadow {
                    continue;
                }
                if segment[p] == 0 {
                    // no target: environment reflectance
                    for b in 0..nb {
                        if !aod_bands[b] {
                            continue;
                        }
                        esum[b] += toa_[b][p] as f64 / 10000.0;
                    }
                    en += 1.0;
                } else if segment[p] - 1 == o as i32 && lapse[p] {
                    // a target and the pixel is lapsing: target reflectance
                    for b in 0..nb {
                        if !aod_bands[b] {
                            continue;
                        }
                        tsum[b] += toa_[b][p] as f64 / 10000.0;
                    }
                    tn += 1.0;
                }
            }
        }

        // if less than 10 pixels were valid targets, skip
        if tn < 10.0 {
            dobj[o].valid = false;
            continue;
        }
        // if less than 10 pixels were valid environment, use the global one
        if en < 10.0 {
            for b in 0..nb {
                esum[b] = etoa[b] as f64;
            }
            en = 1.0;
        }

        let d = &mut dobj[o];
        // mean illumination angle, sky view and elevation
        d.cosi = (csum[o] / d.k as f64) as f32;
        d.sky = (ssum[o] / d.k as f64) as f32;
        d.z = (zsum[o] / d.k as f64) as f32;
        // MOD elevation correction factor
        d.hr = mod_elev_factor(d.z);
        // target centroid
        d.i = (d.imax + d.imin) / 2;
        d.j = (d.jmax + d.jmin) / 2;
        // target centroid in the coarse grid (convert_brick_ji2jip)
        d.e = (d.i as f64 * toa.grid.res / atc.xy_aod.res).floor() as i32;
        d.f = (d.j as f64 * toa.grid.res / atc.xy_aod.res).floor() as i32;
        d.g = d.e * atc.xy_aod.nx as i32 + d.f;
        let g = d.g as usize;
        // sun angles
        d.ms = atc.xy_sun.bands[C_ZEN][g];
        d.mv = atc.xy_view.bands[C_ZEN][g];
        d.razi = atc.xy_view.bands[AZI][g] - atc.xy_sun.bands[AZI][g];
        // target and environment reflectance
        for b in 0..nb {
            if !aod_bands[b] {
                continue;
            }
            d.etoa[b] = (esum[b] / en) as f32;
            d.ttoa[b] = (tsum[b] / tn) as f32;
        }
    }
    Ok(dobj)
}

// ---------------------------------------------------------------------------------------------------------------
// the spectral libraries
// ---------------------------------------------------------------------------------------------------------------

/// A spectral library (`speclib_t`): `n` spectra of `nb` bands.
#[derive(Clone, Debug)]
pub struct SpecLib {
    pub n: usize,
    pub s: Vec<Vec<f32>>,
}

/// `water_lib`: the water spectra of FORCE (`_AERO_WATERLIB_`, 26 spectra on 410-900 nm) convolved with the relative
/// spectral response (`rsr_band[b]`, the row of `_RSR_` of band `b`) of each band.
pub fn water_lib(nb: usize, rsr_band: &[usize]) -> SpecLib {
    let n = tab::AERO_WATERLIB_N;
    let wdim = tab::AERO_WATERLIB_W;
    let mut s = vec![vec![0.0f32; nb]; n];
    for (i, row) in s.iter_mut().enumerate() {
        let spec = &tab::AERO_WATERLIB[i * wdim..(i + 1) * wdim];
        for (b, out) in row.iter_mut().enumerate() {
            let (start, rsr) = tab::RSR[rsr_band[b]];
            let mut v = 0.0f64;
            let mut sum = 0.0f64;
            // the response is stored on its non-zero span only; the zeros add nothing
            for (k, &r) in rsr.iter().enumerate() {
                let w = start + k;
                if w >= wdim {
                    break;
                }
                v += (r * spec[w]) as f64; // float * float
                sum += r as f64;
            }
            if sum > 0.0 {
                *out = (v / sum) as f32;
            }
        }
    }
    SpecLib { n, s }
}

/// `land_lib`: FORCE's land library is unavailable in this version ("to be updated"): one spectrum of zeros.
pub fn land_lib(nb: usize) -> SpecLib {
    SpecLib { n: 1, s: vec![vec![0.0; nb]] }
}

/// `veg_lib`: the vegetation spectra of FORCE: a grid of blue/red reflectance (1% to 5% in steps of 0.5%) and green
/// reflectance above it (1% to 5% in steps of 0.5%), on the bands `blue`, `green` and `red`.
pub fn veg_lib(nb: usize, blue: usize, green: usize, red: usize) -> SpecLib {
    let mut s = Vec::new();
    // `float rb, g; for (rb=0.01; rb<=0.05; rb+=0.005) for (g=0.01; g<=0.05; g+=0.005)`
    let mut rb = 0.01f64 as f32;
    while rb as f64 <= 0.05 {
        let mut g = 0.01f64 as f32;
        while g as f64 <= 0.05 {
            let mut spec = vec![0.0f32; nb];
            spec[blue] = rb;
            spec[green] = rb + g;
            spec[red] = rb;
            s.push(spec);
            g = (g as f64 + 0.005) as f32;
        }
        rb = (rb as f64 + 0.005) as f32;
    }
    SpecLib { n: s.len(), s }
}

// ---------------------------------------------------------------------------------------------------------------
// aod_lib_to_target
// ---------------------------------------------------------------------------------------------------------------

/// `aod_lib_to_target`: for one target, the AOD at which the target's TOA reflectance and each atmospherically
/// contaminated reference spectrum of `lib` match best, per band (`aodest[w][b]`), found by increasing the
/// contamination from -0.2 to 3 in steps of 0.001 (a float accumulation, as in the C) until the environment
/// reflectance becomes unreasonable. Entries that are not written (a band that is not used for the AOD, or a loop
/// that stops at once) keep the value they had. `res` is the pixel size (m), `multi` selects the multiple
/// scattering path reflectance.
pub fn aod_lib_to_target(
    atc: &Atc,
    res: f64,
    multi: bool,
    dobj: &DarkObj,
    lib: &SpecLib,
    kind: TargetType,
    aodest: &mut [Vec<f32>],
) -> Result<(), AodError> {
    let g = dobj.g as usize;
    let nb = atc.xy_aod.nb();
    let ms = atc.xy_sun.bands[C_ZEN][g];
    let mv = atc.xy_view.bands[C_ZEN][g];
    let pr = atc.xy_pr.bands[0][g];
    let pa = atc.xy_pa.bands[0][g];
    let fresnel = atc.xy_fresnel.bands[0][g];
    atc.xy_aod.find_domain("BLUE").ok_or(AodError::NoDomain("BLUE"))?;
    atc.xy_aod.find_domain("GREEN").ok_or(AodError::NoDomain("GREEN"))?;
    atc.xy_aod.find_domain("RED").ok_or(AodError::NoDomain("RED"))?;
    let sw2 = atc.xy_aod.find_domain("SWIR2").ok_or(AodError::NoDomain("SWIR2"))?;

    let mut bestmatch = vec![0.0f32; lib.n];

    // do separately for each band
    for b in 0..nb {
        if !atc.aod_bands[b] || b == sw2 {
            continue;
        }
        // BRDF adjustment for vegetation targets
        let brdf = atc.xy_brdf.bands[b][g];
        // initialize the best match (INT_MAX as a float)
        bestmatch.iter_mut().for_each(|v| *v = i32::MAX as f32);

        // exoatmospheric solar irradiance, ozone corrected
        let tso = atc.xy_tso.bands[b][g];
        let tvo = atc.xy_tvo.bands[b][g];
        let e0_ = atc.e0[b] * tso * tvo;
        // total gaseous transmittance
        let tg = atc.xy_tg.bands[b][g];
        // elevation scale factor for MOD
        let mut mod_ = atc.xy_mod.bands[b][g];
        let hr = atc.xy_hr.bands[0][g];
        mod_ = mod_elev_scale(mod_, hr, dobj.hr);

        // iteratively increase the AOD
        let mut aod = -0.2f64 as f32;
        while aod < 3.0 {
            // total optical depth
            let od = optical_depth(aod, mod_);
            // scattering transmittances
            let st = scatt_transmitt(aod, mod_, od, ms, mv);
            let (t, ts, tsd, tss, tvd, tvs) = (st.total, st.ts, st.tsd, st.tss, st.tvd, st.tvs);
            // global irradiance at the ground
            let eg = e0_ * tsd + e0_ * tss;
            // anisotropy index
            let k = (e0_ * tsd) / atc.e0[b];
            // topographic correction factor
            let egc = if dobj.cosi > 0.0 {
                e0_ * tsd * dobj.cosi / ms + e0_ * tss * k * dobj.cosi / ms + e0_ * tss * (1.0 - k) * dobj.sky
            } else {
                e0_ * tss * (1.0 - k) * dobj.sky
            };
            // path reflectance
            let rhop = path_ref(multi, atc.sob, aod, mod_, od, pa, pr, tsd, tvd, ms, mv);
            // spherical albedo
            let s = sphere_albedo(aod, mod_, od);

            // reasonability check
            if ((dobj.etoa[b] - rhop) as f64) < -0.05 {
                break;
            }

            // simplified reflectance environment
            let tmp = (dobj.etoa[b] - rhop) / tg;
            let rhoe = tmp / (t + s * tmp);

            // environmental weighting function, F(r)
            let km = (dobj.r as f64 * res / 1000.0 + res / 1000.0 / 2.0) as f32;
            let fr = env_weight_molecular(km);
            let fa = env_weight_aerosol(km);
            let f = env_weight(aod, mod_, fa, fr);

            // compute the TOA reference reflectance and compare to the measured TOA reflectance
            for w in 0..lib.n {
                // reference reflectance including Fresnel
                let mut rhow = if kind == TargetType::Water { (lib.s[w][b] + fresnel * tss) / ts } else { lib.s[w][b] };
                // reference reflectance including topography and BRDF
                if kind == TargetType::Shadow {
                    rhow = rhow * egc / eg;
                }
                if kind == TargetType::Vegetation {
                    rhow = rhow * egc / (brdf * eg);
                }
                // background reflectance
                let rhob = rhow * f + (1.0 - f) * rhoe;
                // computed TOA reflectance
                let rhoc = tg * (rhop + (ts * (tvd * rhow + tvs * rhob)) / (1.0 - rhob * s));
                // match similarity
                let match_ = (rhoc - dobj.ttoa[b]).abs();
                if match_ < bestmatch[w] {
                    bestmatch[w] = match_;
                    aodest[w][b] = aod;
                }
            }
            aod = (aod as f64 + 0.001) as f32;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// the fits of the AOD spectrum
// ---------------------------------------------------------------------------------------------------------------

/// `aod_linear_fit`: the Angstrom expression `a0 * wvl ** a1` fitted to the log of the AOD spectrum over the bands
/// used for the AOD (least squares of `ln aod` on `ln wvl`).
pub fn aod_linear_fit(atc: &Atc, logaod: &[f32]) -> (f32, f32) {
    let nb = atc.xy_aod.nb();
    let (mut xsum, mut ysum) = (0.0f32, 0.0f32);
    let mut k = 0i32;
    // mean
    for b in 0..nb {
        if !atc.aod_bands[b] {
            continue;
        }
        xsum += atc.lwvl[b];
        ysum += logaod[b];
        k += 1;
    }
    let xm = xsum / k as f32;
    let ym = ysum / k as f32;
    // covariance and variance
    let (mut csum, mut vsum) = (0.0f32, 0.0f32);
    for b in 0..nb {
        if !atc.aod_bands[b] {
            continue;
        }
        csum += (atc.lwvl[b] - xm) * (logaod[b] - ym);
        vsum += (atc.lwvl[b] - xm) * (atc.lwvl[b] - xm);
    }
    let cv = csum / k as f32;
    let vr = vsum / k as f32;
    let slope = cv / vr;
    let inter = ym - slope * xm;
    (((inter as f64).exp()) as f32, slope)
}

/// `aod_polynomial_minimizer`: the RMSE of the second order polynomial `a0 + a1 x + a2 x^2` (`x` the log of the
/// wavelength) against the log of the AOD; `p` holds the `(ln wvl, ln aod)` pairs as floats (FORCE keeps them in a
/// float array).
fn aod_polynomial_minimizer(v: &[f64], p: &[(f32, f32)]) -> f64 {
    let a0 = v[0] as f32;
    let a1 = v[1] as f32;
    let a2 = v[2] as f32;
    let nb = p.len();
    let mut tmp = 0.0f64;
    for &(lw, la) in p {
        let lwvl = lw as f64;
        let laod = la as f64;
        let y = a0 as f64 + a1 as f64 * lwvl + a2 as f64 * lwvl * lwvl;
        tmp += (y - laod) * (y - laod);
    }
    (tmp / (nb as f32) as f64).sqrt()
}

/// `aod_polynomial_fit`: the modified Angstrom expression `a0 * wvl ** (a1 + a2 ln wvl)`, which accounts for
/// curvature in the AOD spectrum, fitted to the log of the AOD spectrum (the RMSE minimized by the Nelder-Mead
/// simplex `nmsimplex2` of GSL from zero, steps 0.01, 0.01, 0.02, until its size is under 0.01 or 100 iterations).
/// `naod` is the number of bands used for the AOD. Returns `(a0, a1, a2)`.
pub fn aod_polynomial_fit(atc: &Atc, naod: usize, logaod: &[f32]) -> (f32, f32, f32) {
    let nb = atc.xy_aod.nb();
    // (the C stores `naod` and reads that many pairs; `naod` is the number of AOD bands, as here)
    let mut pairs: Vec<(f32, f32)> = Vec::with_capacity(naod);
    for b in 0..nb {
        if !atc.aod_bands[b] {
            continue;
        }
        pairs.push((atc.lwvl[b], logaod[b]));
    }
    let mut f = |v: &[f64]| aod_polynomial_minimizer(v, &pairs);
    let x0 = [0.0f64; 3];
    let step = [0.01f64, 0.01, 0.02];
    let mut x = x0.to_vec();
    // (the C aborts here if the function is not finite at the start: only a non-finite log AOD does that)
    if let Ok(mut s) = NmSimplex::set(&mut f, &x0, &step) {
        let mut iter = 0;
        loop {
            iter += 1;
            if !s.iterate(&mut f) {
                break;
            }
            // gsl_multimin_test_size(size, 1e-2): success if the size is under the tolerance
            if s.size < 1e-2 || iter >= 100 {
                break;
            }
        }
        x = s.x.clone();
    }
    (x[0].exp() as f32, x[1] as f32, x[2] as f32)
}

// ---------------------------------------------------------------------------------------------------------------
// aod_from_target
// ---------------------------------------------------------------------------------------------------------------

/// `aod_from_target`: estimate the AOD over each valid target of `dobj`, one object at a time. A reference library is
/// matched with each target spectrum by [`aod_lib_to_target`] (the band-wise AOD that matches best); the AOD is
/// propagated through the spectrum by fitting a 2nd order polynomial (King and Byrne) or, if that fails the
/// reasonability checks (and for vegetation, always), a linear function (Angstrom) to the log of the AOD and the log of
/// the wavelengths; the spectrum with the highest R2 (over 0.1) wins. Targets without a successful fit are marked not
/// valid. Returns the number of successful targets. For the vegetation type the bands used for the AOD are reset (in
/// `atc.aod_bands`) to the blue, green, red and SWIR2 bands, as in FORCE. `rsr_band` is `meta->cal[b].rsr_band`, `res`
/// the pixel size (m). The shadow type is disabled in FORCE: 0 and nothing is touched.
pub fn aod_from_target(
    atc: &mut Atc,
    rsr_band: &[usize],
    res: f64,
    multi: bool,
    dobj: &mut [DarkObj],
    kind: TargetType,
) -> Result<i32, AodError> {
    let nb = atc.xy_aod.nb();
    let ultrablue = atc.xy_aod.find_domain("ULTRABLUE");
    let blue = atc.xy_aod.find_domain("BLUE").ok_or(AodError::NoDomain("BLUE"))?;
    let green = atc.xy_aod.find_domain("GREEN").ok_or(AodError::NoDomain("GREEN"))?;
    let red = atc.xy_aod.find_domain("RED").ok_or(AodError::NoDomain("RED"))?;
    let sw2 = atc.xy_aod.find_domain("SWIR2").ok_or(AodError::NoDomain("SWIR2"))?;

    // shadow type is disabled until further notice
    if kind == TargetType::Shadow {
        return Ok(0);
    }

    // initialize the spectral library
    let lib = match kind {
        TargetType::Water => water_lib(nb, rsr_band),
        TargetType::Shadow => land_lib(nb),
        TargetType::Vegetation => {
            let lib = veg_lib(nb, blue, green, red);
            atc.aod_bands.iter_mut().for_each(|v| *v = false);
            atc.aod_bands[blue] = true;
            atc.aod_bands[green] = true;
            atc.aod_bands[red] = true;
            atc.aod_bands[sw2] = true;
            lib
        }
    };
    let naod = atc.aod_bands.iter().filter(|&&v| v).count();

    // allocate memory (one set per thread in FORCE: here the one thread's, kept from object to object)
    let mut aodest = vec![vec![0.0f32; nb]; lib.n]; // estimated AOD, band-wise
    let mut aodlog = vec![vec![0.0f32; nb]; lib.n]; // log of the estimated AOD
    let mut aodfit = vec![vec![0.0f32; nb]; lib.n]; // fitted AOD
    let mut k = 0i32;
    let mut err = 0;
    // `coef` and `fit` are private variables of the C's parallel region, never reset between objects
    let mut coef = [0.0f32; 3];
    let mut fit;

    for o in 0..dobj.len() {
        if !dobj[o].valid {
            continue;
        }
        let mut coefbest = [0.0f32; 3];

        // estimate the band-wise AOD that matches best with each pair of target and library reflectance
        if aod_lib_to_target(atc, res, multi, &dobj[o], &lib, kind, &mut aodest).is_err() {
            err += 1;
            continue;
        }

        // fit an aerosol model to each pair of target and library spectrum
        let mut wbest: i32 = -1;
        let mut rsqbest = 0.0f32;
        let mut fitbest = 0;
        for w in 0..lib.n {
            // skip the spectrum if any estimated AOD < 0
            let mut valid = true;
            for b in 0..nb {
                if !atc.aod_bands[b] {
                    continue;
                }
                if aodest[w][b] < 0.0 {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }
            // if vegetation target and red AOD > blue AOD, skip
            if kind == TargetType::Vegetation && aodest[w][red] > aodest[w][blue] {
                continue;
            }
            // fix SWIR2 to a low value (derived from AERONET analysis)
            aodest[w][sw2] = 0.0172f64 as f32;
            // if blue AOD < assumed SWIR2 AOD, skip
            if aodest[w][blue] <= aodest[w][sw2] {
                continue;
            }
            // if ultrablue AOD < blue AOD, skip
            if let Some(u) = ultrablue {
                if atc.aod_bands[u] && aodest[w][u] < aodest[w][blue] {
                    continue;
                }
            }

            // compute the logarithm of the AOD and the mean of the AOD over all bands
            let mut mean = 0.0f32;
            for b in 0..nb {
                if !atc.aod_bands[b] {
                    continue;
                }
                // if any AOD < assumed SWIR2 AOD, force all following bands to this value
                if aodest[w][b] <= aodest[w][sw2] {
                    for bb in b..nb {
                        if !atc.aod_bands[bb] {
                            continue;
                        }
                        aodest[w][bb] = aodest[w][sw2];
                    }
                }
                aodlog[w][b] = (aodest[w][b] as f64).ln() as f32;
                mean += aodest[w][b];
            }
            mean /= naod as f32;

            // fit the polynomial AOD model (King and Byrne)
            if kind != TargetType::Vegetation {
                fit = 0;
                let (a0, a1, a2) = aod_polynomial_fit(atc, naod, &aodlog[w]);
                coef = [a0, a1, a2];
                for b in 0..nb {
                    aodfit[w][b] = power_law(coef[0], atc.wvl[b], coef[1] + coef[2] * atc.lwvl[b]);
                }
                // some reasonability checks
                valid = fit_is_reasonable(&aodfit[w]);
                if coef[0] < 0.0 || coef[0] as f64 > 0.3 {
                    valid = false;
                }
                if coef[1] > 0.0 || coef[1] < -3.0 {
                    valid = false;
                }
                if coef[2] > 0.0 {
                    valid = false;
                }
            } else {
                fit = 0;
                valid = false;
            }

            // if the polynomial fit was not successful or if we have a vegetation target, fit the linear AOD model
            // (Angstrom)
            if !valid {
                fit = 1;
                coef[2] = 0.0;
                let (a0, a1) = aod_linear_fit(atc, &aodlog[w]);
                coef[0] = a0;
                coef[1] = a1;
                for b in 0..nb {
                    aodfit[w][b] = power_law(coef[0], atc.wvl[b], coef[1]);
                }
                // some reasonability checks
                valid = fit_is_reasonable(&aodfit[w]);
                if coef[0] < 0.0 || coef[0] as f64 > 0.3 {
                    valid = false;
                }
                if coef[1] > 0.0 || coef[1] < -3.0 {
                    valid = false;
                }
            }

            // skip if neither fit was successful
            if !valid {
                continue;
            }

            // compute R2 of the fit, and find the best spectrum
            let (mut vres, mut vaod) = (0.0f32, 0.0f32);
            for b in 0..nb {
                if !atc.aod_bands[b] {
                    continue;
                }
                vres += (aodest[w][b] - aodfit[w][b]) * (aodest[w][b] - aodfit[w][b]);
                vaod += (aodest[w][b] - mean) * (aodest[w][b] - mean);
            }
            let rsq = (1.0f64 - (vres / vaod) as f64) as f32;
            if rsq > rsqbest {
                rsqbest = rsq;
                wbest = w as i32;
                fitbest = fit;
                coefbest = coef;
            }
        }

        // if no valid fit, skip
        if wbest < 0 || (rsqbest as f64) < 0.1 {
            dobj[o].valid = false;
            continue;
        }
        let wb = wbest as usize;
        // copy the best AOD
        for b in 0..nb {
            dobj[o].aod[b] = aodfit[wb][b];
            dobj[o].est[b] = aodest[wb][b];
        }
        // copy the coefficients, R2, the spectrum ID and the type of the fit
        dobj[o].coef = coefbest;
        dobj[o].rsq = rsqbest;
        dobj[o].lib_id = wbest;
        dobj[o].ang_fit = fitbest;
        // number of successful targets
        k += 1;
    }
    if err > 0 {
        return Err(AodError::Estimation);
    }
    Ok(k)
}

/// `coef0 * pow(wvl, exponent)` with the C's types: a float coefficient times a double power, stored as a float.
#[inline]
fn power_law(coef0: f32, wvl: f32, exponent: f32) -> f32 {
    (coef0 as f64 * (wvl as f64).powf(exponent as f64)) as f32
}

/// The reasonability checks of the fitted AOD spectrum: no increase over 0.01 between neighbouring bands, no two
/// neighbouring values below -0.01.
fn fit_is_reasonable(fit: &[f32]) -> bool {
    let mut valid = true;
    for b in 1..fit.len() {
        if fit[b] as f64 > fit[b - 1] as f64 && fit[b] as f64 > 0.01 && fit[b - 1] as f64 > 0.01 {
            valid = false;
        }
        if (fit[b] as f64) < -0.01 && (fit[b - 1] as f64) < -0.01 {
            valid = false;
        }
    }
    valid
}

// ---------------------------------------------------------------------------------------------------------------
// aod_elev_dependency
// ---------------------------------------------------------------------------------------------------------------

/// `aod_elevation_minimizer`: the absolute slope of the regression of `aod / exp(-z / x)` on `z`, for a scale height
/// `x`: zero where the AOD of the targets, scaled to sea level with the scale height `x`, does not depend on the
/// elevation.
pub fn aod_elevation_minimizer(x: f64, z: &[f64], aod: &[f64]) -> f64 {
    let n = z.len();
    let (mut xsum, mut ysum) = (0.0f32, 0.0f32);
    // mean
    for k in 0..n {
        xsum = (xsum as f64 + z[k]) as f32;
        ysum = (ysum as f64 + aod[k] / (-z[k] / x).exp()) as f32;
    }
    let xm = xsum / n as f32;
    let ym = ysum / n as f32;
    // covariance and variance
    let (mut csum, mut vsum) = (0.0f32, 0.0f32);
    for k in 0..n {
        csum = (csum as f64 + (z[k] - xm as f64) * (aod[k] / (-z[k] / x).exp() - ym as f64)) as f32;
        vsum = (vsum as f64 + (z[k] - xm as f64) * (z[k] - xm as f64)) as f32;
    }
    let cv = csum / n as f32;
    let vr = vsum / n as f32;
    let slope = cv / vr;
    (slope as f64).abs()
}

/// `aod_elev_dependency`: the elevation dependency of the AOD and the elevation correction factors. The scale height
/// of the aerosols (`atc.hp`, 1000 for an almost flat AOD) is found by minimizing [`aod_elevation_minimizer`] over
/// the green-band AOD (`green` the band) and the elevation of all valid targets, with Brent's method of GSL ([0.1,
/// 1000], start 50, tolerance 0.001, at most 100 iterations; under 1.2 it is set to 1.2), if there are more than 3
/// targets (`dark.nwat + nshd + nveg`), the DEM varies by more than 0.1 and so do the elevations of the targets. Sets
/// `atc.ha` (the scene average), `atc.xy_ha` (each cell with view data) and the `ha` of every valid target.
pub fn aod_elev_dependency(atc: &mut Atc, dark: &mut DarkTargets, green: usize) -> Result<(), AodError> {
    let n = dark.nwat + dark.nshd + dark.nveg;
    // almost flat AOD
    atc.hp = 1000.0;

    if n > 3 && ((atc.dem.max - atc.dem.min) as f64) > 0.1 {
        let n = n as usize;
        let mut zs = vec![0.0f64; n];
        let mut aods = vec![0.0f64; n];
        let mut k = 0usize;
        let mut zmin = i32::MAX as f32;
        let mut zmax = i32::MIN as f32;
        for list in [&dark.wat, &dark.shd, &dark.veg] {
            for o in list.iter() {
                if !o.valid {
                    continue;
                }
                if k >= n {
                    return Err(AodError::Shape("more valid targets than nwat + nshd + nveg".into()));
                }
                zs[k] = o.z as f64;
                if zs[k] < zmin as f64 {
                    zmin = zs[k] as f32;
                }
                if zs[k] > zmax as f64 {
                    zmax = zs[k] as f32;
                }
                aods[k] = o.aod[green] as f64;
                k += 1;
            }
        }

        if (zmax - zmin) as f64 > 0.1 {
            // initialize the method and iterate
            let mut minim = 50.0f64;
            let mut f = |x: f64| aod_elevation_minimizer(x, &zs, &aods);
            // (EINVAL, or a non-finite value: no minimum found, use ~flat AOD)
            if let Ok(mut s) = Brent::set(&mut f, minim, 0.1, 1000.0) {
                let mut iter = 0;
                loop {
                    iter += 1;
                    s.iterate(&mut f).map_err(|_| AodError::Gsl("the AOD elevation minimizer failed"))?;
                    minim = s.x_minimum;
                    let status = gsl_brent::test_interval(s.x_lower, s.x_upper, 0.001, 0.0)
                        .map_err(|_| AodError::Gsl("bad interval"))?;
                    if !(status == IntervalTest::Continue && iter < 100) {
                        break;
                    }
                }
                // unreasonably strong elevation dependency
                if minim < 1.2 {
                    minim = 1.2;
                }
                atc.hp = minim as f32;
            }
        }
    }

    // AOD scaling factor for the scene average
    atc.ha = aod_elev_factor(atc.dem.avg, atc.hp);

    // AOD scaling factor for the grid cells
    let ng = atc.xy_aod.ncell();
    for g in 0..ng {
        if atc.xy_view.is_nodata(0, g) {
            continue;
        }
        let dem = atc.xy_dem.bands[0][g];
        let ha = aod_elev_factor(dem * atc.dem.step + atc.dem.min + atc.dem.step / 2.0, atc.hp);
        atc.xy_ha.bands[0][g] = ha;
    }

    // AOD scaling factor for the dark targets
    for list in [&mut dark.wat, &mut dark.shd, &mut dark.veg] {
        for o in list.iter_mut() {
            if !o.valid {
                continue;
            }
            o.ha = aod_elev_factor(o.z, atc.hp);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// aod_map, interpolate_aod_map, compile_aod
// ---------------------------------------------------------------------------------------------------------------

/// `interpolate_aod_map`: the average AOD maps `map_aod` (`nb` bands of `ng` cells, zero where a cell has no
/// estimate) are interpolated to every cell by inverse distance weighting (GDAL's `GDALGridCreate`, algorithm
/// `"invdist"` with its defaults, over the cells that have a green-band estimate: [`crate::gdal_grid`]) and lowpass
/// filtered (a 3 x 3 mean over the interpolation, in the cells with view data) into `atc.xy_aod`; `atc.aod` is the mean
/// of each band over the cells with an AOD over 0.
fn interpolate_aod_map(atc: &mut Atc, map_aod: &[Vec<f32>]) -> Result<(), AodError> {
    let nb = atc.xy_aod.nb();
    let (nf, ne) = (atc.xy_aod.nx, atc.xy_aod.ny);
    let green = atc.xy_aod.find_domain("GREEN").ok_or(AodError::NoDomain("GREEN"))?;
    if atc.aod.len() != nb {
        atc.aod = vec![0.0; nb];
    }

    // the cells with an AOD estimate
    let mut aod_x = Vec::new();
    let mut aod_y = Vec::new();
    let mut aod_z: Vec<Vec<f64>> = vec![Vec::new(); nb];
    let mut g = 0;
    for e in 0..ne {
        for f in 0..nf {
            if map_aod[green][g] > 0.0 {
                for b in 0..nb {
                    aod_z[b].push(map_aod[b][g] as f64);
                }
                aod_x.push(f as f64);
                aod_y.push(e as f64);
            }
            g += 1;
        }
    }

    // interpolate and smooth AOD
    for b in 0..nb {
        let interpol =
            crate::gdal_grid::invdist_grid(&aod_x, &aod_y, &aod_z[b], 0.0, (nf - 1) as f64, 0.0, (ne - 1) as f64, nf, ne);

        let (mut sum, mut num) = (0.0f64, 0.0f64);
        // lowpass filter
        let mut g = 0;
        for e in 0..ne {
            for f in 0..nf {
                let cell = g;
                g += 1;
                if atc.xy_view.bands[crate::atc::ZEN][cell] < 0.0 {
                    continue;
                }
                let (mut gsum, mut gnum) = (0.0f64, 0.0f64);
                for ee in -1i64..=1 {
                    for ff in -1i64..=1 {
                        let (r, c) = (e as i64 + ee, f as i64 + ff);
                        if r < 0 || c < 0 || r >= ne as i64 || c >= nf as i64 {
                            continue;
                        }
                        gsum += interpol[r as usize * nf + c as usize];
                        gnum += 1.0;
                    }
                }
                let aod = if gnum > 0.0 { (gsum / gnum) as f32 } else { 0.0 };
                atc.xy_aod.bands[b][cell] = aod;
                if aod > 0.0 {
                    sum += aod as f64;
                    num += 1.0;
                }
            }
        }
        atc.aod[b] = if num > 0.0 { (sum / num) as f32 } else { 0.0 };
    }
    Ok(())
}

/// `aod_map`: the continuous AOD field at the resolution of the coarse grid. The AOD of each valid target (water,
/// shadow, vegetation), scaled to the elevation of the scene average with the target's and the scene's elevation
/// factors (`Ha`), is averaged per grid cell weighted by the squared R2 of its fit; the cells without a target are
/// marked in `atc.xy_interp` and filled by [`interpolate_aod_map`], which also sets `atc.aod`.
fn aod_map(atc: &mut Atc, dark: &DarkTargets) -> Result<(), AodError> {
    let nb = atc.xy_aod.nb();
    let ng = atc.xy_aod.ncell();
    let mut map = vec![vec![0.0f32; ng]; nb];
    let mut weight = vec![0.0f32; ng];

    // add the water, shade and vegetation measurements to the map
    for list in [&dark.wat, &dark.shd, &dark.veg] {
        for o in list.iter() {
            if !o.valid {
                continue;
            }
            let g = o.g as usize;
            if o.g < 0 || g >= ng || o.aod.len() != nb {
                return Err(AodError::Shape("a target lies outside the coarse grid".into()));
            }
            let w = o.rsq * o.rsq;
            for b in 0..nb {
                let aod = aod_elev_scale(o.aod[b], o.ha, atc.ha);
                map[b][g] += aod * w;
            }
            weight[g] += w;
        }
    }

    // compute the AOD average for the coarse grid cells
    for g in 0..ng {
        for b in 0..nb {
            if weight[g] > 0.0 {
                map[b][g] /= weight[g];
            } else {
                atc.xy_interp.bands[0][g] = 1.0;
            }
        }
    }

    // interpolate the AOD map
    interpolate_aod_map(atc, &map)
}

/// The parameters of `compile_aod` that FORCE reads from its parameter file, its metadata and the image: whether to
/// estimate the AOD (`doaod`), the directory of the AOD files (`d_aod`; `None` for `"NULL"`), the scene centre in
/// geographic coordinates and the day of the year (for [`aodfileread`]), `meta->cal[b].rsr_band` of each band, and
/// whether the objects are matched with the multiple scattering (`domulti`).
#[derive(Clone, Debug)]
pub struct AodParams<'a> {
    pub doaod: bool,
    pub dir: Option<&'a str>,
    pub center_lon: f64,
    pub center_lat: f64,
    pub doy: i32,
    pub rsr_band: &'a [usize],
    pub multi: bool,
}

/// `compile_aod`: the entry point of the AOD estimation. The fallback is the global constant or the spectrum of the
/// AOD file ([`aodfileread`]); without `par.doaod` that is `atc.aod`. Else the bands of the AOD estimation are the
/// visible ones and the SWIR2 (added to `atc.aod_bands`, which is not cleared), the dark targets are extracted from the
/// image ([`extract_dark_target`], water, shadow, vegetation), the AOD is estimated per target ([`aod_from_target`]),
/// the elevation dependency is estimated ([`aod_elev_dependency`]) and, if there is any valid target, the AOD map is
/// compiled (`atc.xy_aod`, `atc.xy_interp`, `atc.aod`, `atc.aodmap = true`); else the fallback is `atc.aod` and
/// `atc.aodmap` is false. Returns the targets (FORCE frees them; `nwat`, `nshd` and `nveg` count the valid ones; empty
/// without `doaod`). `top` (FORCE's `top_t`) is an input until `compile_topography` is ported. `toa` and `qai` are on
/// the image grid. The message FORCE prints at the end is not here.
pub fn compile_aod(atc: &mut Atc, par: &AodParams, toa: &Toa, qai: &[i16], top: &Top) -> Result<DarkTargets, AodError> {
    let nb = toa.bands.len();
    let res = toa.grid.res;
    let green = toa.find_domain("GREEN").ok_or(AodError::NoDomain("GREEN"))?;
    let sw2 = toa.find_domain("SWIR2").ok_or(AodError::NoDomain("SWIR2"))?;

    // read the external AOD file
    let aod_lut = aodfileread(atc, par.dir, par.center_lon, par.center_lat, par.doy)?;
    atc.aod = vec![0.0; atc.xy_aod.nb()];

    if !par.doaod {
        for b in 0..nb {
            atc.aod[b] = aod_lut[b];
        }
        return Ok(DarkTargets::default());
    }

    // estimate AOD from the image: all visible bands and the SWIR2
    if atc.aod_bands.len() != nb || atc.wvl.len() != nb {
        return Err(AodError::Shape("the atc needs nb bands".into()));
    }
    for b in 0..nb {
        if (atc.wvl[b] as f64) < 0.7 || b == sw2 {
            atc.aod_bands[b] = true;
        }
    }

    // extract the dark targets from the image and tabulate the information
    let mut dark = DarkTargets {
        wat: extract_dark_target(atc, toa, qai, top, TargetType::Water)?,
        shd: extract_dark_target(atc, toa, qai, top, TargetType::Shadow)?,
        veg: extract_dark_target(atc, toa, qai, top, TargetType::Vegetation)?,
        ..Default::default()
    };
    // object-based estimation of AOD
    dark.nwat = aod_from_target(atc, par.rsr_band, res, par.multi, &mut dark.wat, TargetType::Water)?;
    dark.nshd = aod_from_target(atc, par.rsr_band, res, par.multi, &mut dark.shd, TargetType::Shadow)?;
    dark.nveg = aod_from_target(atc, par.rsr_band, res, par.multi, &mut dark.veg, TargetType::Vegetation)?;

    // estimate the elevation dependency
    aod_elev_dependency(atc, &mut dark, green)?;

    // compile the AOD map or use the fallback
    if dark.nwat + dark.nshd + dark.nveg > 0 {
        aod_map(atc, &dark)?;
        atc.aodmap = true;
    } else {
        for b in 0..nb {
            atc.aod[b] = aod_lut[b];
        }
        atc.aodmap = false;
    }
    Ok(dark)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atc::{DemStats, N_ELEV};
    use crate::gas::SceneGrid;
    use crate::radtran::RSR_START_SEN2A;

    const NB: usize = 13;
    const NX: usize = 60;
    const NY: usize = 45;
    const CS: usize = 15;
    const GNX: usize = 4;
    const GNY: usize = 3;
    const DOMAIN: [&str; NB] =
        ["ULTRABLUE", "BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "VAPOR", "CIRRUS", "SWIR1", "SWIR2"];
    const WVL: [f32; NB] = [0.4375, 0.5, 0.5625, 0.65625, 0.703125, 0.75, 0.78125, 0.84375, 0.875, 0.9375, 1.375, 1.609375, 2.1875];
    /// `ln(WVL)` as C's `float` (bits).
    const LWVL: [u32; NB] = [
        0xbf53a135, 0xbf317218, 0xbf134b11, 0xbed7a94b, 0xbeb45642, 0xbe934b11, 0xbe7cc8e3, 0xbe2dfa03, 0xbe08bc74, 0xbd842cc6,
        0x3ea30c5e, 0x3ef3a213, 0x3f4862eb,
    ];
    const E0: [f32; NB] = [1920.0, 1984.0, 1856.0, 1536.0, 1408.0, 1280.0, 1216.0, 1088.0, 960.0, 928.0, 360.0, 240.0, 90.0];
    const MODB: [f32; NB] = [
        0.25, 0.1875, 0.125, 0.0625, 0.0390625, 0.03125, 0.0234375, 0.015625, 0.01171875, 0.0078125, 0.001953125, 0.0009765625,
        0.00048828125,
    ];
    // the TOA reflectance (x 10000) of a water, a vegetation and a land surface through the forward model of the radiative
    // transfer (one cell, AOD 0.15 at 550 nm); the scene adds a pattern of a few counts
    const T_W: [i16; NB] = [1459, 1332, 1079, 637, 470, 376, 324, 262, 232, 105, 102, 83, 54];
    const T_V: [i16; NB] = [1369, 1105, 1079, 620, 1051, 1993, 2860, 3135, 3245, 1691, 182, 1424, 323];
    const T_L: [i16; NB] = [1733, 1603, 1564, 1624, 2174, 2505, 2772, 2867, 2974, 1545, 981, 3323, 2607];

    struct Tiny {
        atc: Atc,
        toa: Toa,
        qai: Vec<i16>,
        top: Top,
        rsr: Vec<usize>,
    }

    /// The scene of the C harness (`gen_tiny`): 60 x 45 pixels of 10 m, a coarse grid of 4 x 3 cells of 150 m, the last
    /// column of cells outside the swath (off, no view); water: lake A (11 x 11), lake B (11 x 12 with an island and a
    /// shadow pixel), lake C (steep, above 1 km: no target), a pond of 9 pixels, a stream 4 wide; vegetation: V1, V2 and
    /// V3 (steep: no target); cloud, shadow and off pixels in them and a cloud over land.
    fn tiny() -> Tiny {
        let gn = GNX * GNY;
        let mut atc = Atc::zeros(GNX, GNY, 150.0, NB, -9999);
        atc.xy_aod.domain = DOMAIN.iter().map(|s| s.to_string()).collect();
        atc.wvl = WVL.to_vec();
        atc.lwvl = LWVL.iter().map(|&b| f32::from_bits(b)).collect();
        atc.e0 = E0.to_vec();
        atc.aod_bands = (0..NB).map(|b| WVL[b] < 0.7 || b == 12).collect();
        atc.dem = DemStats { avg: 0.75, min: 0.0, max: 2.0, step: 0.0078125 };
        atc.sob = 0.84375;
        for g in 0..gn {
            let (gi, gj) = ((g / GNX) as f32, (g % GNX) as f32);
            atc.xy_sun.bands[C_ZEN][g] = 0.625 + 0.015625 * gi + 0.0078125 * gj;
            atc.xy_sun.bands[AZI][g] = 150.0 - g as f32;
            let out = g % GNX == GNX - 1;
            atc.xy_view.bands[C_ZEN][g] = if out { -9999.0 } else { 0.96875 - 0.0078125 * (gi + gj) };
            atc.xy_view.bands[AZI][g] = if out { -9999.0 } else { 90.0 + 4.0 * g as f32 };
            let hr = 0.96875 + g as f32 / 256.0;
            atc.xy_hr.bands[0][g] = hr;
            atc.xy_pr.bands[0][g] = 1.0625;
            atc.xy_pa.bands[0][g] = 0.5 + 0.0625 * (g % 4) as f32;
            atc.xy_fresnel.bands[0][g] = 0.03125 + 0.00390625 * (g % 3) as f32;
            atc.xy_dem.bands[0][g] = 32.0 + 40.0 * gi + 8.0 * gj;
            for b in 0..NB {
                atc.xy_tso.bands[b][g] = 0.96875 + 0.00390625 * (b % 4) as f32;
                atc.xy_tvo.bands[b][g] = 0.9765625 + 0.00390625 * (g % 2) as f32;
                atc.xy_tg.bands[b][g] = if b == 9 { 0.5 } else { 0.90625 + 0.0078125 * (b % 5) as f32 };
                atc.xy_mod.bands[b][g] = MODB[b] * hr;
                atc.xy_brdf.bands[b][g] = 1.0 + (b % 3) as f32 / 32.0;
            }
        }
        let npix = NX * NY;
        let mut cls = vec![0u8; npix];
        let mut fill = |r: std::ops::Range<usize>, c: std::ops::Range<usize>, v: u8| {
            for i in r {
                for j in c.clone() {
                    cls[i * NX + j] = v;
                }
            }
        };
        fill(2..13, 18..29, 1);
        fill(18..29, 2..14, 1);
        fill(22..25, 6..9, 0);
        fill(33..44, 33..44, 1);
        fill(3..6, 3..6, 1);
        fill(31..35, 15..30, 1);
        fill(33..44, 3..14, 2);
        fill(3..14, 33..44, 2);
        fill(20..29, 33..44, 2);
        let mut bands = vec![vec![0i16; npix]; NB];
        let mut qai = vec![0i16; npix];
        let mut top = Top { dem: vec![0; npix], slope: vec![0; npix], ill: vec![0; npix], sky: vec![0; npix] };
        for i in 0..NY {
            for j in 0..NX {
                let p = i * NX + j;
                let wig = ((i + j) % 3) as i16;
                for b in 0..NB {
                    let v = match cls[p] {
                        1 => T_W[b],
                        2 => T_V[b],
                        _ => T_L[b] + 7 * ((i * 3 + j * 5) % 11) as i16,
                    };
                    bands[b][p] = v + wig;
                }
                let mut q = 0i16;
                if j >= 45 {
                    q |= 1;
                }
                if (2..4).contains(&i) && (18..21).contains(&j) {
                    q |= 1 << 1;
                }
                if (i, j) == (19, 3) {
                    q |= 1 << 3;
                }
                if (i, j) == (7, 23) {
                    q |= 1;
                }
                if (i, j) == (35, 5) {
                    q |= 2 << 1;
                }
                if (i, j) == (5, 35) {
                    q |= 1 << 3;
                }
                if (36..41).contains(&i) && (20..31).contains(&j) {
                    q |= 3 << 1;
                    for band in bands.iter_mut() {
                        band[p] = 4500;
                    }
                }
                if q & 1 != 0 {
                    for band in bands.iter_mut() {
                        band[p] = -9999;
                    }
                }
                qai[p] = q;
                top.dem[p] = (32 + 40 * (i / CS) + 8 * (j / CS).min(GNX - 1) + (i + j) % 3) as u8;
                top.slope[p] = if (33..44).contains(&i) && (33..44).contains(&j) {
                    300
                } else if (20..29).contains(&i) && (33..44).contains(&j) {
                    3000
                } else {
                    (40 + (i * 7 + j * 3) % 90) as u16
                };
                top.ill[p] = (8000 + (i * 13 + j * 7) % 1500) as i16;
                top.sky[p] = (9500 + (i * 5 + j * 11) % 400) as u16;
            }
        }
        let domain = DOMAIN.iter().map(|s| s.to_string()).collect();
        Tiny {
            atc,
            toa: Toa { grid: SceneGrid { nx: NX, ny: NY, res: 10.0 }, bands, domain },
            qai,
            top,
            rsr: (0..NB).map(|b| RSR_START_SEN2A + b).collect(),
        }
    }

    fn bits(v: f32) -> u32 {
        v.to_bits()
    }

    /// The values the C harness printed (FORCE's aod-ll.c, GSL 2.7, one thread) for the scene above.
    #[test]
    fn values_of_the_c_harness() {
        let mut t = tiny();
        let wat = extract_dark_target(&t.atc, &t.toa, &t.qai, &t.top, TargetType::Water).unwrap();
        let veg = extract_dark_target(&t.atc, &t.toa, &t.qai, &t.top, TargetType::Vegetation).unwrap();
        let shd = extract_dark_target(&t.atc, &t.toa, &t.qai, &t.top, TargetType::Shadow).unwrap();
        assert_eq!((wat.len(), veg.len(), shd.len()), (4, 2, 0));
        let ints = |o: &DarkObj| [o.k, o.valid as i32, o.imin, o.imax, o.jmin, o.jmax, o.i, o.j, o.e, o.f, o.g, o.r];
        assert_eq!(ints(&wat[0]), [114, 1, 2, 12, 18, 28, 7, 23, 0, 1, 1, 3]); // lake A
        assert_eq!(ints(&wat[1]), [9, 0, 3, 5, 3, 5, 0, 0, 0, 0, 0, 2]); // the pond: under 10 pixels
        assert_eq!(ints(&wat[2]), [122, 1, 18, 28, 2, 13, 23, 7, 1, 0, 4, 3]); // lake B: the island and the shadow pixel
        assert_eq!(ints(&wat[3]), [56, 0, 31, 34, 16, 29, 0, 0, 0, 0, 0, 2]); // the stream: radius 2
        assert_eq!(ints(&veg[0]), [80, 1, 4, 12, 34, 42, 8, 38, 0, 2, 2, 6]);
        assert_eq!(ints(&veg[1]), [80, 1, 34, 42, 4, 12, 38, 8, 2, 0, 8, 6]);
        let flo = |o: &DarkObj| [o.z, o.cosi, o.sky, o.hr, o.ttoa[1], o.etoa[1]].map(bits);
        assert_eq!(flo(&wat[0]), [0x3ea60000, 0x3f535e25, 0x3f7aa993, 0x3f764921, 0x3e087fcc, 0x3e20c79f]);
        assert_eq!(flo(&wat[2]), [0x3f130000, 0x3f55d291, 0x3f7847ee, 0x3f6efe1d, 0x3e087fcc, 0x3e218df6]);
        assert_eq!(flo(&veg[0]), [0x3ec60000, 0x3f564a8c, 0x3f74f4c7, 0x3f747246, 0x3de28241, 0x3e1c31c4]);
        assert_eq!(flo(&veg[1]), [0x3f630000, 0x3f5ae632, 0x3f7a5508, 0x3f661d53, 0x3de28241, 0x3e1ff1b2]);
        // the invalid objects are not tabulated beyond their size
        assert_eq!((wat[1].cosi, wat[1].z, wat[3].ttoa[1]), (0.0, 0.0, 0.0));

        // the band-wise AOD that matches best, for each spectrum of the library
        let sum_bits = |a: &[Vec<f32>]| a.iter().flatten().map(|v| v.to_bits() as u64).sum::<u64>();
        let wlib = water_lib(NB, &t.rsr);
        assert_eq!((wlib.n, veg_lib(NB, 1, 2, 3).n), (26, 81));
        for (o, want_sum, want_b1) in [(0, 163181446271u64, [0x3e6c8b66u32, 0x3e25e36d, 0x3e343973]), (2, 159002343490, [0x3e839590, 0x3e3c6a9b, 0x3e4bc6c6])] {
            let mut aodest = vec![vec![0.0f32; NB]; wlib.n];
            aod_lib_to_target(&t.atc, 10.0, true, &wat[o], &wlib, TargetType::Water, &mut aodest).unwrap();
            assert_eq!(sum_bits(&aodest), want_sum);
            assert_eq!([aodest[0][1], aodest[1][1], aodest[5][1]].map(bits), want_b1);
        }
        let mut atc_veg = t.atc.clone();
        atc_veg.aod_bands = (0..NB).map(|b| matches!(b, 1 | 2 | 3 | 12)).collect();
        let vlib = veg_lib(NB, 1, 2, 3);
        for (o, want_sum) in [(0, 442601072498u64), (1, 405873465065)] {
            let mut aodest = vec![vec![0.0f32; NB]; vlib.n];
            aod_lib_to_target(&atc_veg, 10.0, true, &veg[o], &vlib, TargetType::Vegetation, &mut aodest).unwrap();
            assert_eq!(sum_bits(&aodest), want_sum);
        }

        // the AOD of each target
        let mut dark = DarkTargets { wat, shd, veg, ..Default::default() };
        dark.nwat = aod_from_target(&mut t.atc, &t.rsr, 10.0, true, &mut dark.wat, TargetType::Water).unwrap();
        dark.nshd = aod_from_target(&mut t.atc, &t.rsr, 10.0, true, &mut dark.shd, TargetType::Shadow).unwrap();
        dark.nveg = aod_from_target(&mut t.atc, &t.rsr, 10.0, true, &mut dark.veg, TargetType::Vegetation).unwrap();
        assert_eq!((dark.nwat, dark.nshd, dark.nveg), (2, 0, 2));
        // the vegetation estimation restricts the bands used
        assert_eq!(t.atc.aod_bands.iter().filter(|&&v| v).count(), 4);
        let fit = |o: &DarkObj| ([o.valid as i32, o.lib_id, o.ang_fit], [o.rsq, o.coef[0], o.coef[1], o.coef[2], o.aod[1], o.est[1]].map(bits));
        assert_eq!(fit(&dark.wat[0]), ([1, 5, 1], [0x3f2f1172, 0x3d3129cf, 0xbfb33eea, 0, 0x3de9d38a, 0x3e343973]));
        assert_eq!(fit(&dark.wat[2]), ([1, 5, 1], [0x3f37693a, 0x3d402268, 0xbfbf0b5b, 0, 0x3e072885, 0x3e4bc6c6]));
        assert_eq!(fit(&dark.wat[1]), ([0, 0, 0], [0; 6]));
        assert_eq!(fit(&dark.wat[3]), ([0, 0, 0], [0; 6]));
        assert_eq!(fit(&dark.veg[0]), ([1, 3, 1], [0x3f7fb391, 0x3d4d248c, 0xbfaf4b5d, 0, 0x3e0482e6, 0x3e051ecd]));
        assert_eq!(fit(&dark.veg[1]), ([1, 3, 1], [0x3f7f16b0, 0x3d6d0a3f, 0xbfc7262b, 0, 0x3e2e3a91, 0x3e2e1495]));

        // the elevation dependency: the interval does not enclose a minimum, the AOD is flat
        aod_elev_dependency(&mut t.atc, &mut dark, 2).unwrap();
        assert_eq!(bits(t.atc.hp), 0x447a0000);
        assert_eq!(bits(t.atc.ha), 0x3f7fcede);
        let xy: Vec<u32> = t.atc.xy_ha.bands[0].iter().map(|&v| v.to_bits()).collect();
        assert_eq!(
            xy,
            [0x3f7fef5d, 0x3f7feb44, 0x3f7fe72c, 0x3f7fe314, 0x3f7fdae4, 0x3f7fd6cc, 0x3f7fd2b4, 0x3f7fce9c, 0x3f7fc66d, 0x3f7fc255, 0x3f7fbe3e, 0x3f7fba26]
        );
        let ha: Vec<u32> = dark.wat.iter().chain(dark.veg.iter()).map(|o| o.ha.to_bits()).collect();
        assert_eq!(ha, [0x3f7feac1, 0, 0x3f7fda61, 0, 0x3f7fe6a9, 0x3f7fc5ea]);
    }

    /// What the harness does before `compile_aod`: the sun zenith band of the view brick is 30 where there is a view.
    fn with_view_zenith(mut t: Tiny) -> Tiny {
        for g in 0..GNX * GNY {
            let cz = t.atc.xy_view.bands[C_ZEN][g];
            t.atc.xy_view.bands[crate::atc::ZEN][g] = if cz < 0.0 { cz } else { 30.0 };
        }
        t
    }

    fn params<'a>(t: &'a Tiny, doaod: bool) -> AodParams<'a> {
        AodParams { doaod, dir: None, center_lon: 0.0, center_lat: 0.0, doy: 1, rsr_band: &t.rsr, multi: true }
    }

    /// `compile_aod` on the scene above: the values the C harness printed (FORCE's `compile_aod` as it is, GSL 2.7, GDAL
    /// 3.9.3, one thread).
    #[test]
    fn compile_aod_values_of_the_c_harness() {
        let mut t = with_view_zenith(tiny());
        let rsr = t.rsr.clone();
        let par = AodParams { doaod: true, dir: None, center_lon: 0.0, center_lat: 0.0, doy: 1, rsr_band: &rsr, multi: true };
        let dark = compile_aod(&mut t.atc, &par, &t.toa, &t.qai, &t.top).unwrap();
        assert_eq!((dark.nwat, dark.nshd, dark.nveg), (2, 0, 2));
        assert!(t.atc.aodmap);
        let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        // the map (and the AOD averaged over it) goes through GDAL's approximate 1/r2 (`rcpps`), whose table is the CPU
        // maker's: the bits of the harness on an Intel CPU, the values to that approximation elsewhere
        let same_or_close = |got: &[f32], want: &[u32]| {
            if crate::gdal_grid::rcp_is_intel() {
                assert_eq!(b(got), want);
            } else {
                assert_eq!(got.len(), want.len());
                for (g, &w) in got.iter().zip(want) {
                    let w = f32::from_bits(w);
                    assert!((g - w).abs() <= 2e-3 * w.abs(), "{g} vs {w}");
                }
            }
        };
        same_or_close(
            &t.atc.aod,
            &[
                0x3e25374f, 0x3e0811e2, 0x3de55694, 0x3db75797, 0x3da5deec, 0x3d970a89, 0x3d8e5ade, 0x3d7ea295, 0x3d718d7b, 0x3d5a8e8b,
                0x3cfaf5b9, 0x3cc7d4d7, 0x3c803349
            ],
        );
        assert_eq!((t.atc.hp.to_bits(), t.atc.ha.to_bits()), (0x447a0000, 0x3f7fcede));
        let bands: Vec<bool> = (0..NB).map(|b| matches!(b, 1 | 2 | 3 | 12)).collect();
        assert_eq!(t.atc.aod_bands, bands); // the vegetation estimation leaves the blue, green, red and SWIR2 bands
        // the cells outside the swath (the last column) keep no AOD; the other cells are interpolated or averaged
        same_or_close(
            &t.atc.xy_aod.bands[2],
            &[
                0x3ddc3e32, 0x3ddd8ed8, 0x3dddc3a2, 0, 0x3de7fa0d, 0x3de68414, 0x3de2c664, 0, 0x3df1f892, 0x3dede578, 0x3de757f9, 0,
            ],
        );
        same_or_close(
            &t.atc.xy_aod.bands[12],
            &[
                0x3c763d06, 0x3c7db0d1, 0x3c83067d, 0, 0x3c7bf8a9, 0x3c805623, 0x3c82e91c, 0, 0x3c80066b, 0x3c8101a5, 0x3c828c84, 0,
            ],
        );
        // the cells without a target of their own are the ones marked as interpolated (the targets are in 1 and 4)
        assert_eq!(t.atc.xy_interp.bands[0], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(b(&t.atc.xy_ha.bands[0])[..3], [0x3f7fef5d, 0x3f7feb44, 0x3f7fe72c]);
    }

    /// No target (the fallback), only invalid targets, `doaod` off and an AOD file: `atc.aod` is the fallback spectrum.
    #[test]
    fn compile_aod_without_targets_uses_the_fallback() {
        let t = with_view_zenith(tiny());
        let fallback = aodfileread(&t.atc, None, 0.0, 0.0, 1).unwrap();
        let check = |toa: &Toa, qai: &[i16], doaod: bool| {
            let mut atc = t.atc.clone();
            let dark = compile_aod(&mut atc, &params(&t, doaod), toa, qai, &t.top).unwrap();
            assert_eq!(atc.aod, fallback);
            assert!(!atc.aodmap);
            assert!(atc.xy_aod.bands.iter().flatten().all(|&v| v == 0.0));
            assert!(atc.xy_interp.bands[0].iter().all(|&v| v == 0.0));
            (dark.nwat, dark.nveg)
        };
        // land only: no candidates
        let mut land = Toa { grid: t.toa.grid, bands: t.toa.bands.clone(), domain: t.toa.domain.clone() };
        for (b, band) in land.bands.iter_mut().enumerate() {
            for (p, v) in band.iter_mut().enumerate() {
                if *v != -9999 {
                    *v = T_L[b] + (p % 5) as i16;
                }
            }
        }
        assert_eq!(check(&land, &t.qai, true), (0, 0));
        // all cloud: every candidate is invalid
        let cloud: Vec<i16> = t.qai.iter().map(|&q| q | (1 << 1)).collect();
        assert_eq!(check(&t.toa, &cloud, true), (0, 0));
        // doaod off: nothing is extracted, whatever is in the image
        assert_eq!(check(&t.toa, &t.qai, false), (0, 0));
        // the bands of the estimation are not touched without doaod
        let mut atc = t.atc.clone();
        atc.aod_bands = vec![false; NB];
        compile_aod(&mut atc, &params(&t, false), &t.toa, &t.qai, &t.top).unwrap();
        assert!(atc.aod_bands.iter().all(|&v| !v));

        // an AOD file: the line closest to the scene centre
        let dir = std::env::temp_dir().join(format!("zeit_aod_compile_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("AOD_045.txt"), "1.0 45.0 -2.5 -1.3 -0.05\n2.0 46.0 -2.2 -1.1 -0.02\n").unwrap();
        let mut atc = t.atc.clone();
        let par = AodParams { dir: dir.to_str(), center_lon: 1.9, center_lat: 45.7, doy: 45, ..params(&t, false) };
        compile_aod(&mut atc, &par, &t.toa, &t.qai, &t.top).unwrap();
        assert_eq!(atc.aod, aodfileread(&t.atc, dir.to_str(), 1.9, 45.7, 45).unwrap());
        assert_ne!(atc.aod, fallback);
        // a missing file is a failure
        let par = AodParams { doy: 46, ..par };
        assert!(matches!(compile_aod(&mut atc, &par, &t.toa, &t.qai, &t.top), Err(AodError::OpenFile(_))));
        std::fs::remove_dir_all(&dir).unwrap();

        // a TOA brick without the green or the SWIR2 domain
        let mut atc = t.atc.clone();
        let mut bad = Toa { grid: t.toa.grid, bands: t.toa.bands.clone(), domain: t.toa.domain.clone() };
        bad.domain[2] = String::new();
        assert_eq!(compile_aod(&mut atc, &params(&t, true), &bad, &t.qai, &t.top).unwrap_err(), AodError::NoDomain("GREEN"));
    }

    /// A single estimate is the whole field: the interpolation of one point is its value (to a rounding of the float
    /// division) everywhere with view data.
    #[test]
    fn the_map_of_one_cell_is_flat() {
        let mut t = with_view_zenith(tiny());
        let mut dark = DarkTargets::default();
        dark.wat.push(DarkObj { valid: true, g: 5, rsq: 0.5, ha: 1.0, aod: (0..NB).map(|b| 0.125 + 0.015625 * b as f32).collect(), ..Default::default() });
        t.atc.ha = 1.0;
        aod_map(&mut t.atc, &dark).unwrap();
        for b in 0..NB {
            let want = 0.125 + 0.015625 * b as f32;
            for g in 0..GNX * GNY {
                let got = t.atc.xy_aod.bands[b][g];
                if g % GNX == GNX - 1 {
                    assert_eq!(got, 0.0, "outside the swath"); // the zenith band is the nodata value there
                } else {
                    assert!((got - want).abs() <= want * 1e-6, "band {b} cell {g}: {got} vs {want}");
                }
            }
            assert!((t.atc.aod[b] - want).abs() <= want * 1e-6);
        }
        assert_eq!(t.atc.xy_interp.bands[0].iter().filter(|&&v| v == 0.0).count(), 1);
        // a target outside the grid is an error, not a crash
        dark.wat[0].g = 99;
        assert!(matches!(aod_map(&mut t.atc, &dark), Err(AodError::Shape(_))));
    }

    /// `aod_elev_dependency` on lists of targets; the values the C harness printed for the same inputs (Brent's method).
    #[test]
    fn the_elevation_dependency_of_the_c_harness() {
        let zs = [0.0625f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
        let a1: [u32; 8] = [0x3e758d71, 0x3e58b307, 0x3e376e99, 0x3e1b4598, 0x3e036f4e, 0x3dde83a8, 0x3dbc5ab2, 0x3d86f634];
        let a3: [u32; 8] = [0x3e80ea41, 0x3e4ddd46, 0x3e409a86, 0x3e13821d, 0x3e0a01ab, 0x3dd36379, 0x3dc5c5a1, 0x3d8036b1];
        let run = |mn: f32, mx: f32, a: &[u32; 8]| -> Vec<u32> {
            let mut atc = Atc::zeros(2, 2, 1.0, 3, -9999);
            atc.dem = DemStats { avg: 0.75, min: mn, max: mx, step: 0.0078125 };
            atc.xy_view.bands[0].copy_from_slice(&[0.96875, 0.96875, -9999.0, 0.96875]);
            atc.xy_dem.bands[0].copy_from_slice(&[16.0, 64.0, 128.0, 200.0]);
            let wat = zs.iter().zip(a).map(|(&z, &b)| DarkObj { valid: true, z, aod: vec![0.0, f32::from_bits(b), 0.0], ..Default::default() }).collect();
            let mut dark = DarkTargets { wat, nwat: 8, ..Default::default() };
            aod_elev_dependency(&mut atc, &mut dark, 1).unwrap();
            let mut v = vec![atc.hp.to_bits(), atc.ha.to_bits()];
            v.extend(atc.xy_ha.bands[0].iter().map(|x| x.to_bits()));
            v.extend(dark.wat.iter().map(|t| t.ha.to_bits()));
            v
        };
        // exponential decay with the scale height 1.5 km: found (1.4999 km)
        assert_eq!(
            run(0.0, 2.0, &a1),
            [0x3fbffeb7, 0x3f1b4513, 0x3f6aeb14, 0x3f36f40e, 0, 0x3eb43150, 0x3f758d5f, 0x3f58b2c9, 0x3f376e30, 0x3f1b4513, 0x3f036eb8, 0x3ede826a, 0x3ebc596f, 0x3e86f500]
        );
        // a flat DEM: the flat AOD, 1000
        assert_eq!(
            run(0.5, 0.5, &a1),
            [0x447a0000, 0x3f7fcede, 0x3f7fd6cc, 0x3f7fbe3e, 0, 0x3f7f78b7, 0x3f7ffbe7, 0x3f7fef9e, 0x3f7fdf3d, 0x3f7fcede, 0x3f7fbe7f, 0x3f7fae22, 0x3f7f9dc5, 0x3f7f7d0f]
        );
        // the same with noise on the AOD
        assert_eq!(
            run(0.0, 2.0, &a3),
            [0x3fba6dd5, 0x3f18f819, 0x3f6a50f9, 0x3f3520a5, 0, 0x3eaea947, 0x3f753f38, 0x3f579f6d, 0x3f359d2c, 0x3f18f819, 0x3f00d7a0, 0x3ed90aa9, 0x3eb6cf1d, 0x3e81b0ac]
        );
    }

    #[test]
    fn no_candidates_clouds_and_missing_domains() {
        let t = tiny();
        // land only: no target
        let mut land = Toa { grid: t.toa.grid, bands: t.toa.bands.clone(), domain: t.toa.domain.clone() };
        for (b, band) in land.bands.iter_mut().enumerate() {
            for (p, v) in band.iter_mut().enumerate() {
                if *v != -9999 {
                    *v = T_L[b] + (p % 5) as i16;
                }
            }
        }
        for kind in [TargetType::Water, TargetType::Vegetation] {
            assert!(extract_dark_target(&t.atc, &land, &t.qai, &t.top, kind).unwrap().is_empty());
        }
        // all cloud: the candidates exist but none is valid (a cloud pixel is neither target nor environment)
        let cloud: Vec<i16> = t.qai.iter().map(|&q| q | (1 << 1)).collect();
        let wat = extract_dark_target(&t.atc, &t.toa, &cloud, &t.top, TargetType::Water).unwrap();
        assert!(!wat.is_empty() && wat.iter().all(|o| !o.valid && o.k == 0));
        let mut atc = t.atc.clone();
        let mut wat = wat;
        let n = aod_from_target(&mut atc, &t.rsr, 10.0, true, &mut wat, TargetType::Water).unwrap();
        assert_eq!(n, 0);
        // everything off: no segments at all
        let off: Vec<i16> = t.qai.iter().map(|&q| q | 1).collect();
        assert!(extract_dark_target(&t.atc, &t.toa, &off, &t.top, TargetType::Water).unwrap().is_empty());
        // a TOA brick without the SWIR2 domain
        let mut bad = Toa { grid: t.toa.grid, bands: t.toa.bands.clone(), domain: t.toa.domain.clone() };
        bad.domain[12] = String::new();
        assert_eq!(
            extract_dark_target(&t.atc, &bad, &t.qai, &t.top, TargetType::Water).unwrap_err(),
            AodError::NoDomain("SWIR2")
        );
        // a coarse grid finer than the image, and inputs of the wrong size
        let mut fine = t.atc.clone();
        fine.xy_aod.res = 5.0;
        assert!(matches!(extract_dark_target(&fine, &t.toa, &t.qai, &t.top, TargetType::Water), Err(AodError::Shape(_))));
        assert!(matches!(extract_dark_target(&t.atc, &t.toa, &t.qai[1..], &t.top, TargetType::Water), Err(AodError::Shape(_))));
    }

    #[test]
    fn the_shadow_type_is_disabled_as_in_force() {
        let mut t = tiny();
        assert!(extract_dark_target(&t.atc, &t.toa, &t.qai, &t.top, TargetType::Shadow).unwrap().is_empty());
        let mut none: Vec<DarkObj> = Vec::new();
        assert_eq!(aod_from_target(&mut t.atc, &t.rsr, 10.0, true, &mut none, TargetType::Shadow), Ok(0));
        assert_eq!(t.atc.aod_bands.iter().filter(|&&v| v).count(), 5); // untouched
        assert_eq!(land_lib(NB).n, 1);
    }

    #[test]
    fn the_libraries() {
        let rsr: Vec<usize> = (0..NB).map(|b| RSR_START_SEN2A + b).collect();
        let w = water_lib(NB, &rsr);
        assert_eq!((w.n, w.s.len(), w.s[0].len()), (26, 26, NB));
        // blue and green have water reflectance, SWIR2 none (the library ends at 900 nm)
        assert!(w.s.iter().all(|s| s[1] > 0.0 && s[2] > 0.0 && s[12] == 0.0 && s[1].is_finite()));
        let v = veg_lib(NB, 1, 2, 3);
        assert_eq!(v.n, 81);
        // blue and red 1% to 5% in steps of 0.5%, green that much above them: the loops are in float
        assert_eq!((v.s[0][1], v.s[0][2], v.s[0][3], v.s[0][0]), (0.01, 0.02, 0.01, 0.0));
        let last = &v.s[80];
        assert!((last[1] - 0.05).abs() < 1e-6 && (last[2] - 0.10).abs() < 1e-6);
        assert!(v.s.iter().all(|s| s[1] == s[3] && s[2] > s[1]));
    }

    /// `dist_transform_` against the definition, `connectedcomponents_` on shapes.
    #[test]
    fn the_distance_transform_and_the_components() {
        let (nx, ny) = (23usize, 17usize);
        let mut state = 12345u32;
        let img: Vec<u8> = (0..nx * ny)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 24) < 40) as u8
            })
            .collect();
        let d = dist_transform(&img, nx, ny);
        for i in 0..ny {
            for j in 0..nx {
                let best = (0..ny)
                    .flat_map(|a| (0..nx).map(move |b| (a, b)))
                    .filter(|&(a, b)| img[a * nx + b] != 0)
                    .map(|(a, b)| ((a as i64 - i as i64).pow(2) + (b as i64 - j as i64).pow(2)) as f64)
                    .fold(f64::INFINITY, f64::min);
                assert_eq!(d[i * nx + j], best.sqrt() as u16, "({i}, {j})");
            }
        }
        // components: two blobs touching diagonally are one (8-connected), a ring is one, ids by first pixel
        let pic = ["11000000", "11000110", "00100111", "00000001", "01110000", "01010000", "01110000"];
        let (nx, ny) = (8, pic.len());
        let img: Vec<u8> = pic.iter().flat_map(|r| r.bytes().map(|c| (c == b'1') as u8)).collect();
        let (lab, n) = connected_components(&img, nx, ny);
        assert_eq!(n, 3);
        let at = |i: usize, j: usize| lab[i * nx + j];
        assert_eq!((at(0, 0), at(1, 1), at(2, 2)), (1, 1, 1)); // joined by the diagonal
        assert_eq!((at(1, 5), at(2, 7), at(3, 7)), (2, 2, 2));
        assert_eq!((at(4, 1), at(5, 1), at(6, 3), at(5, 2)), (3, 3, 3, 0)); // the ring, not its hole
        assert_eq!(lab.iter().filter(|&&v| v == 0).count(), img.iter().filter(|&&v| v == 0).count());
    }

    #[test]
    fn the_fits_recover_a_power_law() {
        let mut atc = Atc::zeros(1, 1, 1.0, NB, -9999);
        atc.wvl = WVL.to_vec();
        atc.lwvl = LWVL.iter().map(|&b| f32::from_bits(b)).collect();
        atc.aod_bands = (0..NB).map(|b| WVL[b] < 0.7 || b == 12).collect();
        // Angstrom: aod = 0.12 * wvl^-1.3
        let log: Vec<f32> = (0..NB).map(|b| (0.12f64.ln() - 1.3 * (WVL[b] as f64).ln()) as f32).collect();
        let (a0, a1) = aod_linear_fit(&atc, &log);
        assert!((a0 - 0.12).abs() < 1e-5 && (a1 + 1.3).abs() < 1e-5, "{a0} {a1}");
        // curved: ln aod = ln 0.1 - 1.2 ln wvl - 0.3 (ln wvl)^2
        let log: Vec<f32> = (0..NB)
            .map(|b| {
                let l = (WVL[b] as f64).ln();
                (0.1f64.ln() - 1.2 * l - 0.3 * l * l) as f32
            })
            .collect();
        let (a0, a1, a2) = aod_polynomial_fit(&atc, 5, &log);
        // the simplex stops at its size 0.01: close, not exact
        assert!((a0 - 0.1).abs() < 0.02 && (a1 + 1.2).abs() < 0.1 && (a2 + 0.3).abs() < 0.1, "{a0} {a1} {a2}");
    }

    #[test]
    fn the_elevation_minimizer_is_zero_at_the_true_scale_height() {
        let z: Vec<f64> = (1..=10).map(|k| 0.2 * k as f64).collect();
        let aod: Vec<f64> = z.iter().map(|&zz| 0.3 * (-zz / 1.7f64).exp()).collect();
        let at_truth = aod_elevation_minimizer(1.7, &z, &aod);
        assert!(at_truth < 1e-6, "{at_truth}");
        assert!(aod_elevation_minimizer(0.8, &z, &aod) > 100.0 * at_truth.max(1e-9));
        assert!(aod_elevation_minimizer(5.0, &z, &aod) > 100.0 * at_truth.max(1e-9));
    }

    #[test]
    fn the_aod_file_and_the_global_fallback() {
        let mut atc = Atc::zeros(1, 1, 1.0, 3, -9999);
        atc.wvl = vec![0.5, 0.65, 2.2];
        atc.lwvl = atc.wvl.iter().map(|&w| (w as f64).ln() as f32).collect();
        // the global median AOD of AERONET
        let fb = aodfileread(&atc, None, 0.0, 0.0, 1).unwrap();
        for (b, &v) in fb.iter().enumerate() {
            let w = atc.wvl[b] as f64;
            let want = (-2.8665f32 as f64).exp() * w.powf((-1.4787f32 + -0.0486f32 * atc.lwvl[b]) as f64);
            assert_eq!(v, want as f32);
        }
        assert!(fb[0] > fb[1] && fb[1] > fb[2]);
        let d = std::env::temp_dir().join(format!("zeit-aod-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        // two sites; a line with too few fields; doubled blanks, CRLF
        std::fs::write(d.join("AOD_045.txt"), "10.0 50.0 -2.0 -1.0 -0.1\r\nbad line\n12.0  52.0 -3.0 -1.5 0.0\n").unwrap();
        let dir = d.to_str().unwrap();
        let near_first = aodfileread(&atc, Some(dir), 10.5, 50.5, 45).unwrap();
        let near_second = aodfileread(&atc, Some(dir), 11.9, 52.1, 45).unwrap();
        let want = |c0: f32, c1: f32, c2: f32, b: usize| ((c0 as f64).exp() * (atc.wvl[b] as f64).powf((c1 + c2 * atc.lwvl[b]) as f64)) as f32;
        assert_eq!(near_first[1], want(-2.0, -1.0, -0.1, 1));
        assert_eq!(near_second[2], want(-3.0, -1.5, 0.0, 2));
        // no file for the day: an error (FORCE prints "Unable to open aod file!" and returns NULL)
        assert!(matches!(aodfileread(&atc, Some(dir), 10.0, 50.0, 46), Err(AodError::OpenFile(_))));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_atc_grid_helpers() {
        let atc = Atc::zeros(2, 2, 1.0, 3, -9999);
        assert_eq!((atc.xyz_rho_p.len(), N_ELEV), (256, 256));
        assert_eq!(atc.xy_aod.nb(), 3);
        assert!(!atc.xy_view.is_nodata(0, 0));
    }

    /// Writes the scene above in the layout of `benchmarks/vs_originals/force_aod/main.inc` (to the file named by
    /// `ZEIT_TINY_SCENE`), for the C harness that gave the values of the tests of `compile_aod`:
    /// `cargo test --release --lib write_tiny_scene -- --ignored`, then `aod_harness compile <file> <out> 1 NULL 0 0 1`.
    #[test]
    #[ignore]
    fn write_tiny_scene_for_the_c_harness() {
        use std::io::Write;
        let path = std::env::var("ZEIT_TINY_SCENE").expect("ZEIT_TINY_SCENE");
        let t = tiny();
        let mut f = std::fs::File::create(path).unwrap();
        let mut w = |b: &[u8]| f.write_all(b).unwrap();
        let (gn, npix) = (GNX * GNY, NX * NY);
        for v in [NB as i32, NX as i32, NY as i32] {
            w(&v.to_le_bytes());
        }
        w(&10.0f64.to_le_bytes());
        for v in [GNX as i32, GNY as i32] {
            w(&v.to_le_bytes());
        }
        w(&150.0f64.to_le_bytes());
        for v in [-9999i32, 1] {
            w(&v.to_le_bytes());
        }
        for d in DOMAIN {
            let mut b = [0u8; 16];
            b[..d.len()].copy_from_slice(d.as_bytes());
            w(&b);
        }
        for r in &t.rsr {
            w(&(*r as i32).to_le_bytes());
        }
        for v in &t.atc.aod_bands {
            w(&[*v as u8]);
        }
        let fs = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for v in [&t.atc.wvl[..], &t.atc.lwvl, &t.atc.e0, &[t.atc.dem.avg, t.atc.dem.min, t.atc.dem.max, t.atc.dem.step, t.atc.sob]] {
            w(&fs(v));
        }
        for v in [&t.atc.xy_sun.bands[C_ZEN], &t.atc.xy_sun.bands[AZI], &t.atc.xy_view.bands[C_ZEN], &t.atc.xy_view.bands[AZI]] {
            w(&fs(v));
        }
        for brick in [&t.atc.xy_tso, &t.atc.xy_tvo, &t.atc.xy_tg, &t.atc.xy_mod, &t.atc.xy_brdf] {
            for b in 0..NB {
                w(&fs(&brick.bands[b]));
            }
        }
        for v in [&t.atc.xy_pr, &t.atc.xy_pa, &t.atc.xy_hr, &t.atc.xy_fresnel, &t.atc.xy_dem] {
            w(&fs(&v.bands[0]));
        }
        let _ = (gn, npix);
        for b in 0..NB {
            w(&t.toa.bands[b].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        }
        w(&t.qai.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        w(&t.top.dem);
        w(&t.top.slope.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        w(&t.top.ill.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        w(&t.top.sky.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    }
}
