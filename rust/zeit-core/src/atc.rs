//! The atmospheric correction factors of one image: FORCE's `atc_t` (`atc-ll.h`) and its allocation (`atc-ll.c`:
//! `init_atc`, `init_atc_xy`, `init_atc_xyz`, `allocate_atc`; the `free_*` functions are the drop of the structure).
//! Ported from FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force).
//!
//! FORCE keeps each factor in a `brick_t`: a stack of bands, each a grid of `nx * ny` coarse cells (`atc->res` metres a
//! side: 5000 m in FORCE 3.11), or, for the factors that depend on the elevation, 256 of them (`xyz_*`, one per
//! elevation bin `z`, `_BYTE_LEN_`). Here a [`CoarseBrick`] is one such stack, and the [`Atc`] struct holds every
//! brick and parameter of `atc_t`. Where this differs from the C:
//!
//! * **The type of the bricks.** Every brick holds `f32` (the C has `small` for `xy_interp` and `int` for `xy_dem`; the
//!   values are small integers, which `f32` holds exactly, and the functions that write them truncate as `set_brick` on
//!   the C brick does).
//! * **The geography.** The `brick_t` of the C also carries the geotransformation, the projection and the date of the
//!   image, which `atc_t`'s bricks copy from the digital numbers; here the functions that need them (`sunview.rs`,
//!   `atmo.rs`) take them as a [`crate::sunview::SceneGeo`].
//! * **`Tw_lut`.** The water vapour transmittance table of `atc_t` is [`crate::gas::WvLut`], which the caller holds
//!   (nothing of `atmo-ll.c` reads it).
//! * **The datatype of the sizes.** `nx`, `ny`, `nc` and `res` are `i32` as in the C (initialized to the no-data value,
//!   -9999, by [`Atc::allocate`]).
//! * **Errors.** Where the C prints a message and exits or returns `FAILURE`, this returns an [`AtmoError`].

use crate::meta::Meta;
use crate::radtran::e0;
use std::fmt;

/// Number of elevation bins of the `xyz_*` bricks (`_BYTE_LEN_` of `enum-cl.h`).
pub const N_ELEV: usize = 256;

/// Bands of the sun and view geometry bricks (`enum { ZEN, AZI, cZEN, cAZI, sZEN, sAZI, tZEN, tAZI }`): zenith, azimuth,
/// cosine of the zenith, cosine of the azimuth, sine of the zenith (`sZEN`), sine of the azimuth, tangent of both.
pub const ZEN: usize = 0;
pub const AZI: usize = 1;
pub const C_ZEN: usize = 2;
pub const C_AZI: usize = 3;
pub const S_ZEN: usize = 4;
pub const S_AZI: usize = 5;
pub const T_ZEN: usize = 6;
pub const T_AZI: usize = 7;

/// The no-data value of `atc->nodata` (`_FORCE_NO_DATA_`).
pub const FORCE_NO_DATA: i32 = -9999;

/// The size of the coarse grid cells (`svgrid` of `init_atc_xy`), metres.
pub const SVGRID: i32 = 5000;

/// What the functions of the atmospheric correction that build or read the coarse grid report instead of printing a
/// message and returning `FAILURE` (or exiting).
#[derive(Clone, Debug, PartialEq)]
pub enum AtmoError {
    /// `too many coarse grid cells.`: more than `USHRT_MAX` cells.
    TooManyCells(i32),
    /// An array does not have the size its grid needs.
    Shape(String),
    /// The domain of a band the function needs (`GREEN`) is not among the bands.
    Domain(&'static str),
}

impl fmt::Display for AtmoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtmoError::TooManyCells(n) => write!(f, "too many coarse grid cells ({n}, at most {})", u16::MAX),
            AtmoError::Shape(s) => write!(f, "{s}"),
            AtmoError::Domain(d) => write!(f, "no band of the domain {d}"),
        }
    }
}

impl std::error::Error for AtmoError {}

/// A stack of float bands on a grid of `nx * ny` cells of `res` metres (FORCE's `brick_t` of floats): band `b`, cell
/// `g = row * nx + col` at `bands[b][g]`.
#[derive(Clone, Debug)]
pub struct CoarseBrick {
    pub nx: usize,
    pub ny: usize,
    /// Resolution of the grid in metres (`geotran[_GT_RES_]`).
    pub res: f64,
    /// The nodata value of the bands (`get_brick_nodata`; FORCE keeps one per band, set equal for all of them).
    pub nodata: i32,
    pub bands: Vec<Vec<f32>>,
    /// The domain of each band (`get_brick_domain`: `"BLUE"`, `"GREEN"`, ...), empty where FORCE sets none.
    pub domain: Vec<String>,
}

impl CoarseBrick {
    /// `nb` bands of zeros on an `nx * ny` grid.
    pub fn new(nx: usize, ny: usize, res: f64, nb: usize, nodata: i32) -> Self {
        CoarseBrick { nx, ny, res, nodata, bands: vec![vec![0.0; nx * ny]; nb], domain: vec![String::new(); nb] }
    }

    /// Number of bands (`get_brick_nbands`).
    pub fn nb(&self) -> usize {
        self.bands.len()
    }

    /// Number of cells of one band.
    pub fn ncell(&self) -> usize {
        self.nx * self.ny
    }

    /// The cells of band `b`.
    pub fn band(&self, b: usize) -> &[f32] {
        &self.bands[b]
    }

    /// `find_domain`: the first band of this domain.
    pub fn find_domain(&self, domain: &str) -> Option<usize> {
        self.domain.iter().position(|d| d == domain)
    }

    /// `is_brick_nodata(brick, b, g)`: the cell equals the nodata value (`fequal`, within `FLT_EPSILON`).
    pub fn is_nodata(&self, b: usize, g: usize) -> bool {
        let (x, y) = (self.bands[b][g], self.nodata as f32);
        let diff = (x - y).abs();
        let max = if y.abs() > x.abs() { y.abs() } else { x.abs() };
        diff <= max * f32::EPSILON
    }

    /// `get_brick_range(brick, b, &min, &max)`: the smallest and largest value of band `b` that is not no-data; with no
    /// such value the C's initial values, `LONG_MAX` and `LONG_MIN` (64 bit `long`) as floats.
    pub fn range(&self, b: usize) -> (f32, f32) {
        let (mut mx, mut mn) = (i64::MIN as f32, i64::MAX as f32);
        for g in 0..self.ncell() {
            if self.is_nodata(b, g) {
                continue;
            }
            let tmp = self.bands[b][g];
            if tmp < mn {
                mn = tmp;
            }
            if tmp > mx {
                mx = tmp;
            }
        }
        (mn, mx)
    }
}

/// The elevation statistics and bins of the image (`dem_t`): the DEM is binned in `step` metres from `min` (the
/// elevation of the bin `n` is `n * step + min + step / 2`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DemStats {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// Number of elevation bins minus one (`cnum`; `stats_topography` sets it to `_BYTE_LEN_ - 1`).
    pub cnum: i32,
}

/// The viewing geometry of an image (`view_t`): the nadir line of a Landsat scene in image coordinates
/// (`a x + b y + c = 0`, `ab = sqrt(a^2 + b^2)`), its angle in geographic coordinates, and the height of the sensor
/// (`H`, `H2` its square) in metres.
#[derive(Clone, Copy, Debug, Default)]
pub struct View {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub ab: f32,
    pub geo_angle_nadir: f32,
    pub h: f32,
    pub h2: f32,
}

/// The two-term Henyey-Greenstein coefficients (`tthg_t`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Tthg {
    pub g1: f32,
    pub g2: f32,
    pub alpha: f32,
    pub hg: [f32; 6],
    /// The asymmetry of the phase function used by the multiple scattering path reflectance.
    pub sob: f32,
}

impl Tthg {
    /// The values of `init_atc` (`g1 = 0.801`, `g2 = 0.580`, `alpha = 0.948` and the terms that follow), with the C's
    /// arithmetic: the products of the `float`s are `float`, the sums with the `double` constants are `double`.
    pub fn standard() -> Self {
        let (g1, g2, alpha) = (0.801f64 as f32, 0.580f64 as f32, 0.948f64 as f32);
        let hg = [
            ((1.0 - (g1 * g1) as f64) * alpha as f64) as f32,
            ((1.0 - (g2 * g2) as f64) * (1.0 - alpha as f64)) as f32,
            (1.0 + (g1 * g1) as f64) as f32,
            (1.0 + (g2 * g2) as f64) as f32,
            (2.0 * g1 as f64) as f32,
            (2.0 * g2 as f64) as f32,
        ];
        let sob = (3.0 * (alpha * (g1 + g2) - g2) as f64) as f32;
        Tthg { g1, g2, alpha, hg, sob }
    }
}

/// The part of the digital numbers brick (`DN`) that `allocate_atc` reads: the size and resolution of the image and
/// the wavelength (um) and domain of each band.
#[derive(Clone, Debug)]
pub struct Dn {
    /// Columns and rows of the image (`get_brick_ncols`, `get_brick_nrows`).
    pub nx: usize,
    pub ny: usize,
    /// Pixel size in metres (`get_brick_res`).
    pub res: f64,
    /// Wavelength of each band, um (`get_brick_wavelength`).
    pub wavelength: Vec<f32>,
    /// Domain of each band (`"BLUE"`, `"GREEN"`, ...).
    pub domain: Vec<String>,
}

/// The atmospheric correction factors of one image (FORCE's `atc_t`).
///
/// The geographic coordinates of the coarse cells, which FORCE gets from the projection of the bricks
/// (`get_brick_geo`), are not here: the functions that need them take a [`crate::sunview::SceneGeo`].
#[derive(Clone, Debug)]
pub struct Atc {
    /// Number of cells of the coarse grid in x and y, in total, and their size in metres (`nx`, `ny`, `nc`, `res`).
    pub nx: i32,
    pub ny: i32,
    pub nc: i32,
    pub res: i32,
    /// The no-data value (`nodata`).
    pub nodata: f32,
    /// Viewing geometry (`view`).
    pub view: View,
    /// Wavelength of each band in um (`wvl`), its log (`lwvl`), the square of the log (`lwvl2`) and the exoatmospheric
    /// solar irradiance (`E0`).
    pub wvl: Vec<f32>,
    pub lwvl: Vec<f32>,
    pub lwvl2: Vec<f32>,
    pub e0: Vec<f32>,
    /// Cloud cover (`cc`).
    pub cc: f32,
    /// Elevation statistics and binning (`dem`).
    pub dem: DemStats,
    /// Min and max of the cosine of the sun zenith over the image (`cosszen`).
    pub cosszen: [f32; 2],
    /// Min and max of the cosine of the view zenith over the image (`cosvzen`).
    pub cosvzen: [f32; 2],
    /// Total, molecular and aerosol optical depth of the scene per band (`od`, `mod`, `aod`): `aod` is the mean of the
    /// map, or the fallback spectrum.
    pub od: Vec<f32>,
    pub mod_: Vec<f32>,
    pub aod: Vec<f32>,
    /// The elevation factors of the AOD of the scene average (`Ha`) and of the molecular optical depth (`Hr`), and the
    /// scale height of the aerosols (`Hp`).
    pub hr: f32,
    pub ha: f32,
    pub hp: f32,
    /// The two-term Henyey-Greenstein coefficients (`tthg`).
    pub tthg: Tthg,
    /// Bands used to estimate the AOD (`aod_bands`).
    pub aod_bands: Vec<bool>,
    /// Was an AOD map compiled from dark targets (`aodmap`), or is the fallback used?
    pub aodmap: bool,
    /// The environmental reflectance weighting: the distance (`km`) and the weights of the molecules and aerosols
    /// (`Fr`, `Fa`).
    pub fr: f32,
    pub fa: f32,
    pub km: f32,
    /// Water vapour of the image in cm (`wvp`), set from the LUT or a constant for sensors without a vapour band.
    pub wvp: f32,

    /// Molecular optical depth per band and coarse cell (`xy_mod`; its band domains are the bands of the image's).
    pub xy_mod: CoarseBrick,
    /// AOD per band and coarse cell (`xy_aod`; its band domains name the bands: the grid also defines the cells the
    /// dark targets fall in).
    pub xy_aod: CoarseBrick,
    /// Phase function of the molecular and aerosol scattering (`xy_Pr`, `xy_Pa`; band 0).
    pub xy_pr: CoarseBrick,
    pub xy_pa: CoarseBrick,
    /// Elevation correction factors of the MOD and the AOD per cell (`xy_Hr`, `xy_Ha`; band 0).
    pub xy_hr: CoarseBrick,
    pub xy_ha: CoarseBrick,
    /// Water vapour transmittance, view path (up) and sun path (down), per band and coarse cell (`xy_Tvw`, `xy_Tsw`).
    pub xy_tvw: CoarseBrick,
    pub xy_tsw: CoarseBrick,
    /// Ozone transmittance, view path (up) and sun path (down) (`xy_Tvo`, `xy_Tso`).
    pub xy_tvo: CoarseBrick,
    pub xy_tso: CoarseBrick,
    /// Gaseous transmittance (`xy_Tg`).
    pub xy_tg: CoarseBrick,
    /// BRDF correction factor per band (`xy_brdf`) and Fresnel reflection (`xy_fresnel`, band 0).
    pub xy_brdf: CoarseBrick,
    pub xy_fresnel: CoarseBrick,
    /// Elevation bin per cell (`xy_dem`, band 0).
    pub xy_dem: CoarseBrick,
    /// Cells of the AOD map without an estimate of their own (`xy_interp`, band 0: 1 where the cell was interpolated).
    pub xy_interp: CoarseBrick,
    /// View geometry per coarse cell (bands [`ZEN`], [`AZI`], ...); cells outside the swath hold the nodata value.
    pub xy_view: CoarseBrick,
    /// Sun geometry per coarse cell (bands as `xy_view`).
    pub xy_sun: CoarseBrick,
    /// Backscattering angle per cell (`xy_psi`, band 0).
    pub xy_psi: CoarseBrick,

    /// Total, molecular and aerosol optical depth per elevation bin (`xyz_od`, `xyz_mod`, `xyz_aod`): [`N_ELEV`] bricks
    /// of one band per sensor band.
    pub xyz_od: Vec<CoarseBrick>,
    pub xyz_mod: Vec<CoarseBrick>,
    pub xyz_aod: Vec<CoarseBrick>,
    /// Elevation correction factors of the MOD and the AOD per elevation bin (`xyz_Hr`, `xyz_Ha`; band 0).
    pub xyz_hr: Vec<CoarseBrick>,
    pub xyz_ha: Vec<CoarseBrick>,
    /// Path reflectance per elevation bin (`xyz_rho_p`).
    pub xyz_rho_p: Vec<CoarseBrick>,
    /// Scattering transmittance, sun path (down): total, direct and diffuse (`xyz_Ts`, `xyz_tsd`, `xyz_tss`). The
    /// f-factor of the C-factor of `topo-ll.c` reads `tss` and `tsd`.
    pub xyz_ts: Vec<CoarseBrick>,
    pub xyz_tsd: Vec<CoarseBrick>,
    pub xyz_tss: Vec<CoarseBrick>,
    /// Scattering transmittance, view path (up): total, direct and diffuse (`xyz_Tv`, `xyz_tvd`, `xyz_tvs`).
    pub xyz_tv: Vec<CoarseBrick>,
    pub xyz_tvd: Vec<CoarseBrick>,
    pub xyz_tvs: Vec<CoarseBrick>,
    /// Total scattering transmittance per elevation bin (`xyz_T`).
    pub xyz_t: Vec<CoarseBrick>,
    /// Spherical albedo per elevation bin (`xyz_s`).
    pub xyz_s: Vec<CoarseBrick>,
    /// Environmental reflectance weighting function per elevation bin (`xyz_F`).
    pub xyz_f: Vec<CoarseBrick>,
    /// Gaseous transmittance per elevation bin (`xyz_Tg`).
    pub xyz_tg: Vec<CoarseBrick>,
}

/// The bricks of `atc_t` on an `nx * ny` grid of `res` metres for `nb` sensor bands, with the band domains `domain`
/// (empty strings if it is empty): the allocation of `init_atc_xy` and `init_atc_xyz`. FORCE builds them with
/// `copy_brick` from `xy_mod`, which copies the metadata of band `b` to band `b` if both have as many bands, and of band
/// 0 to every band otherwise.
fn bricks(nx: usize, ny: usize, res: f64, nb: usize, nodata: i32, domain: &[String]) -> Bricks {
    let make = |n: usize| {
        let mut b = CoarseBrick::new(nx, ny, res, n, nodata);
        if !domain.is_empty() {
            for (k, d) in b.domain.iter_mut().enumerate() {
                *d = if n == nb { domain[k].clone() } else { domain[0].clone() };
            }
        }
        b
    };
    let z = |n: usize| vec![make(n); N_ELEV];
    Bricks {
        xy_mod: make(nb),
        xy_aod: make(nb),
        xy_pr: make(1),
        xy_pa: make(1),
        xy_hr: make(1),
        xy_ha: make(1),
        xy_tvw: make(nb),
        xy_tsw: make(nb),
        xy_tvo: make(nb),
        xy_tso: make(nb),
        xy_tg: make(nb),
        xy_brdf: make(nb),
        xy_fresnel: make(1),
        xy_dem: make(1),
        xy_interp: make(1),
        xy_view: make(8),
        xy_sun: make(8),
        xy_psi: make(1),
        xyz_od: z(nb),
        xyz_mod: z(nb),
        xyz_aod: z(nb),
        xyz_hr: z(1),
        xyz_ha: z(1),
        xyz_rho_p: z(nb),
        xyz_ts: z(nb),
        xyz_tsd: z(nb),
        xyz_tss: z(nb),
        xyz_tv: z(nb),
        xyz_tvd: z(nb),
        xyz_tvs: z(nb),
        xyz_t: z(nb),
        xyz_s: z(nb),
        xyz_f: z(nb),
        xyz_tg: z(nb),
    }
}

/// The bricks of an [`Atc`] (see [`bricks`]).
struct Bricks {
    xy_mod: CoarseBrick,
    xy_aod: CoarseBrick,
    xy_pr: CoarseBrick,
    xy_pa: CoarseBrick,
    xy_hr: CoarseBrick,
    xy_ha: CoarseBrick,
    xy_tvw: CoarseBrick,
    xy_tsw: CoarseBrick,
    xy_tvo: CoarseBrick,
    xy_tso: CoarseBrick,
    xy_tg: CoarseBrick,
    xy_brdf: CoarseBrick,
    xy_fresnel: CoarseBrick,
    xy_dem: CoarseBrick,
    xy_interp: CoarseBrick,
    xy_view: CoarseBrick,
    xy_sun: CoarseBrick,
    xy_psi: CoarseBrick,
    xyz_od: Vec<CoarseBrick>,
    xyz_mod: Vec<CoarseBrick>,
    xyz_aod: Vec<CoarseBrick>,
    xyz_hr: Vec<CoarseBrick>,
    xyz_ha: Vec<CoarseBrick>,
    xyz_rho_p: Vec<CoarseBrick>,
    xyz_ts: Vec<CoarseBrick>,
    xyz_tsd: Vec<CoarseBrick>,
    xyz_tss: Vec<CoarseBrick>,
    xyz_tv: Vec<CoarseBrick>,
    xyz_tvd: Vec<CoarseBrick>,
    xyz_tvs: Vec<CoarseBrick>,
    xyz_t: Vec<CoarseBrick>,
    xyz_s: Vec<CoarseBrick>,
    xyz_f: Vec<CoarseBrick>,
    xyz_tg: Vec<CoarseBrick>,
}

impl Atc {
    /// An `Atc` of zeros for `nb` sensor bands on an `nx * ny` grid of `res` metres, nodata `nodata`: the bricks of
    /// [`Atc::allocate`] without the band domains and the initial values of `init_atc` (every parameter is 0, the
    /// Henyey-Greenstein coefficients too). A starting point for tests and for callers that set every field.
    pub fn zeros(nx: usize, ny: usize, res: f64, nb: usize, nodata: i32) -> Self {
        Atc::from_bricks(bricks(nx, ny, res, nb, nodata, &[]), nx, ny, res, nb, nodata)
    }

    /// The structure around the bricks `b`, every parameter 0.
    fn from_bricks(b: Bricks, nx: usize, ny: usize, res: f64, nb: usize, nodata: i32) -> Self {
        Atc {
            nx: nx as i32,
            ny: ny as i32,
            nc: (nx * ny) as i32,
            res: res as i32,
            nodata: nodata as f32,
            view: View::default(),
            wvl: vec![0.0; nb],
            lwvl: vec![0.0; nb],
            lwvl2: vec![0.0; nb],
            e0: vec![0.0; nb],
            cc: 0.0,
            dem: DemStats::default(),
            cosszen: [0.0; 2],
            cosvzen: [0.0; 2],
            od: vec![0.0; nb],
            mod_: vec![0.0; nb],
            aod: vec![0.0; nb],
            hr: 0.0,
            ha: 0.0,
            hp: 0.0,
            tthg: Tthg::default(),
            aod_bands: vec![false; nb],
            aodmap: false,
            fr: 0.0,
            fa: 0.0,
            km: 0.0,
            wvp: 0.0,
            xy_mod: b.xy_mod,
            xy_aod: b.xy_aod,
            xy_pr: b.xy_pr,
            xy_pa: b.xy_pa,
            xy_hr: b.xy_hr,
            xy_ha: b.xy_ha,
            xy_tvw: b.xy_tvw,
            xy_tsw: b.xy_tsw,
            xy_tvo: b.xy_tvo,
            xy_tso: b.xy_tso,
            xy_tg: b.xy_tg,
            xy_brdf: b.xy_brdf,
            xy_fresnel: b.xy_fresnel,
            xy_dem: b.xy_dem,
            xy_interp: b.xy_interp,
            xy_view: b.xy_view,
            xy_sun: b.xy_sun,
            xy_psi: b.xy_psi,
            xyz_od: b.xyz_od,
            xyz_mod: b.xyz_mod,
            xyz_aod: b.xyz_aod,
            xyz_hr: b.xyz_hr,
            xyz_ha: b.xyz_ha,
            xyz_rho_p: b.xyz_rho_p,
            xyz_ts: b.xyz_ts,
            xyz_tsd: b.xyz_tsd,
            xyz_tss: b.xyz_tss,
            xyz_tv: b.xyz_tv,
            xyz_tvd: b.xyz_tvd,
            xyz_tvs: b.xyz_tvs,
            xyz_t: b.xyz_t,
            xyz_s: b.xyz_s,
            xyz_f: b.xyz_f,
            xyz_tg: b.xyz_tg,
        }
    }

    /// `allocate_atc` (`init_atc`, `init_atc_xy`, `init_atc_xyz`): the structure for an image `dn` of the sensor
    /// described by `meta`, with the initial values of the C (the parameters that later stages set are the no-data
    /// value, -9999; the Henyey-Greenstein coefficients, the environmental weighting, the wavelengths and the solar
    /// irradiances are computed) and the bricks on the coarse grid of [`SVGRID`] metres.
    pub fn allocate(dn: &Dn, meta: &Meta) -> Result<Atc, AtmoError> {
        let nb = dn.wavelength.len();
        if dn.domain.len() != nb || meta.cal.len() < nb || nb == 0 {
            return Err(AtmoError::Shape(format!(
                "{} wavelengths, {} domains and {} calibrations: every band needs one of each",
                nb,
                dn.domain.len(),
                meta.cal.len()
            )));
        }
        let nodata = FORCE_NO_DATA as f32;

        // init_atc_xy: resolution and number of cells in x- and y-direction, from the width in map units
        // (`nx * res`, a double) divided by the float `svgrid`
        let width = dn.nx as f64 * dn.res;
        let height = dn.ny as f64 * dn.res;
        let nx = (width / (SVGRID as f32) as f64).ceil() as i32;
        let ny = (height / (SVGRID as f32) as f64).ceil() as i32;
        let nc = nx * ny;
        if nc > u16::MAX as i32 {
            return Err(AtmoError::TooManyCells(nc));
        }
        let bk = bricks(nx as usize, ny as usize, SVGRID as f64, nb, FORCE_NO_DATA, &dn.domain);
        let mut atc = Atc::from_bricks(bk, nx as usize, ny as usize, SVGRID as f64, nb, FORCE_NO_DATA);
        atc.res = SVGRID;

        // init_atc: the nodata of the parameters later stages set
        atc.nodata = nodata;
        atc.tthg = Tthg::standard();
        // the environmental weighting functions for MOD/AOD: r = 1 ring of coarse neighbours
        let r = 1;
        let km = r as f64 * (dn.res / 1000.0) + 0.5 * dn.res / 1000.0;
        atc.km = km as f32;
        let kmf = atc.km as f64;
        atc.fr = (1.0 - (0.930 * (-0.8 * kmf).exp() + 0.070 * (-1.10 * kmf).exp())) as f32;
        atc.fa = (1.0 - (0.375 * (-0.2 * kmf).exp() + 0.625 * (-1.83 * kmf).exp())) as f32;
        atc.aodmap = false;
        atc.cc = nodata;
        atc.wvp = nodata;
        atc.cosszen = [nodata; 2];
        atc.cosvzen = [nodata; 2];
        atc.hr = nodata;
        atc.ha = nodata;
        atc.hp = nodata;
        atc.dem = DemStats { avg: nodata, min: nodata, max: nodata, step: nodata, cnum: FORCE_NO_DATA };
        atc.view = View {
            a: nodata,
            b: nodata,
            c: nodata,
            ab: nodata,
            geo_angle_nadir: nodata,
            h: nodata,
            h2: nodata,
        };
        for bb in 0..nb {
            atc.od[bb] = nodata;
            atc.mod_[bb] = nodata;
            atc.aod[bb] = nodata;
            atc.aod_bands[bb] = false;
            atc.wvl[bb] = dn.wavelength[bb];
            atc.lwvl[bb] = (atc.wvl[bb] as f64).ln() as f32;
            atc.lwvl2[bb] = atc.lwvl[bb] * atc.lwvl[bb];
            atc.e0[bb] = e0(meta.cal[bb].rsr_band);
        }
        Ok(atc)
    }
}
