//! The atmospheric correction factors on the coarse grid: the minimal form of FORCE's `atc_t` (`atc-ll.h`), with the
//! fields the water vapour and gas transmittance functions of `gas-ll.c` read. Ported from FORCE 3.11.00 (Copyright
//! (C) 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force).
//!
//! FORCE keeps each factor in a `brick_t`: a stack of bands, each a grid of `nx * ny` coarse cells (`atc->res`
//! metres a side; 3 km in FORCE's default for a Sentinel-2 tile), or, for the factors that depend on the elevation, 256
//! of them (`xyz_*`, one per elevation bin `z`, `_BYTE_LEN_`). Here a [`CoarseBrick`] is one such stack, and the
//! [`Atc`] struct holds the bricks the gas functions and the dark target functions of `aod-ll.c` need. The fields the
//! next stages (the rest of `aod-ll.c`, `atmo-ll.c`) need are added to [`Atc`] as they are ported (Ts, Tv, F, ...);
//! the layout of a brick stays as it is.

/// Number of elevation bins of the `xyz_*` bricks (`_BYTE_LEN_` of `enum-cl.h`).
pub const N_ELEV: usize = 256;

/// Bands of the sun and view geometry bricks (`enum { ZEN, AZI, cZEN, cAZI, sZEN, sAZI, tZEN, tAZI }`): zenith, azimuth,
/// cosine of the zenith, cosine of the azimuth, ... Only the first four are named here.
pub const ZEN: usize = 0;
pub const AZI: usize = 1;
pub const C_ZEN: usize = 2;
pub const C_AZI: usize = 3;

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
}

/// The elevation statistics and bins of the image (`dem_t`): the DEM is binned in `step` metres from `min` (the
/// elevation of the bin `n` is `n * step + min + step / 2`).
#[derive(Clone, Copy, Debug, Default)]
pub struct DemStats {
    pub avg: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

/// The atmospheric correction factors of one image (the part of FORCE's `atc_t` the gas functions use).
///
/// The scene centre in geographic coordinates, which FORCE gets from the projection of the `xy_sun` brick
/// (`get_brick_geo`), is not here: the functions that need it take it as an argument.
#[derive(Clone, Debug)]
pub struct Atc {
    /// Min and max of the cosine of the sun zenith over the image (`cosszen`).
    pub cosszen: [f32; 2],
    /// Min and max of the cosine of the view zenith over the image (`cosvzen`).
    pub cosvzen: [f32; 2],
    /// Water vapour of the image in cm (`wvp`), set from the LUT or a constant for sensors without a vapour band.
    pub wvp: f32,
    /// Sun geometry per coarse cell (bands [`ZEN`], [`AZI`], [`C_ZEN`], ...).
    pub xy_sun: CoarseBrick,
    /// View geometry per coarse cell (bands as `xy_sun`); cells outside the swath hold the nodata value.
    pub xy_view: CoarseBrick,
    /// Ozone transmittance, sun path (down), per band and coarse cell (`xy_Tso`).
    pub xy_tso: CoarseBrick,
    /// Ozone transmittance, view path (up) (`xy_Tvo`).
    pub xy_tvo: CoarseBrick,
    /// Gaseous transmittance (`xy_Tg`); the gas functions read its number of bands only.
    pub xy_tg: CoarseBrick,
    /// Path reflectance per elevation bin (`xyz_rho_p`), [`N_ELEV`] bricks of one band per sensor band.
    pub xyz_rho_p: Vec<CoarseBrick>,
    /// Total scattering transmittance per elevation bin (`xyz_T`).
    pub xyz_t: Vec<CoarseBrick>,
    /// Spherical albedo per elevation bin (`xyz_s`).
    pub xyz_s: Vec<CoarseBrick>,
    /// Wavelength of each band in um (`wvl`), its log (`lwvl`) and the exoatmospheric solar irradiance (`E0`).
    pub wvl: Vec<f32>,
    pub lwvl: Vec<f32>,
    pub e0: Vec<f32>,
    /// Bands used to estimate the AOD (`aod_bands`).
    pub aod_bands: Vec<bool>,
    /// Elevation statistics and binning (`dem`).
    pub dem: DemStats,
    /// Scale height of the aerosols (`Hp`) and the elevation factors of the AOD of the scene average (`Ha`) and of the
    /// molecular optical depth (`Hr`).
    pub hp: f32,
    pub ha: f32,
    pub hr: f32,
    /// Asymmetry of the two-term Henyey-Greenstein phase function (`tthg.sob`).
    pub sob: f32,
    /// AOD per band and coarse cell (`xy_aod`; its band domains name the bands: the grid also defines the cells the
    /// dark targets fall in).
    pub xy_aod: CoarseBrick,
    /// Molecular optical depth per band (`xy_mod`).
    pub xy_mod: CoarseBrick,
    /// Phase function of the molecular and aerosol scattering (`xy_Pr`, `xy_Pa`; band 0).
    pub xy_pr: CoarseBrick,
    pub xy_pa: CoarseBrick,
    /// Elevation correction factors of the MOD and the AOD per cell (`xy_Hr`, `xy_Ha`; band 0).
    pub xy_hr: CoarseBrick,
    pub xy_ha: CoarseBrick,
    /// BRDF correction factor per band (`xy_brdf`) and Fresnel reflection (`xy_fresnel`, band 0).
    pub xy_brdf: CoarseBrick,
    pub xy_fresnel: CoarseBrick,
    /// Elevation bin per cell (`xy_dem`, band 0).
    pub xy_dem: CoarseBrick,
}

impl Atc {
    /// An `Atc` of zeros for `nb` sensor bands on an `nx * ny` grid of `res` metres, nodata `nodata`.
    pub fn zeros(nx: usize, ny: usize, res: f64, nb: usize, nodata: i32) -> Self {
        let brick = |n| CoarseBrick::new(nx, ny, res, n, nodata);
        Atc {
            cosszen: [0.0; 2],
            cosvzen: [0.0; 2],
            wvp: 0.0,
            xy_sun: brick(8),
            xy_view: brick(8),
            xy_tso: brick(nb),
            xy_tvo: brick(nb),
            xy_tg: brick(nb),
            xyz_rho_p: vec![brick(nb); N_ELEV],
            xyz_t: vec![brick(nb); N_ELEV],
            xyz_s: vec![brick(nb); N_ELEV],
            wvl: vec![0.0; nb],
            lwvl: vec![0.0; nb],
            e0: vec![0.0; nb],
            aod_bands: vec![false; nb],
            dem: DemStats::default(),
            hp: 0.0,
            ha: 0.0,
            hr: 0.0,
            sob: 0.0,
            xy_aod: brick(nb),
            xy_mod: brick(nb),
            xy_pr: brick(1),
            xy_pa: brick(1),
            xy_hr: brick(1),
            xy_ha: brick(1),
            xy_brdf: brick(nb),
            xy_fresnel: brick(1),
            xy_dem: brick(1),
        }
    }
}
