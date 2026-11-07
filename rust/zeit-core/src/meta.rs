//! The metadata of an image as the atmospheric correction reads it: the fields of FORCE's `meta_t` (`meta-ll.h`) that
//! `atc-ll.c`, `sunview-ll.c` and `atmo-ll.c` use. Ported from FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz,
//! GPL-3.0-or-later, https://github.com/davidfrantz/force).
//!
//! Only these fields are here: the response function band of each band of the sensor (`cal_t::rsr_band`) and the grids
//! of the view angles of Sentinel-2 (`s2_meta`: the view zenith and azimuth in degrees on the coarse grid, and the value
//! for a cell without data). The rest of `meta_t` (the calibration constants, the sun angle grids, the reference system,
//! the file names) and the reading of the metadata files (`meta-ll.c`) come with the stage that reads them.

/// The calibration of one band (`cal_t`): its row in the table of spectral response functions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cal {
    /// `rsr_band`: ID in the RSR table ([`crate::radtran`]'s `wavelength`, `e0`, `wvp_transmitt`, ...).
    pub rsr_band: usize,
}

/// The Sentinel-2 grids of the metadata (`s2_meta`): `ny` rows of `nx` cells (the coarse grid of the atmospheric
/// correction), row `e`, column `f` at `e * nx + f`.
#[derive(Clone, Debug, Default)]
pub struct S2Meta {
    /// The value of a cell without data (`nodata`).
    pub nodata: f32,
    /// The view zenith and azimuth in degrees (`vzen`, `vazi`).
    pub vzen: Vec<f32>,
    pub vazi: Vec<f32>,
    /// The grid: cells in x and y (`nx`, `ny`).
    pub nx: usize,
    pub ny: usize,
}

impl S2Meta {
    /// `meta->s2.vzen[e][f]`.
    #[inline]
    pub fn vzen_at(&self, e: usize, f: usize) -> f32 {
        self.vzen[e * self.nx + f]
    }

    /// `meta->s2.vazi[e][f]`.
    #[inline]
    pub fn vazi_at(&self, e: usize, f: usize) -> f32 {
        self.vazi[e * self.nx + f]
    }
}

/// The metadata (`meta_t`): the calibration of every band and the Sentinel-2 angle grids.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub cal: Vec<Cal>,
    pub s2: S2Meta,
}

impl Meta {
    /// Metadata with the given response function bands (`cal[b].rsr_band`) and no Sentinel-2 grids.
    pub fn from_rsr_bands(rsr_band: &[usize]) -> Self {
        Meta { cal: rsr_band.iter().map(|&b| Cal { rsr_band: b }).collect(), s2: S2Meta::default() }
    }
}
