//! Fmask 4.7 for Sentinel-2, end to end: the DN to TOA step of the reading (`nd2toarbt_msi2.m` and `nd2toarbt_msi.m`,
//! the math only), the chain of `autoFmask.m` from the TOA bands on, and the output for FORCE's QAI. A port of the
//! MATLAB of Fmask 4.7 (https://github.com/GERSL/Fmask, commit e32f5bd, MIT License, Copyright (c) 2018 Global
//! Environmental Remote Sensing Lab; Qiu, Zhu and Woodcock 2019, *Remote Sensing of Environment* 231:111205).
//!
//! * [`dn_to_toa20`]: the bands as `u16` digital numbers at their native resolution (10 m: B02, B03, B04, B08; 20 m:
//!   B8A, B11, B12, B07; 60 m: B10) to the single TOA bands (times 10000) of the 20 m grid: DN 0 is no data (NaN),
//!   the 10 m bands go through [`mlab::resize_box_half`], the 60 m one through [`mlab::resize_replicate3`], a pixel
//!   with no data in any band is -9999 (see the note on [`Radiometry`] and `compat`), the saturation flags of the
//!   visible bands are `TOA == 65535` as in the .m (where an offset or a quantification other than 10000 makes them
//!   rare). The reading of the files, the metadata and the angles are the caller's (`crate::s2l1c`).
//! * [`run`]: `autoFmask.m` lines 135-251 for the S2: observation mask, the DEM filled where NaN, the indices and the
//!   CDI, snow, water, potential pixels, absolute snow, potential cloud, false positives, the erosion of the
//!   commissions, the potential shadow (SCS+C with a DEM), the matching, the buffers and the stacking into `cs_final`
//!   (0 clear land, 1 water, 2 shadow, 3 snow, 4 cloud, 255 outside). The result of every step can be kept.
//! * [`to_qai`]: the QAI flags (`qai.rs`) from the Fmask result, on the 20 m grid or on any finer grid that is a whole
//!   multiple of it.
//!
//! `compat`: the .m has three behaviours that look like mistakes and that `compat = true` reproduces and `compat =
//! false` (the default) does not:
//!
//! 1. `autoFmask.m:218-222`: when the scene has no more than 40000 clear pixels the shadow is `pshadow = ~pcloud`,
//!    that is, every pixel that is not cloud is called cloud shadow. Without `compat` there is no shadow then.
//! 2. `MatchCloudShadow.m:50-52`: when the matching is skipped (`ptm <= 40000` or 90 % of cloud) the matched shadow
//!    is set where there is NO potential shadow (`data_shadow_matched(data_shadow_potential==false)=1`). Without
//!    `compat` there is no shadow then. (The `shadow` module is faithful; the choice is made here, through
//!    [`super::shadow::MatchOutput::skipped`].)
//! 3. `nd2toarbt_msi2.m:134` (and the same line of `nd2toarbt_msi.m`): `TOAref(id_missing) = -9999` indexes the
//!    three-dimensional `TOAref` (rows x cols x 9) with a logical matrix of rows x cols, so it writes only the first
//!    band (blue); the other bands keep NaN where any band is missing. Without `compat` all the bands are -9999 there
//!    (what the comment of the .m, "only processing pixels where all bands have values", means). The observation mask
//!    is made from the blue band, so it is the same; the difference is in what the other bands hold outside it.
//!
//! Everything else is equal in the two modes. One more difference from the .m that is in no mode: the view angles
//! in `MatchCloudShadow.m:277` are read from the wrong grid; here [`RunInput::vza`] and `vaa` are read at the 20 m
//! pixel of the cloud pixel, which is what the line means to do (and what the harness gives both sides).
//!
//! Slope and aspect: `autoFmask.m` takes them from `LoadAuxiData.m` (`arcslope`, `aspect`, degrees; the aspect is the
//! direction the slope faces, clockwise from north, which is what `cos(azimuth - aspect)` of the SCS+C in
//! `DetectPotentialCloudShadow.m` asks for). [`slope_aspect_deg`] gives the same quantities from the DEM through
//! Horn's kernel of [`crate::topocorr::slope_aspect`] (0 aspect on the flat); its edge pixels are one-sided
//! differences of the first/last row and column, which are not the edges of the original `arcslope`.

use super::commission;
use super::detect::{self, pmap, FmaskParameters, Toa20};
use super::mlab;
use super::shadow::{self, MatchInput};
use crate::parallax;
use crate::qai;
use rayon::prelude::*;

// ---------------------------------------------------------------------------------------------------------------
// the digital numbers to the TOA reflectance (nd2toarbt_msi2.m, nd2toarbt_msi.m)
// ---------------------------------------------------------------------------------------------------------------

/// The Sentinel-2 L1C bands `nd2toarbt_msi2.m` reads, as `u16` digital numbers at the native resolution, row-major.
/// The 20 m grid is `rows x cols`: the 10 m bands are `2 rows x 2 cols`, the 60 m one `rows / 3 x cols / 3` (so `rows`
/// and `cols` are multiples of 3, as the 5490 x 5490 of a tile is).
pub struct DnBands<'a> {
    pub b02: &'a [u16],
    pub b03: &'a [u16],
    pub b04: &'a [u16],
    pub b8a: &'a [u16],
    pub b11: &'a [u16],
    pub b12: &'a [u16],
    pub b10: &'a [u16],
    pub b07: &'a [u16],
    pub b08: &'a [u16],
}

/// How the DN becomes the reflectance times 10000.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Radiometry {
    /// `nd2toarbt_msi2.m` (baseline >= 04.00): `10000 .* (single(DN) + offset) ./ quantification` in single. The offset
    /// is the `RADIO_ADD_OFFSET` of each band in the order B02, B03, B04, B8A, B11, B12, B10, B07, B08 (the band ids 1,
    /// 2, 3, 8, 11, 12, 10, 6, 7 of the metadata; [`offsets_by_band_id`]) and the quantification the
    /// `QUANTIFICATION_VALUE`.
    Baseline04 { offsets: [f64; 9], quantification: f64 },
    /// `nd2toarbt_msi.m` (older products): the DN itself, in single.
    Legacy,
}

/// The `band_offsets(BandCorrId(iB) == band_offsets(:,1), 2)` of `nd2toarbt_msi2.m`: the offsets, in the order of the
/// bands read, from a table of `(band_id, offset)`; `None` if a band has no entry.
pub fn offsets_by_band_id(table: &[(i32, f64)]) -> Option<[f64; 9]> {
    const IDS: [i32; 9] = [1, 2, 3, 8, 11, 12, 10, 6, 7];
    let mut out = [0f64; 9];
    for (k, id) in IDS.iter().enumerate() {
        out[k] = table.iter().find(|t| t.0 == *id)?.1;
    }
    Some(out)
}

/// The TOA bands of the 20 m grid (single, times 10000, -9999 where a band has no data) and the saturation flags.
pub struct Toa20Bands {
    pub blue: Vec<f32>,
    pub green: Vec<f32>,
    pub red: Vec<f32>,
    /// Band 8A.
    pub nir: Vec<f32>,
    pub swir1: Vec<f32>,
    pub swir2: Vec<f32>,
    /// Band 10.
    pub cirrus: Vec<f32>,
    /// Band 07 (the third red edge band of the CDI).
    pub b7: Vec<f32>,
    /// Band 08 (the broad NIR of the CDI).
    pub b8: Vec<f32>,
    pub satu_blue: Vec<u8>,
    pub satu_green: Vec<u8>,
    pub satu_red: Vec<u8>,
}

impl Toa20Bands {
    /// The view of the bands that [`detect::detect_potential_pixels`] reads.
    pub fn toa(&self) -> Toa20<'_> {
        Toa20 { blue: &self.blue, green: &self.green, red: &self.red, nir: &self.nir, swir1: &self.swir1, swir2: &self.swir2, cirrus: &self.cirrus }
    }
}

/// One band: the DN to single (NaN where the DN is 0) and to the 20 m grid.
fn band_to_20m(dn: &[u16], ratio: Ratio, rows: usize, cols: usize, radio: Radiometry, band: usize) -> Vec<f32> {
    let f: Vec<f32> = match radio {
        Radiometry::Baseline04 { offsets, quantification } => {
            let off = offsets[band] as f32;
            let q = quantification as f32;
            dn.par_iter().map(|&d| if d == 0 { f32::NAN } else { 10000.0f32 * (d as f32 + off) / q }).collect()
        }
        Radiometry::Legacy => dn.par_iter().map(|&d| if d == 0 { f32::NAN } else { d as f32 }).collect(),
    };
    match ratio {
        Ratio::Box => {
            assert_eq!(f.len(), 4 * rows * cols, "a 10 m band must have 2 rows x 2 cols of the 20 m grid");
            mlab::resize_box_half(&f, 2 * rows, 2 * cols)
        }
        Ratio::Same => {
            assert_eq!(f.len(), rows * cols, "a 20 m band must have rows x cols values");
            f
        }
        Ratio::Replicate => {
            assert!(rows % 3 == 0 && cols % 3 == 0 && f.len() == (rows / 3) * (cols / 3), "the 60 m band must have rows / 3 x cols / 3 values");
            mlab::resize_replicate3(&f, rows / 3, cols / 3)
        }
    }
}

#[derive(Clone, Copy)]
enum Ratio {
    Box,
    Same,
    Replicate,
}

/// `nd2toarbt_msi2.m` / `nd2toarbt_msi.m` from the DN on (lines 85-147 / 100-147): see the module. With `compat` the
/// -9999 of the pixels with no data in some band is written to the blue band only, as the .m does.
pub fn dn_to_toa20(dn: &DnBands, rows: usize, cols: usize, radio: Radiometry, compat: bool) -> Toa20Bands {
    let n = rows * cols;
    // BandCorr = {'02','03','04','8A','11','12','10','07','08'}, Ratio = [10 10 10 20 20 20 60 20 10] / 20
    let mut blue = band_to_20m(dn.b02, Ratio::Box, rows, cols, radio, 0);
    let mut green = band_to_20m(dn.b03, Ratio::Box, rows, cols, radio, 1);
    let mut red = band_to_20m(dn.b04, Ratio::Box, rows, cols, radio, 2);
    let mut nir = band_to_20m(dn.b8a, Ratio::Same, rows, cols, radio, 3);
    let mut swir1 = band_to_20m(dn.b11, Ratio::Same, rows, cols, radio, 4);
    let mut swir2 = band_to_20m(dn.b12, Ratio::Same, rows, cols, radio, 5);
    let mut cirrus = band_to_20m(dn.b10, Ratio::Replicate, rows, cols, radio, 6);
    let mut b7 = band_to_20m(dn.b07, Ratio::Same, rows, cols, radio, 7);
    let mut b8 = band_to_20m(dn.b08, Ratio::Box, rows, cols, radio, 8);
    // id_missing = any band is NaN
    let missing: Vec<bool> = pmap(n, |i| {
        blue[i].is_nan() || green[i].is_nan() || red[i].is_nan() || nir[i].is_nan() || swir1[i].is_nan() || swir2[i].is_nan() || cirrus[i].is_nan() || b7[i].is_nan() || b8[i].is_nan()
    });
    let fill = |b: &mut Vec<f32>| {
        b.par_iter_mut().zip(missing.par_iter()).for_each(|(v, &m)| {
            if m {
                *v = -9999.0;
            }
        })
    };
    fill(&mut blue);
    if !compat {
        for b in [&mut green, &mut red, &mut nir, &mut swir1, &mut swir2, &mut cirrus, &mut b7, &mut b8] {
            fill(b);
        }
    }
    // B1Satu = TOAref(:,:,1) == 65535 (after the scaling)
    let sat = |b: &[f32]| -> Vec<u8> { pmap(n, |i| (b[i] == 65535.0f32) as u8) };
    let (satu_blue, satu_green, satu_red) = (sat(&blue), sat(&green), sat(&red));
    Toa20Bands { blue, green, red, nir, swir1, swir2, cirrus, b7, b8, satu_blue, satu_green, satu_red }
}

// ---------------------------------------------------------------------------------------------------------------
// slope and aspect from the DEM
// ---------------------------------------------------------------------------------------------------------------

/// Slope and aspect of a north-up DEM with square pixels of `resolution` metres, in single degrees as
/// `LoadAuxiData.m` gives them (the slope from the horizontal; the aspect clockwise from north, the direction the
/// slope faces, 0 on the flat): [`crate::topocorr::slope_aspect`] (Horn) converted from radians. NaN in the DEM spreads
/// to its neighbours (give the DEM after the fill of [`run`] if that matters). The edges are not those of `arcslope`.
pub fn slope_aspect_deg(dem: &[f32], rows: usize, cols: usize, resolution: f64, n_jobs: i32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(dem.len(), rows * cols);
    let z: Vec<f64> = dem.iter().map(|&v| v as f64).collect();
    let (s, a) = crate::topocorr::slope_aspect(&z, rows, cols, resolution, -resolution, n_jobs);
    let deg = |v: Vec<f32>| -> Vec<f32> { v.into_par_iter().map(|x| (x as f64).to_degrees() as f32).collect() };
    (deg(s), deg(a))
}

// ---------------------------------------------------------------------------------------------------------------
// autoFmask.m
// ---------------------------------------------------------------------------------------------------------------

/// The inputs of the chain, on the 20 m grid (`rows x cols`, row-major).
pub struct RunInput<'a> {
    pub rows: usize,
    pub cols: usize,
    /// The bands of `data_toabt` (single, times 10000, -9999 without data); `cirrus` is the raw band 10.
    pub toa: Toa20<'a>,
    /// Band 07 and band 08 (the CDI); empty slices when there are none (`CDI` gives `[]`).
    pub b7: &'a [f32],
    pub b8: &'a [f32],
    /// Saturation of the visible bands (0/1).
    pub satu_blue: &'a [u8],
    pub satu_green: &'a [u8],
    pub satu_red: &'a [u8],
    /// `dem` (single metres; NaN is filled), `slope` and `aspect` (single degrees) and the GSWO occurrence (uint8).
    pub dem: Option<&'a [f32]>,
    pub slope: Option<&'a [f32]>,
    pub aspect: Option<&'a [f32]>,
    pub gswo: Option<&'a [u8]>,
    /// Sun zenith and azimuth, degrees.
    pub sun_zenith: f64,
    pub sun_azimuth: f64,
    /// View zenith and azimuth angle of each pixel of the 20 m grid, hundredths of a degree (single).
    pub vza: &'a [f32],
    pub vaa: &'a [f32],
    /// `data_meta.Resolution(1)`, metres (20).
    pub resolution: f64,
}

/// The options of [`run`].
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub params: FmaskParameters,
    /// Reproduce the oddities of the .m (see the module); false by default.
    pub compat: bool,
    /// Keep the result of every step in [`RunOutput::steps`].
    pub keep_steps: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig { params: FmaskParameters::default(), compat: false, keep_steps: false }
    }
}

/// The results of the steps, for the comparison with the .m.
#[derive(Clone, Debug, Default)]
pub struct Steps {
    pub psnow0: Vec<u8>,
    pub water_all: Vec<u8>,
    /// `idplcd` after the absolute snow is taken out.
    pub idplcd: Vec<u8>,
    /// `DetectPotentialCloud`'s `cloud`.
    pub cloud: Vec<u8>,
    pub idused: Option<Vec<u8>>,
    pub pfpl: Vec<u8>,
    /// `ErodeCommissons`'s output.
    pub pcloud_eroded: Vec<u8>,
    /// `DetectPotentialCloudShadow`'s output (the scene had enough clear pixels).
    pub pshadow_potential: Option<Vec<u8>>,
    /// The outputs of `MatchCloudShadow`, before `BufferMasks`.
    pub match_cloud: Option<Vec<u8>>,
    pub match_shadow: Option<Vec<u8>>,
    pub topo: Option<shadow::TopoInfo>,
}

/// What `autoFmask.m` produces, and what the QAI needs besides `cs_final`.
#[derive(Clone, Debug)]
pub struct RunOutput {
    pub rows: usize,
    pub cols: usize,
    /// `cs_final`: 0 clear land, 1 water, 2 cloud shadow, 3 snow, 4 cloud, 255 outside the observation.
    pub cs_final: Vec<u8>,
    /// The cloud before the buffer of Fmask (the output of `MatchCloudShadow`, 0/1).
    pub pcloud: Vec<u8>,
    /// The shadow and the snow of `cs_final` (after their buffers), and the water.
    pub pshadow: Vec<u8>,
    pub psnow: Vec<u8>,
    pub water: Vec<u8>,
    /// `ObservMask`.
    pub mask: Vec<u8>,
    pub sum_clr: usize,
    /// The scene had no more than 40000 clear pixels (`autoFmask.m:218`): no matching was tried.
    pub few_clear: bool,
    /// `MatchCloudShadow` skipped the matching (`ptm <= 40000` or 90 % of cloud); false when it was not called.
    pub match_skipped: bool,
    /// `clr_pct` of `autoFmask.m:254`: the percent of the observed pixels that are not cloud (buffer included).
    pub clr_pct: f64,
    pub steps: Option<Box<Steps>>,
}

impl RunOutput {
    /// [`to_qai`] of this result: `factor` is the number of QAI pixels along a 20 m pixel (`1` for 20 m, `2` for 10 m).
    pub fn to_qai(&self, factor: usize, resolution: f64, cloud_buffer_m: f64, n_jobs: i32) -> Vec<i16> {
        to_qai(self.rows, self.cols, factor, resolution, &self.mask, &self.pcloud, &self.water, &self.psnow, &self.pshadow, cloud_buffer_m, n_jobs)
    }
}

/// `autoFmask.m` lines 135-251 for Sentinel-2. Panics on inputs of the wrong size.
pub fn run(inp: &RunInput, cfg: &RunConfig) -> RunOutput {
    let (rows, cols) = (inp.rows, inp.cols);
    let n = rows * cols;
    let prm = &cfg.params;
    let t = &inp.toa;
    for l in [t.blue.len(), t.green.len(), t.red.len(), t.nir.len(), t.swir1.len(), t.swir2.len(), t.cirrus.len(), inp.satu_blue.len(), inp.satu_green.len(), inp.satu_red.len(), inp.vza.len(), inp.vaa.len()] {
        assert_eq!(l, n, "every band must have rows x cols values");
    }
    for o in [inp.dem.map(|d| d.len()), inp.slope.map(|d| d.len()), inp.aspect.map(|d| d.len()), inp.gswo.map(|d| d.len())].into_iter().flatten() {
        assert_eq!(o, n, "dem, slope, aspect and gswo must have rows x cols values");
    }
    let keep = cfg.keep_steps;
    let mut steps = Steps::default();
    // resolution and the pixel counts of the parameters (autoFmask.m:101-107)
    let resolution = prm.output_resolution;
    let cldpix = prm.cloud_buffer as usize;
    let sdpix = prm.cloud_shadow_buffer as usize;
    let snpix = prm.snow_buffer as usize;
    let expdpix = mlab::round_half_away(prm.pfpc_layer_extension_radius / resolution) as usize;
    let erdpix = mlab::round_half_away(prm.pfpc_erosion_radius / resolution) as usize;

    // mask = ObservMask(blue)
    let mask: Vec<u8> = pmap(n, |i| (t.blue[i] > -9999.0f32) as u8);

    // dem(isnan(dem)) = double(prctile(dem(~isnan(dem) & mask), 0.001))
    let dem_filled: Option<Vec<f32>> = inp.dem.map(|d| {
        let mut d = d.to_vec();
        if d.iter().any(|v| v.is_nan()) {
            let valid: Vec<f32> = d.iter().zip(&mask).filter(|(v, &m)| !v.is_nan() && m != 0).map(|(&v, _)| v).collect();
            let fill = mlab::prctile(&valid, 0.001);
            d.par_iter_mut().for_each(|v| {
                if v.is_nan() {
                    *v = fill;
                }
            });
        }
        d
    });
    let dem = dem_filled.as_deref();

    let ndvi = detect::ndvi(t.red, t.nir);
    let ndsi = detect::ndsi(t.green, t.swir1);
    let cdi = commission::cdi(inp.b7, inp.b8, t.nir, rows, cols);
    let satu_bv = detect::saturate(inp.satu_blue, inp.satu_green, inp.satu_red);

    let psnow0 = detect::detect_snow(t.green, t.nir, &ndsi);
    let (water, water_all) = detect::detect_water(&mask, t.nir, &ndvi, &psnow0, inp.gswo);

    let pp = detect::detect_potential_pixels(&mask, t, dem, &ndvi, &ndsi, &satu_bv);
    drop(satu_bv);
    let mut idplcd = pp.idplcd;

    // abs_snow: pure snow/ice is no potential cloud
    if let Some(abs_snow) = detect::detect_abs_snow(t.green, inp.satu_green, &ndsi, &psnow0, rows, cols, inp.resolution) {
        idplcd.par_iter_mut().zip(abs_snow.par_iter()).for_each(|(p, &a)| {
            if a == 1 {
                *p = 0;
            }
        });
    }

    let ndbi = detect::ndbi(t.nir, t.swir1);
    let pc = detect::detect_potential_cloud(
        &mask,
        &water,
        t.swir1,
        &pp.cirrus_normal,
        inp.satu_green,
        inp.satu_red,
        &ndvi,
        &ndsi,
        &ndbi,
        &idplcd,
        &pp.whiteness,
        &pp.hot,
        prm.thin_weight,
        prm.cloud_probability_threshold,
    );
    let sum_clr = pc.sum_clr;
    drop(pp.cirrus_normal);

    let mut pfpl = commission::detect_potential_false_positive_pixels(&mask, &psnow0, inp.slope, &ndbi, &ndvi, cdi.as_deref(), &water, rows, cols, inp.resolution);
    drop((ndbi, ndvi, ndsi));
    if expdpix > 0 {
        pfpl = mlab::dilate_bin_square(&pfpl, rows, cols, 2 * expdpix + 1);
    }
    let pcloud_eroded = commission::erode_commissons(&pc.cloud, &pfpl, &water, cdi.as_deref(), erdpix, rows, cols);
    drop(cdi);

    let few_clear = sum_clr <= 40000;
    let mut match_skipped = false;
    // pcloud0: the cloud before the buffer of Fmask; pcloud_buf: the one that is stacked
    let (pcloud0, pcloud_buf, pshadow, psnow);
    if keep {
        steps.pcloud_eroded = pcloud_eroded.clone();
    }
    if few_clear {
        // autoFmask.m:218-222: every potential cloud is cloud, no buffers
        let pc0: Vec<u8> = pmap(n, |i| (pcloud_eroded[i] > 0) as u8);
        pshadow = if cfg.compat { pmap(n, |i| (pc0[i] == 0) as u8) } else { vec![0u8; n] };
        psnow = psnow0.clone();
        pcloud_buf = pc0.clone();
        pcloud0 = pc0;
    } else {
        let idused = pc.idused.as_deref().expect("idused when sum_clr > 40000");
        let topo = match (inp.slope, inp.aspect) {
            (Some(s), Some(a)) => Some((s, a)),
            _ => None,
        };
        let (ps, info) = shadow::detect_potential_cloud_shadow(rows, cols, t.nir, t.swir1, idused, &mask, topo, inp.sun_zenith, inp.sun_azimuth);
        let m = shadow::match_cloud_shadow(&MatchInput {
            rows,
            cols,
            mask: &mask,
            plcim: &pcloud_eroded,
            plsim: &ps,
            is_shadow_water: prm.shadow_water != 0.0,
            water_all: &water_all,
            dem,
            sun_zenith_deg: inp.sun_zenith,
            sun_azimuth_deg: inp.sun_azimuth,
            resolution: inp.resolution,
            ptm: sum_clr as f64,
            num_near: 14,
            vza: inp.vza,
            vaa: inp.vaa,
        });
        match_skipped = m.skipped;
        let shadow_matched = if m.skipped && !cfg.compat { vec![0u8; n] } else { m.shadow_matched };
        if keep {
            steps.pshadow_potential = Some(ps);
            steps.match_cloud = Some(m.cloud_matched.clone());
            steps.match_shadow = Some(shadow_matched.clone());
            steps.topo = info;
        }
        // autoFmask.m:241
        let (c, s, sn) = shadow::buffer_masks(rows, cols, &m.cloud_matched, cldpix, &shadow_matched, sdpix, &psnow0, snpix);
        pcloud0 = m.cloud_matched;
        pcloud_buf = c;
        pshadow = s;
        psnow = sn;
    }
    let steps_out = keep.then(|| {
        steps.psnow0 = psnow0;
        steps.water_all = water_all;
        steps.idplcd = idplcd;
        steps.cloud = pc.cloud;
        steps.idused = pc.idused;
        steps.pfpl = pfpl;
        Box::new(steps)
    });
    finish(rows, cols, n, mask, water, pcloud_buf, pcloud0, pshadow, psnow, sum_clr, few_clear, match_skipped, steps_out)
}

/// The stacking of `autoFmask.m:245-254`.
#[allow(clippy::too_many_arguments)]
fn finish(
    rows: usize,
    cols: usize,
    n: usize,
    mask: Vec<u8>,
    water: Vec<u8>,
    pcloud_buffered: Vec<u8>,
    pcloud0: Vec<u8>,
    pshadow: Vec<u8>,
    psnow: Vec<u8>,
    sum_clr: usize,
    few_clear: bool,
    match_skipped: bool,
    steps: Option<Box<Steps>>,
) -> RunOutput {
    let cs_final: Vec<u8> = pmap(n, |i| {
        let mut v = (water[i] == 1) as u8; // water first: always the lowest priority
        if psnow[i] == 1 {
            v = 3;
        }
        if pshadow[i] == 1 {
            v = 2;
        }
        if pcloud_buffered[i] == 1 {
            v = 4;
        }
        if mask[i] == 0 {
            v = 255;
        }
        v
    });
    let n_cloud: usize = pcloud_buffered.iter().map(|&v| v as usize).sum();
    let n_mask: usize = mask.iter().map(|&v| v as usize).sum();
    let clr_pct = 100.0 * (1.0 - n_cloud as f64 / n_mask as f64);
    RunOutput { rows, cols, cs_final, pcloud: pcloud0, pshadow, psnow, water, mask, sum_clr, few_clear, match_skipped, clr_pct, steps }
}

// ---------------------------------------------------------------------------------------------------------------
// the QAI
// ---------------------------------------------------------------------------------------------------------------

/// `set_qai` for a field of `bits` bits at `bit`: clear it, then set it to `val`. (`qai.rs` has the setters of the
/// off, aerosol, low sun, subzero, saturation, slope, illumination and vapour fill flags, not of the cloud, shadow,
/// snow and water ones, which are the ones written here.)
#[inline]
fn set_field(v: &mut i16, bit: u32, bits: u32, val: i16) {
    let m = ((1i16 << bits) - 1) << bit;
    *v = (*v & !m) | (val << bit);
}

/// The QAI flags from the Fmask result, on a grid `factor` times finer than the 20 m one (`factor = 1` gives the 20 m
/// grid, `2` the 10 m one: every 20 m pixel is repeated `factor x factor` times), `(rows * factor) x (cols *
/// factor)`, row-major.
///
/// * outside the observation (`mask == 0`, 255 of `cs_final`): the off flag only.
/// * confident cloud (state 2): the cloud of Fmask before its own buffer (`pcloud`), over everything: no other flag,
///   as in the stacking of `cs_final`.
/// * less confident cloud (state 1): the ring of FORCE's 300 m buffer (`cloud_buffer_m`; the midpoint circle of
///   [`parallax::buffer`] with `force = true`, of `cloud_buffer_m / (resolution / factor)` pixels of this grid) around
///   the confident cloud. The buffer of 3 pixels of Fmask is not used.
/// * cirrus (state 3) is never set: Fmask 4 has no cirrus class.
/// * shadow, snow and water: the classes of the stacking of `cs_final` without the cloud, shadow over snow over
///   water (`pshadow` with its Fmask buffer, `psnow`, `water`), flags that are independent of the cloud state, so
///   the pixels of the ring of the buffer keep the flag of the class below them.
///
/// Nothing in `aod`, `gas` or `l2` needs a combination other than these: they test `off` first and then the cloud
/// state and the shadow flag separately.
#[allow(clippy::too_many_arguments)]
pub fn to_qai(
    rows: usize,
    cols: usize,
    factor: usize,
    resolution: f64,
    mask: &[u8],
    pcloud: &[u8],
    water: &[u8],
    psnow: &[u8],
    pshadow: &[u8],
    cloud_buffer_m: f64,
    n_jobs: i32,
) -> Vec<i16> {
    let n = rows * cols;
    assert!(factor >= 1 && mask.len() == n && pcloud.len() == n && water.len() == n && psnow.len() == n && pshadow.len() == n);
    let (fr, fc) = (rows * factor, cols * factor);
    let fine = |i: usize, j: usize| (i / factor) * cols + j / factor;
    // the confident cloud on the QAI grid, then its buffer
    let cloud_fine: Vec<bool> = (0..fr * fc).into_par_iter().map(|p| pcloud[fine(p / fc, p % fc)] == 1 && mask[fine(p / fc, p % fc)] != 0).collect();
    let r = (cloud_buffer_m / (resolution / factor as f64)).round().max(0.0) as usize;
    let ring = if r > 0 { Some(parallax::buffer(&cloud_fine, fc, fr, r, true, n_jobs)) } else { None };
    let mut out = vec![0i16; fr * fc];
    out.par_chunks_mut(fc).enumerate().for_each(|(i, row)| {
        for (j, v) in row.iter_mut().enumerate() {
            let c = fine(i, j);
            let p = i * fc + j;
            if mask[c] == 0 {
                qai::set_off(v, true);
                continue;
            }
            if cloud_fine[p] {
                set_field(v, qai::BIT_CLD, 2, 2);
                continue;
            }
            if let Some(ring) = &ring {
                if ring[p] {
                    set_field(v, qai::BIT_CLD, 2, 1);
                }
            }
            if pshadow[c] == 1 {
                set_field(v, qai::BIT_SHD, 1, 1);
            } else if psnow[c] == 1 {
                set_field(v, qai::BIT_SNW, 1, 1);
            } else if water[c] == 1 {
                set_field(v, qai::BIT_WTR, 1, 1);
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dn_to_toa_scales_resizes_and_fills() {
        let (rows, cols) = (6usize, 6usize);
        let n = rows * cols;
        let ten = |v: u16| vec![v; 4 * n];
        let mut b02 = ten(2000);
        let twenty = |v: u16| vec![v; n];
        let b10 = vec![500u16; (rows / 3) * (cols / 3)];
        // 10 m: one pixel of a 2 x 2 block is 0 -> the 20 m pixel (0, 0) is NaN -> -9999 everywhere
        b02[0] = 0;
        let mut b03 = ten(1500);
        b03[2] = 0; // the 20 m pixel (0, 1): the green band has no data there
        let dn = DnBands { b02: &b02, b03: &b03, b04: &ten(1000), b8a: &twenty(3000), b11: &twenty(2500), b12: &twenty(2000), b10: &b10, b07: &twenty(2900), b08: &ten(3100) };
        let off = [-1000.0; 9];
        let radio = Radiometry::Baseline04 { offsets: off, quantification: 10000.0 };
        let t = dn_to_toa20(&dn, rows, cols, radio, false);
        assert_eq!(t.blue.len(), n);
        assert_eq!(t.blue[0], -9999.0);
        assert_eq!(t.cirrus[0], -9999.0);
        assert_eq!(t.nir[0], -9999.0);
        assert_eq!(t.blue[1], -9999.0);
        assert_eq!(t.nir[1], -9999.0);
        assert_eq!(t.blue[2], 10000.0f32 * (2000.0f32 - 1000.0f32) / 10000.0f32);
        assert_eq!(t.cirrus[n - 1], 10000.0f32 * (500.0f32 - 1000.0f32) / 10000.0f32); // negative: not clipped
        // the compat mode writes only the blue band
        let c = dn_to_toa20(&dn, rows, cols, radio, true);
        assert_eq!(c.blue[0], -9999.0);
        assert_eq!(c.blue[1], -9999.0);
        assert!(c.green[1].is_nan());
        assert_eq!(c.nir[1], 2000.0);
        assert_eq!(c.green[0], 10000.0f32 * (1500.0f32 - 1000.0f32) / 10000.0f32);
        // the old products: the DN as it is
        let l = dn_to_toa20(&dn, rows, cols, Radiometry::Legacy, false);
        assert_eq!(l.nir[2], 3000.0);
        assert_eq!(l.green[3], 1500.0);
        assert!(l.satu_blue.iter().all(|&s| s == 0));
        // 60 m band repeated 3 x 3
        let mut b10v = vec![0u16; (rows / 3) * (cols / 3)];
        b10v[1] = 777;
        b10v[0] = 100;
        b10v[2] = 100;
        b10v[3] = 100;
        let dn2 = DnBands { b10: &b10v, ..dn };
        let l2 = dn_to_toa20(&dn2, rows, cols, Radiometry::Legacy, false);
        assert_eq!(l2.cirrus[3], 777.0);
        assert_eq!(l2.cirrus[5 * cols + 5], 100.0);
        let ids = [(1, -1000.0), (2, -1000.0), (3, -1000.0), (8, -1.0), (11, -1000.0), (12, -1000.0), (10, 0.0), (6, 5.0), (7, 7.0)];
        assert_eq!(offsets_by_band_id(&ids).unwrap(), [-1000.0, -1000.0, -1000.0, -1.0, -1000.0, -1000.0, 0.0, 5.0, 7.0]);
        assert!(offsets_by_band_id(&ids[..8]).is_none());
    }

    /// A scene of bright land and bright white cloud blocks, at the scale where the tests of the steps fit.
    fn scene(rows: usize, cols: usize, cloud_frac: f64) -> (Toa20Bands, Vec<f32>, Vec<f32>) {
        let n = rows * cols;
        let mut s = 12345u64;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let cutoff = (cloud_frac * rows as f64) as usize;
        let mut b = Toa20Bands {
            blue: vec![0.0; n],
            green: vec![0.0; n],
            red: vec![0.0; n],
            nir: vec![0.0; n],
            swir1: vec![0.0; n],
            swir2: vec![0.0; n],
            cirrus: vec![0.0; n],
            b7: vec![0.0; n],
            b8: vec![0.0; n],
            satu_blue: vec![0; n],
            satu_green: vec![0; n],
            satu_red: vec![0; n],
        };
        for i in 0..rows {
            for j in 0..cols {
                let p = i * cols + j;
                let u = rnd() as f32;
                if i < cutoff {
                    let c = 4000.0 + 3000.0 * u;
                    b.blue[p] = c * 0.95;
                    b.green[p] = c;
                    b.red[p] = c * 0.98;
                    b.nir[p] = c * 0.95;
                    b.swir1[p] = c * 0.7;
                    b.swir2[p] = c * 0.55;
                    b.cirrus[p] = 400.0 + 400.0 * u;
                } else {
                    b.blue[p] = 400.0 + 50.0 * u;
                    b.green[p] = 650.0 + 50.0 * u;
                    b.red[p] = 500.0 + 50.0 * u;
                    b.nir[p] = 3000.0 + 300.0 * u;
                    b.swir1[p] = 1500.0 + 100.0 * u;
                    b.swir2[p] = 900.0 + 100.0 * u;
                    b.cirrus[p] = 15.0 + 10.0 * u;
                }
                b.b7[p] = 0.9 * b.nir[p];
                b.b8[p] = b.nir[p] * (1.0 + 0.02 * u);
            }
        }
        (b, vec![500.0; n], vec![9000.0; n])
    }

    fn input<'a>(t: &'a Toa20Bands, rows: usize, cols: usize, vza: &'a [f32], vaa: &'a [f32]) -> RunInput<'a> {
        RunInput {
            rows,
            cols,
            toa: t.toa(),
            b7: &t.b7,
            b8: &t.b8,
            satu_blue: &t.satu_blue,
            satu_green: &t.satu_green,
            satu_red: &t.satu_red,
            dem: None,
            slope: None,
            aspect: None,
            gswo: None,
            sun_zenith: 35.0,
            sun_azimuth: 150.0,
            vza,
            vaa,
            resolution: 20.0,
        }
    }

    #[test]
    fn few_clear_pixels_compat_and_default() {
        // under 40000 clear pixels: the .m calls every pixel that is not cloud a cloud shadow
        let (rows, cols) = (120usize, 150usize);
        let (t, vza, vaa) = scene(rows, cols, 0.5);
        let inp = input(&t, rows, cols, &vza, &vaa);
        let c = run(&inp, &RunConfig { compat: true, keep_steps: true, ..RunConfig::default() });
        assert!(c.few_clear && !c.match_skipped);
        assert!(c.pshadow.iter().zip(&c.pcloud).all(|(&s, &p)| s == (p == 0) as u8));
        assert!(c.cs_final.iter().any(|&v| v == 2));
        let d = run(&inp, &RunConfig { compat: false, keep_steps: true, ..RunConfig::default() });
        assert!(d.few_clear);
        assert!(d.pshadow.iter().all(|&s| s == 0));
        assert!(!d.cs_final.iter().any(|&v| v == 2));
        // the cloud and the rest are equal in both modes
        assert_eq!(c.pcloud, d.pcloud);
        assert_eq!(c.psnow, d.psnow);
        assert!(d.pcloud.iter().any(|&v| v == 1));
        assert_eq!(c.cs_final.iter().zip(&d.cs_final).filter(|(a, b)| a != b).count(), c.cs_final.iter().filter(|&&v| v == 2).count());
    }

    #[test]
    fn skipped_matching_compat_and_default() {
        // more than 40000 clear pixels and more than 90 % of cloud: the matching is skipped, and the .m puts the shadow
        // where there is no potential shadow
        let (rows, cols) = (700usize, 700usize);
        let (t, vza, vaa) = scene(rows, cols, 0.90);
        let mut inp = input(&t, rows, cols, &vza, &vaa);
        inp.sun_zenith = 35.0;
        let c = run(&inp, &RunConfig { compat: true, keep_steps: true, ..RunConfig::default() });
        let d = run(&inp, &RunConfig { compat: false, keep_steps: true, ..RunConfig::default() });
        assert!(!c.few_clear, "clear pixels: {}", c.sum_clr);
        let cloud_pix = c.pcloud.iter().map(|&v| v as usize).sum::<usize>();
        assert!(c.match_skipped && d.match_skipped, "cloud pixels {} of {}", cloud_pix, rows * cols);
        let sc = c.steps.as_ref().unwrap();
        let pot = sc.pshadow_potential.as_ref().unwrap();
        // the .m: shadow = not potential shadow
        assert!(c.steps.as_ref().unwrap().match_shadow.as_ref().unwrap().iter().zip(pot).all(|(&m, &p)| m == (p != 1) as u8));
        assert!(c.pshadow.iter().any(|&v| v == 1));
        assert!(d.pshadow.iter().all(|&v| v == 0));
        assert_eq!(c.pcloud, d.pcloud);
        assert_eq!(c.psnow, d.psnow);
    }

    #[test]
    fn qai_flags() {
        // 20 x 20 pixels at 20 m (400 m): a cloud pixel at (9, 9), a shadow pixel that the ring covers, snow, water, off
        let (rows, cols) = (40usize, 40usize);
        let n = rows * cols;
        let mut mask = vec![1u8; n];
        let mut pcloud = vec![0u8; n];
        let mut water = vec![0u8; n];
        let mut snow = vec![0u8; n];
        let mut shadow = vec![0u8; n];
        pcloud[20 * cols + 20] = 1;
        shadow[20 * cols + 24] = 1; // inside the ring of 15 pixels
        snow[20 * cols + 26] = 1;
        water[20 * cols + 28] = 1;
        water[30 * cols + 35] = 1; // outside the ring
        shadow[3 * cols + 3] = 1;
        snow[3 * cols + 3] = 1; // shadow over snow
        mask[0] = 0;
        pcloud[0] = 0;
        let q = to_qai(rows, cols, 1, 20.0, &mask, &pcloud, &water, &snow, &shadow, 300.0, 1);
        assert_eq!(q.len(), n);
        let at = |i: usize, j: usize| q[i * cols + j];
        assert!(qai::off(at(0, 0)) && qai::cloud(at(0, 0)) == 0);
        assert_eq!(at(0, 0), 1);
        assert_eq!(qai::cloud(at(20, 20)), 2);
        assert_eq!(at(20, 20) & !(3 << 1), 0, "no other flag under the confident cloud");
        // the ring: radius 15 pixels (300 m / 20 m), the midpoint circle
        assert_eq!(qai::cloud(at(20, 20 + 14)), 1);
        assert_eq!(qai::cloud(at(20, 20 - 15)), 1);
        assert_eq!(qai::cloud(at(20, 20 + 16)), 0);
        // the ring keeps the class below
        assert_eq!((qai::cloud(at(20, 24)), qai::shadow(at(20, 24))), (1, true));
        assert_eq!((qai::cloud(at(20, 26)), qai::snow(at(20, 26)), qai::shadow(at(20, 26))), (1, true, false));
        assert_eq!((qai::cloud(at(20, 28)), qai::water(at(20, 28))), (1, true));
        // outside the ring: only the class
        assert_eq!((qai::cloud(at(30, 35)), qai::water(at(30, 35))), (0, true));
        // shadow over snow
        assert_eq!((qai::shadow(at(3, 3)), qai::snow(at(3, 3))), (true, false));
        // cirrus (3) is never set, anywhere
        assert!(q.iter().all(|&v| qai::cloud(v) != 3));
        // cloud too close to the edge: FORCE's buffer loses it, the cloud itself stays
        let mut pc2 = vec![0u8; n];
        pc2[2 * cols + 2] = 1;
        let q2 = to_qai(rows, cols, 1, 20.0, &mask, &pc2, &water, &snow, &shadow, 300.0, 1);
        assert_eq!(qai::cloud(q2[2 * cols + 2]), 2);
        assert_eq!(q2.iter().filter(|&&v| qai::cloud(v) == 1).count(), 0);
        // no buffer
        let q0 = to_qai(rows, cols, 1, 20.0, &mask, &pcloud, &water, &snow, &shadow, 0.0, 1);
        assert_eq!(q0.iter().filter(|&&v| qai::cloud(v) == 1).count(), 0);
    }

    #[test]
    fn qai_at_10_m() {
        let (rows, cols) = (50usize, 50usize);
        let n = rows * cols;
        let mask = vec![1u8; n];
        let mut pcloud = vec![0u8; n];
        let zero = vec![0u8; n];
        pcloud[25 * cols + 25] = 1;
        let q10 = to_qai(rows, cols, 2, 20.0, &mask, &pcloud, &zero, &zero, &zero, 300.0, 1);
        assert_eq!(q10.len(), 4 * n);
        let w = 2 * cols;
        // the 20 m pixel is a block of 2 x 2 confident-cloud pixels
        for (i, j) in [(50, 50), (50, 51), (51, 50), (51, 51)] {
            assert_eq!(qai::cloud(q10[i * w + j]), 2);
        }
        assert_eq!(q10.iter().filter(|&&v| qai::cloud(v) == 2).count(), 4);
        // the ring is 300 m = 30 pixels of 10 m, from the edge of the block
        assert_eq!(qai::cloud(q10[50 * w + 51 + 29]), 1);
        assert_eq!(qai::cloud(q10[50 * w + 51 + 31]), 0);
        assert_eq!(qai::cloud(q10[50 * w + 50 - 30]), 1);
        let q20 = to_qai(rows, cols, 1, 20.0, &mask, &pcloud, &zero, &zero, &zero, 300.0, 1);
        assert_eq!(qai::cloud(q20[25 * cols + 25 + 15]), 1);
        assert_eq!(qai::cloud(q20[25 * cols + 25 + 16]), 0);
    }

    #[test]
    fn slope_aspect_of_a_plane() {
        // rises to the east: faces west (270 degrees), 45 degrees of slope at 20 m
        let (rows, cols) = (10usize, 12usize);
        let dem: Vec<f32> = (0..rows * cols).map(|p| 20.0 * (p % cols) as f32).collect();
        let (s, a) = slope_aspect_deg(&dem, rows, cols, 20.0, 1);
        assert!((s[5 * cols + 5] - 45.0).abs() < 1e-3, "{}", s[5 * cols + 5]);
        assert!((a[5 * cols + 5] - 270.0).abs() < 1e-3, "{}", a[5 * cols + 5]);
    }
}
