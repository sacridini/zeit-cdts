//! The potential pixels of Fmask 4.7 for Sentinel-2 (20 m): indices, saturation, snow, water, potential cloud pixels,
//! the normalized cirrus band, absolute snow and the potential cloud mask. A port of the MATLAB of Fmask 4.7
//! (https://github.com/GERSL/Fmask, commit e32f5bd, MIT License, Copyright (c) 2018 Global Environmental Remote
//! Sensing Lab; Qiu, Zhu and Woodcock 2019, *Remote Sensing of Environment* 231:111205): `FmaskParameters.m`, `NDVI.m`,
//! `NDSI.m`, `NDBI.m`, `Saturate.m`, `DetectSnow.m`, `DetectWater.m`, `DetectPotentialPixels.m`,
//! `NormalizaCirrusDEM.m`, `DetectAbsSnow.m`, `DetectPotentialCloud.m`, `probThin.m`, `probwBrightness.m`,
//! `problBrightness.m` and `problSpectralVaribility.m`; one function for each, taking and giving what the .m does,
//! so that each is compared alone with the original (`benchmarks/vs_originals/fmask4.py`).
//!
//! Only the Sentinel-2 route is ported: where the .m branches by sensor or by the presence of the brightness
//! temperature (`data_toabt.BandBT`, Landsat only) the S2 branch is the one here (no BT, `NormalizeBT`,
//! `probwTemperature`, `problTemperature`, the cold cloud test), and the Landsat ones are not ported.
//!
//! The input is the TOA reflectance at 20 m, `f32` times 10000, -9999 where there is no data (`mask` is
//! `band > -9999`), row-major slices (the MATLAB column order shows only in [`super::mlab`]). The numeric classes of
//! the MATLAB are kept: the bands, indices, whiteness and HOT are single (`f32`, and a double constant meeting a single
//! is rounded to single, as MATLAB does); the normalized cirrus band is double (`zeros(..., 'double')`), so
//! everything it touches (`prob_thin`, the final probabilities, the thresholds) is double; `dem` is single
//! (`LoadAuxiData.m` makes it `single`); GSWO is `uint8`; the masks are `u8` (0/1) for MATLAB's `logical`/`uint8`.
//! NaN follows MATLAB: a comparison with it is false, `max` leaves it out, `prctile` ignores it.

use super::mlab::{self, prctile, prctile_f64};
use rayon::prelude::*;

/// `(0..n).map(f)` in parallel, in order.
pub(super) fn pmap<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    (0..n).into_par_iter().map(f).collect()
}

// ---------------------------------------------------------------------------------------------------------------
// FmaskParameters.m
// ---------------------------------------------------------------------------------------------------------------

/// `FmaskParameters.m`: the parameters, here the Sentinel-2 ones (`'S_MSI'`); the Landsat sets are not ported.
#[derive(Clone, Debug, PartialEq)]
pub struct FmaskParameters {
    /// Pixels.
    pub cloud_buffer: f64,
    /// Pixels.
    pub cloud_shadow_buffer: f64,
    /// Pixels.
    pub snow_buffer: f64,
    pub thin_weight: f64,
    pub cloud_probability_threshold: f64,
    /// Meters.
    pub pfpc_layer_extension_radius: f64,
    /// Meters.
    pub pfpc_erosion_radius: f64,
    /// Meters.
    pub output_resolution: f64,
    /// Mask the cloud shadow over water (0 = no).
    pub shadow_water: f64,
}

impl Default for FmaskParameters {
    fn default() -> Self {
        FmaskParameters {
            cloud_buffer: 3.0,
            cloud_shadow_buffer: 3.0,
            snow_buffer: 0.0,
            pfpc_layer_extension_radius: 0.0,
            shadow_water: 0.0,
            // case 'S_MSI'
            thin_weight: 0.5,
            cloud_probability_threshold: 20.00,
            output_resolution: 20.0,
            pfpc_erosion_radius: 90.0, // mirrored from Landsat 8 (the .m)
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// NDVI.m, NDSI.m, NDBI.m, Saturate.m
// ---------------------------------------------------------------------------------------------------------------

/// `NDVI.m`: `(nir - red) ./ (nir + red)` in single, 0.01 where `nir + red == 0`.
pub fn ndvi(red: &[f32], nir: &[f32]) -> Vec<f32> {
    normalized_difference(nir, red)
}

/// `NDSI.m`: `(green - swir) ./ (green + swir)` in single, 0.01 where the sum is 0.
pub fn ndsi(green: &[f32], swir: &[f32]) -> Vec<f32> {
    normalized_difference(green, swir)
}

/// `NDBI.m`: `(swir - nir) ./ (swir + nir)` in single; unlike the others, the zero sum is NOT fixed (the line is
/// commented out in the .m).
pub fn ndbi(nir: &[f32], swir: &[f32]) -> Vec<f32> {
    pmap(nir.len(), |i| (swir[i] - nir[i]) / (swir[i] + nir[i]))
}

/// `(a - b) ./ (a + b)` in single, 0.01 where `a + b == 0`.
fn normalized_difference(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    pmap(a.len(), |i| {
        let d = a[i] + b[i];
        if d == 0.0 {
            0.01f32
        } else {
            (a[i] - b[i]) / d
        }
    })
}

/// `Saturate.m`: saturated at any visible band (`blue + green + red >= 1`).
pub fn saturate(blue: &[u8], green: &[u8], red: &[u8]) -> Vec<u8> {
    pmap(blue.len(), |i| (blue[i] as u32 + green[i] as u32 + red[i] as u32 >= 1) as u8)
}

// ---------------------------------------------------------------------------------------------------------------
// DetectSnow.m
// ---------------------------------------------------------------------------------------------------------------

/// `DetectSnow.m`: `ndsi > 0.15 & nir > 1100 & green > 1000` (the BT test, `band_bt < 1000`, is Landsat's and not
/// ported).
pub fn detect_snow(green: &[f32], nir: &[f32], ndsi: &[f32]) -> Vec<u8> {
    pmap(green.len(), |i| (ndsi[i] > 0.15f32 && nir[i] > 1100.0 && green[i] > 1000.0) as u8)
}

// ---------------------------------------------------------------------------------------------------------------
// DetectWater.m
// ---------------------------------------------------------------------------------------------------------------

/// `DetectWater.m`: `(water, waterAll)`. The spectral water is `(ndvi < 0.01 & nir < 1100) | (ndvi < 0.1 & ndvi > 0 &
/// nir < 500)` inside `mask`; with GSWO (`gswater`, uint8; `None` for `[]`) it is extended where the occurrence is
/// above its 17.5th percentile over the spectral water minus 5, if that is above 0: `waterAll` everywhere,
/// `water` but where `psnow`. (`slope` is an argument of the .m that it does not use.)
pub fn detect_water(mask: &[u8], nir: &[f32], ndvi: &[f32], psnow: &[u8], gswater: Option<&[u8]>) -> (Vec<u8>, Vec<u8>) {
    let n = mask.len();
    let mut water: Vec<u8> = pmap(n, |i| {
        let (v, r) = (ndvi[i], nir[i]);
        let w = (v < 0.01f32 && r < 1100.0) || (v < 0.1f32 && v > 0.0 && r < 500.0);
        (w && mask[i] != 0) as u8
    });
    let mut water_all = water.clone();
    if water.iter().any(|&w| w != 0) {
        if let Some(gs) = gswater {
            if gs.iter().any(|&g| g != 0) {
                let vals: Vec<f64> = gs.iter().zip(&water).filter(|(_, &w)| w == 1).map(|(&g, _)| g as f64).collect();
                let occur = prctile_f64(&vals, 17.5) - 5.0;
                if occur > 0.0 {
                    for i in 0..n {
                        let gs_w = gs[i] as f64 > occur;
                        if gs_w {
                            water_all[i] = 1;
                            if psnow[i] == 0 {
                                water[i] = 1;
                            }
                        }
                        if mask[i] == 0 {
                            water[i] = 0;
                            water_all[i] = 0;
                        }
                    }
                }
            }
        }
    }
    (water, water_all)
}

// ---------------------------------------------------------------------------------------------------------------
// NormalizaCirrusDEM.m
// ---------------------------------------------------------------------------------------------------------------

/// The elements of MATLAB's `a:100:b` for single `a` and `b` (the base plus `i` steps, in single).
fn colon_single(a: f32, step: f32, b: f32) -> Vec<f32> {
    if a.is_nan() || b.is_nan() {
        return Vec::new();
    }
    let q = (b as f64 - a as f64) / step as f64;
    if q < 0.0 {
        return Vec::new();
    }
    let n = (q * (1.0 + 4.0 * f64::EPSILON)).floor() as usize + 1;
    (0..n).map(|i| a + (i as f32) * step).collect()
}

/// `NormalizaCirrusDEM.m`: the cirrus band minus the darkest clear value (the 2nd percentile of the clear pixels
/// `idplcd == false & mask`), per interval of 100 m of the DEM (`dem`; the 0.001th and 99.999th percentiles of the
/// DEM inside the mask give the range) or, without DEM, in all; negative values 0. Returns double. A slice with no
/// clear pixel keeps the previous slice's value (0 at first), as in the .m.
pub fn normaliza_cirrus_dem(mask: &[u8], idplcd: &[u8], cirrus: &[f32], dem: Option<&[f32]>) -> Vec<f64> {
    let n = mask.len();
    let l_pt = 2.0;
    let idclr: Vec<bool> = pmap(n, |i| idplcd[i] == 0 && mask[i] != 0);
    let mut out = vec![0f64; n];
    match dem {
        None => {
            let clr: Vec<f32> = (0..n).filter(|&i| idclr[i]).map(|i| cirrus[i]).collect();
            let low = prctile(&clr, l_pt);
            for i in 0..n {
                if mask[i] != 0 {
                    out[i] = (cirrus[i] - low) as f64;
                }
            }
        }
        Some(dem) => {
            let dm: Vec<f32> = (0..n).filter(|&i| mask[i] != 0).map(|i| dem[i]).collect();
            let dem_start = prctile(&dm, 0.001);
            let dem_end = prctile(&dm, 99.999);
            let lut = colon_single(dem_start, 100.0, dem_end);
            let n_slice = lut.len();
            let mut cirrus_lowest = 0f32;
            for (i, &c) in lut.iter().enumerate() {
                let (lo, hi) = (c - 50.0f32, c + 50.0f32);
                let inter = |d: f32| -> bool {
                    if i + 1 == n_slice {
                        d > lo
                    } else if i == 0 {
                        d < hi
                    } else {
                        d > lo && d < hi
                    }
                };
                let clr: Vec<f32> = (0..n).filter(|&k| idclr[k] && inter(dem[k])).map(|k| cirrus[k]).collect();
                if !clr.is_empty() {
                    cirrus_lowest = prctile(&clr, l_pt);
                }
                out.par_iter_mut().enumerate().for_each(|(k, o)| {
                    if inter(dem[k]) {
                        *o = (cirrus[k] - cirrus_lowest) as f64;
                    }
                });
            }
        }
    }
    out.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------------------------
// DetectPotentialPixels.m
// ---------------------------------------------------------------------------------------------------------------

/// The TOA bands of `data_toabt` that [`detect_potential_pixels`] reads (single, x10000).
pub struct Toa20<'a> {
    pub blue: &'a [f32],
    pub green: &'a [f32],
    pub red: &'a [f32],
    pub nir: &'a [f32],
    pub swir1: &'a [f32],
    pub swir2: &'a [f32],
    pub cirrus: &'a [f32],
}

/// The outputs of `DetectPotentialPixels.m`.
pub struct PotentialPixels {
    /// `idplcd`: the potential cloud pixels.
    pub idplcd: Vec<u8>,
    /// `BandCirrusNormal` (double).
    pub cirrus_normal: Vec<f64>,
    /// Whiteness (single).
    pub whiteness: Vec<f32>,
    /// HOT (single).
    pub hot: Vec<f32>,
}

/// `DetectPotentialPixels.m`: the basic test (`ndsi < 0.8 & ndvi < 0.8 & swir2 > 300`), the whiteness test (`< 0.7`,
/// 0 where a visible band is saturated), the haze test (`HOT > 0` or saturated), the ratio test (`nir / swir1 >
/// 0.75`) and, with the cirrus band (always, for S2), the cirrus test (normalized cirrus `> 100`). The BT test
/// (Landsat) is not ported.
pub fn detect_potential_pixels(mask: &[u8], toa: &Toa20, dem: Option<&[f32]>, ndvi: &[f32], ndsi: &[f32], satu_bv: &[u8]) -> PotentialPixels {
    let n = mask.len();
    let v: Vec<(bool, f32, f32)> = pmap(n, |i| {
        let (b, g, r) = (toa.blue[i], toa.green[i], toa.red[i]);
        let mut p = ndsi[i] < 0.8f32 && ndvi[i] < 0.8f32 && toa.swir2[i] > 300.0;
        let visimean = (b + g + r) / 3.0f32;
        let mut whiteness = ((b - visimean).abs() + (g - visimean).abs() + (r - visimean).abs()) / visimean;
        if satu_bv[i] == 1 {
            whiteness = 0.0;
        }
        p = p && whiteness < 0.7f32;
        let hot = b - 0.5f32 * r - 800.0f32;
        p = p && (hot > 0.0 || satu_bv[i] == 1);
        let ratio = toa.nir[i] / toa.swir1[i];
        p = p && ratio > 0.75f32;
        (p, whiteness, hot)
    });
    let mut idplcd: Vec<u8> = v.iter().map(|t| t.0 as u8).collect();
    let whiteness = v.iter().map(|t| t.1).collect();
    let hot = v.iter().map(|t| t.2).collect();
    drop(v);
    let cirrus_normal = normaliza_cirrus_dem(mask, &idplcd, toa.cirrus, dem);
    for i in 0..n {
        idplcd[i] = (idplcd[i] != 0 || cirrus_normal[i] > 100.0) as u8;
    }
    PotentialPixels { idplcd, cirrus_normal, whiteness, hot }
}

// ---------------------------------------------------------------------------------------------------------------
// DetectAbsSnow.m
// ---------------------------------------------------------------------------------------------------------------

/// `DetectAbsSnow.m`: absolute snow/ice, `scsi < 9 & psnow & ~saturated green`, with the spectral-contextual snow
/// index `scsi = stdfilt(green, true(2 * fix(5000 / resolution) + 1)) .* (1 - ndsi)`, computed in 5 clips of rows
/// (each with `radius` rows more on both sides). `None` (the .m gives `[]`) when there are not more than 110889
/// potential snow pixels.
///
/// The .m is ported as it is, including its clip arithmetic: when the expanded start of a clip other than the first
/// is before row 1 (`row_st_exp < 1`, i.e. clips shorter than the radius, images under about 1250 rows), it takes
/// `row_st_small = 1` and `row_ed_small = row_nums(i_row)` (the code written for the first clip) and so reads the
/// scsi of the FIRST rows of the expanded window, not of the clip's rows (DetectAbsSnow.m lines 71-76).
/// An index out of the window is an error in MATLAB (here a panic).
pub fn detect_abs_snow(green: &[f32], satu_green: &[u8], ndsi: &[f32], psnow: &[u8], rows: usize, cols: usize, resolution: f64) -> Option<Vec<u8>> {
    detect_abs_snow_scsi(green, satu_green, ndsi, psnow, rows, cols, resolution).map(|r| r.0)
}

/// [`detect_abs_snow`] and also the `scsi` the .m builds inside (single values stored in double), for the harness.
pub fn detect_abs_snow_scsi(green: &[f32], satu_green: &[u8], ndsi: &[f32], psnow: &[u8], rows: usize, cols: usize, resolution: f64) -> Option<(Vec<u8>, Vec<f64>)> {
    let n = rows * cols;
    assert!(green.len() == n && ndsi.len() == n && psnow.len() == n && satu_green.len() == n);
    if psnow.iter().map(|&p| p as usize).sum::<usize>() <= 110889 {
        return None;
    }
    // fillmissing(band_green, 'constant', 0)
    let filled: Vec<f32> = green.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
    let radius = (5000.0 / resolution).trunc() as i64;
    let win = (2 * radius + 1) as usize;
    let num_clips = 5usize;
    let size_row = rows as i64;
    let num_per_row = mlab::round_half_away(size_row as f64 / num_clips as f64) as i64;
    let mut row_nums = vec![num_per_row; num_clips];
    row_nums[num_clips - 1] = size_row - num_per_row * (num_clips as i64 - 1);
    let mut scsi = vec![0f64; n];
    for i_row in 0..num_clips {
        let row_ed: i64 = row_nums[..=i_row].iter().sum();
        let row_st = row_ed - row_nums[i_row] + 1;
        let mut row_st_exp = row_st - radius;
        let mut row_ed_exp = row_ed + radius;
        let mut row_st_small = radius + 1;
        let mut row_ed_small = radius + row_nums[i_row];
        if row_st_exp < 1 {
            row_st_exp = 1;
            row_st_small = 1;
            row_ed_small = row_nums[i_row];
        }
        if row_ed_exp > size_row {
            row_ed_exp = size_row;
        }
        let (r0, r1) = ((row_st_exp - 1) as usize, row_ed_exp as usize); // rows r0..r1 (0-based, end exclusive)
        let sub_rows = r1 - r0;
        let std = mlab::stdfilt(&filled[r0 * cols..r1 * cols], sub_rows, cols, win);
        assert!(row_st_small >= 1 && row_ed_small as usize <= sub_rows && row_ed_small - row_st_small == row_ed - row_st, "DetectAbsSnow: index out of range (an error in MATLAB)");
        for k in 0..(row_ed - row_st + 1) as usize {
            let src = (row_st_small - 1) as usize + k; // row of the window
            let dst = (row_st - 1) as usize + k; // row of the image
            for c in 0..cols {
                let w = r0 + src; // row of the image the window row is
                // double .* single is single (the double is converted first); the assignment to scsi widens it back
                scsi[dst * cols + c] = ((std[src * cols + c] as f32) * (1.0f32 - ndsi[w * cols + c])) as f64;
            }
        }
    }
    let abs_snow = pmap(n, |i| (scsi[i] < 9.0 && psnow[i] != 0 && satu_green[i] == 0) as u8);
    Some((abs_snow, scsi))
}

// ---------------------------------------------------------------------------------------------------------------
// probThin.m, probwBrightness.m, problBrightness.m, problSpectralVaribility.m
// ---------------------------------------------------------------------------------------------------------------

/// `probThin.m`: `cirrus / 400`, negatives 0 (not capped at 1). Double, as the normalized cirrus band is.
pub fn prob_thin(cirrus: &[f64]) -> Vec<f64> {
    pmap(cirrus.len(), |i| {
        let v = cirrus[i] / 400.0;
        if v < 0.0 {
            0.0
        } else {
            v
        }
    })
}

/// `probwBrightness.m`: `swir / 1100` clipped to 0..1 (NaN stays), single.
pub fn probw_brightness(swir: &[f32]) -> Vec<f32> {
    pmap(swir.len(), |i| {
        let mut v = swir[i] / 1100.0f32;
        if v > 1.0 {
            v = 1.0;
        }
        if v < 0.0 {
            v = 0.0;
        }
        v
    })
}

/// `problBrightness.m`: the HOT probability, `(HOT - t_l) / (t_h - t_l)` clipped to 0..1, with `t_l` and `t_h` the
/// `100 * l_pt` and `100 * h_pt` percentiles of the HOT over `idlnd`, minus 400 and plus 400. Single.
pub fn probl_brightness(hot: &[f32], idlnd: &[u8], l_pt: f64, h_pt: f64) -> Vec<f32> {
    let f_hot: Vec<f32> = hot.iter().zip(idlnd).filter(|(_, &m)| m != 0).map(|(&h, _)| h).collect();
    let t_hotl = prctile(&f_hot, 100.0 * l_pt) - 400.0f32;
    let t_hoth = prctile(&f_hot, 100.0 * h_pt) + 400.0f32;
    pmap(hot.len(), |i| {
        let mut v = (hot[i] - t_hotl) / (t_hoth - t_hotl);
        if v < 0.0 {
            v = 0.0;
        }
        if v > 1.0 {
            v = 1.0;
        }
        v
    })
}

/// `problSpectralVaribility.m`: `1 - max(|ndsi|, |ndvi|, |ndbi|, whiteness)` (NaN left out by `max`), with `ndsi`
/// set to 0 where green is saturated and `ndsi < 0`, and `ndvi` to 0 where red is saturated and `ndvi > 0`. Single.
pub fn probl_spectral_variability(ndvi: &[f32], ndsi: &[f32], ndbi: &[f32], whiteness: &[f32], satu_green: &[u8], satu_red: &[u8]) -> Vec<f32> {
    pmap(ndvi.len(), |i| {
        let mut s = ndsi[i];
        if satu_green[i] != 0 && s < 0.0 {
            s = 0.0;
        }
        let mut v = ndvi[i];
        if satu_red[i] != 0 && v > 0.0 {
            v = 0.0;
        }
        1.0f32 - s.abs().max(v.abs()).max(ndbi[i].abs()).max(whiteness[i])
    })
}

// ---------------------------------------------------------------------------------------------------------------
// DetectPotentialCloud.m
// ---------------------------------------------------------------------------------------------------------------

/// The outputs of `DetectPotentialCloud.m`.
pub struct PotentialCloud {
    /// `sum_clr`: the number of clear sky pixels.
    pub sum_clr: usize,
    /// `cloud`: the potential clouds.
    pub cloud: Vec<u8>,
    /// `idused` (the clear land pixels, or all clear ones when under 0.1 % of the observed): `None` (`[]`) when
    /// `sum_clr <= 40000`.
    pub idused: Option<Vec<u8>>,
    /// Low and high temperature levels: 0, they are set only with a BT band (Landsat, not ported).
    pub t_templ: f64,
    pub t_temph: f64,
}

/// What `DetectPotentialCloud.m` computes inside (single): the final probabilities over land and water and the two
/// thresholds, for the harness (`detect_potential_cloud_probs`).
pub struct CloudProbs {
    pub lprob_final: Vec<f32>,
    pub wprob_final: Vec<f32>,
    pub clr_max: f32,
    pub wclr_max: f32,
}

/// `DetectPotentialCloud.m`: with at most 40000 clear pixels all the potential cloud pixels are cloud; otherwise
/// the cloud probability over land (HOT probability, spectral variability and the thin cirrus term) and over water
/// (SWIR1 brightness and the thin term) must exceed the 82.5th percentile over the clear pixels plus `cldprob`.
/// `cirrus` is the normalized cirrus band (double), `wpt` the thin weight (0.5 for S2) and `cldprob` the cloud
/// probability threshold (20 for S2). The number of pixels is that of the arrays (`data_meta.Dim`).
#[allow(clippy::too_many_arguments)]
pub fn detect_potential_cloud(
    mask: &[u8],
    water: &[u8],
    swir1: &[f32],
    cirrus: &[f64],
    satu_green: &[u8],
    satu_red: &[u8],
    ndvi: &[f32],
    ndsi: &[f32],
    ndbi: &[f32],
    idplcd: &[u8],
    whiteness: &[f32],
    hot: &[f32],
    wpt: f64,
    cldprob: f64,
) -> PotentialCloud {
    detect_potential_cloud_probs(mask, water, swir1, cirrus, satu_green, satu_red, ndvi, ndsi, ndbi, idplcd, whiteness, hot, wpt, cldprob).0
}

/// [`detect_potential_cloud`] and also the internal probabilities and thresholds (`None` when `sum_clr <= 40000`).
#[allow(clippy::too_many_arguments)]
pub fn detect_potential_cloud_probs(
    mask: &[u8],
    water: &[u8],
    swir1: &[f32],
    cirrus: &[f64],
    satu_green: &[u8],
    satu_red: &[u8],
    ndvi: &[f32],
    ndsi: &[f32],
    ndbi: &[f32],
    idplcd: &[u8],
    whiteness: &[f32],
    hot: &[f32],
    wpt: f64,
    cldprob: f64,
) -> (PotentialCloud, Option<CloudProbs>) {
    let n = mask.len();
    let l_pt = 0.175f64;
    let h_pt = 1.0 - l_pt;
    let idclr: Vec<u8> = pmap(n, |i| (idplcd[i] == 0 && mask[i] == 1) as u8);
    let sum_clr = idclr.iter().map(|&c| c as usize).sum::<usize>();
    let idlnd: Vec<u8> = pmap(n, |i| (idclr[i] != 0 && water[i] == 0) as u8);
    let idwt: Vec<u8> = pmap(n, |i| (idclr[i] != 0 && water[i] == 1) as u8);
    if sum_clr <= 40000 {
        let cloud = pmap(n, |i| (idplcd[i] == 1 && mask[i] != 0) as u8);
        return (PotentialCloud { sum_clr, cloud, idused: None, t_templ: 0.0, t_temph: 0.0 }, None);
    }
    let prob_thin = prob_thin(cirrus);
    let wprob_brightness = probw_brightness(swir1);
    let n_lnd = idlnd.iter().map(|&c| c as usize).sum::<usize>();
    let n_obs = mask.iter().map(|&c| c as usize).sum::<usize>();
    let lndptm = 100.0 * n_lnd as f64 / n_obs as f64;
    let idused = if lndptm >= 0.1 { idlnd.clone() } else { idclr.clone() };
    let lprob_brightness = probl_brightness(hot, &idused, l_pt, h_pt);
    let lprob_vari = probl_spectral_variability(ndvi, ndsi, ndbi, whiteness, satu_green, satu_red);
    // wprob_final = 1 .* wprob_brightness + wpt .* prob_thin, then 100 .*: wpt .* prob_thin is double, and single + double
    // is single (the double is converted first), so the probabilities, their percentiles and the thresholds are single
    let wprob_final: Vec<f32> = pmap(n, |i| 100.0f32 * (wprob_brightness[i] + (wpt * prob_thin[i]) as f32));
    let lprob_final: Vec<f32> = pmap(n, |i| 100.0f32 * (lprob_vari[i] * lprob_brightness[i] + (wpt * prob_thin[i]) as f32));
    let take = |v: &[f32], m: &[u8]| -> Vec<f32> { v.iter().zip(m).filter(|(_, &k)| k != 0).map(|(&x, _)| x).collect() };
    let wclr_h = prctile(&take(&wprob_final, &idwt), 100.0 * h_pt);
    let clr_h = prctile(&take(&lprob_final, &idlnd), 100.0 * h_pt);
    let wclr_max = wclr_h + cldprob as f32;
    let clr_max = clr_h + cldprob as f32;
    // SegementClouds (without the BT test), then cloud(mask == 0) = 0
    let cloud = pmap(n, |i| {
        let c = idplcd[i] == 1 && ((lprob_final[i] > clr_max && water[i] == 0) || (wprob_final[i] > wclr_max && water[i] == 1));
        (c && mask[i] != 0) as u8
    });
    (PotentialCloud { sum_clr, cloud, idused: Some(idused), t_templ: 0.0, t_temph: 0.0 }, Some(CloudProbs { lprob_final, wprob_final, clr_max, wclr_max }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_s2() {
        let p = FmaskParameters::default();
        assert_eq!((p.thin_weight, p.cloud_probability_threshold, p.output_resolution, p.pfpc_erosion_radius), (0.5, 20.0, 20.0, 90.0));
        assert_eq!((p.cloud_buffer, p.cloud_shadow_buffer, p.snow_buffer, p.pfpc_layer_extension_radius, p.shadow_water), (3.0, 3.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn indices() {
        assert_eq!(ndvi(&[1000.0, 0.0, -5.0], &[3000.0, 0.0, 5.0]), vec![0.5, 0.01, 0.01]);
        assert_eq!(ndsi(&[3000.0, 0.0], &[1000.0, 0.0]), vec![0.5, 0.01]);
        // NDBI keeps the zero sum: 0/0 is NaN
        let b = ndbi(&[0.0, 1000.0], &[0.0, 3000.0]);
        assert!(b[0].is_nan() && b[1] == 0.5);
        assert_eq!(saturate(&[0, 1, 0], &[0, 0, 0], &[0, 0, 1]), vec![0, 1, 1]);
    }

    #[test]
    fn snow_and_water() {
        assert_eq!(detect_snow(&[1500.0, 900.0, 1500.0], &[1500.0, 1500.0, 1000.0], &[0.3, 0.3, 0.3]), vec![1, 0, 0]);
        let mask = [1u8, 1, 1, 0, 1];
        let nir = [200.0f32, 2000.0, 400.0, 100.0, 1000.0];
        let ndvi = [0.005f32, 0.005, 0.05, 0.0, -0.2];
        let (w, wa) = detect_water(&mask, &nir, &ndvi, &[0; 5], None);
        // 0: ndvi<0.01 & nir<1100; 1: nir high; 2: 0<ndvi<0.1 & nir<500; 3: outside the mask; 4: ndvi<0.01, nir<1100
        assert_eq!(w, vec![1, 0, 1, 0, 1]);
        assert_eq!(w, wa);
        // GSWO: occurrence 10, 80, 90, .., the water's 17.5th percentile is 10 + (0.175*3+0.5-1)*(90-10)... > 5
        let gs = [10u8, 80, 90, 90, 0];
        let psnow = [0u8, 0, 1, 0, 0];
        let (w2, wa2) = detect_water(&mask, &nir, &ndvi, &psnow, Some(&gs));
        // water pixels' values 10, 90, 0 -> sorted 0, 10, 90: pos .175*3+.5 = 1.025 -> 0.25 -> occur < 5? 0 + 0.025*10 -5 < 0: no extension
        assert_eq!((&w2, &wa2), (&w, &wa));
        let gs2 = [60u8, 80, 90, 90, 60];
        let (w3, wa3) = detect_water(&mask, &nir, &ndvi, &psnow, Some(&gs2));
        // water 60, 90, 60 -> sorted 60 60 90: occur = 60 + 0.025*0 - 5 = 55; > 55: pixels 1, 2, 3 (outside mask)
        assert_eq!(wa3, vec![1, 1, 1, 0, 1]);
        assert_eq!(w3, vec![1, 1, 1, 0, 1]); // pixel 2 is snow (water stays from the spectral test), 1 gets water
        let (w4, _) = detect_water(&mask, &nir, &ndvi, &[0, 0, 0, 0, 0], Some(&[0u8; 5]));
        assert_eq!(w4, w);
    }

    #[test]
    fn cirrus_without_dem() {
        let mask = vec![1u8; 6];
        let idplcd = [0u8, 0, 0, 0, 1, 0];
        let cirrus = [10.0f32, 20.0, 30.0, 40.0, 500.0, 50.0];
        let c = normaliza_cirrus_dem(&mask, &idplcd, &cirrus, None);
        // clear values 10, 20, 30, 40, 50: the 2nd percentile is the minimum (pos 0.02*5+0.5 = 0.6 <= 1): 10
        assert_eq!(c, vec![0.0, 10.0, 20.0, 30.0, 490.0, 40.0]);
    }

    #[test]
    fn cirrus_with_dem() {
        // two slices of the DEM: 0 m and 200 m, each with its own darkest cirrus
        let mask = vec![1u8; 8];
        let idplcd = vec![0u8; 8];
        let dem = [0.0f32, 0.0, 0.0, 0.0, 200.0, 200.0, 200.0, 200.0];
        let cirrus = [5.0f32, 6.0, 7.0, 8.0, 50.0, 51.0, 52.0, 53.0];
        let c = normaliza_cirrus_dem(&mask, &idplcd, &cirrus, Some(&dem));
        assert_eq!(c, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(colon_single(0.0, 100.0, 200.0), vec![0.0, 100.0, 200.0]);
        assert_eq!(colon_single(0.0, 100.0, 199.0), vec![0.0, 100.0]);
        assert_eq!(colon_single(5.0, 100.0, 5.0), vec![5.0]);
    }

    #[test]
    fn probabilities() {
        assert_eq!(prob_thin(&[-10.0, 200.0, 800.0]), vec![0.0, 0.5, 2.0]);
        assert_eq!(probw_brightness(&[-5.0, 550.0, 2000.0]), vec![0.0, 0.5, 1.0]);
        let hot: Vec<f32> = (0..10).map(|i| i as f32 * 100.0).collect();
        let sel = vec![1u8; 10];
        let p = probl_brightness(&hot, &sel, 0.175, 0.825);
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(p.windows(2).all(|w| w[0] <= w[1]));
        // spectral variability: max leaves NaN out; saturation zeroes the sign-specific indices
        let v = probl_spectral_variability(&[0.5, 0.5], &[-0.4, -0.4], &[f32::NAN, 0.1], &[0.2, 0.2], &[0, 1], &[1, 0]);
        assert_eq!(v, vec![1.0 - 0.4, 1.0 - 0.5]);
    }

    #[test]
    fn potential_cloud_branches() {
        let n = 300 * 300;
        let ones = vec![1u8; n];
        let zeros = vec![0u8; n];
        let f = vec![0.1f32; n];
        let hot: Vec<f32> = (0..n).map(|i| (i % 100) as f32).collect();
        let cirrus = vec![0f64; n];
        for (k, branch_all) in [(39999usize, true), (40000, true), (40001, false)] {
            // idplcd true for all but k pixels: sum_clr = k
            let idp: Vec<u8> = (0..n).map(|i| (i >= k) as u8).collect();
            let r = detect_potential_cloud(&ones, &zeros, &f, &cirrus, &zeros, &zeros, &f, &f, &f, &idp, &f, &hot, 0.5, 20.0);
            assert_eq!(r.sum_clr, k);
            if branch_all {
                assert!(r.idused.is_none());
                assert_eq!(r.cloud, idp);
            } else {
                assert!(r.idused.is_some());
            }
        }
        // mask == 0 is never cloud
        let mut mk = ones.clone();
        mk[n - 1] = 0;
        let idp = vec![1u8; n];
        let r = detect_potential_cloud(&mk, &zeros, &f, &cirrus, &zeros, &zeros, &f, &f, &f, &idp, &f, &hot, 0.5, 20.0);
        assert_eq!((r.cloud[n - 1], r.cloud[0]), (0, 1));
    }

    #[test]
    fn abs_snow_threshold() {
        let (rows, cols) = (400, 300);
        let n = rows * cols;
        let green = vec![3000.0f32; n];
        let ndsi = vec![0.5f32; n];
        let sat = vec![0u8; n];
        let few: Vec<u8> = (0..n).map(|i| (i < 110889) as u8).collect();
        assert!(detect_abs_snow(&green, &sat, &ndsi, &few, rows, cols, 20.0).is_none());
        let many: Vec<u8> = (0..n).map(|i| (i < 110890) as u8).collect();
        let a = detect_abs_snow(&green, &sat, &ndsi, &many, rows, cols, 20.0).unwrap();
        // a flat green: scsi 0 < 9 where psnow
        assert_eq!(a, many);
    }
}
