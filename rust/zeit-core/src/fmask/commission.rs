//! The commission errors of Fmask 4.7 for Sentinel-2: the Cloud Displacement Index, the potential false positive
//! pixels (bright urban surfaces and mountain snow) and the removal of the cloud commissions. A port of the MATLAB of
//! Fmask 4.7 (https://github.com/GERSL/Fmask, commit e32f5bd, MIT License, Copyright (c) 2018 Global Environmental
//! Remote Sensing Lab; Qiu, Zhu and Woodcock 2019, *Remote Sensing of Environment* 231:111205): `CDI.m`,
//! `EnhanceLine.m` (its active part), `DetectPotentialFalsePositivePixels.m` and `ErodeCommissons.m`, one function
//! for each, taking and giving what the .m does. The CDI follows Frantz et al. 2018, *Remote Sensing of Environment*
//! 215:471-481 (the one in [`crate::parallax`] is Frantz's own, FORCE's; this one is Fmask's).
//!
//! Only the Sentinel-2 route is ported: the brightness temperature branch (Landsat; `graythresh`, `imbinarize`) of
//! `DetectPotentialFalsePositivePixels.m` is not. Classes as in [`super::detect`]: bands, ratios and the enhanced
//! NDBI are single, the `stdfilt` of the CDI is double (so the CDI is), masks are `u8` 0/1. A NaN CDI is neither
//! `< -0.8` nor `< -0.5`.

use super::detect::pmap;
use super::mlab::{self, Strel};

/// `CDI.m`: the Cloud Displacement Index from the red edge 3 (band 7), the broad NIR (band 8) and the narrow NIR
/// (band 8A): the variances over 7 x 7 pixels (`stdfilt(.., true(7)) .^ 2`) of the ratios `band8 ./ band8A` (single)
/// and `band7 ./ band8A`, `(var_8A_7 - var_8A_8) ./ (var_8A_8 + var_8A_7)`, double. `None` (the .m gives `[]`) when
/// `band7` or `band8` is empty.
pub fn cdi(band7: &[f32], band8: &[f32], band8a: &[f32], rows: usize, cols: usize) -> Option<Vec<f64>> {
    if band7.is_empty() || band8.is_empty() {
        return None;
    }
    let n = rows * cols;
    assert!(band7.len() == n && band8.len() == n && band8a.len() == n);
    let ratio_8a_8: Vec<f32> = pmap(n, |i| band8[i] / band8a[i]);
    let ratio_8a_7: Vec<f32> = pmap(n, |i| band7[i] / band8a[i]);
    let var8: Vec<f64> = mlab::stdfilt(&ratio_8a_8, rows, cols, 7).into_iter().map(|s| s.powi(2)).collect();
    let var7: Vec<f64> = mlab::stdfilt(&ratio_8a_7, rows, cols, 7).into_iter().map(|s| s.powi(2)).collect();
    Some(pmap(n, |i| (var7[i] - var8[i]) / (var8[i] + var7[i])))
}

/// `EnhanceLine.m` (the part that is not commented out): the maximum of the four line detectors of `band`, the 3 x 3
/// templates `[-1 2 -1; -1 2 -1; -1 2 -1]`, `[-1 -1 -1; 2 2 2; -1 -1 -1]`, `[2 -1 -1; -1 2 -1; -1 -1 2]` and
/// `[-1 -1 2; -1 2 -1; 2 -1 -1]`, each divided by 6 (double, taken to single by `imfilter`), through
/// [`mlab::imfilter3`] (correlation, zero padding), `max` in single leaving NaN out.
pub fn enhance_line(band: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const T: [[f64; 9]; 4] = [
        [-1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0],
        [-1.0, -1.0, -1.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0],
        [2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0],
        [-1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0],
    ];
    let mut out = Vec::new();
    for (k, t) in T.iter().enumerate() {
        let mut tk = [0f32; 9];
        for i in 0..9 {
            tk[i] = (t[i] / 6.0) as f32;
        }
        let f = mlab::imfilter3(band, rows, cols, &tk);
        if k == 0 {
            out = f;
        } else {
            for (o, v) in out.iter_mut().zip(f) {
                *o = v.max(*o);
            }
        }
    }
    out
}

/// `DetectPotentialFalsePositivePixels.m`: [`enhance_line`] of the NDBI, then [`pfpl_from_enhanced`].
#[allow(clippy::too_many_arguments)]
pub fn detect_potential_false_positive_pixels(
    mask: &[u8],
    psnow: &[u8],
    slope: Option<&[f32]>,
    ndbi: &[f32],
    ndvi: &[f32],
    cdi: Option<&[f64]>,
    water: &[u8],
    rows: usize,
    cols: usize,
    resolution: f64,
) -> Vec<u8> {
    let enh = enhance_line(ndbi, rows, cols);
    pfpl_from_enhanced(mask, psnow, slope, &enh, ndvi, cdi, water, rows, cols, resolution)
}

/// The rest of `DetectPotentialFalsePositivePixels.m` after `ndbi = EnhanceLine(ndbi)`: urban pixels
/// (`ndbi > 0 & ndbi > ndvi & mask & water == 0`), which the CDI (`cdi < -0.8`, when there is one) takes back to
/// cloud; mountain snow (`psnow & slope > 20`, with `slope`); all dilated by a square of `2 * fix(250 / resolution)
/// + 1` pixels; plus the potential snow; inside the mask. (Separate from [`detect_potential_false_positive_pixels`]
/// so that the harness can feed it the Octave `imfilter` result.)
#[allow(clippy::too_many_arguments)]
pub fn pfpl_from_enhanced(
    mask: &[u8],
    psnow: &[u8],
    slope: Option<&[f32]>,
    ndbi_enhanced: &[f32],
    ndvi: &[f32],
    cdi: Option<&[f64]>,
    water: &[u8],
    rows: usize,
    cols: usize,
    resolution: f64,
) -> Vec<u8> {
    let n = rows * cols;
    let mut pfpl: Vec<u8> = pmap(n, |i| ((ndbi_enhanced[i] > 0.0 && ndbi_enhanced[i] > ndvi[i]) && mask[i] != 0 && water[i] == 0) as u8);
    if pfpl.iter().any(|&p| p != 0) {
        if let Some(cdi) = cdi {
            for i in 0..n {
                if cdi[i] < -0.8 {
                    pfpl[i] = 0;
                }
            }
        }
    }
    if let Some(slope) = slope {
        for i in 0..n {
            if psnow[i] == 1 && slope[i] > 20.0 {
                pfpl[i] = 1;
            }
        }
    }
    let width_pixels = (250.0 / resolution).trunc() as usize;
    let mut pfpl = mlab::dilate_bin_square(&pfpl, rows, cols, 2 * width_pixels + 1);
    for i in 0..n {
        pfpl[i] = ((pfpl[i] != 0 || psnow[i] != 0) && mask[i] != 0) as u8;
    }
    pfpl
}

/// `ErodeCommissons.m`: removes the commission errors from the potential cloud layer `pcloud`.
///
/// 1. The cloud (`pcloud > 0`) is eroded by `strel('disk', erdpix)`; its pixels that are gone and are potential false
///    positives (`pfpl`) are taken out of the cloud (`cloud_eroded`), and that is dilated by `strel('disk', 2 *
///    erdpix)`.
/// 2. Only the objects (8-connected, of the cloud before the erosion) that keep a pixel are left; the cloud is the
///    dilated eroded cloud inside them, plus the cloud over water.
/// 3. With a CDI: of the objects of fewer than 10000 pixels (8-connected) the ones with no confident pixel
///    (`cdi < -0.5`) are bright surfaces and are removed.
/// 4. Objects of fewer than 3 pixels go.
///
/// The numbering of the labels does not matter anywhere (they only tell which pixels are together), and the label
/// 0 that `ismember` sees in the .m (background) selects only background pixels, which no step reads. `erdpix` is
/// `round(90 / 20) = 5` for Sentinel-2.
pub fn erode_commissons(pcloud: &[u8], pfpl: &[u8], water: &[u8], cdi: Option<&[f64]>, erdpix: usize, rows: usize, cols: usize) -> Vec<u8> {
    let n = rows * cols;
    let cloud: Vec<u8> = pmap(n, |i| (pcloud[i] > 0) as u8);
    let ces: Strel = mlab::strel_disk(erdpix);
    let eroded = mlab::erode_bin(&cloud, rows, cols, &ces);
    let cloud_eroded: Vec<u8> = pmap(n, |i| if eroded[i] == 0 && pfpl[i] != 0 { 0 } else { cloud[i] });
    drop(eroded);
    let dilated = mlab::dilate_bin(&cloud_eroded, rows, cols, &mlab::strel_disk(2 * erdpix));
    let (objs, nobj) = mlab::bwlabel(&cloud, rows, cols);
    let mut keep = vec![false; nobj + 1];
    for i in 0..n {
        if cloud_eroded[i] != 0 && objs[i] > 0 {
            keep[objs[i] as usize] = true;
        }
    }
    let mut cloud: Vec<u8> = pmap(n, |i| {
        let remaining = objs[i] > 0 && keep[objs[i] as usize];
        ((dilated[i] != 0 && remaining) || (water[i] != 0 && cloud[i] != 0)) as u8
    });
    if let Some(cdi) = cdi {
        let large = mlab::bwareaopen(&cloud, rows, cols, 10000);
        let small: Vec<u8> = pmap(n, |i| (cloud[i] == 1 && large[i] == 0) as u8);
        let (lab, nl) = mlab::bwlabel(&small, rows, cols);
        let mut confident = vec![false; nl + 1];
        for i in 0..n {
            if lab[i] > 0 && cdi[i] < -0.5 {
                confident[lab[i] as usize] = true;
            }
        }
        for i in 0..n {
            if small[i] == 1 && !confident[lab[i] as usize] {
                cloud[i] = 0;
            }
        }
    }
    mlab::bwareaopen(&cloud, rows, cols, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdi_flat_and_empty() {
        assert!(cdi(&[], &[1.0], &[1.0], 1, 1).is_none());
        let n = 12 * 12;
        let a: Vec<f32> = (0..n).map(|i| 1000.0 + (i % 7) as f32).collect();
        // band 7 flat ratio, band 8 varying: the 8-ratio variance is larger, CDI < 0
        let b7: Vec<f32> = a.iter().map(|&v| v * 0.9).collect();
        let b8: Vec<f32> = a.iter().enumerate().map(|(i, &v)| v * (1.0 + 0.2 * ((i * 7 % 5) as f32))).collect();
        let c = cdi(&b7, &b8, &a, 12, 12).unwrap();
        assert!(c.iter().all(|&v| v < -0.9), "{:?}", &c[..4]);
        // equal variances: CDI 0
        let c0 = cdi(&b8, &b8, &a, 12, 12).unwrap();
        assert!(c0.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn enhance_line_detects_lines() {
        // a bright vertical line on a dark field: the first template responds at its centre
        let (rows, cols) = (7, 7);
        let mut b = vec![0f32; rows * cols];
        for r in 0..rows {
            b[r * cols + 3] = 6.0;
        }
        let e = enhance_line(&b, rows, cols);
        assert!((e[3 * cols + 3] - 6.0).abs() < 1e-5);
        // NaN is left out by max: a NaN pixel spreads over its 3 x 3 in every template, and the max of NaNs is NaN there
        let mut c = b.clone();
        c[0] = f32::NAN;
        let e2 = enhance_line(&c, rows, cols);
        assert!(e2[0].is_nan() && !e2[3 * cols + 3].is_nan());
    }

    #[test]
    fn pfpl_cases() {
        let (rows, cols) = (60, 60);
        let n = rows * cols;
        let mask = vec![1u8; n];
        let water = vec![0u8; n];
        let psnow = vec![0u8; n];
        let ndvi = vec![0.0f32; n];
        // an urban block: ndbi 1 in a 20 x 20 square (the line detectors respond inside it)
        let mut ndbi = vec![0.0f32; n];
        for r in 20..40 {
            for c in 20..40 {
                ndbi[r * cols + c] = 1.0;
            }
        }
        let p = detect_potential_false_positive_pixels(&mask, &psnow, None, &ndbi, &ndvi, None, &water, rows, cols, 20.0);
        assert_eq!(p[30 * cols + 30], 1);
        assert_eq!(p[0], 0);
        // dilated by 12 pixels: 20 - 12 = 8 is in, 7 is out (for the pixels the detector marks)
        let urban: Vec<usize> = (0..n).filter(|&i| p[i] == 1).collect();
        assert!(urban.len() > 400);
        // a confident negative CDI takes the urban pixels back
        let cdi = vec![-0.9f64; n];
        let p2 = detect_potential_false_positive_pixels(&mask, &psnow, None, &ndbi, &ndvi, Some(&cdi), &water, rows, cols, 20.0);
        assert!(p2.iter().all(|&v| v == 0));
        // mountain snow (psnow and slope > 20) is added, and the potential snow itself
        let mut ps = psnow.clone();
        ps[5 * cols + 5] = 1;
        let slope = vec![25.0f32; n];
        let p3 = detect_potential_false_positive_pixels(&mask, &ps, Some(&slope), &vec![0.0; n], &ndvi, None, &water, rows, cols, 20.0);
        assert_eq!(p3[5 * cols + 5], 1);
        assert_eq!(p3[0], 1); // within 12 pixels of it
        assert_eq!(p3[40 * cols + 40], 0);
        // outside the mask
        let mut m = mask.clone();
        m[5 * cols + 5] = 0;
        let p4 = detect_potential_false_positive_pixels(&m, &ps, Some(&slope), &vec![0.0; n], &ndvi, None, &water, rows, cols, 20.0);
        assert_eq!(p4[5 * cols + 5], 0);
    }

    #[test]
    fn erode_commissons_rules() {
        let (rows, cols) = (80, 120);
        let n = rows * cols;
        let mut cloud = vec![0u8; n];
        let fill = |c: &mut Vec<u8>, r0: usize, r1: usize, c0: usize, c1: usize| {
            for r in r0..r1 {
                for q in c0..c1 {
                    c[r * cols + q] = 1;
                }
            }
        };
        fill(&mut cloud, 10, 40, 10, 40); // a solid cloud
        fill(&mut cloud, 60, 63, 90, 92); // a small object (6 pixels)
        fill(&mut cloud, 0, 2, 0, 1); // 2 pixels at the corner
        let water = vec![0u8; n];
        let pfpl = vec![0u8; n];
        let none = erode_commissons(&cloud, &pfpl, &water, None, 5, rows, cols);
        // no false positive layer: nothing is taken out of the cloud, the small object (6 pixels) stays and the
        // 2 pixels at the corner go (fewer than 3)
        assert_eq!(none[20 * cols + 20], 1);
        assert_eq!(none[61 * cols + 90], 1);
        assert_eq!(none[0], 0);
        // with pfpl on the whole cloud, the eroded interior stays; the border ring of the cloud is taken out
        let pf = vec![1u8; n];
        let r = erode_commissons(&cloud, &pf, &water, None, 5, rows, cols);
        assert_eq!(r[25 * cols + 25], 1);
        assert_eq!(r[61 * cols + 90], 0); // eroded away, a potential false positive: the whole object goes
        // the CDI removes the small object with no confident pixel and keeps the one with
        let mut cloud2 = cloud.clone();
        fill(&mut cloud2, 60, 75, 20, 35);
        let mut cdi = vec![0.0f64; n];
        cdi[20 * cols + 20] = -0.6; // confident inside the first (900 px, small: < 10000)
        let r2 = erode_commissons(&cloud2, &pfpl, &water, Some(&cdi), 5, rows, cols);
        assert_eq!(r2[20 * cols + 20], 1);
        assert_eq!(r2[65 * cols + 25], 0);
        // cloud over water is kept whatever the erosion says
        let mut w = vec![0u8; n];
        for r in 60..63 {
            for q in 90..92 {
                w[r * cols + q] = 1;
            }
        }
        let r3 = erode_commissons(&cloud, &pf, &w, None, 5, rows, cols);
        assert_eq!(r3[61 * cols + 90], 1);
    }
}
