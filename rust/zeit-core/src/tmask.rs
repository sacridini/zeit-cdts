//! Tmask (Zhu & Woodcock 2014): a port of src/tmask.cpp, same operations in
//! the same order, so the same bits. A robust fit (robustfit.rs) of
//! a0 + c1 t + a1 cos(wt) + b1 sin(wt), w = 2 pi / 365.25, to the green and the
//! SWIR series of a pixel's clear candidates; an observation whose green
//! residual is above `cloud_threshold` (cloud) or whose SWIR residual is below
//! `-shadow_threshold` (shadow), in reflectance (values / scale), is flagged.

use crate::parallel;
use crate::robustfit::robustfit_bisquare;

/// The annual cycle.
const W: f64 = 2.0 * std::f64::consts::PI / 365.25;

/// Tmask of one pixel's observations: 1 clear, 0 flagged; nothing flagged
/// with fewer than 5. `days` in days, any origin.
pub fn tmask_pixel(
    days: &[f64],
    green: &[f64],
    swir: &[f64],
    scale: f64,
    cloud_threshold: f64,
    shadow_threshold: f64,
) -> Result<Vec<u8>, &'static str> {
    let n = days.len();
    if green.len() != n || swir.len() != n {
        return Err("tmask_pixel: days, green and swir must have the same length");
    }
    let cos: Vec<f64> = days.iter().map(|&d| (W * d).cos()).collect();
    let sin: Vec<f64> = days.iter().map(|&d| (W * d).sin()).collect();
    Ok(screen(
        days,
        &cos,
        &sin,
        green,
        swir,
        scale,
        cloud_threshold,
        shadow_threshold,
    ))
}

/// tmask_pixel with cos(w d) and sin(w d) of the days given.
#[allow(clippy::too_many_arguments)]
fn screen(
    days: &[f64],
    cos: &[f64],
    sin: &[f64],
    green: &[f64],
    swir: &[f64],
    scale: f64,
    cloud_threshold: f64,
    shadow_threshold: f64,
) -> Vec<u8> {
    let n = days.len();
    let mut clear = vec![1u8; n];
    if n < 5 {
        return clear;
    }
    // Days from the first date: the fit is the same with any origin, and better conditioned.
    let t0 = days.iter().copied().fold(days[0], |m, d| if d < m { d } else { m });
    let mut x = vec![0.0; n * 3];
    for i in 0..n {
        x[3 * i] = days[i] - t0;
        x[3 * i + 1] = cos[i];
        x[3 * i + 2] = sin[i];
    }
    let g: Vec<f64> = green.iter().map(|v| v / scale).collect();
    let s: Vec<f64> = swir.iter().map(|v| v / scale).collect();
    let bg = robustfit_bisquare(&x, 3, &g);
    let bs = robustfit_bisquare(&x, 3, &s);
    for i in 0..n {
        let xi = &x[3 * i..3 * i + 3];
        let pg = bg[0] + bg[1] * xi[0] + bg[2] * xi[1] + bg[3] * xi[2];
        let ps = bs[0] + bs[1] * xi[0] + bs[2] * xi[1] + bs[3] * xi[2];
        if g[i] - pg > cloud_threshold || s[i] - ps < -shadow_threshold {
            clear[i] = 0;
        }
    }
    clear
}

/// Tmask of every pixel of `green` and `swir` (pixels x T, row-major): an
/// observation is valid when both values are finite and above 0; pixels with
/// more than `min_observations` valid ones are screened, the others have their
/// valid observations clear. 1 = clear, 0 = flagged or not valid.
#[allow(clippy::too_many_arguments)]
pub fn tmask_batch(
    green: &[f64],
    swir: &[f64],
    days: &[f64],
    pixels: usize,
    scale: f64,
    min_observations: i64,
    cloud_threshold: f64,
    shadow_threshold: f64,
    n_jobs: i32,
) -> Vec<u8> {
    let t = days.len();
    assert!(green.len() == pixels * t && swir.len() == pixels * t);
    let mut out = vec![0u8; pixels * t];
    // cos(w d) and sin(w d) of every date, once for all the pixels
    let cos_all: Vec<f64> = days.iter().map(|&d| (W * d).cos()).collect();
    let sin_all: Vec<f64> = days.iter().map(|&d| (W * d).sin()).collect();
    parallel::for_each_row(&mut out, t, n_jobs, Vec::new, |idx: &mut Vec<usize>, p, row| {
        let (gp, sp) = (&green[p * t..(p + 1) * t], &swir[p * t..(p + 1) * t]);
        idx.clear();
        idx.extend((0..t).filter(|&i| gp[i].is_finite() && sp[i].is_finite() && gp[i] > 0.0 && sp[i] > 0.0));
        if idx.len() as i64 <= min_observations {
            for &i in idx.iter() {
                row[i] = 1;
            }
            return;
        }
        let pick = |v: &[f64]| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let clear = screen(
            &pick(days),
            &pick(&cos_all),
            &pick(&sin_all),
            &pick(gp),
            &pick(sp),
            scale,
            cloud_threshold,
            shadow_threshold,
        );
        for (k, &i) in idx.iter().enumerate() {
            row[i] = clear[k];
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_a_cloud_and_a_shadow() {
        let days: Vec<f64> = (0..60).map(|i| 730120.0 + i as f64 * 16.0).collect();
        let mut g: Vec<f64> = days.iter().map(|d| 800.0 + 200.0 * (W * d).sin()).collect();
        let mut s: Vec<f64> = days.iter().map(|d| 2000.0 + 300.0 * (W * d).cos()).collect();
        g[20] += 3000.0;
        s[40] -= 1500.0;
        let clear = tmask_pixel(&days, &g, &s, 10000.0, 0.04, 0.04).unwrap();
        assert_eq!(clear[20], 0);
        assert_eq!(clear[40], 0);
        assert_eq!(clear.iter().filter(|&&c| c == 0).count(), 2);
    }
}
