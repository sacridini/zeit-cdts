//! `np.percentile` (linear method, numpy's default) and `skimage.exposure.rescale_intensity(image,
//! in_range=(imin, imax), out_range=(omin, omax))`.

/// numpy's linear percentiles of `values` at each `q` in 0..=100 (a NaN in the values gives NaN),
/// as `np.percentile(values, q)` with `q` a sequence or an array (a scalar Python `q` keeps a float32
/// input in float32, which this does not do). The virtual index is (n - 1) * (q / 100) and the
/// interpolation numpy's `_lerp`, in float64.
pub fn percentiles(values: &[f64], qs: &[f64]) -> Vec<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; qs.len()];
    }
    let mut v = values.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    lerp_sorted(&v, qs, |a, b| b - a)
}

/// The same for a float32 image: numpy subtracts the two neighbours in float32 before weighing the
/// difference by the float64 gamma, so the result is float64 with that rounding.
pub fn percentiles_f32(values: &[f32], qs: &[f64]) -> Vec<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; qs.len()];
    }
    let mut v = values.to_vec();
    v.sort_unstable_by(f32::total_cmp);
    let v: Vec<f64> = v.into_iter().map(f64::from).collect();
    lerp_sorted(&v, qs, |a, b| (b as f32 - a as f32) as f64)
}

fn lerp_sorted(v: &[f64], qs: &[f64], sub: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let n = v.len();
    qs.iter()
        .map(|&q| {
            let virt = (n - 1) as f64 * (q / 100.0);
            let prev = virt.floor();
            let gamma = virt - prev;
            let i = prev as usize;
            let a = v[i.min(n - 1)];
            let b = v[(i + 1).min(n - 1)];
            let diff = sub(a, b);
            if gamma >= 0.5 {
                b - diff * (1.0 - gamma)
            } else {
                a + diff * gamma
            }
        })
        .collect()
}

/// `np.clip(x, lo, hi)`: NaN stays NaN.
#[inline]
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// rescale_intensity of a float64 image: clipped to [imin, imax], then mapped to [omin, omax]
/// (a constant range is only clipped to the output range).
pub fn rescale_intensity(values: &[f64], imin: f64, imax: f64, omin: f64, omax: f64) -> Vec<f64> {
    if imin != imax {
        let (din, dout) = (imax - imin, omax - omin);
        values
            .iter()
            .map(|&x| (clip(x, imin, imax) - imin) / din * dout + omin)
            .collect()
    } else {
        values.iter().map(|&x| clip(clip(x, imin, imax), omin, omax)).collect()
    }
}

/// The same for a float32 image, in float32 arithmetic as numpy does it (the Python floats are
/// weak: the limits and the two ranges' widths, computed in f64, are rounded to f32 when used).
pub fn rescale_intensity_f32(values: &[f32], imin: f64, imax: f64, omin: f64, omax: f64) -> Vec<f32> {
    let (lo, hi) = (imin as f32, imax as f32);
    if imin != imax {
        let (din, dout, o) = ((imax - imin) as f32, (omax - omin) as f32, omin as f32);
        values
            .iter()
            .map(|&x| (x.max(lo).min(hi) - lo) / din * dout + o)
            .collect()
    } else {
        let (olo, ohi) = (omin as f32, omax as f32);
        values.iter().map(|&x| x.max(lo).min(hi).max(olo).min(ohi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_numpys_linear() {
        let v = [4.0, 1.0, 3.0, 2.0, 5.0];
        let p = percentiles(&v, &[0.0, 25.0, 50.0, 100.0, 10.0]);
        assert_eq!(p[..4], [1.0, 2.0, 3.0, 5.0]);
        assert!((p[4] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn rescale_maps_and_clips() {
        let r = rescale_intensity(&[-1.0, 0.0, 5.0, 10.0, 20.0], 0.0, 10.0, 0.0, 1.0);
        assert_eq!(r, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        let c = rescale_intensity(&[3.0, 9.0], 5.0, 5.0, 0.0, 1.0);
        assert_eq!(c, vec![1.0, 1.0]);
    }
}
