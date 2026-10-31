//! Texture of an image in a moving window: the grey-level co-occurrence
//! matrix's (GLCM) properties (Haralick, Shanmugam & Dinstein 1973) as
//! scikit-image's `graycomatrix` and `graycoprops` give them, and three of
//! spfeas's simpler features (Graesser et al. 2012): the window's mean, its
//! standard deviation and its mean gradient (Sobel) magnitude. Meant for a
//! temporal summary of a series (a median NDVI, a season's amplitude): the
//! texture of fields, pastures and forests as inputs of a classifier.
//!
//! The image is quantized to `levels` grey levels over [lo, hi] (values
//! outside clipped; NaN none). At each pixel, the pairs of valid pixels of its
//! window `distance` apart along each of the four angles (0, 45, 90 and 135
//! degrees, scikit-image's `angles=[0, pi/4, pi/2, 3pi/4]` with the row offset
//! of `-sin`) make a symmetric, normalized GLCM (scikit-image's
//! `symmetric=True, normed=True`); each property is its mean over the angles
//! with pairs. A window with no pair: NaN.

use crate::parallel;
use rayon::prelude::*;

/// A texture feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// sum P(i,j) (i - j)^2
    Contrast,
    /// sum P(i,j) |i - j|
    Dissimilarity,
    /// sum P(i,j) / (1 + (i - j)^2)
    Homogeneity,
    /// sum P(i,j)^2 (angular second moment)
    Asm,
    /// sqrt(ASM)
    Energy,
    /// sum P(i,j) (i - mu)(j - mu) / sigma^2 (1 where sigma is 0)
    Correlation,
    /// -sum P(i,j) log2 P(i,j)
    Entropy,
    /// the GLCM's mean, sum P(i,j) i
    GlcmMean,
    /// the GLCM's variance, sum P(i,j) (i - mu)^2
    GlcmVariance,
    /// the window's mean of the values (not quantized)
    Mean,
    /// the window's standard deviation of the values (population)
    Std,
    /// the window's mean Sobel gradient magnitude of the values (over its pixels
    /// with all eight neighbours)
    Gradient,
}

impl Feature {
    pub const ALL: [Feature; 12] = [
        Feature::Contrast,
        Feature::Dissimilarity,
        Feature::Homogeneity,
        Feature::Asm,
        Feature::Energy,
        Feature::Correlation,
        Feature::Entropy,
        Feature::GlcmMean,
        Feature::GlcmVariance,
        Feature::Mean,
        Feature::Std,
        Feature::Gradient,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Feature::Contrast => "contrast",
            Feature::Dissimilarity => "dissimilarity",
            Feature::Homogeneity => "homogeneity",
            Feature::Asm => "asm",
            Feature::Energy => "energy",
            Feature::Correlation => "correlation",
            Feature::Entropy => "entropy",
            Feature::GlcmMean => "glcm_mean",
            Feature::GlcmVariance => "glcm_variance",
            Feature::Mean => "mean",
            Feature::Std => "std",
            Feature::Gradient => "gradient",
        }
    }

    pub fn parse(s: &str) -> Result<Feature, String> {
        Feature::ALL
            .into_iter()
            .find(|f| f.name() == s.to_lowercase())
            .ok_or_else(|| {
                format!(
                    "unknown texture feature {s:?} (one of {})",
                    Feature::ALL.map(|f| f.name()).join(", ")
                )
            })
    }

    fn glcm(self) -> bool {
        !matches!(self, Feature::Mean | Feature::Std | Feature::Gradient)
    }
}

/// The texture's settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// the window's side (odd)
    pub window: usize,
    /// grey levels (2..=256)
    pub levels: usize,
    /// the pairs' distance in pixels
    pub distance: usize,
    /// the values' range quantized (NaN: the image's 2nd and 98th percentiles)
    pub lo: f64,
    pub hi: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options { window: 7, levels: 32, distance: 1, lo: f64::NAN, hi: f64::NAN }
    }
}

/// No grey level (a NaN value).
pub const NONE: u8 = 255;

/// The q-th quantile (0..1) of the finite values, linear between order
/// statistics (numpy's default).
pub fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    if i + 1 < v.len() {
        v[i] + f * (v[i + 1] - v[i])
    } else {
        v[i]
    }
}

/// Grey levels 0..levels of the values over [lo, hi] (clipped): level
/// floor((v - lo) / (hi - lo) * levels), the top value in the last; NONE for
/// NaN.
pub fn quantize(values: &[f64], lo: f64, hi: f64, levels: usize) -> Vec<u8> {
    let span = hi - lo;
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return NONE;
            }
            if !(span > 0.0) {
                return 0;
            }
            let k = ((v - lo) / span * levels as f64).floor();
            k.clamp(0.0, (levels - 1) as f64) as u8
        })
        .collect()
}

/// The four angles' (row, column) offsets for a distance d: scikit-image's
/// round(sin(angle) * d) rows up and round(cos(angle) * d) columns.
fn offsets(d: usize) -> [(isize, isize); 4] {
    let d = d as f64;
    [0.0f64, 45.0, 90.0, 135.0].map(|a| {
        let r = a.to_radians();
        ((-(r.sin() * d)).round() as isize, ((r.cos() * d).round()) as isize)
    })
}

/// The texture features of an image (rows x cols, row by row; NaN none):
/// [feature][rows][cols].
pub fn texture(
    image: &[f64],
    rows: usize,
    cols: usize,
    features: &[Feature],
    opts: &Options,
    n_jobs: i32,
) -> Result<Vec<f64>, String> {
    if image.len() != rows * cols {
        return Err(format!("the image has {} values, not {rows} x {cols}", image.len()));
    }
    if opts.window < 3 || opts.window % 2 == 0 {
        return Err(format!("the window must be odd and at least 3, not {}", opts.window));
    }
    if !(2..=254).contains(&opts.levels) {
        return Err(format!("levels must be 2 to 254, not {}", opts.levels));
    }
    if opts.distance == 0 || opts.distance >= opts.window {
        return Err(format!("the distance must be 1 to window - 1, not {}", opts.distance));
    }
    let (lo, hi) = (
        if opts.lo.is_finite() { opts.lo } else { quantile(image, 0.02) },
        if opts.hi.is_finite() { opts.hi } else { quantile(image, 0.98) },
    );
    let grey = quantize(image, lo, hi, opts.levels);
    let need_glcm = features.iter().any(|f| f.glcm());
    let need_grad = features.contains(&Feature::Gradient);
    let grad = if need_grad { sobel(image, rows, cols) } else { vec![] };
    let n = rows * cols;
    let nf = features.len();
    let half = (opts.window / 2) as isize;
    let offs = offsets(opts.distance);
    let levels = opts.levels;
    let mut out = vec![f64::NAN; nf * n];
    // row by row in parallel: each row's features [cols][nf]
    let per_row: Vec<Vec<f64>> = parallel::pool(parallel::threads(n_jobs)).install(|| {
        (0..rows)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![f64::NAN; cols * nf];
                let mut codes: Vec<u16> = Vec::with_capacity(opts.window * opts.window * 2);
                let mut props = [0f64; 9];
                for x in 0..cols {
                    let at = |yy: isize, xx: isize| -> Option<usize> {
                        if yy < 0 || xx < 0 || yy >= rows as isize || xx >= cols as isize {
                            return None;
                        }
                        // inside the window of (y, x)
                        if (yy - y as isize).abs() > half || (xx - x as isize).abs() > half {
                            return None;
                        }
                        Some(yy as usize * cols + xx as usize)
                    };
                    // the GLCM's properties averaged over the angles with pairs
                    let mut angles = 0;
                    if need_glcm {
                        props = [0.0; 9];
                        for &(dr, dc) in &offs {
                            codes.clear();
                            for yy in y as isize - half..=y as isize + half {
                                for xx in x as isize - half..=x as isize + half {
                                    let (Some(a), Some(b)) = (at(yy, xx), at(yy + dr, xx + dc)) else { continue };
                                    let (ga, gb) = (grey[a], grey[b]);
                                    if ga == NONE || gb == NONE {
                                        continue;
                                    }
                                    codes.push(ga as u16 * levels as u16 + gb as u16);
                                    codes.push(gb as u16 * levels as u16 + ga as u16);
                                }
                            }
                            if codes.is_empty() {
                                continue;
                            }
                            angles += 1;
                            codes.sort_unstable();
                            let total = codes.len() as f64;
                            // first pass: the cells' probabilities, the linear sums
                            let (mut con, mut dis, mut hom, mut asm, mut ent, mut mean) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                            let mut k = 0;
                            while k < codes.len() {
                                let c = codes[k];
                                let mut m = k + 1;
                                while m < codes.len() && codes[m] == c {
                                    m += 1;
                                }
                                let p = (m - k) as f64 / total;
                                let (i, j) = ((c as usize / levels) as f64, (c as usize % levels) as f64);
                                let d = i - j;
                                con += p * d * d;
                                dis += p * d.abs();
                                hom += p / (1.0 + d * d);
                                asm += p * p;
                                ent -= p * p.log2();
                                mean += p * i;
                                k = m;
                            }
                            // second pass: the variance and the correlation about the mean
                            let (mut var, mut cov) = (0.0, 0.0);
                            let mut k = 0;
                            while k < codes.len() {
                                let c = codes[k];
                                let mut m = k + 1;
                                while m < codes.len() && codes[m] == c {
                                    m += 1;
                                }
                                let p = (m - k) as f64 / total;
                                let (i, j) = ((c as usize / levels) as f64, (c as usize % levels) as f64);
                                var += p * (i - mean) * (i - mean);
                                cov += p * (i - mean) * (j - mean);
                                k = m;
                            }
                            // scikit-image: correlation 1 where a standard deviation is below 1e-15
                            let corr = if var.sqrt() < 1e-15 { 1.0 } else { cov / var };
                            for (s, v) in props.iter_mut().zip([con, dis, hom, asm, asm.sqrt(), corr, ent, mean, var]) {
                                *s += v;
                            }
                        }
                    }
                    // the window's values
                    let (mut sum, mut sq, mut gs, mut cnt, mut gcnt) = (0.0, 0.0, 0.0, 0usize, 0usize);
                    if features.iter().any(|f| !f.glcm()) {
                        for yy in y as isize - half..=y as isize + half {
                            for xx in x as isize - half..=x as isize + half {
                                let Some(a) = at(yy, xx) else { continue };
                                let v = image[a];
                                if !v.is_finite() {
                                    continue;
                                }
                                sum += v;
                                sq += v * v;
                                if need_grad && grad[a].is_finite() {
                                    gs += grad[a];
                                    gcnt += 1;
                                }
                                cnt += 1;
                            }
                        }
                    }
                    for (fi, f) in features.iter().enumerate() {
                        let v = match f {
                            Feature::Mean if cnt > 0 => sum / cnt as f64,
                            Feature::Std if cnt > 0 => {
                                let m = sum / cnt as f64;
                                (sq / cnt as f64 - m * m).max(0.0).sqrt()
                            }
                            Feature::Gradient if gcnt > 0 => gs / gcnt as f64,
                            Feature::Mean | Feature::Std | Feature::Gradient => f64::NAN,
                            _ if angles == 0 => f64::NAN,
                            f => {
                                let i = match f {
                                    Feature::Contrast => 0,
                                    Feature::Dissimilarity => 1,
                                    Feature::Homogeneity => 2,
                                    Feature::Asm => 3,
                                    Feature::Energy => 4,
                                    Feature::Correlation => 5,
                                    Feature::Entropy => 6,
                                    Feature::GlcmMean => 7,
                                    _ => 8,
                                };
                                props[i] / angles as f64
                            }
                        };
                        row[x * nf + fi] = v;
                    }
                }
                row
            })
            .collect()
    });
    for (y, row) in per_row.into_iter().enumerate() {
        for x in 0..cols {
            for fi in 0..nf {
                out[fi * n + y * cols + x] = row[x * nf + fi];
            }
        }
    }
    Ok(out)
}

/// Each pixel's Sobel gradient magnitude, sqrt(gx^2 + gy^2) of the 3 x 3
/// kernels; NaN at the image's border and next to a NaN.
pub fn sobel(image: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut g = vec![f64::NAN; rows * cols];
    if rows < 3 || cols < 3 {
        return g;
    }
    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let v = |dy: isize, dx: isize| image[(y as isize + dy) as usize * cols + (x as isize + dx) as usize];
            let gx = (v(-1, 1) + 2.0 * v(0, 1) + v(1, 1)) - (v(-1, -1) + 2.0 * v(0, -1) + v(1, -1));
            let gy = (v(1, -1) + 2.0 * v(1, 0) + v(1, 1)) - (v(-1, -1) + 2.0 * v(-1, 0) + v(-1, 1));
            g[y * cols + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_scikit_images() {
        assert_eq!(offsets(1), [(0, 1), (-1, 1), (-1, 0), (-1, -1)]);
        assert_eq!(offsets(2), [(0, 2), (-1, 1), (-2, 0), (-1, -1)]);
    }

    #[test]
    fn a_flat_window_has_no_contrast_and_full_energy() {
        let img = vec![5.0; 49];
        let t = texture(&img, 7, 7, &Feature::ALL, &Options { lo: 0.0, hi: 10.0, ..Default::default() }, 1).unwrap();
        let at = |f: usize| t[f * 49 + 3 * 7 + 3];
        assert_eq!(at(0), 0.0); // contrast
        assert_eq!(at(3), 1.0); // asm
        assert_eq!(at(5), 1.0); // correlation where sigma is 0
        assert_eq!(at(6), 0.0); // entropy
        assert_eq!(at(9), 5.0); // mean
        assert_eq!(at(10), 0.0); // std
    }

    #[test]
    fn stripes_have_contrast_across_them_only() {
        // vertical stripes 0, 9, 0, 9 ...: horizontal pairs always differ
        let (r, c) = (9, 9);
        let img: Vec<f64> = (0..r * c).map(|i| if (i % c) % 2 == 0 { 0.0 } else { 9.0 }).collect();
        let o = Options { window: 3, levels: 10, distance: 1, lo: 0.0, hi: 9.999 };
        let t = texture(&img, r, c, &[Feature::Contrast], &o, 1).unwrap();
        // 0 degrees: all pairs differ by 9 (81); 45 and 135: too; 90: none differ: mean 3/4 of 81
        let v = t[4 * c + 4];
        assert!((v - 81.0 * 3.0 / 4.0).abs() < 1e-12, "{v}");
    }
}
