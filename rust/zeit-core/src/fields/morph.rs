//! Grayscale morphology equal to `skimage.morphology` 0.26 (which calls `scipy.ndimage.grey_*`):
//! `dilation`, `erosion`, `opening`, `closing`, `white_tophat` and `black_tophat` with a footprint
//! (square, disk, cross or any boolean mask of odd size) and a border mode (`reflect`, scipy's and
//! scikit-image's default, `constant` with `cval`, `nearest`, `mirror`, `wrap`).
//!
//! As in scikit-image, `erosion` is the minimum and `dilation` the maximum of the image over the
//! footprint placed on each pixel (no mirroring); `opening` and `closing` apply the second step with
//! the mirrored footprint, which only matters for asymmetric ones. Values are f64 (integer images
//! are exact in it); the top-hats subtract in f64, which for f32 or integer inputs equals the
//! subtraction in their own type once cast back. NaN is not supported.

/// How the borders are extended, as in `scipy.ndimage`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Border {
    /// d c b a | a b c d | d c b a
    Reflect,
    /// the constant value on every side
    Constant(f64),
    /// a a a a | a b c d | d d d d
    Nearest,
    /// d c b | a b c d | c b a
    Mirror,
    /// a b c d | a b c d | a b c d
    Wrap,
}

impl Border {
    /// scipy's name of the mode (`cval` used by "constant" only).
    pub fn parse(name: &str, cval: f64) -> Result<Border, String> {
        match name {
            "reflect" => Ok(Border::Reflect),
            "constant" => Ok(Border::Constant(cval)),
            "nearest" => Ok(Border::Nearest),
            "mirror" => Ok(Border::Mirror),
            "wrap" => Ok(Border::Wrap),
            _ => Err(format!(
                "unsupported mode {name:?} (reflect, constant, nearest, mirror or wrap)"
            )),
        }
    }

    /// The index in `0..n` that `i` stands for, None for the constant outside.
    fn map(self, i: isize, n: usize) -> Option<usize> {
        let ni = n as isize;
        if (0..ni).contains(&i) {
            return Some(i as usize);
        }
        match self {
            Border::Constant(_) => None,
            Border::Nearest => Some(i.clamp(0, ni - 1) as usize),
            Border::Wrap => Some(i.rem_euclid(ni) as usize),
            Border::Reflect => {
                let m = i.rem_euclid(2 * ni);
                Some(if m >= ni { 2 * ni - 1 - m } else { m } as usize)
            }
            Border::Mirror => {
                if ni == 1 {
                    return Some(0);
                }
                let m = i.rem_euclid(2 * ni - 2);
                Some(if m >= ni { 2 * ni - 2 - m } else { m } as usize)
            }
        }
    }
}

/// A footprint: the offsets (row, col) from the centre of its true cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Footprint {
    pub offsets: Vec<(isize, isize)>,
}

impl Footprint {
    /// From a boolean mask of odd size [rows, cols], its centre the middle cell.
    pub fn from_mask(mask: &[bool], rows: usize, cols: usize) -> Result<Footprint, String> {
        if mask.len() != rows * cols || rows.is_multiple_of(2) || cols.is_multiple_of(2) {
            return Err("the footprint must have an odd number of rows and columns".into());
        }
        let (cr, cc) = ((rows / 2) as isize, (cols / 2) as isize);
        let mut offsets = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                if mask[r * cols + c] {
                    offsets.push((r as isize - cr, c as isize - cc));
                }
            }
        }
        if offsets.is_empty() {
            return Err("the footprint is empty".into());
        }
        Ok(Footprint { offsets })
    }

    /// `footprint_rectangle((size, size))`, `size` odd.
    pub fn square(size: usize) -> Result<Footprint, String> {
        Footprint::from_mask(&vec![true; size * size], size, size)
    }

    /// `skimage.morphology.disk(radius)`: x^2 + y^2 <= radius^2.
    pub fn disk(radius: usize) -> Footprint {
        let r = radius as isize;
        let mut offsets = Vec::new();
        for dr in -r..=r {
            for dc in -r..=r {
                if dr * dr + dc * dc <= r * r {
                    offsets.push((dr, dc));
                }
            }
        }
        Footprint { offsets }
    }

    /// The default footprint of scikit-image: the 3x3 cross (4-neighbourhood).
    pub fn cross() -> Footprint {
        Footprint {
            offsets: vec![(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
        }
    }

    /// The footprint reversed in every axis.
    pub fn mirrored(&self) -> Footprint {
        Footprint {
            offsets: self.offsets.iter().map(|&(r, c)| (-r, -c)).collect(),
        }
    }
}

fn filter(
    img: &[f64],
    rows: usize,
    cols: usize,
    fp: &Footprint,
    border: Border,
    is_max: bool,
) -> Result<Vec<f64>, String> {
    if img.len() != rows * cols {
        return Err("the image must have rows * cols values".into());
    }
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let pr = fp.offsets.iter().map(|o| o.0.unsigned_abs()).max().unwrap_or(0);
    let pc = fp.offsets.iter().map(|o| o.1.unsigned_abs()).max().unwrap_or(0);
    let (ph, pw) = (rows + 2 * pr, cols + 2 * pc);
    let cval = if let Border::Constant(v) = border { v } else { 0.0 };
    let mut pad = vec![cval; ph * pw];
    for r in 0..ph {
        let Some(sr) = border.map(r as isize - pr as isize, rows) else {
            continue;
        };
        for c in 0..pw {
            if let Some(sc) = border.map(c as isize - pc as isize, cols) {
                pad[r * pw + c] = img[sr * cols + sc];
            }
        }
    }
    let mut out = vec![if is_max { f64::NEG_INFINITY } else { f64::INFINITY }; rows * cols];
    for &(dr, dc) in &fp.offsets {
        for r in 0..rows {
            let src = &pad[(r + pr).wrapping_add_signed(dr) * pw..][..pw];
            let src = &src[pc.wrapping_add_signed(dc)..][..cols];
            let dst = &mut out[r * cols..(r + 1) * cols];
            if is_max {
                for (o, &v) in dst.iter_mut().zip(src) {
                    if v > *o {
                        *o = v;
                    }
                }
            } else {
                for (o, &v) in dst.iter_mut().zip(src) {
                    if v < *o {
                        *o = v;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `skimage.morphology.dilation(image, footprint, mode=, cval=)`.
pub fn dilation(img: &[f64], rows: usize, cols: usize, fp: &Footprint, border: Border) -> Result<Vec<f64>, String> {
    filter(img, rows, cols, fp, border, true)
}

/// `skimage.morphology.erosion(image, footprint, mode=, cval=)`.
pub fn erosion(img: &[f64], rows: usize, cols: usize, fp: &Footprint, border: Border) -> Result<Vec<f64>, String> {
    filter(img, rows, cols, fp, border, false)
}

/// `skimage.morphology.opening`: the erosion, then the dilation with the mirrored footprint.
pub fn opening(img: &[f64], rows: usize, cols: usize, fp: &Footprint, border: Border) -> Result<Vec<f64>, String> {
    let e = erosion(img, rows, cols, fp, border)?;
    dilation(&e, rows, cols, &fp.mirrored(), border)
}

/// `skimage.morphology.closing`: the dilation, then the erosion with the mirrored footprint.
pub fn closing(img: &[f64], rows: usize, cols: usize, fp: &Footprint, border: Border) -> Result<Vec<f64>, String> {
    let d = dilation(img, rows, cols, fp, border)?;
    erosion(&d, rows, cols, &fp.mirrored(), border)
}

/// `skimage.morphology.white_tophat`: the image minus its opening.
pub fn white_tophat(img: &[f64], rows: usize, cols: usize, fp: &Footprint, border: Border) -> Result<Vec<f64>, String> {
    let mut o = opening(img, rows, cols, fp, border)?;
    for (o, &v) in o.iter_mut().zip(img) {
        *o = v - *o;
    }
    Ok(o)
}

/// `skimage.morphology.black_tophat`: the closing minus the image.
pub fn black_tophat(img: &[f64], rows: usize, cols: usize, fp: &Footprint, border: Border) -> Result<Vec<f64>, String> {
    let mut c = closing(img, rows, cols, fp, border)?;
    for (c, &v) in c.iter_mut().zip(img) {
        *c -= v;
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_matches_skimage() {
        // skimage.morphology.disk(3) has 29 cells; disk(1) is the cross
        assert_eq!(Footprint::disk(3).offsets.len(), 29);
        assert_eq!(Footprint::disk(1).offsets.len(), 5);
        assert_eq!(Footprint::disk(0).offsets, vec![(0, 0)]);
    }

    #[test]
    fn dilation_and_erosion_with_reflect() {
        // 1 x 5 row, footprint 1 x 3
        let img = [1.0, 5.0, 2.0, 2.0, 0.0];
        let fp = Footprint::from_mask(&[true; 3], 1, 3).unwrap();
        assert_eq!(
            dilation(&img, 1, 5, &fp, Border::Reflect).unwrap(),
            vec![5.0, 5.0, 5.0, 2.0, 2.0]
        );
        assert_eq!(
            erosion(&img, 1, 5, &fp, Border::Reflect).unwrap(),
            vec![1.0, 1.0, 2.0, 0.0, 0.0]
        );
        // constant 9 outside: the dilation of the ends is 9, the erosion is as before
        assert_eq!(
            dilation(&img, 1, 5, &fp, Border::Constant(9.0)).unwrap(),
            vec![9.0, 5.0, 5.0, 2.0, 9.0]
        );
    }

    #[test]
    fn border_maps() {
        assert_eq!(Border::Reflect.map(-1, 4), Some(0));
        assert_eq!(Border::Reflect.map(-2, 4), Some(1));
        assert_eq!(Border::Reflect.map(4, 4), Some(3));
        assert_eq!(Border::Reflect.map(9, 4), Some(1));
        assert_eq!(Border::Mirror.map(-1, 4), Some(1));
        assert_eq!(Border::Mirror.map(4, 4), Some(2));
        assert_eq!(Border::Wrap.map(-1, 4), Some(3));
        assert_eq!(Border::Nearest.map(7, 4), Some(3));
        assert_eq!(Border::Constant(0.0).map(-1, 4), None);
    }

    #[test]
    fn tophats_pick_peaks_and_pits() {
        let mut img = vec![1.0; 25];
        img[12] = 4.0;
        let fp = Footprint::square(3).unwrap();
        let w = white_tophat(&img, 5, 5, &fp, Border::Reflect).unwrap();
        assert_eq!(w[12], 3.0);
        assert_eq!(w.iter().sum::<f64>(), 3.0);
        let mut img = vec![4.0; 25];
        img[12] = 1.0;
        let b = black_tophat(&img, 5, 5, &fp, Border::Reflect).unwrap();
        assert_eq!(b[12], 3.0);
        assert_eq!(b.iter().sum::<f64>(), 3.0);
    }

    #[test]
    fn even_footprints_are_refused() {
        assert!(Footprint::from_mask(&[true; 4], 2, 2).is_err());
    }
}
