//! Connected-component labelling (`scipy.ndimage.label`, `skimage.measure.label`),
//! `skimage.filters.apply_hysteresis_threshold`, `skimage.segmentation.find_boundaries(mode="thick")`,
//! `skimage.segmentation.relabel_sequential` and `skimage.morphology.remove_small_objects`.
//!
//! Components are numbered in raster order of their first pixel, as both libraries do. `connectivity`
//! is skimage's: 1 is 4-connected, 2 or more (clamped to the 2 dimensions) 8-connected.

use super::watershed::neighbour_offsets;

/// Components of equal value (`same_value`, background 0: `skimage.measure.label` of an integer
/// image) or of any non-zero value (`ndimage.label`, `skimage.measure.label` of a boolean image).
/// Returns the labels (0 background) and their number.
pub fn label_ints(
    img: &[i32],
    rows: usize,
    cols: usize,
    connectivity: usize,
    same_value: bool,
) -> Result<(Vec<i32>, usize), String> {
    if img.len() != rows * cols {
        return Err("the image must have rows * cols values".into());
    }
    let offsets = neighbour_offsets(connectivity);
    let mut out = vec![0i32; img.len()];
    let mut n = 0usize;
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..img.len() {
        if img[start] == 0 || out[start] != 0 {
            continue;
        }
        n += 1;
        out[start] = n as i32;
        stack.push(start as u32);
        while let Some(i) = stack.pop() {
            let i = i as usize;
            let (r, c) = ((i / cols) as isize, (i % cols) as isize);
            for &(dr, dc) in offsets {
                let (nr, nc) = (r + dr, c + dc);
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let j = nr as usize * cols + nc as usize;
                if out[j] != 0 || img[j] == 0 || (same_value && img[j] != img[start]) {
                    continue;
                }
                out[j] = n as i32;
                stack.push(j as u32);
            }
        }
    }
    Ok((out, n))
}

/// The components of a boolean image.
pub fn label_mask(mask: &[bool], rows: usize, cols: usize, connectivity: usize) -> Result<(Vec<i32>, usize), String> {
    let v: Vec<i32> = mask.iter().map(|&b| b as i32).collect();
    label_ints(&v, rows, cols, connectivity, false)
}

/// `apply_hysteresis_threshold(image, low, high)`: the 4-connected components of `image > low` that
/// hold a pixel `> high` (`low` is clipped to `high`, as in scikit-image).
pub fn hysteresis_threshold(image: &[f64], rows: usize, cols: usize, low: f64, high: f64) -> Result<Vec<bool>, String> {
    if image.len() != rows * cols {
        return Err("the image must have rows * cols values".into());
    }
    let low = low.min(high);
    let mut out = vec![false; image.len()];
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..image.len() {
        if !(image[start] > high) || out[start] {
            continue;
        }
        out[start] = true;
        stack.push(start as u32);
        while let Some(i) = stack.pop() {
            let i = i as usize;
            let (r, c) = (i / cols, i % cols);
            let mut visit = |j: usize| {
                if !out[j] && image[j] > low {
                    out[j] = true;
                    stack.push(j as u32);
                }
            };
            if r > 0 {
                visit(i - cols);
            }
            if c > 0 {
                visit(i - 1);
            }
            if c + 1 < cols {
                visit(i + 1);
            }
            if r + 1 < rows {
                visit(i + cols);
            }
        }
    }
    Ok(out)
}

/// `find_boundaries(labels, connectivity, mode="thick")`: the pixels whose neighbourhood (itself
/// included) holds more than one label, that is `dilation != erosion`; the border reflects, so
/// nothing is a boundary because of it.
pub fn find_boundaries(labels: &[i32], rows: usize, cols: usize, connectivity: usize) -> Result<Vec<bool>, String> {
    if labels.len() != rows * cols {
        return Err("the labels must have rows * cols values".into());
    }
    let offsets = neighbour_offsets(connectivity);
    let mut out = vec![false; labels.len()];
    for r in 0..rows {
        for c in 0..cols {
            let v = labels[r * cols + c];
            out[r * cols + c] = offsets.iter().any(|&(dr, dc)| {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                nr >= 0
                    && nc >= 0
                    && nr < rows as isize
                    && nc < cols as isize
                    && labels[nr as usize * cols + nc as usize] != v
            });
        }
    }
    Ok(out)
}

/// `relabel_sequential(labels, offset)`: 0 stays 0 and the other values become `offset`,
/// `offset + 1`, ... in increasing order. Returns the new labels and the old values in the order of
/// the new ones (`inverse[k]` is the old value of label `offset + k`). Negative labels are refused.
pub fn relabel_sequential(labels: &[i32], offset: i32) -> Result<(Vec<i32>, Vec<i32>), String> {
    if offset <= 0 {
        return Err("Offset must be strictly positive.".into());
    }
    if labels.iter().any(|&l| l < 0) {
        return Err("Cannot relabel array that contains negative values.".into());
    }
    let mut uniq: Vec<i32> = labels.iter().copied().filter(|&l| l != 0).collect();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() as i64 + offset as i64 - 1 > i32::MAX as i64 {
        return Err("the new labels overflow int32".into());
    }
    let out = labels
        .iter()
        .map(|&l| {
            if l == 0 {
                0
            } else {
                offset + uniq.binary_search(&l).unwrap() as i32
            }
        })
        .collect();
    Ok((out, uniq))
}

/// `remove_small_objects` of a label image: the labels (any non-zero value, whatever their
/// connectivity) with `max_size` pixels or fewer are set to 0 (skimage 0.26's `max_size`;
/// the older `min_size` is `max_size = min_size - 1`).
pub fn remove_small_labels(labels: &[i32], max_size: usize) -> Result<Vec<i32>, String> {
    if labels.iter().any(|&l| l < 0) {
        return Err("Negative value labels are not supported.".into());
    }
    let mut out = labels.to_vec();
    if max_size == 0 {
        return Ok(out);
    }
    let top = labels.iter().copied().max().unwrap_or(0).max(0) as usize;
    let mut size = vec![0usize; top + 1];
    for &l in labels {
        size[l as usize] += 1;
    }
    for l in out.iter_mut() {
        if size[*l as usize] <= max_size {
            *l = 0;
        }
    }
    Ok(out)
}

/// `remove_small_objects` of a boolean image: the components (of `connectivity`) with `max_size`
/// pixels or fewer are cleared.
pub fn remove_small_objects_mask(
    mask: &[bool],
    rows: usize,
    cols: usize,
    max_size: usize,
    connectivity: usize,
) -> Result<Vec<bool>, String> {
    if max_size == 0 {
        return Ok(mask.to_vec());
    }
    let (lab, n) = label_mask(mask, rows, cols, connectivity)?;
    let mut size = vec![0usize; n + 1];
    for &l in &lab {
        size[l as usize] += 1;
    }
    Ok(lab
        .iter()
        .zip(mask)
        .map(|(&l, &m)| m && size[l as usize] > max_size)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_in_raster_order_and_connectivity() {
        // 1 0 0 / 0 1 0 / 0 0 1: three 4-connected pieces, one 8-connected
        let m = [true, false, false, false, true, false, false, false, true];
        let (l4, n4) = label_mask(&m, 3, 3, 1).unwrap();
        assert_eq!((n4, l4), (3, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]));
        let (l8, n8) = label_mask(&m, 3, 3, 2).unwrap();
        assert_eq!((n8, l8), (1, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]));
    }

    #[test]
    fn equal_values_only_connect() {
        let v = [5, 5, 7, 5, 0, 7];
        let (l, n) = label_ints(&v, 2, 3, 1, true).unwrap();
        assert_eq!((n, l), (2, vec![1, 1, 2, 1, 0, 2]));
        let (l, n) = label_ints(&v, 2, 3, 1, false).unwrap();
        assert_eq!((n, l), (1, vec![1, 1, 1, 1, 0, 1]));
    }

    #[test]
    fn hysteresis_keeps_what_touches_high() {
        let img = [0.0, 0.5, 0.9, 0.0, 0.5, 0.5, 0.0, 0.0];
        let h = hysteresis_threshold(&img, 2, 4, 0.4, 0.8).unwrap();
        assert_eq!(h, vec![false, true, true, false, true, true, false, false]);
        let h = hysteresis_threshold(&[0.5, 0.0, 0.5, 0.0], 1, 4, 0.4, 0.8).unwrap();
        assert_eq!(h, vec![false; 4]);
    }

    #[test]
    fn boundaries_are_thick() {
        let l = [1, 1, 2, 2];
        assert_eq!(find_boundaries(&l, 1, 4, 1).unwrap(), vec![false, true, true, false]);
        assert_eq!(find_boundaries(&[3; 4], 2, 2, 2).unwrap(), vec![false; 4]);
    }

    #[test]
    fn relabel_and_remove_small() {
        let (o, inv) = relabel_sequential(&[0, 5, 5, 42, 0, 9], 1).unwrap();
        assert_eq!(o, vec![0, 1, 1, 3, 0, 2]);
        assert_eq!(inv, vec![5, 9, 42]);
        let (o, _) = relabel_sequential(&[0, 5, 9], 5).unwrap();
        assert_eq!(o, vec![0, 5, 6]);
        assert!(relabel_sequential(&[-1], 1).is_err());
        let r = remove_small_labels(&[1, 1, 2, 3, 3, 3], 1).unwrap();
        assert_eq!(r, vec![1, 1, 0, 3, 3, 3]);
        let m = [true, true, false, true];
        assert_eq!(
            remove_small_objects_mask(&m, 1, 4, 1, 1).unwrap(),
            vec![true, true, false, false]
        );
    }
}
