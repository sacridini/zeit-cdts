//! Marker-based watershed equal to `skimage.segmentation.watershed(image, markers,
//! connectivity=1|2, mask=mask)` (compactness 0, no watershed line).
//!
//! Like scikit-image (`_watershed_cy.pyx`), the pixels wait in a binary heap ordered by
//! (value, age), age being the order of entry into the queue (markers all 0), so a tie on a plateau
//! goes to the marker that reached it first. The heap is the same one (a plain binary heap, the same
//! sift up and down), because markers of equal value and age 0 leave it in an order that depends on
//! its layout, and that order decides later ties. Neighbours are visited as scikit-image sorts them:
//! by distance, then in raster order of the footprint: up, left, right, down, then the diagonals
//! (-1,-1), (-1,1), (1,-1), (1,1). The image is taken as float64, as scikit-image casts it.
//! In this simplest case (no compactness, no watershed line) scikit-image labels a pixel when it is
//! pushed, so it is pushed once, and a pushed value is raised to the value of the pixel it came from;
//! checked against `_watershed_cy.pyx` of v0.26.0.

#[derive(Clone, Copy)]
struct Item {
    value: f64,
    age: u64,
    index: u32,
}

#[inline]
fn smaller(a: &Item, b: &Item) -> bool {
    if a.value != b.value {
        a.value < b.value
    } else {
        a.age < b.age
    }
}

struct Heap {
    data: Vec<Item>,
}

impl Heap {
    fn push(&mut self, item: Item) {
        let mut child = self.data.len();
        self.data.push(item);
        while child > 0 {
            let parent = (child - 1) / 2;
            if smaller(&item, &self.data[parent]) {
                self.data[child] = self.data[parent];
                child = parent;
            } else {
                break;
            }
        }
        self.data[child] = item;
    }

    fn pop(&mut self) -> Option<Item> {
        let top = *self.data.first()?;
        let last = self.data.pop().unwrap();
        let n = self.data.len();
        if n == 0 {
            return Some(top);
        }
        self.data[0] = last;
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut smallest = i;
            if l < n {
                if smaller(&self.data[l], &self.data[i]) {
                    smallest = l;
                }
                if r < n && smaller(&self.data[r], &self.data[smallest]) {
                    smallest = r;
                }
            } else {
                break;
            }
            if smallest == i {
                break;
            }
            self.data.swap(i, smallest);
            i = smallest;
        }
        Some(top)
    }
}

const NEIGHBOURS_4: [(isize, isize); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];
const NEIGHBOURS_8: [(isize, isize); 8] = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)];

/// The neighbours of a pixel in scikit-image's order; `connectivity` <= 1 is 4-connected, more is
/// 8-connected (scikit-image clamps it to the number of dimensions).
pub fn neighbour_offsets(connectivity: usize) -> &'static [(isize, isize)] {
    if connectivity <= 1 {
        &NEIGHBOURS_4
    } else {
        &NEIGHBOURS_8
    }
}

/// The watershed of `image` flooded from `markers` (non-zero labels), inside `mask` (None: everywhere).
/// Markers outside the mask are dropped. Returns the labels (0 where nothing reached).
pub fn watershed(
    image: &[f64],
    markers: &[i32],
    rows: usize,
    cols: usize,
    connectivity: usize,
    mask: Option<&[bool]>,
) -> Result<Vec<i32>, String> {
    let n = rows * cols;
    if image.len() != n || markers.len() != n || mask.is_some_and(|m| m.len() != n) {
        return Err("image, markers and mask must have rows * cols values".into());
    }
    if n > u32::MAX as usize {
        return Err("the image has more than 2^32 pixels".into());
    }
    let in_mask = |i: usize| mask.is_none_or(|m| m[i]);
    let mut out: Vec<i32> = (0..n).map(|i| if in_mask(i) { markers[i] } else { 0 }).collect();
    let mut heap = Heap { data: Vec::new() };
    for i in 0..n {
        if out[i] != 0 {
            heap.push(Item {
                value: image[i],
                age: 0,
                index: i as u32,
            });
        }
    }
    let offsets = neighbour_offsets(connectivity);
    let mut age: u64 = 1;
    while let Some(elem) = heap.pop() {
        let idx = elem.index as usize;
        let label = out[idx];
        let (r, c) = ((idx / cols) as isize, (idx % cols) as isize);
        for &(dr, dc) in offsets {
            let (nr, nc) = (r + dr, c + dc);
            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let ni = nr as usize * cols + nc as usize;
            if !in_mask(ni) || out[ni] != 0 {
                continue;
            }
            age += 1;
            // labelled when pushed (the first pixel to reach it claims it), and never lower than
            // the pixel it came from
            out[ni] = label;
            let mut value = image[ni];
            if value < elem.value {
                value = elem.value;
            }
            heap.push(Item {
                value,
                age,
                index: ni as u32,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateau_ties_go_to_the_first_to_arrive() {
        let img = vec![0.0; 7];
        let mut m = vec![0; 7];
        m[0] = 1;
        m[6] = 2;
        let w = watershed(&img, &m, 1, 7, 1, None).unwrap();
        assert_eq!(w, vec![1, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn plateau_in_2d_follows_the_age_order() {
        let img = vec![0.0; 9];
        let mut m = vec![0; 9];
        m[0] = 1;
        m[8] = 2;
        let w = watershed(&img, &m, 3, 3, 1, None).unwrap();
        assert_eq!(w, vec![1, 1, 1, 1, 1, 2, 1, 2, 2]);
    }

    #[test]
    fn the_value_beats_the_age() {
        // the wall of 5 is crossed last: pixel 2 (value 1) is flooded by marker 2 before the wall
        let img = vec![0.0, 5.0, 1.0, 0.0];
        let m = vec![1, 0, 0, 2];
        let w = watershed(&img, &m, 1, 4, 1, None).unwrap();
        assert_eq!(w, vec![1, 1, 2, 2]);
    }

    #[test]
    fn the_mask_stops_the_flood_and_drops_outside_markers() {
        let img = vec![0.0; 5];
        let mask = [true, true, false, true, true];
        let m = vec![1, 0, 7, 0, 2];
        let w = watershed(&img, &m, 1, 5, 1, Some(&mask)).unwrap();
        assert_eq!(w, vec![1, 1, 0, 2, 2]);
    }

    #[test]
    fn a_marker_on_the_mask_edge_floods_inwards() {
        let img = vec![0.0; 6];
        let mask = [false, true, true, true, true, false];
        let m = vec![0, 3, 0, 0, 0, 0];
        let w = watershed(&img, &m, 1, 6, 1, Some(&mask)).unwrap();
        assert_eq!(w, vec![0, 3, 3, 3, 3, 0]);
    }

    #[test]
    fn diagonal_connectivity_crosses_corners() {
        // 4-connectivity cannot reach the corner pixel through a diagonal gap
        let img = vec![0.0; 4];
        let mask = [true, false, false, true];
        let m = vec![1, 0, 0, 0];
        assert_eq!(watershed(&img, &m, 2, 2, 1, Some(&mask)).unwrap(), vec![1, 0, 0, 0]);
        assert_eq!(watershed(&img, &m, 2, 2, 2, Some(&mask)).unwrap(), vec![1, 0, 0, 1]);
    }

    #[test]
    fn marker_labels_are_kept_and_shape_is_checked() {
        let img = vec![0.0; 4];
        assert!(watershed(&img, &[1, 0, 0], 2, 2, 1, None).is_err());
        let w = watershed(&img, &[-3, 0, 0, 0], 2, 2, 1, None).unwrap();
        assert_eq!(w, vec![-3; 4]);
    }
}
