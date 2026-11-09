// SPDX-License-Identifier: GPL-3.0-or-later
//! The DST product (distance to the nearest cloud, shadow or snow) from the QAI. The rule comes from a
//! specification written from FORCE's public documentation and from the product's outputs, in a clean room.

use crate::aod::dist_transform;
use crate::gas::SceneGrid;
use crate::qai;

/// Distance to the nearest target in map units, `int16`, `nodata` at off pixels.
///
/// Targets are pixels (not off) with cloud state 1, 2 or 3, shadow or snow; water is not a target. Target pixels
/// give 0; the others give `floor(d_pixels) * res`, with `d_pixels` the exact Euclidean distance to the centre of
/// the nearest target inside the grid (off pixels are plain non-targets).
///
/// Cases the specification leaves unobserved:
/// - no target in the image: every non-off pixel gets `i16::MAX`;
/// - a distance that does not fit `int16`: saturates to `i16::MAX`;
/// - `res != 10`: the floor is taken in pixels, then multiplied by `res` (rounded to the nearest integer);
/// - off pixel with other bits set: still off, so the output is `nodata`.
pub fn cloud_distance(qai: &[i16], grid: &SceneGrid, nodata: i16) -> Vec<i16> {
    let (nx, ny) = (grid.nx, grid.ny);
    assert_eq!(qai.len(), nx * ny, "cloud_distance: qai length does not match the grid");
    let target: Vec<u8> = qai
        .iter()
        .map(|&v| (!qai::off(v) && (qai::cloud(v) != 0 || qai::shadow(v) || qai::snow(v))) as u8)
        .collect();
    let any = target.iter().any(|&t| t != 0);
    let dist = if any { dist_transform(&target, nx, ny) } else { Vec::new() };
    (0..nx * ny)
        .map(|i| {
            if qai::off(qai[i]) {
                nodata
            } else if target[i] != 0 {
                0
            } else if !any {
                i16::MAX
            } else {
                (dist[i] as f64 * grid.res).round().min(i16::MAX as f64) as i16
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ND: i16 = -9999;

    fn grid(nx: usize, ny: usize, res: f64) -> SceneGrid {
        SceneGrid { nx, ny, res }
    }

    // bit layout: off = 0, cloud = 1-2, shadow = 3, snow = 4, water = 5
    fn one(nx: usize, ny: usize, x: usize, y: usize, v: i16) -> Vec<i16> {
        let mut q = vec![0i16; nx * ny];
        q[y * nx + x] = v;
        q
    }

    #[test]
    fn each_class_is_a_target() {
        for v in [1i16 << 1, 2 << 1, 3 << 1, 1 << 3, 1 << 4] {
            let q = one(5, 1, 0, 0, v);
            let d = cloud_distance(&q, &grid(5, 1, 10.0), ND);
            assert_eq!(d, vec![0, 10, 20, 30, 40], "class {v}");
        }
    }

    #[test]
    fn water_is_not_a_target() {
        let q = one(4, 1, 0, 0, 1 << 5);
        let d = cloud_distance(&q, &grid(4, 1, 10.0), ND);
        assert_eq!(d, vec![i16::MAX; 4]);
        // water together with a target class is still a target
        let q = one(4, 1, 0, 0, (1 << 5) | (1 << 3));
        assert_eq!(cloud_distance(&q, &grid(4, 1, 10.0), ND), vec![0, 10, 20, 30]);
    }

    #[test]
    fn off_pixels_are_nodata_and_block_nothing() {
        let mut q = one(5, 1, 0, 0, 1 << 3);
        q[1] = 1;
        q[2] = 1 | (1 << 3); // off with a target bit: nodata, not a target
        let d = cloud_distance(&q, &grid(5, 1, 10.0), ND);
        assert_eq!(d, vec![0, ND, ND, 30, 40]);
    }

    #[test]
    fn off_is_not_a_target() {
        // the off pixel at the end carries a snow bit but does not count as a target
        let mut q = one(4, 1, 0, 0, 1 << 4);
        q[3] = 1 | (1 << 4);
        let d = cloud_distance(&q, &grid(4, 1, 10.0), ND);
        assert_eq!(d, vec![0, 10, 20, ND]);
        // and an off pixel alone leaves the image without targets
        let q = one(3, 1, 1, 0, 1);
        assert_eq!(cloud_distance(&q, &grid(3, 1, 10.0), ND), vec![i16::MAX, ND, i16::MAX]);
    }

    #[test]
    fn truncation_with_a_one_one_offset() {
        // distance sqrt(2) = 1.41 pixels: floor gives 1, so 10 m (rounding would also give 1; the 2,1 case differs)
        let q = one(4, 4, 0, 0, 1 << 3);
        let d = cloud_distance(&q, &grid(4, 4, 10.0), ND);
        assert_eq!(d[4 + 1], 10);
        // (2,1): sqrt(5) = 2.24 -> 2; (2,2): 2.83 -> 2 (rounding would give 3)
        assert_eq!(d[4 + 2], 20);
        assert_eq!(d[2 * 4 + 2], 20);
        // (3,3): 4.24 -> 4
        assert_eq!(d[3 * 4 + 3], 40);
    }

    #[test]
    fn border_targets_only_inside_the_grid() {
        // a target in the corner: the opposite corner is at the full distance, nothing mirrored
        let q = one(6, 3, 0, 0, 1 << 1);
        let d = cloud_distance(&q, &grid(6, 3, 10.0), ND);
        assert_eq!(d[0], 0);
        assert_eq!(d[5], 50);
        assert_eq!(d[2 * 6 + 5], 50); // sqrt(25+4) = 5.38 -> 5
    }

    #[test]
    fn other_resolution_multiplies_after_the_floor() {
        let q = one(4, 4, 0, 0, 1 << 3);
        let d = cloud_distance(&q, &grid(4, 4, 20.0), ND);
        assert_eq!(d[2 * 4 + 2], 40); // floor(2.83) = 2, times 20
        assert_eq!(d[3], 60);
    }

    #[test]
    fn no_target_and_overflow_saturate() {
        let q = vec![0i16; 6];
        assert_eq!(cloud_distance(&q, &grid(3, 2, 10.0), ND), vec![i16::MAX; 6]);
        let q = one(4, 1, 0, 0, 1 << 3);
        let d = cloud_distance(&q, &grid(4, 1, 20000.0), ND);
        assert_eq!(d, vec![0, 20000, i16::MAX, i16::MAX]);
    }
}
