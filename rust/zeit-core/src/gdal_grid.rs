//! Inverse distance weighting of scattered points onto a regular grid, as GDAL's `GDALGridCreate` does it for the
//! algorithm `"invdist"` with its default options (power 2, smoothing 0, no search ellipse, no point limit, nodata 0):
//! the way FORCE 3.11.00's `interpolate_aod_map` (`aod-ll.c`) calls it. A port of GDAL 3.9.3 (the version of the conda
//! env `force-ref` FORCE is compared against), `alg/gdalgrid.cpp` and `alg/gdalgridavx.cpp`, Copyright (c) 2007 Andrey
//! Kiselev, (c) 2009-2013 Even Rouault, MIT license (see `THIRD_PARTY_NOTICES.md`).
//!
//! With those options `GDALGridContextCreate` picks `GDALGridInverseDistanceToAPower2NoSmoothingNoSearchAVX` (if GDAL
//! was built with AVX and the CPU has it, which holds for every x86-64 CPU since 2011; `GDAL_USE_AVX=NO` selects the
//! SSE version instead, which has four lanes, and a CPU without AVX the same). That function works in `float`, not in
//! `double`: the coordinates and values are rounded to `float`, sixteen points at a time are summed into eight lanes
//! of `float` (two vectors of eight, each lane adds the two vectors' terms one after the other), the lanes are added
//! left to right, the points that remain are added one by one, and the reciprocal `1 / r2` of the vector loop is the
//! hardware's approximation `_mm256_rcp_ps` (relative error up to 1.5 * 2^-12), not a division. This module does the
//! same operation by operation.
//!
//! * **The reciprocal.** `RCPPS`/`RCPSS` returns an approximation from a table in the CPU: Intel's and AMD's tables
//!   differ, so GDAL (and FORCE) give slightly different AOD maps on the two. On x86-64 this port executes the same
//!   instruction (`_mm_rcp_ss`, the scalar form; same table), so on a given CPU it is GDAL's bits. On other
//!   architectures `1 / x` (exact) is used, which differs from GDAL's result by about 1e-3 in relative terms.
//! * **Threads.** GDAL splits the rows of the grid among `GDAL_NUM_THREADS` threads; each cell is computed by one
//!   thread from all the points in the same order, so the result does not depend on the number of threads. This port is
//!   sequential.
//! * **Not ported.** The other algorithms and options of `GDALGridCreate` (the search ellipse, `max_points`,
//!   `min_points`, other powers, smoothing, the quadtree for nearest neighbours): FORCE does not use them. The vector
//!   code is written lane by lane, so the number of points is not limited to what an `f32` index holds, and the points
//!   are not required to be aligned.

/// The reciprocal as the vector loop of GDAL computes it (`_mm256_rcp_ps`): the CPU's approximation on x86-64, `1 / x`
/// elsewhere.
#[inline(always)]
fn rcp(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_rcp_ss, _mm_set_ss};
        // SAFETY: SSE is part of the x86-64 baseline.
        #[allow(unused_unsafe)]
        unsafe {
            _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0 / x
    }
}

/// Whether the CPU is an Intel one, whose `rcpps` table is the one the values of GDAL and FORCE in the tests were
/// printed with (an AMD CPU has another table; elsewhere `rcp` is `1 / x`): the tests compare the bits only there and
/// to the approximation (1.5 * 2^-12 per weight) otherwise.
pub fn rcp_is_intel() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: CPUID leaf 0 exists on every x86-64 CPU.
        #[allow(unused_unsafe)]
        let r = unsafe { std::arch::x86_64::__cpuid(0) };
        let mut v = [0u8; 12];
        v[..4].copy_from_slice(&r.ebx.to_le_bytes());
        v[4..8].copy_from_slice(&r.edx.to_le_bytes());
        v[8..].copy_from_slice(&r.ecx.to_le_bytes());
        &v == b"GenuineIntel"
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `GDALGridInverseDistanceToAPower2NoSmoothingNoSearchAVX`: the value at the node `(x_point, y_point)` from the
/// points `(px, py, pz)` already rounded to `f32`; `nodata` where no point contributes.
fn node_avx(px: &[f32], py: &[f32], pz: &[f32], x_point: f64, y_point: f64, nodata: f64) -> f64 {
    const LOOP_SIZE: usize = 16;
    let n = px.len();
    let epsilon = 0.0000000000001f32;
    let fx = x_point as f32;
    let fy = y_point as f32;
    let mut nom = [0.0f32; 8];
    let mut den = [0.0f32; 8];
    let rounded = n / LOOP_SIZE * LOOP_SIZE;
    let mut i = 0;
    while i < rounded {
        let mut r2 = [0.0f32; LOOP_SIZE];
        for l in 0..LOOP_SIZE {
            let rx = px[i + l] - fx;
            let ry = py[i + l] - fy;
            r2[l] = rx * rx + ry * ry;
        }
        // if (r2 < fEpsilon): the first lane of the block that is, whatever the others hold
        if let Some(j) = r2.iter().position(|&v| v < epsilon) {
            return pz[i + j] as f64;
        }
        for l in 0..8 {
            let inv0 = rcp(r2[l]);
            let inv1 = rcp(r2[l + 8]);
            nom[l] += inv0 * pz[i + l];
            nom[l] += inv1 * pz[i + 8 + l];
            den[l] += inv0;
            den[l] += inv1;
        }
        i += LOOP_SIZE;
    }

    let mut nominator = nom[0] + nom[1] + nom[2] + nom[3] + nom[4] + nom[5] + nom[6] + nom[7];
    let mut denominator = den[0] + den[1] + den[2] + den[3] + den[4] + den[5] + den[6] + den[7];

    // the few remaining points (the test is against a double here, against a float in the vector loop)
    while i < n {
        let rx = px[i] - fx;
        let ry = py[i] - fy;
        let r2 = rx * rx + ry * ry;
        if (r2 as f64) < 0.0000000000001 {
            return pz[i] as f64;
        }
        let inv = 1.0f32 / r2;
        nominator += inv * pz[i];
        denominator += inv;
        i += 1;
    }

    if denominator == 0.0 {
        nodata
    } else {
        (nominator / denominator) as f64
    }
}

/// `GDALGridCreate(GGA_InverseDistanceToAPower, <"invdist" defaults>, n, x, y, z, xmin, xmax, ymin, ymax, nx, ny,
/// GDT_Float64, ...)`: the grid of `nx * ny` values, row by row from the top (`ymin`), of the nodes `xmin + (ix +
/// 0.5) * (xmax - xmin) / nx` and `ymin + (iy + 0.5) * (ymax - ymin) / ny`; a node on a point (`r2 < 1e-13`)
/// takes the value of that point, a node with no points the nodata value, 0.
#[allow(clippy::too_many_arguments)]
pub fn invdist_grid(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    assert!(x.len() == y.len() && x.len() == z.len(), "x, y and z need the same length");
    assert!(nx > 0 && ny > 0, "the grid needs a positive size");
    let px: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let py: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let pz: Vec<f32> = z.iter().map(|&v| v as f32).collect();
    let dx = (xmax - xmin) / nx as f64;
    let dy = (ymax - ymin) / ny as f64;
    let mut out = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        let y_point = ymin + (iy as f64 + 0.5) * dy;
        for ix in 0..nx {
            let x_point = xmin + (ix as f64 + 0.5) * dx;
            out.push(node_avx(&px, &py, &pz, x_point, y_point, 0.0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_points_is_nodata_and_a_node_on_a_point_is_that_point() {
        assert_eq!(invdist_grid(&[], &[], &[], 0.0, 3.0, 0.0, 2.0, 3, 2), vec![0.0; 6]);
        // the nodes are at (0.5 * 2 / 2 ..): x = 0.5 and 1.5, y = 0.5 and 1.5 for a 2 x 2 grid over [0, 2]
        let g = invdist_grid(&[0.5], &[1.5], &[0.25], 0.0, 2.0, 0.0, 2.0, 2, 2);
        assert_eq!(g, vec![0.25; 4]);
        let g = invdist_grid(&[0.5, 1.5], &[0.5, 0.5], &[2.0, 4.0], 0.0, 2.0, 0.0, 2.0, 2, 2);
        assert_eq!((g[0], g[1]), (2.0, 4.0));
    }

    #[test]
    fn the_weights_are_inverse_squared_distances() {
        // two points, a node at distances 1 and 2: (a / 1 + b / 4) / (1 + 1/4)
        let g = invdist_grid(&[0.0, 3.0], &[0.0, 0.0], &[10.0, 20.0], 1.0, 2.0, 0.0, 1.0, 1, 1);
        // the node is (1.5, 0.5): r2 = 1.5 for both, so the mean
        assert!((g[0] - 15.0).abs() < 1e-2, "{}", g[0]);
        let g = invdist_grid(&[0.0, 4.0], &[0.5, 0.5], &[10.0, 20.0], 1.0, 2.0, 0.0, 1.0, 1, 1);
        let (r1, r2) = (1.5f64 * 1.5, 2.5f64 * 2.5);
        let want = (10.0 / r1 + 20.0 / r2) / (1.0 / r1 + 1.0 / r2);
        // the reciprocal is the CPU's approximation, good to 2^-11
        assert!((g[0] - want).abs() < want * 2e-3, "{} vs {}", g[0], want);
    }

    #[test]
    fn the_blocks_of_sixteen_and_the_remainder_agree_in_value() {
        // many points: the vector loop (16 at a time) and the tail both run; compare with the double sum
        let n = 37;
        let x: Vec<f64> = (0..n).map(|i| (i * 7 % 13) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i * 5 % 11) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| 0.0625 * (i % 9) as f64 + 0.03125).collect();
        let g = invdist_grid(&x, &y, &z, 0.0, 13.0, 0.0, 11.0, 13, 11);
        for iy in 0..11 {
            for ix in 0..13 {
                let (xp, yp) = (0.0 + (ix as f64 + 0.5), 0.0 + (iy as f64 + 0.5));
                let (mut a, mut b) = (0.0, 0.0);
                for k in 0..n {
                    let r2 = (x[k] - xp).powi(2) + (y[k] - yp).powi(2);
                    a += z[k] / r2;
                    b += 1.0 / r2;
                }
                let want = a / b;
                assert!((g[iy * 13 + ix] - want).abs() < want * 4e-3, "{ix},{iy}: {} vs {want}", g[iy * 13 + ix]);
            }
        }
    }

    /// Values GDAL 3.9.3 (`GDALGridCreate`, algorithm `invdist`, its AVX path, an Intel CPU) printed for the same
    /// points: the bits of the `f64` results, whose low 29 bits are zero (they are `float`s).
    #[test]
    fn values_of_gdal() {
        // 20 points, one block of sixteen and a tail of four: the cells g of a 6 x 5 grid with g % 3 != 0
        let cells: Vec<usize> = (0..30).filter(|g| g % 3 != 0).collect();
        let x: Vec<f64> = cells.iter().map(|g| (g % 6) as f64).collect();
        let y: Vec<f64> = cells.iter().map(|g| (g / 6) as f64).collect();
        let z: Vec<f64> = cells.iter().map(|g| 0.0625 * (g % 7 + 1) as f64).collect();
        let g = invdist_grid(&x, &y, &z, 0.0, 5.0, 0.0, 4.0, 6, 5);
        same_or_close(
            &g,
            &[
                0x3fc42d9840000000, 0x3fc2abe0e0000000, 0x3fc6bb18a0000000, 0x3fcbb9ed00000000, 0x3fd17bac20000000, 0x3fd3359be0000000, 0x3fc77f8800000000, 0x3fc0d6d700000000, 0x3fc1f31560000000, 0x3fca533680000000, 0x3fcf0761e0000000, 0x3fd129be20000000, 0x3fd4020aa0000000, 0x3fd75b9320000000, 0x3fb22e6980000000, 0x3fca94ec40000000, 0x3fc91c0200000000, 0x3fcd164020000000, 0x3fd4cc0aa0000000, 0x3fd62be600000000, 0x3fd8ed4800000000, 0x3fceb60480000000, 0x3fc41e3d80000000, 0x3fc7a23220000000, 0x3fd423b0c0000000, 0x3fd4e32400000000, 0x3fd63a5bc0000000, 0x3fcfdf64a0000000, 0x3fc1809100000000, 0x3fc34d8d00000000
            ],
        );
        // 4 points: only the tail
        let g = invdist_grid(&[0.0, 4.0, 2.0, 1.0], &[0.0, 3.0, 1.0, 3.0], &[0.25, 0.5, 0.125, 0.75], 0.0, 4.0, 0.0, 3.0, 5, 4);
        same_or_close(
            &g,
            &[
                0x3fd0a69e80000000, 0x3fcdbb3b00000000, 0x3fc61891a0000000, 0x3fcbeadec0000000, 0x3fd3048fe0000000, 0x3fd458b9e0000000, 0x3fce6a7b80000000, 0x3fc06ba6a0000000, 0x3fcbbb2aa0000000, 0x3fd5447aa0000000, 0x3fdea08ce0000000, 0x3fdb804ba0000000, 0x3fd33e1c00000000, 0x3fd543f460000000, 0x3fdac89740000000, 0x3fe534f740000000, 0x3fe684a3a0000000, 0x3fe11786a0000000, 0x3fdd2e23a0000000, 0x3fdf2a2b20000000
            ],
        );
    }

    /// The bits on an Intel CPU (where they were printed), the values to the approximation of `rcpps` elsewhere.
    fn same_or_close(got: &[f64], want_bits: &[u64]) {
        assert_eq!(got.len(), want_bits.len());
        if rcp_is_intel() {
            let bits: Vec<u64> = got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, want_bits);
        } else {
            for (g, &w) in got.iter().zip(want_bits) {
                let w = f64::from_bits(w);
                assert!((g - w).abs() <= 2e-3 * w.abs(), "{g} vs {w}");
            }
        }
    }
}
