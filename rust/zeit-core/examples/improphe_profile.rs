//! Timing of ImproPhe on synthetic standardized components and a blocky coarse field:
//! `cargo run -p zeit-core --release --example improphe_profile`.
//! The kernel (`improphe_grid`) for h = 10, 25, 83 on 1 and 8 threads, then the stages before it on a larger image.
use std::time::Instant;
use zeit_core::improphe::{average_season, focal_sd, improphe_grid, min_neighbors, pca, standardize, Grid};

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Five components: smooth fields (a continuum of similar pixels, as land cover has) plus noise.
fn components(ny: usize, nx: usize, npc: usize, s: &mut u64) -> Vec<f32> {
    let npix = ny * nx;
    let mut pcs = vec![0f32; npc * npix];
    for b in 0..npc {
        let (fi, fj, ph) = (17.0 + 5.0 * b as f64, 23.0 - 3.0 * b as f64, b as f64);
        for i in 0..ny {
            for j in 0..nx {
                let v = ((i as f64 / fi + ph).sin() + (j as f64 / fj - ph).cos()) * 0.9 + (lcg(s) - 0.5) * 0.3;
                pcs[b * npix + i * nx + j] = v as f32;
            }
        }
    }
    pcs
}

fn main() {
    let mut s = 99u64;
    let npc = 5;
    let nb_mr = 1;
    let mut sink = 0.0f64;
    for (h, ny, nx) in [(10usize, 400usize, 400usize), (25, 300, 300), (83, 160, 160)] {
        let npix = ny * nx;
        let pcs = components(ny, nx, npc, &mut s);
        let mut tex = vec![0f32; npix];
        focal_sd(&pcs, ny, nx, 3, 0..npc, &mut tex, 8);
        // the coarse field: blocks of 7 x 7 pixels of a smooth function of the components
        let mr: Vec<f32> = (0..npix)
            .map(|p| {
                let (i, j) = (p / nx / 7 * 7, p % nx / 7 * 7);
                let q = i * nx + j;
                (1000.0 * (1.5 + f64::from(pcs[q]) + 0.5 * f64::from(pcs[npix + q]))) as f32
            })
            .collect();
        let mut mr_tex = vec![0f32; npix];
        focal_sd(&mr, ny, nx, 3, 0..1, &mut mr_tex, 8);
        let g = Grid {
            hr: &pcs,
            npc,
            hr_tex: &tex,
            mr: &mr,
            mr_tex: &mr_tex,
            nb_mr,
            ny,
            nx,
            h,
            mink: min_neighbors(h),
        };
        let mut out = vec![0f32; nb_mr * npix];
        for threads in [1, 8] {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                improphe_grid(&g, &mut out, threads);
                best = best.min(t.elapsed().as_secs_f64());
                sink += f64::from(out[npix / 2]);
            }
            println!(
                "h {h:3} ({ny}x{nx}, disc {} px, mink {}) {threads} thread(s): {best:.3} s, {:.0} pixel/s",
                (std::f64::consts::PI * (h * h) as f64) as usize,
                min_neighbors(h),
                npix as f64 / best
            );
        }
    }

    // the stages before the kernel: 12 scenes of 10 bands, 1500 x 1500 pixels (2.25 M)
    let (ny, nx, nt, nb) = (1500usize, 1500usize, 12usize, 10usize);
    let npix = ny * nx;
    let x: Vec<f32> = (0..nt * nb * npix)
        .map(|i| {
            let (b, p) = ((i / npix) % nb, i % npix);
            (1000.0
                + 150.0 * b as f64
                + 800.0 * ((p / nx) as f64 / 90.0).sin() * (1.0 + 0.1 * b as f64)
                + lcg(&mut s) * 60.0) as f32
        })
        .collect();
    let doy: Vec<i32> = (0..nt as i32).map(|t| 15 + 30 * t).collect();
    let year = vec![2020; nt];
    let dwin = [1, 91, 181, 271, 365];
    let mut seasonal = vec![0f32; 4 * nb * npix];
    for threads in [1, 8] {
        let t = Instant::now();
        average_season(
            &x,
            nt,
            nb,
            npix,
            &doy,
            &year,
            &dwin,
            Some(2020),
            true,
            &mut seasonal,
            threads,
        );
        let t_avg = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut p = pca(&seasonal, 4 * nb, npix, 0.975, true, threads);
        let t_pca = t.elapsed().as_secs_f64();
        let t = Instant::now();
        standardize(&mut p.pcs, npix, threads);
        let t_std = t.elapsed().as_secs_f64();
        let mut tex = vec![0f32; npix];
        let t = Instant::now();
        focal_sd(&p.pcs, ny, nx, 11, 0..p.npc, &mut tex, threads);
        let t_tex = t.elapsed().as_secs_f64();
        println!(
            "{threads} thread(s), {} px, {} scenes x {nb} bands, {} PCs: average_season {t_avg:.2} s, pca {t_pca:.2} s, standardize {t_std:.2} s, focal_sd (k = 11) {t_tex:.2} s",
            npix, nt, p.npc
        );
        sink += f64::from(tex[npix / 2]);
    }
    println!("{sink}");
}
