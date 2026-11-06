//! Timing of the NBAR c-factor on random geometries (Sentinel-2-like: sza 20-60, vza 0-11):
//! `cargo run -p zeit-core --release --example nbar_profile`.
use std::time::Instant;
use zeit_core::brdf::{self, Geometry};

fn main() {
    let n = 4_000_000usize;
    let mut s = 12345u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let sza: Vec<f32> = (0..n).map(|_| (20.0 + 40.0 * rnd()) as f32).collect();
    let saa: Vec<f32> = (0..n).map(|_| (100.0 + 40.0 * rnd()) as f32).collect();
    let vza: Vec<f32> = (0..n).map(|_| (11.0 * rnd()) as f32).collect();
    let vaa: Vec<f32> = (0..n).map(|_| (360.0 * rnd()) as f32).collect();
    let geo = Geometry { sza: &sza, saa: &saa, vza: &vza, vaa: &vaa };
    let one_band: Vec<f32> = (0..n).map(|_| (0.4 * rnd()) as f32).collect();
    let ten: Vec<f32> = (0..10).flat_map(|_| one_band.iter().copied()).collect();
    let coefs: Vec<_> = brdf::ROLES.iter().skip(1).map(|r| brdf::role(r).unwrap()).collect();
    let mut out1 = vec![0.0f32; n];
    let mut out10 = vec![0.0f32; 10 * n];
    let mut sink = 0.0f32;
    for threads in [1, 8] {
        let mut best1 = f64::INFINITY;
        let mut best10 = f64::INFINITY;
        for _ in 0..4 {
            let t = Instant::now();
            brdf::nbar_batch(&one_band, &geo, brdf::RED, 35.0, &mut out1, threads);
            best1 = best1.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            brdf::nbar_bands(&ten, &geo, &coefs, 35.0, &mut out10, None, threads);
            best10 = best10.min(t.elapsed().as_secs_f64());
            sink += out1[7] + out10[n + 3];
        }
        println!(
            "{threads} thread(s): 1 band {:.3} s, {:.1} Mpixel/s; 10 bands {:.3} s, {:.1} Mpixel/s, {:.1} Mband-pixel/s",
            best1,
            n as f64 / best1 / 1e6,
            best10,
            n as f64 / best10 / 1e6,
            10.0 * n as f64 / best10 / 1e6
        );
    }
    println!("{sink}");
}
