//! Timing of the spectral adjustment on random realistic spectra (vegetation, soil, water, snow, urban):
//! `cargo run -p zeit-core --release --example specadjust_profile`.
use std::time::Instant;
use zeit_core::specadjust::{spectral_adjust_batch, spectral_adjust_batch_as, Quirk, Sensor};

fn main() {
    let n = 2_000_000usize;
    let mut s = 12345u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    // BLUE GREEN RED NIR SWIR1 SWIR2, x 10000
    let kinds: [[f64; 6]; 5] = [
        [400.0, 700.0, 500.0, 3800.0, 1800.0, 900.0],
        [1200.0, 1700.0, 2200.0, 2900.0, 3800.0, 3200.0],
        [600.0, 700.0, 400.0, 200.0, 100.0, 50.0],
        [8000.0, 8200.0, 8000.0, 7000.0, 1500.0, 900.0],
        [1500.0, 1700.0, 1900.0, 2300.0, 2600.0, 2200.0],
    ];
    let src: Vec<f32> = (0..n)
        .flat_map(|i| {
            let k = kinds[(rnd() * 5.0) as usize % 5];
            let scale = 0.6 + 0.8 * rnd();
            let _ = i;
            (0..6).map(|b| (k[b] * scale * (0.9 + 0.2 * rnd())) as f32).collect::<Vec<_>>()
        })
        .collect();
    let mut out = vec![0.0f64; n * 10];
    let mut sink = 0.0;
    for (name, quirk) in [("fixed", Quirk::Fixed), ("force", Quirk::Force)] {
        for threads in [1, 8] {
            let mut best = f64::INFINITY;
            for _ in 0..4 {
                let t = Instant::now();
                spectral_adjust_batch_as(&src, Sensor::Lnd08, &mut out, threads, quirk);
                best = best.min(t.elapsed().as_secs_f64());
                sink += out[7];
            }
            println!("LND08 {name} {threads} thread(s): {:.3} s, {:.2} Mpixel/s, {:.3} us/px", best, n as f64 / best / 1e6, best / n as f64 * 1e6);
        }
    }
    let _ = spectral_adjust_batch::<f32>;
    println!("{sink}");
}
