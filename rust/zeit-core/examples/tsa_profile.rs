//! Single-thread timing of the TSA kernels on a synthetic cube (a step every 5 days, half of them missing):
//! `cargo run -p zeit-core --release --example tsa_profile`.
use std::time::Instant;
use zeit_core::tsa::interp::{interp_batch, Method};
use zeit_core::tsa::stats::{fold_batch, Metric};

fn best<F: FnMut()>(mut f: F) -> f64 {
    f();
    (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let (np, nt) = (20_000usize, 325usize);
    let mut s = 12345u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let t: Vec<f64> = (0..nt).map(|i| 737000.0 + 5.6 * i as f64).map(f64::floor).collect();
    let cube: Vec<f64> = (0..np * nt)
        .map(|_| if rnd() < 0.5 { f64::NAN } else { (10000.0 * rnd()).floor() - 1000.0 })
        .collect();
    let cube32: Vec<f32> = cube.iter().map(|&v| v as f32).collect();
    let grid: Vec<f64> = (0..114).map(|i| 737000.0 + 16.0 * i as f64).collect();
    let us = |sec: f64| sec / np as f64 * 1e6;
    let mut sink = 0.0;
    let r = best(|| {
        let mut o = vec![0.0f64; np * 114];
        for (i, v) in o.iter_mut().enumerate() {
            *v = i as f64;
        }
        sink += o[5];
    });
    println!("alloc+write output {:.3} us/px", us(r));
    let mut o = vec![0.0f64; np * 114];
    let r = best(|| {
        for (i, v) in o.iter_mut().enumerate() {
            *v = i as f64;
        }
        sink += o[5];
    });
    println!("write output (warm) {:.3} us/px", us(r));
    let r = best(|| sink += cube.iter().sum::<f64>());
    println!("read cube f64 (sum) {:.3} us/px", us(r));
    let r = best(|| sink += interp_batch(&cube32, np, nt, &t, &grid[..1], &Method::Linear, 1).unwrap()[0]);
    println!("linear f32, one output {:.3} us/px", us(r));
    let r = best(|| sink += interp_batch(&cube32, np, nt, &t, &grid[..1], &Method::Moving { window: 16.0 }, 1).unwrap()[0]);
    println!("moving f32, one output {:.3} us/px", us(r));
    let r = best(|| sink += interp_batch(&cube, np, nt, &t, &grid, &Method::Linear, 1).unwrap()[7]);
    println!("linear f64 {:.3} us/px", us(r));
    let r = best(|| sink += interp_batch(&cube32, np, nt, &t, &grid, &Method::Linear, 1).unwrap()[7]);
    println!("linear f32 {:.3} us/px", us(r));
    let r = best(|| sink += interp_batch(&cube, np, nt, &t, &grid, &Method::Moving { window: 16.0 }, 1).unwrap()[7]);
    println!("moving f64 {:.3} us/px", us(r));
    let r = best(|| sink += interp_batch(&cube32, np, nt, &t, &grid, &Method::Moving { window: 16.0 }, 1).unwrap()[7]);
    println!("moving f32 {:.3} us/px", us(r));
    let m = Method::Rbf { sigma: vec![8.0, 16.0, 32.0], cutoff: 0.95 };
    let r = best(|| sink += interp_batch(&cube, np, nt, &t, &grid, &m, 1).unwrap()[7]);
    println!("rbf f64 {:.3} us/px", us(r));
    let tsi: Vec<f64> = (0..np * 114).map(|_| (10000.0 * rnd()).floor()).collect();
    let group: Vec<i32> = (0..114).map(|i| (i / 23).min(4)).collect();
    let mean = [Metric::Mean];
    let r = best(|| sink += fold_batch(&tsi, np, 114, &group, 5, &mean, 1.0, 1)[3]);
    println!("year fold mean {:.3} us/px", us(r));
    println!("({sink})");
}
