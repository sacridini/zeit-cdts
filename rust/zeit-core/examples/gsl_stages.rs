//! The stages of the port of `gsl_eigen_symmv` on a matrix read from a file, written as raw doubles in the format of
//! `benchmarks/vs_originals/gsl_stages.c` (which writes GSL's own), to find the first operation at which they differ:
//!
//!     cargo run --release -p zeit-core --example gsl_stages -- <matrix.bin> <n> <out prefix>
//!
//! Writes `<out>_tau`, `_d`, `_sd`, `_q` (after the Householder reduction) and `_eval`, `_evec` (`gsl_eigen_symmv`,
//! unsorted).
use std::io::Write;

fn put(prefix: &str, name: &str, v: &[f64]) {
    let mut f = std::fs::File::create(format!("{prefix}_{name}")).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args[2].parse().unwrap();
    let bytes = std::fs::read(&args[1]).unwrap();
    let a: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(a.len(), n * n);
    let (tau, d, sd, q) = zeit_core::gsl_symmv::symmtd_stages(&a, n);
    put(&args[3], "tau", &tau);
    put(&args[3], "d", &d);
    put(&args[3], "sd", &sd);
    put(&args[3], "q", &q);
    let mut work = a.clone();
    let (e, v) = zeit_core::gsl_symmv::eigen_symmv(&mut work, n);
    put(&args[3], "eval", &e);
    put(&args[3], "evec", &v);
}
