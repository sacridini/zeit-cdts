//! Rbeast's random numbers: the PCG32 stream of `abc_rand_pcg_local_generic.c`
//! and the Gaussian (a 64-bin ziggurat) and gamma (Marsaglia's squeeze for
//! shapes above 1) samplers of `abc_rand_pcg_local.c` and
//! `abc_rand_pcg_global.c`, number for number: the same seed gives the same
//! draws, so the same chain.

#![allow(clippy::excessive_precision)]

use std::sync::OnceLock;

const MULTIPLIER: u64 = 6364136223846793005;
const DEFAULT_STATE: u64 = 0x853c49e6748fea9b;
const DEFAULT_INCREMENT: u64 = 0xda3e39cb94b95bdb;
const INV_2P25: f64 = 1.0 / (1u64 << 25) as f64;
const INV_2P32: f64 = 1.0 / (1u64 << 32) as f64;

/// A PCG32 stream (`local_pcg32_random_t`).
#[derive(Clone)]
pub struct Pcg {
    state: u64,
    increment: u64,
}

impl Pcg {
    /// `local_pcg_set_seed(stream, 0x853c49e6748fea9b, seed)`, which is what
    /// `r_vslNewStream` does with BEAST's seed.
    pub fn new(seed: u64) -> Self {
        let mut initstate = DEFAULT_STATE ^ seed;
        if initstate == 0 {
            initstate = DEFAULT_STATE;
        }
        let initseq = if seed == 0 { DEFAULT_INCREMENT } else { seed };
        let mut rng = Pcg {
            state: 0,
            increment: (initseq << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(initstate);
        rng.next_u32();
        rng
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.increment);
        xorshifted.rotate_right(rot)
    }

    /// `local_pcg_random`.
    pub fn fill(&mut self, out: &mut [u32]) {
        for v in out {
            *v = self.next_u32();
        }
    }
}

struct GaussTable {
    x: [f32; 64],
    y_ratio: [f32; 63],
    exp_lamda: f32,
    inflection: i64,
}

#[rustfmt::skip]
const GAUSS_X: [f32; 64] = [
    0.0, 0.019584285230127, 0.039176085503098, 0.058782936068943, 0.078412412733112,
    0.098072152488661, 0.117769874579095, 0.137513402144336, 0.157310684610171, 0.177169820991740, 0.197099084294312,
    0.217106947210130, 0.237202109328788, 0.257393526100938, 0.277690439821577, 0.298102412930487, 0.318639363964375,
    0.339311606538817, 0.360129891789569, 0.381105454763557, 0.402250065321725, 0.423576084201200, 0.445096524985516,
    0.466825122852590, 0.488776411114670, 0.510965806738247, 0.533409706241281, 0.556125593618691, 0.579132162255556,
    0.602449453164424, 0.626099012346421, 0.650104070647995, 0.674489750196082, 0.699283302383220, 0.724514383492366,
    0.750215375467941, 0.776421761147928, 0.803172565597918, 0.830510878205399, 0.858484474141832, 0.887146559018876,
    0.916556667533113, 0.946781756301046, 0.977897543940542, 1.009990169249582, 1.043158263318454, 1.077515567040280,
    1.113194277160929, 1.150349380376008, 1.189164350199337, 1.229858759216589, 1.272698641190536, 1.318010897303537,
    1.366203816372098, 1.417797137996268, 1.473467577947101, 1.534120544352547, 1.601008664886076, 1.675939722773444,
    1.761670410363067, 1.862731867421652, 1.987427885929896, 2.153874694061456, 2.41755901623650,
];

fn gauss_table() -> &'static GaussTable {
    static TABLE: OnceLock<GaussTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let x = GAUSS_X;
        let mut y_ratio = [0f32; 63];
        for i in 0..63 {
            // exp(-0.5 * (x[i+1]*x[i+1] - x[i]*x[i])), the difference in float.
            let d = x[i + 1] * x[i + 1] - x[i] * x[i];
            y_ratio[i] = (-0.5 * d as f64).exp() as f32;
        }
        let inflection = (0..63).find(|&i| x[i + 1] as f64 >= 1.0).unwrap_or(0) as i64;
        let a = x[63];
        let exp_lamda = ((a as f64 + ((a * a + 4.0) as f64).sqrt()) / 2.0) as f32;
        GaussTable {
            x,
            y_ratio,
            exp_lamda,
            inflection,
        }
    })
}

/// `local_pcg_gauss`: `out.len()` standard normal draws.
pub fn gauss(rng: &mut Pcg, out: &mut [f32]) {
    let g = gauss_table();
    let n = out.len() as i64;
    let mut rand = [0u32; 128];
    let size_minus2 = (2 * n).min(128) - 2;
    let mut cursor = size_minus2 + 1;
    let refill = |rng: &mut Pcg, rand: &mut [u32; 128]| rng.fill(&mut rand[..(size_minus2 + 2) as usize]);
    for o in out.iter_mut() {
        if cursor > size_minus2 {
            refill(rng, &mut rand);
            cursor = 0;
        }
        let r = rand[cursor as usize];
        let mut u = (r >> 7) as f64 * INV_2P25;
        let bin = (r & 0x3f) as usize;
        let sign = r & 0x40;
        cursor += 1;
        let x: f32;
        if bin < 63 {
            let delta = g.x[bin + 1] - g.x[bin];
            let y_ratio = g.y_ratio[bin];
            loop {
                if u <= y_ratio as f64 {
                    x = (g.x[bin] as f64 + delta as f64 * u / y_ratio as f64) as f32;
                    break;
                }
                let u1 = rand[cursor as usize] as f64 * INV_2P32;
                cursor += 1;
                let check = u <= y_ratio as f64 + (1.0f32 - y_ratio) as f64 * u1;
                let xx = (g.x[bin + 1] as f64 - delta as f64 * u1) as f32;
                let skip_log = (bin as i64) > g.inflection && !check;
                if (bin as i64) < g.inflection && check {
                    x = xx;
                    break;
                }
                if !skip_log && u.ln() <= (-0.5f32 * (xx * xx - g.x[bin] * g.x[bin])) as f64 {
                    x = xx;
                    break;
                }
                if cursor > size_minus2 {
                    refill(rng, &mut rand);
                    cursor = 0;
                }
                u = rand[cursor as usize] as f64 * INV_2P32;
                cursor += 1;
            }
        } else {
            loop {
                let u1 = rand[cursor as usize] as f64 * INV_2P32;
                cursor += 1;
                let xx = (g.x[63] as f64 - u1.ln() / g.exp_lamda as f64) as f32;
                let d = xx - g.exp_lamda;
                if u.ln() < (-0.5f32 * (d * d)) as f64 {
                    x = xx;
                    break;
                }
                if cursor > size_minus2 {
                    refill(rng, &mut rand);
                    cursor = 0;
                }
                u = rand[cursor as usize] as f64 * INV_2P32;
                cursor += 1;
            }
        }
        *o = if sign != 0 { x } else { -x };
    }
}

/// `logf` of the C runtime Rbeast is built with: the double log, rounded.
#[inline]
pub fn logf(x: f32) -> f32 {
    (x as f64).ln() as f32
}

/// `expf`, likewise.
#[inline]
pub fn expf(x: f32) -> f32 {
    (x as f64).exp() as f32
}

/// `local_pcg_gamma(rng, out, a, out.len())`: gamma draws of shape `a`, scale 1.
pub fn gamma(rng: &mut Pcg, out: &mut [f32], a: f32) {
    const INV_MAX: f32 = 2.328306436538696e-10;
    let n = out.len() as i64;
    let mut rand = [0u32; 200];
    let size_minus2 = (if 2 * n < 200 { 2 * n + 2 } else { 200 }) - 2;
    let fill = (size_minus2 + 2) as usize;
    rng.fill(&mut rand[..fill]);
    let mut cursor: i64 = 0;
    let mut next2 = |rng: &mut Pcg, rand: &mut [u32; 200]| {
        if cursor > size_minus2 {
            rng.fill(&mut rand[..fill]);
            cursor = 0;
        }
        let u0 = rand[cursor as usize] as f32 * INV_MAX;
        let u1 = rand[cursor as usize + 1] as f32 * INV_MAX;
        cursor += 2;
        (u0, u1)
    };
    if a > 1.0 {
        let b = a - 1.0;
        let c = a + a + a - 0.75;
        for o in out.iter_mut() {
            let gam;
            loop {
                let (u0, u1) = next2(rng, &mut rand);
                let w = u0 * (1.0 - u0);
                let y = (c / w).sqrt() * (u0 - 0.5);
                let g = b + y;
                if g >= 0.0 {
                    let z = 64.0f32 * (w * w * w) * (u1 * u1);
                    if z <= 1.0 - 2.0 * (y * y) / g {
                        gam = g;
                        break;
                    }
                    if logf(z) <= 2.0 * (b * logf(g / b) - y) {
                        gam = g;
                        break;
                    }
                }
            }
            *o = gam;
        }
    } else if a == 1.0 {
        let b = a - 1.0;
        let c = a + a + a - 0.75;
        for o in out.iter_mut() {
            let gam;
            loop {
                let (u0, u1) = next2(rng, &mut rand);
                let w = u0 * (1.0 - u0);
                let y = (c / w).sqrt() * (u0 - 0.5);
                let g = b + y;
                if g >= 0.0 {
                    let z = 64.0f32 * (w * w * w) * (u1 * u1);
                    if z <= 1.0 - 2.0 * (y * y) / g {
                        gam = g;
                        break;
                    }
                    if logf(z) <= -2.0 * y {
                        gam = g;
                        break;
                    }
                }
            }
            *o = gam;
        }
    } else if a > 0.0 {
        let b = 1.0 + 0.3678794f32 * a;
        for o in out.iter_mut() {
            let gam;
            loop {
                let (u0, u1) = next2(rng, &mut rand);
                let c = b * u0;
                if c < 1.0 {
                    let g = expf(logf(c) / a);
                    if -logf(1.0 - u1) >= g {
                        gam = g;
                        break;
                    }
                } else {
                    let g = -logf((b - c) / a);
                    if -logf(1.0 - u1) >= (1.0 - a) * logf(g) {
                        gam = g;
                        break;
                    }
                }
            }
            *o = gam;
        }
    } else if a < 0.0 {
        out.fill(f32::NAN);
    } else {
        out.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_reference() {
        // The reference PCG32 (pcg32_srandom_r(42, 54)) gives 0xa15c02b7 first;
        // Rbeast's seeding takes initstate = DEFAULT_STATE ^ initseq, so check the
        // generator itself through its state update instead.
        let mut r = Pcg {
            state: 0,
            increment: (54 << 1) | 1,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(42);
        r.next_u32();
        assert_eq!(r.next_u32(), 0xa15c02b7);
        assert_eq!(r.next_u32(), 0x7b47f409);
    }

    #[test]
    fn gauss_moments() {
        let mut r = Pcg::new(7);
        let mut v = vec![0f32; 200_000];
        gauss(&mut r, &mut v);
        let m = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let s = v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64;
        assert!(m.abs() < 0.01, "{m}");
        assert!((s - 1.0).abs() < 0.01, "{s}");
    }

    #[test]
    fn gamma_moments() {
        let mut r = Pcg::new(9);
        for a in [0.5f32, 1.0, 3.5, 60.0] {
            let mut v = vec![0f32; 100_000];
            gamma(&mut r, &mut v, a);
            let m = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
            assert!((m - a as f64).abs() < 0.03 * a as f64 + 0.01, "{a} {m}");
        }
    }
}
