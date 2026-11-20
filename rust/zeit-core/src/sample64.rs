//! The sample types the change-detection kernels (`imad`, `cva`, `omnibus`) convert to float64 for their statistics.

/// A sample type converted to float64 (exactly, for every type here).
pub trait Sample64: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

macro_rules! sample64 {
    ($($t:ty),*) => {$(
        impl Sample64 for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    )*};
}
sample64!(i16, u16, i32, u32, f32, f64);

/// Whether `v` is a usable value: finite and none of the `missing` numbers.
#[inline(always)]
pub fn present(v: f64, missing: &[f64]) -> bool {
    v.is_finite() && !missing.contains(&v)
}

/// A small seeded generator for the tests of the change-detection kernels (`zeit-core` has no `rand`): SplitMix64,
/// Box-Muller normals and Marsaglia-Tsang gamma variates.
#[cfg(test)]
pub(crate) mod rng {
    pub struct Rng {
        state: u64,
        spare: Option<f64>,
    }

    impl Rng {
        pub fn new(seed: u64) -> Rng {
            Rng {
                state: seed,
                spare: None,
            }
        }

        pub fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in (0, 1).
        pub fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        pub fn normal(&mut self) -> f64 {
            if let Some(s) = self.spare.take() {
                return s;
            }
            let (u, v) = (self.uniform(), self.uniform());
            let r = (-2.0 * u.ln()).sqrt();
            let t = 2.0 * std::f64::consts::PI * v;
            self.spare = Some(r * t.sin());
            r * t.cos()
        }

        /// Gamma with shape `a` and scale 1.
        pub fn gamma(&mut self, a: f64) -> f64 {
            if a < 1.0 {
                let g = self.gamma(a + 1.0);
                return g * self.uniform().powf(1.0 / a);
            }
            let d = a - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                let x = self.normal();
                let v = (1.0 + c * x).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u = self.uniform();
                if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                    return d * v;
                }
            }
        }
    }
}
