//! BEAST's options as Rbeast settles them (`beastv2_io_in_args.c`): the
//! orders, the minimum separations, the margins and the numbers of
//! changepoints clamped to what the series allows, the largest number of
//! terms, the move step and the credible-interval subsampling.

/// The prior on the precision of the coefficients (`precPriorType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecType {
    /// One precision, never resampled (`'constant'`).
    Const,
    /// One precision for every term (`'uniform'`).
    Uniform,
    /// One precision per component, season and trend (`'componentwise'`).
    ComponentWise,
}

/// What a caller sets (the arguments of Rbeast's `beast()`).
#[derive(Clone, Debug)]
pub struct Options {
    /// A harmonic season (`season = 'harmonic'`) or none (`'none'`).
    pub season: bool,
    /// The period, in the units of the times (`period`).
    pub period: f32,
    /// The time of the first value and the time between two (`start`, `deltat`).
    pub start: f32,
    pub dt: f32,
    /// `scp.minmax`, `sorder.minmax`, `tcp.minmax`, `torder.minmax`.
    pub scp: (i32, i32),
    pub sorder: (i32, i32),
    pub tcp: (i32, i32),
    pub torder: (i32, i32),
    /// `sseg.min`, `tseg.min`: the shortest segment, in values (None: Rbeast's).
    pub sseg_min: Option<i32>,
    pub tseg_min: Option<i32>,
    /// `mcmc.samples`, `mcmc.burnin`, `mcmc.chains`, `mcmc.thin`, `mcmc.seed`.
    pub samples: u32,
    pub burnin: u32,
    pub chains: u32,
    pub thin: u32,
    pub seed: u64,
    /// `precValue`, `precPriorType`.
    pub prec_value: f32,
    pub prec_type: PrecType,
    /// `dump.ci`: the credible intervals (their subsampling draws from the
    /// stream, so the chain differs with and without them, as in Rbeast).
    pub ci: bool,
    /// `maxMissingRate`: a series with more missing values is not fitted.
    pub max_missing_rate: f32,
}

impl Default for Options {
    /// Rbeast's `beast()` defaults, with seed 42.
    fn default() -> Self {
        Options {
            season: true,
            period: 1.0,
            start: 1.0,
            dt: 1.0,
            scp: (0, 10),
            sorder: (0, 5),
            tcp: (0, 10),
            torder: (0, 1),
            sseg_min: None,
            tseg_min: None,
            samples: 8000,
            burnin: 200,
            chains: 3,
            thin: 5,
            seed: 42,
            prec_value: 1.5,
            prec_type: PrecType::ComponentWise,
            ci: true,
            max_missing_rate: 0.75,
        }
    }
}

/// The prior of one component: its orders, numbers of changepoints, minimum
/// separation and margins.
#[derive(Clone, Copy, Debug, Default)]
pub struct BasisPrior {
    pub min_order: i32,
    pub max_order: i32,
    pub min_knot: i32,
    pub max_knot: i32,
    pub min_sep: i32,
    pub left: i32,
    pub right: i32,
    pub complexity: f32,
}

/// The options settled for a series of `n` values.
#[derive(Clone, Debug)]
pub struct Config {
    pub n: usize,
    pub season: bool,
    /// The period in values (`meta.period`, a float).
    pub period: f32,
    pub start: f32,
    pub dt: f32,
    pub sp: BasisPrior,
    pub tp: BasisPrior,
    pub k_max: i32,
    /// The prior the precision functions use (componentwise becomes uniform
    /// without a season, as in Rbeast) and the basis types its option chose.
    pub prec_type: PrecType,
    pub season_btype: u8,
    pub trend_btype: u8,
    pub prec_value: f32,
    pub sig2: f32,
    pub alpha1: f32,
    pub alpha2: f32,
    pub delta1: f32,
    pub delta2: f32,
    pub sig_factor: f32,
    pub samples: u32,
    pub burnin: u32,
    pub chains: u32,
    pub thin: u32,
    pub seed: u64,
    pub move_step: i32,
    pub t_resample: f32,
    pub s_resample: f32,
    pub ci: bool,
    pub ci_fast: bool,
    pub ci_k: usize,
    pub ci_sub: u16,
    pub max_missing_rate: f32,
}

fn margins(n: i32, min_sep: i32) -> (i32, i32, i32) {
    let (mut left, mut right) = (min_sep, min_sep);
    let mut nleft = n - (1 + left + right);
    if nleft < 1 {
        if n - (1 + min_sep * 2) < 1 {
            left = 0;
            right = 0;
        } else {
            left = min_sep;
            right = min_sep;
        }
        nleft = n - (1 + left + right);
    }
    let max_cp = (nleft as f64 / (min_sep as f64 + 1.0)).ceil() as i32;
    (left, right, max_cp)
}

impl Config {
    pub fn new(o: &Options, n: usize) -> Result<Config, String> {
        let nn = n as i32;
        if n < 2 {
            return Err(format!("BEAST needs at least 2 values, got {n}"));
        }
        if !(o.dt > 0.0) {
            return Err(format!("deltat must be positive, got {}", o.dt));
        }
        let season = o.season;
        if season && !(o.period > 0.0) {
            return Err(format!("a harmonic season needs a positive period, got {}", o.period));
        }
        if season && !(o.period / o.dt >= 2.0) {
            return Err(format!(
                "period / deltat = {} / {} gives fewer than 2 values per period: use season='none'",
                o.period, o.dt
            ));
        }
        let period = if season { o.period / o.dt } else { 0.0 };
        let p = period as i32;

        let mut sp = BasisPrior::default();
        if season {
            let mut smin = o.sorder.0.min(p / 2 - 1);
            smin = smin.max(1);
            let mut smax = o.sorder.1.min(p / 2 - 1);
            smax = smax.max(smin);
            let mut sep = match o.sseg_min {
                Some(v) if v > 0 => v,
                _ => p / 2,
            };
            sep = sep.max(smax);
            sep = sep.min(nn / 2 - 1);
            let (left, right, max_cp) = margins(nn, sep);
            let (mut kmin, mut kmax) = o.scp;
            kmin = kmax.min(kmin);
            kmin = kmin.max(0);
            kmin = kmin.min(max_cp);
            kmax = kmax.min(max_cp);
            kmax = kmax.max(kmin);
            sp = BasisPrior {
                min_order: smin,
                max_order: smax,
                min_knot: kmin,
                max_knot: kmax,
                min_sep: sep,
                left,
                right,
                complexity: 0.0,
            };
        }
        let tmin = o.torder.0.max(0);
        let tmax = o.torder.1.max(tmin);
        let mut tsep = match o.tseg_min {
            Some(v) if v > 0 => v,
            _ => {
                if season {
                    p / 2
                } else {
                    3
                }
            }
        };
        tsep = tsep.max(tmax + 1);
        tsep = tsep.min(nn / 2 - 1);
        let (left, right, max_cp) = margins(nn, tsep);
        let (mut kmin, mut kmax) = o.tcp;
        kmin = kmax.min(kmin).max(0);
        kmin = kmin.min(kmax);
        kmin = kmin.max(0);
        kmin = kmin.min(max_cp);
        kmax = kmax.min(max_cp);
        kmax = kmax.max(kmin);
        let tp = BasisPrior {
            min_order: tmin,
            max_order: tmax,
            min_knot: kmin,
            max_knot: kmax,
            min_sep: tsep,
            left,
            right,
            complexity: 0.0,
        };

        let mut k_max = 0;
        if season {
            k_max += (sp.max_order * 2) * (sp.max_knot + 1);
        }
        k_max += (tp.max_order + 1) * (tp.max_knot + 1);
        if nn < 5000 {
            k_max = k_max.min(nn);
        } else if nn < 15000 {
            k_max = k_max.min(5000.min(nn / 2));
        } else {
            k_max = k_max.min(7500);
        }
        let half_up = |a: i32, b: i32| ((a + b) as f64 / 2.0).ceil() as i32;
        let mut k_init = 0;
        if season {
            k_init += (half_up(sp.max_order, sp.min_order) * 2) * (half_up(sp.min_knot, sp.max_knot) + 1);
        }
        k_init += (half_up(tp.max_order, tp.min_order) + 1) * (half_up(tp.min_knot, tp.max_knot) + 1);
        k_max = k_max.max(k_init);

        let samples = if o.samples == 0 { 3000 } else { o.samples }.max(800);
        let thin = o.thin.max(1);
        let burnin = if o.burnin == 0 { 150 } else { o.burnin }.max(150);
        let chains = if o.chains == 0 { 3 } else { o.chains }.max(1);
        let move_step = if season { period as i32 } else { tp.min_sep + 1 };
        let t_resample = if tp.max_order == tp.min_order { 0.0 } else { 0.1f32 };
        let s_resample = if sp.max_order == sp.min_order { 0.0 } else { 0.17f32 };

        let btype = match o.prec_type {
            PrecType::Const | PrecType::Uniform => 0,
            PrecType::ComponentWise => 1,
        };
        let prec_type = if !season && o.prec_type == PrecType::ComponentWise {
            PrecType::Uniform
        } else {
            o.prec_type
        };

        // ConstructCIStruct: 100 values in each tail of a subsample when the
        // chains are long enough, else every sample's tail.
        let alpha = ((1.0 - 0.95f32 as f64) / 2.0) as f32;
        let mut ci_fast = true;
        let mut ci_sub = 0u16;
        let mut ci_k = 100usize;
        let total = 100.0f32 / alpha;
        let fraction = total as f64 / samples as f32 as f64;
        if fraction < 0.99f32 as f64 {
            ci_sub = (fraction * 65535.0) as u16;
        } else {
            ci_fast = false;
        }
        if !ci_fast {
            ci_k = (samples as f32 * alpha) as usize;
        }

        if !(o.prec_value > 0.0) {
            return Err(format!("prec_value must be positive, got {}", o.prec_value));
        }
        Ok(Config {
            n,
            season,
            period,
            start: o.start,
            dt: o.dt,
            sp,
            tp,
            k_max,
            prec_type,
            season_btype: btype,
            trend_btype: btype,
            prec_value: if (o.prec_value as f64) > 1e-32 {
                o.prec_value
            } else {
                1e-32
            },
            sig2: 0.2,
            alpha1: 0.00000001,
            alpha2: 0.00000001,
            delta1: 0.00000001,
            delta2: 0.00000001,
            sig_factor: 1.8,
            samples,
            burnin,
            chains,
            thin,
            seed: o.seed,
            move_step,
            t_resample,
            s_resample,
            ci: o.ci,
            ci_fast,
            ci_k: ci_k.max(1),
            ci_sub,
            max_missing_rate: o.max_missing_rate,
        })
    }

    /// The number of terms one proposal can add at most (the columns of the
    /// new terms' scratch matrix).
    pub fn max_new_terms(&self) -> usize {
        let t = 2 * (self.tp.max_order + 1) as usize;
        let s = if self.season { 4 * self.sp.max_order as usize } else { 0 };
        t.max(s).max(4)
    }
}
