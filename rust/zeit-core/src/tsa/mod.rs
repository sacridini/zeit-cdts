//! Time series analysis of regular and irregular series, per pixel: the
//! statistics of a series or of its folds by year, quarter, month, week and day
//! of the year, the standardization of a series, its interpolation to new
//! dates, its linear trend and greatest change, and the clear observations of
//! windows of months and the gaps between them, and the polar phenology metrics of the seasons. Ported from FORCE's higher-level TSA module (see each file's header).

pub mod cso;
pub mod interp;
pub mod polar;
pub mod stats;
pub mod trend;

/// A type the series may be stored in: the work is always in f64, the cube is
/// not converted as a whole (a float32 cube is half the memory traffic of a
/// float64 one). NaN marks a missing observation.
pub trait Sample: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

impl Sample for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

impl Sample for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
