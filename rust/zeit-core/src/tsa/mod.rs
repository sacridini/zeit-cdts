//! Time series analysis of regular and irregular series, per pixel: the
//! statistics of a series or of its folds by year, quarter, month, week and day
//! of the year, the standardization of a series, its interpolation to new
//! dates, and its linear trend and greatest change. Ported from FORCE's higher-level TSA module (see each file's header).

pub mod interp;
pub mod stats;
pub mod trend;
