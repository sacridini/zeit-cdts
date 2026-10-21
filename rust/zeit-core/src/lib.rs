//! The algorithms of zeit, in Rust. Each module is a port of the C++ core of
//! zeit 1.x (src/, removed in 2.0; see RUST_MIGRATION.md) operation by
//! operation, the same sums in the same order, and gives the same bits: "the
//! C++" in the comments is that core. The Python bindings live in the zeit-py
//! crate.

// The C++'s tests such as `!(s > 0.0)` are kept as they are: they fail on NaN,
// where `s <= 0.0` would pass it.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod bfast;
pub mod bfast_lite;
pub mod bfast_monitor;
pub mod ccdc;
pub mod cmath;
pub mod coded;
pub mod eigen;
pub mod landtrendr;
pub mod mann_kendall;
pub mod medoid;
pub mod parallel;
pub mod phenology;
pub mod robustfit;
pub mod sma;
pub mod snic;
pub mod som;
pub mod stl;
pub mod tmask;
pub mod twdtw;
pub mod warp;
pub mod whittaker;
