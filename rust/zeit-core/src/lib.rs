//! The algorithms of zeit, in Rust: the replacement of src/ (C++), module by
//! module (see RUST_MIGRATION.md). Each module is a port of its C++ counterpart
//! operation by operation (the same sums in the same order), so that both give
//! the same bits; the Python bindings live in the zeit-py crate.

pub mod cmath;
pub mod eigen;
pub mod mann_kendall;
pub mod parallel;
pub mod snic;
pub mod twdtw;
