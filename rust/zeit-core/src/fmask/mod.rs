//! Fmask 4.7 (Function of mask, Zhu and Woodcock 2012, 2014; Qiu, Zhu and Woodcock 2019, 2020), in Rust.
//!
//! The cloud, cloud shadow and snow detection of the Fmask 4.7 repository (https://github.com/GERSL/Fmask, commit
//! e32f5bd, MIT License, Copyright (c) 2018 Global Environmental Remote Sensing Lab) is ported in steps; the
//! modules of this one hold:
//!
//! * [`mlab`]: the primitives of MATLAB (the Image Processing and Statistics toolboxes) that the detection uses,
//!   with their semantics: `prctile`, `imresize` by 2 and 3, `stdfilt`, `imerode`/`imdilate` with `strel`
//!   (`square`, `disk`, arbitrary), `bwlabel`, `bwareaopen`, `regionprops`, `imfill`, `imfilter` and `round`.
//!   They are written from MATLAB's documented behaviour and compared with GNU Octave
//!   (`benchmarks/vs_originals/fmask4.py`), not copied from any code.
//!
//! * [`detect`]: the Sentinel-2 route of the potential pixels (`FmaskParameters`, `NDVI`, `NDSI`, `NDBI`, `Saturate`,
//!   `DetectSnow`, `DetectWater`, `DetectPotentialPixels`, `NormalizaCirrusDEM`, `DetectAbsSnow`,
//!   `DetectPotentialCloud` and the probabilities), one function per .m file.
//!
//! * [`commission`]: `CDI`, `EnhanceLine`, `DetectPotentialFalsePositivePixels` and `ErodeCommissons`.
//!
//! The Fmask MATLAB code works on column-major matrices; the images of this port are row-major slices
//! (`rows`, `cols`), and where the order matters (the labels of `bwlabel`, the pixel lists of `regionprops`)
//! the MATLAB (column) order is kept.

pub mod commission;
pub mod detect;
pub mod mlab;
pub mod run;
pub mod shadow;
