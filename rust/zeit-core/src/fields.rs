//! The building blocks of field-boundary delineation by the stochastic watershed (phase 24a):
//! the marker-based watershed of scikit-image, grayscale morphology (scipy.ndimage and
//! skimage.morphology), percentile rescaling, hysteresis thresholding, `find_boundaries`, component
//! labelling, `relabel_sequential`, `remove_small_objects` and the greedy thinning of random
//! markers. Each function equals its scikit-image / scipy counterpart (0.26 / 1.18) value for value;
//! images are row-major `[rows, cols]` slices.

pub mod intensity;
pub mod label;
pub mod morph;
pub mod points;
pub mod watershed;
