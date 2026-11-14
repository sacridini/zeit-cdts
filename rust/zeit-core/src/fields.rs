//! The building blocks of field-boundary delineation by the stochastic watershed (phase 24a):
//! the marker-based watershed of scikit-image, grayscale morphology (scipy.ndimage and
//! skimage.morphology), percentile rescaling, hysteresis thresholding, `find_boundaries`, component
//! labelling, `relabel_sequential`, `remove_small_objects` and the greedy thinning of random
//! markers; and (phase 24b) the Monte Carlo of the stochastic watershed on top of them
//! ([`stochastic`]). Each function equals its scikit-image / scipy counterpart (0.26 / 1.18) value for value;
//! images are row-major `[rows, cols]` slices.

pub mod exact;
pub mod intensity;
pub mod label;
pub mod morph;
pub mod points;
pub mod stochastic;
pub mod watershed;
