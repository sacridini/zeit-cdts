# Third-party notices

Zeit is licensed under the GNU General Public License, version 3 or (at your
option) any later version (`GPL-3.0-or-later`, see `LICENSE`).

Several algorithms in Zeit are ports of, or are derived from, the software
listed below. Their licenses are all compatible with distributing Zeit under
the GPL-3.0-or-later. The parts that used to derive from GPL-2.0-only code were
rewritten from the papers and the products' documentation, without reading the
original code: the CCDC lasso (`rust/zeit-core/src/lasso.rs`, from Friedman,
Hastie & Tibshirani 2010, instead of GLMnet), the Bayesian smoothing of class
probabilities (`rust/zeit-core/src/bayes_smooth.rs`, from Camara et al. 2024,
instead of sits), the phenology (`rust/zeit-core/src/phenology/`, from the
papers, instead of phenofit), the QA weights (`zeit/qc.py`, from the
products' guides, instead of phenofit's `qcFUN.R`), and the TempCNN and
LightTAE networks (`zeit/ai/tempcnn.py`, `zeit/ai/ltae.py`, from the papers and
the authors' repositories, instead of sits). None of GLMnet, phenofit or sits
is in Zeit any more.

## Ported or derived code

| Zeit component | Derived from | License |
| :--- | :--- | :--- |
| `rust/zeit-core/src/ccdc.rs` (CCDC, apart from the lasso, which is written from the paper), `rust/zeit-core/src/robustfit.rs` | [GERSL/CCDC](https://github.com/GERSL/CCDC), Zhu & Woodcock (2014), MATLAB | MIT |
| `rust/zeit-core/src/bfast.rs`, `bfast_monitor.rs`, `bfast_lite.rs` | R package [bfast](https://github.com/bfast2/bfast), Verbesselt et al. | GPL-2.0-or-later |
| `rust/zeit-core/src/bfast*.rs` (efp/MOSUM process, critical values, breakpoints) | R package [strucchangeRcpp](https://github.com/bfast2/strucchangeRcpp), Zeileis et al. | GPL-2.0 or GPL-3.0 |
| `rust/zeit-core/src/stl.rs` | R `stats::stl()` (`src/library/stats/src/stl.c`), R Core Team | GPL-2.0 or GPL-3.0 |
| `rust/zeit-core/src/optimize.rs` (Brent's minimizer, `fmin` and `fmin_netlib`) | R `stats::optimize()` (`src/library/stats/src/optimize.c`, `Brent_fmin`, itself a translation of netlib's `fmin.f`, Forsythe, Malcolm & Moler 1977), R Core Team | GPL-2.0 or GPL-3.0 |
| `rust/zeit-core/src/mann_kendall.rs` | [pymannkendall](https://github.com/mmhs013/pymannkendall), Hussain & Mahmud (2019) | MIT |
| `rust/zeit-core/src/eigen.rs`, `eigen/` (QR, LDLT, matrix products, sparse Cholesky with AMD ordering, Levenberg-Marquardt), ported operation by operation so the results are those of zeit 1.x | [Eigen](https://eigen.tuxfamily.org) 3.4 | MPL-2.0 |
| `rust/zeit-core/src/eigen/lm.rs` (Levenberg-Marquardt, from Eigen's `unsupported/NonLinearOptimization`) | MINPACK, University of Chicago | MINPACK license (see below) |
| `zeit/ai/utae.py` (U-TAE, L-TAE) | [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps), Garnot & Landrieu (2021) | MIT |
| `zeit/ai/presto.py` (Presto encoder; its pretrained weights are downloaded, not bundled) | [nasaharvest/presto](https://github.com/nasaharvest/presto), Tseng et al. (2023) | MIT |
| `zeit/ai/prithvi.py` (Prithvi-EO-2.0 encoder; its pretrained weights are downloaded, not bundled) | `prithvi_mae.py` of [ibm-nasa-geospatial/Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial), Szwarcman et al. (2024), IBM Corp. | Apache-2.0 |
| `rust/zeit-core/src/beast.rs`, `beast/` (BEAST) | C code of the R package [Rbeast](https://github.com/zhaokg/Rbeast) 1.0.2, Zhao et al. (2019) | GPL (>= 2) |
| `rust/zeit-core/src/som.rs`, `zeit/_som.py` (SOM) | [MiniSom](https://github.com/JustGlowing/minisom) 2.3.x, Vettigli | MIT |
| `zeit/ai/tempcnn.py` (TempCNN) | [charlotte-pel/temporalCNN](https://github.com/charlotte-pel/temporalCNN), Pelletier et al. (2019) | GPL-3.0 |
| `zeit/ai/ltae.py` (LightTAE, the L-TAE) | [VSainteuf/lightweight-temporal-attention-pytorch](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch), Sainte Fare Garnot & Landrieu (2020) | MIT |
| `zeit/segmentation.py` (seed grids) | R package [snic](https://github.com/rolfsimoes/snic), `snic_grid()` | GPL (>= 2) |
| `rust/zeit-core/src/warp.rs`, `warp/` (resampling kernels of `load_raster(..., like=)`) | [GDAL](https://gdal.org) 3.12, `alg/gdalwarpkernel.cpp`, Warmerdam, Rouault et al. (via landschaft, same author as Zeit) | MIT |
| `rust/zeit-core/src/tsa/` (spectral-temporal metrics and folds, standardization, interpolation, linear trend, change-aftereffect-trend, clear-observation statistics, polar phenology metrics; `stats.rs` from `stm-hl.c`, `fold-hl.c`, `standardize-hl.c` and `stats-cl.c`, `interp.rs` from `interpolate-hl.c`, `trend.rs` from `trend-hl.c` and `stats-cl.c`, `cso.rs` from `cso-hl.c`, `polar.rs` from `polar-hl.c`) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2019) | GPL-3.0-or-later |
| `rust/zeit-core/src/specadjust.rs` and `specadjust_tables.rs` (spectral adjustment of Landsat and MODIS to Sentinel-2A, `zeit.spectral_adjust`; the code from `spec-adjust-hl.c`, the tables generated from the same file by `rust/zeit-core/tools/gen_specadjust.py`, which keeps its copyright line; the method and the tables are Scheffler, Frantz and Segl 2020, *Remote Sensing of Environment* 241:111723) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
| `zeit/_tasseled.py` (Tasseled Cap, `zeit.tasseled_cap`: the arithmetic of `index_tasseled` in `index-compute-hl.c`; the matrix is Crist 1985, *Remote Sensing of Environment* 17:301-306, and the matrices of Huang et al. 2002 and Baig et al. 2014 are from their papers) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
| `rust/zeit-core/src/improphe.rs` and `zeit/_improphe.py` (ImproPhe, `zeit.improphe`: the seasonal means, the standardization, the focal standard deviation, the neighbourhood kernel and the two drivers from `improphe-hl.c`, `cf-improphe-hl.c` and `l2-improphe-hl.c`; the principal components from `pca-cl.c`; the distance kernel of `imagefuns-cl.c`; the method is Frantz et al. 2016, *IEEE Transactions on Geoscience and Remote Sensing* 54(7):4153-4164) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
| `rust/zeit-core/src/resmerge.rs` and `zeit/_resmerge.py` (Sentinel-2 20 m to 10 m resolution merge, `zeit.sentinel2_resmerge`: `resolution_merge_1`, `resolution_merge_2`, `improphe_resmerge` and `rescale_weight_resmerge` of `resmerge-ll.c`, with `distance_kernel` of `imagefuns-cl.c`; the STARFM of the same file, `resolution_merge_3`, USDA-ARS code under its own terms, is not ported) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
| `rust/zeit-core/src/resmerge.rs` (the least squares fit of `zeit.sentinel2_resmerge(method="regression")`, for its FORCE arithmetic: `gsl_multifit_linear` and `gsl_multifit_linear_est` of GSL 2.7, `multifit/multilinear.c` and `multifit/linear_common.c`, with the column balancing, `gsl_linalg_SV_decomp_mod` and `gsl_linalg_SV_decomp` (`linalg/svd.c`, `linalg/bidiag.c`, `linalg/svdstep.c`, `linalg/householder.c`) and the reference CBLAS routines they call) | [GNU Scientific Library](https://www.gnu.org/software/gsl/) 2.7 | GPL-3.0-or-later |
| `rust/zeit-core/src/parallax.rs` and `zeit/_parallax.py` (Sentinel-2 parallax cloud test and cloud buffer, `zeit.parallax_clouds`, `zeit.cloud_displacement_index`, `zeit.buffer_mask`: `cloud_parallax` of `cloud-ll.c`, the Cloud Displacement Index of Frantz, Haß, Uhl, Stoffels and Hill 2018, *Remote Sensing of Environment* 215:471-481; with `gauss_kernel` and `buffer_` of `imagefuns-cl.c`, `covar_recurrence` and `variance` of `stats-cl.c` and the queue of `queue-cl.c`. `cloud-ll.c` as a whole is derived from Zhu's Fmask, but only this step, which is Frantz's own, is ported: nothing of Fmask's own parts (the potential cloud pixels, the land and water probabilities, the shadow matching) is here, they are inputs of the function. Fmask's current repository, https://github.com/GERSL/Fmask, is MIT-licensed (checked 2026-11-07 through the GitHub API), and none of its parts is used; the variability probability when it is not given is written from the formula of Zhu and Woodcock 2012, *Remote Sensing of Environment* 118:83-94) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
| `rust/zeit-core/src/radtran.rs` and `rust/zeit-core/src/radtran_tables.rs` (the radiative transfer of the atmospheric correction, the pieces of FORCE's `force-l2ps`: all of `radtran-ll.c` (`molecular_optical_depth`, `scatt_transmitt`, `path_ref`, `sphere_albedo`, `env_weight`, `backscatter`, the phase functions, `wvp_transmitt`, `ozone_transmitt`, `fresnel_reflection`, `illumin`, the C-factors and `f_factor`), `wavelength` and `E0` of `table-ll.c` with the tables they read (the Thuillier exoatmospheric irradiance, the water vapour (HITRAN 2016) and ozone (Bird and Riordan 1986) absorption, and the relative spectral response of the Landsat 4-8 and Sentinel-2A/B bands, courtesy of USGS/ESA, generated by `tools/gen_radtran_tables.py`), `ozone_amount` of `gas-ll.c`, `localtime2gmt` and `standard_sunzenith` of `sunview-ll.c`, and the day-of-year routines of `date-cl.c`; `brdf-ll.c` is not ported; FORCE's `sun-cl.c` is not ported either: the sun position behind `sunpos` is `rust/zeit-core/src/solpos.rs`, a clean-room implementation from Meeus, Astronomical Algorithms (1998), and NOAA's published solar calculation documentation, GPL-3.0-or-later (Zeit)) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
| `rust/zeit-core/src/gas.rs`, `rust/zeit-core/src/atc.rs`, `rust/zeit-core/src/qai.rs` and `rust/zeit-core/src/gsl_nmsimplex.rs` (the water vapour and gas transmittance of the atmospheric correction: `wvp_transmitt_lut`, `fun_wvp`, `water_vapor`, `gas_transmittance` and `water_vapor_from_lut` of FORCE's `gas-ll.c`, with the minimal forms of `atc_t` (`atc-ll.h`) and of the quality bits of `quality-cl.c` they read; `gsl_nmsimplex.rs` is `multimin/simplex2.c` of the GNU Scientific Library 2.7, Copyright (C) 2007-2009 Brian Gough, 2002 Tuomo Keskitalo and Ivo Alxneit, in one dimension and in n dimensions, with the reference `dnrm2`, `ddot` and `daxpy` of GSL's CBLAS) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022); [GSL](https://www.gnu.org/software/gsl/) 2.7 | GPL-3.0-or-later |
| `rust/zeit-core/src/aod.rs` and `rust/zeit-core/src/gsl_brent.rs` (the dark targets and the aerosol optical depth over each of them, FORCE's `aod-ll.c`, with its AOD map and entry point (`aod_map`, `interpolate_aod_map`, `compile_aod`): `aodfileread`, `extract_dark_target`, `water_lib`, `land_lib`, `veg_lib`, `aod_lib_to_target`, `aod_from_target`, `aod_elevation_minimizer`, `aod_elev_dependency`, `aod_linear_fit` and `aod_polynomial_fit`, with the connected components labelling (Chang, Chen and Lu 2004) and the distance transform (Meijster, Roerdink and Hesselink 2006) of `imagefuns-cl.c` and `get_cloud` of `quality-cl.c`; the water aerosol library `_AERO_WATERLIB_` of `table-ll.c` is in `radtran_tables.rs`; `gsl_brent.rs` is `min/brent.c`, `min/fsolver.c` and `min/convergence.c` of the GNU Scientific Library 2.7, Copyright (C) 1996-2000 and 2007 Brian Gough) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022); | `rust/zeit-core/src/gdal_grid.rs` (the inverse distance weighting of `GDALGridCreate` with the options FORCE's `interpolate_aod_map` passes, algorithm `invdist` with its defaults: `GDALGridInverseDistanceToAPower2NoSmoothingNoSearchAVX` of `alg/gdalgridavx.cpp`, as chosen by `GDALGridContextCreate` and run by `GDALGridJobProcess` and `GDALGridContextProcess` of `alg/gdalgrid.cpp`, GDAL **3.9.3**, the version of the conda env FORCE is compared against; Copyright (c) 2007 Andrey Kiselev, (c) 2009-2013 Even Rouault; the notice is in the MIT section below) | [GDAL](https://gdal.org) 3.9.3 (OSGeo), `alg/gdalgrid.cpp`, `alg/gdalgridavx.cpp` | MIT |
| `rust/zeit-core/src/topo.rs` (the topography of the atmospheric correction, FORCE's `topo-ll.c`: `ocean_topography`, `smooth_topography`, `exposition_topography` (slope and aspect with Horn's method, in radians times 10000), `stats_topography`, `illumination_topography`, `compile_topography`, `cfactor_topography` (the C-factor of the topographic correction) and `average_elevation_cell`, with `covar_recurrence`, `variance`, `covariance` and `linreg_coefs` of `stats-cl.c` and the QAI setters of `quality-cl.c`) | [FORCE](https://github.com/davidfrantz/force) 3.11.00, David Frantz (2013-2022) | GPL-3.0-or-later |
[GSL](https://www.gnu.org/software/gsl/) 2.7 | GPL-3.0-or-later |
| `rust/zeit-core/src/coreg.rs`, `rust/zeit-core/src/coreg_local.rs` and `zeit/_coreg.py` (global co-registration, `zeit.coregister`: `COREG` of `arosics/CoReg.py`, the matching window, the shifted cross power spectrum, the integer shift and its validation, the subpixel shift, the reliability and the structural similarity check; local co-registration, `zeit.coregister_local`: `COREG_LOCAL` of `arosics/CoReg_local.py` and `arosics/Tie_Point_Grid.py`, the grid of tie points, the match of each point, the reliability, SSIM and RANSAC filters (`Tie_Point_Refiner`) and the GCPs for the warp; the method is Scheffler, Hollstein, Hofmann and Bochow 2017, *Remote Sensing* 9(7):676. AROSICS has no NOTICE file; its structural similarity is scikit-image's `structural_similarity` (BSD-3-Clause) written out, and so is scikit-image's `ransac` of an `AffineTransform` (BSD-3-Clause) with numpy's `default_rng` (PCG64, `SeedSequence` and `Generator.choice`; BSD-3-Clause, and the PCG family, Apache-2.0 or MIT) reproduced from their documented algorithms; the FFTs are the `rustfft` crate (MIT or Apache-2.0); the local warp is GDAL's, through rasterio) | [AROSICS](https://git.gfz-potsdam.de/danschef/arosics) 1.14.0, Daniel Scheffler and the GFZ Helmholtz Centre for Geosciences, Potsdam (2017-2026) | Apache-2.0 |
| `rust/zeit-core/src/gsl_symmv.rs` (`gsl_eigen_symmv` and `gsl_eigen_symmv_sort` with the Householder reduction and the QR iteration, for `zeit.improphe(force=True)`, whose FORCE-style components take their signs and last bits from GSL 2.7: `eigen/symmv.c`, `eigen/qrstep.c` (Brian Gough 2001-2010), `eigen/sort.c` (Gerard Jungman, Patrick Alken 1996-2007), `linalg/symmtd.c` (Brian Gough 2001, 2007; Patrick Alken 2019), `linalg/householder.c` (Gerard Jungman, Brian Gough 1996-2010) and the reference CBLAS routines they call, `cblas/source_*.h` (Gerard Jungman 1996-2000)) | [GNU Scientific Library](https://www.gnu.org/software/gsl/) 2.7 | GPL-3.0-or-later |
| `rust/zeit-core/src/tsa/interp.rs` (the bisquare iteratively reweighted least squares of the harmonic interpolation) | [GNU Scientific Library](https://www.gnu.org/software/gsl/) `multifit/multirobust.c` and `robust_wfun.c`, Patrick Alken (Copyright 2013) | GPL-3.0-or-later |
| `rust/zeit-core/src/landtrendr.rs` (LandTrendr) | [KennedyResearch/LandTrendr-2012](https://github.com/KennedyResearch/LandTrendr-2012), Kennedy et al. (2010), IDL | no license published — see below |

SNIC (`rust/zeit-core/src/snic.rs`) is implemented from the paper (Achanta & Süsstrunk,
CVPR 2017). The authors' reference code is not included; the test suite only
stores outputs of it (`tests/data/snic_reference_parity.npz`) to check that
Zeit gives the same labels.

The remotePARTS methods (`rust/zeit-core/src/ar_reml.rs`, `gls.rs`,
`spatial.rs`, `parts.rs`) are implemented from the paper (Ives et al., Remote
Sensing of Environment 266, 112678, 2021) and the statistics it defines; the
AR(1) fit, the nugget of a fit alone and the statistics between partitions
are computed by other algorithms than the R package's. The p-value of the
correlated chi-square test integrates Imhof's (1961) formula. The package's code
(GPL-3.0-or-later) is not included, because Zeit's algorithms are other ones; the test suite only stores outputs of it
(`rust/zeit-core/tests/data/remoteparts_*.json`, made by
`tests/data/make_remoteparts_fixtures.R`) to check that Zeit gives the same fits.

The seed grids of `snic_grid` (`zeit/segmentation.py`) follow the R package snic, which is GPL (>= 2).

**LandTrendr-2012**: the original repository does not state a license. Zeit's
port is a derivative of that code; permission from the authors is being
sought. Until then, the LandTrendr port's redistribution terms are unresolved.

## Bundled or linked libraries

The Rust crates compiled into `zeit._core` (from `rust/Cargo.lock`; crates
used only to build it are left out):

| Library | Use | License |
| :--- | :--- | :--- |
| [PyO3](https://github.com/PyO3/pyo3) (`pyo3`, `pyo3-ffi`) | Python bindings | MIT OR Apache-2.0 |
| [rust-numpy](https://github.com/PyO3/rust-numpy) (`numpy`) | NumPy arrays in the bindings | BSD-2-Clause (see below) |
| [ndarray](https://github.com/rust-ndarray/ndarray), `matrixmultiply`, `rawpointer`, `num-traits`, `num-complex`, `num-integer` | array views (through rust-numpy) | MIT OR Apache-2.0 |
| [rayon](https://github.com/rayon-rs/rayon) (`rayon`, `rayon-core`, `crossbeam-*`, `either`) | threads | MIT OR Apache-2.0 |
| [mimalloc](https://github.com/purpleprotocol/mimalloc_rust) (`mimalloc`, `libmimalloc-sys`, with Microsoft's [mimalloc](https://github.com/microsoft/mimalloc) C library) | memory allocator | MIT |
| [faer](https://codeberg.org/sarah-quinones/faer) (`faer`, `faer-traits`, `gemm`, `gemm-*`, `nano-gemm`, `nano-gemm-*`, `private-gemm-x86`, `pulp`, `dyn-stack`, `reborrow`, `equator`, `qd`, `spindle`) | dense linear algebra of the GLS (Cholesky, tridiagonalization) | MIT; parts derived from Eigen (MPL-2.0), LAPACK and SuiteSparse AMD/COLAMD (BSD-3-Clause, see below) |
| `raw-cpuid`, `libm` | support (through faer) | MIT |
| `atomic-wait` | support (through faer) | BSD-2-Clause (see below) |
| `libc`, `once_cell`, `portable-atomic`, `portable-atomic-util`, `rustc-hash`, `bitflags`, `bytemuck`, `cfg-if`, `generativity`, `num_cpus`, `windows-sys` | support | MIT OR Apache-2.0 (`bytemuck`: or Zlib) |

The parts of Eigen ported to Rust (above), and those faer derives from Eigen,
are covered by the Mozilla Public License 2.0 (<https://mozilla.org/MPL/2.0/>), which allows distributing them
as part of a GPL work.

This product includes software developed by the University of Chicago, as
Operator of Argonne National Laboratory (MINPACK, via Eigen's
NonLinearOptimization module, ported to Rust in
`rust/zeit-core/src/eigen/lm.rs`). The MINPACK copyright notice:

> Minpack Copyright Notice (1999) University of Chicago. All rights reserved
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. The end-user documentation included with the redistribution, if any, must
>    include the following acknowledgment: "This product includes software
>    developed by the University of Chicago, as Operator of Argonne National
>    Laboratory." Alternately, this acknowledgment may appear in the software
>    itself, if and wherever such third-party acknowledgments normally appear.
> 4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF
>    ANY KIND. THE COPYRIGHT HOLDER, THE UNITED STATES, THE UNITED STATES
>    DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES,
>    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF
>    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
>    NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY
>    FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT
>    REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED
>    RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED,
>    THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.
> 5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT HOLDER, THE UNITED
>    STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES: BE
>    LIABLE FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE
>    DAMAGES OF ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
>    PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER SUCH LIABILITY
>    IS ASSERTED ON THE BASIS OF CONTRACT, TORT (INCLUDING NEGLIGENCE OR STRICT
>    LIABILITY), OR OTHERWISE, EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF
>    THE POSSIBILITY OF SUCH LOSS OR DAMAGES.

## BSD-2-Clause notices (rust-numpy, atomic-wait)

> Copyright (c) 2017, Toshiki Teramura
> All rights reserved.
>
> Copyright (c) 2022, Mara Bos <m-ou.se@m-ou.se>
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## BSD-3-Clause notices (LAPACK, SuiteSparse AMD and COLAMD, through faer)

> Copyright (c) 1992-2025 The University of Tennessee and The University of
> Tennessee Research Foundation. All rights reserved.
> Copyright (c) 2000-2025 The University of California Berkeley. All rights
> reserved.
> Copyright (c) 2006-2025 The University of Colorado Denver. All rights
> reserved. (LAPACK)
>
> AMD, Copyright (c), 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
> Iain S. Duff. All Rights Reserved.
>
> COLAMD, Copyright 1998-2022, Timothy A. Davis.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of the copyright holders, nor of the organizations to which
>   the authors are affiliated, nor the names of its contributors may be used to
>   endorse or promote products derived from this software without specific
>   prior written permission.
>
> The copyright holders provide no reassurances that the source code provided
> does not infringe any patent, copyright, or any other intellectual property
> rights of third parties. The copyright holders disclaim any liability to any
> recipient for claims brought against recipient by any third party for
> infringement of that parties intellectual property rights. (LAPACK)
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## MIT License notices

The following copyright notices apply to the MIT-licensed code listed above:

- Copyright (c) 2003, Frank Warmerdam; Copyright (c) 2008-2013, Even Rouault (GDAL, `gdalwarpkernel.cpp`)
- Copyright (c) 2007 Andrey Kiselev; Copyright (c) 2009-2013 Even Rouault (GDAL 3.9.3, `gdalgrid.cpp`); Copyright (c) 2013 Even Rouault (GDAL 3.9.3, `gdalgridavx.cpp`)
- Copyright (c) 2018-2025 Microsoft Corporation, Daan Leijen (mimalloc); Copyright 2019 Octavian Oncescu (mimalloc crates)
- Copyright (c) 2018 Global Environmental Remote Sensing Lab (GERSL/CCDC)
- Copyright (c) 2021-2025 sarah quiñones el kazdadi (faer, faer-traits, gemm, nano-gemm, private-gemm-x86, pulp, dyn-stack, reborrow, equator, qd, spindle)
- Copyright (c) 2015 Gerd Zellweger (raw-cpuid)
- The Rust Project Developers (libm)
- Copyright (c) 2019 Md. Manjurul Hussain Shourov (pymannkendall)
- Copyright (c) 2021 VSainteuf (utae-paps)
- Copyright (c) 2020 VSainteuf (lightweight-temporal-attention-pytorch)
- Copyright (c) 2024 Presto Authors (nasaharvest/presto)

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The full texts of the GPL versions referred to above are available at
<https://www.gnu.org/licenses/>.
