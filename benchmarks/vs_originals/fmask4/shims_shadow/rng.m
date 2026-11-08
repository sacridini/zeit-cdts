## Shim of rng for the comparison of Zeit's Fmask cloud shadow step with GNU Octave (fmask4_shadow.py).
##
## Fmask's stratiedSampleHanlder.m and MatchCloudShadow.m call "rng(1)" before "randperm" ("static seed"). MATLAB's
## generator cannot be reproduced in Octave, so both sides use the same deterministic draw (a 64-bit LCG and a partial
## Fisher-Yates shuffle, zeit_core::fmask::shadow::fmask_randperm): rng (1) puts the LCG at the state 1, and the
## randperm shim (next to this file) advances it. It is a deviation from MATLAB, documented in shadow.rs.
function rng (varargin)
  global FMASK_RNG_HI FMASK_RNG_LO
  if (numel (varargin) != 1 || ! isnumeric (varargin{1}) || varargin{1} != 1)
    error ("rng shim: only rng (1) is supported");
  endif
  FMASK_RNG_HI = uint64 (0);
  FMASK_RNG_LO = uint64 (1);
endfunction
