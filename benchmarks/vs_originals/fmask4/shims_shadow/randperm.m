## Shim of randperm for the comparison of Zeit's Fmask cloud shadow step with GNU Octave (see rng.m).
##
## randperm (n) / randperm (n, k) as zeit_core::fmask::shadow::fmask_randperm: a 64-bit LCG (s = s * 6364136223846793005 +
## 1442695040888963407 mod 2^64, after rng (1) at s = 1) and a partial Fisher-Yates shuffle, in which step i (0-based)
## swaps the positions i and i + mod (s >> 32, n - i), s being the state after i + 1 steps. The states of all the steps
## are computed at once (the LCG is an affine map: its j-th power is built by doubling, in 32-bit halves held in uint64
## so that no product overflows), the swaps are a loop. The state is left after the k steps.
function p = randperm (n, k)
  global FMASK_RNG_HI FMASK_RNG_LO
  if (nargin < 2)
    k = n;
  endif
  if (isempty (FMASK_RNG_LO))
    error ("randperm shim: call rng (1) first");
  endif
  p = 1:n;
  if (k < 1)
    p = p(1:0);
    return;
  endif
  [sh, sl] = fm_lcg_states (k, FMASK_RNG_HI, FMASK_RNG_LO);
  i = (0:k-1)';
  j = i + mod (double (sh), n - i);          # 0-based partner of every step
  j = j + 1;
  for t = 1:k
    u = j(t);
    if (u != t)
      v = p(t); p(t) = p(u); p(u) = v;
    endif
  endfor
  p = p(1:k);
  FMASK_RNG_HI = sh(k);
  FMASK_RNG_LO = sl(k);
endfunction
