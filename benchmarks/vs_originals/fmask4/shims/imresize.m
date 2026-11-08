## Shim of imresize for the comparison of Zeit's Fmask primitives with GNU Octave (image 2.20).
##
## Octave's imresize (A, 0.5, "box") does not average the 2 x 2 blocks (it returns one pixel of each: [1..16] as 4 x 4
## gives [6 8; 14 16], where MATLAB's box kernel gives the means [3.5 5.5; 11.5 13.5]). Fmask uses the box resize to
## bring 10 m to 20 m, so this shim returns the mean of the blocks: the pairs of rows, then the pairs of columns, in
## the class of A. Every other call goes to the original, whose handle the driver kept in FMASK_ORIG_IMRESIZE before
## putting this folder first in the path.
function B = imresize (A, varargin)
  global FMASK_ORIG_IMRESIZE
  if (numel (varargin) == 2 && ischar (varargin{2}) && strcmp (varargin{2}, "box") && isscalar (varargin{1}) && varargin{1} == 0.5)
    B = (A(1:2:end, :) + A(2:2:end, :)) / 2;
    B = (B(:, 1:2:end) + B(:, 2:2:end)) / 2;
  else
    B = FMASK_ORIG_IMRESIZE (A, varargin{:});
  endif
endfunction
