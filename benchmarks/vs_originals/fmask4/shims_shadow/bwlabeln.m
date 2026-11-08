## Shim of bwlabeln for the comparison of Zeit's Fmask cloud shadow step with GNU Octave.
##
## Octave's bwlabeln does not number the objects in the order of their first pixel in a scan by columns (MATLAB's
## numbering, which Zeit's bwlabel keeps), and in MatchCloudShadow.m the order of the objects changes the result (the
## first 14 are revisited, the matched heights are those of the earlier ones). This shim calls the original (its handle
## kept in FMASK_ORIG_BWLABELN by the driver before this folder is put on the path) and renumbers the objects.
function [L, n] = bwlabeln (varargin)
  global FMASK_ORIG_BWLABELN
  [L, n] = FMASK_ORIG_BWLABELN (varargin{:});
  idx = find (L);
  if (isempty (idx))
    return;
  endif
  labs = L(idx);
  [~, ia] = unique (labs, "first");        # first position of every label in the scan by columns
  [~, ord] = sort (ia);
  u = labs(ia(ord));                       # the labels in the order of their first pixel
  perm = zeros (double (max (labs(:))), 1);
  perm(double (u)) = 1:numel (u);
  L(idx) = cast (perm(double (labs)), class (L));
endfunction
