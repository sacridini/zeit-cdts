## The neighbourhood of strel ("disk", r) of MATLAB (N = 4), for the shim of strel in fmask4.py (Octave's strel only
## takes N = 0, the exact disk).
##
## Written from the description of the approximation by periodic lines (Adams 1993; Jones and Soille 1996), in a way
## independent of the Rust one (shifts of a grid here, not point sets): for r < 3 the exact disk; otherwise the sum
## (dilation) of the four periodic lines of the directions (1,0), (1,1), (0,1), (-1,1), each i * v for i in -rp..rp
## with rp = floor (k / |v|), k = 2 r / (cot (pi/8) + 1 / sin (pi/8)); then ext is the half width in x of the result,
## len = 2 * (r - ext) - 1 and, if len >= 3, a horizontal and a vertical line of len pixels (UNCONFIRMED for the radii
## where no published shape was found, r = 10 among them: the same formula as the Rust).
function nh = fmask_disk (r)
  if (r < 3)
    [x, y] = meshgrid (-r:r, -r:r);
    nh = (x.^2 + y.^2) <= r^2;
    return;
  endif
  theta = pi / 8;
  k = 2 * r / (cot (theta) + 1 / sin (theta));
  v = [1 0; 1 1; 0 1; -1 1];
  M = 4 * r + 5;                      # half size of the working grid (never reached)
  g = false (2*M + 1);
  g(M+1, M+1) = true;
  for d = 1:4
    rp = floor (k / norm (v(d,:)));
    h = false (2*M + 1);
    for i = -rp:rp
      h |= circshift (g, [i * v(d,2), i * v(d,1)]);
    endfor
    g = h;
  endfor
  [yy, xx] = find (g);
  ext = max (abs (xx - (M + 1)));
  len = 2 * (r - ext) - 1;
  if (len >= 3)
    hl = (len - 1) / 2;
    h = false (2*M + 1);
    for i = -hl:hl
      h |= circshift (g, [0, i]);
    endfor
    g = h;
    h = false (2*M + 1);
    for i = -hl:hl
      h |= circshift (g, [i, 0]);
    endfor
    g = h;
  endif
  [yy, xx] = find (g);
  half = max ([abs(yy - (M + 1)); abs(xx - (M + 1))]);
  nh = g(M+1-half:M+1+half, M+1-half:M+1+half);
endfunction
