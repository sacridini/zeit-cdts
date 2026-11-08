## The first k states (halves of 32 bits, as uint64 column vectors) of the 64-bit LCG s = s * A + C mod 2^64, A =
## 0x5851F42D4C957F2D, C = 0x14057B7EF767814F, after the state (h0, l0). The j-th step is the affine map x -> A_j x + C_j;
## A_j and C_j for j = 1..k are built by doubling (x_{m+j} = A_j (A_m x + C_m) + C_j), every product of two numbers of
## 32 bits being below 2^64, so that the uint64 arithmetic of Octave (which saturates) never does.
function [sh, sl] = fm_lcg_states (k, h0, l0)
  M = uint64 (4294967295);
  ah = uint64 (hex2dec ("5851F42D")); al = uint64 (hex2dec ("4C957F2D"));
  ch = uint64 (hex2dec ("14057B7E")); cl = uint64 (hex2dec ("F767814F"));
  Ah = ah; Al = al; Ch = ch; Cl = cl;
  m = 1;
  while (m < k)
    amh = Ah(m); aml = Al(m); cmh = Ch(m); cml = Cl(m);
    [nah, nal] = mul64 (Ah, Al, amh, aml, M);
    [th, tl] = mul64 (Ah, Al, cmh, cml, M);
    [nch, ncl] = add64 (th, tl, Ch, Cl, M);
    Ah = [Ah; nah]; Al = [Al; nal]; Ch = [Ch; nch]; Cl = [Cl; ncl];
    m = 2 * m;
  endwhile
  Ah = Ah(1:k); Al = Al(1:k); Ch = Ch(1:k); Cl = Cl(1:k);
  [th, tl] = mul64 (Ah, Al, uint64 (h0), uint64 (l0), M);
  [sh, sl] = add64 (th, tl, Ch, Cl, M);
endfunction

function [rh, rl] = mul64 (ah, al, bh, bl, M)
  L = al .* bl;
  cross = bitand (ah .* bl, M) + bitand (al .* bh, M);
  rl = bitand (L, M);
  rh = bitand (bitshift (L, -32) + cross, M);
endfunction

function [rh, rl] = add64 (ah, al, bh, bl, M)
  s = al + bl;
  rl = bitand (s, M);
  rh = bitand (ah + bh + bitshift (s, -32), M);
endfunction
