## See fm_rec_c.m.
function fm_rec_match (cur, type, base_h, thresh, near)
  global FM_REC
  ## every value is made double before they meet in a row (a single in it would make the row single)
  FM_REC(end+1, :) = [double(cur) double(type) double(base_h) double(thresh) double(near)];
endfunction
