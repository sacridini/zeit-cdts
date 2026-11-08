## Recorders of the instrumented copies (fmask4_shadow.py adds a call to them in a copy of the .m; the originals are
## not touched). fm_rec_c: the c of the SCS+C; fm_rec_ids: the samples of the stratified sampling; fm_rec_match: the
## values of a match.
function fm_rec_c (c)
  global FM_C
  FM_C(end+1) = c;
endfunction
