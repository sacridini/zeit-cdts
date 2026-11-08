## DetectPotentialCloud.m alone, on arrays made by the harness (to put sum_clr at 39999, 40000 and 40001 exactly, and
## other corners). Files: <id>.{mask,water,sg,sr,idplcd} uint8, <id>.{swir1,ndvi,ndsi,ndbi,whiteness,hot} float32,
## <id>.cn double; writes <id>.{sumclr,cloud,idused,idusedempty}.
function fm_pc_direct (dir, id, rows, cols)
  b = [dir "/" id];
  td.BandSWIR1  = rd ([b ".swir1"], rows, cols, "single");
  td.BandCirrus = rd ([b ".cn"], rows, cols, "double");
  td.BandBT     = [];
  td.SatuGreen  = logical (rd ([b ".sg"], rows, cols, "uint8"));
  td.SatuRed    = logical (rd ([b ".sr"], rows, cols, "uint8"));
  mask = logical (rd ([b ".mask"], rows, cols, "uint8"));
  water = rd ([b ".water"], rows, cols, "uint8");
  idplcd = logical (rd ([b ".idplcd"], rows, cols, "uint8"));
  ndvi = rd ([b ".ndvi"], rows, cols, "single");
  ndsi = rd ([b ".ndsi"], rows, cols, "single");
  ndbi = rd ([b ".ndbi"], rows, cols, "single");
  whiteness = rd ([b ".whiteness"], rows, cols, "single");
  hot = rd ([b ".hot"], rows, cols, "single");
  meta.Dim = [rows cols];
  meta.Resolution = [20 20];
  [sum_clr, cloud, idused] = DetectPotentialCloud (meta, mask, water, td, [], ndvi, ndsi, ndbi, idplcd, whiteness, hot, 0.5, 20.0);
  wr ([b ".sumclr"], sum_clr, "double");
  wr ([b ".cloud"], cloud, "uint8");
  wr ([b ".idusedempty"], isempty (idused), "uint8");
  if (! isempty (idused))
    wr ([b ".idused"], idused, "uint8");
  endif
endfunction
