## The chain of the cloud shadow step of autoFmask.m (lines 226-241) for one Sentinel-2 20 m scene, run with the original
## .m files of Fmask 4.7 (none is edited) and, a second time, with copies instrumented by fmask4_shadow.py (a line added
## that records the values the .m keeps inside: the c of the SCS+C, the samples, and at every match the height, the
## similarity and the height of the neighbours), whose outputs must be equal to the originals'.
##
## args: zen azi hasdem hastopo ptm isShadowater
## Files in dir (float32 unless said, rows x cols, row-major): <id>.{nir,swir} (TOA times 10000, -9999 without data),
## <id>.{clear,mask,cloud,waterall,snow} (uint8), <id>.{vza,vaa} (hundredths of a degree), <id>.dem, <id>.slope,
## <id>.aspect (if hasdem / hastopo).
function fm_shadow_scene (dir, id, rows, cols, args)
  global FM_REC FM_C FM_IDS
  b = [dir "/" id];
  zen = str2double (args{1}); azi = str2double (args{2});
  hasdem = str2double (args{3}); hastopo = str2double (args{4});
  ptm = str2double (args{5}); isw = str2double (args{6}) != 0;
  nir = rdv ([b ".nir"], rows, cols, "single");
  swir = rdv ([b ".swir"], rows, cols, "single");
  clearland = logical (rdv ([b ".clear"], rows, cols, "uint8"));
  mask = logical (rdv ([b ".mask"], rows, cols, "uint8"));
  pcloud = rdv ([b ".cloud"], rows, cols, "uint8");
  waterall = logical (rdv ([b ".waterall"], rows, cols, "uint8"));
  psnow = rdv ([b ".snow"], rows, cols, "uint8");
  angles.VZA = rdv ([b ".vza"], rows, cols, "single");
  angles.VAA = rdv ([b ".vaa"], rows, cols, "single");
  dem = []; slope = []; aspect = [];
  if (hasdem) dem = rdv ([b ".dem"], rows, cols, "single"); endif
  if (hastopo)
    slope = rdv ([b ".slope"], rows, cols, "single");
    aspect = rdv ([b ".aspect"], rows, cols, "single");
  endif
  meta.Zen = zen; meta.Azi = azi; meta.Dim = [rows cols]; meta.Resolution = [20 20]; meta.Sensor = "S_MSI";

  ## stage 1: autoFmask.m:226
  tic;
  ps = DetectPotentialCloudShadow (meta, nir, swir, clearland, mask, slope, aspect);
  t1 = toc;
  wrv ([b ".ps"], ps, "uint8");
  FM_C = []; FM_IDS = [];
  ps2 = DetectPotentialCloudShadow_rec (meta, nir, swir, clearland, mask, slope, aspect);
  if (! isequal (ps, ps2)) printf ("ERRDIFF %s: the instrumented DetectPotentialCloudShadow differs\n", id); endif
  wrv ([b ".c"], FM_C(:)', "double");
  wrv ([b ".ids"], FM_IDS(:)', "double");
  clear nir swir ps2;

  ## stage 2: autoFmask.m:235 (the input is the output of stage 1)
  tic;
  [sim, cl, sh] = MatchCloudShadow (mask, pcloud, ps, isw, waterall, dem, [], [], [], meta, ptm, 14, angles);
  t2 = toc;
  wrv ([b ".sim"], sim(:)', "double");
  wrv ([b ".cl"], cl, "uint8");
  wrv ([b ".sh"], sh, "uint8");
  FM_REC = zeros (0, 5);
  [sim2, cl2, sh2] = MatchCloudShadow_rec (mask, pcloud, ps, isw, waterall, dem, [], [], [], meta, ptm, 14, angles);
  if (! (isequal (sim, sim2) && isequal (cl, cl2) && isequal (sh, sh2)))
    printf ("ERRDIFF %s: the instrumented MatchCloudShadow differs\n", id);
  endif
  wrv ([b ".rec"], FM_REC, "double");
  wrv ([b ".nrec"], size (FM_REC, 1), "double");

  ## autoFmask.m:241 with the default buffers of 3 pixels
  [bc, bs, bn] = BufferMasks (cl, 3, sh, 3, psnow, 3);
  wrv ([b ".bc"], bc, "uint8");
  wrv ([b ".bs"], bs, "uint8");
  wrv ([b ".bn"], bn, "uint8");
  printf ("  scene %s: DetectPotentialCloudShadow %.1f s, MatchCloudShadow %.1f s\n", id, t1, t2);
endfunction

function x = rdv (f, rows, cols, cls)
  fid = fopen (f, "rb");
  x = fread (fid, [cols, rows], [cls "=>" cls])';
  fclose (fid);
endfunction

function wrv (f, x, cls)
  fid = fopen (f, "wb");
  fwrite (fid, x', cls);
  fclose (fid);
endfunction
