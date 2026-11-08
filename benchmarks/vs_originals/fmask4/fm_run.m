## The Octave side of benchmarks/vs_originals/fmask4_run.py: for each line of FMASK_DIR/cases.txt,
##   "scene <id> rows cols hasdem hasgs hastopo zen azi"   the chain of autoFmask.m for one Sentinel-2 20 m scene
##   "toa <id> rows cols <legacy|b04|b04z> quantif"      the DN to TOA math of nd2toarbt_msi2.m / nd2toarbt_msi.m
## The original .m files of Fmask 4.7 (FMASK_SRC on the path) are called as they are; none is edited. The shims
## (FMASK_SHIMS: imresize, fmask_disk, the patched @strel, bwlabeln, rng, randperm) are first in the path.
pkg load image statistics
global FMASK_ORIG_IMRESIZE FMASK_ORIG_BWLABELN FMASK_RNG_HI FMASK_RNG_LO
FMASK_ORIG_IMRESIZE = @imresize;
FMASK_ORIG_BWLABELN = @bwlabeln;
FMASK_RNG_HI = uint64 (0);
FMASK_RNG_LO = uint64 (1);
if (! isempty (getenv ("FMASK_SRC")))
  addpath (getenv ("FMASK_SRC"));
endif
addpath (getenv ("FMASK_SHIMS"));
dir = getenv ("FMASK_DIR");

function x = rd (f, rows, cols, cls)
  fid = fopen (f, "rb");
  x = fread (fid, [cols, rows], [cls "=>" cls])';
  fclose (fid);
endfunction

function wr (f, x, cls)
  fid = fopen (f, "wb");
  fwrite (fid, x', cls);
  fclose (fid);
endfunction

## autoFmask.m lines 135-251 for the S2, from the TOA bands on (LoadData and the reading of the files are not run). The
## calls, their arguments and their order are those of autoFmask.m; the only things that are not: the data come from
## files, and every result is also written.
function fm_run_scene (dir, id, rows, cols, hasdem, hasgs, hastopo, zen, azi)
  b = [dir "/" id];
  td.BandBlue   = rd ([b ".blue"],   rows, cols, "single");
  td.BandGreen  = rd ([b ".green"],  rows, cols, "single");
  td.BandRed    = rd ([b ".red"],    rows, cols, "single");
  td.BandNIR    = rd ([b ".nir"],    rows, cols, "single");
  td.BandSWIR1  = rd ([b ".swir1"],  rows, cols, "single");
  td.BandSWIR2  = rd ([b ".swir2"],  rows, cols, "single");
  td.BandCirrus = rd ([b ".cirrus"], rows, cols, "single");
  td.BandVRE3   = rd ([b ".b7"],     rows, cols, "single");
  td.BandNIR8   = rd ([b ".b8"],     rows, cols, "single");
  td.BandBT     = [];
  td.SatuBlue   = logical (rd ([b ".sb"], rows, cols, "uint8"));
  td.SatuGreen  = logical (rd ([b ".sg"], rows, cols, "uint8"));
  td.SatuRed    = logical (rd ([b ".sr"], rows, cols, "uint8"));
  dem = []; slope = []; aspect = []; water_occur = [];
  if (hasdem)
    dem = rd ([b ".dem"], rows, cols, "single");
  endif
  if (hastopo)
    slope = rd ([b ".slope"], rows, cols, "single");
    aspect = rd ([b ".aspect"], rows, cols, "single");
  endif
  if (hasgs)
    water_occur = rd ([b ".gs"], rows, cols, "uint8");
  endif
  angles_view.VZA = rd ([b ".vza"], rows, cols, "single");
  angles_view.VAA = rd ([b ".vaa"], rows, cols, "single");
  data_meta.Dim = [rows cols];
  data_meta.Resolution = [20 20];
  data_meta.Sensor = "S_MSI";
  data_meta.Zen = zen;
  data_meta.Azi = azi;

  default_paras = FmaskParameters ("S_MSI");
  tpw = default_paras.ThinWeight;
  resolution = default_paras.OutputResolution;
  cldpix = default_paras.CloudBuffer;
  sdpix = default_paras.CloudShadowBuffer;
  snpix = default_paras.SnowBuffer;
  expdpix = round (default_paras.PFPCLayerExtensinRadius / resolution);
  erdpix = round (default_paras.PFPCErosionRadius / resolution);
  cldprob = default_paras.CloudProbabilityThershold;
  isShadowater = default_paras.ShadowWater;

  mask = ObservMask (td.BandBlue);
  wr ([b ".mask"], mask, "uint8");
  if (! isempty (dem))
    dem_nan = isnan (dem);
    dem(dem_nan) = double (prctile (dem(! dem_nan & mask), 0.001));
    clear dem_nan;
  endif
  ndvi = NDVI (td.BandRed, td.BandNIR);
  ndsi = NDSI (td.BandGreen, td.BandSWIR1);
  cdi = CDI (td.BandVRE3, td.BandNIR8, td.BandNIR);
  td.BandVRE3 = [];
  td.BandNIR8 = [];
  satu_Bv = Saturate (td.SatuBlue, td.SatuGreen, td.SatuRed);
  td.SatuBlue = [];

  psnow = DetectSnow (data_meta.Dim, td.BandGreen, td.BandNIR, td.BandBT, ndsi);
  wr ([b ".psnow0"], psnow, "uint8");
  [water, waterAll] = DetectWater (data_meta.Dim, mask, td.BandNIR, ndvi, psnow, slope, water_occur);
  clear water_occur;
  wr ([b ".water"], water, "uint8");
  wr ([b ".waterall"], waterAll, "uint8");

  [idplcd, BandCirrusNormal, whiteness, HOT] = DetectPotentialPixels (mask, td, dem, ndvi, ndsi, satu_Bv);
  td.BandBlue = [];
  td.BandRed = [];
  td.BandSWIR2 = [];
  clear satu_Bv;
  td.BandCirrus = BandCirrusNormal;
  clear BandCirrusNormal;

  abs_snow = DetectAbsSnow (td.BandGreen, td.SatuGreen, ndsi, psnow, data_meta.Resolution(1));
  if (! isnan (abs_snow))
    idplcd(abs_snow == 1) = 0; clear abs_snow;
  endif
  wr ([b ".idplcd"], idplcd, "uint8");

  ndbi = NDBI (td.BandNIR, td.BandSWIR1);
  [sum_clr, pcloud_all, idlnd, t_templ, t_temph] = DetectPotentialCloud (data_meta, mask, water, td, dem, ndvi, ndsi, ndbi, idplcd, whiteness, HOT, tpw, cldprob);
  clear ndsi idplcd whiteness HOT tpw cldprob;
  wr ([b ".sumclr"], sum_clr, "double");
  wr ([b ".cloud"], pcloud_all, "uint8");
  wr ([b ".idlndempty"], isempty (idlnd), "uint8");
  if (! isempty (idlnd))
    wr ([b ".idlnd"], idlnd, "uint8");
  endif
  td.SatuGreen = [];
  td.SatuRed = [];
  td.BandCirrus = [];

  pfpl = DetectPotentialFalsePositivePixels (mask, psnow, slope, ndbi, ndvi, td.BandBT, cdi, water, data_meta.Resolution(1));
  clear ndbi ndvi;
  if (expdpix > 0)
    PFPCEs = strel ("square", 2 * expdpix + 1);
    pfpl = imdilate (pfpl, PFPCEs);
    clear PFPCEs;
  endif
  wr ([b ".pfpl"], pfpl, "uint8");

  pcloud = ErodeCommissons (data_meta, pcloud_all, pfpl, water, cdi, erdpix);
  clear cdi;
  wr ([b ".pcloud_eroded"], pcloud, "uint8");

  cs_final = zeros (data_meta.Dim, "uint8");
  cs_final(water == 1) = 1;

  tshadow = 0; tmatch = 0;
  if (sum_clr <= 40000)
    pcloud = pcloud > 0;
    pshadow = ! pcloud;
    wr ([b ".path"], 0, "uint8");
    pcloud0 = pcloud;
  else
    tic;
    pshadow = DetectPotentialCloudShadow (data_meta, td.BandNIR, td.BandSWIR1, idlnd, mask, slope, aspect);
    tshadow = toc;
    wr ([b ".pshadow_pot"], pshadow, "uint8");
    td.BandNIR = [];
    td.BandSWIR1 = [];
    data_bt_c = td.BandBT;
    tic;
    [sim, pcloud, pshadow] = MatchCloudShadow (mask, pcloud, pshadow, isShadowater, waterAll, dem, data_bt_c, t_templ, t_temph, data_meta, sum_clr, 14, angles_view);
    tmatch = toc;
    wr ([b ".sim"], sim(:)', "double");
    wr ([b ".match_cloud"], pcloud, "uint8");
    wr ([b ".match_shadow"], pshadow, "uint8");
    wr ([b ".path"], 1, "uint8");
    pcloud0 = pcloud;
    [pcloud, pshadow, psnow] = BufferMasks (pcloud, cldpix, pshadow, sdpix, psnow, snpix);
  endif
  wr ([b ".pcloud0"], pcloud0, "uint8");
  wr ([b ".pcloud"], pcloud, "uint8");
  wr ([b ".pshadow"], pshadow, "uint8");
  wr ([b ".psnow"], psnow, "uint8");

  cs_final(psnow == 1) = 3;
  cs_final(pshadow == 1) = 2;
  cs_final(pcloud == 1) = 4;
  cs_final(mask == 0) = 255;
  wr ([b ".cs_final"], cs_final, "uint8");
  clr_pct = 100 * (1 - sum (pcloud(:)) / sum (mask(:)));
  wr ([b ".clrpct"], clr_pct, "double");
  printf ("  scene %s: DetectPotentialCloudShadow %.1f s, MatchCloudShadow %.1f s\n", id, tshadow, tmatch);
endfunction

## nd2toarbt_msi2.m lines 85-147 (b04) or nd2toarbt_msi.m lines 85-147 (legacy), on arrays instead of the .jp2 files.
## Files in dir: <id>.d<k> (uint16, the DN of band k = 1..9 in the order BandCorr, at its native size, row-major).
function fm_toa_scene (dir, id, rows, cols, kind, quantif)
  b = [dir "/" id];
  BandCorr = {"02","03","04","8A","11","12","10","07","08"};
  BandCorrId = [1, 2, 3, 8, 11, 12, 10, 6, 7];
  band_offsets = [1 -1000; 2 -900; 3 -1000; 4 -1; 5 -100; 6 -1; 7 -1000; 8 -500; 9 -1000; 10 -50; 11 -1000; 12 -1000; 13 -1000];   # as the metadata has them
  if (strcmp (kind, "b04z"))
    band_offsets(:, 2) = 0;       # zero offsets: the saturation 65535 can then be reached after the scaling
  endif
  resolu = [20 20];
  Ratio = [10 10 10 20 20 20 60 20 10] / (resolu(1));
  nat = {2, 2, 2, 1, 1, 1, 1/3, 1, 2};
  ijdim_ref = [rows cols];
  id_missing = zeros (ijdim_ref, "uint8");
  for iB = 1:length (BandCorr)
    r = round (rows * nat{iB}); c = round (cols * nat{iB});
    dum = rd ([b ".d" num2str(iB)], r, c, "uint16");
    if (strncmp (kind, "b04", 3))
      mask_filled = dum == 0;
      dum = 10000 .* (single (dum) + band_offsets(BandCorrId(iB) == band_offsets(:, 1), 2)) ./ quantif;
      dum(mask_filled) = NaN;
      clear mask_filled;
    else
      dum = single (dum);
      dum(dum == 0) = NaN;
    endif
    if (Ratio(iB) < 1)
      TOAref(:,:,iB) = imresize (dum, Ratio(iB), "box");
    elseif (Ratio(iB) > 1)
      TOAref(:,:,iB) = imresize (dum, Ratio(iB), "nearest");
    elseif (Ratio(iB) == 1)
      TOAref(:,:,iB) = dum;
    endif
    clear dum;
    id_missing = id_missing | isnan (TOAref(:,:,iB));
  endfor
  TOAref(id_missing) = -9999;
  B1Satu = TOAref(:,:,1) == 65535;
  B2Satu = TOAref(:,:,2) == 65535;
  B3Satu = TOAref(:,:,3) == 65535;
  for k = 1:9
    wr ([b ".t" num2str(k)], TOAref(:,:,k), "single");
  endfor
  wr ([b ".sb"], B1Satu, "uint8");
  wr ([b ".sg"], B2Satu, "uint8");
  wr ([b ".sr"], B3Satu, "uint8");
  wr ([b ".class"], strcmp (class (TOAref), "single"), "uint8");
endfunction

fid = fopen ([dir "/cases.txt"], "r");
while (true)
  ln = fgetl (fid);
  if (! ischar (ln)) break; endif
  t = strsplit (strtrim (ln), " ");
  op = t{1}; id = t{2}; rows = str2double (t{3}); cols = str2double (t{4});
  tic;
  try
    switch (op)
      case "scene"
        fm_run_scene (dir, id, rows, cols, str2double (t{5}), str2double (t{6}), str2double (t{7}), str2double (t{8}), str2double (t{9}));
      case "toa"
        fm_toa_scene (dir, id, rows, cols, t{5}, str2double (t{6}));
    endswitch
  catch err
    fid2 = fopen ([dir "/" id ".err"], "w"); fputs (fid2, [err.message " @ " err.stack(1).name ":" num2str(err.stack(1).line)]); fclose (fid2);
    printf ("ERROR %s: %s\n", id, err.message);
  end_try_catch
  printf ("%s %s %.3f s\n", op, id, toc);
  fflush (stdout);
endwhile
fclose (fid);
