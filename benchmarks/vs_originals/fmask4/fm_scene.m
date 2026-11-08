## The chain of autoFmask.m for one Sentinel-2 20 m scene (from the TOA bands on), run with the original .m files of
## Fmask 4.7 (FMASK_SRC is on the path; none is edited), writing every intermediate result for the comparison with
## Zeit's (benchmarks/vs_originals/fmask4.py). The calls and their arguments are those of autoFmask.m (lines 142-188);
## only the Sentinel-2 route is run (no BT, no CDI). Besides the chain, the functions that take the output of another
## are also called alone on it (the 'cn2' and 'p*' files), so that each stage can be compared with the same input.
##
## Files in dir: <id>.{blue,green,red,nir,swir1,swir2,cirrus} (float32, rows x cols, row-major), <id>.{sb,sg,sr}
## (uint8 saturation), <id>.dem (float32, if hasdem), <id>.gs (uint8, if hasgs).
function fm_scene (dir, id, rows, cols, hasdem, hasgs)
  b = [dir "/" id];
  td.BandBlue   = rd ([b ".blue"],   rows, cols, "single");
  td.BandGreen  = rd ([b ".green"],  rows, cols, "single");
  td.BandRed    = rd ([b ".red"],    rows, cols, "single");
  td.BandNIR    = rd ([b ".nir"],    rows, cols, "single");
  td.BandSWIR1  = rd ([b ".swir1"],  rows, cols, "single");
  td.BandSWIR2  = rd ([b ".swir2"],  rows, cols, "single");
  td.BandCirrus = rd ([b ".cirrus"], rows, cols, "single");
  td.BandBT     = [];
  td.SatuBlue   = logical (rd ([b ".sb"], rows, cols, "uint8"));
  td.SatuGreen  = logical (rd ([b ".sg"], rows, cols, "uint8"));
  td.SatuRed    = logical (rd ([b ".sr"], rows, cols, "uint8"));
  dem = [];
  if (hasdem)
    dem = rd ([b ".dem"], rows, cols, "single");
  endif
  gs = [];
  if (hasgs)
    gs = rd ([b ".gs"], rows, cols, "uint8");
  endif
  meta.Dim = [rows cols];
  meta.Resolution = [20 20];
  hascdi = exist ([b ".b7"], "file") == 2;
  hasslope = exist ([b ".slope"], "file") == 2;

  prm = FmaskParameters ('S_MSI');
  wr ([b ".prm"], [prm.CloudBuffer prm.CloudShadowBuffer prm.SnowBuffer prm.ThinWeight prm.CloudProbabilityThershold ...
                   prm.PFPCLayerExtensinRadius prm.PFPCErosionRadius prm.OutputResolution prm.ShadowWater], "double");

  mask = ObservMask (td.BandBlue);
  ndvi = NDVI (td.BandRed, td.BandNIR);
  ndsi = NDSI (td.BandGreen, td.BandSWIR1);
  satu_Bv = Saturate (td.SatuBlue, td.SatuGreen, td.SatuRed);
  wr ([b ".mask"], mask, "uint8");
  wr ([b ".ndvi"], ndvi, "single");
  wr ([b ".ndsi"], ndsi, "single");
  wr ([b ".satubv"], satu_Bv, "uint8");
  wr ([b ".ndvi_class"], strcmp (class (ndvi), "single"), "uint8");

  psnow = DetectSnow (meta.Dim, td.BandGreen, td.BandNIR, td.BandBT, ndsi);
  wr ([b ".psnow"], psnow, "uint8");
  [water, waterAll] = DetectWater (meta.Dim, mask, td.BandNIR, ndvi, psnow, [], gs);
  wr ([b ".water"], water, "uint8");
  wr ([b ".waterall"], waterAll, "uint8");

  [idplcd, cn, whiteness, HOT] = DetectPotentialPixels (mask, td, dem, ndvi, ndsi, satu_Bv);
  wr ([b ".idplcd"], idplcd, "uint8");
  wr ([b ".cn"], cn, "double");
  wr ([b ".whiteness"], whiteness, "single");
  wr ([b ".hot"], HOT, "single");
  wr ([b ".cn_class"], strcmp (class (cn), "double"), "uint8");

  ## NormalizaCirrusDEM alone, on the potential pixels as input
  cn2 = NormalizaCirrusDEM (mask, idplcd, td.BandCirrus, dem);
  wr ([b ".cn2"], cn2, "double");

  abs_snow = DetectAbsSnow (td.BandGreen, td.SatuGreen, ndsi, psnow, meta.Resolution(1));
  wr ([b ".absempty"], isempty (abs_snow), "uint8");
  if (! isempty (abs_snow))
    wr ([b ".abssnow"], abs_snow, "uint8");
  endif
  wr ([b ".psnow_n"], sum (psnow(:)), "double");

  ndbi = NDBI (td.BandNIR, td.BandSWIR1);
  wr ([b ".ndbi"], ndbi, "single");
  idplcd2 = idplcd;
  if (! isempty (abs_snow))
    idplcd2(abs_snow == 1) = 0;
  endif
  wr ([b ".idplcd2"], idplcd2, "uint8");

  td.BandCirrus = cn;
  [sum_clr, cloud, idused, tl, th] = DetectPotentialCloud (meta, mask, water, td, dem, ndvi, ndsi, ndbi, idplcd2, whiteness, HOT, prm.ThinWeight, prm.CloudProbabilityThershold);
  wr ([b ".sumclr"], sum_clr, "double");
  wr ([b ".cloud"], cloud, "uint8");
  wr ([b ".idusedempty"], isempty (idused), "uint8");
  if (! isempty (idused))
    wr ([b ".idused"], idused, "uint8");
  endif
  wr ([b ".tl"], [tl th], "double");

  ## the probabilities, alone
  idclr = idplcd2 == false & mask == 1;
  idlnd = idclr & water == false;
  wr ([b ".pthin"], probThin (cn), "double");
  wr ([b ".pwb"], probwBrightness (td.BandSWIR1), "single");
  if (nnz (idlnd) > 0)
    wr ([b ".plb"], problBrightness (HOT, idlnd, 0.175, 1 - 0.175), "single");
  endif
  wr ([b ".psv"], problSpectralVaribility (ndvi, ndsi, ndbi, whiteness, td.SatuGreen, td.SatuRed), "single");
  wr ([b ".idlnd"], idlnd, "uint8");

  ## (Part A) the values the .m keeps inside, recomputed by a literal replica of DetectPotentialCloud.m 112-126
  if (sum_clr > 40000)
    l_pt = 0.175; h_pt = 1 - l_pt;
    idwt_ = idclr & water == true;
    wpt = prm.ThinWeight; cldprob = prm.CloudProbabilityThershold;
    prob_thin = probThin (cn);
    wprob_brightness = probwBrightness (td.BandSWIR1);
    lprob_brightness = problBrightness (HOT, idused, l_pt, h_pt);
    lprob_vari = problSpectralVaribility (ndvi, ndsi, ndbi, whiteness, td.SatuGreen, td.SatuRed);
    wprob_temp = 1; lprob_temp = 1;
    wprob_final=wprob_temp.*wprob_brightness + wpt.*prob_thin;
    wprob_final=100.*wprob_final;
    wclr_h=prctile(wprob_final(idwt_),100*h_pt);
    lprob_final=lprob_temp.*lprob_vari.*lprob_brightness + wpt.*prob_thin;
    lprob_final=100.*lprob_final;
    clr_h=prctile(lprob_final(idlnd),100*h_pt);
    wclr_max=wclr_h+cldprob;
    clr_max=clr_h+cldprob;
    wr ([b ".lprob"], lprob_final, "single");
    wr ([b ".wprob"], wprob_final, "single");
    wr ([b ".thr"], [clr_max wclr_max], "single");
    wr ([b ".probclass"], [strcmp(class(lprob_final),"single") strcmp(class(wprob_final),"single") strcmp(class(clr_max),"single") strcmp(class(wclr_max),"single")], "uint8");
  endif
  if (! isempty (abs_snow))
    [scsi, scls] = fm_scsi (td.BandGreen, ndsi, psnow, meta.Resolution(1));
    wr ([b ".scsi"], scsi, "double");
    wr ([b ".scsiclass"], strcmp (scls, "single"), "uint8");
  endif

  ## (Part B) CDI, EnhanceLine, DetectPotentialFalsePositivePixels, ErodeCommissons (autoFmask.m 147, 196, 209)
  if (hascdi)
    cdi = CDI (rd ([b ".b7"], rows, cols, "single"), rd ([b ".b8"], rows, cols, "single"), td.BandNIR);
    wr ([b ".cdi"], cdi, "double");
    wr ([b ".cdiclass"], strcmp (class (cdi), "double"), "uint8");
    wr ([b ".ndbienh"], EnhanceLine (ndbi), "single");
    slope = [];
    if (hasslope)
      slope = rd ([b ".slope"], rows, cols, "single");
    endif
    erdpix = round (90 / 20);
    pa = DetectPotentialFalsePositivePixels (mask, psnow, slope, ndbi, ndvi, [], cdi, water, 20);
    pb = DetectPotentialFalsePositivePixels (mask, psnow, [], ndbi, ndvi, [], cdi, water, 20);
    pc = DetectPotentialFalsePositivePixels (mask, psnow, [], ndbi, ndvi, [], [], water, 20);
    wr ([b ".pfpl_a"], pa, "uint8");
    wr ([b ".pfpl_b"], pb, "uint8");
    wr ([b ".pfpl_c"], pc, "uint8");
    wr ([b ".ero_a"], ErodeCommissons (meta, cloud, pa, water, cdi, erdpix), "uint8");
    wr ([b ".ero_b"], ErodeCommissons (meta, cloud, pb, water, cdi, erdpix), "uint8");
    wr ([b ".ero_c"], ErodeCommissons (meta, cloud, pc, water, [], erdpix), "uint8");
    wr ([b ".ero_d"], ErodeCommissons (meta, cloud, pa, water, [], erdpix), "uint8");
  endif
endfunction
