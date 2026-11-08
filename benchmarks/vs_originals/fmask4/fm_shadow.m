## The Octave side of benchmarks/vs_originals/fmask4_shadow.py: runs the original .m files of the cloud shadow step of
## Fmask 4.7 (DetectPotentialCloudShadow, stratiedSampleHanlder, ProjectDEM2Plane, getRealCloudPositionS2,
## MatchCloudShadow, BufferMasks; FMASK_SRC is on the path and none is edited) on the cases of FMASK_DIR/cases.txt,
## with the shims of FMASK_SHIMS in front (bwlabeln renumbered by columns, rng and randperm of the common draw), and
## writes the results. Each line of cases.txt is "op id rows cols args...", the inputs are <id>.* (raw, row-major;
## see fm_shadow_scene.m for the scenes), the outputs <id>.out*.
pkg load image statistics
global FMASK_ORIG_BWLABELN FMASK_RNG_HI FMASK_RNG_LO
FMASK_ORIG_BWLABELN = @bwlabeln;
FMASK_RNG_HI = uint64 (0);
FMASK_RNG_LO = uint64 (1);
addpath (getenv ("FMASK_SRC"));            # the original .m files of Fmask 4.7, as they are
addpath (fileparts (mfilename ("fullpath")));   # fm_shadow_scene.m and the recorders
addpath (getenv ("FMASK_SHIMS"));          # first in the path (the shims, and the instrumented copies _rec)
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

fid = fopen ([dir "/cases.txt"], "r");
while (true)
  ln = fgetl (fid);
  if (! ischar (ln)) break; endif
  t = strsplit (strtrim (ln), " ");
  op = t{1}; id = t{2}; rows = str2double (t{3}); cols = str2double (t{4});
  args = t(5:end);
  base = [dir "/" id];
  tic;
  try
  switch (op)
    case "range"
      ## args: class b i l; the pattern of the .m, a single base and limit with a double step
      b = str2double (args{2}); i = str2double (args{3}); l = str2double (args{4});
      if (strcmp (args{1}, "single"))
        r = single (b):i:single (l);
        wr ([base ".out"], r(:), "single");
        wr ([base ".out2"], strcmp (class (r), "single"), "uint8");
      else
        r = b:i:l;
        wr ([base ".out"], r(:), "double");
      endif
    case "randperm"
      n = str2double (args{1}); k = str2double (args{2});
      rng (1);
      p = randperm (n);
      wr ([base ".out"], p(1:k), "double");
    case "strat"
      x = rd ([base ".in"], rows, 1, "single");
      ids = stratiedSampleHanlder (x, single (str2double (args{1})), single (str2double (args{2})), [rows 1], ...
                                   str2double (args{3}), str2double (args{4}), 0, 20);
      wr ([base ".out"], ids, "double");
    case "polyfit"
      x = rd ([base ".in"], rows, 1, "double");
      y = rd ([base ".in2"], rows, 1, "double");
      wr ([base ".out"], polyfit (x, y, 1), "double");
    case "realpos"
      x = rd ([base ".in"], rows, 1, "double");
      y = rd ([base ".in2"], rows, 1, "double");
      h = str2double (args{1});
      if (strcmp (args{2}, "single")) h = single (h); endif
      [xn, yn] = getRealCloudPositionS2 (x, y, h, single (str2double (args{3})), single (str2double (args{4})), [20 20]);
      wr ([base ".out"], xn, "single");
      wr ([base ".out2"], yn, "single");
      wr ([base ".out3"], [strcmp(class(xn),"single") strcmp(class(yn),"single")], "uint8");
    case "plane"
      ## args: hasdem dem_base elev tazi azi dim_expd
      dem = [];
      if (str2double (args{1}))
        dem = rd ([base ".dem"], rows, cols, "single");
      endif
      [pr, pc] = ProjectDEM2Plane ([rows cols], [], dem, str2double (args{2}), str2double (args{3}), str2double (args{4}), ...
                                   str2double (args{5}), str2double (args{6}), [20 20]);
      wr ([base ".out"], pr, "uint16");
      wr ([base ".out2"], pc, "uint16");
    case "buffer"
      c = rd ([base ".in"], rows, cols, "uint8");
      s = rd ([base ".in2"], rows, cols, "uint8");
      n = rd ([base ".in3"], rows, cols, "uint8");
      [bc, bs, bn] = BufferMasks (c, str2double (args{1}), s, str2double (args{2}), n, str2double (args{3}));
      wr ([base ".out"], bc, "uint8");
      wr ([base ".out2"], bs, "uint8");
      wr ([base ".out3"], bn, "uint8");
    case "knn"
      x = rd ([base ".in"], rows, 2, "double");
      q = [str2double(args{1}) str2double(args{2})];
      [idx, d] = knnsearch (x, q, "k", str2double (args{3}), "distance", "cityblock");
      wr ([base ".out"], idx, "double");
      wr ([base ".out2"], d, "double");
    case "shadow"
      fm_shadow_scene (dir, id, rows, cols, args);
  endswitch
  catch err
    fid2 = fopen ([base ".err"], "w"); fputs (fid2, [err.message " @ " err.stack(1).name ":" num2str(err.stack(1).line)]); fclose (fid2);
    printf ("ERROR %s: %s @ %s:%d\n", id, err.message, err.stack(1).name, err.stack(1).line);
  end_try_catch
  printf ("%s %s %.3f s\n", op, id, toc);
  fflush (stdout);
endwhile
fclose (fid);
