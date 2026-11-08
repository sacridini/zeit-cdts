## The Octave side of benchmarks/vs_originals/fmask4.py: runs the MATLAB primitives on the cases in FMASK_DIR and
## writes the results. Each line of cases.txt is "op id rows cols args..."; the input is <id>.in (raw, row-major:
## float32, or uint8 for the binary ops, or float64 for round), the element of the morphology <id>.mask (uint8).
## The outputs are <id>.out (and <id>.out2, <id>.out3 for the ops with several). The folder of the shims (the shim
## of imresize, fmask_disk, and the copy of the @strel class patched to accept strel ("disk", r)) is FMASK_SHIMS.
pkg load image statistics
global FMASK_ORIG_IMRESIZE
FMASK_ORIG_IMRESIZE = @imresize;
if (! isempty (getenv ("FMASK_SRC")))
  addpath (getenv ("FMASK_SRC"));         # the original .m files of Fmask 4.7, as they are
endif
addpath (fileparts (mfilename ("fullpath")));   # fm_scene.m, fm_pc_direct.m
addpath (getenv ("FMASK_SHIMS"));         # first in the path
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

function se = make_se (dir, id, kind, a, b)
  switch (kind)
    case "disk"
      se = strel ("disk", a);
    case "square"
      se = strel ("square", a);
    case "arb"
      se = strel ("arbitrary", logical (rd ([dir "/" id ".mask"], a, b, "uint8")));
  endswitch
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
    case "prctile"
      x = rd ([base ".in"], rows, cols, "single");
      p = cellfun (@str2double, args);
      r = zeros (1, numel (p), "single");
      for i = 1:numel (p)
        r(i) = prctile (x(:), p(i));
      endfor
      wr ([base ".out"], r, "single");
    case "resize_half"
      x = rd ([base ".in"], rows, cols, "single");
      wr ([base ".out"], imresize (x, 0.5, "box"), "single");
    case "resize_native_box"
      ## Octave's own, without the shim, to report its error
      x = rd ([base ".in"], rows, cols, "single");
      wr ([base ".out"], FMASK_ORIG_IMRESIZE (x, 0.5, "box"), "single");
    case "rep3"
      x = rd ([base ".in"], rows, cols, "single");
      wr ([base ".out"], imresize (x, 3, "nearest"), "single");
    case "stdfilt"
      x = rd ([base ".in"], rows, cols, "single");
      k = str2double (args{1});
      s = stdfilt (x, true (k));
      wr ([base ".out"], s, "double");
      wr ([base ".out2"], strcmp (class (s), "double"), "uint8");
    case "morph_bin"
      x = logical (rd ([base ".in"], rows, cols, "uint8"));
      se = make_se (dir, id, args{2}, str2double (args{3}), str2double (args{4}));
      if (strcmp (args{1}, "erode"))
        r = imerode (x, se);
      else
        r = imdilate (x, se);
      endif
      wr ([base ".out"], r, "uint8");
    case "morph_gray"
      x = rd ([base ".in"], rows, cols, "single");
      se = make_se (dir, id, args{2}, str2double (args{3}), str2double (args{4}));
      if (strcmp (args{1}, "erode"))
        r = imerode (x, se);
      else
        r = imdilate (x, se);
      endif
      wr ([base ".out"], r, "single");
    case "strel_disk"
      r = str2double (args{1});
      wr ([base ".out"], fmask_disk (r), "uint8");
      s = strel ("disk", r);
      wr ([base ".out2"], getnhood (s), "uint8");
    case "bwlabel"
      x = logical (rd ([base ".in"], rows, cols, "uint8"));
      [l, n] = bwlabeln (x, 8);
      wr ([base ".out"], l, "int32");
      [l2, n2] = bwlabel (x, 8);
      wr ([base ".out2"], l2, "int32");
      wr ([base ".out3"], [n n2], "int32");
      s = regionprops (l, "Area", "Centroid", "PixelList");
      area = [s.Area];
      cen = vertcat (s.Centroid);
      pix = vertcat (s.PixelList);
      wr ([base ".area"], area(:)', "double");
      wr ([base ".cen"], cen, "double");
      wr ([base ".pix"], pix, "double");
    case "bwareaopen"
      x = logical (rd ([base ".in"], rows, cols, "uint8"));
      wr ([base ".out"], bwareaopen (x, str2double (args{1}), 8), "uint8");
    case "imfill"
      x = rd ([base ".in"], rows, cols, "single");
      wr ([base ".out"], imfill (x), "single");
      wr ([base ".out2"], imfill (x, 4), "single");
      wr ([base ".out3"], imfill (x, 8), "single");
    case "imfilter"
      x = rd ([base ".in"], rows, cols, "single");
      tt = cellfun (@str2double, args);
      tm = reshape (tt, 3, 3)';
      wr ([base ".out"], imfilter (x, single (tm)), "single");
      wr ([base ".out2"], imfilter (x, tm), "single");
    case "fm_scene"
      fm_scene (dir, id, rows, cols, str2double (args{1}), str2double (args{2}));
    case "fm_pc_direct"
      fm_pc_direct (dir, id, rows, cols);
    case "round"
      x = rd ([base ".in"], rows, cols, "double");
      wr ([base ".out"], round (x), "double");
  endswitch
  catch err
    fid2 = fopen ([base ".err"], "w"); fputs (fid2, [err.message " @ " err.stack(1).name ":" num2str(err.stack(1).line)]); fclose (fid2);
    printf ("ERROR %s: %s\n", id, err.message);
  end_try_catch
  printf ("%s %s %.3f s\n", op, id, toc);
  fflush (stdout);
endwhile
fclose (fid);
