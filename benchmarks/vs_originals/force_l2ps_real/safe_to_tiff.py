"""Copies a Sentinel-2 L1C .SAFE with the 13 band files (JPEG 2000) turned into GeoTIFFs that keep the name ``*_Bxx.jp2``.

Why: the GDAL of the conda env ``force-ref`` (3.9.3, where force-l2ps runs) has no JPEG 2000 driver and cannot be given one
without changing the env that built the reference. GDAL opens a file by its content, not its extension, so FORCE finds the
``_Bxx.jp2`` files it looks for and reads them with its own code. The data are the decoded digital numbers (uint16, the same
values for both sides, so that a difference between decoders cannot be taken for a difference of the correction).

    python safe_to_tiff.py SAFE_DIR OUT_DIR
"""
import shutil
import sys
import xml.etree.ElementTree  # noqa: F401  (before rasterio: DLL clash on Windows)
from pathlib import Path

import rasterio

src, out = Path(sys.argv[1]), Path(sys.argv[2])
dst = out / src.name
for p in sorted(src.rglob("*")):
    rel = p.relative_to(src)
    if p.is_dir():
        (dst / rel).mkdir(parents=True, exist_ok=True)
        continue
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".jp2" and p.parent.name == "IMG_DATA":
        if (dst / rel).exists():
            continue
        with rasterio.open(p) as d:
            prof = d.profile
            prof.update(driver="GTiff", compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512)
            prof.pop("nbits", None)
            with rasterio.open(dst / rel, "w", **prof) as o:
                o.write(d.read())
        print("converted", rel, flush=True)
    elif p.suffix == ".jp2":
        continue  # the masks of QI_DATA are not read
    else:
        shutil.copy2(p, dst / rel)
for d in ("AUX_DATA", "HTML"):
    (dst / d).mkdir(exist_ok=True)
