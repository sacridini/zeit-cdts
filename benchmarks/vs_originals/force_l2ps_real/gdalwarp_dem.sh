#!/bin/bash
# The DEM on the grid of the scene as GDAL's warp makes it (what FORCE's warp_from_disc_to_known_brick does:
# GDALWarpOperation, bilinear, exact transformer (error threshold 0), the no-data value on both sides), with the gdalwarp of
# the conda env force-ref (GDAL 3.9.3), to compare with Zeit's DEM.
#   wsl -e bash -lc 'bash .../gdalwarp_dem.sh DEM.tif OUT.tif ulx lry lrx uly epsg'
set -eu
source ~/force-ref-work/env.sh
export OMP_NUM_THREADS=8 GDAL_NUM_THREADS=1
gdalwarp -overwrite -t_srs EPSG:$7 -te $3 $4 $5 $6 -tr 10 10 -r bilinear -et 0 -srcnodata -32767 -dstnodata -32767 \
  -ot Float32 -wo INIT_DEST=-32767 -co COMPRESS=DEFLATE -co TILED=YES "$1" "$2"
