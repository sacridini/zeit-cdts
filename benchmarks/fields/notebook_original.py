# Source: "MappingDemo 2 (1).html", a Jupyter notebook exported to HTML
# ("Mapping field boundaries in Landsat TM imagery", Version I, August 2020),
# by Baumann; inspired by Emmanuelle Gouillart's segmentation tutorial
# (http://emmanuelle.github.io/a-tutorial-on-segmentation.html).
# The author authorised porting the code directly.
#
# The code cells below are extracted verbatim, in order (the markdown cells
# are kept as comments). This file is NOT meant to run: it needs gdal,
# baumiTools and the author's G: drive files, cell 1 contains the IPython magic
# `%matplotlib inline` (a syntax error in plain Python), and cells use
# `np.int` (removed in numpy 1.24). See notebook_reference.py for the
# runnable, parameterised version.

# Mapping field boundaries in Landsat TM imagery
# Inspiration from: http://emmanuelle.github.io/a-tutorial-on-segmentation.html
# Version I (August 2020)
# Preparation
# Load the needed libraries
# Define the paths, load images into arrays
# Load properties of image

# %% [cell 1]
import gdal
import skimage
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %% [cell 2]
path = "G:/Baumann/_ANALYSES/FieldSize/"
img2000 = gdal.Open(path + "tileID_305_B5-medoid_2000.tif")
img2019 = gdal.Open(path + "tileID_305_B5-medoid_2019.tif")

# %% [cell 3]
gt = img2000.GetGeoTransform()
pr = img2000.GetProjection()

# Load images into arrays
# Define some subset boundaries based on ULx, ULy, cols, rows
# Load the arrays

# %% [cell 4]
cols = 500#img2000.RasterXSize#500
rows = 500#img2000.RasterYSize#500
UL_x = 0
UL_y = 0
sub_gt = [(gt[0]+UL_x*gt[1]), gt[1], 0, (gt[3]-UL_y*gt[1]), 0, gt[5]]

# %% [cell 5]
arr2000 = img2000.GetRasterBand(1).ReadAsArray(UL_x, UL_y, cols, rows)
arr2019 = img2019.GetRasterBand(1).ReadAsArray(UL_x, UL_y, cols, rows)

# %% [cell 6]
plt.rcParams['figure.figsize'] = [10, 10]
f, axarr = plt.subplots(1,2)
axarr[0].imshow(arr2000, cmap='gray')
axarr[1].imshow(arr2019, cmap='gray')

# Run the segmentation
# Step I - Variation 1 : Denoise the image --> DON't DO IT
# Use the non-local means filter, that averages a pixel with other pixels under the condition that patches centered on the two pixels look similar.
# This filter has the property to preserve textures well: here, it is important to preserve well the weak edges between grains, while smoothing significantly the interior of the grains.

# %% [cell 7]
from skimage import restoration
from skimage import img_as_float
arr2000_denoised = restoration.denoise_nl_means(img_as_float(arr2000), patch_size = 2, patch_distance = 2, h=0.01)
arr2019_denoised = restoration.denoise_nl_means(img_as_float(arr2019), patch_size = 2, patch_distance = 2, h=0.01)
#
# Plot them next to each other
plt.rcParams['figure.figsize'] = [10, 10]
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(arr2000, cmap='gray')
axarr[0,1].imshow(arr2019, cmap='gray')
axarr[1,0].imshow(arr2000_denoised, cmap='gray')
axarr[1,1].imshow(arr2019_denoised, cmap='gray')

# Step I - Variation 2: Stretch the histogram of the image

# %% [cell 8]
def StretchHisto(arr):
    from skimage.exposure import rescale_intensity
    p2, p98 = np.percentile(arr, (2, 98))
    arr = rescale_intensity(arr, in_range=(p2, p98), out_range=(0, 1))
    return arr
arr2000_stretch = StretchHisto(arr2000)
arr2019_stretch = StretchHisto(arr2019)
# Plot them next to each other
plt.rcParams['figure.figsize'] = [10, 10]
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(arr2000, cmap='gray')
axarr[0,1].imshow(arr2019, cmap='gray')
axarr[1,0].imshow(arr2000_stretch, cmap='gray')
axarr[1,1].imshow(arr2019_stretch, cmap='gray')

# Step II: Create a gradient image
# Make use of the darker gray levels of boundaries and use a morphological operation called black tophat, that detects structures that are darker than their surroundings, and smaller than a structuring element.
# In order to enhance edges, use a linear combination of the black tophat and the denoised image.

# %% [cell 9]
from scipy import ndimage
from skimage import morphology
# Black tophat transformation (see https://en.wikipedia.org/wiki/Top-hat_transform)
hat_2000 = ndimage.black_tophat(arr2000_stretch, 3)
hat_2019 = ndimage.black_tophat(arr2019_stretch, 3)
# Combine with denoised image
hat_2000_lc = hat_2000 - 0.5 * arr2000_stretch
hat_2019_lc = hat_2019 - 0.5 * arr2019_stretch
# Morphological dilation to try to remove some holes in hat image
hat_2000_morph = morphology.dilation(hat_2000_lc)
hat_2019_morph = morphology.dilation(hat_2019_lc)
#
# Plot them in the series to see the changes
plt.rcParams['figure.figsize'] = [16, 10]
f, axarr = plt.subplots(2,4)
axarr[0,0].imshow(arr2000_stretch, cmap='gray')
axarr[0,1].imshow(hat_2000, cmap='gray')
axarr[0,2].imshow(hat_2000_lc, cmap='gray')
axarr[0,3].imshow(hat_2000_morph, cmap='gray')

axarr[1,0].imshow(arr2019_stretch, cmap='gray')
axarr[1,1].imshow(hat_2019, cmap='gray')
axarr[1,2].imshow(hat_2019_lc, cmap='gray')
axarr[1,3].imshow(hat_2019_morph, cmap='gray')

# Step III: Build a function for automatic point sampling and thinning
# A possible way to determine markers is to draw random pixels of the image. The underlying idea of using random markers is that there are some "natural boundaries" inside the image, corresponding to the grain boundaries that are looked for, and that a segmentation using random markers will have a fraction of its boundaries lying on such natural boundaries.
# By averaging on a large number of segmentations corresponding to different realizations of the markers, one hopes that pixels lying on the looked-for boundaries will lie on segmentation boundaries more often than other pixels.
# We know that the objects to segment have a minimal size (e.g., field size), markers that are too close to each other lead to boundaries that are seldom found on grain boundaries.
# Impose that random markers should have a minimal distance between each other, using scipy.spatial.KDTree to compute pairs of markers closer than a certain distance. We will also use the coloring trick to reduce the number of labels.

# %% [cell 10]
from scipy import spatial
def trim_close_points(points, distance):#=10):
    """
    Greedy method to remove some points so that
    all points are separated by a distance greater
    than ``distance``.
    
    points : array of shape (2, n_points)
        Coordinates of 2-D points
        
    distance : float
        Minimal distance between points
    """
    x, y = points
    tree = spatial.KDTree(np.array([x, y]).T)
    pairs = tree.query_pairs(distance)
    remove_indices = []
    for pair in pairs:
        if pair[0] in remove_indices:
            continue
        if pair[1] in remove_indices:
            continue
        else:
            remove_indices.append(pair[1])
    keep_indices = np.setdiff1d(np.arange(len(x)), remove_indices)
    return np.array([x[keep_indices], y[keep_indices]])

# %% [cell 11]
# Check result on simple example
x, y = np.random.random((2, cols))
xk, yk = trim_close_points((x, y), 0.1)
plt.rcParams['figure.figsize'] = [5, 3]
plt.plot(x, y, 'o')
plt.plot(xk, yk, 'or')

# Step  IV: run a large number of segmentations using random markers
# For speed purposes, we use watershed segmentation. We can vary it by using Random Walker
# Is an EPP, next step: parallelize the loop

# %% [cell 12]
from time import time
from sklearn.neighbors import kneighbors_graph
from pyamg.graph import vertex_coloring
from skimage import segmentation
n_real = 500
n_markers = 1000
segs_2000_ws = []
segs_2019_ws = []
segs_2000_rw = []
segs_2019_rw = []
for real in range(n_real):
    # Random markers
    x, y = np.random.random((2, n_markers))
    x *= hat_2000_morph.shape[0]
    y *= hat_2000_morph.shape[1]
    # Remove points too close to each other
    xk, yk = trim_close_points((x, y), 20)
    mat = kneighbors_graph(np.array([xk, yk]).T, 20)
    colors = vertex_coloring(mat)
    # Array of markers
    markers_rw = np.zeros(hat_2000_morph.shape, dtype=np.int)
    markers_rw[xk.astype(np.int), yk.astype(np.int)] = colors + 1
    markers_rw = morphology.dilation(markers_rw, morphology.disk(3))
    # Segmentation using watershed
    seg2000 = segmentation.watershed(hat_2000_morph, markers_rw, mask = np.where(arr2000 <= 2000, 0, 1))
    segs_2000_ws.append(seg2000)
    seg2019 = segmentation.watershed(hat_2019_morph, markers_rw, mask = np.where(arr2019 <= 2000, 0, 1))
    segs_2019_ws.append(seg2019)
    # Segmentation using random walker
    #seg2000 = segmentation.random_walker(hat_2000_morph, markers_rw, beta=25000, mode='cg_mg')
    #segs_2000_rw.append(seg2000)
    #seg2019 = segmentation.random_walker(hat_2019_morph, markers_rw, beta=25000, mode='cg_mg')
    #segs_2019_rw.append(seg2019)

# Step V:  Count the segmentations on top of each other

# %% [cell 13]
segs_2000 = np.array(segs_2000_ws)
boundaries_2000 = np.zeros_like(hat_2000_morph)
for seg in segs_2000_ws:
    boundaries_2000 += segmentation.find_boundaries(seg, connectivity=4)
segs_2019 = np.array(segs_2019_ws)
boundaries_2019 = np.zeros_like(hat_2019_morph)
for seg in segs_2019_ws:
    boundaries_2019 += segmentation.find_boundaries(seg, connectivity=4)

# %% [cell 14]
plt.rcParams['figure.figsize'] = [16, 11]
f, axarr = plt.subplots(2,3)
orig = axarr[0,0].imshow(arr2000, cmap='gray')
morph = axarr[0,1].imshow(hat_2000_morph, cmap='gray')
bound = axarr[0,2].imshow(boundaries_2000, cmap='gist_heat')
f.colorbar(bound, ax=axarr[0,2])

orig = axarr[1,0].imshow(arr2019, cmap='gray')
morph = axarr[1,1].imshow(hat_2019_morph, cmap='gray')
bound = axarr[1,2].imshow(boundaries_2019, cmap='gist_heat')
f.colorbar(bound, ax=axarr[1,2])

# Step VI: Apply some thresholding to extract the boundaries
# Use here first the implemented function in skimage (https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html)

# %% [cell 15]
from skimage import measure, color
def ApplyHyst_and_relabel(boundArr, th_low, th_high):
    from skimage import filters
    low = th_low*n_real
    high = th_high*n_real
    # Apply hysteresis
    hyst = filters.apply_hysteresis_threshold(boundArr, low, high)
    regions = measure.label(np.logical_not(hyst), background=0, connectivity=1)
    # Remove small objets and relabel
    regions_clean = morphology.remove_small_objects(regions + 1, min_size=50)
    regions_clean, _, _ = segmentation.relabel_sequential(regions_clean)
    regions_clean = np.random.permutation(regions_clean.max() + 1)[regions_clean]
    
    return hyst, regions_clean

# %% [cell 16]
hyst2000, regions2000 = ApplyHyst_and_relabel(boundaries_2000, 0.3, 0.6)
hyst2019, regions2019 = ApplyHyst_and_relabel(boundaries_2019, 0.3, 0.6)


plt.rcParams['figure.figsize'] = [16, 11]
f, axarr = plt.subplots(2,3)
orig = axarr[0,0].imshow(arr2000, cmap='gray')
h = axarr[0,1].imshow(hyst2000, cmap='gist_heat')
segs = axarr[0,2].imshow(regions2000, cmap='CMRmap_r')#cmap='tab20c'

orig = axarr[1,0].imshow(arr2019, cmap='gray')
h = axarr[1,1].imshow(hyst2019, cmap='gist_heat')
segs = axarr[1,2].imshow(regions2019, cmap='CMRmap_r')#cmap='tab20c'

# Step VII: Clean up the segmentation by having the watershed segmentation run over the original image
# Finish the segmentation and  label pixels that have been labeled as boundaries (some of them being "real boundaries", other not) using the random walker to attribute the remaining unknown pixels.
# Roughly also mask out areas of remaining forest (simple threshold of 2000)

# %% [cell 17]
#final_2000 = segmentation.random_walker(arr2000, regions2000, beta=100, mode='cg_j')
final_2000 = segmentation.watershed(arr2000, regions2000)#, mask = np.where(arr2000 <= 2000, 0, 1))
final_2000 = np.random.permutation(final_2000.max() + 1)[final_2000]
final_2000 = np.where(arr2000 <= 2000, 0, final_2000)

#final_2019 = segmentation.random_walker(arr2019, regions2019, beta=100, mode='cg_j')
final_2019 = segmentation.watershed(arr2019, regions2019)#, mask = np.where(arr2019 <= 2000, 0, 1))
final_2019 = np.random.permutation(final_2019.max() + 1)[final_2019]
final_2019 = np.where(arr2019 <= 2000, 0, final_2019)

# %% [cell 18]
plt.rcParams['figure.figsize'] = [16, 11]
f, axarr = plt.subplots(2,2)
orig = axarr[0,0].imshow(arr2000, cmap='gray')
segs = axarr[0,1].imshow(final_2000, cmap='CMRmap_r')#cmap='tab20c'

orig = axarr[1,0].imshow(arr2019, cmap='gray')
segs = axarr[1,1].imshow(final_2019, cmap='CMRmap_r')#cmap='tab20c'

# %% [cell 19]
import baumiTools as bt
from gdal import gdalconst
# Safe them to disc in order to see what they look like in GIS
def CopyTestRasterToDisk(producedArray, cols, rows, gt, pr, type, outName):
    drvMemR = gdal.GetDriverByName('MEM')
    outRas = drvMemR.Create('', cols, rows, 1, type)
    outRas.SetProjection(pr)
    outRas.SetGeoTransform(gt)
    outRas.GetRasterBand(1).WriteArray(producedArray, 0, 0)
    bt.baumiRT.CopyMEMtoDisk(outRas, outName)
CopyTestRasterToDisk(final_2000, cols, rows, sub_gt, pr, gdalconst.GDT_UInt32, path + "y2000_segmented.tif")
CopyTestRasterToDisk(final_2019, cols, rows, sub_gt, pr, gdalconst.GDT_UInt32, path + "y2019_segmented.tif")

# %% [cell 20]
 

