# slice_width.py
#
# Reads any DICOM images within the directory and analyses them with the available scripts
#
# Slice width
#
# Simon Shah
# 23rd Oct 2018


import pydicom
import numpy as np
from scipy import ndimage
import sys
import matplotlib.pyplot as plt

# Read DICOM image
# Read the DICOM file - not needed if called from parent script
image = pydicom.dcmread( 'ANNUALQA.MR.HEAD_GENERAL.tra.slice_width.IMA')
# Prepare image for processing
idata = image.pixel_array  # Read the pixel values into an array
idata = np.array(idata)  # Make it a numpy array
idown = ((idata / idata.max()) * 256).astype('uint8')  # Downscale to uint8 for openCV techniques


# threshold and binaries the image in order to locate the rods.
# this is achieved by masking the
img_max = np.max(np.array(idown))  # maximum number of img intensity
no_region = [None] * img_max

# smooth the image with a 0.5sig kernal - this is to avoid noise being counted in .label function
img_tmp = ndimage.gaussian_filter(np.array(idown), 0.5)

# step over a range of threshold levels from 0 to the max in the image
# using the ndimage.label function to count the features for each threshold
for x in range(0, img_max):
    tmp = img_tmp <= x
    labeled_array, num_features = ndimage.label(tmp.astype(np.int))
    no_region[x] = num_features
# find the indices that correspond to 10 regions and pick the median
index = [i for i,val in enumerate(no_region) if val==10]
thres_ind = np.median(index).astype(np.int)

# Generate the labeled array with the threshold choson
img_threshold = img_tmp <= thres_ind
labeled_array, num_features = ndimage.label(img_threshold.astype(np.int))
# check that we have got the 10 rods!
if num_features != 10:
    sys.exit("Did not find the 9 rods")


# find the indices of each of the rods and
rod_rows = [None]*9
rod_cols = [None]*9
for x in range(0,11):
    if np.count_nonzero(labeled_array == x) < 40:
        rowi, coli = np.where(labeled_array == x)
        rod_rows[x - 2] = np.mean(rowi).astype(np.float32)
        rod_cols[x - 2] = np.mean(coli).astype(np.float32)

print(rod_rows)
print(rod_cols)
# find the horizontal and vertical distances for the rods (top mid bot and left mid right
# the rod indices are ordered as:
# 3 2 1
# 6 5 4
# 9 8 7
# Slightly worried here that the order could go wrong...
horz_dist = [None]*3
vert_dist = [None]*3
horz_dist[0] = (((rod_rows[2] - rod_rows[0]) ** 2) + (rod_cols[2] - rod_cols[0]) ** 2) ** 0.5
horz_dist[1] = (((rod_rows[5] - rod_rows[3]) ** 2) + (rod_cols[5] - rod_cols[3]) ** 2) ** 0.5
horz_dist[2] = (((rod_rows[8] - rod_rows[6]) ** 2) + (rod_cols[8] - rod_cols[6]) ** 2) ** 0.5

vert_dist[0] = (((rod_rows[2] - rod_rows[8]) ** 2) + (rod_cols[2] - rod_cols[8]) ** 2) ** 0.5
vert_dist[1] = (((rod_rows[1] - rod_rows[7]) ** 2) + (rod_cols[1] - rod_cols[7]) ** 2) ** 0.5
vert_dist[2] = (((rod_rows[0] - rod_rows[6]) ** 2) + (rod_cols[0] - rod_cols[6]) ** 2) ** 0.5


## the distances into mm and the FOV
# could do with adding catches if array > 2 and not equal to what we expect!
getFOV = np.array(image[0x0018,0x1310].value)
FOV = np.where(getFOV)[0]
PixelSpace = (image.PixelSpacing)
FOVx = getFOV[FOV[0]] * PixelSpace[0]
FOVy = getFOV[FOV[1]] * PixelSpace[1]

# calculate the horizontal and vertical distances
horz_dist_mm = np.mean(np.multiply(PixelSpace[0], horz_dist))
vert_dist_mm = np.mean(np.multiply(PixelSpace[0], vert_dist))
# Calculate the distortion in the horizontal and vertical directions
horz_distortion = 100 * (np.std(np.multiply(PixelSpace[0], horz_dist))) / horz_dist_mm
vert_distortion = 100 * (np.std(np.multiply(PixelSpace[0], vert_dist))) / vert_dist_mm

# Calculate the correction coeff for the top and bottom profiles
# divided by 120mm - as this is the distance between the rods
Correction_Coeff = [None]*2
Correction_Coeff[0] = np.mean(np.multiply(PixelSpace[0], horz_dist[1:2]))/120 # Top profile
Correction_Coeff[1] = np.mean(np.multiply(PixelSpace[0], horz_dist[0:1]))/120 # Bottom profile

# Round the rod positions, so they can be assigned to a pixel
rod_rows_int = np.round(rod_rows).astype(int)
rod_cols_int = np.round(rod_cols).astype(int)

# Find the central y-axis point for the top and bottom profiles
# done by finding the distance between the mid-distances of the central rods
top_profile_centre = np.round(((rod_rows_int[4] - rod_rows_int[1]) / 2 ) + rod_rows_int[1]).astype(int)
bot_profile_centre = np.round(((rod_rows_int[7] - rod_rows_int[4]) / 2 ) + rod_rows_int[4]).astype(int)

# Selected 20mm around the mid-distances and take the average to find the line profiles
Top_Profile = idown[(top_profile_centre-10):(top_profile_centre+9),rod_cols_int[5]:rod_cols_int[3]]
Top_Profile_av = np.mean(Top_Profile, axis=0)

Bot_Profile = idown[(bot_profile_centre-10):(bot_profile_centre+9),rod_cols_int[5]:rod_cols_int[3]]
Bot_Profile_av = np.mean(Bot_Profile, axis=0)


# FOR NOW JUST DO ONE PROFILE AND THE CLEAN SCRIPT UP
# First fit a quadratic curve to outer edges (30 pixels each side)
length_profile = len(Top_Profile_av)
print(length_profile)
test = np.array(Top_Profile_av)
test2 = test[0:10:1, 100:110:1]

# Plot annotated image for user
fig = plt.figure(1)
plt.plot(Bot_Profile_av)
plt.show()
