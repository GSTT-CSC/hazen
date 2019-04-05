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
from math import pi
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import sys
import matplotlib.pyplot as plt

# Read DICOM image
# Read the DICOM file - not needed if called from parent script
image = pydicom.dcmread('ANNUALQA.MR.HEAD_GENERAL.tra.slice_width.IMA')
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
# concat the top and bottom profiles
combined = np.row_stack((Top_Profile_av, Bot_Profile_av))
# test = [Top_Profile_av],[Bot_Profile_av]
# test = np.array(test).astype(float)
# print("wtf", combined)
# print(combined[:][0])
# print("Top ", Top_Profile_av)
TrapzFit = np.zeros((2,5))
for ii in range(0,2):
    Profile_tmp = combined[:][ii]
    # print(Profile_tmp)
    length_profile = len(Profile_tmp)
    left = np.array(Profile_tmp[0:30])
    right = np.array(Profile_tmp[90:])
    outer_profile = np.concatenate([left, right])

    # create the x axis for the outer profile
    x_left = np.arange(30)
    x_right = np.arange(30)+90
    x_outer = np.concatenate([x_left, x_right])

    # seconds order poly fit of the outer profile
    Pfit = np.poly1d(np.polyfit(x_outer, outer_profile, 2))

    # use the poly fit to generate a quadratric curve with 0.25 space (high res)
    sample_spacing = 0.25
    x_interp = np.arange(0, 120, sample_spacing)
    x = np.arange(0, 120)
    baseline_interp = Pfit(x_interp)
    baseline = Pfit(x)

    # Remove the baseline effects from the profiles
    Profile_corrected = Profile_tmp - baseline

    f = interp1d(x, Profile_corrected, fill_value="extrapolate")
    Profile_corrected_interp = f(x_interp)

    Profile_interp = Profile_corrected_interp + baseline_interp


    # Slice thinkness - need to get from dicom metadata
    def traps_ss(x, base, start, stop, topleft, topright):
        # n_ramp
        # n_plateau
        # n_left_baseline
        # n_right_baseline
        # plateau_amplitude

        y = np.zeros(len(x))

        # gradient from bottom left to top left
        a = np.float((topleft - start) / topleft)
        # range of values on the slope up
        z = np.arange(int(topleft) - int(start))
        # print("start = ", int(start),"Top left = ", int(topleft),"Top right =", int(topright),"stop = ", int(stop))
        # print("a =", a)
        # print("z length = ", len(z))
        # print("base = ", base)

        # need to set some conditions on start and stop, top left and top right so they are the same length.
        stop = int(topright) + (int(topleft) - int(start))
        y[:int(start)] = base

        y[int(start):int(topleft)] = base + (a * z)

        y[int(topleft):int(topright)] = base + (a * z[-1])

        y[int(topright):int(stop)] = (base + (a * z[-1])) - (z * a)

        y[int(stop):] = base

        fwhm = (topright - topleft) + (topleft-start)

        print(fwhm)

        return y



    xdata = np.linspace(0, len(Profile_corrected_interp),len(Profile_corrected_interp))

    popt, pcov = curve_fit(traps_ss, xdata, abs(Profile_corrected_interp), p0=[0, 150, 300, 200, 250],
                           bounds=([-10., 50., 201., 60., 200.], [10., 200., 400., 280., 400.]),method='trf')


    print(popt)
    TrapzFit[ii][:] = popt

    fit = traps_ss(xdata,*popt)

    fig = plt.figure(ii)
    plt.plot(fit)
    plt.plot(abs(Profile_corrected_interp))
    plt.show()



#
print(TrapzFit)
# # now to convert to slice width from fit.
fwhm = [None]*2
Slice_Width_mm = [None]*2
Slice_Width_mm_Geometry_Corrected = [None]*2
Slice_Width_mm_AAPM = [None]*2
Slice_Width_mm_AAPM_Corrected = [None]*2
for ii in range(0,2):

    fwhm[ii] = ((TrapzFit[ii][4]) - (TrapzFit[ii][3])) + ((TrapzFit[ii][3]) - (TrapzFit[ii][1]))
    print(fwhm)
    Slice_Width_mm[ii] = fwhm[ii] * sample_spacing * PixelSpace[0] * np.tan((11.3 * pi) / 180)
    Slice_Width_mm_Geometry_Corrected[ii] = Slice_Width_mm[ii] / Correction_Coeff[ii]
    Slice_Width_mm_AAPM[ii] = fwhm[ii] * sample_spacing * PixelSpace[0]
    Slice_Width_mm_AAPM_Corrected[ii] = (fwhm[ii] * sample_spacing * PixelSpace[0]) / Correction_Coeff[ii]

# %Geometric mean of slice widths (pg 34 of IPEM Report 80)
slicewidth_geo_mean_mm = (Slice_Width_mm[0] * Slice_Width_mm[1])**(0.5)
slicewidth_Geometry_Corrected_geo_mean_mm = (Slice_Width_mm_Geometry_Corrected[0] * Slice_Width_mm_Geometry_Corrected[1])**(0.5)
#
# %AAPM method directly incorporating phantom tilt
theta = (180.0 - 2.0 * 11.3) * pi/180.0
term1 = (np.cos(theta))**2.0 * (Slice_Width_mm_AAPM[1] - Slice_Width_mm_AAPM[0])**2.0 + (4.0 * Slice_Width_mm_AAPM[1] * Slice_Width_mm_AAPM[0])
term2 = (Slice_Width_mm_AAPM[1] + Slice_Width_mm_AAPM[0]) * np.cos(theta)
term3 = 2.0 * np.sin(theta)
#
slicewidth_mean_mm_alternative = ((term1**0.5) + term2)/term3
phantom_tilt = np.arctan(slicewidth_mean_mm_alternative/Slice_Width_mm_AAPM[1]) + (theta/2.0) - pi/2.0
phantom_tilt_deg = phantom_tilt * (180.0/pi)
#
phantom_tilt_check = -np.arctan(slicewidth_mean_mm_alternative/Slice_Width_mm_AAPM[0]) - (theta/2.0) + pi/2.0
phantom_tilt_check_deg = phantom_tilt_check * (180.0/pi)
#
# %AAPM method directly incorporating phantom tilt and independent of geometric linearity
theta = (180.0 - 2.0 * 11.3) * pi/180.0
term1 = (np.cos(theta))**2.0 * (Slice_Width_mm_AAPM_Corrected[1] - Slice_Width_mm_AAPM_Corrected[0])**2.0 + \
        (4.0 * Slice_Width_mm_AAPM_Corrected[1] * Slice_Width_mm_AAPM_Corrected[0])
term2 = (Slice_Width_mm_AAPM_Corrected[1] + Slice_Width_mm_AAPM_Corrected[0]) * np.cos(theta)
term3 = 2.0 * np.sin(theta)
#
slicewidth_mean_mm_geo_corr = ((term1**0.5) + term2)/term3
phantom_tilt_corr = np.arctan(slicewidth_mean_mm_geo_corr/Slice_Width_mm_AAPM_Corrected[1]) + (theta/2.0) - pi/2.0
phantom_tilt_deg_corr = phantom_tilt_corr * (180.0/pi)
#
phantom_tilt_check_corr = -np.arctan(slicewidth_mean_mm_geo_corr/Slice_Width_mm_AAPM_Corrected[0]) - (theta/2.0) + pi/2.0
phantom_tilt_check_deg_corr = phantom_tilt_check_corr * (180.0/pi)
##

f = open("test.txt", "w+")
f.write('Series Description: \s' + image.SeriesDescription + '\n')
f.write("Width: %d" % image.Rows + '\n')
f.write("Height: %d" % image.Columns + '\n')
f.write("Slice Thinkness (mm): %d" % image.SliceThickness + '\n')
f.write("Field of View (mm) : %d" % FOVx + '\n')
f.write("bandwidth (Hz/Px) : %d" % image.PixelBandwidth + '\n')
f.write("TR  (ms) : %d" % image.RepetitionTime + '\n')
f.write("TE  (ms) : %d" % image.EchoTime + '\n')
f.write("Flip Angle  (deg) : %d" % image.FlipAngle + '\n')


f.write("Horizontal line bottom (mm) %f" % horz_dist[0] + '\n')
f.write("Horizontal line middle (mm) %f" % horz_dist[1] + '\n')
f.write("Horizontal line top (mm) %f" % horz_dist[2] + '\n')
f.write("Horizontal Linearity (mm)\ %f" % horz_dist_mm + '\n')
f.write("Horizontal Distortion \ %f" % horz_distortion + '\n')

f.write("Vertical line bottom (mm) %f" % vert_dist[0] + '\n')
f.write("Vertical line middle (mm) %f" % vert_dist[1] + '\n')
f.write("Vertical line top (mm) %f" % vert_dist[2] + '\n')
f.write("Vertical Linearity (mm)\ %f" % vert_dist_mm + '\n')
f.write("Vertical Distortion \ %f" % vert_distortion + '\n')

f.write("Slice width top (mm) \ %f" % Slice_Width_mm[0] + '\n')
f.write("Slice width bottom (mm) \ %f" % Slice_Width_mm[1] + '\n')
f.write("Phantom tilt (deg) \ %f" % phantom_tilt_deg + '\n')
f.write("Slice width AAPM geometry corrected (mm) %f" % slicewidth_mean_mm_geo_corr + '\n')
f.close()

f = open("test2.txt", "w+")
f.write("%d" % image.Rows + '\n')
f.write("%d" % image.Columns + '\n')
f.write("%d" % image.SliceThickness + '\n')
f.write("%d" % FOVx + '\n')
f.write("%d" % image.PixelBandwidth + '\n')
f.write("%d" % image.RepetitionTime + '\n')
f.write("%d" % image.EchoTime + '\n')
f.write("%d" % image.FlipAngle + '\n')


f.write("%f" % horz_dist[0] + '\n')
f.write("%f" % horz_dist[1] + '\n')
f.write("%f" % horz_dist[2] + '\n')
f.write("%f" % horz_dist_mm + '\n')
f.write("%f" % horz_distortion + '\n')

f.write("%f" % vert_dist[0] + '\n')
f.write("%f" % vert_dist[1] + '\n')
f.write("%f" % vert_dist[2] + '\n')
f.write("%f" % vert_dist_mm + '\n')
f.write("%f" % vert_distortion + '\n')

f.write("%f" % Slice_Width_mm[0] + '\n')
f.write("%f" % Slice_Width_mm[1] + '\n')
f.write("%f" % phantom_tilt_deg + '\n')
f.write("%f" % slicewidth_mean_mm_geo_corr + '\n')
f.close()