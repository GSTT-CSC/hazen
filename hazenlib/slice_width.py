# slice_width.py
#
# Reads any DICOM images within the directory and analyses them with the available scripts
#
# Slice width
#
# Simon Shah
# 23rd Oct 2018

import os
from math import pi
import sys

import pydicom
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
from lmfit import Model


class Rod:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Rod: {self.x}, {self.y}'

    @property
    def centroid(self):
        return self.x, self.y

    def __lt__(self, other):
        """Using "reading order" in a coordinate system where 0,0 is bottom left"""
        try:
            x0, y0 = self.centroid
            x1, y1 = other.centroid
            return (-y0, x0) < (-y1, x1)
        except AttributeError:
            return NotImplemented

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def get_rods(dcm):
    fig, ax = plt.subplots(nrows=2, ncols=2)

    arr = dcm.pixel_array
    ax[0][0].imshow(arr, cmap='gray')
    # threshold and binaries the image in order to locate the rods.
    # this is achieved by masking the
    img_max = np.max(arr)  # maximum number of img intensity
    no_region = [None] * img_max

    # smooth the image with a 0.5sig kernal - this is to avoid noise being counted in .label function
    # img_tmp = ndimage.gaussian_filter(arr, 0.5)
    # commented out smoothing as not in original MATLAB - Haris
    img_tmp = arr
    # step over a range of threshold levels from 0 to the max in the image
    # using the ndimage.label function to count the features for each threshold
    for x in range(0, img_max):
        tmp = img_tmp <= x
        labeled_array, num_features = ndimage.label(tmp.astype(np.int))
        no_region[x] = num_features

    # find the indices that correspond to 10 regions and pick the median
    index = [i for i, val in enumerate(no_region) if val == 10]

    thres_ind = np.median(index).astype(np.int)

    # Generate the labeled array with the threshold chosen
    img_threshold = img_tmp <= thres_ind

    ax[0][1].imshow(img_threshold, cmap='gray')

    labeled_array, num_features = ndimage.label(img_threshold.astype(np.int))
    ax[1][0].imshow(labeled_array, cmap='gray')

    # check that we have got the 10 rods!
    if num_features != 10:
        sys.exit("Did not find the 9 rods")

    rods = ndimage.measurements.center_of_mass(arr, labeled_array, range(2, 11))

    rods = [Rod(x=x[1], y=x[0]) for x in rods]

    rods = sorted(rods)

    ax[1][1].imshow(arr, cmap='gray')

    mark = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    for idx, i in enumerate(rods):
        ax[1][1].scatter(x=i.x, y=i.y, marker=f"$ {mark[idx]} $", s=5, linewidths=0.5)
    # ax[1][1].scatter(x=[i.x for i in rods], y=[i.y for i in rods], marker="+", s=1, linewidths=0.5)
    fig.savefig('rods.png')
    return rods


def get_rod_distances(rods):
    """
    Calculate horizontal and vertical distances of rods

    Args:
        rods:

    Returns:

    """
    horz_dist = [None] * 3
    vert_dist = [None] * 3
    horz_dist[2] = (((rods[2].y - rods[0].y) ** 2) + (rods[2].x - rods[0].x) ** 2) ** 0.5
    horz_dist[1] = (((rods[5].y - rods[3].y) ** 2) + (rods[5].x - rods[3].x) ** 2) ** 0.5
    horz_dist[0] = (((rods[8].y - rods[6].y) ** 2) + (rods[8].x - rods[6].x) ** 2) ** 0.5

    vert_dist[0] = (((rods[2].y - rods[8].y) ** 2) + (rods[2].x - rods[8].x) ** 2) ** 0.5
    vert_dist[1] = (((rods[1].y - rods[7].y) ** 2) + (rods[1].x - rods[7].x) ** 2) ** 0.5
    vert_dist[2] = (((rods[0].y - rods[6].y) ** 2) + (rods[0].x - rods[6].x) ** 2) ** 0.5

    return horz_dist, vert_dist


def get_rod_distortions(rods, dcm):
    # find the horizontal and vertical distances for the rods (top mid bot and left mid right
    # the rod indices are ordered as:
    # 789
    # 456
    # 123
    pixel_spacing = dcm.PixelSpacing
    getFOV = np.array(dcm[0x0018, 0x1310].value)
    FOV = np.where(getFOV)[0]
    FOVx = getFOV[FOV[0]] * pixel_spacing[0]
    FOVy = getFOV[FOV[1]] * pixel_spacing[1]

    horz_dist, vert_dist = get_rod_distances(rods)

    # calculate the horizontal and vertical distances
    horz_dist_mm = np.mean(np.multiply(pixel_spacing[0], horz_dist))
    vert_dist_mm = np.mean(np.multiply(pixel_spacing[1], vert_dist))

    # Calculate the distortion in the horizontal and vertical directions
    horz_distortion = 100 * (np.std(np.multiply(pixel_spacing[0], horz_dist))) / horz_dist_mm
    vert_distortion = 100 * (np.std(np.multiply(pixel_spacing[0], vert_dist))) / vert_dist_mm

    return horz_distortion, vert_distortion


def baseline_correction(profile):
    """
    Calculates quadratic fit of the baseline and subtracts from profile

    Args:
        profile:

    Returns:

    """
    print(f"length of profile: {len(profile)}, shape: {profile.shape}")
    left = np.array(profile[:30])
    right = np.array(profile[-30:])
    outer_profile = np.concatenate([left, right])

    # create the x axis for the outer profile
    x_left = np.arange(30)
    x_right = np.arange(30) + 90
    x_outer = np.concatenate([x_left, x_right])

    # seconds order poly fit of the outer profile
    poly_fit = np.poly1d(np.polyfit(x_outer, outer_profile, 2))

    # use the poly fit to generate a quadratic curve with 0.25 space (high res)
    sample_spacing = 0.25
    x_interp = np.arange(0, 120, sample_spacing)
    x = np.arange(0, 120)
    baseline_interp = poly_fit(x_interp)
    plt.figure()
    plt.plot(baseline_interp)
    plt.title('Baseline ')
    baseline = poly_fit(x)

    plt.figure()
    plt.plot(profile, label='profile')
    plt.plot(baseline, label='baseline')
    plt.legend()
    plt.title('Baseline fitted')

    # Remove the baseline effects from the profiles
    profile_corrected = profile - baseline

    f = interp1d(x, profile_corrected, fill_value="extrapolate")
    plt.figure()

    profile_corrected_interp = f(x_interp)
    plt.plot(profile_corrected_interp)
    plt.title("profile_corrected_interp")
    profile_interp = profile_corrected_interp + baseline_interp

    return profile_interp


def trapezoid(x, base, start, stop, topleft, topright):
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

    # need to set some conditions on start and stop, top left and top right so they are the same length.
    stop = int(topright) + (int(topleft) - int(start))
    y[:int(start)] = base

    y[int(start):int(topleft)] = base + (a * z)

    y[int(topleft):int(topright)] = base + (a * z[-1])

    y[int(topright):int(stop)] = (base + (a * z[-1])) - (z * a)

    y[int(stop):] = base

    plt.figure()
    plt.plot(y)

    fwhm = (topright - topleft) + (topleft - start)

    print(fwhm)

    return y


def get_profile(arr, x, y, width, n):
    pass


def get_slice_width(dcm):
    """
    Calculates slice width using double wedge image

    Args:
        dcm:

    Returns:

    """

    slice_width = 0.0

    arr = dcm.pixel_array

    rods = get_rods(dcm)
    horz_dist, vert_dist = get_rod_distances(rods)

    correction_coeff = [None] * 2
    correction_coeff[0] = np.mean(np.multiply(dcm.PixelSpacing[0], horz_dist[1:2])) / 120  # Top profile
    correction_coeff[1] = np.mean(np.multiply(dcm.PixelSpacing[0], horz_dist[0:1])) / 120  # Bottom profile

    # Find the central y-axis point for the top and bottom profiles
    # done by finding the distance between the mid-distances of the central rods

    top_profile_centre = np.round(((rods[5].y - rods[8].y) / 2) + rods[8].y).astype(int)
    bottom_profile_centre = np.round(((rods[2].y - rods[5].y) / 2) + rods[5].y).astype(int)

    # Selected 20mm around the mid-distances and take the average to find the line profiles
    top_profile = arr[(top_profile_centre-10):(top_profile_centre+10), int(rods[3].x):int(rods[5].x)]

    plt.figure()
    plt.imshow(arr)

    mark = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    for idx, i in enumerate(rods):
        plt.scatter(x=i.x, y=i.y, marker=f"$ {mark[idx]} $", s=15, linewidths=0.5)

    for i, val in enumerate(top_profile):
        plt.plot(range(int(rods[3].x), int(rods[5].x)), [top_profile_centre-10+i]*120, '-', color='red')

    plt.figure()
    for i in top_profile:
        plt.plot(i)
    plt.plot(top_profile[1])

    top_profile_mean = np.mean(top_profile, axis=0)

    plt.figure()
    plt.plot(top_profile_mean)

    bottom_profile = arr[(bottom_profile_centre-10):(bottom_profile_centre+10), int(rods[0].x):int(rods[2].x)]
    bottom_profile_mean = np.mean(bottom_profile, axis=0)

    top_profile_mean_corrected = baseline_correction(profile=top_profile_mean)
    plt.figure()
    plt.plot(top_profile_mean_corrected)
    plt.title('top_profile_mean_corrected')

    bottom_profile_mean_corrected = baseline_correction(profile=bottom_profile_mean)
    plt.figure()
    plt.plot(bottom_profile_mean_corrected)
    plt.title('bottom_profile_mean_corrected')
    plt.show()

    return slice_width


def main(data: list) -> list:
    print(f"Measuring slice width from image: {data}")

    if len(data) != 1:
        raise Exception('Need one DICOM file only')

    dcm = pydicom.read_file(data[0])

    results = get_slice_width(dcm)

    return [results]


def simon_slice_width():
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
    top_profile_centre = np.round(((rod_rows_int[4] - rod_rows_int[1]) / 2) + rod_rows_int[1]).astype(int)
    bot_profile_centre = np.round(((rod_rows_int[7] - rod_rows_int[4]) / 2) + rod_rows_int[4]).astype(int)

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




        xdata = np.linspace(0, len(Profile_corrected_interp),len(Profile_corrected_interp))

        popt, pcov = curve_fit(traps_ss, xdata, abs(Profile_corrected_interp), p0=[0, 150, 300, 200, 250],
                               bounds=([-10., 50., 201., 60., 200.], [10., 200., 400., 280., 400.]),method='trf')


        print(popt)
        TrapzFit[ii][:] = popt

        fit = traps_ss(xdata, *popt)

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


if __name__ == "__main__":
    main([os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1])])