# mrqa_functions
#
# This file contains python modules required to perform a range of MRI QA functions.
# Importing this file is a requirement of the Daily_MR_QA.py script
#
#
# Last updated 18/05/2018
#
# Neil Heraghty
# neil.heraghty@nhs.net
#

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

def snr(image):
    """
    Calculates the SNR for a single-slice image of a uniform MRI phantom

    This script utilises the smoothed subtraction method described in McCann 2013:
    A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790

    parameters:
    ---------------
    image: dicom image read using dcmread

    returns:
    ---------------
    mean_snr    :   calculated mean snr for the selected image
    """

    # Prepare image for processing
    idata = image.pixel_array  # Read the pixel values into an array
    idata = np.array(idata)  # Make it a numpy array
    idown = ((idata / idata.max()) * 256).astype('uint8')  # Downscale to uint8 for openCV techniques

    # Find phantom centre and radius
    (cenx, ceny, cradius) = find_circle(idown)

    # Create noise image
    imnoise = image_noise(idata)

    # Measure signal and noise within 5 20x20 pixel ROIs (central and +-40pixels in x & y)
    sig = [None] * 5
    noise = [None] * 5

    sig[0] = np.mean(idata[(cenx - 10):(cenx + 10), (ceny - 10):(ceny + 10)])
    sig[1] = np.mean(idata[(cenx - 50):(cenx - 30), (ceny - 50):(ceny - 30)])
    sig[2] = np.mean(idata[(cenx + 30):(cenx + 50), (ceny - 50):(ceny - 30)])
    sig[3] = np.mean(idata[(cenx - 50):(cenx - 10), (ceny + 30):(ceny + 50)])
    sig[4] = np.mean(idata[(cenx + 30):(cenx + 50), (ceny + 30):(ceny + 50)])

    noise[0] = np.std(imnoise[(cenx - 10):(cenx + 10), (ceny - 10):(ceny + 10)])
    noise[1] = np.std(imnoise[(cenx - 50):(cenx - 30), (ceny - 50):(ceny - 30)])
    noise[2] = np.std(imnoise[(cenx + 30):(cenx + 50), (ceny - 50):(ceny - 30)])
    noise[3] = np.std(imnoise[(cenx - 50):(cenx - 10), (ceny + 30):(ceny + 50)])
    noise[4] = np.std(imnoise[(cenx + 30):(cenx + 50), (ceny + 30):(ceny + 50)])

    # Draw regions for testing
    cv.rectangle(idown, ((cenx - 10), (ceny - 10)), ((cenx + 10), (ceny + 10)), 128, 2)
    cv.rectangle(idown, ((cenx - 50), (ceny - 50)), ((cenx - 30), (ceny - 30)), 128, 2)
    cv.rectangle(idown, ((cenx + 30), (ceny - 50)), ((cenx + 50), (ceny - 30)), 128, 2)
    cv.rectangle(idown, ((cenx - 50), (ceny + 30)), ((cenx - 30), (ceny + 50)), 128, 2)
    cv.rectangle(idown, ((cenx + 30), (ceny + 30)), ((cenx + 50), (ceny + 50)), 128, 2)

    # Plot annotated image for user
    fig = plt.figure(1)
    plt.imshow(idown, cmap='gray')
    plt.show()

    # Calculate SNR for each ROI and average
    snr = np.divide(sig, noise)
    mean_snr = np.mean(snr)

    # print("Measured SNR: ",int(round(mean_snr)))

    return mean_snr


def uniformity(image):
    """
    Calculates uniformity for a single-slice image of a uniform MRI phantom
    This script implements the IPEM/MAGNET method of measuring fractional uniformity.
    It also calculates integral uniformity using a 75% area FOV ROI and CoV for the same ROI.

    This script also measures Ghosting within a single image of a uniform phantom.
    This follows the guidance from ACR for testing their large phantom.

    A simple measurement of distortion is also made by comparing the height and width of the circular phantom.

    parameters:
    ---------------
    image: dicom image read using dcmread

    returns:
    ---------------
    fract_uniformity_hor :  fractional uniformity in horizontal direction
    fract_uniformity_ver :  fractional uniformity in vertical direction
    intuniform           :  integral uniformity (%) using central 75% of phantom
    cov                  :  coefficient of variation (%) using central 75% of phantom
    ghosting             :  percentage ghosting according to ACR method
    idistort             :  distortion measured as percentage difference in phantom width/height
    """

    # Prepare image for processing
    idata = image.pixel_array              # Read the pixel values into an array
    idata = np.array(idata)                # Make it a numpy array
    idown = ((idata / idata.max()) * 256).astype('uint8') # Downscale to uint8 for openCV techniques

    # Find phantom centre and radius
    (cenx, ceny, cradius) = find_circle(idown)

    # Create central 10x10 ROI and measure modal value
    cenroi = idata[(ceny-5):(ceny+5),(cenx-5):(cenx+5)]
    cenroi_flat = cenroi.flatten()
    (cenmode, modenum) = mode(cenroi_flat)

    # Create 160-pixel profiles (horizontal and vertical, centred at cenx,ceny)
    horroi = idata[(ceny-5):(ceny+5),(cenx-80):(cenx+80)]
    horprof = np.mean(horroi, axis=0)
    verroi = idata[(ceny-80):(ceny+80),(cenx-5):(cenx+5)]
    verprof = np.mean(verroi, axis=1)

    # Count how many elements are within 0.9-1.1 times the modal value
    hornum = np.where(np.logical_and((horprof > (0.9*cenmode)),(horprof < (1.1*cenmode))))
    hornum = len(hornum[0])
    vernum = np.where(np.logical_and((verprof > (0.9*cenmode)),(verprof < (1.1*cenmode))))
    vernum = len(vernum[0])

    # Calculate fractional uniformity
    fract_uniformity_hor = hornum/160
    fract_uniformity_ver = vernum/160

    #print("Horizontal Fractional Uniformity: ",np.round(fract_uniformity_hor,2))
    #print("Vertical Fractional Uniformity: ",np.round(fract_uniformity_ver,2))

    # Define 75% area mask for alternative uniformity measures
    r = cradius*0.865
    n = len(idata)
    y,x = np.ogrid[:n,:n]
    dist_from_center = np.sqrt((x - cenx)**2 + (y-ceny)**2)

    mask = dist_from_center <= r

    # Draw new area for checking
    cv.circle(idown, (cenx.astype('int'), ceny.astype('int')), r.astype('int'), 128, 2)

    # Calculate stats for masked region
    roimean = np.mean(idata[mask])
    roistd = np.std(idata[mask])
    roimax = np.amax(idata[mask])
    roimin = np.amin(idata[mask])

    # Calculate CoV and integral uniformity
    cov = 100*roistd/roimean
    intuniform = (1-(roimax-roimin)/(roimax+roimin))*100

    #print("CoV: ",np.round(cov,1),"%")
    #print("Integral Uniformity (ACR): ",np.round(intuniform,1),"%")

    # ------ Ghosting ------

    # Measure mean pixel value within rectangular ROIs slightly outside the top/bottom/left/right of phantom
    meantop = np.mean(idata[(cenx-50):(cenx+50),(ceny-cradius-20):(ceny-cradius-10)])
    meanbot = np.mean(idata[(cenx-50):(cenx+50),(ceny+cradius+10):(ceny+cradius+20)])
    meanl = np.mean(idata[(cenx-cradius-20):(cenx-cradius-10),(ceny-50):(ceny+50)])
    meanr = np.mean(idata[(cenx+cradius+10):(cenx+cradius+20),(ceny-50):(ceny+50)])

    # Calculate percentage ghosting
    ghosting = np.abs(((meantop+meanbot)-(meanl+meanr))/(2*roimean))*100

    #print("Ghosting: ",np.round(ghosting,1), "%")

    # ------ Distortion ------

    # Threshold using half of roimin as a cutoff
    thresh = idata < roimin/2
    ithresh = idata
    ithresh[thresh] = 0

    # Find the indices of thresholded pixels
    bbox = np.argwhere(ithresh)
    (bbystart, bbxstart), (bbystop, bbxstop) = bbox.min(0), bbox.max(0) + 1

    idistort = (bbxstop-bbxstart)/(bbystop-bbystart)
    idistort =np.abs(idistort-1)

    #print("Distortion: ", np.round(idistort,2),"%")

    # Plot annotated image for user
    fig = plt.figure(1)
    plt.imshow(idown, cmap='gray')
    plt.show()

    return fract_uniformity_hor, fract_uniformity_ver, cov, intuniform, ghosting, idistort


def fwhm(image):
    """
    Determines the spatial resolution of image by measuring the FWHM across the edges of
    a uniform phantom

    parameters:
    ---------------
    image: dicom image read using dcmread

    returns:
    ---------------
    hor_fwhm:   fwhm (mm) for horizontal edge
    ver_fwhm:   fwhm (mm) for vertical edge
    """

    # Read pixel size
    pixelsize = image[0x28,0x30].value

    # Prepare image for processing
    idata = image.pixel_array              # Read the pixel values into an array
    idata = np.array(idata)                # Make it a numpy array
    idown = ((idata / idata.max()) * 256).astype('uint8')  # Downscale to uint8 for openCV techniques

    # Find phantom centre and radius
    (cenx, ceny, cradius) = find_circle(idown)

    # Create profile through edges
    lprof = idata[ceny,(cenx-cradius-20):(cenx-cradius+20)]
    bprof = idata[(ceny+cradius-20):(ceny+cradius+20), cenx]
    bprof = np.flipud(bprof)

    # Draw lines on image for checking
    cv.line(idown,((cenx-cradius-20),ceny),((cenx-cradius+20),ceny),128,2)
    cv.line(idown, (cenx, (ceny+cradius-20)), (cenx, (ceny+cradius+20)), 128, 2)

    # Differentiate profiles to obtain LSF
    llsf = np.gradient(lprof.astype(int))
    blsf = np.gradient(bprof.astype(int))

    # Calculate FWHM of LSFs
    hor_fwhm = calc_fwhm(llsf)*pixelsize[0]
    ver_fwhm = calc_fwhm(blsf)*pixelsize[0]

    # Plot annotated image for user
    fig = plt.figure(1)
    plt.imshow(idown, cmap='gray')
    plt.show()

    #print("Horizontal FWHM: ", np.round(hor_fwhm,2),"mm")
    #print("Vertical FWHM: ", np.round(ver_fwhm,2),"mm")

    return hor_fwhm, ver_fwhm


def find_circle(a):
    """
    Finds a circle that fits the outline of the phantom using the Hough Transform

    parameters:
    ---------------
    a: image array (should be uint8)

    returns:
    ---------------
    cenx, ceny: xy pixel coordinates of the centre of the phantom
    cradius : radius of the phantom in pixels
    errCircle: error message if good circle isn't found
    """

    # Perform Hough transform to find circle
    circles = cv.HoughCircles(a, cv.HOUGH_GRADIENT, 1, 200, param1=30,
                              param2=45, minRadius=0, maxRadius=0)

    # Check that a single phantom was found
    if len(circles) == 1:
        pass
        # errCircle = "1 circle found."
    else:
        errCircle = "Wrong number of circles detected, check image."
        print(errCircle)

    # Draw circle onto original image
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(a, (i[0], i[1]), i[2], 128, 2)
        # draw the center of the circle
        cv.circle(a, (i[0], i[1]), 2, 128, 2)

    cenx = i[0]
    ceny = i[1]
    cradius = i[2]
    return (cenx, ceny, cradius)


def conv2d(a, f):
    """
    Performs a 2D convolution (for filtering images)

    parameters:
    ---------------
    a: array to be filtered
    f: filter kernel

    returns:
    ---------------
    filtered numpy array
    """

    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def image_noise(a):
    """
    Separates the image noise by smoothing the image and subtracing the smoothed image
    from the original.

    parameters:
    ---------------
    a: image array from dcmread and .pixelarray

    returns:
    ---------------
    Inoise: image representing the image noise
    """

    # Create 3x3 boxcar kernel (recommended size - adjustments will affect results)
    size = (3, 3)
    kernel = np.ones(size) / 9

    # Convolve image with boxcar kernel
    imsmoothed = conv2d(a, kernel)
    # Pad the array (to counteract the pixels lost from the convolution)
    imsmoothed = np.pad(imsmoothed, 1, 'minimum')
    # Subtract smoothed array from original
    imnoise = a - imsmoothed

    return imnoise


def mode(a, axis=0):
    """
        Finds the modal value of an array. From scipy.stats.mode

        parameters:
        ---------------
        a: array

        returns:
        ---------------
        mostfrequent: the modal value
        oldcounts: the number of times this value was counted (check this)
        """

    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts


def calc_fwhm(lsf):
    """
    Determines the FWHM of an LSF

    parameters:
    ---------------
    lsf: image array (1D)

    returns:
    ---------------
    fwhm: the measured fwhm in pixels
    """

    # Find the peak pixel
    ind1 = np.argmax(lsf)

    # Create array of values around the peak
    lsfp = np.asarray([lsf[ind1-1], lsf[ind1], lsf[ind1+1]])
    # Create index array
    xloc = np.asarray([ind1-1, ind1, ind1+1])

    # Fit a 2nd order polynomial to the points
    fit = poly.polyfit(xloc,lsfp,2)

    # Find the peak value of this polynomial
    x = np.linspace(int(ind1-1), int(ind1+1), 1000)
    y = fit[0] + fit[1]*x + fit[2]*x**2
    peakval = np.max(y)
    halfpeak = peakval/2

    # Find indices where lsf > halfpeak
    gthalf = np.where(lsf > halfpeak)
    gthalf = gthalf[0]

    # Find left & right edges of profile and corresponding values
    leftx = np.asarray([gthalf[0]-1,gthalf[0]])
    lefty = lsf[leftx]
    rightx = np.asarray([gthalf[len(gthalf)-1],gthalf[len(gthalf)-1]+1])
    righty = lsf[rightx]

    # Use linear interpolation to find half maximum locations and calculate fwhm (pixels)
    lefthm = np.interp(halfpeak,lefty,leftx)
    righthm = np.interp(halfpeak,np.flip(righty,0),np.flip(rightx,0))
    fwhm = righthm-lefthm

    return fwhm


