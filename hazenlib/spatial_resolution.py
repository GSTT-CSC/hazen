"""
Spatial Resolution

This script determines the spatial resolution of image by measuring the FWHM across the edges of
a uniform phantom

Created by Neil Heraghty
neil.heraghty@nhs.net

16/05/2018

"""
import cv2 as cv
import numpy as np
import numpy.polynomial.polynomial as poly
from PIL import Image
import pydicom
import pylab
from scipy.signal import hilbert, chirp


def maivis_deriv(a, h=1, n=1, axis=-1):
    """
    Performs differentiation the same way as done with MAIVIS to find the line spread function (LSF). This function has been re-implemented from IDL code written by Ioannis.

        ;differentiate the ESF the same way as done with MAIVIS to find the line spread function (LSF)
        max_deriv=FIX(n_elements(ESF)/4)

        lsf=FLTARR(4*max_deriv)

        for lsf_element=0, 4*max_deriv-4 do begin

        aa=ESF(lsf_element)+ESF(lsf_element+1)
        bb=ESF(lsf_element+2)+ESF(lsf_element+3)
        lsf(lsf_element)=(bb-aa)/2

        end

        ;pad the last 3 elements of the lsf
        for lsf_element=4*max_deriv-3, 4*max_deriv-1 do begin
        lsf(lsf_element)=lsf(4*max_deriv-4)
        end

     Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    Returns
    -------
    diff : ndarray
        The `n` order differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`.
    See Also
    --------
    idl_deriv
    numpy.diff

    """
    print(len(a))
    print(a)
    a = np.interp(np.arange(0, 128, 1), np.arange(len(a)), a)
    print(len(a))
    print(a)
    max_deriv = int(len(a) / 4)
    b = [(-a[i + 2] - a[i + 3] - a[i] + a[i + 1]) / 2 for i in range(4 * max_deriv - 4)]
    # pad last 3 elements of b
    print([b[-1]] * 3)
    b.extend([b[-1]] * 3)
    print(len(b))
    print(b)
    return np.asanyarray(b)


def idl_deriv(a, h=1, n=1, axis=-1):
    """
    Performs centred three-point Langrangian interpolation differentiation of a using IDL implementation from their documentation:

        "For an evenly-spaced array of values Y[0...N–1], the three-point Lagrangian interpolation reduces to:
            Y'[0] = (–3*Y[0] + 4*Y[1] – Y[2]) / 2
            Y'[i] = (Y[i+1] – Y[i–1]) / 2 ; i = 1...N–2
            Y'[N–1] = (3*Y[N–1] – 4*Y[N–2] + Y[N–3]) / 2
        This routine is written in the IDL language. You can find more details for all of the above equations
        in the source code in the file deriv.pro in the lib subdirectory of the IDL distribution."

    See: https://www.harrisgeospatial.com/docs/DERIV.html

     Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    Returns
    -------
    diff : ndarray
        The `n` order differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`.
    See Also
    --------
    numpy.diff

    """
    diff = [(a[i + 1] - a[i - 1]) / 2 for i in range(len(a) - 2)]
    diff[0] = (-3 * a[0] + 4 * a[1] - a[2]) / (2)
    diff[-1] = (3 * a[len(a) - 1] - 4 * a[len(a) - 2] + a[len(a) - 3]) / 2
    return np.asanyarray(diff)


def create_line_iterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    #  define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #  difference and absolute difference between points
    #  used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #  predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    #  Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


def rescale_to_byte(array):
    image_histogram, bins = np.histogram(array.flatten(), 255, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(array.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(array.shape).astype('uint8')


def get_circles(image):
    v = np.median(image)
    upper = int(min(255, (1.0 + 5) * v))
    i = 40
    while True:
        circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1.2, 256,
                                  param1=upper, param2=i, minRadius=100, maxRadius=256)
        i -= 1
        if circles is None:
            pass
        else:
            circles = np.uint16(np.around(circles))
            break
    return circles


def pshow_circles(images, circles, title='Circles'):
    if circles is None or len(circles) == 0:
        return
    #  print len(circles)
    all_images_and_circles = []
    for index, img in enumerate(images):
        cimage = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        print(len(circles[index][0, :]))
        for i in circles[index][0, :]:
            cv.circle(cimage, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(cimage, (i[0], i[1]), 2, (0, 0, 255), 3)
        all_images_and_circles.append(cimage)
    pylab.show(pylab.imshow(np.hstack(all_images_and_circles), cmap=pylab.cm.gray))
    return np.hstack(all_images_and_circles)


def thresh_image(img, bound=150):
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    thresh = cv.threshold(blurred, bound, 255, cv.THRESH_TOZERO_INV)[1]
    # pylab.imshow(thresh, cmap=pylab.cm.gray)
    # pylab.show()
    return thresh


def find_square(img):
    cnts = cv.findContours(img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts[1]:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.1 * perimeter, True)
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            rect = cv.minAreaRect(approx)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            w, h = rect[1]
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            if 0.95 < ar < 1.05:
                return box


def get_bisecting_normal(vector, centre, length_factor=0.25):
    # calculate coordinates of bisecting normal
    nrx_1 = centre["x"] - int(length_factor * vector["y"])
    nry_1 = centre["y"] + int(length_factor * vector["x"])
    nrx_2 = centre["x"] + int(length_factor * vector["y"])
    nry_2 = centre["y"] - int(length_factor * vector["x"])
    return nrx_1, nry_1, nrx_2, nry_2


def get_right_edge_normal_profile(img, square):

    # Calculate dx and dy
    right_edge_profile_vector = {"x": square[3][0] - square[0][0], "y": square[3][1] - square[0][1]}

    # Calculate centre (x,y) of edge
    right_edge_profile_roi_centre = {"x": square[0][0] + int(right_edge_profile_vector["x"] / 2),
                                     "y": square[0][1] + int(right_edge_profile_vector["y"] / 2)}

    n1x, n1y, n2x, n2y = get_bisecting_normal(right_edge_profile_vector, right_edge_profile_roi_centre)
    right_edge_profile = create_line_iterator([n1x, n1y], [n2x, n2y], img)

    intensities = right_edge_profile[:, -1]

    return intensities


def calculate_mtf(dicom):

    img = rescale_to_byte(dicom.pixel_array)
    thresh = thresh_image(img)
    square = find_square(thresh)
    profile = get_right_edge_normal_profile(img, square)

    spacing = dicom.PixelSpacing
    lsf = spacing * np.diff(profile[::-1], n=1)
    my_lsf = spacing * idl_deriv(profile[::-1])

    # pylab.plot(range(len(lsf)),lsf)
    # pylab.plot(range(len(my_lsf)), my_lsf)

    mtf = np.fft.fft(lsf)
    my_mtf = np.fft.fft(my_lsf)
    my_mtf = my_mtf / my_mtf[0]
    # pylab.plot(range(len(mtf)),mtf)
    # pylab.plot(range(len(my_mtf)), my_mtf)

    analytic_signal = hilbert(np.real(mtf.astype(int)))
    my_analytic_signal = hilbert(np.real(my_mtf.astype(int)))
    amplitude_envelope = np.abs(analytic_signal)
    my_amplitude_envelope = np.abs(my_analytic_signal)
    # pylab.plot(range(len(mtf)), amplitude_envelope)
    # pylab.plot(range(len(my_mtf)), my_amplitude_envelope)


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
    circles = cv.HoughCircles(a, cv.HOUGH_GRADIENT, 1, 100, param1=50,
                              param2=30, minRadius=0, maxRadius=0)

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
        cv.circle(a, (i[0], i[1]), i[2], 30, 2)
        # draw the center of the circle
        cv.circle(a, (i[0], i[1]), 2, 30, 2)

    cenx = i[0]
    ceny = i[1]
    cradius = i[2]
    return (cenx, ceny, cradius)


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
    lsfp = np.asarray([lsf[ind1 - 1], lsf[ind1], lsf[ind1 + 1]])
    # Create index array
    xloc = np.asarray([ind1 - 1, ind1, ind1 + 1])

    # Fit a 2nd order polynomial to the points
    fit = poly.polyfit(xloc, lsfp, 2)

    # Find the peak value of this polynomial
    x = np.linspace(int(ind1 - 1), int(ind1 + 1), 1000)
    y = fit[0] + fit[1] * x + fit[2] * x ** 2
    peakval = np.max(y)
    halfpeak = peakval / 2

    # Find indices where lsf > halfpeak
    gthalf = np.where(lsf > halfpeak)
    gthalf = gthalf[0]

    # Find left & right edges of profile and corresponding values
    leftx = np.asarray([gthalf[0] - 1, gthalf[0]])
    lefty = lsf[leftx]
    rightx = np.asarray([gthalf[len(gthalf) - 1], gthalf[len(gthalf) - 1] + 1])
    righty = lsf[rightx]

    # Use linear interpolation to find half maximum locations and calculate fwhm (pixels)
    lefthm = np.interp(halfpeak, lefty, leftx)
    righthm = np.interp(halfpeak, np.flip(righty, 0), np.flip(rightx, 0))
    fwhm = righthm - lefthm

    return fwhm


def main(image):
    # Read pixel size
    pixelsize = image[0x28, 0x30].value

    # Prepare image for processing
    idata = image.pixel_array  # Read the pixel values into an array
    idata = np.array(idata)  # Make it a numpy array
    idown = (idata / 256).astype('uint8')  # Downscale to uint8 for openCV techniques

    # Find phantom centre and radius
    (cenx, ceny, cradius) = find_circle(idown)

    # Create profile through edges
    lprof = idata[ceny, (cenx - cradius - 20):(cenx - cradius + 20)]
    bprof = idata[(ceny + cradius - 20):(ceny + cradius + 20), cenx]
    bprof = np.flipud(bprof)

    # Differentiate profiles to obtain LSF
    llsf = np.gradient(lprof)
    blsf = np.gradient(bprof)

    # Calculate FWHM of LSFs
    hor_fwhm = calc_fwhm(llsf) * pixelsize[0]
    ver_fwhm = calc_fwhm(blsf) * pixelsize[0]

    # print("Horizontal FWHM: ", np.round(hor_fwhm,2),"mm")
    # print("Vertical FWHM: ", np.round(ver_fwhm,2),"mm")

    return hor_fwhm, ver_fwhm
