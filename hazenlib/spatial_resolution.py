"""
Spatial Resolution

Contributors:
Haris Shuaib, haris.shuaib@gstt.nhs.uk
Neil Heraghty, neil.heraghty@nhs.net, 16/05/2018

.. todo::
    Replace shape finding functions with hazenlib.tools equivalents
    
"""
import copy

import cv2 as cv
import numpy as np
import numpy.polynomial.polynomial as poly
import pydicom


def maivis_deriv(x, a, h=1, n=1, axis=-1):
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
    max_deriv = len(a) // 4
    b = [(a[i + 2] + a[i + 3] - a[i] - a[i + 1]) / 2 for i in range(4 * max_deriv - 3)]
    # pad last 3 elements of b
    b.extend([b[-1]] * 3)
    return b


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
    image_histogram, bins = np.histogram(array.flatten(), 255)
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
                return box, rect[2]


def get_roi(pixels, centre, size=20):
    x, y = centre
    arr = pixels[x - size // 2: x + size // 2, y - size // 2: y + size // 2]
    return arr


def get_void_roi(pixels, circle, size=20):
    centre_x = circle[0][0][0]
    centre_y = circle[0][0][1]
    return get_roi(pixels=pixels, centre=(centre_x, centre_y), size=size)


def get_edge_roi(pixels, square, size=20):
    _, centre = get_right_edge_vector_and_centre(square)
    return get_roi(pixels, centre=(centre["x"], centre["y"]), size=size)


def edge_is_vertical(edge_roi, mean) -> bool:
    """
    control_parameter_01=0  ;a control parameter that will be equal to 1 if the edge is vertical and 0 if it is horizontal

for column=0, event.MTF_roi_size-2 do begin
if MTF_Data(column, 0 ) EQ mean_value then control_parameter_01=1
if (MTF_Data(column, 0) LT mean_value) AND (MTF_Data(column+1, 0) GT mean_value) then control_parameter_01=1
if (MTF_Data(column, 0) GT mean_value) AND (MTF_Data(column+1, 0) LT mean_value) then control_parameter_01=1
end
    Returns:

    """
    for col in range(edge_roi.shape[0] - 1):

        if edge_roi[col, 0] == mean:
            return True
        if edge_roi[col, 0] < mean < edge_roi[col + 1, 0]:
            return True
        if edge_roi[col, 0] > mean > edge_roi[col + 1, 0]:
            return True

    return False


def get_bisecting_normal(vector, centre, length_factor=0.25):
    # calculate coordinates of bisecting normal
    nrx_1 = centre["x"] - int(length_factor * vector["y"])
    nry_1 = centre["y"] + int(length_factor * vector["x"])
    nrx_2 = centre["x"] + int(length_factor * vector["y"])
    nry_2 = centre["y"] - int(length_factor * vector["x"])
    return nrx_1, nry_1, nrx_2, nry_2


def get_right_edge_vector_and_centre(square):
    # Calculate dx and dy
    right_edge_profile_vector = {"x": square[3][0] - square[0][0], "y": square[3][1] - square[0][1]}

    # Calculate centre (x,y) of edge
    right_edge_profile_roi_centre = {"x": square[0][0] + int(right_edge_profile_vector["x"] / 2),
                                     "y": square[0][1] + int(right_edge_profile_vector["y"] / 2)}
    return right_edge_profile_vector, right_edge_profile_roi_centre


def get_right_edge_normal_profile(img, square):
    right_edge_profile_vector, right_edge_profile_roi_centre = get_right_edge_vector_and_centre(square)

    n1x, n1y, n2x, n2y = get_bisecting_normal(right_edge_profile_vector, right_edge_profile_roi_centre)
    right_edge_profile = create_line_iterator([n1x, n1y], [n2x, n2y], img)

    intensities = right_edge_profile[:, -1]

    return intensities


def get_signal_roi(pixels, square, circle, size=20):
    _, square_centre = get_right_edge_vector_and_centre(square)
    circle_r = circle[0][0][2]
    x = square_centre["x"] + circle_r // 2
    y = square_centre["y"]
    return get_roi(pixels=pixels, centre=(x, y), size=size)


def get_edge(edge_arr, mean_value, spacing):
    if edge_is_vertical(edge_arr, mean_value):
        edge_arr = np.rot90(edge_arr)

    x_edge = [0] * 20
    y_edge = [0] * 20

    for row in range(20):
        for col in range(19):
            control_parameter_02 = 0
            #         print(f"signal_arr[col, row]={signal_arr[col, row]}")
            #         print(f"signal_arr[col, row+1]={signal_arr[col, row]}")
            if edge_arr[row, col] == mean_value:
                control_parameter_02 = 1
            if (edge_arr[row, col] < mean_value) and (edge_arr[row, col + 1] > mean_value):
                control_parameter_02 = 1
            if (edge_arr[row, col] > mean_value) and (edge_arr[row, col + 1] < mean_value):
                control_parameter_02 = 1

            if control_parameter_02 == 1:
                x_edge[row] = row * spacing[0]
                y_edge[row] = col * spacing[1]

    return x_edge, y_edge, edge_arr


def get_edge_angle_and_intercept(x_edge, y_edge):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #  Apply least squares method for the edge
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    mean_x = np.mean(x_edge)
    mean_y = np.mean(y_edge)

    slope_up = np.sum((x_edge - mean_x) * (y_edge - mean_y))
    slope_down = np.sum((x_edge - mean_x) * (x_edge - mean_x))
    slope = slope_up / slope_down
    angle = np.arctan(slope)
    intercept = mean_y - slope * mean_x
    return angle, intercept


def get_edge_profile_coords(angle, intercept, spacing):
    # ;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ; translate and rotate the data's coordinates according to the slope and intercept
    # ;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    original_mtf_x_position = np.array([x * spacing[0] for x in range(20)])
    original_mtf_x_positions = copy.copy(original_mtf_x_position)
    for row in range(19):
        original_mtf_x_positions = np.row_stack((original_mtf_x_positions, original_mtf_x_position))

    original_mtf_y_position = np.array([x * spacing[0] for x in range(20)])
    original_mtf_y_positions = copy.copy(original_mtf_y_position)
    for row in range(19):
        original_mtf_y_positions = np.column_stack((original_mtf_y_positions, original_mtf_y_position))

    # we are only interested in the rotated y positions as there correspond to the distance of the data from the edge
    rotated_mtf_y_positions = -original_mtf_x_positions * np.sin(angle) + (
            original_mtf_y_positions - intercept) * np.cos(angle)

    return original_mtf_x_positions, rotated_mtf_y_positions


def get_esf(edge_arr, y):
    # ;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ;extract the edge response function
    # ;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ;extract the distance from the edge and the corresponding data as vectors

    edge_distance = copy.copy(y[0, :])

    for row in range(1, 20):
        edge_distance = np.append(edge_distance, y[row, :])

    esf_data = copy.copy(edge_arr[:, 0])
    for row in range(1, 20):
        esf_data = np.append(esf_data, edge_arr[:, row])

    # sort the distances and the data accordingly
    ind_edge_distance = np.argsort(edge_distance)
    sorted_edge_distance = edge_distance[ind_edge_distance]
    sorted_esf_data = esf_data[ind_edge_distance]

    # get rid of duplicates (if two data correspond to the same distance) and replace them with their average
    temp_array01 = np.array([sorted_edge_distance[0]])
    temp_array02 = np.array([sorted_esf_data[0]])

    for element in range(1, len(sorted_edge_distance)):

        if not (sorted_edge_distance[element] - temp_array01[-1]).all():
            temp_array02[-1] = (temp_array02[-1] + sorted_esf_data[element]) / 2

        else:
            temp_array01 = np.append(temp_array01, sorted_edge_distance[element])
            temp_array02 = np.append(temp_array02, sorted_esf_data[element])

    # ;interpolate the edge response function (ESF) so that it only has 128 elements
    u = np.linspace(temp_array01[0], temp_array01[-1], 128)
    esf = np.interp(u, temp_array01, temp_array02)

    return u, esf


def calculate_mtf(dicom):
    pixels = dicom.pixel_array
    img = rescale_to_byte(pixels)  # rescale for OpenCV operations
    thresh = thresh_image(img)
    circle = get_circles(img)
    square, tilt = find_square(thresh)
    _, centre = get_right_edge_vector_and_centre(square)

    void_arr = get_void_roi(pixels, circle)
    edge_arr = get_edge_roi(pixels, square)
    signal_arr = get_signal_roi(pixels, square, circle)

    spacing = dicom.PixelSpacing
    mean = np.mean([void_arr, signal_arr])

    x_edge, y_edge, edge_arr = get_edge(edge_arr, mean, spacing)
    angle, intercept = get_edge_angle_and_intercept(x_edge, y_edge)
    x, y = get_edge_profile_coords(angle, intercept, spacing)
    u, esf = get_esf(edge_arr, y)
    lsf = maivis_deriv(u, esf)
    mtf = abs(np.fft.fft(lsf))
    norm_mtf = mtf / mtf[0]
    mtf_50 = min([i for i in range(len(norm_mtf) - 1) if norm_mtf[i] >= 0.5 >= norm_mtf[i + 1]])
    profile_length = max(y.flatten()) - min(y.flatten())
    mtf_frequency = 10.0 * mtf_50 / profile_length
    res = 10 / (2 * mtf_frequency)

    return res


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


def main(data: list) -> dict:

    results = calculate_mtf(data[0])

    return {'spatial_resolution': results}
