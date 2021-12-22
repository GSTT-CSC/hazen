"""
Spatial Resolution

Contributors:
Haris Shuaib, haris.shuaib@gstt.nhs.uk
Neil Heraghty, neil.heraghty@nhs.net, 16/05/2018

.. todo::
    Replace shape finding functions with hazenlib.tools equivalents
    
"""
import copy
import sys
import traceback
from hazenlib.logger import logger



import cv2 as cv
import numpy as np
from numpy.fft import fftfreq

import hazenlib

def deri(a):
    # This function calculated the LSF by taking the derivative of the ESF. Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3643984/
    b = np.gradient(a)
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


def get_circles(image):
    v = np.median(image)
    upper = int(min(255, (1.0 + 5) * v))
    i = 40

    while True:
        circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1.2, 256,
                                  param1=upper, param2=i, minRadius=80, maxRadius=200)
        # min and max radius need to accomodate at least 256 and 512 matrix sizes
        i -= 1
        if circles is None:
            pass
        else:
            circles = np.uint16(np.around(circles))
            break

    # img = cv.circle(image, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (255, 0, 0))
    # plt.imshow(img)
    # plt.show()
    return circles


def thresh_image(img, bound=150):
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    thresh = cv.threshold(blurred, bound, 255, cv.THRESH_TOZERO_INV)[1]
    return thresh


def find_square(img):
    cnts = cv.findContours(img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

    for c in cnts:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.1 * perimeter, True)
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            rect = cv.minAreaRect(approx)

            # OpenCV 4.5 adjustment
            # - cv.minAreaRect() output tuple order changed since v3.4
            # - swap rect[1] order & rotate rect[2] by -90
            # – convert tuple>list>tuple to do this
            rectAsList = list(rect)
            rectAsList[1] = (rectAsList[1][1], rectAsList[1][0])
            rectAsList[2] = rectAsList[2] - 90
            rect = tuple(rectAsList)

            box = cv.boxPoints(rect)
            box = np.int0(box)
            w, h = rect[1]
            ar = w / float(h)

            # make sure that the width of the square is reasonable size taking into account 256 and 512 matrix
            if not 20 < w < 100:

                continue

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            if 0.92 < ar < 1.08:
                break

    # points should start at top-right and go anti-clockwise
    top_corners = sorted(box, key=lambda x: x[1])[:2]
    top_corners = sorted(top_corners, key=lambda x: x[0], reverse=True)

    bottom_corners = sorted(box, key=lambda x: x[1])[2:]
    bottom_corners = sorted(bottom_corners, key=lambda x: x[0])
    return top_corners + bottom_corners, box


def get_roi(pixels, centre, size=20):
    y, x = centre
    arr = pixels[x - size // 2: x + size // 2, y - size // 2: y + size // 2]
    return arr


def get_void_roi(pixels, circle, size=20):
    centre_x = circle[0][0][0]
    centre_y = circle[0][0][1]
    return get_roi(pixels=pixels, centre=(centre_x, centre_y), size=size)


def get_edge_roi(pixels, edge_centre, size=20):
    return get_roi(pixels, centre=(edge_centre["x"], edge_centre["y"]), size=size)


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


def get_top_edge_vector_and_centre(square):
    # Calculate dx and dy
    top_edge_profile_vector = {"x": (square[0][0] + square[1][0])//2, "y": (square[0][1] + square[1][1])//2}

    # Calculate centre (x,y) of edge
    top_edge_profile_roi_centre = {"x": (square[0][0] + square[1][0])//2,
                                   "y": (square[0][1] + square[1][1])//2}

    return top_edge_profile_vector, top_edge_profile_roi_centre


def get_right_edge_vector_and_centre(square):
    # Calculate dx and dy
    right_edge_profile_vector = {"x": square[3][0] - square[0][0], "y": square[3][1] - square[0][1]}  # nonsense

    # Calculate centre (x,y) of edge
    right_edge_profile_roi_centre = {"x": (square[3][0] + square[0][0])//2,
                                     "y": (square[3][1] + square[0][1])//2}
    return right_edge_profile_vector, right_edge_profile_roi_centre


def get_right_edge_normal_profile(img, square):
    right_edge_profile_vector, right_edge_profile_roi_centre = get_right_edge_vector_and_centre(square)

    n1x, n1y, n2x, n2y = get_bisecting_normal(right_edge_profile_vector, right_edge_profile_roi_centre)
    right_edge_profile = create_line_iterator([n1x, n1y], [n2x, n2y], img)

    intensities = right_edge_profile[:, -1]

    return intensities


def get_signal_roi(pixels, edge, edge_centre, circle, size=20):
    circle_r = circle[0][0][2]
    if edge == 'right':
        x = edge_centre["x"] + circle_r // 2
        y = edge_centre["y"]
    elif edge == 'top':
        x = edge_centre["x"]
        y = edge_centre["y"] - circle_r // 2

    return get_roi(pixels=pixels, centre=(x, y), size=size)


def get_edge(edge_arr, mean_value, spacing):

    if edge_is_vertical(edge_arr, mean_value):
        edge_arr = np.rot90(edge_arr)

    x_edge = [0] * 20
    y_edge = [0] * 20

    for row in range(20):
        for col in range(19):
            control_parameter_02 = 0

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
    # ;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ;Apply least squares method for the edge
    # ;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    original_mtf_y_position = np.array([x * spacing[1] for x in range(20)])
    original_mtf_y_positions = copy.copy(original_mtf_y_position)
    for row in range(19):
        original_mtf_y_positions = np.column_stack((original_mtf_y_positions, original_mtf_y_position))

    # we are only interested in the rotated y positions as there correspond to the distance of the data from the edge
    rotated_mtf_y_positions = -original_mtf_x_positions * np.sin(angle) + (
            original_mtf_y_positions - intercept) * np.cos(angle)
    
    rotated_mtf_x_positions = original_mtf_x_positions*np.cos(angle) + (
        original_mtf_y_positions - intercept) * np.sin(angle)

    return rotated_mtf_x_positions, rotated_mtf_y_positions


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




def calculate_mtf_for_edge(dicom, edge, report_path=False):
    pixels = dicom.pixel_array
    pe = dicom.InPlanePhaseEncodingDirection

    img = hazenlib.rescale_to_byte(pixels)  # rescale for OpenCV operations
    thresh = thresh_image(img)
    circle = get_circles(img)
    square, box = find_square(thresh)
    if edge == 'right':
        _, centre = get_right_edge_vector_and_centre(square)
    else:
        _, centre = get_top_edge_vector_and_centre(square)

    edge_arr = get_edge_roi(pixels, centre)
    void_arr = get_void_roi(pixels, circle)
    signal_arr = get_signal_roi(pixels, edge, centre, circle)
    spacing = hazenlib.get_pixel_size(dicom)
    mean = np.mean([void_arr, signal_arr])
    x_edge, y_edge, edge_arr = get_edge(edge_arr, mean, spacing)
    angle, intercept = get_edge_angle_and_intercept(x_edge, y_edge)
    x, y = get_edge_profile_coords(angle, intercept, spacing)
    u, esf = get_esf(edge_arr, y)
    lsf = deri(esf)
    lsf = np.array(lsf)
    n=lsf.size
    mtf = abs(np.fft.fft(lsf))
    norm_mtf = mtf / mtf[0]
    mtf_50 = min([i for i in range(len(norm_mtf) - 1) if norm_mtf[i] >= 0.5 >= norm_mtf[i + 1]])
    profile_length = max(y.flatten()) - min(y.flatten())
    freqs= fftfreq(n, profile_length/n)
    mask = freqs >= 0
    mtf_frequency = 10.0 * mtf_50 / profile_length
    res = 10 / (2 * mtf_frequency)


    if report_path:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(11, 1)
        fig.set_size_inches(5, 36)
        fig.tight_layout(pad=4)
        axes[0].set_title('raw pixels')
        axes[0].imshow(pixels, cmap='gray')
        axes[1].set_title('rescaled to byte')
        axes[1].imshow(img, cmap='gray')
        axes[2].set_title('thresholded')
        axes[2].imshow(thresh, cmap='gray')
        axes[3].set_title('finding circle')
        c = cv.circle(img, (circle[0][0][0], circle[0][0][1]), circle[0][0][2], (255, 0, 0))
        axes[3].imshow(c)
        box = cv.drawContours(img, [box], 0, (255, 0, 0), 1)
        axes[4].set_title('finding MTF square')
        axes[4].imshow(box)
        axes[5].set_title('edge ROI')
        axes[5].imshow(edge_arr, cmap='gray')
        axes[6].set_title('void ROI')
        im = axes[6].imshow(void_arr, cmap='gray')
        fig.colorbar(im, ax=axes[6])
        axes[7].set_title('signal ROI')
        im = axes[7].imshow(signal_arr, cmap='gray')
        fig.colorbar(im, ax=axes[7])
        axes[8].set_title('edge spread function')
        axes[8].plot(esf)
        axes[8].set_xlabel('mm')
        axes[9].set_title('line spread function')
        axes[9].plot(lsf)
        axes[9].set_xlabel('mm')
        axes[10].set_title('normalised MTF')
        axes[10].plot(freqs[mask],norm_mtf[mask])
        axes[10].set_xlabel('lp/mm')
        fig.savefig(f'{report_path}_{pe}_{edge}.png')







    return res


def calculate_mtf(dicom, report_path=False):

    pe = dicom.InPlanePhaseEncodingDirection
    pe_result, fe_result = None, None

    if pe == 'COL':
        pe_result = calculate_mtf_for_edge(dicom, 'top', report_path)
        fe_result = calculate_mtf_for_edge(dicom, 'right', report_path)
    elif pe == 'ROW':
        pe_result = calculate_mtf_for_edge(dicom, 'right', report_path)
        fe_result = calculate_mtf_for_edge(dicom, 'top', report_path)

    return {'phase_encoding_direction': pe_result, 'frequency_encoding_direction': fe_result}


def main(data: list, report_path=False) -> dict:
    results = {}
    for dcm in data:
        try:
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_{dcm.InstanceNumber}"
            if report_path:
                report_path = key
        except AttributeError as e:
            logger.info(e)
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}"
        try:
            results[key] = calculate_mtf(dcm, report_path)

        except Exception as e:
            print(f"Could not calculate the spatial resolution for {key} because of : {e}")
            traceback.print_exc(file=sys.stdout)
            continue

    return results

