"""
Spatial Resolution

Contributors:
Haris Shuaib, haris.shuaib@gstt.nhs.uk
Neil Heraghty, neil.heraghty@nhs.net

16/05/2018

.. todo::
    Replace shape finding functions with hazenlib.utils equivalents
    
"""
import copy
import os
import sys
import traceback
from hazenlib.logger import logger

import cv2 as cv
import numpy as np
from numpy.fft import fftfreq

from hazenlib.HazenTask import HazenTask
from hazenlib.utils import rescale_to_byte, get_pixel_size, ShapeDetector


class SpatialResolution(HazenTask):
    """Task to measure spatial resolution using a MagNET phantom
    No additional arguments.

    Args:
        HazenTask: inherits from the HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        """Main function to run task with specified args

        Returns:
            results (dict): dictionary of task - value pair and optionally
                        a images key with value listing image paths
        """
        results = {}

        for dcm in self.data:
            try:
                logger.info("Calculating spatial resolution for image {}".format(
                    self.key(dcm)))
                result = self.calculate_mtf(dcm)
            except Exception as e:
                print(f"Could not calculate the spatial resolution for {self.key(dcm)} because of : {e}")
                traceback.print_exc(file=sys.stdout)
                continue

            results[self.key(dcm)] = result

        # only return reports if requested
        if self.report:
            results['reports'] = {'images': self.report_files}

        return results


    def find_circle(self, img):
        """Find circle in image

        Args:
            img (rescaled pixel array): _description_

        Returns:
            x, y, r (tuplr): centre coordinates x, y and radius r
        """
        v = np.median(img)
        upper = int(min(255, (1.0 + 5) * v))
        i = 40

        while True:
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.2, 256,
                        param1=upper, param2=i, minRadius=80, maxRadius=200)
            # min and max radius need to accomodate at least 256 and 512 matrix sizes
            i -= 1
            if circles is not None:
                circles = np.uint16(np.around(circles))
                break
        x = circles[0][0][0]
        y = circles[0][0][1]
        r = circles[0][0][2]

        # img = cv.circle(img, (x,y), r, (255, 0, 0))
        # plt.imshow(img)
        # plt.show()
        return x, y, r

    def get_thresh_image(self, img, bound=150):
        """Create an image of pixels above threshold

        Args:
            img (rescaled pixel array): _description_
            bound (int, optional): might be boundary? ie threshold?.
                    Defaults to 150.

        Returns:
            threshold image: np.ndarray
        """
        blurred = cv.GaussianBlur(img, (5, 5), 0)
        thresh = cv.threshold(blurred, bound, 255, cv.THRESH_TOZERO_INV)[1]
        return thresh

    def find_square(self, img):
        """Find square in image

        Args:
            img (np.ndarray): threshold image

        Returns:
            square and box: np.array
                square is a list of array/coordinates of the corners
                box is a np.ndarray, unordered corner coordinates
        """
        # TODO Replace shape finding functions with hazenlib.utils equivalents
        # arr = dcm.pixel_array
        # shape_detector = hazenlib.utils.ShapeDetector(arr=arr)
        # (x, y), size, angle = shape_detector.get_shape('rectangle')

        cnts = cv.findContours(img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

        for c in cnts:
            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.1 * perimeter, True)
            if len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x,y), (h,w), angle = cv.minAreaRect(approx)

                # OpenCV 4.5 adjustment
                # - cv.minAreaRect() output tuple order changed since v3.4
                # - swap rect[1] order & rotate rect[2] by -90
                # – convert tuple>list>tuple to do this
                angle = angle-90

                ar = w / float(h)
                # make sure that the width of the square is reasonable size taking into account 256 and 512 matrix
                if not 20 < w < 100:
                    continue

                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                if 0.92 < ar < 1.08:
                    # calculate coordinates of corner points --> np.ndarray
                    box = cv.boxPoints(((x,y), (w,h), angle))
                    # Convert values to int - TODO losing accuracy!!!
                    box = np.int0(box)
                    break

        # points should start at top-right and go anti-clockwise
        # TODO have a better way of ordering corner coords
        top_corners = sorted(box, key=lambda x: x[1])[:2]
        top_corners = sorted(top_corners, key=lambda x: x[0], reverse=True)

        bottom_corners = sorted(box, key=lambda x: x[1])[2:]
        bottom_corners = sorted(bottom_corners, key=lambda x: x[0])
        return top_corners + bottom_corners, box

    def get_centre(self, edge, square, circle_r):
        """Get coordinates of the centre of edge and define coordinates for the
        centre of the signal ROI

        Args:
            edge (string): "top" or "right" edge to find centre of
            square (list of np.array): coordinates of the square corners
            circle_r (int): radius of the located circle

        Returns:
            tuples: x,y coordinates of the centre of the selected edge and
                    x,y coordinates of the centre of the signal ROI
        """
        # Calculate centre (x,y) of edge, horizontal or vertical
        if edge == "top":
            edge_x = (square[0][0] + square[1][0]) // 2
            edge_y = (square[0][1] + square[1][1]) // 2

            signal_x = edge_x
            signal_y = edge_y - circle_r // 2
        elif edge == "right":
            edge_x = (square[3][0] + square[0][0]) // 2
            edge_y = (square[3][1] + square[0][1]) // 2

            signal_x = edge_x + circle_r // 2
            signal_y = edge_y
        return (edge_x, edge_y), (signal_x, signal_y)

    def get_roi(self, pixels, centre, size=10):
        """Get a subset of pixel values (array) correspoding to a region of interest

        Args:
            pixels (dcm.pixel_array): DICOM pixel array
            centre (tuple): x,y (np.array) coordinates of centre
            size (int, optional): HALF the size of required ROI.
                Defaults to 10, resulting in 20x20 ROI

        Returns:
            np.ndarray: ndarray of pixel values
        """
        y, x = centre # TODO check whether it is intentionally swapped?
        arr = pixels[x - size: x + size, y - size: y + size]
        return arr

    def is_edge_vertical(self, edge_arr, mean_value) -> bool:
        """Determine whether edge is vertical or horizontal

        Args:
            edge_arr (np.ndarray): 20x20 ROI of pixel values
            mean_value (np.mean): mean of the void and signal arrays

        Returns:
            boolean: True/False whether edge is vertical
        """
        """
            control_parameter_01=0  ;a control parameter that will be equal to 1 if the edge is vertical and 0 if it is horizontal

            for column=0, event.MTF_roi_size-2 do begin
            if MTF_Data(column, 0 ) EQ mean_value then control_parameter_01=1
            if (MTF_Data(column, 0) LT mean_value) AND (MTF_Data(column+1, 0) GT mean_value) then control_parameter_01=1
            if (MTF_Data(column, 0) GT mean_value) AND (MTF_Data(column+1, 0) LT mean_value) then control_parameter_01=1
            end
        """
        for col in range(edge_arr.shape[0] - 1):

            if edge_arr[col, 0] == mean_value:
                return True
            if edge_arr[col, 0] < mean_value < edge_arr[col + 1, 0]:
                return True
            if edge_arr[col, 0] > mean_value > edge_arr[col + 1, 0]:
                return True

        return False

    def get_edges(self, edge_arr, mean_value, spacing):
        """Get arrays of pixel values along horizontal and vertical edges

        Args:
            edge_arr (np.ndarray): 20x20 ROI of pixel values
            mean_value (np.mean): _description_
            spacing (tuple of float): pixel sizes in x, y

        Returns:
            tuple: list of values along edge, length 20
        """
        # Initialise lists that hold pixel values for each edge
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

        return x_edge, y_edge

    def get_angle_and_intercept(self, x_edge, y_edge):
        """Apply least squares method for the edge to obtain
        the coordinates and angle of intercept of the edges

        Args:
            x_edge (list): _description_
            y_edge (list): _description_

        Returns:
            tuple of angle, intercept:
                angle: np.arctan(slope)
                intercept: np.float
        """

        mean_x = np.mean(x_edge)
        mean_y = np.mean(y_edge)

        slope_up = np.sum((x_edge - mean_x) * (y_edge - mean_y))
        slope_down = np.sum((x_edge - mean_x) * (x_edge - mean_x))
        slope = slope_up / slope_down
        angle = np.arctan(slope)
        intercept = mean_y - slope * mean_x
        return angle, intercept

    def get_edge_profile_coords(self, angle, intercept, spacing):
        """Translate and rotate the data's coordinates
            according to the slope and intercept

        Args:
            angle (np.arctan(slope)): angle calculated from tangent of slope
            intercept (np.float64): intercept value
            spacing (tuple of float): pixel sizes in x, y

        Returns:
            tuple of np.array-s:
                rotated_mtf_x_positions, rotated_mtf_y_positions
        """

        original_mtf_x_position = np.array([x * spacing[0] for x in range(20)])
        original_mtf_x_positions = copy.copy(original_mtf_x_position)
        for row in range(19):
            original_mtf_x_positions = np.row_stack((original_mtf_x_positions, original_mtf_x_position))

        original_mtf_y_position = np.array([x * spacing[1] for x in range(20)])
        original_mtf_y_positions = copy.copy(original_mtf_y_position)
        for row in range(19):
            original_mtf_y_positions = np.column_stack((original_mtf_y_positions, original_mtf_y_position))

        # we are only interested in the rotated y positions as these correspond to the distance of the data from the edge
        rotated_mtf_y_positions = -original_mtf_x_positions * np.sin(angle) + (
                original_mtf_y_positions - intercept) * np.cos(angle)

        rotated_mtf_x_positions = original_mtf_x_positions * np.cos(angle) + (
                original_mtf_y_positions - intercept) * np.sin(angle)

        return rotated_mtf_x_positions, rotated_mtf_y_positions

    def get_esf(self, edge_arr, y):
        """Extract the edge response function
        extract the distance from the edge and the corresponding data as vectors

        Args:
            edge_arr (np.ndarray): 20x20 ROI of pixel values
            y (np.array): rotated_mtf_y_positions

        Returns:
            tuple of: u, esf
                u: evenly spaced numbers over a specified interval (np.ndarray)
                esf: One-dimensional linear interpolation (np.interp)
        """

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

        # interpolate the edge response function (ESF) so that it only has 128 elements
        u = np.linspace(temp_array01[0], temp_array01[-1], 128)
        esf = np.interp(u, temp_array01, temp_array02)

        return u, esf

    def calculate_mtf_for_edge(self, dcm, edge) -> float:
        """Calculate MTF along the edge

        Args:
            dcm (DICOM): DICOM image object
            edge (string): "top" or "right" edge of the image

        Returns:
            float: resolution (MTF value) along the selected edge
        """
        pixels = dcm.pixel_array
        pe = dcm.InPlanePhaseEncodingDirection

        img = rescale_to_byte(pixels)  # rescale for OpenCV operations
        circle_x, circle_y, circle_r = self.find_circle(img)

        thresh = self.get_thresh_image(img)

        square, box = self.find_square(thresh)
        edge_centre, signal_centre = self.get_centre(edge, square, circle_r)

        edge_roi = self.get_roi(pixels, edge_centre)
        void_arr = self.get_roi(pixels, (circle_x, circle_y))
        signal_arr = self.get_roi(pixels, signal_centre)
        mean_value = np.mean([void_arr, signal_arr])

        spacing = get_pixel_size(dcm)
        # Rotate the edge array if it is vertical
        if self.is_edge_vertical(edge_roi, mean_value):
            edge_roi = np.rot90(edge_roi)

        x_edge, y_edge = self.get_edges(edge_roi, mean_value, spacing)
        angle, intercept = self.get_angle_and_intercept(x_edge, y_edge)
        x, y = self.get_edge_profile_coords(angle, intercept, spacing)
        u, esf = self.get_esf(edge_roi, y)
        # get the derivative of ESF function as an np.ndarray
        lsf = np.gradient(esf)
        mtf = abs(np.fft.fft(lsf))
        norm_mtf = mtf / mtf[0]
        mtf_50 = min([i for i in range(len(norm_mtf) - 1) if norm_mtf[i] >= 0.5 >= norm_mtf[i + 1]])
        profile_length = max(y.flatten()) - min(y.flatten())
        freqs = fftfreq(lsf.size, profile_length / lsf.size)
        mask = freqs >= 0
        mtf_frequency = 10.0 * mtf_50 / profile_length
        res = 10 / (2 * mtf_frequency)

        if self.report:
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
            c = cv.circle(img, (circle_x, circle_y), circle_r, (255, 0, 0))
            axes[3].imshow(c)
            draw_box = cv.drawContours(img, [box], 0, (255, 0, 0), 1)
            axes[4].set_title('finding MTF square')
            axes[4].imshow(draw_box)
            axes[5].set_title('edge ROI')
            axes[5].imshow(edge_roi, cmap='gray')
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
            axes[10].plot(freqs[mask], norm_mtf[mask])
            axes[10].set_xlabel('lp/mm')

            pe = dcm.InPlanePhaseEncodingDirection
            img_name = f"{self.key(dcm)}_{pe}_{edge}.png"
            logger.info(f'Writing report image: {img_name}')
            img_path = os.path.realpath(os.path.join(self.report_path, img_name))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return res

    def calculate_mtf(self, dcm) -> dict:
        """Calculate MTF

        Args:
            dcm (DICOM): DICOM image object

        Returns:
            dict: dictionary of spatial resolution in the phase and frequency
                    encoding directions
        """
        pe = dcm.InPlanePhaseEncodingDirection
        top = self.calculate_mtf_for_edge(dcm, 'top')
        right = self.calculate_mtf_for_edge(dcm, 'right')

        if pe == 'COL':
            result = {
                        'phase_encoding_direction': top,
                        'frequency_encoding_direction': right
                    }
        elif pe == 'ROW':
            result = {
                        'phase_encoding_direction': right,
                        'frequency_encoding_direction': top
                    }
        # TODO should there be an else?

        return result
