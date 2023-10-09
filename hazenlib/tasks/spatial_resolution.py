"""
Spatial Resolution

Contributors:
Haris Shuaib, haris.shuaib@gstt.nhs.uk
Neil Heraghty, neil.heraghty@nhs.net, 16/05/2018

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

import hazenlib.utils
from hazenlib.HazenTask import HazenTask


class SpatialResolution(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.single_dcm = self.dcm_list[0]

    def run(self) -> dict:
        results = self.init_result_dict()
        results['file'] = self.img_desc(self.single_dcm)
        try:
            pe_result, fe_result = self.calculate_mtf(self.single_dcm)
            results['measurement'] = {
                'phase encoding direction mm': round(pe_result, 2),
                'frequency encoding direction mm': round(fe_result, 2)
            }
        except Exception as e:
            print(f"Could not calculate the spatial resolution for {self.img_desc(self.single_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def deri(self, a):
        # This function calculated the LSF by taking the derivative of the ESF. Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3643984/
        b = np.gradient(a)
        return b


    def get_circles(self, image):
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

    def thresh_image(self, img, bound=150):
        blurred = cv.GaussianBlur(img, (5, 5), 0)
        thresh = cv.threshold(blurred, bound, 255, cv.THRESH_TOZERO_INV)[1]
        return thresh

    def find_square(self, img):
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

    def get_roi(self, pixels, centre, size=20):
        y, x = centre
        arr = pixels[x - size // 2: x + size // 2, y - size // 2: y + size // 2]
        return arr

    def get_void_roi(self, pixels, circle, size=20):
        centre_x = circle[0][0][0]
        centre_y = circle[0][0][1]
        return self.get_roi(pixels=pixels, centre=(centre_x, centre_y), size=size)

    def get_edge_roi(self, pixels, edge_centre, size=20):
        return self.get_roi(pixels, centre=(edge_centre["x"], edge_centre["y"]), size=size)

    def edge_is_vertical(self, edge_roi, mean) -> bool:
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

    def get_bisecting_normal(self, vector, centre, length_factor=0.25):
        # calculate coordinates of bisecting normal
        nrx_1 = centre["x"] - int(length_factor * vector["y"])
        nry_1 = centre["y"] + int(length_factor * vector["x"])
        nrx_2 = centre["x"] + int(length_factor * vector["y"])
        nry_2 = centre["y"] - int(length_factor * vector["x"])
        return nrx_1, nry_1, nrx_2, nry_2

    def get_top_edge_vector_and_centre(self, square):
        # Calculate dx and dy
        top_edge_profile_vector = {"x": (square[0][0] + square[1][0]) // 2, "y": (square[0][1] + square[1][1]) // 2}

        # Calculate centre (x,y) of edge
        top_edge_profile_roi_centre = {"x": (square[0][0] + square[1][0]) // 2,
                                       "y": (square[0][1] + square[1][1]) // 2}

        return top_edge_profile_vector, top_edge_profile_roi_centre

    def get_right_edge_vector_and_centre(self, square):
        # Calculate dx and dy
        right_edge_profile_vector = {"x": square[3][0] - square[0][0], "y": square[3][1] - square[0][1]}  # nonsense

        # Calculate centre (x,y) of edge
        right_edge_profile_roi_centre = {"x": (square[3][0] + square[0][0]) // 2,
                                         "y": (square[3][1] + square[0][1]) // 2}
        return right_edge_profile_vector, right_edge_profile_roi_centre


    def get_signal_roi(self, pixels, edge, edge_centre, circle, size=20):
        circle_r = circle[0][0][2]
        if edge == 'right':
            x = edge_centre["x"] + circle_r // 2
            y = edge_centre["y"]
        elif edge == 'top':
            x = edge_centre["x"]
            y = edge_centre["y"] - circle_r // 2

        return self.get_roi(pixels=pixels, centre=(x, y), size=size)

    def get_edge(self, edge_arr, mean_value, spacing):
        if self.edge_is_vertical(edge_arr, mean_value):
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

    def get_edge_angle_and_intercept(self, x_edge, y_edge):
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

    def get_edge_profile_coords(self, angle, intercept, spacing):
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

        rotated_mtf_x_positions = original_mtf_x_positions * np.cos(angle) + (
                original_mtf_y_positions - intercept) * np.sin(angle)

        return rotated_mtf_x_positions, rotated_mtf_y_positions

    def get_esf(self, edge_arr, y):
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

    def calculate_mtf_for_edge(self, dicom, edge):
        pixels = dicom.pixel_array
        pe = dicom.InPlanePhaseEncodingDirection

        img = hazenlib.utils.rescale_to_byte(pixels)  # rescale for OpenCV operations
        thresh = self.thresh_image(img)
        circle = self.get_circles(img)
        square, box = self.find_square(thresh)
        if edge == 'right':
            _, centre = self.get_right_edge_vector_and_centre(square)
        else:
            _, centre = self.get_top_edge_vector_and_centre(square)

        edge_arr = self.get_edge_roi(pixels, centre)
        void_arr = self.get_void_roi(pixels, circle)
        signal_arr = self.get_signal_roi(pixels, edge, centre, circle)
        spacing = hazenlib.utils.get_pixel_size(dicom)
        mean = np.mean([void_arr, signal_arr])
        x_edge, y_edge, edge_arr = self.get_edge(edge_arr, mean, spacing)
        angle, intercept = self.get_edge_angle_and_intercept(x_edge, y_edge)
        x, y = self.get_edge_profile_coords(angle, intercept, spacing)
        u, esf = self.get_esf(edge_arr, y)
        lsf = self.deri(esf)
        lsf = np.array(lsf)
        n = lsf.size
        mtf = abs(np.fft.fft(lsf))
        norm_mtf = mtf / mtf[0]
        mtf_50 = min([i for i in range(len(norm_mtf) - 1) if norm_mtf[i] >= 0.5 >= norm_mtf[i + 1]])
        profile_length = max(y.flatten()) - min(y.flatten())
        freqs = fftfreq(n, profile_length / n)
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
            axes[10].plot(freqs[mask], norm_mtf[mask])
            axes[10].set_xlabel('lp/mm')
            logger.debug(f'Writing report image: {self.report_path}_{pe}_{edge}.png')
            img_path = os.path.realpath(os.path.join(self.report_path,
                                    f'{self.img_desc(dicom)}_{pe}_{edge}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return res

    def calculate_mtf(self, dicom) -> tuple:
        pe = dicom.InPlanePhaseEncodingDirection
        pe_result, fe_result = None, None

        if pe == 'COL':
            pe_result = self.calculate_mtf_for_edge(dicom, 'top')
            fe_result = self.calculate_mtf_for_edge(dicom, 'right')
        elif pe == 'ROW':
            pe_result = self.calculate_mtf_for_edge(dicom, 'right')
            fe_result = self.calculate_mtf_for_edge(dicom, 'top')

        return pe_result, fe_result
