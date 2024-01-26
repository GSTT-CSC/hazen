"""
Spatial Resolution

Contributors:
Haris Shuaib, haris.shuaib@gstt.nhs.uk \n
Neil Heraghty, neil.heraghty@nhs.net, 16/05/2018 \n
Sophie Ratkai , sophie.ratkai@nhs.net, 25/01/2024
"""
import os
import sys
import copy
import traceback

import cv2 as cv
import numpy as np
from numpy.fft import fftfreq

import hazenlib.utils
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger


class SpatialResolution(HazenTask):
    """Spatial resolution measurement class for DICOM images of the MagNet phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.single_dcm = self.dcm_list[0]

    def run(self) -> dict:
        """Main function for performing spatial resolution measurement

        Returns:
            dict: results are returned in a standardised dictionary structurespecifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        results = self.init_result_dict()
        results["file"] = self.img_desc(self.single_dcm)

        pe = self.single_dcm.InPlanePhaseEncodingDirection
        print(pe)
        try:
            horizontal, vertical = self.calculate_mtf(self.single_dcm)
        except Exception as e:
            print(
                f"Could not calculate the spatial resolution for {self.img_desc(self.single_dcm)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        if pe == "COL":  # FE right, PE top
            pe_result = horizontal
            fe_result = vertical
        elif pe == "ROW":  # PE right, FE top
            pe_result = vertical  # self.calculate_mtf(dcm, "right")
            fe_result = horizontal  # self.calculate_mtf(dcm, "top")

        results["measurement"] = {
            "phase encoding direction mm": round(pe_result, 10),
            "frequency encoding direction mm": round(fe_result, 10),
        }

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_roi(self, arr: np.ndarray, centre: tuple, half_size=10):
        """Get coordinates of the region of interest

        Args:
            arr (np.ndarray): _description_
            centre (tuple): x,y (int) coordinates
            size (int, optional): diameter of the region of interest. Defaults to 20.

        Returns:
            np.ndarray: subset of the pixel array
        """
        y, x = centre
        roi_arr = arr[x - half_size : x + half_size, y - half_size : y + half_size]
        return roi_arr

    def edge_is_vertical(self, edge_roi: np.ndarray, mean) -> bool:
        """Determine whether edge is vertical
            control_parameter_01=0  ;a control parameter that will be equal to 1 if the edge is vertical and 0 if it is horizontal

        Args:
            edge_roi (np.ndarray): pixel array in ROI
            mean (float): array of mean pixel values

        Returns:
            bool: True or false whether edge is vertical
        """
        for col in range(19):
            # print(col)
            # # print(edge_roi[col, 0])
            # print(edge_roi[col, 0] - edge_roi[col + 1, 0])
            # print(mean - edge_roi[col + 1, 0])
            # print()
            if edge_roi[col, 0] <= mean <= edge_roi[col + 1, 0]:
                return True
            if edge_roi[col, 0] >= mean >= edge_roi[col + 1, 0]:
                return True

        return False

    def get_edge(self, edge_arr, mean_value, spacing):
        """Obtain list of values corrseponding to????

        Args:
            edge_arr (np.ndarray): edge ROI pixel array
            mean_value (float): mean pixel value between void and signal
            spacing (tuple of float): pixel spacing from DICOM header

        Returns:
            tuple of list: x_edge and y_edge are a list of values
        """
        x_edge = []
        y_edge = []

        # Iterate over every pixel value in the ROI array
        # save values to list when adjacent pixel values are
        # on either side of the mean pixel value
        for row in range(20):
            for col in range(19):
                if (edge_arr[row, col] <= mean_value) and (
                    edge_arr[row, col + 1] >= mean_value
                ):
                    x_edge.append(row * spacing[0])
                    y_edge.append(col * spacing[1])
                    break

                elif (edge_arr[row, col] >= mean_value) and (
                    edge_arr[row, col + 1] <= mean_value
                ):
                    x_edge.append(row * spacing[0])
                    y_edge.append(col * spacing[1])
                    break
                else:
                    pass

        # 0.0, 0.48828125, 0.9765625, 1.46484375, 1.953125, 2.44140625, 2.9296875, 3.41796875, 3.90625, 4.39453125, 4.8828125, 5.37109375, 5.859375, 6.34765625, 6.8359375, 7.32421875, 7.8125, 8.30078125, 8.7890625, 9.27734375]
        # [5.859375, 5.37109375, 5.37109375, 5.37109375, 5.37109375, 5.37109375, 5.37109375, 5.37109375, 4.8828125, 4.8828125, 4.8828125, 4.8828125, 4.8828125, 4.8828125, 4.39453125, 4.39453125, 4.39453125, 4.39453125, 4.39453125, 4.39453125]

        return x_edge, y_edge

    def get_edge_angle_and_intercept(self, x_edge, y_edge):
        """Get edge (slope) angle and intercept value

        Args:
            x_edge (np.ndarray): _description_
            y_edge (np.ndarray): _description_

        Returns:
            tuple: angle and intercept values
        """
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
        """translate and rotate the data's coordinates according to the slope and intercept

        Args:
            angle (ndarray or scalar): angle of slope
            intercept (ndarray or scalar): intercept of slope
            spacing (tuple/list): spacing value in x and y directions

        Returns:
            tuple: of np.ndarrays of the rotated MTF positions in x and y directions
        """

        original_mtf_x_position = np.array([x * spacing[0] for x in range(20)])
        original_mtf_x_positions = copy.copy(original_mtf_x_position)
        for row in range(19):
            original_mtf_x_positions = np.row_stack(
                (original_mtf_x_positions, original_mtf_x_position)
            )

        original_mtf_y_position = np.array([x * spacing[1] for x in range(20)])
        original_mtf_y_positions = copy.copy(original_mtf_y_position)
        for row in range(19):
            original_mtf_y_positions = np.column_stack(
                (original_mtf_y_positions, original_mtf_y_position)
            )

        # we are only interested in the rotated y positions as there correspond to the distance of the data from the edge
        rotated_mtf_y_positions = -original_mtf_x_positions * np.sin(angle) + (
            original_mtf_y_positions - intercept
        ) * np.cos(angle)

        rotated_mtf_x_positions = original_mtf_x_positions * np.cos(angle) + (
            original_mtf_y_positions - intercept
        ) * np.sin(angle)

        return rotated_mtf_x_positions, rotated_mtf_y_positions

    def get_esf(self, edge_arr, y):
        """Extract the edge response function

        Args:
            edge_arr (np.ndarray): _description_
            y (np.ndarray): _description_

        Returns:
            tuple: u and esf - 'normal' and interpolated edge response function (ESF)
        """

        # extract the distance from the edge and the corresponding data as vectors

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

    def calculate_mtf(self, dcm) -> tuple:
        """Calculate MTF in horizontal and verrtical directions

        Args:
        dcm (pydicom.Dataset): DICOM image object

        Returns:
            tuple: MTF in the horizontal and vertical direction
        """
        arr = dcm.pixel_array
        pe = dcm.InPlanePhaseEncodingDirection
        spacing = hazenlib.utils.get_pixel_size(dcm)

        """
            # 1. Detect circle to locate phantom centre
            # img = hazenlib.utils.rescale_to_byte(arr)  # rescale for OpenCV operations
            # thresh = self.thresh_image(img)
            # circle = self.get_circles(img)
            # circle_radius = circle[2]
            # print(circle_radius)

            # [254 256 197]
            # 255.67379760742188 256.6147155761719 196.54336547851562
            # 256 257 197
        """

        # 1a. Using ShapeDetector
        shape_detector = hazenlib.utils.ShapeDetector(arr)
        x, y, r = shape_detector.get_shape("circle")
        circle_x, circle_y, circle_radius = round(x), round(y), round(r)
        void_arr = self.get_roi(arr=arr, centre=(circle_x, circle_y))

        """
            # 2. Detect square and return coordinates of corners
            # square, box = self.find_square(thresh)
            # # array of tuples, coordinates of the 4 corners
            # print("square")
            # print(square)
            # # 2d array of coordinates of the 4 corners
            # print("box")
            # print(box)
            # print(box.shape)
        """

        # 2a. Return centre coords, size and angle
        (x, y), size, angle = shape_detector.get_shape("rectangle")
        # print(x, y, size, angle)
        # Get coordinates of the corners
        detected_box = cv.boxPoints(((x, y), size, angle))
        box_coords = detected_box.round().astype(int)
        # =========================================================

        # 3. Get edge ROI, signal ROI (void ROI - based on circle)
        # a. Calculate centre coordinates of edges
        top_edge_centre = (
            (box_coords[3][0] + box_coords[2][0]) // 2,
            (box_coords[3][1] + box_coords[2][1]) // 2,
        )

        top_signal_roi_x = top_edge_centre[0]
        top_signal_roi_y = top_edge_centre[1] - circle_radius // 2

        signal_arr = self.get_roi(arr=arr, centre=(top_signal_roi_x, top_signal_roi_y))

        mean = np.mean([void_arr, signal_arr])

        ##### Get edge ROI
        edge_arr = self.get_roi(arr, top_edge_centre)

        if self.edge_is_vertical(edge_arr, mean):
            logger.info("TOP edge is vertical, rotating pixel array")
            horizontal_edge_arr = np.rot90(edge_arr)
        else:
            horizontal_edge_arr = edge_arr

        x_edge, y_edge = self.get_edge(horizontal_edge_arr, mean, spacing)
        angle, intercept = self.get_edge_angle_and_intercept(x_edge, y_edge)
        x, y = self.get_edge_profile_coords(angle, intercept, spacing)
        u, esf = self.get_esf(horizontal_edge_arr, y)
        # This function calculated the LSF by taking the derivative of the ESF.
        # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3643984/
        lsf = np.gradient(esf)
        lsf = np.array(lsf)
        n = lsf.size
        mtf = abs(np.fft.fft(lsf))
        norm_mtf = mtf / mtf[0]
        mtf_50 = min(
            [
                i
                for i in range(len(norm_mtf) - 1)
                if norm_mtf[i] >= 0.5 >= norm_mtf[i + 1]
            ]
        )
        profile_length = max(y.flatten()) - min(y.flatten())
        freqs = fftfreq(n, profile_length / n)
        mask = freqs >= 0
        mtf_frequency = 10.0 * mtf_50 / profile_length
        res1 = 10 / (2 * mtf_frequency)
        print(f"TOP resolution is {res1}")

        # =========================================================
        right_edge_centre = (
            (box_coords[0][0] + box_coords[3][0]) // 2,
            (box_coords[0][1] + box_coords[3][1]) // 2,
        )

        right_signal_roi_x = right_edge_centre[0] + circle_radius // 2
        right_signal_roi_y = right_edge_centre[1]

        signal_arr = self.get_roi(
            arr=arr, centre=(right_signal_roi_x, right_signal_roi_y)
        )
        mean = np.mean([void_arr, signal_arr])

        ##### Get edge ROI
        edge_arr = self.get_roi(arr, right_edge_centre)

        if self.edge_is_vertical(edge_arr, mean):
            logger.info("RIGHT edge is vertical, rotating pixel array")
            horizontal_edge_arr = np.rot90(edge_arr)
        else:
            horizontal_edge_arr = edge_arr

        x_edge, y_edge = self.get_edge(horizontal_edge_arr, mean, spacing)
        angle, intercept = self.get_edge_angle_and_intercept(x_edge, y_edge)
        x, y = self.get_edge_profile_coords(angle, intercept, spacing)
        u, esf = self.get_esf(horizontal_edge_arr, y)
        # This function calculated the LSF by taking the derivative of the ESF.
        # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3643984/
        lsf = np.gradient(esf)
        lsf = np.array(lsf)
        n = lsf.size
        mtf = abs(np.fft.fft(lsf))
        norm_mtf = mtf / mtf[0]
        mtf_50 = min(
            [
                i
                for i in range(len(norm_mtf) - 1)
                if norm_mtf[i] >= 0.5 >= norm_mtf[i + 1]
            ]
        )
        profile_length = max(y.flatten()) - min(y.flatten())
        freqs = fftfreq(n, profile_length / n)
        mask = freqs >= 0
        mtf_frequency = 10.0 * mtf_50 / profile_length
        res2 = 10 / (2 * mtf_frequency)
        print(f"RIGHT resolution is {res2}")

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(11, 1)
            fig.set_size_inches(5, 36)
            fig.tight_layout(pad=4)
            axes[0].set_title("raw pixels")
            axes[0].imshow(arr, cmap="gray")
            # axes[1].set_title("rescaled to byte")
            # axes[1].imshow(img, cmap="gray")
            # axes[2].set_title("thresholded")
            # axes[2].imshow(thresh, cmap="gray")
            axes[3].set_title("finding circle")
            c = cv.circle(arr, (circle_x, circle_y), circle_radius, (255, 0, 0))
            axes[3].imshow(c)
            box = cv.drawContours(arr, [box_coords], 0, (255, 0, 0), 1)
            axes[4].set_title("finding MTF square")
            axes[4].imshow(box)
            axes[5].set_title("edge ROI")
            axes[5].imshow(edge_arr, cmap="gray")
            axes[6].set_title("void ROI")
            im = axes[6].imshow(void_arr, cmap="gray")
            fig.colorbar(im, ax=axes[6])
            axes[7].set_title("signal ROI")
            im = axes[7].imshow(signal_arr, cmap="gray")
            fig.colorbar(im, ax=axes[7])
            axes[8].set_title("edge spread function")
            axes[8].plot(esf)
            axes[8].set_xlabel("mm")
            axes[9].set_title("line spread function")
            axes[9].plot(lsf)
            axes[9].set_xlabel("mm")
            axes[10].set_title("normalised MTF")
            axes[10].plot(freqs[mask], norm_mtf[mask])
            axes[10].set_xlabel("lp/mm")
            logger.debug(f"Writing report image: {self.report_path}_{pe}_{edge}.png")
            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_{pe}_{edge}.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return res1, res2
