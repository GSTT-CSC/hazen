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
        # set contants
        self.ROI_half_size = 10

        # Obtain values from DICOM object
        self.single_dcm = self.dcm_list[0]
        self.arr = self.single_dcm.pixel_array
        self.spacing = hazenlib.utils.get_pixel_size(self.single_dcm)

        # 1. Detect phantom (circle) using ShapeDetector
        shape_detector = hazenlib.utils.ShapeDetector(self.arr)
        # Return centre coords, and radius
        x, y, r = shape_detector.get_shape("circle")
        circle_x, circle_y, circle_radius = round(x), round(y), round(r)
        logger.debug(
            "Centre coordinates, size and rotation angle of the detected square",
            args=(x, y, size, angle),
        )
        # TODO: visualise_shape()

        # 2. Detect square within phantom using ShapeDetector
        # Return centre coords, size and angle
        (x, y), size, angle = shape_detector.get_shape("rectangle")
        logger.debug(
            "Centre coordinates, size and rotation angle of the detected square",
            args=(x, y, size, angle),
        )
        # Get coordinates of the corners
        detected_box = cv.boxPoints(((x, y), size, angle))
        box_coords = detected_box.round().astype(int)
        # TODO: visualise_shape()

        # 3. Define regions to sample background (void),
        #    signal and edges in horizontal and vertical directions
        void_arr = self.get_roi(self.arr, (circle_x, circle_y))
        # TODO: visualise_shape()
        (
            horizontal_edge_centre,
            horizontal_signal_centre,
        ) = self.get_horizontal_roi_centres(box_coords, circle_radius)
        (
            vertical_edge_centre,
            vertical_signal_centre,
        ) = self.get_vertical_roi_centres(box_coords, circle_radius)

        # Common functions
        horizontal_mean = self.get_mean_signal(horizontal_signal_centre, void_arr)
        vertical_mean = self.get_mean_signal(vertical_signal_centre, void_arr)

        horizontal_edge_roi = self.get_edge_roi(horizontal_edge_centre, horizontal_mean)
        vertical_edge_roi = self.get_edge_roi(vertical_edge_centre, vertical_mean)
        # TODO: visualise_shape()

        # 4. Infer resolution from edge functions
        self.horizontal_resolution = self.calculate_mtf(
            horizontal_edge_roi, horizontal_mean
        )
        self.vertical_resolution = self.calculate_mtf(vertical_edge_roi, vertical_mean)
        # TODO: visualise_shape()

        print(f"horizontal resolution is {self.horizontal_resolution}")
        print(f"vertical resolution is {self.vertical_resolution}")

    def get_roi(self, arr: np.ndarray, centre: tuple):
        """Get pixel array within the region of interest

        Args:
            arr (np.ndarray): _description_
            centre (tuple): x,y (int) coordinates

        Returns:
            np.ndarray: subset of the pixel array
        """
        y, x = centre
        roi_arr = arr[
            x - self.ROI_half_size : x + self.ROI_half_size,
            y - self.ROI_half_size : y + self.ROI_half_size,
        ]
        return roi_arr

    def get_horizontal_roi_centres(self, box_coords, circle_radius) -> tuple:
        """Get the centre coordinates of the horizontal edge and signal ROI

        Args:
            box_coords (list[tuple]): x, y coordinates of the box corners
            circle_radius (int or float): radius of the phantom (circle)

        Returns:
            tuple of tuple: edge_centre and signal_centre coordinates
        """
        edge_centre = (
            (box_coords[3][0] + box_coords[2][0]) // 2,
            (box_coords[3][1] + box_coords[2][1]) // 2,
        )

        signal_centre = (edge_centre[0], edge_centre[1] - circle_radius // 2)

        return edge_centre, signal_centre

    def get_vertical_roi_centres(self, box_coords, circle_radius) -> tuple:
        """Get the centre coordinates of the vertical edge and signal ROI

        Args:
            box_coords (list[tuple]): x, y coordinates of the box corners
            circle_radius (int or float): radius of the phantom (circle)

        Returns:
            tuple of tuple: edge_centre and signal_centre coordinates
        """
        edge_centre = (
            (box_coords[0][0] + box_coords[3][0]) // 2,
            (box_coords[0][1] + box_coords[3][1]) // 2,
        )

        signal_centre = (edge_centre[0] + circle_radius // 2, edge_centre[1])

        return edge_centre, signal_centre

    def get_mean_signal(self, signal_centre: tuple, void_arr: np.ndarray) -> float:
        """Determine the mean signal value between the signal and background ROI

        Args:
            signal_centre (tuple): x, y coordinates of the centre for the signal ROI
            void_arr (np.ndarray): pixel array corresponding to the background ROI

        Returns:
            float: mean signal value
        """
        signal_arr = self.get_roi(self.arr, signal_centre)
        mean = np.mean([void_arr, signal_arr])
        return mean

    def is_edge_vertical(self, edge_roi: np.ndarray, mean: float) -> bool:
        """Determine whether edge is vertical

        Args:
            edge_roi (np.ndarray): pixel array in ROI
            mean (float): array of mean pixel values

        Returns:
            bool: True or False whether edge is vertical
        """
        # Iterate over first row of pixel array to determine whether there are
        # any consecutive pixel values on either side of the mean value
        for col in range(self.ROI_half_size * 2 - 1):
            if edge_roi[col, 0] <= mean <= edge_roi[col + 1, 0]:
                return True
            if edge_roi[col, 0] >= mean >= edge_roi[col + 1, 0]:
                return True

        return False

    def get_edge_roi(self, edge_centre: tuple, mean: float) -> np.ndarray:
        """Get pixel array in the edge region, in which the edge is horizontal

        Args:
            edge_centre (tuple of int): x, y coordinates of the edge ROI centre
            mean (float): mean pixel value

        Returns:
            np.ndarray: a pixel array in which the edge is horizontal
        """
        edge_arr = self.get_roi(self.arr, edge_centre)

        if self.is_edge_vertical(edge_arr, mean):
            logger.info("edge is vertical, rotating pixel array by 90 degrees")
            horizontal_edge_arr = np.rot90(edge_arr)
        else:
            horizontal_edge_arr = edge_arr

        return horizontal_edge_arr

    def get_edges(self, edge_arr: np.ndarray, mean_value: float) -> tuple[list]:
        """Obtain list of values corrseponding to ????

        Args:
            edge_arr (np.ndarray): edge ROI pixel array
            mean_value (float): mean pixel value between void and signal

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
                    x_edge.append(row * self.spacing[0])
                    y_edge.append(col * self.spacing[1])
                    break

                elif (edge_arr[row, col] >= mean_value) and (
                    edge_arr[row, col + 1] <= mean_value
                ):
                    x_edge.append(row * self.spacing[0])
                    y_edge.append(col * self.spacing[1])
                    break
                else:
                    pass

        return x_edge, y_edge

    def get_edge_angle_and_intercept(self, x_edge: list, y_edge: list) -> tuple[float]:
        """Get edge (slope) angle and intercept value, using least squares method for the edge

        Args:
            x_edge (list): _description_
            y_edge (list): _description_

        Returns:
            tuple of float: angle and intercept values
        """
        mean_x = np.mean(x_edge)
        mean_y = np.mean(y_edge)

        slope_up = np.sum((x_edge - mean_x) * (y_edge - mean_y))
        slope_down = np.sum((x_edge - mean_x) * (x_edge - mean_x))
        slope = slope_up / slope_down
        angle = np.arctan(slope)
        intercept = mean_y - slope * mean_x

        return angle, intercept

    def get_edge_profile_coords(self, angle: float, intercept: float):
        """translate and rotate the data's coordinates according to the slope and intercept

        Args:
            angle (float): angle of slope
            intercept (float): intercept of slope

        Returns:
            tuple: of np.ndarrays of the rotated MTF positions in x and y directions
        """

        original_mtf_x_position = np.array([x * self.spacing[0] for x in range(20)])
        original_mtf_x_positions = copy.copy(original_mtf_x_position)
        for row in range(19):
            original_mtf_x_positions = np.row_stack(
                (original_mtf_x_positions, original_mtf_x_position)
            )

        original_mtf_y_position = np.array([x * self.spacing[1] for x in range(20)])
        original_mtf_y_positions = copy.copy(original_mtf_y_position)
        for row in range(19):
            original_mtf_y_positions = np.column_stack(
                (original_mtf_y_positions, original_mtf_y_position)
            )

        # we are only interested in the rotated y positions
        # as they correspond to the distance of the data from the edge
        rotated_mtf_y_positions = -original_mtf_x_positions * np.sin(angle) + (
            original_mtf_y_positions - intercept
        ) * np.cos(angle)

        rotated_mtf_x_positions = original_mtf_x_positions * np.cos(angle) + (
            original_mtf_y_positions - intercept
        ) * np.sin(angle)

        return rotated_mtf_x_positions, rotated_mtf_y_positions

    def calculate_esf(self, edge_arr: np.ndarray, y: np.ndarray):
        """Interpolate the edge response function (ESF) as applied to the edge ROI

        Args:
            edge_arr (np.ndarray): edge ROI pixel array
            y (np.ndarray): _description_

        Returns:
            tuple: u and esf - 'normal' and interpolated edge response function (ESF)
        """

        # extract the distance from the edge and the corresponding data as vectors
        edge_distance = copy.copy(y[0, :])

        for row in range(1, self.ROI_half_size * 2):
            edge_distance = np.append(edge_distance, y[row, :])

        esf_data = copy.copy(edge_arr[:, 0])
        for row in range(1, self.ROI_half_size * 2):
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

    def calculate_mtf(self, edge_arr: np.ndarray, mean: float) -> float:
        """Calculate MTF of an edge response function
        Args:
            edge_arr (np.ndarray): edge ROI pixel array
            mean (float): mean pixel value between void and signal

        Returns:
            float: spatial resolution value
        """
        x_edge, y_edge = self.get_edges(edge_arr, mean)
        angle, intercept = self.get_edge_angle_and_intercept(x_edge, y_edge)
        x, y = self.get_edge_profile_coords(angle, intercept)
        u, esf = self.calculate_esf(edge_arr, y)
        # This function calculated the LSF by taking the derivative of the ESF.
        # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3643984/
        lsf = np.gradient(esf)
        lsf = np.array(lsf)
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
        freqs = fftfreq(lsf.size, profile_length / lsf.size)
        mask = freqs >= 0
        mtf_frequency = 10.0 * mtf_50 / profile_length
        resolution = 10 / (2 * mtf_frequency)

        return resolution

    def run(self) -> dict:
        """Main function for performing spatial resolution measurement

        Returns:
            dict: results are returned in a standardised dictionary structurespecifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        results = self.init_result_dict()
        results["file"] = self.img_desc(self.single_dcm)

        pe = self.single_dcm.InPlanePhaseEncodingDirection

        if pe == "COL":  # FE right, PE top
            pe_result = self.horizontal_resolution
            fe_result = self.vertical_resolution
        elif pe == "ROW":  # PE right, FE top
            pe_result = self.vertical_resolution
            fe_result = self.horizontal_resolution

        results["measurement"] = {
            "phase encoding direction mm": round(pe_result, 10),
            "frequency encoding direction mm": round(fe_result, 10),
        }

        # only return reports if requested
        if self.report:
            # TODO: separate plots and visualisation into their respective functions
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(11, 1)
            fig.set_size_inches(5, 36)
            fig.tight_layout(pad=4)
            axes[0].set_title("raw pixels")
            axes[0].imshow(self.arr, cmap="gray")
            # axes[1].set_title("rescaled to byte")
            # axes[1].imshow(img, cmap="gray")
            # axes[2].set_title("thresholded")
            # axes[2].imshow(thresh, cmap="gray")
            axes[3].set_title("finding circle")
            c = cv.circle(self.arr, (circle_x, circle_y), circle_radius, (255, 0, 0))
            axes[3].imshow(c)
            box = cv.drawContours(self.arr, [box_coords], 0, (255, 0, 0), 1)
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
                os.path.join(
                    self.report_path,
                    f"{self.img_desc(self.single_dcm)}_{pe}_{edge}.png",
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)
            results["report_image"] = self.report_files

        return results
