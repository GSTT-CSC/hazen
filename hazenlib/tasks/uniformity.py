"""
Uniformity

Calculates uniformity for a single-slice image of a uniform MRI phantom

This script implements the IPEM/MAGNET method of measuring fractional uniformity.
It also calculates integral uniformity using a 75% area FOV ROI and CoV for the same ROI.

Created by Neil Heraghty neil.heraghty@nhs.net \n
14/05/2018

Updated by Sophie Ratkai sophie.ratkai@nhs.net \n
25/01/2024
"""

import os
import sys
import traceback
import numpy as np
from scipy import stats

from hazenlib.utils import ShapeDetector, get_image_orientation
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
import hazenlib.exceptions as exc


class Uniformity(HazenTask):
    """Uniformity measurement class for DICOM images of the MagNet phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the single DICOM input to be the first in the list
        self.single_dcm = self.dcm_list[0]

    def run(self) -> dict:
        """Main function for performing uniformity measurement

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        results = self.init_result_dict()
        img_desc = self.img_desc(self.single_dcm)
        results["file"] = img_desc
        logger.debug("------------------------------")
        logger.debug(img_desc)

        try:
            horizontal_uniformity, vertical_uniformity = self.get_fractional_uniformity(
                self.single_dcm
            )
            results["measurement"] = {
                "horizontal %": round(horizontal_uniformity, 2),
                "vertical %": round(vertical_uniformity, 2),
            }
        except Exception as e:
            logger.warning(
                f"Could not calculate the uniformity for {img_desc} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_object_centre(self, arr, orientation):
        """Locate centre coordinates

        Args:
            dcm (pydicom.FileDataset): DICOM image object

        Returns:
            tuple: x and y coordinates
        """
        shape_detector = ShapeDetector(arr)

        if orientation in ["Sagittal", "Coronal"]:
            # orientation is sagittal to patient
            try:
                (x, y), size, angle = shape_detector.get_shape("rectangle")
            except exc.ShapeError:
                raise

        else:
            # orientation == "Transverse" or axial
            x, y, r = shape_detector.get_shape("circle")

        return int(x), int(y)

    def get_profile_fraction(self, roi_array, central_roi_mode, direction):
        """Calculate fractional value of pixel values within threshold across a profile

        Args:
            roi_array (np.ndarray): pixel array
            central_roi_mode (int): modal value within central ROI
            direction (int): 0 or 1 corresponding to horizontal and vertical axis

        Returns:
            float: profile fraction
        """
        profile_mean = np.mean(roi_array, axis=direction, out=None)
        and_arr = np.logical_and(
            (profile_mean > (0.9 * central_roi_mode)),
            (profile_mean < (1.1 * central_roi_mode)),
        )
        profile_count = np.where(and_arr)
        profile_fraction = len(profile_count[0]) / profile_mean.shape[0]
        return profile_fraction

    def get_fractional_uniformity(self, dcm):
        """Get fractional uniformity

        Args:
            dcm (pydicom.FileDataset): DICOM image object

        Returns:
            tuple: values of horizontal and vertical fractional uniformity
        """

        # 1. Determine shape of phantom - depends on orientation
        arr = dcm.pixel_array
        orientation = get_image_orientation(dcm.ImageOrientationPatient)

        # Locate phantom centre - based on shape detected
        x, y = self.get_object_centre(arr, orientation)
        logger.debug(f"Phantom centre coordinates are {x, y}")

        # Create central 10x10 ROI and get modal value
        central_roi = arr[(y - 5) : (y + 5), (x - 5) : (x + 5)]
        mode_result = stats.mode(central_roi, axis=None, keepdims=False)
        central_roi_mode = mode_result.mode
        logger.debug(f"Modal value in central ROI is {central_roi_mode}")

        # Create 160x10 pixel profiles (horizontal and vertical, centred at x,y)
        horizontal_roi = arr[(y - 5) : (y + 5), (x - 80) : (x + 80)]
        vertical_roi = arr[(y - 80) : (y + 80), (x - 5) : (x + 5)]

        # Count how many elements are within 0.9-1.1 times the modal value
        # Calculate fractional uniformity
        logger.debug(f"Calculating fractional uniformity along profiles")
        fractional_uniformity_horizontal = self.get_profile_fraction(
            horizontal_roi, central_roi_mode, direction=0
        )
        fractional_uniformity_vertical = self.get_profile_fraction(
            vertical_roi, central_roi_mode, direction=1
        )

        if self.report:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection

            fig, ax = plt.subplots()
            rects = [
                Rectangle(
                    (x - 5, y - 5),
                    10,
                    10,
                    facecolor="None",
                    edgecolor="red",
                    linewidth=3,
                ),
                Rectangle(
                    (x - 80, y - 5), 160, 10, facecolor="None", edgecolor="green"
                ),
                Rectangle(
                    (x - 5, y - 80), 10, 160, facecolor="None", edgecolor="yellow"
                ),
            ]
            pc = PatchCollection(rects, match_original=True)
            ax.imshow(arr, cmap="gray")
            ax.add_collection(pc)
            ax.scatter(x, y, 5)

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return fractional_uniformity_horizontal, fractional_uniformity_vertical
