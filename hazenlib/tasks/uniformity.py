"""
Uniformity + Ghosting & Distortion

Calculates uniformity for a single-slice image of a uniform MRI phantom

This script implements the IPEM/MAGNET method of measuring fractional uniformity.
It also calculates integral uniformity using a 75% area FOV ROI and CoV for the same ROI.

This script also measures Ghosting within a single image of a uniform phantom.
This follows the guidance from ACR for testing their large phantom.

A simple measurement of distortion is also made by comparing the height and width of the circular phantom.

Created by Neil Heraghty
neil.heraghty@nhs.net

14/05/2018

"""

import sys
import traceback
import os
import numpy as np

import hazenlib.utils
import hazenlib.exceptions as exc
from hazenlib.HazenTask import HazenTask


class Uniformity(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.single_dcm = self.dcm_list[0]

    def run(self) -> dict:
        results = self.init_result_dict()
        results['file'] = self.img_desc(self.single_dcm)

        try:
            horizontal_uniformity, vertical_uniformity = self.get_fractional_uniformity(self.single_dcm)
            results['measurement'] = {
                "horizontal %": round(horizontal_uniformity, 2),
                "vertical %": round(vertical_uniformity, 2),
                }
        except Exception as e:
            print(f"Could test not calculate the uniformity for {self.img_desc(self.single_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def mode(self, a, axis=0):
        """
        Finds the modal value of an array. From scipy.stats.mode

        Parameters:
        ---------------
        a: array

        Returns:
        ---------------
        most_frequent: the modal value
        old_counts: the number of times this value was counted (check this)
        """
        scores = np.unique(np.ravel(a))  # get ALL unique values
        test_shape = list(a.shape)
        test_shape[axis] = 1
        old_most_frequent = np.zeros(test_shape)
        old_counts = np.zeros(test_shape)
        most_frequent = None

        for score in scores:
            template = (a == score)
            counts = np.expand_dims(np.sum(template, axis), axis)
            most_frequent = np.where(counts > old_counts, score, old_most_frequent)
            old_counts = np.maximum(counts, old_counts)
            old_most_frequent = most_frequent

        return most_frequent, old_counts

    def get_object_centre(self, dcm):
        arr = dcm.pixel_array
        shape_detector = hazenlib.utils.ShapeDetector(arr=arr)
        orientation = hazenlib.utils.get_image_orientation(dcm.ImageOrientationPatient)

        if orientation in ['Sagittal', 'Coronal']:
            # orientation is sagittal to patient
            try:
                (x, y), size, angle = shape_detector.get_shape('rectangle')
            except exc.ShapeError:
                raise

        elif orientation == 'Transverse':
            # orientation is axial
            x, y, r = shape_detector.get_shape('circle')

        else:
            raise Exception("Direction must be Transverse, Sagittal or Coronal.")

        return int(x), int(y)

    def get_fractional_uniformity(self, dcm):

        arr = dcm.pixel_array
        x, y = self.get_object_centre(dcm)

        central_roi = arr[(y - 5):(y + 5), (x - 5):(x + 5)].flatten()
        # Create central 10x10 ROI and measure modal value

        central_roi_mode, mode_popularity = self.mode(central_roi)

        # Create 160-pixel profiles (horizontal and vertical, centred at x,y)
        horizontal_roi = arr[(y - 5):(y + 5), (x - 80):(x + 80)]
        horizontal_profile = np.mean(horizontal_roi, axis=0)
        vertical_roi = arr[(y - 80):(y + 80), (x - 5):(x + 5)]
        vertical_profile = np.mean(vertical_roi, axis=1)

        # Count how many elements are within 0.9-1.1 times the modal value
        horizontal_count = np.where(
            np.logical_and((horizontal_profile > (0.9 * central_roi_mode)), (horizontal_profile < (
                    1.1 * central_roi_mode))))
        horizontal_count = len(horizontal_count[0])
        vertical_count = np.where(np.logical_and((vertical_profile > (0.9 * central_roi_mode)), (vertical_profile < (
                1.1 * central_roi_mode))))
        vertical_count = len(vertical_count[0])

        # Calculate fractional uniformity
        fractional_uniformity_horizontal = horizontal_count / 160
        fractional_uniformity_vertical = vertical_count / 160

        if self.report:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            fig, ax = plt.subplots()
            rects = [Rectangle((x - 5, y - 5), 10, 10, facecolor="None", edgecolor='red', linewidth=3),
                     Rectangle((x - 80, y - 5), 160, 10, facecolor="None", edgecolor='green'),
                     Rectangle((x - 5, y - 80), 10, 160, facecolor="None", edgecolor='yellow')]
            pc = PatchCollection(rects, match_original=True)
            ax.imshow(arr, cmap='gray')
            ax.add_collection(pc)
            ax.scatter(x, y, 5)

            img_path = os.path.realpath(os.path.join(self.report_path,
                                            f'{self.img_desc(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return fractional_uniformity_horizontal, fractional_uniformity_vertical
