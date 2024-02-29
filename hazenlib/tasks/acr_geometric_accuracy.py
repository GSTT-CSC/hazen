"""
ACR Geometric Accuracy

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for slices 1 and 5 of the ACR phantom.

This script calculates the horizontal and vertical lengths of the ACR phantom in Slice 1 in accordance with the ACR
Guidance.
This script calculates the horizontal, vertical and diagonal lengths of the ACR phantom in Slice 5 in accordance with
the ACR Guidance.
The average distance measurement error, maximum distance measurement error and coefficient of variation of all distance
measurements is reported as recommended by IPEM Report 112, "Quality Control and Artefacts in Magnetic Resonance
Imaging".

This is done by first producing a binary mask for each respective slice. Line profiles are drawn with aid of rotation
matrices around the centre of the test object to determine each respective length. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

18/11/2022
"""

import os
import sys
import traceback
import numpy as np

import skimage.measure
import skimage.transform
import skimage.morphology

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRGeometricAccuracy(HazenTask):
    """Geometric accuracy measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing geometric accuracy measurement using the first and fifth slices from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = [
            self.img_desc(self.ACR_obj.slice_stack[0]),
            self.img_desc(self.ACR_obj.slice_stack[4]),
        ]

        try:
            lengths_1 = self.get_geometric_accuracy(0)
            results["measurement"][self.img_desc(self.ACR_obj.slice_stack[0])] = {
                "Horizontal distance": round(lengths_1[0], 2),
                "Vertical distance": round(lengths_1[1], 2),
            }
        except Exception as e:
            print(
                f"Could not calculate the geometric accuracy for {self.img_desc(self.ACR_obj.slice_stack[0])} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        try:
            lengths_5 = self.get_geometric_accuracy(4)
            results["measurement"][self.img_desc(self.ACR_obj.slice_stack[4])] = {
                "Horizontal distance": round(lengths_5[0], 2),
                "Vertical distance": round(lengths_5[1], 2),
                "Diagonal distance SW": round(lengths_5[2], 2),
                "Diagonal distance SE": round(lengths_5[3], 2),
            }
        except Exception as e:
            print(
                f"Could not calculate the geometric accuracy for {self.img_desc(self.ACR_obj.slice_stack[4])} because of : {e}"
            )

            traceback.print_exc(file=sys.stdout)

        L = lengths_1 + lengths_5

        mean_err, max_err, cov_l = self.get_distortion_metrics(L)

        results["measurement"]["distortion"] = {
            "Mean relative measurement error": round(mean_err, 2),
            "Max absolute measurement error": round(max_err, 2),
            "Coefficient of variation %": round(cov_l, 2),
        }

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_geometric_accuracy(self, slice_index):
        """Measure geometric accuracy for input slice. \n
        Creates a mask over the phantom from the pixel array of the DICOM image.
        Uses the centre and shape of the mask to determine horizontal and vertical lengths,
        and also diagonal lengths in slice 5.

        Args:
            slice_index (int): the index of the slice position, for example slice 5 is at index 4.

        Returns:
            tuple of float: horizontal and vertical distances.
        """
        img_dcm = self.ACR_obj.slice_stack[slice_index]
        img = img_dcm.pixel_array
        mask = self.ACR_obj.get_mask_image(img)
        cxy, _ = self.ACR_obj.find_phantom_center(img, self.ACR_obj.dx, self.ACR_obj.dy)

        length_dict = self.ACR_obj.measure_orthogonal_lengths(mask, slice_index)
        if slice_index == 4:
            sw_dict, se_dict = self.diagonal_lengths(mask, cxy, 4)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1)
            fig.set_size_inches(8, 24)
            fig.tight_layout(pad=4)

            if slice_index == 0:
                axes[0].imshow(img)
                axes[0].scatter(cxy[0], cxy[1], c="red")
                axes[0].set_title("Centroid Location")

                axes[1].imshow(mask)
                axes[1].set_title("Thresholding Result")

                axes[2].imshow(img)
                axes[2].arrow(
                    length_dict["Horizontal Extent"][0],
                    cxy[1],
                    length_dict["Horizontal Extent"][-1]
                    - length_dict["Horizontal Extent"][0],
                    1,
                    color="blue",
                    length_includes_head=True,
                    head_width=5,
                )
                axes[2].arrow(
                    cxy[0],
                    length_dict["Vertical Extent"][0],
                    1,
                    length_dict["Vertical Extent"][-1]
                    - length_dict["Vertical Extent"][0],
                    color="orange",
                    length_includes_head=True,
                    head_width=5,
                )
                axes[2].legend(
                    [
                        str(np.round(length_dict["Horizontal Distance"], 2)) + "mm",
                        str(np.round(length_dict["Vertical Distance"], 2)) + "mm",
                    ]
                )
                axes[2].axis("off")
                axes[2].set_title("Geometric Accuracy for Slice 1")

                img_path = os.path.realpath(
                    os.path.join(self.report_path, f"{self.img_desc(img_dcm)}.png")
                )
                fig.savefig(img_path)
                self.report_files.append(img_path)

            if slice_index == 4:
                axes[0].imshow(img)
                axes[0].scatter(cxy[0], cxy[1], c="red")
                axes[0].axis("off")
                axes[0].set_title("Centroid Location")

                axes[1].imshow(mask)
                axes[1].axis("off")
                axes[1].set_title("Thresholding Result")

                axes[2].imshow(img)
                axes[2].arrow(
                    length_dict["Horizontal Extent"][0],
                    cxy[1],
                    length_dict["Horizontal Extent"][-1]
                    - length_dict["Horizontal Extent"][0],
                    1,
                    color="blue",
                    length_includes_head=True,
                    head_width=5,
                )
                axes[2].arrow(
                    cxy[0],
                    length_dict["Vertical Extent"][0],
                    1,
                    length_dict["Vertical Extent"][-1]
                    - length_dict["Vertical Extent"][0],
                    color="orange",
                    length_includes_head=True,
                    head_width=5,
                )
                axes[2].arrow(
                    se_dict["Start"][0],
                    se_dict["Start"][1],
                    se_dict["Extent"][0],
                    se_dict["Extent"][1],
                    color="purple",
                    length_includes_head=True,
                    head_width=5,
                )
                axes[2].arrow(
                    sw_dict["Start"][0],
                    sw_dict["Start"][1],
                    sw_dict["Extent"][0],
                    sw_dict["Extent"][1],
                    color="yellow",
                    length_includes_head=True,
                    head_width=5,
                )

                axes[2].legend(
                    [
                        str(np.round(length_dict["Horizontal Distance"], 2)) + "mm",
                        str(np.round(length_dict["Vertical Distance"], 2)) + "mm",
                        str(np.round(sw_dict["Distance"], 2)) + "mm",
                        str(np.round(se_dict["Distance"], 2)) + "mm",
                    ]
                )
                axes[2].axis("off")
                axes[2].set_title("Geometric Accuracy for Slice 5")

                img_path = os.path.realpath(
                    os.path.join(self.report_path, f"{self.img_desc(img_dcm)}.png")
                )
                fig.savefig(img_path)
                self.report_files.append(img_path)

        if slice_index == 4:
            return (
                length_dict["Horizontal Distance"],
                length_dict["Vertical Distance"],
                sw_dict["Distance"],
                se_dict["Distance"],
            )
        else:
            return length_dict["Horizontal Distance"], length_dict["Vertical Distance"]

    def diagonal_lengths(self, img, cxy, slice_index):
        """Measure diagonal lengths. \n
        Rotates the pixel array by 45° and measures the horizontal and vertical distances.

        Args:
            img (np.ndarray): pixel array of the slice (dcm.pixel_array).
            cxy (tuple): x,y coordinates of the circle centre.
            slice_index (int): index of the slice number.

        Returns:
            tuple of dictionaries: for both the south-east (SE) diagonal length and the south-west (SW) diagonal length: \n
            "start" and "end" indicate the start and end x and y positions of the lengths; "Extent" is the distance (in
            pixels) of the lengths; "Distance" is "Extent" with factors applied to convert from pixels to mm.
        """
        # Calculate geometric mean of the x and y pixel spacing components,
        # due to the possibility of pixels being rectangular,
        # ie. the length and width of pixels can differ.
        eff_res = np.sqrt(np.mean(np.square((self.ACR_obj.dx, self.ACR_obj.dy))))
        img_rotate = skimage.transform.rotate(img, 45, center=(cxy[0], cxy[1]))

        length_dict = self.ACR_obj.measure_orthogonal_lengths(img_rotate, slice_index)
        extent_h = length_dict["Horizontal Extent"]

        origin = (cxy[0], cxy[1])
        start = (extent_h[0], cxy[1])
        end = (extent_h[-1], cxy[1])
        se_x_start, se_y_start = ACRObject.rotate_point(origin, start, 45)
        se_x_end, se_y_end = ACRObject.rotate_point(origin, end, 45)

        dist_se = (
            np.sqrt(np.sum(np.square([se_x_end - se_x_start, se_y_end - se_y_start])))
            * eff_res
        )
        se_dict = {
            "Start": (se_x_start, se_y_start),
            "End": (se_x_end, se_y_end),
            "Extent": (se_x_end - se_x_start, se_y_end - se_y_start),
            "Distance": dist_se,
        }

        extent_v = length_dict["Vertical Extent"]

        start = (cxy[0], extent_v[0])
        end = (cxy[0], extent_v[-1])
        sw_x_start, sw_y_start = ACRObject.rotate_point(origin, start, 45)
        sw_x_end, sw_y_end = ACRObject.rotate_point(origin, end, 45)

        dist_sw = (
            np.sqrt(np.sum(np.square([sw_x_end - sw_x_start, sw_y_end - sw_y_start])))
            * eff_res
        )
        sw_dict = {
            "Start": (sw_x_start, sw_y_start),
            "End": (sw_x_end, sw_y_end),
            "Extent": (sw_x_end - sw_x_start, sw_y_end - sw_y_start),
            "Distance": dist_sw,
        }

        return sw_dict, se_dict

    @staticmethod
    def get_distortion_metrics(L):
        """Calculates the mean error, the maximum error and the coefficient of
        variation between the horizontal and vertical distances
        measured on slices 1 and 5.

        Args:
            L (tuple): horizontal and vertical distances from slices 1 and 5.

        Returns:
            tuple of floats: mean_err, max_err, cov_l
        """
        err = [x - 190 for x in L]
        mean_err = np.mean(err)

        max_err = np.max(np.absolute(err))
        cov_l = 100 * np.std(L) / np.mean(L)

        return mean_err, max_err, cov_l
