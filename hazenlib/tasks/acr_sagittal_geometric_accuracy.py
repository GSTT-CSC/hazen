"""
ACR Sagittal Geometric Accuracy
__________________________________________

Reference
_________

`ACR Large Phantom Guidance PDF <https://accreditationsupport.acr.org/helpdesk/attachments/11093487417>`_

Intro
_____

The ACR Geometric Accuracy Task has an additional task in which you obtain the Sagittal Localizer and measure the
vertical distance from top to bottom of the rectangular shape. The measurement is performed about 15 mm offset from
center per their example. I think the idea is that you can avoid any phantom positional errors that might present
towards the edges of the rectangle. At any rate, this task is a modified version of the axial task.

ACR Guidelines
______________

ACR Algorithm
+++++++++++++

    #. Display the ACR sagittal localizer image. Adjust the display window and level as described below.
    #. Measure the superior to inferior (head to foot) length of the phantom along a line close to the middle
        of the phantom as shown in Figure 2.
                                    ...
    #. Display slice 1 of the ACR T1 series. Adjust the display window and level as described below.
    #. Measure the diameter of the phantom in 2 directions: top-to-bottom and left-to-right (Figure 2).
    #. Determine the window level setting to measure slice 5 of the ACR T1 series, as described below.
        Display slice 5 with that window and level.
    #. Measure the diameter of the phantom in 4 directions: top-to-bottom, left-to-right, and both diagonals.

ACR Scoring Rubric
++++++++++++++++++

Phantom     Sagittal length     Pass/Fail Limit     Axial diameter      Pass/Fail Limit
            (mm)                (mm)                (mm)                (mm)
_______     _______________     _______________     ______________      _______________
Large       148                 148 +/- 3.0         190                 190 +/- 3.0
Medium      134                 134 +/- 2.0         165                 165 +/- 2.0

Notes
_____

.. note::

    Some MR vendors provide the ability to select gradient distortion correction at the operator console. For
    these systems be sure that the distortion correction option is turned on; leaving distortion correction off can
    cause geometric accuracy failure.

.. note::

    A common cause of failure of this test is miscalibration of one or more gradients. A mis-calibrated gradient
    causes its associated image dimension (x, y, or z) to appear longer or shorter than it really is. Mis-calibrated
    gradients also can cause slice position errors. It is normal for gradient calibration to drift over time and to
    require recalibration by the service engineer.

.. note::

    Another possible cause of failure is use of an excessively low acquisition bandwidth. It is common practice
    on low B0 field scanners and at some facilities to reduce acquisition bandwidth, especially on long TE
    acquisitions, to increase signal-to-noise ratio (SNR). This can be pushed to the point that the normal
    inhomogeneities in B0 manifest themselves as spatial distortions in the image. On most scanners the default
    bandwidth for T1-weighted acquisitions is set high enough to avoid this problem. If the geometric accuracy
    test measurements fail, and the ACR T1 series was acquired at low bandwidth, try acquiring that series again
    at a higher bandwidth to see if the problem is eliminated.

Created by Luis M. Santos, M.D.
luis.santos2@nih.gov

2/20/2025
"""

# Python imports
import os
import sys
import traceback

# Module imports
import numpy as np
from hazenlib import logger
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import Measurement


class ACRSagittalGeometricAccuracy(HazenTask):
    """Geometric accuracy measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing geometric accuracy measurement using the first and fifth slices from the
        ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name,
                input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs,
                optionally path to the generated images for visualisation.
        """
        dcm = self.ACR_obj.slice_stack[0]
        img_desc = self.img_desc(dcm)

        # Initialise results dictionary
        results = self.init_result_dict(desc=img_desc)

        try:
            lengths_1 = self.get_geometric_accuracy(dcm)
            logger.info(lengths_1)
            results.add_measurement(
                Measurement(
                    name="GeometricAccuracy",
                    value=round(lengths_1, 2),
                    subtype="Sagittal Localizer",
                    unit="mm",
                    description="Superior to Inferior",
                ),
            )
        except Exception as e:
            logger.error(
                f"Could not calculate the geometric accuracy for {self.img_desc(self.ACR_obj.slice_stack[0])} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    def write_report(self, img, dcm, length_dict, mask, cxy, offset):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1)
        fig.set_size_inches(8, 24)
        fig.tight_layout(pad=4)

        axes[0].imshow(img)
        axes[0].scatter(cxy[0], cxy[1], c="red")
        axes[0].set_title("Centroid Location")

        axes[1].imshow(mask)
        axes[1].set_title("Thresholding Result")

        axes[2].imshow(img)
        axes[2].arrow(
            cxy[0] + offset[0],
            length_dict["Vertical Extent"][0],
            1 + offset[1],
            length_dict["Vertical Extent"][-1]
            - length_dict["Vertical Extent"][0],
            color="orange",
            length_includes_head=True,
            head_width=5,
        )
        axes[2].legend(
            [
                str(np.round(length_dict["Vertical Distance"], 2)) + "mm",
            ]
        )
        axes[2].axis("off")
        axes[2].set_title("Geometric Accuracy for Slice 1")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)

    def get_geometric_accuracy(self, dcm):
        """Measures Vertical distance of the Sagittal Localizer.

        Steps
        _____

            #. Grabs slice 1
            #. Creates a mask over the phantom from the pixel array of the DICOM image.
            #. Finds center of phantom
            #. Computes offset from center.
            #. Samples a vertical line profile at that offset.
            #. Computes distance.

        Args:
            dcm (pydicom.Dataset): slice 1

        Returns:
            float: vertical distance.

        """
        img, _, _ = self.ACR_obj.get_presentation_pixels(dcm)

        cxy, _ = self.ACR_obj.find_phantom_center(
            img,
            self.ACR_obj.dx,
            self.ACR_obj.dy,
            axial=False,
        )
        offset = (int(np.round(-15 / self.ACR_obj.dx)), 0)

        length_dicts = []
        for threshold in np.linspace(0.05, 0.09, 10, endpoint=True):
            mask = self.ACR_obj.get_mask_image(
                img,
                cxy,
                mag_threshold=threshold,
            )
            length_dicts.append(
                self.ACR_obj.measure_orthogonal_lengths(
                    mask,
                    cxy,
                    v_offset=offset,
                ),
            )

        length_dict = {}
        for key in ("Vertical Distance",):
            length_dict[key] = np.mean([d[key] for d in length_dicts])

        if self.report:
            length_dict["Vertical Extent"] = length_dicts[
                len(length_dicts) // 2
            ]["Vertical Extent"]
            self.write_report(img, dcm, length_dict, mask, cxy, offset)

        return length_dict["Vertical Distance"]
