"""
ACR Uniformity

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the percentage integral uniformity for slice 7 of the ACR phantom.

This script calculates the percentage integral uniformity in accordance with the ACR Guidance.
This is done by first defining a large 200cm2 ROI before placing 1cm2 ROIs at every pixel within
the large ROI. At each point, the mean of the 1cm2 ROI is calculated. The ROIs with the maximum and
minimum mean value are used to calculate the integral uniformity. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

13/01/2022
"""

import os
import sys
import traceback

import numpy as np
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.types import Measurement, Result
from hazenlib.utils import (
    compute_radius_from_area,
    create_circular_mask,
    create_circular_mean_kernel,
    create_circular_roi_at,
    detect_roi_center,
)
from matplotlib.pyplot import subplots as plt_subplots
from scipy.signal import convolve2d


class ACRUniformity(HazenTask):
    """Uniformity measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)

        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        # Required pixel radius to produce ~200cm2 ROI
        # (this produces 25000 mm2??)
        self.r_large = compute_radius_from_area(200, self.ACR_obj.dx)
        # Required pixel radius to produce ~1cm2 ROI (produces 150mm2)
        self.r_small = compute_radius_from_area(1, self.ACR_obj.dx)
        # Kernel we can use to convolve on input array to obtain an ROI mean
        self.r_small_kernel = create_circular_mean_kernel(self.r_small)
        logger.info(
            f"Generated 2D circular kernel for target => \n{self.r_small_kernel}"
        )
        # Required pixel radius to produce ~200cm2 ROI - 1cm to ensure small rois live fully within large ROI
        self.r_large_filter = self.r_large - self.r_small
        # Offset used when adding labels to plots. They display 10 mm to the bottom and right of the ROI
        self.label_x_offset = np.ceil(np.divide(10, self.ACR_obj.dx))

    def run(self) -> Result:
        """Main function for performing uniformity measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM
                Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path
                to the generated images for visualisation
        """
        # Initialise results dictionary
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = self.img_desc(self.ACR_obj.slice_stack[6])

        try:
            result = self.get_integral_uniformity(self.ACR_obj.slice_stack[6])
            results.add_measurement(
                Measurement(
                    name="Uniformity",
                    type="measured",
                    subtype="Integral uniformity",
                    unit="%",
                    value=round(result, 2),
                ),
            )
        except Exception as e:
            logger.exception(
                "Could not calculate the percent integral uniformity for %s"
                " because of : %s",
                self.img_desc(self.ACR_obj.slice_stack[6]),
                e,
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    def write_report(self, img, centre, min_roi, max_roi, piu, dcm):
        (centre_x, centre_y) = centre

        # To keep convention with the rest of this module
        # we'll unpack these variables x, y
        # with the note that x = rows and y = columns
        # which is different to the usual convention
        # of x = columns and y = rows.
        x_max, y_max, max_value = max_roi
        x_min, y_min, min_value = min_roi

        fig, axes = plt_subplots(2, 1)
        fig.set_size_inches(8, 16)
        fig.tight_layout(pad=4)

        theta = np.linspace(0, 2 * np.pi, 360)

        axes[0].imshow(img)
        axes[0].scatter(centre_y, centre_x, c="red")
        axes[0].axis("off")
        axes[0].set_title("Centroid Location")

        axes[1].imshow(img)
        axes[1].scatter([x_max, x_min], [y_max, y_min], c="red", marker="x")
        axes[1].plot(
            self.r_small * np.sin(theta) + x_max,
            self.r_small * np.cos(theta) + y_max,
            c="yellow",
        )
        axes[1].annotate(
            "Min = " + str(np.round(min_value, 1)),
            [x_min + self.label_x_offset, y_min],
            c="white",
        )

        axes[1].plot(
            self.r_small * np.sin(theta) + x_min,
            self.r_small * np.cos(theta) + y_min,
            c="yellow",
        )
        axes[1].annotate(
            "Max = " + str(np.round(max_value, 1)),
            [x_max + self.label_x_offset, y_max],
            c="white",
        )
        axes[1].plot(
            self.r_large * np.cos(theta) + centre_x,
            self.r_large * np.sin(theta) + centre_y,
            c="black",
        )
        axes[1].axis("off")
        axes[1].set_title(
            "Percent Integral Uniformity = " + str(np.round(piu, 2)) + "%"
        )

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)

    def calculate_uniformity(self, min_value, max_value):
        """Calculates the percent integral uniformity (PIU) using formula `PIU = 100 * (1 - (max - min) / (max + min))`.

        It is broken down into constituent elements to shift the math to numpy as much as possible to squeeze
        performance as much as possible.

        Args:
            min_value (int): minimum mean value.
            max_value (int): maximum mean value.

        Returns:
            float: value of integral uniformity.

        """
        subtraction = np.subtract(max_value, min_value)
        addition = np.add(max_value, min_value)
        division = np.divide(subtraction, addition)
        fraction = np.subtract(1, division)
        return np.multiply(100, fraction)

    def get_mean_roi_values(self, img, search_space_mask=None):
        """
        This method gets the mean small rois for the areas of minimum and maximum intensities.

        Below is an excerpt from the ACR Large Phantom Guidance

        For each series, the measurements are made according to the following procedure:

            #. Display slice location 7.
            #. Place a large, circular region-of-interest (ROI) on the image as shown in Figure 15.
                *. The area of the ROI depends on whether the large (200 cm2) or medium (160 cm2) phantom
                    was scanned (Table 4).
                *. This large ROI defines the boundary of the region in which the image uniformity is measured.
                *. Although the mean pixel intensity inside this ROI is not needed for the uniformity test, it is
                    used in the percent signal ghosting test (section 6.0), so it should be noted.
            #. Set the display window to its minimum, and lower the level until the entire area inside the large ROI is
            white.
                *. The goal now is to raise the level slowly until a small, roughly 1 cm2 region of dark pixels
                    develops inside the ROI. This is the region of lowest signal in the large ROI.
                *. Sometimes more than one region of dark pixels will appear. In that case, focus attention on the
                    largest dark region.
                *. In some cases, rather than having a well-defined dark region, one or more wide, poorly defined
                    dark areas or areas of mixed black and white pixels are apparent.
                *. In that case, make a visual estimate of the location of the darkest 1 cm2 portion of the largest
                    dark area should be made.
            #. Place a 1 cm2 circular ROI on the low-signal region identified in step 3.
                *. If measuring in the Medium phantom be sure that this ROI does not include any of the notch at
                the top of the phantom.
                *. Figures 16a and 17a show what typical Large and Medium phantom images look like at this
                    point.
                *. Record the mean pixel value for this 1 cm2 ROI. This is the measured low-signal value.
                *. If there is uncertainty about where to place the ROI because there is no single obviously darkest
                location, try several locations and select the one having the lowest mean pixel value.
            #. Raise the level until all but a small, roughly 1 cm2 region of white pixels remains inside the large ROI.
                *. This is the region of highest signal.
                *. Sometimes more than 1 region of white pixels will remain.
                    In that case, focus attention on the largest white region. It can happen that rather than having a
                    well-defined white region, one ends up with 1 or more diffuse areas of mixed black and white pixels.
                    In that case, make a best estimate of the location of the brightest 1 cm2 portion of the largest
                    bright area.
            #. Place a 1 cm2 circular ROI on the high-signal region identified in step 5.
                *. Record the average pixel value for this 1 cm2 ROI. This is the measured high-signal value.
                *. If there is uncertainty about where to place the ROI because there is no single obviously
                    brightest location, try several locations and select the one having the highest mean pixel value.

        Args:
            img (np.ma.MaskedArray): Large ROI pixel data.
            search_space_mask (np.ndarray): Mask delineating any restrictions on the search space used to look for the
                minima and maxima.

        Returns:
            float: value of integral uniformity.

        """
        # First, let's make sure we zero out the periphery of our large ROI so that the convolution can ignore it.
        mask = img.mask
        large_roi = img.copy()
        large_roi[mask] = 0
        # Convolve the large ROI with our kernel.
        # See https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        mean_large_roi = convolve2d(
            large_roi, self.r_small_kernel, mode="same"
        )
        # We want to bias the valid centers to be fully within the large ROI.
        # Per the ACR example, it looks like it should be biased by one radius (or two radii in our case due to implementation).
        valid_search_mask = (
            mask if search_space_mask is None else search_space_mask
        )
        mean_large_roi = np.ma.masked_array(
            mean_large_roi, mask=valid_search_mask, fill_value=0
        )
        # Find mean (which is already averaged) and x and y coordinates of that point in the averaged data.
        # Do this for both the minimum and maximum
        min_mean = mean_large_roi.min()
        x_min, y_min = detect_roi_center(
            mean_large_roi, mean_large_roi.argmin()
        )
        max_mean = mean_large_roi.max()
        x_max, y_max = detect_roi_center(
            mean_large_roi, mean_large_roi.argmax()
        )
        return x_min, y_min, min_mean, x_max, y_max, max_mean

    def get_integral_uniformity(self, dcm):
        """Calculates the percent integral uniformity (PIU) of a DICOM pixel array. \n
        Iterates with a ~1 cm^2 ROI through a ~200 cm^2 ROI inside the phantom region,
        and calculates the mean non-zero pixel value inside each ~1 cm^2 ROI. \n
        The PIU is defined as: `PIU = 100 * (1 - (max - min) / (max + min))`, where \n
        'max' and 'min' represent the maximum and minimum of the mean non-zero pixel values of each ~1 cm^2 ROI.

        Args:
            dcm (pydicom.Dataset): DICOM image object to calculate uniformity from.

        Returns:
            float: value of integral uniformity.
        """
        img = dcm.pixel_array

        (center_x, center_y), _ = self.ACR_obj.find_phantom_center(
            img, self.ACR_obj.dx, self.ACR_obj.dy
        )
        logger.info(f"Adjusted centroid to ({center_x}, {center_y})")

        logger.info("Getting large ROI in image...")
        large_roi = create_circular_roi_at(
            img, self.r_large, center_x, center_y
        )
        large_roi_valid_space = create_circular_mask(
            img, self.r_large_filter, center_x, center_y
        )

        logger.info("Getting the min and max mean ROIs in image...")
        x_min, y_min, min_value, x_max, y_max, max_value = (
            self.get_mean_roi_values(large_roi, ~large_roi_valid_space)
        )
        logger.info(f"Mean Min ROI => ({x_min}, {y_min}) = {min_value}")
        logger.info(f"Mean Max ROI => ({x_max}, {y_max}) = {max_value}")

        # Uniformity calculation
        piu = self.calculate_uniformity(min_value, max_value)
        logger.info(
            f"PIU[{piu}] = 100 * (1 - ({max_value} - {min_value}) / ({max_value} + {min_value}))"
        )

        if self.report:
            logger.info("Writing report ... ")
            self.write_report(
                img,
                (center_x, center_y),
                (x_min, y_min, min_value),
                (x_max, y_max, max_value),
                piu,
                dcm,
            )

        return piu
