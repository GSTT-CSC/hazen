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
import pydicom

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from scipy.ndimage import center_of_mass, shift as nd_shift

class ACRUniformity(HazenTask):
    """Uniformity measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def detect_best_rotation(self, ref_img):
        """
        Detects the best rotation/flip to align the ACR phantom automatically.
        Returns a tuple: (rotation_name, rotation_function).
        """
        transforms = [
            (lambda x: x, 'identity'),
            (lambda x: np.rot90(x, 1), 'rot90'),
            (lambda x: np.rot90(x, 2), 'rot180'),
            (lambda x: np.rot90(x, 3), 'rot270'),
            (lambda x: np.fliplr(x), 'fliplr'),
            (lambda x: np.flipud(x), 'flipud'),
            (lambda x: np.fliplr(np.rot90(x, 1)), 'rot90+fliplr'),
            (lambda x: np.flipud(np.rot90(x, 1)), 'rot90+flipud'),
        ]

        ref_img = ref_img.astype(np.float64)
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min() + 1e-12)

        # Use first slice as reference.
        mov_img = self.ACR_obj.slice_stack[0].pixel_array.astype(np.float64)
        mov_img = (mov_img - mov_img.min()) / (mov_img.max() - mov_img.min() + 1e-12)

        best_mse = np.inf
        best_name = 'identity'
        best_func = lambda x: x

        for func, name in transforms:
            transformed = func(mov_img)
            # Align by centroid
            mask_ref = ref_img > 0.1
            mask_mov = transformed > 0.1
            cy_ref, cx_ref = center_of_mass(mask_ref)
            cy_mov, cx_mov = center_of_mass(mask_mov)
            shift_y = cy_ref - cy_mov
            shift_x = cx_ref - cx_mov
            aligned = nd_shift(transformed, shift=(shift_y, shift_x), order=1, mode='constant', cval=0.0)

            # Compute mean squared error.
            min_shape = tuple(np.minimum(ref_img.shape, aligned.shape))
            mse = np.mean((ref_img[:min_shape[0], :min_shape[1]] - aligned[:min_shape[0], :min_shape[1]]) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_name = name
                best_func = func

        logger.info(f"Best rotation/flip detected: {best_name} (MSE={best_mse:.4e})")
        return best_name, best_func

    def apply_rotation_to_stack(self, rotation_func):
        """
        Applies a rotation/flip to all slices in self.ACR_obj.slice_stack.
        Modifies pixel arrays in memory (safe, not written to disk).
        """
        for dcm in self.ACR_obj.slice_stack:
            rotated = rotation_func(dcm.pixel_array)
            dcm.PixelData = rotated.astype(dcm.pixel_array.dtype).tobytes()

    def run(self) -> dict:
        """Main function for performing uniformity measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """

        ref_dcm = pydicom.dcmread("tests/data/acr/Siemens/0.dcm")
        ref_img = ref_dcm.pixel_array.astype(np.float64)
        ref_img = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
        _, rotation_func = self.detect_best_rotation(ref_img)
        self.apply_rotation_to_stack(rotation_func)

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(self.ACR_obj.slice_stack[6])

        try:
            result = self.get_integral_uniformity(self.ACR_obj.slice_stack[6])
            results["measurement"] = {"integral uniformity %": round(result, 2)}
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
            results["report_image"] = self.report_files

        return results

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
        # Required pixel radius to produce ~200cm2 ROI
        r_large = np.ceil(75 / self.ACR_obj.dx).astype(int)
        # Required pixel radius to produce ~1cm2 ROI
        r_small = np.ceil(np.sqrt(100 / np.pi) / self.ACR_obj.dx).astype(int)
        # Offset distance for rectangular void at top of phantom
        d_void = np.ceil(5 / self.ACR_obj.dy).astype(int)
        dims = img.shape  # Dimensions of image

        mask = self.ACR_obj.get_mask_image(img)
        cy, cx = center_of_mass(mask)
        (centre_x, centre_y) = (cx, cy)

        # Dummy circular mask at centroid
        base_mask = ACRObject.circular_mask((centre_x, centre_y + d_void), r_small, dims)
        coords = np.nonzero(base_mask)  # Coordinates of mask

        # TODO: ensure that shifting the sampling circle centre
        # is in the correct direction by a correct factor

        # List to store the results from each small ROI
        results = []
        height,width = img.shape

        # Iterating through the large ROI with the small ROI and storing the results
        for x in range(r_small, width - r_small):
            for y in range (r_small, height - r_small):
                y_grid, x_grid = np.ogrid[:height, :width]
                mask = (x_grid - x)**2 +(y_grid - y)**2 <= r_small**2
                roi_values = img[mask]
                mean_val = np.mean(roi_values)
                results.append((x, y, mean_val))


        filtered_results = []
        for x, y, mean_val in results:
            # Distance from centre of small ROI to centre of large ROI
            distance_to_centre = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance_to_centre + r_small <= r_large:
                # Filtering small ROIs to only include those that fall completely within the larger ROI
                filtered_results.append((x, y, mean_val))
        # Get the small ROIs containing the maximum mean and minimum mean values
        max_mean_tuple = max(filtered_results, key = lambda item: item[2])
        min_mean_tuple = min(filtered_results, key=lambda item: item[2])
        max_value = max_mean_tuple[2]
        min_value = min_mean_tuple[2]
        x_max, y_max = max_mean_tuple[0], max_mean_tuple[1]
        x_min, y_min = min_mean_tuple[0], min_mean_tuple[1]
        min_loc = [x_min, y_min]
        max_loc = [x_max, y_max]

        # Uniformity calculation
        piu = 100 * (1 - (max_value - min_value) / (max_value + min_value))

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            theta = np.linspace(0, 2 * np.pi, 360)

            # centre_x = column index, centre_y = row index (from center_of_mass)
            axes[0].imshow(img)
            axes[0].scatter(centre_x, centre_y, c="red")  # x=col, y=row
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            # Prepare coordinates for max/min ROIs.
            x_max, y_max = max_loc[0], max_loc[1]
            x_min, y_min = min_loc[0], min_loc[1]

            axes[1].imshow(img)
            axes[1].scatter([x_max, x_min], [y_max, y_min], c="red", marker="x")

            axes[1].plot(
                x_max + r_small * np.cos(theta),
                y_max + r_small * np.sin(theta),
                linewidth=1,
                color="yellow",
            )
            axes[1].annotate(
                "Max = " + str(np.round(max_value, 1)),
                (x_max, y_max + (10 / self.ACR_obj.dy)),
                color="white",
            )

            axes[1].plot(
                x_min + r_small * np.cos(theta),
                y_min + r_small * np.sin(theta),
                linewidth=1,
                color="yellow",
            )
            axes[1].annotate(
                "Min = " + str(np.round(min_value, 1)),
                (x_min, y_min + (10 / self.ACR_obj.dy)),
                color="white",
            )

            circle_cx = centre_x
            circle_cy = centre_y

            axes[1].plot(
                circle_cx + r_large * np.cos(theta),
                circle_cy + r_large * np.sin(theta),
                color="black",
                linewidth=1.5,
            )

            axes[1].axis("off")
            axes[1].set_title("Percent Integral Uniformity = " + str(np.round(piu, 2)) + "%")

            img_path = os.path.realpath(os.path.join(self.report_path, f"{self.img_desc(dcm)}.png"))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return piu
