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

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRUniformity(HazenTask):
    """Uniformity measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing uniformity measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(self.ACR_obj.slice_stack[6])

        try:
            result = self.get_integral_uniformity(self.ACR_obj.slice_stack[6])
            results["measurement"] = {"integral uniformity %": round(result, 2)}
        except Exception as e:
            print(
                f"Could not calculate the percent integral uniformity for"
                f"{self.img_desc(self.ACR_obj.slice_stack[6])} because of : {e}"
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
        r_large = np.ceil(80 / self.ACR_obj.dx).astype(int)
        # Required pixel radius to produce ~1cm2 ROI
        r_small = np.ceil(np.sqrt(100 / np.pi) / self.ACR_obj.dx).astype(int)
        # Offset distance for rectangular void at top of phantom
        d_void = np.ceil(5 / self.ACR_obj.dx).astype(int)
        dims = img.shape  # Dimensions of image

        (centre_x, centre_y), _ = self.ACR_obj.find_phantom_center(
            img, self.ACR_obj.dx, self.ACR_obj.dy
        )
        # Dummy circular mask at centroid
        base_mask = ACRObject.circular_mask(
            (centre_x, centre_y + d_void), r_small, dims
        )
        coords = np.nonzero(base_mask)  # Coordinates of mask

        # TODO: ensure that shifting the sampling circle centre
        # is in the correct direction by a correct factor
        lroi = self.ACR_obj.circular_mask([centre_x, centre_y + d_void], r_large, dims)
        img_masked = lroi * img
        half_max = np.percentile(img_masked[np.nonzero(img_masked)], 50)

        min_image = img_masked * (img_masked < half_max)
        max_image = img_masked * (img_masked > half_max)

        min_rows, min_cols = np.nonzero(min_image)[0], np.nonzero(min_image)[1]
        max_rows, max_cols = np.nonzero(max_image)[0], np.nonzero(max_image)[1]

        mean_array = np.zeros(img_masked.shape)

        def uniformity_iterator(masked_image, sample_mask, rows, cols):
            """Iterates spatially through the pixel array with a circular ROI and calculates the mean non-zero pixel
            value within the circular ROI at each iteration.

            Args:
                masked_image (np.array): subset of pixel array.
                sample_mask (np.array): _description_.
                rows (np.array): 1D array.
                cols (np.array): 1D array.

            Returns:
                np.array: array of mean values.
            """
            # Coordinates of mask
            coords = np.nonzero(sample_mask)
            for idx, (row, col) in enumerate(zip(rows, cols)):
                centre = [row, col]
                translate_mask = [
                    coords[0] + centre[0] - centre_x - d_void,
                    coords[1] + centre[1] - centre_y,
                ]
                values = masked_image[translate_mask[0], translate_mask[1]]
                if np.count_nonzero(values) < np.count_nonzero(sample_mask):
                    mean_val = 0
                else:
                    mean_val = np.mean(values[np.nonzero(values)])

                mean_array[row, col] = mean_val

            return mean_array

        min_data = uniformity_iterator(min_image, base_mask, min_rows, min_cols)
        max_data = uniformity_iterator(max_image, base_mask, max_rows, max_cols)

        sig_max = np.max(max_data)
        sig_min = np.min(min_data[np.nonzero(min_data)])

        max_loc = np.where(max_data == sig_max)
        min_loc = np.where(min_data == sig_min)

        piu = 100 * (1 - (sig_max - sig_min) / (sig_max + sig_min))

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            theta = np.linspace(0, 2 * np.pi, 360)

            axes[0].imshow(img)
            axes[0].scatter(centre_x, centre_y, c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].imshow(img)
            axes[1].scatter(
                [max_loc[1], min_loc[1]], [max_loc[0], min_loc[0]], c="red", marker="x"
            )
            axes[1].plot(
                r_small * np.cos(theta) + max_loc[1],
                r_small * np.sin(theta) + max_loc[0],
                c="yellow",
            )
            axes[1].annotate(
                "Min = " + str(np.round(sig_min, 1)),
                [min_loc[1], min_loc[0] + 10 / self.ACR_obj.dx],
                c="white",
            )

            axes[1].plot(
                r_small * np.cos(theta) + min_loc[1],
                r_small * np.sin(theta) + min_loc[0],
                c="yellow",
            )
            axes[1].annotate(
                "Max = " + str(np.round(sig_max, 1)),
                [max_loc[1], max_loc[0] + 10 / self.ACR_obj.dx],
                c="white",
            )
            axes[1].plot(
                r_large * np.cos(theta) + centre_y,
                r_large * np.sin(theta) + centre_x + 5 / self.ACR_obj.dy,
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

        return piu
