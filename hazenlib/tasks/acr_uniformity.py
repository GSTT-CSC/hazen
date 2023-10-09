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

import sys
import traceback
import os
import numpy as np

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRUniformity(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)


    def run(self) -> dict:
        # Initialise results dictionary
        results = self.init_result_dict()
        results['file'] = self.img_desc(self.ACR_obj.slice7_dcm)

        try:
            result = self.get_integral_uniformity(self.ACR_obj.slice7_dcm)
            results['measurement'] = {
                "integral uniformity %": round(result, 2)
                }
        except Exception as e:
            print(
                f"Could not calculate the percent integral uniformity for"
                f"{self.img_desc(self.ACR_obj.slice7_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def get_integral_uniformity(self, dcm):
        # Calculate the integral uniformity in accordance with ACR guidance.
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        r_large = np.ceil(80 / res[0]).astype(int)  # Required pixel radius to produce ~200cm2 ROI
        r_small = np.ceil(np.sqrt(100 / np.pi) / res[0]).astype(int)  # Required pixel radius to produce ~1cm2 ROI
        d_void = np.ceil(5 / res[0]).astype(int)  # Offset distance for rectangular void at top of phantom
        dims = img.shape  # Dimensions of image

        cxy = self.ACR_obj.centre
        base_mask = ACRObject.circular_mask((cxy[0], cxy[1] + d_void), r_small, dims)  # Dummy circular mask at
        # centroid
        coords = np.nonzero(base_mask)  # Coordinates of mask

        lroi = self.ACR_obj.circular_mask([cxy[0], cxy[1] + d_void], r_large, dims)
        img_masked = lroi * img
        half_max = np.percentile(img_masked[np.nonzero(img_masked)], 50)

        min_image = img_masked * (img_masked < half_max)
        max_image = img_masked * (img_masked > half_max)

        min_rows, min_cols = np.nonzero(min_image)[0], np.nonzero(min_image)[1]
        max_rows, max_cols = np.nonzero(max_image)[0], np.nonzero(max_image)[1]

        mean_array = np.zeros(img_masked.shape)

        def uniformity_iterator(masked_image, sample_mask, rows, cols):
            coords = np.nonzero(sample_mask)  # Coordinates of mask
            for idx, (row, col) in enumerate(zip(rows, cols)):
                centre = [row, col]
                translate_mask = [coords[0] + centre[0] - cxy[0] - d_void,
                                  coords[1] + centre[1] - cxy[1]]
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
            axes[0].scatter(cxy[0], cxy[1], c='red')
            axes[0].axis('off')
            axes[0].set_title('Centroid Location')

            axes[1].imshow(img)
            axes[1].scatter([max_loc[1], min_loc[1]], [max_loc[0], min_loc[0]], c='red', marker='x')
            axes[1].plot(r_small * np.cos(theta) + max_loc[1], r_small * np.sin(theta) + max_loc[0], c='yellow')
            axes[1].annotate('Min = ' + str(np.round(sig_min, 1)), [min_loc[1], min_loc[0] + 10 / res[0]], c='white')

            axes[1].plot(r_small * np.cos(theta) + min_loc[1], r_small * np.sin(theta) + min_loc[0], c='yellow')
            axes[1].annotate('Max = ' + str(np.round(sig_max, 1)), [max_loc[1], max_loc[0] + 10 / res[0]], c='white')
            axes[1].plot(r_large * np.cos(theta) + cxy[1], r_large * np.sin(theta) + cxy[0] + 5 / res[1], c='black')
            axes[1].axis('off')
            axes[1].set_title('Percent Integral Uniformity = ' + str(np.round(piu, 2)) + '%')

            img_path = os.path.realpath(os.path.join(
                self.report_path, f'{self.img_desc(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return piu
