"""
ACR Uniformity

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates uniformity for slice 7 of the ACR phantom.

This script calculates the integral uniformity in accordance with the ACR Guidance.
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
from hazenlib.acr_tools import ACRTools


class ACRUniformity(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = None

    def run(self) -> dict:
        results = {}
        self.ACR_obj = ACRTools(self.data)
        uniformity_dcm = self.ACR_obj.dcm[6]
        try:
            result = self.get_integral_uniformity(uniformity_dcm)
            results[self.key(uniformity_dcm)] = result

            results['reports'] = {'images': self.report_files}
        except Exception as e:
            print(
                f"Could not calculate the percent integral uniformity for {self.key(uniformity_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        return results

    def get_integral_uniformity(self, dcm):
        # Calculate the integral uniformity in accordance with ACR guidance.
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        r_large = np.ceil(80 / res[0]).astype(int)  # Required pixel radius to produce ~200cm2 ROI
        r_small = np.ceil(np.sqrt(100 / np.pi) / res[0]).astype(int)  # Required pixel radius to produce ~1cm2 ROI
        d_void = np.ceil(5 / res[0]).astype(int)  # Offset distance for rectangular void at top of phantom
        dims = img.shape  # Dimensions of image

        cxy = self.ACR_obj.find_phantom_center(img)
        base_mask = self.ACR_obj.circular_mask([cxy[0], cxy[1] + d_void], r_small, dims)  # Dummy circular mask at
        # centroid
        coords = np.nonzero(base_mask)  # Coordinates of mask

        lroi = self.ACR_obj.circular_mask([cxy[0], cxy[1] + d_void], r_large, dims)
        img_masked = lroi * img

        lroi_rows, lroi_cols = np.nonzero(lroi)[0], np.nonzero(lroi)[1]

        mean_val = np.zeros(lroi_rows.shape)
        mean_array = np.zeros(img_masked.shape)

        for idx, (row, col) in enumerate(zip(lroi_rows, lroi_cols)):
            centre = [row, col]  # Extract coordinates of new mask centre within large ROI
            trans_mask = [coords[0] + centre[0] - cxy[0] - d_void,
                          coords[1] + centre[1] - cxy[1]]  # Translate mask within the limits of the large ROI
            sroi_val = img_masked[trans_mask[0], trans_mask[1]]  # Extract values within translated mask
            if np.count_nonzero(sroi_val) < np.count_nonzero(base_mask):
                mean_val[idx] = 0
            else:
                mean_val[idx] = np.mean(sroi_val[np.nonzero(sroi_val)])
            mean_array[row, col] = mean_val[idx]

        sig_max = np.max(mean_val)
        sig_min = np.min(mean_val[np.nonzero(mean_val)])

        max_loc = np.where(mean_array == sig_max)
        min_loc = np.where(mean_array == sig_min)

        piu = 100 * (1 - (sig_max - sig_min) / (sig_max + sig_min))

        piu = np.round(piu, 2)
        if self.report:
            import matplotlib.pyplot as plt
            theta = np.linspace(0, 2 * np.pi, 360)
            fig = plt.figure()
            fig.set_size_inches(8, 8)
            plt.imshow(img)

            plt.scatter([max_loc[1], min_loc[1]], [max_loc[0], min_loc[0]], c='red', marker='x')
            plt.plot(r_small * np.cos(theta) + max_loc[1], r_small * np.sin(theta) + max_loc[0], c='yellow')
            plt.annotate('Min = ' + str(np.round(sig_min, 1)), [min_loc[1], min_loc[0] + 10 / res[0]], c='white')

            plt.plot(r_small * np.cos(theta) + min_loc[1], r_small * np.sin(theta) + min_loc[0], c='yellow')
            plt.annotate('Max = ' + str(np.round(sig_max, 1)), [max_loc[1], max_loc[0] + 10 / res[0]], c='white')
            plt.plot(r_large * np.cos(theta) + cxy[1], r_large * np.sin(theta) + cxy[0] + 5 / res[1], c='black')
            plt.axis('off')
            plt.title('Percent Integral Uniformity = ' + str(np.round(piu, 2)) + '%')
            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return piu
