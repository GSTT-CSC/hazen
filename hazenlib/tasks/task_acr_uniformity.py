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
import skimage.morphology

from hazenlib.HazenTask import HazenTask


class ACRUniformity(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        results = {}
        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        for dcm in self.data:
            if dcm.ImagePositionPatient[2] == z[idx_sort[6]]:
                try:
                    result = self.get_integral_uniformity(dcm)
                except Exception as e:
                    print(f"Could not calculate the percent integral uniformity for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result

        results['reports'] = {'images': self.report_files}

        return results

    def centroid_com(self, dcm):
        # Calculate centroid of object using a centre-of-mass calculation
        thresh_img = dcm > 0.25 * np.max(dcm)
        open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
        bhull = skimage.morphology.convex_hull_image(open_img)
        coords = np.nonzero(bhull)  # row major - first array is columns

        sum_x = np.sum(coords[1])
        sum_y = np.sum(coords[0])
        cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

        cxy = [cxy[0].astype(int), cxy[1].astype(int)]
        return cxy

    def circular_mask(self, centre, radius, dims):
        # Define a circular logical mask
        nx = np.linspace(1, dims[0], dims[0])
        ny = np.linspace(1, dims[1], dims[1])

        x, y = np.meshgrid(nx, ny)
        mask = np.square(x - centre[0]) + np.square(y - centre[1]) <= radius ** 2
        return mask

    def get_integral_uniformity(self, dcm):
        # Calculate the integral uniformity in accordance with ACR guidance.
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        r_large = np.ceil(80 / res[0]).astype(int)  # Required pixel radius to produce ~200cm2 ROI
        r_small = np.ceil(np.sqrt(100 / np.pi) / res[0]).astype(int)  # Required pixel radius to produce ~1cm2 ROI
        d_void = np.ceil(5 / res[0]).astype(int)  # Offset distance for rectangular void at top of phantom
        dims = img.shape  # Dimensions of image

        cxy = self.centroid_com(img)
        base_mask = self.circular_mask([cxy[0], cxy[1] + d_void], r_small, dims)  # Dummy circular mask at centroid
        coords = np.nonzero(base_mask)  # Coordinates of mask

        lroi = self.circular_mask([cxy[0], cxy[1] + d_void], r_large, dims)
        img_masked = lroi * img

        lroi_extent = np.nonzero(lroi)

        mean_val = np.zeros(lroi_extent[0].shape)
        mean_array = np.zeros(img_masked.shape)

        for ii in range(0, len(lroi_extent[0])):
            centre = [lroi_extent[0][ii], lroi_extent[1][ii]]  # Extract coordinates of new mask centre within large ROI
            trans_mask = [coords[0] + centre[0] - cxy[0] - d_void,
                          coords[1] + centre[1] - cxy[1]]  # Translate mask within the limits of the large ROI
            sroi_val = img_masked[trans_mask[0], trans_mask[1]]  # Extract values within translated mask
            if np.count_nonzero(sroi_val) < np.count_nonzero(base_mask):
                mean_val[ii] = 0
            else:
                mean_val[ii] = np.mean(sroi_val[np.nonzero(sroi_val)])
            mean_array[lroi_extent[0][ii], lroi_extent[1][ii]] = mean_val[ii]

        sig_max = np.max(mean_val)
        sig_min = np.min(mean_val[np.nonzero(mean_val)])

        max_loc = np.where(mean_array == sig_max)
        min_loc = np.where(mean_array == sig_min)

        piu = 100 * (1 - (sig_max - sig_min) / (sig_max + sig_min))

        piu = np.round(piu,2)
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
