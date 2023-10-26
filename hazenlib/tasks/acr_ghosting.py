"""
ACR Ghosting

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the percent-signal ghosting for slice 7 of the ACR phantom.

This script calculates the percentage signal ghosting in accordance with the ACR Guidance.
This is done by first defining a large 200cm2 ROI before placing 10cm2 elliptical ROIs outside the phantom along the
cardinal directions. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

14/11/2022
"""

import sys
import traceback
import os
import numpy as np

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRGhosting(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        # Initialise results dictionary
        results = self.init_result_dict()
        results['file'] = self.img_desc(self.ACR_obj.slice7_dcm)

        try:
            result = self.get_signal_ghosting(self.ACR_obj.slice7_dcm)
            results['measurement'] = {
                "signal ghosting %": round(result, 3)
                }
        except Exception as e:
            print(f"Could not calculate the percent-signal ghosting for {self.img_desc(self.ACR_obj.slice7_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def get_signal_ghosting(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        r_large = np.ceil(80 / res[0]).astype(int)  # Required pixel radius to produce ~200cm2 ROI
        dims = img.shape

        mask = self.ACR_obj.mask_image
        cxy = self.ACR_obj.centre

        nx = np.linspace(1, dims[0], dims[0])
        ny = np.linspace(1, dims[1], dims[1])

        x, y = np.meshgrid(nx, ny)

        lroi = np.square(x - cxy[0]) + np.square(y - cxy[1] - np.divide(5, res[1])) <= np.square(r_large)
        sad = 2 * np.ceil(
            np.sqrt(1000 / (4 * np.pi)) / res[0])  # Short axis diameter for an ellipse of 10cm2 with a 1:4 axis ratio

        # WEST ELLIPSE
        w_point = np.argwhere(np.sum(mask, 0) > 0)[0]  # find first column in mask
        w_centre = [cxy[1], np.floor(w_point / 2)]  # initialise centre of ellipse
        left_fov_to_centre = w_centre[1] - sad / 2 - 5  # edge of ellipse towards left FoV (+ tolerance)
        centre_to_left_phantom = w_centre[1] + sad / 2 + 5  # edge of ellipse towards left side of phantom (+ tolerance)
        if left_fov_to_centre < 0 or centre_to_left_phantom > w_point:
            diffs = [left_fov_to_centre, centre_to_left_phantom - w_point]
            ind = diffs.index(max(diffs, key=abs))
            w_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))  # ellipse scaling factor
        else:
            w_factor = 1

        w_ellipse = np.square((y - w_centre[0]) / (4 * w_factor)) + np.square((x - w_centre[1]) * w_factor) <= \
                    np.square(10 / res[0])  # generate ellipse mask

        # EAST ELLIPSE
        e_point = np.argwhere(np.sum(mask, 0) > 0)[-1]  # find last column in mask
        e_centre = [cxy[1], e_point + np.ceil((dims[1] - e_point) / 2)]  # initialise centre of ellipse
        right_fov_to_centre = e_centre[1] + sad / 2 + 5  # edge of ellipse towards right FoV (+ tolerance)
        centre_to_right_phantom = e_centre[
                                      1] - sad / 2 - 5  # edge of ellipse towards right side of phantom (+ tolerance)
        if right_fov_to_centre > dims[1] - 1 or centre_to_right_phantom < e_point:
            diffs = [dims[1] - 1 - right_fov_to_centre, centre_to_right_phantom - e_point]
            ind = diffs.index(max(diffs, key=abs))
            e_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))  # ellipse scaling factor
        else:
            e_factor = 1

        e_ellipse = np.square((y - e_centre[0]) / (4 * e_factor)) + np.square(
            (x - e_centre[1]) * e_factor) <= np.square(
            10 / res[0])  # generate ellipse mask

        # NORTH ELLIPSE
        n_point = np.argwhere(np.sum(mask, 1) > 0)[0]  # find first row in mask
        n_centre = [np.round(n_point / 2), cxy[0]]  # initialise centre of ellipse
        top_fov_to_centre = n_centre[0] - sad / 2 - 5  # edge of ellipse towards top FoV (+ tolerance)
        centre_to_top_phantom = n_centre[0] + sad / 2 + 5  # edge of ellipse towards top side of phantom (+ tolerance)
        if top_fov_to_centre < 0 or centre_to_top_phantom > n_point:
            diffs = [top_fov_to_centre, centre_to_top_phantom - n_point]
            ind = diffs.index(max(diffs, key=abs))
            n_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))  # ellipse scaling factor
        else:
            n_factor = 1

        n_ellipse = np.square((y - n_centre[0]) * n_factor) + np.square(
            (x - n_centre[1]) / (4 * n_factor)) <= np.square(
            10 / res[0])  # generate ellipse mask

        # SOUTH ELLIPSE
        s_point = np.argwhere(np.sum(mask, 1) > 0)[-1]  # find last row in mask
        s_centre = [s_point + np.round((dims[1] - s_point) / 2), cxy[0]]  # initialise centre of ellipse
        bottom_fov_to_centre = s_centre[0] + sad / 2 + 5  # edge of ellipse towards bottom FoV (+ tolerance)
        centre_to_bottom_phantom = s_centre[0] - sad / 2 - 5  # edge of ellipse towards
        if bottom_fov_to_centre > dims[0] - 1 or centre_to_bottom_phantom < s_point:
            diffs = [dims[0] - 1 - bottom_fov_to_centre, centre_to_bottom_phantom - s_point]
            ind = diffs.index(max(diffs, key=abs))
            s_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))  # ellipse scaling factor
        else:
            s_factor = 1

        s_ellipse = np.square((y - s_centre[0]) * s_factor) + np.square(
            (x - s_centre[1]) / (4 * s_factor)) <= np.square(
            10 / res[0])

        large_roi_val = np.mean(img[np.nonzero(lroi)])
        w_ellipse_val = np.mean(img[np.nonzero(w_ellipse)])
        e_ellipse_val = np.mean(img[np.nonzero(e_ellipse)])
        n_ellipse_val = np.mean(img[np.nonzero(n_ellipse)])
        s_ellipse_val = np.mean(img[np.nonzero(s_ellipse)])

        psg = 100 * np.absolute(
            ((n_ellipse_val + s_ellipse_val) - (w_ellipse_val + e_ellipse_val)) / (2 * large_roi_val))

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
            axes[1].plot(r_large * np.cos(theta) + cxy[0], r_large * np.sin(theta) + cxy[1] + 5 / res[1], c='black')
            axes[1].text(cxy[0] - 3 * np.floor(10 / res[0]), cxy[1] + np.floor(10 / res[1]),
                         "Mean = " + str(np.round(large_roi_val, 2)), c='white')

            axes[1].plot(10. / res[0] * np.cos(theta) / w_factor + w_centre[1],
                         10. / res[0] * np.sin(theta) * 4 * w_factor + w_centre[0], c='red')
            axes[1].text(w_centre[1] - np.floor(10 / res[0]), w_centre[0], "Mean = " + str(np.round(w_ellipse_val, 2)),
                         c='white')

            axes[1].plot(10. / res[0] * np.cos(theta) / e_factor + e_centre[1],
                         10. / res[0] * np.sin(theta) * 4 * e_factor + e_centre[0], c='red')
            axes[1].text(e_centre[1] - np.floor(30 / res[0]), e_centre[0], "Mean = " + str(np.round(e_ellipse_val, 2)),
                         c='white')

            axes[1].plot(10. / res[0] * np.cos(theta) * 4 * n_factor + n_centre[1],
                         10. / res[0] * np.sin(theta) / n_factor + n_centre[0], c='red')
            axes[1].text(n_centre[1] - 5 * np.floor(10 / res[0]), n_centre[0],
                         "Mean = " + str(np.round(n_ellipse_val, 2)),
                         c='white')

            axes[1].plot(10. / res[0] * np.cos(theta) * 4 * s_factor + s_centre[1],
                         10. / res[0] * np.sin(theta) / s_factor + s_centre[0], c='red')
            axes[1].text(s_centre[1], s_centre[0], "Mean = " + str(np.round(s_ellipse_val, 2)), c='white')

            axes[1].axis('off')
            axes[1].set_title('Percent Signal Ghosting = ' + str(np.round(psg, 3)) + '%')
            img_path = os.path.realpath(os.path.join(
                self.report_path, f'{self.img_desc(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return psg
