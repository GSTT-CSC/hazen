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

import os
import sys
import traceback
import numpy as np

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRGhosting(HazenTask):
    """Ghosting measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing ghosting measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(self.ACR_obj.slice_stack[6])

        try:
            result = self.get_signal_ghosting(self.ACR_obj.slice_stack[6])
            results["measurement"] = {"signal ghosting %": round(result, 3)}
        except Exception as e:
            print(
                f"Could not calculate the percent-signal ghosting for {self.img_desc(self.ACR_obj.slice_stack[6])} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_signal_ghosting(self, dcm):
        """Calculate the percentage signal ghosting (PSG). \n
        Sample signal intensity from ellipses outside the phantom in
        four directions and calculate the mean signal value within each.
        Percentage signal ghosting (PSG) is then expressed as the mean signal in these four ROIs
        as a percentage of the mean signal in a ROI in the centre of the phantom.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: percentage ghosting value.
        """
        img = dcm.pixel_array
        # Required pixel radius to produce ~200cm2 ROI
        r_large = np.ceil(80 / self.ACR_obj.dx).astype(int)
        dims = img.shape

        mask = self.ACR_obj.get_mask_image(img)
        (centre_x, centre_y), _ = self.ACR_obj.find_phantom_center(
            img, self.ACR_obj.dx, self.ACR_obj.dy
        )

        nx = np.linspace(1, dims[0], dims[0])
        ny = np.linspace(1, dims[1], dims[1])

        x, y = np.meshgrid(nx, ny)

        lroi = np.square(x - centre_x) + np.square(
            y - centre_y - np.divide(5, self.ACR_obj.dy)
        ) <= np.square(r_large)
        # Short axis diameter for an ellipse of 10cm2 with a 1:4 axis ratio
        sad = 2 * np.ceil(np.sqrt(1000 / (4 * np.pi)) / self.ACR_obj.dx)

        # WEST ELLIPSE
        # find first column in mask
        w_point = np.argwhere(np.sum(mask, 0) > 0)[0]
        # initialise centre of ellipse
        w_centre = [centre_y, np.floor(w_point / 2)]
        # edge of ellipse towards left FoV (+ tolerance)
        left_fov_to_centre = w_centre[1] - sad / 2 - 5
        # edge of ellipse towards left side of phantom (+ tolerance)
        centre_to_left_phantom = w_centre[1] + sad / 2 + 5
        if left_fov_to_centre < 0 or centre_to_left_phantom > w_point:
            diffs = [left_fov_to_centre, centre_to_left_phantom - w_point]
            ind = diffs.index(max(diffs, key=abs))
            # ellipse scaling factor
            w_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))
        else:
            w_factor = 1

        # generate ellipse mask
        w_ellipse = np.square((y - w_centre[0]) / (4 * w_factor)) + np.square(
            (x - w_centre[1]) * w_factor
        ) <= np.square(10 / self.ACR_obj.dx)

        # EAST ELLIPSE
        # find last column in mask
        e_point = np.argwhere(np.sum(mask, 0) > 0)[-1]
        # initialise centre of ellipse
        e_centre = [
            centre_y,
            e_point + np.ceil((dims[1] - e_point) / 2),
        ]
        # edge of ellipse towards right FoV (+ tolerance)
        right_fov_to_centre = e_centre[1] + sad / 2 + 5
        # edge of ellipse towards right side of phantom (+ tolerance)
        centre_to_right_phantom = e_centre[1] - sad / 2 - 5
        if right_fov_to_centre > dims[1] - 1 or centre_to_right_phantom < e_point:
            diffs = [
                dims[1] - 1 - right_fov_to_centre,
                centre_to_right_phantom - e_point,
            ]
            ind = diffs.index(max(diffs, key=abs))
            # ellipse scaling factor
            e_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))
        else:
            e_factor = 1

        # generate ellipse mask
        e_ellipse = np.square((y - e_centre[0]) / (4 * e_factor)) + np.square(
            (x - e_centre[1]) * e_factor
        ) <= np.square(10 / self.ACR_obj.dx)

        # NORTH ELLIPSE
        # find first row in mask
        n_point = np.argwhere(np.sum(mask, 1) > 0)[0]
        # initialise centre of ellipse
        n_centre = [np.round(n_point / 2), centre_x]
        # edge of ellipse towards top FoV (+ tolerance)
        top_fov_to_centre = n_centre[0] - sad / 2 - 5
        # edge of ellipse towards top side of phantom (+ tolerance)
        centre_to_top_phantom = n_centre[0] + sad / 2 + 5
        if top_fov_to_centre < 0 or centre_to_top_phantom > n_point:
            diffs = [top_fov_to_centre, centre_to_top_phantom - n_point]
            ind = diffs.index(max(diffs, key=abs))
            # ellipse scaling factor
            n_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))
        else:
            n_factor = 1

        # generate ellipse mask
        n_ellipse = np.square((y - n_centre[0]) * n_factor) + np.square(
            (x - n_centre[1]) / (4 * n_factor)
        ) <= np.square(10 / self.ACR_obj.dx)
        # SOUTH ELLIPSE
        # find last row in mask
        s_point = np.argwhere(np.sum(mask, 1) > 0)[-1]
        # initialise centre of ellipse
        s_centre = [s_point + np.round((dims[1] - s_point) / 2), centre_x]
        # edge of ellipse towards bottom FoV (+ tolerance)
        bottom_fov_to_centre = s_centre[0] + sad / 2 + 5
        # edge of ellipse towards
        centre_to_bottom_phantom = s_centre[0] - sad / 2 - 5
        if bottom_fov_to_centre > dims[0] - 1 or centre_to_bottom_phantom < s_point:
            diffs = [
                dims[0] - 1 - bottom_fov_to_centre,
                centre_to_bottom_phantom - s_point,
            ]
            ind = diffs.index(max(diffs, key=abs))
            # ellipse scaling factor
            s_factor = (sad / 2) / (sad / 2 - np.absolute(diffs[ind]))
        else:
            s_factor = 1

        s_ellipse = np.square((y - s_centre[0]) * s_factor) + np.square(
            (x - s_centre[1]) / (4 * s_factor)
        ) <= np.square(10 / self.ACR_obj.dx)

        large_roi_val = np.mean(img[np.nonzero(lroi)])
        w_ellipse_val = np.mean(img[np.nonzero(w_ellipse)])
        e_ellipse_val = np.mean(img[np.nonzero(e_ellipse)])
        n_ellipse_val = np.mean(img[np.nonzero(n_ellipse)])
        s_ellipse_val = np.mean(img[np.nonzero(s_ellipse)])

        psg = 100 * np.absolute(
            ((n_ellipse_val + s_ellipse_val) - (w_ellipse_val + e_ellipse_val))
            / (2 * large_roi_val)
        )

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
            axes[1].plot(
                r_large * np.cos(theta) + centre_x,
                r_large * np.sin(theta) + centre_y + 5 / self.ACR_obj.dy,
                c="black",
            )
            axes[1].text(
                centre_x - 3 * np.floor(10 / self.ACR_obj.dx),
                centre_y + np.floor(10 / self.ACR_obj.dy),
                "Mean = " + str(np.round(large_roi_val, 2)),
                c="white",
            )

            axes[1].plot(
                10.0 / self.ACR_obj.dx * np.cos(theta) / w_factor + w_centre[1],
                10.0 / self.ACR_obj.dx * np.sin(theta) * 4 * w_factor + w_centre[0],
                c="red",
            )
            axes[1].text(
                w_centre[1] - np.floor(10 / self.ACR_obj.dx),
                w_centre[0],
                "Mean = " + str(np.round(w_ellipse_val, 2)),
                c="white",
            )

            axes[1].plot(
                10.0 / self.ACR_obj.dx * np.cos(theta) / e_factor + e_centre[1],
                10.0 / self.ACR_obj.dx * np.sin(theta) * 4 * e_factor + e_centre[0],
                c="red",
            )
            axes[1].text(
                e_centre[1] - np.floor(30 / self.ACR_obj.dx),
                e_centre[0],
                "Mean = " + str(np.round(e_ellipse_val, 2)),
                c="white",
            )

            axes[1].plot(
                10.0 / self.ACR_obj.dx * np.cos(theta) * 4 * n_factor + n_centre[1],
                10.0 / self.ACR_obj.dx * np.sin(theta) / n_factor + n_centre[0],
                c="red",
            )
            axes[1].text(
                n_centre[1] - 5 * np.floor(10 / self.ACR_obj.dx),
                n_centre[0],
                "Mean = " + str(np.round(n_ellipse_val, 2)),
                c="white",
            )

            axes[1].plot(
                10.0 / self.ACR_obj.dx * np.cos(theta) * 4 * s_factor + s_centre[1],
                10.0 / self.ACR_obj.dx * np.sin(theta) / s_factor + s_centre[0],
                c="red",
            )
            axes[1].text(
                s_centre[1],
                s_centre[0],
                "Mean = " + str(np.round(s_ellipse_val, 2)),
                c="white",
            )

            axes[1].axis("off")
            axes[1].set_title(
                "Percent Signal Ghosting = " + str(np.round(psg, 3)) + "%"
            )
            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return psg
