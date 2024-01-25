import os
import sys
import traceback

import numpy as np
import cv2 as cv

from hazenlib.utils import get_pe_direction, get_pixel_size, rescale_to_byte
from hazenlib.logger import logger
from hazenlib.HazenTask import HazenTask


class Ghosting(HazenTask):
    """Ghosting measurement class for DICOM images of the MagNet phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.single_dcm = self.dcm_list[0]

    def run(self) -> dict:
        """Main function for performing ghosting measurement

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM SeriesDescription + EchoTime + NumberOfAverages, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        results = self.init_result_dict()
        img_desc = self.img_desc(
            self.single_dcm,
            properties=["SeriesDescription", "EchoTime", "NumberOfAverages"],
        )
        results["file"] = img_desc
        print("------------------------------")
        print(img_desc)

        try:
            ghosting_value = self.get_ghosting(self.single_dcm)
            results["measurement"] = {"ghosting %": round(ghosting_value, 3)}

        except Exception as e:
            print(f"Could not calculate the ghosting for {img_desc} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_signal_bounding_box(self, array: np.ndarray):
        """Define coordinates of bounding box around area with top 25% signal strength (25% highest pixel values)

        Args:
            array (np.ndarray): pixel array

        Returns:
            tuple of int: y_min, y_max, x_min, x_max
        """
        # Find highest pixel value
        max_signal = np.max(array)

        # Create threshold of top 25% signal strength
        signal_limit = max_signal * 0.4
        signal = []
        for idx, voxel in np.ndenumerate(array):
            if voxel > signal_limit:
                signal.append(idx)

        signal_column = sorted([voxel[1] for voxel in signal])
        signal_row = sorted([voxel[0] for voxel in signal])

        upper_row = min(signal_row)  #  11 row_min 220 y
        lower_row = max(signal_row)  #  93 row_max 177
        left_column = min(signal_column)  # 217  col_min 25 x
        right_column = max(signal_column)  # 299  col_max 69

        # Create array of pixel coordinates where their value is above threshold
        threshold_array = np.argwhere(array >= signal_limit)

        # Record coordinate of the
        y_min, x_min = threshold_array.min(axis=0)  # row
        y_max, x_max = threshold_array.max(axis=0)  # row, col

        print("x_min, x_max, y_min, y_max")
        print(x_min, x_max, y_min, y_max)

        print("left_column, right_column, upper_row, lower_row")
        print(left_column, right_column, upper_row, lower_row)

        return (
            left_column,
            right_column,
            upper_row,
            lower_row,
        )

    def get_pe_direction(self, dcm):
        return dcm.InPlanePhaseEncodingDirection

    def get_background_roi_centres(self, dcm, pe, signal_centre):
        """Determine the background ROI centre coordinates with respect to the signal and PE direction

        Args:
            dcm (pydicom.Dataset): DICOM image object
            signal_centre (list): x, y coordinates of the centre

        Returns:
            list of tuple of int: x, y coordinates of the centre of background regions of interest
        """

        if pe == "ROW":
            # phase encoding is left -right i.e. increases with columns

            # Determine if phantom is in top or bottom half
            if signal_centre[1] < dcm.Rows * 0.5:
                # phantom is in top half of image
                print("phantom is in top half of image")
                background_rois_row = round(dcm.Rows * 0.75)  # in the bottom quadrant
            else:
                # phantom is in bottom half of image
                print("phantom is in bottom half of image")
                background_rois_row = round(dcm.Rows * 0.25)  # in the top quadrant

            # Determine if phantom is in left or right half
            if signal_centre[0] > round(dcm.Columns / 2):
                # phantom is in right half of image
                print("phantom is in right half of image")
                # need 4 ROIs evenly spaced from 0->background_roi[0]
                gap = round(signal_centre[0] / 4)
                background_rois = [
                    (signal_centre[0] - i * gap, background_rois_row) for i in range(4)
                ]
            else:
                # phantom is in left half of image
                print("phantom is in left half of image")
                # need 4 ROIs evenly spaced from background_roi[0]->end
                gap = round((dcm.Columns - signal_centre[0]) / 4)
                background_rois = [
                    (signal_centre[0] + i * gap, background_rois_row) for i in range(4)
                ]

        else:  # phase encoding is top-down i.e. increases with rows (y-axis)
            # Determine if phantom is in left or right half
            if signal_centre[0] < dcm.Columns * 0.5:
                # phantom is in left half of image
                print("phantom is in left half of image")
                background_rois_column = round(dcm.Columns * 0.75)
                # in the right quadrant
            else:
                # phantom is right half of image
                print("phantom is in right half of image")
                background_rois_column = round(dcm.Columns * 0.25)
                # in the top quadrant

            if signal_centre[1] >= round(dcm.Rows / 2):
                # phantom is bottom half of image
                print("phantom is bottom half of image")
                # need 4 ROIs evenly spaced from 0->background_roi[0]
                gap = round(signal_centre[1] / 4)
                background_rois = [
                    (background_rois_column, signal_centre[1] - i * gap)
                    for i in range(4)
                ]
            else:
                # phantom is top half of image
                print("phantom is top half of image")
                # need 4 ROIs evenly spaced from background_roi[0]->end
                gap = round((dcm.Columns - signal_centre[1]) / 4)
                background_rois = [
                    (background_rois_column, signal_centre[1] + i * gap)
                    for i in range(4)
                ]

        print("background_roi_centres")
        print(background_rois)
        return background_rois

    def get_background_slices(self, background_rois, slice_radius=5):
        """Set location of background ROIs

        Args:
            background_rois (list): list of pixel arrays (np.array)
            slice_radius (int, optional): _description_. Defaults to 5.

        Returns:
            list of np.ndarray: _description_
        """
        slices = [
            (
                np.array(
                    range(roi[0] - slice_radius, roi[0] + slice_radius), dtype=np.intp
                )[:, np.newaxis],
                np.array(
                    range(roi[1] - slice_radius, roi[1] + slice_radius), dtype=np.intp
                ),
            )
            for roi in background_rois
        ]
        # print("Corner coordinates of the first ROI")
        # print(background_rois[0][0] - slice_radius)
        # print(background_rois[0][0] + slice_radius)
        # print(background_rois[0][1] - slice_radius)
        # print(background_rois[0][1] + slice_radius)

        return slices

    def get_eligible_area(self, dcm, pe, signal_bounding_box, slice_radius=5):
        """Get pixel array within ROI from image

        Args:
            signal_bounding_box (_type_): _description_
            dcm (pydicom.Dataset): DICOM image object
            slice_radius (int, optional): _description_. Defaults to 5.

        Returns:
            tuple of lists: corresponding to eligible_columns, eligible_rows
        """
        left_column, right_column, upper_row, lower_row = signal_bounding_box

        # take into account when phantom is off edge of image
        lower_row = min(dcm.Rows - slice_radius, lower_row)
        upper_row = max(slice_radius, upper_row)
        right_column = min(dcm.Columns - slice_radius, right_column)
        left_column = max(slice_radius, left_column)

        padding_from_box = 30  # pixels

        if pe == "ROW":
            if left_column < dcm.Columns / 2:
                # signal is in left half
                eligible_columns = range(
                    right_column + padding_from_box, dcm.Columns - slice_radius
                )
                eligible_rows = range(upper_row, lower_row)
                ghost_slice = np.array(
                    range(right_column + padding_from_box, dcm.Columns - slice_radius),
                    dtype=np.intp,
                )[:, np.newaxis], np.array(range(upper_row, lower_row))
            else:
                # signal is in right half
                eligible_columns = range(slice_radius, left_column - padding_from_box)
                eligible_rows = range(upper_row, lower_row)
                ghost_slice = np.array(
                    range(slice_radius, left_column - padding_from_box), dtype=np.intp
                )[:, np.newaxis], np.array(range(upper_row, lower_row))

        else:
            if upper_row < dcm.Rows / 2:
                # signal is in top half
                eligible_rows = range(
                    lower_row + padding_from_box, dcm.Rows - slice_radius
                )
                eligible_columns = range(left_column, right_column)
                ghost_slice = np.array(
                    range(lower_row + padding_from_box, dcm.Rows - slice_radius),
                    dtype=np.intp,
                )[:, np.newaxis], np.array(range(left_column, right_column))
            else:
                # signal is in bottom half
                eligible_rows = range(slice_radius, upper_row - padding_from_box)
                eligible_columns = range(left_column, right_column)
                ghost_slice = np.array(
                    range(slice_radius, upper_row - padding_from_box), dtype=np.intp
                )[:, np.newaxis], np.array(range(left_column, right_column))

        return eligible_columns, eligible_rows

    def get_ghost_slice(self, eligible_area):
        """Get array of pixel values wihtin bounding box of the ghost slice

        Args:
            eligible_area ():
            # signal_bounding_box (tuple or list): _description_
            # dcm (pydicom.Dataset): DICOM image object
            # slice_radius (int, optional): _description_. Defaults to 5.

        Returns:
            tuple of np.array: ghost_col and ghost_row
        """
        ghost_col = np.array(
            range(min(eligible_area[1]), max(eligible_area[1])), dtype=np.intp
        )[:, np.newaxis]
        ghost_row = np.array(range(min(eligible_area[0]), max(eligible_area[0])))
        return ghost_col, ghost_row

    def get_signal_slice(self, signal_centre, slice_radius=5):
        """Get coordinates of pixels where the signal + slice_radius is

        Args:
            signal_centre (tuple): left_column, right_column
            slice_radius (int, optional): _description_. Defaults to 5.

        Returns:
            tuple of np.array: indeces corresponding to the signal region
        """
        idxs = (
            np.array(
                range(signal_centre[0] - slice_radius, signal_centre[0] + slice_radius),
                dtype=np.intp,
            )[:, np.newaxis],
            np.array(
                range(signal_centre[1] - slice_radius, signal_centre[1] + slice_radius),
                dtype=np.intp,
            ),
        )
        return idxs

    def calculate_ghost_intensity(
        self, ghost: np.ndarray, phantom: np.ndarray, noise: np.ndarray
    ) -> float:
        """Calculates the ghost intensity

        References: IPEM Report 112 - Small Bottle Method
                    MagNET, Ghosting = (Sg-Sn)/(Sp-Sn) x 100%

        Args:
            ghost (np.ndarray):
            phantom (np.ndarray):
            noise (np.ndarray):

        Returns:
            float
        """

        if ghost is None or phantom is None or noise is None:
            raise Exception(
                f"At least one of ghost, phantom and noise ROIs is empty or null"
            )

        ghost_mean = np.mean(ghost)
        phantom_mean = np.mean(phantom)
        noise_mean = np.mean(noise)

        if phantom_mean < ghost_mean or phantom_mean < noise_mean:
            raise Exception(
                f"The mean phantom signal is lower than the ghost or the noise signal. This can't be the case "
            )

        return 100 * abs(ghost_mean - noise_mean) / phantom_mean

    def get_ghosting(self, dcm) -> float:
        """Calculate ghosting percentage

        Args:
            dcm (pydicom.Dataset): DICOM image object

        Returns:
            float: percentage ghosting across eligible area
        """
        pe = get_pe_direction(dcm)
        x, y = get_pixel_size(dcm)  # assume square pixels i.e. x=y
        # ROIs need to be 10mmx10mm
        slice_radius = int(10 // (2 * x))
        print(f"Slice radius is {slice_radius}")

        # locate signal (get coordinates of bounding box)
        logger.debug("get coordinates of signal bounding box")
        bbox = self.get_signal_bounding_box(dcm.pixel_array)
        left_column, right_column, upper_row, lower_row = bbox

        signal_centre = [
            (left_column + right_column) // 2,
            (upper_row + lower_row) // 2,
        ]
        signal_centre_col, signal_centre_row = signal_centre
        # TODO: determine which quadrant the signal is
        logger.debug(
            f"Coordinate of the signal centre is {signal_centre_col, signal_centre_row}"
        )
        # signal ROI
        signal_col, signal_row = self.get_signal_slice(
            signal_centre, slice_radius=slice_radius
        )
        # signal pixel values
        logger.debug("get pixel values in signal bounding box")
        phantom = dcm.pixel_array[(signal_row, signal_col)]
        # phantom2 = dcm.pixel_array[
        #     signal_centre_row - slice_radius : signal_centre_row + slice_radius,
        #     signal_centre_col - slice_radius : signal_centre_col + slice_radius,
        # ]
        # print("did I get the signal right???")
        # print(phantom == phantom2)

        # noise ROIs
        logger.debug("get centre coordinates of noise regions")
        background_roi_centres = self.get_background_roi_centres(dcm, pe, signal_centre)
        # noise pixel values
        logger.debug("get pixel values in noise regions")
        # for all 4
        noise = np.concatenate(
            [
                dcm.pixel_array[(row, col)]
                for col, row in self.get_background_slices(
                    background_roi_centres, slice_radius=slice_radius
                )
            ]
        )

        # slices = self.get_background_slices(
        #     background_roi_centres, slice_radius=slice_radius
        # )
        # noise_0 = dcm.pixel_array[(slices[0][1], slices[0][0])]
        # print("slices[0]")
        # print(slices[0][0])
        # print(slices[0][1])

        # # for first one
        # print(
        #     background_roi_centres[0][0] - slice_radius,
        #     background_roi_centres[0][0] + slice_radius,
        #     background_roi_centres[0][1] - slice_radius,
        #     background_roi_centres[0][1] + slice_radius,
        # )
        # slices0 = dcm.pixel_array[
        #     background_roi_centres[0][1]
        #     - slice_radius : background_roi_centres[0][1]
        #     + slice_radius,
        #     background_roi_centres[0][0]
        #     - slice_radius : background_roi_centres[0][0]
        #     + slice_radius,
        # ]
        # print(noise_0 == slices0)

        # ghost mask
        eligible_area = self.get_eligible_area(dcm, pe, bbox, slice_radius=slice_radius)
        ghost_col, ghost_row = self.get_ghost_slice(eligible_area)
        # ghost pixel values
        ghost = dcm.pixel_array[(ghost_col, ghost_row)]
        print(ghost.shape)

        ghosting = self.calculate_ghost_intensity(ghost, phantom, noise)

        if self.report:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            x1, x2, y1, y2 = bbox

            img = dcm.pixel_array
            img = img.astype("float64")
            # print('this is img',img)
            img *= 255.0 / img.max()
            img = rescale_to_byte(dcm.pixel_array)
            img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (255, 0, 0), 1)

            for x, y in background_roi_centres:
                #  slice_size = 10
                x1 = x - 5
                y1 = y - 5
                x2 = x + 5
                y2 = y + 5
                img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (255, 0, 0), 1)

            x1 = ghost_row.min()
            y1 = ghost_col.min()
            x2 = ghost_row.max()
            y2 = ghost_col.max()
            img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (255, 0, 0), 1)

            x1 = min(eligible_area[0])
            y1 = min(eligible_area[1])
            x2 = max(eligible_area[0])
            y2 = max(eligible_area[1])
            img = cv.rectangle(img.copy(), (x1, y1), (x2, y2), (255, 0, 0), 1)

            ax.imshow(img)
            # fig.savefig(f'{self.report_path}.png')
            img_path = os.path.realpath(
                os.path.join(
                    self.report_path,
                    f"{self.img_desc(dcm, properties=['SeriesDescription', 'EchoTime', 'NumberOfAverages'])}.png",
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return ghosting
