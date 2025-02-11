"""
ACR Slice Thickness

Calculates the slice thickness for slice 1 of the ACR phantom.

The ramps located in the middle of the phantom are located and line profiles are drawn through them. The full-width
half-maximum (FWHM) of each ramp is determined to be their length. Using the formula described in the ACR guidance, the
slice thickness is then calculated.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

31/01/2022
"""

import os
import sys
import traceback
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union, Self

import scipy
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.utils import get_image_orientation, get_dicom_files


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing slice width measurement
        using slice 1 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Identify relevant slice
        slice_thickness_dcm = self.ACR_obj.slice_stack[0]
        # TODO image may be 90 degrees cw or acw, could use code to identify which or could be added as extra arg

        ori = get_image_orientation(slice_thickness_dcm)
        if ori == 'Sagittal':
            # Get the pixel array from the DICOM file
            img = slice_thickness_dcm.pixel_array

            # Rotate the image 90 degrees clockwise

            rotated_img = np.rot90(img, k=-1)  # k=-1 for 90 degrees clockwise

            # Update the pixel array in the DICOM object
            slice_thickness_dcm.PixelData = rotated_img.tobytes()

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(slice_thickness_dcm)

        try:
            result = self.get_slice_thickness(slice_thickness_dcm)
            results["measurement"] = {"slice width mm": round(result, 2)}
        except Exception as e:
            print(
                f"Could not calculate the slice thickness for {self.img_desc(slice_thickness_dcm)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def find_ramps(self, img, centre):
        """Find ramps in the pixel array and return the co-ordinates of their location.

        Args:
            img (np.ndarray): dcm.pixel_array
            centre (list): x,y coordinates of the phantom centre

        Returns:
            tuple: x and y coordinates of ramp.
        """
        # X
        investigate_region = int(np.ceil(5.5 / self.ACR_obj.dy).item())

        if np.mod(investigate_region, 2) == 0:
            investigate_region = investigate_region + 1

        # Line profiles around the central row
        invest_x = [
            skimage.measure.profile_line(
                img, (centre[1] + k, 1), (centre[1] + k, img.shape[1]), mode="constant"
            )
            for k in range(investigate_region)
        ]

        invest_x = np.array(invest_x).T
        mean_x_profile = np.mean(invest_x, 1)
        abs_diff_x_profile = np.absolute(np.diff(mean_x_profile))

        # find the points corresponding to the transition between:
        # [0] - background and the hyperintense phantom
        # [1] - hyperintense phantom and hypointense region with ramps
        # [2] - hypointense region with ramps and hyperintense phantom
        # [3] - hyperintense phantom and background

        x_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_x_profile, 4)
        x_locs = np.sort(x_peaks) - 1

        width_pts = [x_locs[1], x_locs[2]]
        width = np.max(width_pts) - np.min(width_pts)

        # take rough estimate of x points for later line profiles
        x = np.round([np.min(width_pts) + 0.2 * width, np.max(width_pts) - 0.2 * width])

        # Y
        c = skimage.measure.profile_line(
            img,
            (centre[1] - 2 * investigate_region, centre[0]),
            (centre[1] + 2 * investigate_region, centre[0]),
            mode="constant",
        ).flatten()

        abs_diff_y_profile = np.absolute(np.diff(c))

        y_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_y_profile, 2)
        y_locs = centre[1] - 2 * investigate_region + 1 + y_peaks
        height = np.max(y_locs) - np.min(y_locs)

        y = np.round([np.max(y_locs) - 0.25 * height, np.min(y_locs) + 0.25 * height])

        return x, y

    def FWHM(self, data):
        """Calculate full width at half maximum of the line profile.

        Args:
            data (np.ndarray): slice profile curve.

        Returns:
            tuple: co-ordinates of the half-maximum points on the line profile.
        """
        baseline = np.min(data)
        data -= baseline
        # TODO create separate variable so that data value isn't being overwritten
        half_max = np.max(data) * 0.5

        # Naive attempt
        half_max_crossing_indices = np.argwhere(
            np.diff(np.sign(data - half_max))
        ).flatten()

        # Interpolation
        def simple_interp(x_start, ydata):
            """Simple interpolation - obtaining more accurate x co-ordinates.

            Args:
                x_start (int or float): x coordinate of the half maximum.
                ydata (np.ndarray): y coordinates.

            Returns:
                float: true x coordinate of the half maximum.
            """
            x_points = np.arange(x_start - 5, x_start + 6)
            # Check if expected x_pts (indices) will be out of range ( >= len(ydata))
            inrange = np.where(x_points == len(ydata))[0]
            if np.size(inrange) > 0:
                # locate index of where ydata ends within x_pts
                # crop x_pts until len(ydata)
                x_pts = x_points[: inrange.flatten()[0]]
            else:
                x_pts = x_points

            y_pts = ydata[x_pts]

            grad = (y_pts[-1] - y_pts[0]) / (x_pts[-1] - x_pts[0])

            x_true = x_start + (half_max - ydata[x_start]) / grad

            return x_true

        FWHM_pts = simple_interp(half_max_crossing_indices[0], data), simple_interp(
            half_max_crossing_indices[-1], data
        )
        return FWHM_pts

    def get_slice_thickness(self, dcm):
        """Measure slice thickness. \n
        Identify the ramps, measure the line profile, measure the FWHM, and use this to calculate the slice thickness.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: measured slice thickness.
        """
        img = dcm.pixel_array
        
        ############################
        # Added by NC to demonstrate potential improvement for line placement
        lines = self.place_lines(img)
        profiles = [line.get_profile(refImg=img) for line in lines]
        # Below is temporary code to demonstrate placed lines and obtained profiles
        for line in lines:
            plt.plot(*line)
        plt.imshow(img)
        plt.show()

        for profile in profiles:
            plt.plot(profile)
        plt.show()
        ############################
        
        cxy, _ = self.ACR_obj.find_phantom_center(img, self.ACR_obj.dx, self.ACR_obj.dy)
        x_pts, y_pts = self.find_ramps(img, cxy)

        interp_factor = 1 / 5
        interp_factor_dx = interp_factor * self.ACR_obj.dx
        sample = np.arange(1, x_pts[1] - x_pts[0] + 2)
        new_sample = np.arange(1, x_pts[1] - x_pts[0] + interp_factor, interp_factor)
        offsets = np.arange(-3, 4)
        ramp_length = np.zeros((2, 7))

        line_store = []
        fwhm_store = []
        for i, offset in enumerate(offsets):
            lines = [
                skimage.measure.profile_line(
                    img,
                    (offset + y_pts[0], x_pts[0]),
                    (offset + y_pts[0], x_pts[1]),
                    linewidth=2,
                    mode="constant",
                ).flatten(),
                skimage.measure.profile_line(
                    img,
                    (offset + y_pts[1], x_pts[0]),
                    (offset + y_pts[1], x_pts[1]),
                    linewidth=2,
                    mode="constant",
                ).flatten(),
            ]

            interp_lines = [
                scipy.interpolate.interp1d(sample, line)(new_sample) for line in lines
            ]
            fwhm = [self.FWHM(interp_line) for interp_line in interp_lines]
            ramp_length[0, i] = interp_factor_dx * np.diff(fwhm[0])
            ramp_length[1, i] = interp_factor_dx * np.diff(fwhm[1])

            line_store.append(interp_lines)
            fwhm_store.append(fwhm)

        with np.errstate(divide="ignore", invalid="ignore"):
            dz = 0.2 * (np.prod(ramp_length, axis=0)) / np.sum(ramp_length, axis=0)

        dz = dz[~np.isnan(dz)]
        # TODO check this - if it's taking the value closest to the DICOM slice thickness this is potentially not accurate?
        z_ind = np.argmin(np.abs(dcm.SliceThickness - dz))

        slice_thickness = dz[z_ind]

        if self.report:
            fig, axes = plt.subplots(4, 1)
            fig.set_size_inches(8, 24)
            fig.tight_layout(pad=4)

            x_ramp = new_sample * self.ACR_obj.dx
            x_extent = np.max(x_ramp)
            y_ramp = line_store[z_ind][1]
            y_extent = np.max(y_ramp)
            max_loc = np.argmax(y_ramp) * interp_factor_dx

            axes[0].imshow(img)
            axes[0].scatter(cxy[0], cxy[1], c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].imshow(img)
            axes[1].plot(
                [x_pts[0], x_pts[1]], offsets[z_ind] + [y_pts[0], y_pts[0]], "b-"
            )
            axes[1].plot(
                [x_pts[0], x_pts[1]], offsets[z_ind] + [y_pts[1], y_pts[1]], "r-"
            )
            axes[1].axis("off")
            axes[1].set_title("Line Profiles")

            xmin = fwhm_store[z_ind][1][0] * interp_factor_dx / x_extent
            xmax = fwhm_store[z_ind][1][1] * interp_factor_dx / x_extent

            axes[2].plot(
                x_ramp,
                y_ramp,
                "r",
                label=f"FWHM={np.round(ramp_length[1][z_ind], 2)}mm",
            )
            axes[2].axhline(
                0.5 * y_extent, linestyle="dashdot", color="k", xmin=xmin, xmax=xmax
            )
            axes[2].axvline(
                max_loc, linestyle="dashdot", color="k", ymin=0, ymax=10 / 11
            )

            axes[2].set_xlabel("Relative Position (mm)")
            axes[2].set_xlim([0, x_extent])
            axes[2].set_ylim([0, y_extent * 1.1])
            axes[2].set_title("Upper Ramp")
            axes[2].grid()
            axes[2].legend(loc="best")

            xmin = fwhm_store[z_ind][0][0] * interp_factor_dx / x_extent
            xmax = fwhm_store[z_ind][0][1] * interp_factor_dx / x_extent
            x_ramp = new_sample * self.ACR_obj.dx
            x_extent = np.max(x_ramp)
            y_ramp = line_store[z_ind][0]
            y_extent = np.max(y_ramp)
            max_loc = np.argmax(y_ramp) * interp_factor_dx

            axes[3].plot(
                x_ramp,
                y_ramp,
                "b",
                label=f"FWHM={np.round(ramp_length[0][z_ind], 2)}mm",
            )
            axes[3].axhline(
                0.5 * y_extent, xmin=xmin, xmax=xmax, linestyle="dashdot", color="k"
            )
            axes[3].axvline(
                max_loc, ymin=0, ymax=10 / 11, linestyle="dashdot", color="k"
            )

            axes[3].set_xlabel("Relative Position (mm)")
            axes[3].set_xlim([0, x_extent])
            axes[3].set_ylim([0, y_extent * 1.1])
            axes[3].set_title("Lower Ramp")
            axes[3].grid()
            axes[3].legend(loc="best")

            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm)}_slice_thickness.png"
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return slice_thickness

    def place_lines(self, img: np.ndarray) -> list["Line"]:
        """Places line on image within ramps insert.
        Works for a rotated phantom.

        Args:
            img (np.ndarray): Pixel array from DICOM image.

        Returns:
            profiles (list): A list of the two lines as Line objects.
        """
        # Normalize to uint8, enhance contast and binarize using otsu thresh

        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contrastEnhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(img_uint8)
        _, img_binary = cv2.threshold(contrastEnhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contour by x-span sort
        contours, _ = cv2.findContours(
            img_binary.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )
        contours_sorted = sorted(
            contours,
            key=lambda cont: abs(np.max(cont[:, 0, 0]) - np.min(cont[:, 0, 0])),
            reverse=True,
        )
        insertContour = contours_sorted[1]

        # Create list of Point objects for the four corners of the contour
        insertCorners = np.intp(cv2.boxPoints(cv2.minAreaRect(insertContour)))
        insertCorners = insertCorners.astype(float)
        corners = [Point(x=p[0], y=p[1]) for p in insertCorners]

        # Define short sides of contours by list of line objects
        corners = sorted(corners, key=lambda point: corners[0].get_distance_to(point))
        shortSides = [Line(p1=corners[0], p2=corners[1]), Line(p1=corners[2], p2=corners[3])]

        # Get sublines of short sides and force p1 to be higher in y
        sublines = [line.get_subline(percOfOrig=30) for line in shortSides]
        for line in sublines:
            if line.p1.y < line.p2.y:
                line.point_swap()

        # Define connecting lines
        connectingLines = [
            Line(p1=sublines[0].p1, p2=sublines[1].p1),
            Line(p1=sublines[0].p2, p2=sublines[1].p2),
        ]

        # Final lines are sublines of connecting lines
        finalLines = [line.get_subline(percOfOrig=95) for line in connectingLines]

        return finalLines


class Point:
    """Class representing a point in Cartesian Space"""

    def __init__(self, x: Union[int, float], y: Union[int, float]):
        if not isinstance(x, (int, float)):
            raise ValueError("arg x of Point.__init__ should be int or float.")
        if not isinstance(y, (int, float)):
            raise ValueError("arg y of Point.__init__ should be int or float.")
        self._xy = np.array([x, y])

    @property
    def x(self):
        """Getter for x coordinate."""
        return self._xy[0]

    @property
    def y(self):
        """Getter for y coord"""
        return self._xy[1]

    @property
    def xy(self) -> np.ndarray:
        """Getter for xy np array"""
        return self._xy

    def get_distance_to(self, other: Self) -> float:
        """Calculates distance between two point objects.

        Args:
            other (Point): Point to calculate distance to.

        Returns:
            dist (float): The distance between the points.
        """
        if not isinstance(other, type(self)):
            raise ValueError("arg other of Point.get_distance_to should be Point.")
        dist = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        return dist

    def as_int(self) -> Self:
        """Returns an instance of Point with coords mapped to int.

        Returns:
            as_int (Point): Instance of Point with coords mapped to int.
        """
        as_int = type(self)(x=round(self.x), y=round(self.y))
        return as_int

    def __add__(self, other: Self) -> Self:
        """Addition between two point objects

        Args:
            other (Point): Point to add.

        Returns:
            result (Point): Point object for summed coords
        """
        if not isinstance(other, type(self)):
            raise ValueError("arg other of Point.__add__ should be Point")
        result = type(self)(x=self.x + other.x, y=self.y + other.y)
        return result

    def __sub__(self, other: Self) -> Self:
        """Subtraction between two point objects

        Args:
            other (Point): Point to subtract.

        Returns:
            result (Point): Point object for subtracted coords
        """
        if not isinstance(other, type(self)):
            raise ValueError("arg other of Point.__sub__ should be Point")
        result = type(self)(x=self.x - other.x, y=self.y - other.y)
        return result

    def __truediv__(self, scalar: Union[int, float]) -> Self:
        """Divides point by scalar.

        Args:
            scalar (int, float): scalar to divide point by.

        Returns:
            result (Point): Point object for divided coords
        """
        if not isinstance(scalar, (int, float)):
            raise ValueError("arg scalar of Point.__truediv__ should be int or float.")
        result = type(self)(x=self.x / scalar, y=self.y / scalar)
        return result

    def __mul__(self, scalar:  Union[int, float]) -> Self:
        """Multiplies point by scalar

        Args:
            scalar (int, float): scalar to multiply point by.

        Returns:
            result (Point): Point object for multiplied coords
        """
        if not isinstance(scalar, (int, float)):
            raise ValueError("arg scalar of Point.__mul__ should be int or float.")
        result = type(self)(x=self.x * scalar, y=self.y * scalar)
        return result

    def __str__(self) -> str:
        return f"Point{(self.x, self.y)}"


class Line:
    """Class for a line in Cartesian Space"""

    def __init__(self, p1: Point, p2: Point):
        if not isinstance(p1, Point):
            raise ValueError("arg p1 of Line.__init__ should be Point.")
        if not isinstance(p2, Point):
            raise ValueError("arg p2 of Line.__init__ should be Point.")

        self._p1 = p1
        self._p2 = p2

    @property
    def p1(self) -> Point:
        """Getter for point 1 in line"""
        return self._p1

    @property
    def p2(self) -> Point:
        """Getter for point 2 in line"""
        return self._p2

    @property
    def midpoint(self) -> Point:
        """Getter for line midpoint"""
        return (self._p1 + self._p2) / 2

    def get_profile(self, refImg: np.ndarray) -> list:
        """Gets profile across line using pixel values from reference image

        Args:
            refImg (np.ndarray): Reference image for obtaining pixel values

        Returns:
            profile (list): pixel value profile across line
        """
        profile = skimage.measure.profile_line(
            image=refImg,
            src=self._p1.as_int().xy.tolist()[::-1],
            dst=self._p2.as_int().xy.tolist()[::-1],
        )
        return profile

    def point_swap(self):
        """Swaps order of points"""
        self._p1, self._p2 = self._p2, self._p1

    def get_subline(self, percOfOrig: Union[int, float]) -> Self:
        """Returns an instance of self that is reduced in length to be X percent of the original. \n
        The percentage for shrinking is determined by var percOfOrig

        Args:
            percOfOrig (int, float): Percentage of original line to shrink output line to

        Returns:
            subline (Line): subline of original line
        """
        if not isinstance(percOfOrig, (int, float)):
            raise ValueError("arg percOfOrig of Line.get_subline should be int or float.")
        if not 0 < percOfOrig <= 100:
            raise ValueError(
                f"For Line.get_subline method:\n\t   Arg perc should be in bounds (0, 100], received perc={percOfOrig}."
            )
        percOffSide = (100 - percOfOrig) / 2
        vector = self._p1 - self._p2
        p1Prime = self._p1 - vector * percOffSide / 100
        p2Prime = self._p2 + vector * percOffSide / 100
        subline = Line(p1=p1Prime, p2=p2Prime)

        return subline

    def __iter__(self) -> iter:
        x_values = [self._p1.x, self._p2.x]
        y_values = [self._p1.y, self._p2.y]
        return iter([x_values, y_values])

    def __str__(self) -> str:
        """Str representation of Line"""
        return f"Line(\n    p1={self._p1},\n    p2={self._p2}\n)"

stTask = ACRSliceThickness(input_data=get_dicom_files(r"R:\Users Public\Students\Nathan Crossley\MRI\Hazen Project\hazen\hazenlib\tasks\AA32044SP45_Ax"))
stTask.run()
