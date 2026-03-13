"""
ACR Low-Contrast Object Detectability Task
__________________________________________

Reference
_________

`ACR Large Phantom Guidance PDF <https://accreditationsupport.acr.org/helpdesk/attachments/11093487417>`_

Intro
_____

Performs a series of Image Processing steps that attempts to

    #. Isolate the signal of the spots inside the circle ROI in slices 8 through 11.
    #. Predict where spots should be present in ROI.
    #. Walk through the predicted spot ROIs.
    #. Asses if spoke is valid.
    #. Stop at first invalid spoke.
    #. Calculate the slice score.

A lot of the tuning performed attempts to strike a balance between the real number of high intensity signals and
avoiding overestimation. As a result, my current attempt is biased towards underestimation if the signal is too
small to survive thresholding.

ACR Guidelines
______________

ACR Algorithm
+++++++++++++

    #. Display slice 11, which has the highest contrast objects. Adjust the display window width and level
        settings for best visibility of the low contrast objects. This will usually require a fairly narrow window
        width and careful adjustment of the level to best distinguish the objects from the background.
    #. Count the number of complete spokes. Begin counting with the spoke having the largest diameter
        disks; this spoke is at 12 o clock or slightly to the right of 12 o clock, and is referred to as spoke 1.
        Count clockwise from spoke 1 until a spoke is reached where one or more of the disks is not
        discernible from the background. A spoke is complete only if all three disks are discernible. Count
        complete spokes, not individual disks.
    #. The score for this slice is the number of complete spokes. Record the score.
    #. Repeat the procedure to determine the number of visible spokes for the remaining LCD images.

ACR Scoring Rubric
++++++++++++++++++

For each series, record the number of complete spokes visible on each slice, then sum the values for all four
slices to determine the total LCD score. For example, if the ACR T2 series scored 3 spokes in slice 8, 5 spokes
in slice 9, 9 spokes in slice 10, and 10 spokes in slice 11; the total score for the ACR T2 series would be 3 + 5
+ 9 + 10 = 27.

Nominal Field  ACR T1 LCD       ACR T2 LCD
Strength       Limit           Limit
               (total spokes)  (total spokes)
_____________  _______________ ______________
<1.5T           7               7
1.5T - <3T      30              25
3T              37              37

Notes
_____

The approach taken here was to do a Difference of Gaussians with several blurring and thresholding steps such that
the resulting image has most of the noise dropped. Then, we dilate whatever signal is present and hope that the predicted
mask sees the signal. This approach is very error prone with very noisy scans. As a result, we threshold such that we
only consider a very high percentile threshold point based on the available signal. Furthermore, the DoG operation tends
to introduce a rim of noise on the periphery which often blends with the outer ring of spots. Per my tuning, this is
minimized, but still exists. To avoid this artifact, I can decrease the aggressiveness of the DoG operation at the cost
of more residual noise in the final output. The reason I tried this approach first is that I was interested in
avoiding adding more dependencies and training a neural network for this task at this moment. However, current testing
suggests that I should read about blob detection algorithms and consider machine learning as well. It could be a very
productive approach long term.

The Hough Transform does not work reliably for this task. You really are looking into blob detection more so than circle
detection. DoG returns a decently cleaned image for further analysis. The result is just not easy to clean via
thresholding.

TL;DR; I definitely need to return to this task and try a cleaner approach using blob detection.

Created by Luis M. Santos, M.D.
luis.santos2@nih.gov

12/02/2025
"""

# Python imports
import os

# Module imports
import cv2
import numpy as np
from hazenlib import logger
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.utils import (
    compute_radius_from_area,
    create_circular_roi_at,
    expand_data_range,
    wait_on_parallel_results,
)
from hazenlib.types import Measurement
from matplotlib.pyplot import subplots as plt_subplots


class ACRObjectDetectability(HazenTask):
    """Uniformity measurement class for DICOM images of the ACR phantom."""

    DOT_SEPARATION = 12.8  #: 12 to 14 mm from center to center of each circle.
    DOT_ANGLE = np.deg2rad(36)  #: Each spoke is at this angle of separation.
    SLICE_ANGLE_OFFSET = np.deg2rad(9)  #: Each subsequent slice
    START_ANGLE = np.deg2rad(90)  #: Slice 0 has spots at a 90 deg angle
    ORIG_SPOKE_RADII = {
        0: 3.5,
        1: 3.1945,
        2: 2.889,
        3: 2.5835,
        4: 2.278,
        5: 1.9725,
        6: 1.667,
        7: 1.3615,
        8: 1.056,
        9: 0.75,
    }
    SPOKE_RADII = {  #: Radius used for each spoke spot. Meaning, in spoke 0 all spots have the same
        0: 2.0,  #: radius.
        1: 2.0,
        2: 2.0,
        3: 2.0,
        4: 2.0,
        5: 2.0,
        6: 2.0,
        7: 2.0,
        8: 2.0,
        9: 2.0,
    }
    BINARIZATION_THRESHOLD = {1.5: 97.8, 3.0: 97.5}
    FIRST_SLICE_NUM = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        # Required pixel radius to produce ~75cm2 ROI
        self.r_inner = compute_radius_from_area(80, self.ACR_obj.dx)
        # Required pixel radius to produce ~0.25cm2 ROI
        self.r_binarization_sample = compute_radius_from_area(
            45, self.ACR_obj.dx
        )
        # Required pixel radius to produce ~15cm2 ROI
        self.r_noise = compute_radius_from_area(45, self.ACR_obj.dx)
        # Required pixel radius to produce ~0.25cm2 ROI
        self.r_spot = compute_radius_from_area(0.25, self.ACR_obj.dx)

    def run(self) -> dict:
        """Main function for performing uniformity measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM
                Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path
                to the generated images for visualisation
        """
        slices = [
            self.ACR_obj.slice_stack[7],
            self.ACR_obj.slice_stack[8],
            self.ACR_obj.slice_stack[9],
            self.ACR_obj.slice_stack[10],
        ]
        # Initialise results dictionary
        results = self.init_result_dict(
            files=tuple([self.img_desc(sl) for sl in slices]),
        )

        r = self.get_spokes_and_scores(slices)

        if not len(r["meta"]["measurement"]):
            msg = "No low contrast object detection measurements made"
            logger.error(msg)
            raise ValueError(msg)

        for measurement, score in r["meta"]["measurement"].items():
            subtype = (
                f"slice {measurement}"
                if isinstance(measurement, int)
                else "total"
            )

            m = Measurement(
                name="LowContrastObjectDetectability",
                value=score,
                type="measured",
                subtype=subtype,
                description=(
                    f"Field Strength = {r['meta']['field_strength']}T"
                ),
            )
            results.add_measurement(m)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    def write_report(self, slices, results):
        data = results["data"]

        arg_list = [
            (slices[i], data[i], data[i]["center"]) for i in range(len(slices))
        ]
        self.report_files = wait_on_parallel_results(
            self.write_report_slice, arg_list
        )

    def write_report_slice(
        self, dcm, img_result, center, theta=np.linspace(0, 2 * np.pi, 360)
    ):
        (center_x, center_y) = center
        fig, axes = plt_subplots(4, 1)
        fig.set_size_inches(8, 16)
        fig.tight_layout(pad=4)

        # Centroid
        axes[0].imshow(img_result["img"][0], cmap="gray", vmin=0, vmax=255)
        axes[0].scatter(center_x, center_y, c="red")
        axes[0].axis("off")
        axes[0].set_title("Window Leveled + Centroid Location")

        # DoG
        axes[1].imshow(img_result["img"][1], cmap="viridis")
        axes[1].axis("off")
        axes[1].set_title("Difference of Gaussians")

        # Dilated
        axes[2].imshow(img_result["img"][2], cmap="viridis")
        axes[2].axis("off")
        axes[2].set_title("Filtered (binarized + dilated)")

        # axes[2].imshow(img_result['img'][2], cmap='gray', vmin=0, vmax=255)
        axes[3].imshow(img_result["img"][3], cmap="viridis")
        for spoke in img_result["spokes"]:
            spot_center1, spot_center2, spot_center3 = spoke["centers"]
            axes[3].plot(
                self.r_spot * np.cos(theta) + spot_center1[0],
                self.r_spot * np.sin(theta) + spot_center1[1],
                c="green",
            )
            axes[3].plot(
                self.r_spot * np.cos(theta) + spot_center2[0],
                self.r_spot * np.sin(theta) + spot_center2[1],
                c="green",
            )
            axes[3].plot(
                self.r_spot * np.cos(theta) + spot_center3[0],
                self.r_spot * np.sin(theta) + spot_center3[1],
                c="green",
            )
        axes[3].axis("off")
        axes[3].set_title("Valid Spokes (dilated)")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        return img_path

    @staticmethod
    def calculate_spot_location(center, angle, spot_separation, spot):
        x_dist, y_dist = (
            np.floor(np.cos(angle) * spot_separation * spot),
            np.floor(np.sin(angle) * spot_separation * spot),
        )
        return (center[0] + x_dist, center[1] - y_dist)

    def detect_spot(
        self, img, center, angle, spot_separation, spot_radius, spot
    ):
        x, y = self.calculate_spot_location(
            center, angle, spot_separation, spot
        )
        return create_circular_roi_at(img, spot_radius, x, y), (x, y)

    def detect_spoke(self, img, center, slice_num, spoke):
        spot_radius = int(np.ceil(self.SPOKE_RADII[spoke] / self.ACR_obj.dx))
        spot_separation = self.DOT_SEPARATION / self.ACR_obj.dx
        angle = (
            self.START_ANGLE
            - (spoke * self.DOT_ANGLE)
            - (self.SLICE_ANGLE_OFFSET * slice_num)
        )

        # Generate individual spot masks on image.
        spot1 = self.detect_spot(
            img, center, angle, spot_separation, spot_radius, 1
        )
        spot2 = self.detect_spot(
            img, center, angle, spot_separation, spot_radius, 2
        )
        spot3 = self.detect_spot(
            img, center, angle, spot_separation, spot_radius, 3
        )

        return spot1, spot2, spot3

    @staticmethod
    def combine_masks(spots, target_mask):
        # Combine the spot masks into a master spoke mask
        spot1, spot2, spot3 = spots
        combined_mask = np.ma.mask_or(
            ~spot1[0].mask, np.ma.mask_or(~spot2[0].mask, ~spot3[0].mask)
        )
        return np.ma.mask_or(combined_mask, target_mask)

    def compute_score(self, feature_data, center, slice_num):
        spoke_results = {}
        for spoke in range(10):
            spot1, spot2, spot3 = self.detect_spoke(
                feature_data, center, slice_num, spoke
            )
            spot_score = sum(
                [spot1[0].sum() > 0, spot2[0].sum() > 0, spot3[0].sum() > 0]
            )
            valid = spot_score == 3
            if valid:
                spoke_results[spoke] = {
                    "spots": spot_score,
                    "centers": [spot1[1], spot2[1], spot3[1]],
                }
            else:
                break
        return spoke_results

    def get_img_center(self, img):
        (center_x, center_y), _ = self.ACR_obj.find_phantom_center(
            img, self.ACR_obj.dx, self.ACR_obj.dy
        )
        offsetted_y = np.round(center_y + 5 / self.ACR_obj.dy)
        return (np.round(center_x - self.ACR_obj.dx), np.round(offsetted_y))

    def detect_objects(self, dcm, field_strength, slice_num):
        """Performs all preprocessing steps required to isolate the signal from background. Then, runs an ROI based
        algorithm to detect the presence of valid spokes in dataset.

        Steps
        _____

            #. Request presentation level pixel data.
            #. Finds centroid of data.
            #. Restrict workspace to the inner ROI in dataset.
            #. Computes Histogram based Window Center and Window Width
            #. Performs a small Gaussian pass to denoise the input image.
            #. Performs Difference of Gaussians (DoG) pass.
            #. Performs another small Gaussian filter pass to decimate some of the remaining noise.
            #. Performs binarization of dog based on a percentile.
            #. Dilate remaining signal since we have destroyed so much of it by this point.
            #. Detect and count spots/spokes to generate score.
            #. Return scores and relevant images.

        Notes
        _____

        ..note::

            The weird Gaussian filtering passes not associated with the DoG have subtle and profound effects on what
            signal survives all of our filtering levers.

        ..note::

            The DoG implementation used here also allows for the application of gamma correction and multiple passes.
            However, none of these options ended up being necessary on the final version of this algorithm. I made
            the implementations available as individual methods in the ACRObject class for other future image
            manipulation tasks.

        ..warning::

            The scoring algorithm is a very dumb algorithm in that it computes the center of a spot ROI and performs
            a sum on the captured pixel population. Any values > 0 are assumed to imply that the neighborhood signal
            must have come from a strong signal. However, this is only an approximation to reality and thus it is
            weak against inputs that are noisy after all of our filtering steps. As a result, it is strongly encouraged
            to provide a copy of the dilated output in the report to allow for visual inspection of the noise
            distribution. I am not clever enough to figure out how to detect this kind of noise after generating
            results.

        ..alert::

            The GE dataset present in this repository is particularly troublesome because it represents a case in which
            there is a poor SNR on a high resolution (0.5mm) 512x512 FoV. This is a problem because the ACR prescribed
            that the matrix has to be 256x256 with a resolution of 1.0mm. The main issue is if I make the algorithm
            robust to the ACR Phantom requirements, the algorithm becomes weak to the GE data's 8th slice, which is
            pure noise. The algorithm somehow detects the alignment of the noise spots and yields at least 2 valid
            spokes when that should not be the case.

        Args:
            dcm (pydicom.Dataset): DICOM image object to calculate uniformity from.
            field_strength (float): Scanner field strength parameter. Kinda needed to get the correct level of noise
                cleaning.
            slice_num (int): Index of this DICOM slice in the original sorted dicom stack.

        Returns:
            dict: Relevant results that can be reported on. These include the per slice score and intermediate images.

            .. code-block::
                :caption: Results Structure!

                {
                    'id': slice_id,
                    'img': [windowed, dog, dilated, dilated],
                    'spokes': [spoke for spoke in results.values()],
                    'score': len(results),
                    'center': (center_x, center_y)
                }
        """
        slice_id = self.FIRST_SLICE_NUM + slice_num
        logger.info(f"Processing slice # {slice_id}")

        img, rescaled, presentation = self.ACR_obj.get_presentation_pixels(dcm)

        # Step 0, let's obtain a better center
        (center_x, center_y) = self.get_img_center(rescaled)
        logger.info(
            f"Phantom centroid set to {(center_x, center_y)} for slice {slice_id}!"
        )

        # Now, we can ready the ROI on which to focus on extracting the signal
        inner_roi = create_circular_roi_at(
            rescaled, self.r_inner, center_x, center_y
        )
        inner_roi[inner_roi.mask] = 0

        # Find noise sampling ROI
        noise_roi = create_circular_roi_at(
            inner_roi, self.r_noise, center_x, center_y
        )
        noise_roi[noise_roi.mask] = 0

        # Compute the window level and width in the noise sampling area.
        center, width = self.ACR_obj.compute_center_and_width(noise_roi)
        logger.info(f"Target Windowing => {center}, {width}")

        # Apply the previously computed window settings to the inner ROI window.
        # We adjust the windowing here using the clip method which is a custom method. The clip method is not one
        # of the standard DICOM windowing methods. It is a modified linear exact method except we do not rescale the
        # window data. Rescaling the window data yields promotion of noise to the same rank as our signal.
        contrasted = self.ACR_obj.apply_window_center_width(
            inner_roi, center, width * field_strength
        )

        # First, let do a light Gaussian pass to help remove some of the crazy noise and improve SNR.
        # Now, testing against the GE test dataset with a 512x512 matrix from a 1.5T scanner, my algorithm favors a sigma
        # of 1 or higher. Testing against high resolution Philips 3T dataset favors a fractional sigma <= 0.5.
        # The issue is that the GE dataset represents a noisy 1.5T scan acquisition with enough SNR for a human
        # to make a judgement call biased towards overestimating spoke count. This same noise causes a small
        # overestimation from the algorithm on the worst case slice due to two areas of high SNR. However, that GE slice
        # should have yielded 0 valid spokes. As a result, the compromise seems to lie somewhere between a 0.3 to 0.5
        # factor which adjusted by the pixel resolution will yield a sigma > 0.5 for the high resolution GE and
        # 0.3 for the lower (1mm) resolution acquisitions.
        resolution_factor = 1 / self.ACR_obj.dx
        noise_removed = self.ACR_obj.filter_with_gaussian(
            contrasted, 0.7 * resolution_factor
        )
        noise_removed = np.ma.masked_array(
            noise_removed, mask=inner_roi.mask, fill_value=0
        )

        # Perform Difference of Gaussians to further isolate relevant pixels
        # Using a large gamma to allow high intensities to survive the DoG operation.
        # Gamma correction can have a profound effect on remaining signal just like the selection of the sigmas.
        # Gamma correction here helps with fine-tuning to approximate what I thought is reality for the GE dataset which
        # was noisier than more ideal scans. GE => gamma = 20, sigma2 = 3.5
        factor = 3 / field_strength
        dog = self.ACR_obj.filter_with_dog(
            noise_removed,
            (0.1 * factor) / self.ACR_obj.dx,
            (1.5 * factor) / self.ACR_obj.dx,
        )
        dog = self.ACR_obj.filter_with_gaussian(dog, 0.5 / self.ACR_obj.dx)
        dog = np.ma.masked_array(dog, mask=inner_roi.mask, fill_value=0)

        # Binarize the results
        # Results should be so clean at this stage that we do not need to be aggressive (97+) in the thresholding.
        # 92% is pretty safe and accurate in most cases, but the 1.5T can yield more human like results with a slightly
        # lesser percentile. I found that 91st percentile strikes an acceptable balance between surviving noise and
        # keeping the dimmer spots in the signal population.
        binarized = self.ACR_obj.binarize_image(
            dog.copy(), self.BINARIZATION_THRESHOLD.get(field_strength, 92)
        )

        # Dilate the signal that is present.
        dilated = cv2.dilate(
            binarized, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        dilated = np.ma.masked_array(
            dilated, mask=inner_roi.mask, fill_value=0
        )

        # Count spots and spokes clockwise
        # A spot is valid if its max intensity is above relative threshold
        # A spoke is valid if it contains 3 successive spots in diagonal.
        results = self.compute_score(dilated, (center_x, center_y), slice_num)

        windowed = expand_data_range(contrasted)
        windowed[inner_roi.mask] = 0

        return {
            "id": slice_id,
            "img": [windowed, dog, dilated, dilated],
            "spokes": [spoke for spoke in results.values()],
            "score": len(results),
            "center": (center_x, center_y),
        }

    def get_spokes_and_scores(self, slices):
        """For each slice, we go through a series of image filtering steps. Once the image is sufficiently filtered,
        we then predict where spot ROIs are expected to be. If the ROI captures a population of intensities > 0, the
        spot is considered valid. Three valid spots in a radial pattern yields a valid spoke. Scoring stops at the
        first incomplete spoke. We then collect the per slice and overall scores as well as

        Args:
            slices (list of pydicom.Dataset): list of relevant slices (8 to 11) used for generating ACR scores.

        Returns:
            dict: Results including individual slice scores, images, and scanner information.

            .. code-block::
                :caption: Results Structure!

                {
                    "meta": {
                        "field_strength": field_strength,
                        "slice_scores": [],
                        "score": 0
                    },
                    "data": {
                        0: {slice results},
                        1: {slice results},
                        2: {slice results},
                        3: {slice results},
                    }
                }
        """
        field_strength = slices[-1].MagneticFieldStrength

        results = {
            "meta": {"field_strength": field_strength, "measurement": {}},
            "data": {},
        }

        # Run processing jobs
        jobs = [(slices[i], field_strength, i) for i in range(4)]
        result_data = wait_on_parallel_results(self.detect_objects, jobs)

        # Collect data and final score
        score = 0
        for i in range(4):
            results["data"][i] = result_data[i]
            slice_score = results["data"][i]["score"]
            results["meta"]["measurement"][self.FIRST_SLICE_NUM + i] = (
                slice_score
            )
            score += slice_score

        # Append meta data about results
        results["meta"]["measurement"]["total_score"] = score

        # Generate report
        if self.report:
            logger.info("Writing report ... ")
            self.write_report(slices, results)
            logger.info("Finished writing report!")

        return results
