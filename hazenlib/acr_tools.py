import numpy as np
import glob
import pydicom
import cv2
import scipy
import skimage
import matplotlib.pyplot as plt
import time


class ACRTools:
    def __init__(self, dcm):
        self.dcm = dcm

        self.imgs = self.sort_images()
        self.rot_angle = self.determine_rotation()

    def sort_images(self):
        """
        Sort a stack of images based on slice position.

        Returns
        -------
        img_stack : np.array
            A sorted stack of images, where each image is represented as a 2D numpy array.
        """

        z = np.array([dcm_file.ImagePositionPatient[2] for dcm_file in self.dcm])
        img_stack = np.array([dcm_file.pixel_array for dcm_file in self.dcm])

        return img_stack[np.argsort(z)]

    def determine_rotation(self):
        """
        Determine the rotation angle of the phantom using edge detection and the Hough transform.

        Returns
        ------
        rot_angle : float
            The rotation angle in degrees.
        """

        thresh = cv2.threshold(self.imgs[0], 127, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        diff = cv2.absdiff(dilate, thresh)

        h, theta, d = skimage.transform.hough_line(diff)
        _, angles, _ = skimage.transform.hough_line_peaks(h, theta, d)

        angle = np.rad2deg(scipy.stats.mode(angles)[0][0])
        rot_angle = angle + 90 if angle < 0 else angle - 90

        return rot_angle

    def rotate_images(self):
        """
        Rotate the images by a specified angle. The value range and dimensions of the image are preserved.

        Returns
        -------
        np.array:
            The rotated images.
        """

        return skimage.transform.rotate(self.imgs, self.rot_angle, resize=False, preserve_range=True)

    def find_phantom_center(self, img):
        """
        Find the center of a given image.

        Parameters
        ----------
        img : np.array
            Image to find center of.

        Returns
        -------
        centre  : tuple
            Tuple of ints representing the (x, y) center of the image.
        """

        mask = img > 0.25 * np.max(img)
        open_img = skimage.morphology.area_opening(mask, area_threshold=500)
        mask = skimage.morphology.convex_hull_image(open_img)

        coords = np.nonzero(mask)  # row major - first array is columns

        centre = np.sum(coords[1])/coords[1].shape, np.sum(coords[0])/coords[0].shape
        centre = tuple(int(i) for i in centre)

        return centre


paths = glob.glob("C:/Users/yazma/Documents/GitHub/hazen/tests/data/acr/Siemens/*")

dcm_files = []
for path in paths:
    dcm_files.append(pydicom.dcmread(path))

test = ACRTools(dcm_files)

c = test.find_phantom_center(test.imgs[6])
print(c)
