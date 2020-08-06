# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:15:35 2020

@author: Paul Wilson

Overview
========
	1. Get list of DICOM files.
	2. Create container object / array (all-slices) containing:
        a. Target images (to be ordered by TE or TI). Should be same position on same phantom, different TE or TI
		b. Transformation matrix to map template image spcae to target image space
        c. List of coordinates of centres of each sphere in template image (to enable ROI generation)
	3. Image alignment-generate RT transformation matrix
		a. Poss use https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/ , https://alexanderpacha.com/2018/01/29/aligning-images-an-engineers-solution/
		b. Generate coordinates of sphere centres by transforming list of coordinates from template.
			i. CHECK-Display image with overlays showing sampling locations AND mounting pins / coffin (to check alignment).
        c. Create mask for each sphere by placing structuring element (e.g. binary disk, diameter=?5px) centred on taget sphere coordinates.
            i. CHECK-overlay contour map on target image.
	4. For each sphere:
		a. Find array of mean PVs foreach slice.
			i. CHK--is max/min range too big--indicates poor position localisation
		b. Fit array to decay curve.
			i. Use different fitting algorithm for T1, T2. CHK sampling is relevant--i.e. different TIs, TEs at each slice.
		c. Numeric and graphical output. Poss include known values if available.

"""
