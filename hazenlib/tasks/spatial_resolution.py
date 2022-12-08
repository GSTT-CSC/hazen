"""
Spatial Resolution

Contributors:
Haris Shuaib, haris.shuaib@gstt.nhs.uk
Neil Heraghty, neil.heraghty@nhs.net, 16/05/2018

.. todo::
    Replace shape finding functions with hazenlib.tools equivalents
    
"""
import copy
import os
import sys
import traceback
from hazenlib.logger import logger

import cv2 as cv
import numpy as np
from numpy.fft import fftfreq

import hazenlib
from hazenlib.HazenTask import HazenTask


class SpatialResolution(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        results = {}
        for dcm in self.data:
            try:
                results[self.key(dcm)] = self.calculate_mtf(dcm)
            except Exception as e:
                print(f"Could not calculate the spatial resolution for {self.key(dcm)} because of : {e}")
                traceback.print_exc(file=sys.stdout)
                continue

        results['reports'] = {'images': self.report_files}

        return results

    import pydicom
    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt

    # get the matrix intensity from the pydicom
    dataset = pydicom.dcmread('/Users/lce21/Documents/GitHub/hazen/tests/data/resolution/eastkent/256_sag.IMA')
    pixels = dataset.pixel_array

    pitch = dataset.PixelSpacing

    # Bluring the image allows to test if the MTF changes with different image resolutions
    def blur_image(self, dicom, blur_value):
        kernel = np.ones((blur_value, blur_value), np.float32) / (blur_value) ** 2
        img = np.array(dicom.pixel_array)
        img = cv.filter2D(img, -1, kernel)
        return img

    def polynomialfit(self, dicom, order):
        '''
        calculate the polynomial fit of an input for a defined degree
        '''
        x, y = range(len(dicom.pixel_array)), dicom.pixel_array
        coefficients = np.polyfit(x, y, order)
        return np.polyval(coefficients, x)

    # find edge indexes and values
    def find_edge_indexes(self, dicom):
        pixels = dicom.pixel_array
        pitch = dicom.PixelSpacing
        central_col = pixels[120:140, 100:120]  # indexes and values of edge
        columns = central_col.shape[0]  # get number of columns in the image
        der2_index = []
        max_values = []
        max_indexes = []
        for i in range(columns):  # get indeces of edge, this is the max value because it is the max value in the graph of the second derivtive
           derivative = np.gradient(central_col[i, 1:])
           der2 = np.gradient(derivative)
           der2 = np.round(der2)
           max_value = np.amax(der2)
           max_values.append(max_value)
           max_index = np.where(der2 == np.amax(der2))
           max_indexes.append(max_index)
        angle = 6.3
        return max_values

    def find_edge_values(self, dicom):
        pixels = dicom.pixel_array
        pitch = dicom.PixelSpacing
        central_col = pixels[120:140, 100:120]  # indexes and values of edge
        columns = central_col.shape[0]  # get number of columns in the image
        der2_index = []
        max_values = []
        for i in range(columns):  # get indeces of edge, this is the max value because it is the max value in the graph of the second derivtive
           derivative = np.gradient(central_col[i, 1:])
           der2 = np.gradient(derivative)
           der2 = np.round(der2)
           max_value = np.amax(der2)
           max_values.append(max_value)
        angle = 6.3
        return max_values

    # find angle through sin, y value - height of edge
    def find_angle(self, dicom):
        y_edge = len(self.find_edge_indexes(self, dicom))  # this is the height of the image, the number of columns and it's the height of the triangle formed by the edge
        max_indexes = np.asarray(self.find_edge_indexes(self, dicom))
        x_adj = max_indexes[-1] - max_indexes[0]  # this is the width of the triangle
        hyp = np.sqrt(y_edge ** 2 + x_adj ** 2)
        angle = np.arccos(x_adj / hyp)
        return angle

    # project onto line
    def project(self, dicom):
        pixels = dicom.pixel_array
        pitch = dicom.PixelSpacing
        central_col = pixels[120:140, 100:120]  # indexes and values of edge
        esp = []
        arr = np.array(central_col)
        arr = (arr.flatten())
        for i in central_col.flatten():
            x_projection = i / np.tan(self.find_angle(self, dicom))
            esp.append(x_projection)
        out = np.concatenate(esp).ravel().tolist()
        esp = sorted(out)
        return esp

    import pandas as pd
    # project
    esp = []
    new_val = []
    final_x_cords = []
    x_projections = []
    values = []
    rev_esp = []
    tot = []
    esp = np.empty([0, 2])
    for column in range(central_col.shape[1]):
        x_projections = []
        for index, value in np.ndenumerate(central_col[::-1, column]):
            x_projection = (index[0] + 1) * np.tan(0.2) + (column + 1) * np.cos(
                0.2)  # https://www.spiedigitallibrary.org/journals/optical-engineering/volume-57/issue-1/014103/Modified-slanted-edge-method-for-camera-modulation-transfer-function-measurement/10.1117/1.OE.57.1.014103.short?SSO=1
            print(x_projection)
            x_projection = np.round(x_projection, 1)
            tot1 = np.array([x_projection, value])
            tot.append(tot1)


    df = pd.DataFrame(tot, columns=['key', 'values'])
    b = (df.groupby('key').mean()).to_numpy()


    plt.plot(b)
    plt.show()
    #



    esp = polynomialfit(self,esp, 110)


    lsf = np.gradient(esp)
    import numpy as np
    plt.plot(lsf)
    plt.show()
    w = np.hanning(len(lsf));
    lsf = lsf * w;
    plt.plot(lsf)
    plt.show()

    # get MTF
    from numpy.fft import fftfreq
    import matplotlib.pyplot as plt
    lsf = np.array(lsf)
    n = lsf.size
    print(n)
    mtf = abs(np.fft.fft(lsf))
    norm_mtf = mtf / max(mtf)


    profile_length = len(central_col[0])
    freqs = fftfreq(n, profile_length / n)
    mask = freqs >= 0
    plt.plot(freqs[mask], norm_mtf[mask])
    plt.show()

    def calculate_mtf(self, dicom):


        pe_result = self.blur_image(dicom,2)


        return pe_result