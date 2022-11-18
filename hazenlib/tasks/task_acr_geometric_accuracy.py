"""
ACR Geometric Accuracy
https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for slices 1 and 5 of the ACR phantom.

This script calculates the horizontal and vertical lengths of the ACR phantom in Slice 1 in accordance with the ACR Guidance.
This script calculates the horizontal, vertical and diagonal lengths of the ACR phantom in Slice 5 in accordance with the ACR Guidance.
The average distance measurement error, maximum distance measurement error and coefficient of variation of all distance
measurements is reported as recommended by IPEM Report 112, "Quality Control and Artefacts in Magnetic Resonance Imaging".

This is done by first producing a binary mask for each respective slice. Line profiles are drawn with aid of rotation
matrices around the centre of the test object to determine each respective length. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

18/11/2022
"""

import sys
import traceback
import os
import numpy as np
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask

class ACRGeometricAccuracy(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        results = {}
        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        for dcm in self.data:
            if dcm.ImagePositionPatient[2] == z[idx_sort[0]]:
                try:
                    result = self.get_geometric_accuracy_slice1(dcm)
                except Exception as e:
                    print(f"Could not calculate the percent-signal ghosting for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result
            elif dcm.ImagePositionPatient[2] == z[idx_sort[4]]:
                try:
                    result = self.get_geometric_accuracy_slice5(dcm)
                except Exception as e:
                    print(f"Could not calculate the percent-signal ghosting for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result

        results['reports'] = {'images': self.report_files}

        return results

    def get_geometric_accuracy_slice1(self,dcm):
        L = 5
        return L

    def get_geometric_accuracy_slice5(self,dcm):
        L = 25
        return L