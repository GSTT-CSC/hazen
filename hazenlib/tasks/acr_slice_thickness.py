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

import sys
import traceback
import os
import hazenlib
from hazenlib.HazenTask import HazenTask
import numpy as np
import skimage.morphology
import pydicom


class ACRSliceThickness(HazenTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        results = {}
        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        for dcm in self.data:
            curr_z = dcm.ImagePositionPatient[2]
            if curr_z == z[idx_sort[0]]:
                try:
                    result = self.get_slice_thickness(dcm)
                except Exception as e:
                    print(f"Could not calculate the slice thickness for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result

        results['reports'] = {'images': self.report_files}

        return results
