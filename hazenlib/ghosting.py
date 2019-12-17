import os
import sys
from copy import copy
from math import pi

import pydicom
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt


def get_ghosting(dicom_data: list) -> dict:

    return {}


def main(data: list) -> dict:

    data = [pydicom.read_file(dcm) for dcm in data]  # load dicom objects into memory

    results = get_ghosting(data)

    return results


if __name__ == "__main__":
    main([os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1])])
