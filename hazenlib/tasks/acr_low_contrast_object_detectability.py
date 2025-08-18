"""Low-Contrast Object Detectability.

As per section 7 of the Large and Medium Phantom Test Guidance for the
ACR MRI Accreditation Program:

https://www.acraccreditation.org/-/media/ACRAccreditation/Documents/MRI/ACR-Large-Med-Phantom-Guidance-102022.pdf

```
the low contrast detectability test assesses the extent to which objects
of low contrast are discernible in the images.
```

These are performed on slices 8 through 11 by counting
the number of visible spokes.

The implementation follows that of:

A statistical approach to automated analysis of the
low-contrast object detectability test for the large ACR MRI phantom

DOI = {10.1002/acm2.70173}
journal = {Journal of Applied Clinical Medical Physics},
author = {Golestani, Ali M. and Gee, Julia M.},
year = {2025},
month = jul

An implementation by the authors can be found on GitHub:
        https://github.com/aligoles/ACR-Low-Contrast-Object-Detectability

Notes from the paper:

- Images with acquired with:
        - 3.0T Siemens MAGNETOM Vida.
        - 1.5T Philips scanner integrated into an Elekta Unity MR-Linac System.
- 40 Datasets analyzed (20 for each scanner).


Implementation overview:

- Normalise image intensity for each slice (independently) to within [0, 1].
- Background removal process performed using histogram thresholding.
- Contrast disk is identified by detecting and labelling connected components.
- Center of Gravity (CoG) method used to detect center of circle.
- 90 Angular radials profile in a specific angle are generated.
- Known phantom geometry and rotation used to calculate position of first spoke.
        - Circles at 12.5, 25.0 and 38.0mm from CoG.
- 2nd order polynomial fitted to the model and added to general linear model
    (GLM) regressors.
- 3 GLM regressors created for each 1D profile.
- Test passes if every GLM regressors exceed the significance level.
        - Significance level for each slice is set to 0.0125.
- Significance within each slice is adjusted using the Benjamini-Hochberg
    false discovery rate.

Implemented for Hazen by Alex Drysdale: alexander.drysdale@wales.nhs.uk
"""
# Python imports
import logging
from typing import Any

import numpy as np
import pydicom
# Local imports
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import Measurement, Result

logger = logging.getLogger(__name__)

class ACRLowContrastObjectDetectability(HazenTask):
    """Low Contrast Object Detectability (LCOD) class for the ACR phantom."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the LCOD object."""
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )

        super().__init__(**kwargs)

        self.slice_range = slice(7,11)

        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

        # Pass threshold is at least N spokes total for both the T1 and T2
        # acquisitions where:
        # @ 1.5T, N =  7
        # @ 3.0T, N = 37
        match float(self.ACR_obj.slice_stack[0]["MagneticFieldStrength"].value):
            case 3.0:
                self.pass_threshold = 37
            case 1.5:
                self.pass_threshold = 7
            case _:
                logger.error(
                    "No LCOD pass threshold specified for %s T systems"
                    " assuming a pass threshold of at least 7 spokes for"
                    " each sequence",
                    self.ACR_obj.slice_stack[0]["MagneticFieldStrength"].value,
                )

    def run(self) -> Result:
        """Run the LCOD analysis."""
        results = self.init_result_dict()
        results.files = [
            self.img_desc(f)
            for f in self.ACR_obj.slice_stack[self.slice_range]
        ]

        for i, dcm in enumerate(self.ACR_obj.slice_stack[self.slice_range]):
            slice_no = i * self.slice_range.step + self.slice_range.start
            result = self.count_spokes(dcm)
            results.add_measurement(
                Measurement(
                    name="LowContrastObjectDetection",
                    type="measurement",
                    subtype=f"slice {slice_no}",
                    value=result,
                ),
            )

        if self.report:
            results.add_report_image(self.report_files)

        return results


    def count_spokes(self, dcm: pydicom.Dataset) -> int:
        """Count the number of spokes."""
        if np.min(dcm.pixel_array) < 0:
            msg = "Pixel data should be positive"
            logger.critical(
                "%s but got minimum = %f", msg, np.min(dcm.pixel_array),
            )
            raise ValueError(msg)
        norm_img = dcm.pixel_array / np.max(dcm.pixel_array)

        background_rm = self.histogram_threshold(
