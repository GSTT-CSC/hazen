Input requirements
=================================
In order to perform the desired measurements on the correct images, the user must provide images that are in the appropriate format.

Generally, tasks in Hazen require the user to provide a folder path containing the images relevant to that task. Folders should contain image files only, in classic dicom format. There are additional requirements specific to each task and depending on the phantom used.


ACR Large MRI Phantom
------------------------------
Hazen supports analysis of ACR Large MRI Phantom images via the following tasks:

*    *acr_snr*
*    *acr_uniformity*
*    *acr_ghosting*
*    *acr_slice_position*
*    *acr_slice_thickness*
*    *acr_geometric_accuracy*

For each of these ACR tasks, the user should provide a folder path that contains images of all eleven slices of the ACR phantom in one orientation. Each orientation must be analysed separately.

Specific requirements: SNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The acr_snr task has two task options.

*    *measured_slice_width*: slice width as measured by the acr_slice_thickness task can be provided to give a more accurate normalised SNR. The value should be provided without units.
*    *subtract*: SNR can be calculated by the subtraction method using the subtract task option. A second data set should be provided that is an identical repeated acquisition of the first data set.


MagNET test objects
--------------------------
Hazen supports analysis of  MagNET test object images via the following tasks:

*    *snr*
*    *snr_map*
*    *spatial_resolution*
*    *uniformity*
*    *ghosting*
*    *slice_position*
*    *slice_width* (measures slice thickness, linearity and distortion)

Unless specified otherwise below, the user should provide a folder path containing a single image of the relevant test object in each plane that the test is to be performed. Multiple orientations can be tested at once meaning that a transverse, sagittal and coronal image can be included in the same folder. Exceptions to this include snr, ghosting and slice position.

Specific requirements: SNR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the snr task, the user should provide a folder path containing a single image of the MagNET flood field test object in each plane that snr is to be measured. If two images are provided, Hazen will calculate SNR via the smoothing method for each slice and via the subtraction method. Otherwise, Hazen will calculate SNR via the smoothing method only for each image provided. Measured slice width (as measured by the slice_width task) can be provided to give a more accurate normalised SNR. The value should be provided without units.

Specific requirements: Ghosting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user should provide a folder path containing four images, one for each echo time.

Specific requirements: Slice position
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The user should provide a folder path that contains 60 appropriately acquired transverse images of the MagNET slice position test object.