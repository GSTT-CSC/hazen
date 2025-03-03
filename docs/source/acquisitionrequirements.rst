Acquisition requirements
=================================
The *hazen* application provides automatic quantitative analysis for MRI data acquired with the ACR Large MRI phantom and MagNET Test Objects. Images should be acquired as detailed in the ACR\ :footcite:p:`2015:acrQC` and MagNET guidelines respectively. Acquisition requires precise phantom positioning such that the structures in the phantoms are orientated correctly with respect to the scanner.


ACR Large MRI Phantom
------------------------
The phantom should be positioned such that the images are orientated as shown below, with the vertical bars at the top of each slice.

.. figure:: /_static/all_slices.png
   :width: 525
   :height: 400
   :align: center

   Slices 1 to 11 of the ACR phantom used for analysis in Hazen

It should be noted that ACR guidance is limited to acquisition in the transverse plane. If acquiring in the sagittal and coronal planes, it is important to position the phantom such that the images are in the correct orientation:

*    Transverse: the central axis of the phantom should be aligned with the z-axis of the scanner, with the chin marker positioned at the inferior.

*    Sagittal: the central axis of the phantom should be aligned with the x-axis of the scanner, with the chin marker positioned at the right. Images may need to be rotated by 90° either on-line or off-line.

*    Coronal: the central axis of the phantom should be aligned with the y-axis of the scanner, with the chin marker positioned at the anterior.

.. figure:: /_static/ACR_photos.png
   :width: 600
   :height: 210
   :align: center

   ACR Phantom. The central axis is defined as the line which runs along the length of the phantom through the nose marker (highlighted in yellow).

Each ACR associated task in Hazen requires a series of 11 images. Please refer to the ACR Large Phantom guidance for suggested sequence parameters. The user may want to use different filter options depending on the test being performed. Other than these filters, the acquisition for each task should be identical.

The series should be planned on a sagittal localiser such that slices 1 and 11 are prescribed so that they align with the vertices of the pairs of wedges at each end of the phantom. Slice 1 should be positioned at the end labelled CHIN and where the spatial resolution and slice width inserts are positioned.

.. figure:: /_static/localiser.png
   :width: 390
   :height: 320
   :align: center

   Sagittal localiser showing where slices 1 and 11 should be prescribed


MagNET test objects
----------------------
Unlike the ACR phantom, each task designed to be used with the MagNET test objects, requires a unique acquisition with a specific phantom. Please refer to the MagNET guidance for suggested sequence parameters.

Flood field test object
^^^^^^^^^^^^^^^^^^^^^^^^^
The water based (<= 1.5T) and oil based (3T) MagNET flood field phantoms are used to measure snr and uniformity. The phantom should be positioned centrally in the head coil and a single slice acquired in each of the transverse, sagittal and coronal planes. The user may want to repeat the acquisition for the purposes of subtraction snr.

.. figure:: /_static/magnet_flood_field_Acquisition.png
   :width: 864
   :height: 260
   :align: center

   Diagram of MagNET flood field test object highlighting the appropriate positioning and slice planning for snr and uniformity tests

Spatial resolution test object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MagNET spatial resolution test object should be positioned centrally in the head coil in each plane in turn. A single slice is acquired through the centre of the phantom.

.. figure:: /_static/magnet_resolution.png
   :width: 864
   :height: 260
   :align: center

   Diagram of MagNET spatial resolution test object highlighting the appropriate positioning and slice planning for spatial resolution

Ghosting
^^^^^^^^^^^^
Images of an off-centre test object should be acquired at four different echo times (30,60,90,120 ms). The test object can be any small water or oil based phantom. The phantom should be positioned off centre in the head coil and a single image acquired at each echo time in the transverse plane only. The user may choose to acquire data with both one and two averages, to be tested separately.

.. figure:: /_static/ghost.jpg
   :width: 300
   :height: 300
   :align: center

   Acquisition for ghosting test with small bottle method

Slice position test object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MagNET slice position test object should be positioned centrally in the head coil. A series of sixty images is acquired in the transverse plane only, the phantom should be positioned such that the bars in the test object are orientated vertically.

.. figure:: /_static/magnet_slice_position.png
   :width: 400
   :height: 500
   :align: center

   Diagram of MagNET flood field test object highlighting the appropriate positioning and slice planning for snr and uniformity tests.


Geometric/slice width test object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MagNET geometric test object should be positioned centrally in the head coil in each plane in turn. A single slice is acquired through the angled glass plates.

.. figure:: /_static/magnet_geometric.png
   :width: 864
   :height: 260
   :align: center

   Diagram of MagNET geometric test object highlighting the appropriate positioning and slice planning for slice width, linearity and distortion tests.


References
------------------
.. footbibliography::
