# daily_mr_qa
#
# Reads any DICOM images within the directory and analyses them with the available scripts
#
# Daily QA images should be a single transverse slice of a uniform cylindrical or spherical phantom
#
# Neil Heraghty
# neil.heraghty@nhs.net
#
# 16/05/2018

from pydicom.filereader import dcmread
import os
import numpy as np
import mrqa_functions as qa

# Read in DICOM files within this directory
ext = ('.dcm','.IMA')
imagelist = [i for i in os.listdir('.') if i.endswith(ext)]

# Define output lists
scanner = [None]*len(imagelist)
date = [None]*len(imagelist)
site = [None]*len(imagelist)
mean_snr = [None]*len(imagelist)
fract_uniformity_hor = [None]*len(imagelist)
fract_uniformity_ver = [None]*len(imagelist)
cov = [None]*len(imagelist)
intuniform = [None]*len(imagelist)
ghosting= [None]*len(imagelist)
idistort = [None]*len(imagelist)
hor_fwhm = [None]*len(imagelist)
ver_fwhm = [None]*len(imagelist)

for entry in range(len(imagelist)):
    # Read image
    image = dcmread(imagelist[entry])

    # Record some metadata
    scanner[entry] = image[0x0008,0x1010].value
    date[entry] = image[0x0008,0x0020].value
    site[entry] = image[0x0008,0x0080].value

    # Run subroutines and record results
    mean_snr[entry] = qa.snr(image)
    (fract_uniformity_hor[entry], fract_uniformity_ver[entry], cov[entry], intuniform[entry], ghosting[entry], idistort[entry]) = qa.uniformity(image)
    (hor_fwhm[entry], ver_fwhm[entry]) = qa.fwhm(image)

# Print results
for i in range(len(imagelist)):
    print("\n", scanner[i], ',', site[i], ',', date[i])
    print("Measured SNR: ", int(round(mean_snr[i])))
    print("Horizontal Fractional Uniformity: ", np.round(fract_uniformity_hor[i], 2))
    print("Vertical Fractional Uniformity: ", np.round(fract_uniformity_ver[i], 2))
    print("CoV: ", np.round(cov[i], 1), "%")
    print("Integral Uniformity (ACR): ", np.round(intuniform[i], 1), "%")
    print("Ghosting: ", np.round(ghosting[i], 1), "%")
    print("Distortion: ", np.round(idistort[i], 2), "%")
    print("Horizontal FWHM: ", np.round(hor_fwhm[i], 2), "mm")
    print("Vertical FWHM: ", np.round(ver_fwhm[i], 2), "mm")

# Save data to file


# Send images to archive folder