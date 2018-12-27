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
import csv

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

# Generate list of output file names based on scanner ID
csvnames = [None] * len(scanner)
for i in range(len(scanner)):
    csvnames[i] = scanner[i]+'.csv'

# Check if previous data exists and, if not, create a csv file to store data in
os.chdir('datalog')
for i in range(len(csvnames)):
    if os.path.isfile(csvnames[i]):
        pass
    else:
        with open(csvnames[i], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Scanner ID', 'Site Name', 'Date', 'SNR', 'Horizontal Fractional Uniformity',
                            'Vertical Fractional Uniformity', 'CoV/%', 'Integral Uniformity (ACR)/%',
                            'Ghosting/%', 'Distortion/%', 'Horizontal FWHM/mm', 'Vertical FWHM/mm'])

# Write output to csv file
for i in range(len(csvnames)):
    with open(csvnames[i], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([scanner[i], site[i], date[i], int(round(mean_snr[i])),
                         np.round(fract_uniformity_hor[i], 2), np.round(fract_uniformity_ver[i], 2),
                         np.round(cov[i], 1), np.round(intuniform[i], 1), np.round(ghosting[i], 1),
                         np.round(idistort[i], 2), np.round(hor_fwhm[i], 2), np.round(ver_fwhm[i], 2)])

# Send images to archive folder
os.chdir('..')
archnames = [None] * len(imagelist)
for i in range(len(imagelist)):
    archnames[i] = 'archive/' + imagelist[i]
    os.rename(imagelist[i],archnames[i])