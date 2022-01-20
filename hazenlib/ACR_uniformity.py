"""
ACR Uniformity

Calculates uniformity for slice 7 of the ACR phantom.

This script calculates the integral uniformity in accordance with the ACR Guidance.
This is done by first defining a large 200cm2 ROI before placing 1cm2 ROIs at every pixel within
the large ROI. At each point, the mean of the 1cm2 ROI is calculated. The ROIs with the maximum and
minimum mean value are used to calculate the integral uniformity. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

13/01/2022
"""

import sys
import traceback
import numpy as np
import hazenlib.tools
import skimage.morphology


def centroid_com(dcm):
    # Calculate centroid of object using a centre-of-mass calculation
    thresh_img = dcm > 0.25 * np.max(dcm)
    open_img = skimage.morphology.area_opening(thresh_img,area_threshold=500)
    bhull = skimage.morphology.convex_hull_image(open_img)
    coords = np.nonzero(bhull)  # row major - first array is columns

    sum_x = np.sum(coords[0])
    sum_y = np.sum(coords[1])
    cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

    cxy = [cxy[0].astype(int), cxy[1].astype(int)]
    return cxy


def circular_mask(centre,radius,dims):
    # Define a circular logical mask
    nx = np.linspace(1, dims[0], dims[0])
    ny = np.linspace(1, dims[1], dims[1])

    x, y = np.meshgrid(nx, ny)
    mask = np.square(x - centre[1]) + np.square(y - centre[0]) <= np.square(radius)
    return mask


def integral_uniformity(dcm,report_path):
    # Calculate the integral uniformity in accordance with ACR guidance.
    img = dcm[7].pixel_array
    res = img.PixelSpacing  # In-haplane resolution from metadata
    r_large = np.ceil(80 / res[0])  # Required pixel radius to produce ~200cm2 ROI
    r_small = np.ceil(np.sqrt(100 / np.pi) / res[0])  # Required pixel radius to produce ~1cm2 ROI
    d_void = np.ceil(5/res[0]) # Offset distance for rectangular void at top of phantom
    dims = img.shape  # Dimensions of image

    cxy = centroid_com(img)
    base_mask = circular_mask([cxy[0]+d_void,cxy[1]], r_small, dims)  # Dummy circular mask at centroid
    coords = np.nonzero(base_mask)  # Coordinates of mask

    lroi = circular_mask([cxy[0], cxy[1] - np.divide(5 / res[1])], r_large)
    img_masked = lroi * img
    lroi_extent = np.nonzero(lroi)

    mean_val = []
    mean_array = []
    for ii in range(0, len(lroi_extent[0])):
        centre = [lroi_extent[0][ii], lroi_extent[1][ii]]  # Extract coordinates of new mask centre within large ROI
        trans_mask = [coords[0] + centre[0] - cxy[0],
                      coords[1] + centre[1] - np.round(cxy[1])]  # Translate mask within the limits of the large ROI
        sroi_val = img_masked[trans_mask[0], trans_mask[1]]  # Extract values within translated mask
        if np.count_nonzero(sroi_val) < np.count_nonzero(base_mask):
            mean_val[ii] = 0
        else:
            mean_val[ii] = np.mean(sroi_val[np.nonzero(sroi_val)])
        mean_array[lroi_extent[0][ii], lroi_extent[1][ii]] = mean_val[ii]

    sig_max = np.max(mean_val)
    sig_min = np.min(mean_val[np.nonzero(mean_val)])

    max_loc = np.where(mean_array == sig_max)
    min_loc = np.where(mean_array == sig_min)

    piu = 100 * (1 - (sig_max - sig_min) / (sig_max + sig_min))
    if report_path:

        import matplotlib.pyplot as plt
        theta = np.linspace(0, 2 * np.pi, 360)
        fig = plt.figure()
        fig.set_size_inches(8, 8)
        plt.imshow(img)

        plt.scatter([max_loc[1], min_loc[1]], [max_loc[0], min_loc[0]], c='red', marker='x')
        plt.plot(r_small * np.cos(theta) + max_loc[1], r_small * np.sin(theta) + max_loc[0], c='yellow')
        plt.annotate('Min = ' + str(np.round(sig_min, 1)), [min_loc[1], min_loc[0] + 10 / res[0]], c='white')

        plt.plot(r_small * np.cos(theta) + min_loc[1], r_small * np.sin(theta) + min_loc[0], c='yellow')
        plt.annotate('Max = ' + str(np.round(sig_max, 1)), [max_loc[1], max_loc[0] + 10 / res[0]], c='white')
        plt.plot(r_large * np.cos(theta) + cxy[1], r_large * np.sin(theta) + cxy[0] + 5 / res[1], c='black')
        plt.axis('off')
        plt.title('Percent Integral Uniformity = ' + str(np.round(piu, 1)) + '%')

        fig.savefig(report_path + ".png")

    return {'Percentage Integral Uniformity = ' + str(np.round(piu, 1)) + '%'}


def main(data: list, report_path=False) -> dict:

    results = {}
    for dcm in data:
        try:
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_{dcm.InstanceNumber}"
        except AttributeError as e:
            print(e)
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}"

        if report_path:
            report_path = key

        try:
            result = integral_uniformity(dcm, report_path)
        except Exception as e:
            print(f"Could not calculate the uniformity for {key} because of : {e}")
            traceback.print_exc(file=sys.stdout)
            continue

        results[key] = result

    return results


if __name__ == "__main__":
    main([sys.argv[1]])
