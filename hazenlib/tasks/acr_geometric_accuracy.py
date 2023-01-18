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
import skimage.transform

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
                    result1 = self.get_geometric_accuracy_slice1(dcm)
                except Exception as e:
                    print(f"Could not calculate the geometric accuracy for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result1
            elif dcm.ImagePositionPatient[2] == z[idx_sort[4]]:
                try:
                    result5 = self.get_geometric_accuracy_slice5(dcm)
                except Exception as e:
                    print(f"Could not calculate the geometric accuracy for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result5

        results['reports'] = {'images': self.report_files}

        L = result1 + result5
        mean_err, max_err, cov_l = self.distortion_metric(L)
        print(f"Mean relative measurement error is equal to {np.round(mean_err, 2)}mm")
        print(f"Maximum absolute measurement error is equal to {np.round(max_err, 2)}mm")
        print(f"Coefficient of variation of measurements is equal to {np.round(cov_l, 2)}%")
        return results

    def centroid_com(self, dcm):
        # Calculate centroid of object using a centre-of-mass calculation
        thresh_img = dcm > 0.25 * np.max(dcm)
        open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
        bhull = skimage.morphology.convex_hull_image(open_img)
        coords = np.nonzero(bhull)  # row major - first array is columns

        sum_x = np.sum(coords[1])
        sum_y = np.sum(coords[0])
        cx, cy = sum_x / coords[0].shape[0], sum_y / coords[1].shape[0]
        cxy = (round(cx), round(cy))

        return bhull, cxy

    def horizontal_length(self, res, mask, cxy):
        dims = mask.shape
        start_h = (cxy[1], 0)
        end_h = (cxy[1], dims[0] - 1)
        line_profile_h = skimage.measure.profile_line(mask, start_h, end_h, mode='reflect')
        extent_h = np.nonzero(line_profile_h)[0]
        dist_h = (extent_h[-1] - extent_h[0]) * res[0]

        h_dict = {
            'Start': start_h,
            'End': end_h,
            'Extent': extent_h,
            'Distance': dist_h
        }
        return h_dict

    def vertical_length(self, res, mask, cxy):
        dims = mask.shape
        start_v = (0, cxy[0])
        end_v = (dims[1] - 1, cxy[0])
        line_profile_v = skimage.measure.profile_line(mask, start_v, end_v, mode='reflect')
        extent_v = np.nonzero(line_profile_v)[0]
        dist_v = (extent_v[-1] - extent_v[0]) * res[1]

        v_dict = {
            'Start': start_v,
            'End': end_v,
            'Extent': extent_v,
            'Distance': dist_v
        }
        return v_dict

    def rotate_point(self, origin, point, angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)

        x_prime = origin[0] + c * (point[0] - origin[0]) - s * (point[1] - origin[1])
        y_prime = origin[1] + s * (point[0] - origin[0]) + c * (point[1] - origin[1])
        return x_prime, y_prime

    def diagonal_lengths(self, res, mask, cxy):
        eff_res = np.sqrt(np.mean(np.square(res)))
        mask_rotate = skimage.transform.rotate(mask, 45, center=(cxy[0], cxy[1]))

        h_dict = self.horizontal_length(res, mask_rotate, cxy)
        extent_h = h_dict['Extent']

        origin = (cxy[0], cxy[1])
        start = (extent_h[0], cxy[1])
        end = (extent_h[-1], cxy[1])
        se_x_start, se_y_start = self.rotate_point(origin, start, 45)
        se_x_end, se_y_end = self.rotate_point(origin, end, 45)

        dist_se = np.sqrt(np.sum(np.square([se_x_end - se_x_start, se_y_end - se_y_start]))) * eff_res
        se_dict = {
            'Start': (se_x_start, se_y_start),
            'End': (se_x_end, se_y_end),
            'Extent': (se_x_end - se_x_start, se_y_end - se_y_start),
            'Distance': dist_se
        }

        v_dict = self.vertical_length(res, mask_rotate, cxy)
        extent_v = v_dict['Extent']

        start = (cxy[0], extent_v[0])
        end = (cxy[0], extent_v[-1])
        sw_x_start, sw_y_start = self.rotate_point(origin, start, 45)
        sw_x_end, sw_y_end = self.rotate_point(origin, end, 45)

        dist_sw = np.sqrt(np.sum(np.square([sw_x_end - sw_x_start, sw_y_end - sw_y_start]))) * eff_res
        sw_dict = {
            'Start': (sw_x_start, sw_y_start),
            'End': (sw_x_end, sw_y_end),
            'Extent': (sw_x_end - sw_x_start, sw_y_end - sw_y_start),
            'Distance': dist_sw
        }

        return sw_dict, se_dict

    def get_geometric_accuracy_slice1(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing
        mask, cxy = self.centroid_com(img)

        h_dict = self.horizontal_length(res, mask, cxy)
        v_dict = self.vertical_length(res, mask, cxy)

        if self.report:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.set_size_inches(8, 8)
            plt.imshow(img)

            plt.arrow(h_dict['Extent'][0], cxy[1], h_dict['Extent'][-1] - h_dict['Extent'][0], 1, color='blue',
                      length_includes_head=True, head_width=5)
            plt.arrow(cxy[0], v_dict['Extent'][0], 1, v_dict['Extent'][-1] - v_dict['Extent'][0], color='orange',
                      length_includes_head=True, head_width=5)
            plt.legend([str(np.round(h_dict['Distance'], 2)) + 'mm',
                        str(np.round(v_dict['Distance'], 2)) + 'mm'])
            plt.axis('off')
            plt.title('Geometric Accuracy for Slice 1')

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return h_dict['Distance'], v_dict['Distance']

    def get_geometric_accuracy_slice5(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing
        mask, cxy = self.centroid_com(img)

        h_dict = self.horizontal_length(res, mask, cxy)
        v_dict = self.vertical_length(res, mask, cxy)
        sw_dict, se_dict = self.diagonal_lengths(res, mask, cxy)

        if self.report:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.set_size_inches(8, 8)
            plt.imshow(img)

            plt.arrow(h_dict['Extent'][0], cxy[1], h_dict['Extent'][-1] - h_dict['Extent'][0], 1, color='blue',
                      length_includes_head=True, head_width=5)
            plt.arrow(cxy[0], v_dict['Extent'][0], 1, v_dict['Extent'][-1] - v_dict['Extent'][0], color='orange',
                      length_includes_head=True, head_width=5)

            plt.arrow(se_dict['Start'][0], se_dict['Start'][1], se_dict['Extent'][0], se_dict['Extent'][1],
                      color='purple', length_includes_head=True, head_width=5)
            plt.arrow(sw_dict['Start'][0], sw_dict['Start'][1], sw_dict['Extent'][0], sw_dict['Extent'][1],
                      color='yellow', length_includes_head=True, head_width=5)

            plt.legend([str(np.round(h_dict['Distance'], 2)) + 'mm',
                        str(np.round(v_dict['Distance'], 2)) + 'mm',
                        str(np.round(sw_dict['Distance'], 2)) + 'mm',
                        str(np.round(se_dict['Distance'], 2)) + 'mm'])
            plt.axis('off')
            plt.title('Geometric Accuracy for Slice 5')

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return h_dict['Distance'], v_dict['Distance'], sw_dict['Distance'], se_dict['Distance']

    def distortion_metric(self, L):
        err = [x - 190 for x in L]
        mean_err = np.mean(err)

        max_err = np.max(np.absolute(err))
        cov_l = 100 * np.std(L) / np.mean(L)

        return mean_err, max_err, cov_l
