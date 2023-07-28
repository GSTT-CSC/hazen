import os.path
import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data', 'relaxometry')
TEMPLATE_VALUES = {
    'plate3': {
        'sphere_centres_row_col': (),
        'bolt_centres_row_col': (),
        't1': {
            'filename': '',
            'relax_times': []}},

    'plate4': {
        'sphere_centres_row_col': (
            (56, 94), (62, 117), (81, 132), (105, 134), (125, 120), (133, 99),
            (127, 75), (108, 60), (84, 59), (64, 72), (80, 81), (78, 111),
            (109, 113), (111, 82), (148, 118)),
        'bolt_centres_row_col': (),
        't1': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate4_T1_signed'),
            'relax_times': {
                '1.5T':
                    np.array([2376, 2183, 1870, 1539, 1237, 1030, 752.2, 550.2,
                              413.4, 292.9, 194.9, 160.2, 106.4, 83.3, 2700]),
                '3.0T':
                    np.array([2480, 2173, 1907, 1604, 1332, 1044, 801.7, 608.6,
                              458.4, 336.5, 244.2, 176.6, 126.9, 90.9, 2700])}},
        't2': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate4_T2'),
            'relax_times': {
                '1.5T':
                    np.array([939.4, 594.3, 416.5, 267.0, 184.9, 140.6, 91.76,
                              64.84, 45.28, 30.62, 19.76, 15.99, 10.47, 8.15,
                              2400]),
                '3.0T':
                    np.array([581.3, 403.5, 278.1, 190.94, 133.27, 96.89,
                              64.07, 46.42, 31.97, 22.56, 15.813, 11.237,
                              7.911, 5.592, 2400])}}},

    'plate5': {
        'sphere_centres_row_col': (
            (56, 95), (62, 117), (81, 133), (104, 134), (124, 121), (133, 98),
            (127, 75), (109, 61), (84, 60), (64, 72), (80, 81), (78, 111),
            (109, 113), (110, 82), (97, 43)),
        'bolt_centres_row_col': ((52, 80), (92, 141), (138, 85)),
        't1': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate5_T1_signed'),
            'relax_times': {
                '1.5T':
                    np.array([2033, 1489, 1012, 730.8, 514.1, 367.9, 260.1,
                              184.6, 132.7, 92.7, 65.4, 46.32, 32.45, 22.859,
                              2700]),
                '3.0T':
                    np.array([1989, 1454, 984.1, 706, 496.7, 351.5, 247.13,
                              175.3, 125.9, 89.0, 62.7, 44.53, 30.84,
                              21.719, 2700])}},

        't2': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate5_T2'),
            'relax_times': {
                '1.5T':
                    np.array([1669.0, 1244.0, 859.3, 628.5, 446.3, 321.2,
                              227.7, 161.9, 117.1, 81.9, 57.7, 41.0, 28.7,
                              20.2, 2400]),
                '3.0T':
                    np.array([1465, 1076, 717.9, 510.1, 359.6, 255.5, 180.8,
                              127.3, 90.3, 64.3, 45.7, 31.86, 22.38,
                              15.83, 2400])}}}}

SMOOTH_TIMES = {
    "t1": range(0, 1000, 10),
    "t2": range(0, 500, 5)
}

TEMPLATE_FIT_ITERS = 500
TERMINATION_EPS = 1e-10

# Parameters for Rician noise model - used in T2 calculation
MAX_RICIAN_NOISE = 20.0
SEED_RICIAN_NOISE = 5.0
