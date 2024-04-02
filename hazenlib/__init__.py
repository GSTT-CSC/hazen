"""
Welcome to the hazen Command Line Interface

The following Tasks are available:
- ACR phantom:
acr_snr | acr_slice_position | acr_slice_thickness | acr_spatial_resolution | acr_uniformity | acr_ghosting | acr_geometric_accuracy
- MagNET Test Objects:
snr | snr_map | slice_position | slice_width | spatial_resolution | uniformity | ghosting
- Caliber phantom:
relaxometry

All tasks can be run by executing 'hazen <task> <folder>'. Optional flags are available for the Tasks; see the General
Options section below. The 'acr_snr' and 'snr' Tasks have additional optional flags, also detailed below.

Usage:
    hazen <task> <folder> [options]
    hazen snr <folder> [--measured_slice_width=<mm>] [--coil=<head or body>] [options]
    hazen acr_snr <folder> [--measured_slice_width=<mm>] [--subtract=<folder2>] [options]
    hazen relaxometry <folder> --calc=<T1> --plate_number=<4> [options]

    hazen -h | --help
    hazen --version

General Options: available for all Tasks
    --report                     Whether to generate visualisation of the measurement steps.
    --output=<path>              Provide a folder where report images are to be saved.
    --verbose                    Whether to provide additional metadata about the calculation in the result (slice position and relaxometry tasks)
    --log=<level>                Set the level of logging based on severity. Available levels are "debug", "warning", "error", "critical", with "info" as default.

acr_snr & snr Task options:
    --measured_slice_width=<mm>  Provide a slice width to be used for SNR measurement, by default it is parsed from the DICOM (optional for acr_snr and snr)
    --subtract=<folder2>         Provide a second folder path to calculate SNR by subtraction for the ACR phantom (optional for acr_snr)

relaxometry Task options:
    --calc=<n>                   Choose 'T1' or 'T2' for relaxometry measurement (required)
    --plate_number=<n>           Which plate to use for measurement: 4 or 5 (required)
"""

import os
import sys
import json
import inspect
import logging
import importlib

from docopt import docopt
from pydicom import dcmread
from hazenlib.logger import logger
from hazenlib.utils import get_dicom_files, is_enhanced_dicom
from hazenlib._version import __version__

"""Hazen is designed to measure the same parameters from multiple images.
    While some tasks require a set of multiple images (within the same folder),
    such as slice position, SNR and all ACR tasks,
    the majority of the calculations are performed on a single image at a time,
    and bulk processing all images in the input folder with the same task.

    In Sep 2023 a design decision was made to pass the minimum number of files
    to the task.run() functions.
    Below is a list of the single image tasks where the task.run() will be called
    on each image in the folder, while other tasks are being passed ALL image files.
"""
single_image_tasks = [
    "ghosting",
    "uniformity",
    "spatial_resolution",
    "slice_width",
    "snr_map",
]


def init_task(selected_task, files, report, report_dir, **kwargs):
    """Initialise object of the correct HazenTask class

    Args:
        selected_task (string): name of task script/module to load
        files (list): list of filepaths to DICOM images
        report (bool): whether to generate report images
        report_dir (string): path to folder to save report images to
        kwargs: any other key word arguments

    Returns:
        an object of the specified HazenTask class
    """
    task_module = importlib.import_module(f"hazenlib.tasks.{selected_task}")

    try:
        task = getattr(task_module, selected_task.capitalize())(
            input_data=files, report=report, report_dir=report_dir, **kwargs
        )
    except:
        class_list = [
            cls.__name__
            for _, cls in inspect.getmembers(
                sys.modules[task_module.__name__],
                lambda x: inspect.isclass(x) and (x.__module__ == task_module.__name__),
            )
        ]
        if len(class_list) == 1:
            task = getattr(task_module, class_list[0])(
                input_data=files, report=report, report_dir=report_dir, **kwargs
            )
        else:
            raise Exception(
                f"Task {task_module} has multiple class definitions: {class_list}"
            )

    return task


def main():
    """Main entrypoint to hazen"""
    arguments = docopt(__doc__, version=__version__)

    # Set common options
    log_levels = {
        "critical": logging.CRITICAL,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    if arguments["--log"] in log_levels.keys():
        level = log_levels[arguments["--log"]]
        logging.getLogger().setLevel(level)
    else:
        # logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    report = arguments["--report"]
    report_dir = arguments["--output"] if arguments["--output"] else None
    verbose = arguments["--verbose"]

    logger.debug("The following files were identified as valid DICOMs:")
    files = get_dicom_files(arguments["<folder>"])
    logger.debug(
        "%s task will be set off on %s images", arguments["<task>"], len(files)
    )

    # Parse the task and optional arguments:
    if arguments["snr"] or arguments["<task>"] == "snr":
        selected_task = "snr"
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            measured_slice_width=arguments["--measured_slice_width"],
            coil=arguments["--coil"],
        )
        result = task.run()
    elif arguments["acr_snr"] or arguments["<task>"] == "acr_snr":
        selected_task = "acr_snr"
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            subtract=arguments["--subtract"],
            measured_slice_width=arguments["--measured_slice_width"],
        )
        result = task.run()
    elif arguments["relaxometry"] or arguments["<task>"] == "relaxometry":
        selected_task = "relaxometry"
        task = init_task(selected_task, files, report, report_dir)
        result = task.run(
            calc=arguments["--calc"],
            plate_number=arguments["--plate_number"],
            verbose=arguments["--verbose"],
        )
    else:
        selected_task = arguments["<task>"]
        if selected_task in single_image_tasks:
            # Ghosting, Uniformity, Spatial resolution, SNR map, Slice width
            # for now these are most likely not enhanced, single-frame
            for file in files:
                task = init_task(selected_task, [file], report, report_dir)
                result = task.run()
                result_string = json.dumps(result, indent=2)
                print(result_string)
            return
        else:
            # Slice Position task, all ACR tasks except SNR
            # may be enhanced, may be multi-frame
            fns = [os.path.basename(fn) for fn in files]
            print("Processing", fns)
            task = init_task(selected_task, files, report, report_dir, verbose=verbose)
            result = task.run()

    result_string = json.dumps(result, indent=2)
    print(result_string)


if __name__ == "__main__":
    main()
