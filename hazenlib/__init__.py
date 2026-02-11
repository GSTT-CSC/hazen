"""Welcome to the hazen Command Line Interface.

The following Tasks are available:
- ACR phantom:
acr_all | acr_snr | acr_slice_position | acr_slice_thickness |
acr_spatial_resolution | acr_uniformity | acr_ghosting | acr_geometric_accuracy |
acr_low_contrast_object_detectability
- MagNET Test Objects:
snr | snr_map | slice_position | slice_width | spatial_resolution | uniformity | ghosting
- Caliber phantom:
relaxometry

Note that the acr_all task requires 3 directories as arguments
(T1, T2 and Sagittal Localiser) whilst all other commands require
a single positional directory argument. That is:

hazen acr_all /path/to/T1 /path/to/T2 /path/to/SagittalLocaliser
"""

import argparse
import logging
import os

from hazenlib._version import __version__
from hazenlib.execution import timed_execution
from hazenlib.formatters import write_result
from hazenlib.logger import logger
from hazenlib.orchestration import (
    ACRLargePhantomProtocol,
    TASK_REGISTRY,
    init_task,
)
from hazenlib.utils import get_dicom_files


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "task",
        # TODO(@abdrysdale): Add a list of protocols in registry.
        choices=list(TASK_REGISTRY.keys()) + ["acr_all"],
        help="The task to run",
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing DICOM files",
        nargs="+",
    )

    # General options available for all tasks
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Include execution time metadata in results",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Whether to generate visualisation of the measurement steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Provide a folder where report images are to be saved",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Whether to provide additional metadata about the calculation "
            "in the result (slice position and relaxometry tasks)"
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help=(
            "Set the level of logging based on severity. "
            'Available levels are "debug", "warning", "error", "critical", '
            'with "info" as default'
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "tsv"],
        help="Output format for test results. Choices: json (default), csv or tsv",
    )
    parser.add_argument(
        "--result",
        type=str,
        default="-",
        help='Path to the results path. If "-", default, will write to stdout',
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    # Task-specific options
    parser.add_argument(
        "--measured_slice_width",
        type=float,
        default=None,
        help=(
            "Provide a slice width to be used for SNR measurement, "
            "by default it is parsed from the DICOM "
            "(optional for acr_snr and snr)"
        ),
    )
    parser.add_argument(
        "--subtract",
        type=str,
        default=None,
        help=(
            "Provide a second folder path to calculate SNR by subtraction "
            "for the ACR phantom (optional for acr_snr)"
        ),
    )
    parser.add_argument(
        "--coil",
        type=str,
        default=None,
        choices=["head", "body"],
        help="Coil type for SNR measurement (optional for snr)",
    )
    parser.add_argument(
        "--calc",
        type=str,
        default=None,
        choices=["T1", "T2"],
        help=(
            "Choose 'T1' or 'T2' for relaxometry measurement "
            "(required for relaxometry)"
        ),
    )
    parser.add_argument(
        "--plate_number",
        type=int,
        default=None,
        choices=[4, 5],
        help="Which plate to use for measurement: 4 or 5 (required for relaxometry)",
    )
    return parser


def main() -> None:
    """Primary entrypoint to hazen."""
    parser = get_parser()
    args = parser.parse_args()

    execution_wrapper = (
        timed_execution if args.profile else (lambda f, *a, **k: f(*a, **k))
    )

    single_image_tasks = [
        task for task in TASK_REGISTRY.values() if task.single_image
    ]

    # Set common options
    log_levels = {
        "critical": logging.CRITICAL,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    level = log_levels.get(args.log, logging.INFO)
    logging.getLogger().setLevel(level)

    report = args.report
    report_dir = args.output
    verbose = args.verbose
    fmt = args.format
    result_file = args.result

    # Parse the task and optional arguments:
    selected_task = args.task.lower()

    logger.info(f"Hazen version: {__version__}")

    #################################
    # Special Case the ACR ALL Task #
    #################################

    if selected_task == "acr_all":
        task = ACRLargePhantomProtocol(
            args.folder,
            report=report,
            report_dir=report_dir,
            verbose=verbose,
        )
        protocol = execution_wrapper(task.run)
        for result in protocol.results:
            write_result(result, fmt=fmt, path=result_file)
        return
    if len(args.folder) != 1:
        parser.error(
            f"Task '{selected_task}' expects exactly one folder"
            f" as a positional argument, but {len(args.folder)}"
            " were provided",
        )

    #####################
    # Single task usage #
    #####################

    logger.debug("The following files were identified as valid DICOMs:")
    files = get_dicom_files(args.folder[0])
    logger.debug(
        "%s task will be set off on %s images",
        args.task,
        len(files),
    )

    if selected_task == "snr":
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            measured_slice_width=args.measured_slice_width,
            coil=args.coil,
        )
        result = execution_wrapper(task.run)
    elif selected_task == "acr_snr":
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            subtract=args.subtract,
            measured_slice_width=args.measured_slice_width,
        )
        result = execution_wrapper(task.run)
    elif selected_task == "relaxometry":
        missing_args = []
        if args.calc is None:
            missing_args.append("--calc")
        if args.plate_number is None:
            missing_args.append("--plate_number")
        if missing_args:
            parser.error(
                f"relaxometry task requires the following arguments: "
                f"{', '.join(missing_args)}",
            )
        task = init_task(selected_task, files, report, report_dir)
        result = execution_wrapper(
            task.run,
            calc=args.calc,
            plate_number=args.plate_number,
            verbose=verbose,
        )
    else:
        if selected_task in single_image_tasks:
            # Ghosting, Uniformity, Spatial resolution, SNR map, Slice width
            # for now these are most likely not enhanced, single-frame
            for f in files:
                task = init_task(selected_task, [f], report, report_dir)
                result = execution_wrapper(task.run)
                write_result(result, fmt=fmt, path=result_file)
            return
        # Slice Position task, all ACR tasks except SNR
        # may be enhanced, may be multi-frame
        fns = [os.path.basename(fn) for fn in files]
        logger.info("Processing: %s", fns)
        task = init_task(
            selected_task,
            files,
            report,
            report_dir,
            verbose=verbose,
        )
        result = execution_wrapper(task.run)

        write_result(result, fmt=fmt, path=result_file)
        return

    write_result(result, fmt=fmt, path=result_file)


if __name__ == "__main__":
    main()
