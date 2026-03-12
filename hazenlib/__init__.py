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
import shutil
from pathlib import Path
from typing import get_args

from hazenlib._version import __version__
from hazenlib.constants import MEASUREMENT_VISIBILITY
from hazenlib.discovery import generate_batch_config
from hazenlib.execution import timed_execution
from hazenlib.formatters import write_result
from hazenlib.logger import logger
from hazenlib.orchestration import (
    TASK_REGISTRY,
    ACRLargePhantomProtocol,
    BatchConfig,
    init_task,
)
from hazenlib.utils import get_dicom_files


def get_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    ####################
    # Common arguments #
    ####################

    common_parser = argparse.ArgumentParser(add_help=False)

    common_parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Whether to provide additional metadata about the calculation "
            "in the result (slice position and relaxometry tasks)"
        ),
    )
    common_parser.add_argument(
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

    common_parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    ###############
    # Main parser #
    ###############

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common_parser],
    )

    ###########################################
    # General options available for all tasks #
    ###########################################

    task_options_parser = argparse.ArgumentParser(
        add_help=False,
        parents=[common_parser],
    )

    task_options_parser.add_argument(
        "--profile",
        action="store_true",
        help="Include execution time metadata in results",
    )

    task_options_parser.add_argument(
        "--report",
        action="store_true",
        help="Whether to generate visualisation of the measurement steps",
    )
    task_options_parser.add_argument(
        "--report-docx",
        type=str,
        default=None,
        help="Path to save Word report (requires --report for images)",
    )
    task_options_parser.add_argument(
        "--report-template",
        type=str,
        default=None,
        help="Path to template to be used for --report-docx report.",
    )
    task_options_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Provide a folder where report images are to be saved",
    )

    task_options_parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "tsv"],
        help="Output format for test results. Choices: json (default), csv or tsv",
    )
    task_options_parser.add_argument(
        "--result",
        type=str,
        default="-",
        help='Path to the results path. If "-", default, will write to stdout',
    )
    task_options_parser.add_argument(
        "--level",
        type=str,
        default="all",
        choices=[*get_args(MEASUREMENT_VISIBILITY), "all"],
        help=(
            "Filter results by visibility:"
            " 'final' (report-ready metrics),"
            " 'intermediate' (scientist verification),"
            " or 'all'."
        ),
    )

    ######################
    # Set up sub parsers #
    ######################

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    ####################
    # Batch sub parser #
    ####################

    batch_parser = subparsers.add_parser(
        "batch",
        help="Execute task from config files.",
        parents=[common_parser],
    )
    batch_parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default=None,
        help="Path to YAML configuration file (required unless --init is used)",
    )
    batch_parser.add_argument(
        "--init",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Generate a batch config file"
            " from a directory of DICOM acquisitions"
        ),
    )
    batch_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for generated config file (used with --init)",
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and list jobs without executing",
    )

    ###################
    # Task sub parser #
    ###################
    for task_name in TASK_REGISTRY:
        task_parser = subparsers.add_parser(
            task_name,
            parents=[task_options_parser],
        )
        task_parser.add_argument(
            "folder",
            help="Path to folder containing DICOM files",
            nargs=1,
        )

        ########################
        # SNR Specific Options #
        ########################
        if task_name in ["snr", "acr_snr"]:
            task_parser.add_argument(
                "--measured_slice_width",
                type=float,
                default=None,
                help=(
                    "Provide a slice width to be used for SNR measurement, "
                    "by default it is parsed from the DICOM "
                    "(optional for acr_snr and snr)"
                ),
            )
            task_parser.add_argument(
                "--subtract",
                type=str,
                default=None,
                help=(
                    "Provide a second folder to calculate SNR by subtraction "
                    "for the ACR phantom (optional for acr_snr)"
                ),
            )
            task_parser.add_argument(
                "--coil",
                type=str,
                default=None,
                choices=["head", "body"],
                help="Coil type for SNR measurement (optional for snr)",
            )

        ################################
        # Relaxometry specific options #
        ################################

        if task_name == "relaxometry":
            task_parser.add_argument(
                "--calc",
                type=str,
                default=None,
                choices=["T1", "T2"],
                help=(
                    "Choose 'T1' or 'T2' for relaxometry measurement "
                    "(required for relaxometry)"
                ),
            )
            task_parser.add_argument(
                "--plate_number",
                type=int,
                default=None,
                choices=[4, 5],
                help="Which plate to use for measurement: 4 or 5",
            )

    # Replace this with a PROTOCOL_REGISTRY similar to task registry
    for protocol_name in ["acr_all"]:
        protocol_parser = subparsers.add_parser(
            protocol_name,
            parents=[task_options_parser],
        )
        protocol_parser.add_argument(
            "folder",
            help="Path to folder(s) containing DICOM files",
            nargs="+",
        )

    return parser


def main() -> None:
    """Primary entrypoint to hazen."""
    parser = get_parser()
    args = parser.parse_args()

    try:
        execution_wrapper = (
            timed_execution
            if args.profile
            else (lambda f, *a, **k: f(*a, **k))
        )
    # Batch commands always run with timed execution.
    except AttributeError:
        execution_wrapper = timed_execution

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

    logger.info(f"Hazen version: {__version__}")

    ###################################
    # Special Case for batch commands #
    ###################################
    if args.command == "batch":

        # Init (discovery) command
        if args.init:
            init_path = Path(args.init)
            if not init_path.is_dir():
                parser.error(f"Directory not found: {init_path}")

            batch_config = generate_batch_config(init_path)

            # Override output file if specified
            if args.output:
                batch_config._file = Path(args.output)  # noqa: SLF001

            # Write the config file
            output_path = Path(batch_config._file)  # noqa: SLF001
            output_path.parent.mkdir(parents=True, exist_ok=True)
            batch_config.to_yaml(output_path)
            print(f"Batch config written to: {output_path}")    # noqa: T201
            return

        if not args.config:
            parser.error("config is required when not using --init")

        if not Path(args.config).exists():
            parser.error(f"Config file not found: {args.config}")
        batch = BatchConfig.from_config(args.config, dry_run=args.dry_run)

        output = Path(batch.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        results = execution_wrapper(batch.run)

        if args.dry_run:
            return

        if batch.report_docx:
            for level in batch.levels:
                doc = results.to_docx(
                    template_path=batch.report_template,
                    level=level,
                )
                out = Path(batch.report_docx)
                doc.save(out.with_stem(out.stem + f"_{level}"))

        for level in batch.levels:
            for result in results.results:
                write_result(
                    result,
                    fmt=output.suffix.split(".")[-1],
                    path=output.with_stem(
                        output.stem + f"_{level}",
                    ),
                    level=level,
                )

        conf_src = Path(args.config)
        conf_bak = conf_src.with_suffix(conf_src.suffix + ".bak")
        shutil.copy(conf_src, conf_bak)
        print(  # noqa: T201
            "Batch job successfully run!"
            f" Current batch file copied to {conf_bak} as a backup.",
        )
        return

    #############################
    # Gathers non-batch options #
    #############################

    report = args.report
    report_dir = args.output
    verbose = args.verbose
    fmt = args.format
    level = args.level
    result_file = args.result

    # Parse the task and optional arguments:
    selected_task = args.command.lower()

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
        if args.report_docx or args.report_template:
            doc = protocol.to_docx(
                template_path=args.report_template,
                level=level,
            )
            doc.save(args.report_docx)

        for result in protocol.results:
            write_result(result, fmt=fmt, path=result_file, level=level)
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
        args.command,
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
                write_result(result, fmt=fmt, path=result_file, level=level)
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

        write_result(result, fmt=fmt, path=result_file, level=level)
        return

    write_result(result, fmt=fmt, path=result_file, level=level)


if __name__ == "__main__":
    main()
