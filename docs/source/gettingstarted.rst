Getting started
===============
.. attention:: These guidelines are written for developers to install *hazen* on MacOS, Linux and Windows and are a work in progress!

If you would like to use with *hazen* via the CLI or contribute to its development, this is the installation guide for you.

If, however, you are interested in using or contributing to the *hazen web app*, please visit that project's GitHub repository `here <https://github.com/GSTT-CSC/hazen-web-app>`_ for its installation guide.

Prerequisites
-------------

* Python v3.11-3.13
* Git
* uv
* Docker (optional)

Install
-------

Quick usage with uv (no installation required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The fastest way to run *hazen* without permanently installing it is using ``uvx`` (requires `uv <https://docs.astral.sh/uv/>`_):

.. code-block:: bash

   # Install uv if you haven't already
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows:
   # powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Then run hazen directly (no install needed):
   uvx hazen --help
   uvx hazen snr tests/data/snr/Philips

This downloads and runs the latest version in a temporary environment. To force the absolute latest version (bypassing cache):
``uvx --reinstall hazen <command>``

.. note::
   For the Wales-specific version (*hazen-wales*), use ``--from hazen-wales``:
   
   .. code-block:: bash
   
      uvx --from hazen-wales hazen --help

Install with uv (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install *hazen* permanently using ``uv`` (recommended for most users):

.. code-block:: bash

   uv tool install hazen
   
   # Or for the Wales-specific version:
   uv tool install hazen-wales

   # Verify installation
   hazen --help

This installs *hazen* in an isolated environment and adds it to your PATH.

Development install (clone repository)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For developers contributing to *hazen*:

.. code-block:: bash

   git clone https://github.com/GSTT-CSC/hazen.git
   cd hazen

   # Install dependencies (including dev dependencies)
   uv sync --group dev

   # Run tests to ensure everything is working
   uv run pytest tests/

Docker (alternative)
^^^^^^^^^^^^^^^^^^^^
Refer to the `Docker installation instructions <https://docs.docker.com/engine/install>`_ to install Docker on your host computer.

For ease of use, copy the ``hazen-app`` script to a location accessible on the path:

.. code-block:: bash

   cd hazen
   cp ./hazen-app /usr/local/bin

   # run hazen with CLI arguments
   hazen-app snr tests/data/snr/Siemens/

   latest: Pulling from gsttmriphysics/hazen
   Digest: sha256:18603e40b45f3af4bf45f07559a08a7833af92a6efe21cb7306f758e8eeab24a
   Status: Image is up to date for gsttmriphysics/hazen:latest
   docker.io/gsttmriphysics/hazen:latest
   {   'snr_smoothing_measured_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 191.16,
       'snr_smoothing_measured_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1': 195.58,
       'snr_smoothing_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 1866.09,
       'snr_smoothing_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1': 1909.2,
       'snr_subtraction_measured_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 220.73,
       'snr_subtraction_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 2154.69}

Legacy install with pip (not recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Alternatively, hazen can be installed using ``pip`` in a virtual environment, though ``uv`` is strongly preferred for better dependency resolution and reproducibility.

.. code-block:: bash

   # Install dependencies (macOS example with Homebrew)
   brew update
   brew upgrade
   brew install openssl openblas lapack
   
   # Set environment variables
   export LDFLAGS="-L`brew --prefix openssl`/lib -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib"
   export CPPFLAGS="-I`brew --prefix openssl`/include -I/usr/local/opt/openblas/include -I/usr/local/opt/lapack/include"
   export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig:/usr/local/opt/lapack/lib/pkgconfig"

   # Create and activate a virtual environment
   cd hazen
   python3 -m venv ./hazen-venv
   source hazen-venv/bin/activate

   # Install hazen in editable mode with dev dependencies
   pip install --upgrade pip setuptools wheel
   pip install -e ".[dev]"

   # Run tests to ensure everything is working
   pytest tests/

Usage
-----

Command Line
^^^^^^^^^^^^
The CLI version of hazen is designed to be pointed at single folders containing DICOM file(s). Example datasets are provided in the `tests/data/` directory. Depending on your installation method:

* If you used ``uv tool install`` or pip: use ``hazen``
* If you are using ``uvx`` (no installation): use ``uvx hazen``
* If you are using the development install (cloned repo): use ``uv run hazen``
* If you are using Docker: use ``hazen-app``

Examples:

.. code-block:: bash

   # Installed version (uv tool install or pip)
   hazen snr tests/data/snr/Philips

   # Development version (from cloned repo)
   uv run hazen snr tests/data/snr/Philips

   # Docker version
   hazen-app snr tests/data/snr/Siemens/

To see the full list of available tools:

.. code-block:: bash

   hazen --help

The ``--report`` option provides additional information for some of the functions:

.. code-block:: bash

   hazen snr tests/data/snr/Philips --report

Task-Specific Arguments
^^^^^^^^^^^^^^^^^^^^^^^

Several tasks accept additional arguments to modify their behaviour.

Signal-to-Noise Ratio (snr, acr_snr)
"""""""""""""""""""""""""""""""""""""
The SNR tasks support the following optional arguments:

- ``--measured_slice_width``: Provide a float value for the measured slice
  thickness (in mm) to calculate normalised SNR. If omitted, the value is
  parsed from the DICOM header.
- ``--subtract``: Path to a second folder containing an
  identical repeated acquisition to enable calculation via the subtraction
  method.
- ``--coil`` (*snr only*): Specify the coil type as ``"head"`` or ``"body"``.

Example usage:

.. code-block:: bash

   hazen snr tests/data/snr/Philips --measured_slice_width 5.0 --coil head
   hazen acr_snr tests/data/acr/Siemens --subtract tests/data/acr/Siemens_repeat

Relaxometry (relaxometry)
""""""""""""""""""""""""""
The relaxometry task requires the following arguments:

- ``--calc``: Choose calculation type, either ``"T1"`` or ``"T2"``.
- ``--plate_number``: Select which phantom plate to analyse, either ``4`` or
  ``5``.

**Example:**

.. code-block:: bash

   hazen relaxometry tests/data/relaxometry/T1/site1/plate5 --calc T1 --plate_number 5

Profile Mode (--profile)
""""""""""""""""""""""""
Add the ``--profile`` flag to any task to include execution timing metadata
in the results. This is useful for performance monitoring and optimisation.

.. code-block:: bash

   hazen acr_snr tests/data/acr/Philips --profile


Output Formats
^^^^^^^^^^^^^^

Hazen supports multiple output formats controlled by the ``--format`` option:

- ``json`` (default): JavaScript Object Notation, ideal for programmatic processing
- ``csv``: Comma-separated values, suitable for spreadsheet applications  
- ``tsv``: Tab-separated values, useful for data analysis pipelines

.. code-block:: bash

   # CSV format for import into Excel
   hazen snr tests/data/snr/Philips --format csv
   
   # TSV format for data analysis
   hazen acr_snr tests/data/acr/Siemens --format tsv

Result Visibility (--level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Results are categorized by visibility levels to distinguish between final report 
values and intermediate calculations:

- ``final``: Report-ready metrics suitable for QA records and publication
- ``intermediate``: Detailed metrics for scientific verification and validation
- ``all``: All available measurements including internal calculations (default)

.. code-block:: bash

   # Get only final report values
   hazen snr tests/data/snr/Philips --level final
   
   # Include intermediate values for detailed verification
   hazen acr_uniformity tests/data/acr/Philips --level intermediate

Word Reports (--report-docx, --report-template)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate Microsoft Word documents for archival or sharing. This requires the 
``--report`` flag to include measurement visualisations.

.. code-block:: bash

   # Generate Word report with images
   hazen snr tests/data/snr/Philips --report --report-docx ./qa_report.docx
   
   # Use a custom template for institutional styling
   hazen acr_all /path/to/T1 /path/to/T2 /path/to/Localiser \
       --report --report-docx ./monthly_qa.docx \
       --report-template ./hospital_template.docx

When using batch processing, these options correspond to the ``report_docx`` 
and ``report_template`` configuration fields (see Batch Processing section).


Batch Processing
^^^^^^^^^^^^^^^^
For processing multiple acquisitions or running multiple tasks in a single operation, *hazen* provides a batch mode using YAML configuration files.

Generating Configuration Files
""""""""""""""""""""""""""""""""
The easiest way to get started with batch processing is to auto-generate a configuration file from a directory of DICOM files:

.. code-block:: bash

   # Generate config from directory structure
   hazen batch --init /path/to/dicom/data

   # Specify custom output location
   hazen batch --init /path/to/dicom/data --output ./my_batch_config.yml

This scans the directory for DICOM acquisitions and generates a template configuration file with detected tasks and folder paths.

Configuration Schema
""""""""""""""""""""
Batch configuration files are written in YAML with the following structure:

.. code-block:: yaml

   # Required: Configuration format version
   version: "1.0"
   
   # Optional: Constrain compatible hazen versions
   hazen_version_constraint: ">=2.0.0"
   
   # Optional: Description of this batch job
   description: "Monthly QA Analysis"
   
   # Required: Output path for results
   output: "./results/batch_output.json"
   
   # Optional: Result visibility levels to include
   levels: ["final", "all"]
   
   # Optional: Generate Word report (requires --report for images)
   report_docx: "./results/report.docx"
   
   # Optional: Custom template for Word reports
   report_template: "./templates/custom_template.docx"
   
   # Default parameters applied to all jobs
   defaults:
       report: false
       verbose: false
   
   # List of analysis jobs to execute
   jobs:
     - task: acr_all
       folders:
         - "./data/acr/T1"
         - "./data/acr/T2"
         - "./data/acr/SagittalLocaliser"
       overrides:
         report: true
         verbose: true
     
     - task: snr
       folders:
         - "./data/snr/head coil 1"
       overrides:
         subtract: "./data/snr/head coil 2"
         coil: "head"
         measured_slice_width: 5.0

**Key Fields:**

- **version** (required): Configuration file format version (currently "1.0")
- **output** (required): File path for batch results (JSON, CSV, or TSV based on extension)
- **levels**: List of visibility levels to output. Options: ``"final"``, ``"intermediate"``, ``"all"``
- **defaults**: Global parameters applied to all jobs (e.g., ``report``, ``verbose``)
- **jobs**: List of analysis tasks, each specifying:
  - **task**: Task name (e.g., ``acr_snr``, ``snr``, ``uniformity``)
  - **folders**: List of paths to DICOM directories
  - **overrides** (optional): Task-specific parameters that override defaults

Validation with Dry-Run
"""""""""""""""""""""""""
Before executing a batch, validate the configuration without running analysis:

.. code-block:: bash

   # Validate config and list jobs without executing
   hazen batch my_config.yml --dry-run

This parses the configuration, checks folder existence, validates task names, and displays the execution plan. Use this to verify paths and parameters before processing.

Running Batch Jobs
""""""""""""""""""
Execute the batch configuration:

.. code-block:: bash

   # Run batch with default logging
   hazen batch my_config.yml
   
   # Run with detailed logging
   hazen batch my_config.yml --log=debug
   
   # Profile execution time for each job
   # (Note: batch mode always runs with timed execution)

Output and Reports
""""""""""""""""""
Results are written to the path specified in the ``output`` field. When ``report_docx`` is specified, Word documents are generated for each visibility level (e.g., ``report_final.docx``, ``report_all.docx``).

The original configuration file is backed up with a ``.bak`` extension upon successful completion.

Example: Complete ACR Protocol
""""""""""""""""""""""""""""""
Process a full ACR Large Phantom protocol in one command:

.. code-block:: yaml

   version: "1.0"
   output: "./acr_monthly.json"
   levels: ["final"]
   report_docx: "./acr_report.docx"
   
   defaults:
       report: true
       verbose: false
   
   jobs:
     - task: acr_all
       folders:
         - "./data/acr/T1"
         - "./data/acr/T2"
         - "./data/acr/sag_localiser"


Web interface
^^^^^^^^^^^^^
Please refer to the *hazen web app* GitHub repository `here <https://github.com/GSTT-CSC/hazen-web-app>`_ for more information.

Contributing
------------

Developers should use the **Development install** method above with ``uv sync --group dev``.

Key points for contributors:

* Create a Git feature-branch for your ticket
* Ensure all tests pass: ``uv run pytest tests/``
* Dependencies are managed via ``pyproject.toml``; the ``uv.lock`` file is automatically maintained by ``uv`` and should be committed to the repository
* The code is properly formatted by either running ``uv run ruff format`` or ``make format``

The ``Makefile`` provides a lot of convenience for testing, formatting, building documentation and more. Run: ``make help`` for more information.

If you would like to contribute to the development of *hazen*, please take a look at the `Contributing`_ page.

Releasing
---------

The Release Manager should ensure:
* Dependencies are managed using ``uv`` and specified in ``pyproject.toml``
* The ``uv.lock`` file is committed to the repository (automatically maintained by ``uv``)
* All outstanding issues for the current release have been closed, or, transferred to future release.
* All tests are passing on GitHub Actions.
* The version number has been updated in ``pyproject.toml``:
* The ``release`` branch has been merged into ``main`` branch
* A new release has been created with a new version tag (tag = version number)
* RMs of other branches should update their release from the latest release as soon as possible and deal with any merge conflicts.
* RMs: Ryan Satnarine, Daniel West

Users
-----
Please `raise an Issue <https://github.com/GSTT-CSC/hazen/issues>`_ if you have any problems installing or running *hazen*.

We have used *hazen* with MRI data from a handful of different MRI scanners, including multiple different vendors. If your MRI data doesn't work with *hazen*, or the results are unexpected, please submit an Issue and we will investigate.

