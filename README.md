<!-- PROJECT HEADING -->
<br />
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/a/a3/Ibn_Al-Haytham_portrait.jpg" alt="Ibn Al-Haytham" width="512">
</p>   
<h1 align="center">hazen</h1>
<p align="center">
Quality assurance framework for Magnetic Resonance Imaging
<br />
<a href="https://hazen.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
<br />
<br />
<a href="https://github.com/GSTT-CSC/hazen">View repo</a>
·
<a href="https://github.com/GSTT-CSC/hazen/issues">Report Bug</a>
·
<a href="https://github.com/GSTT-CSC/hazen/issues">Request Feature</a>
</p>
<p align="center">
<img src="https://github.com/GSTT-CSC/hazen/actions/workflows/tests_release.yml/badge.svg?branch=main">
<img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/laurencejackson/ba102d5f3e592fcd50451c2eff8a803d/raw/hazen_pytest-coverage-comment.json">
</p>
<p align="center">Please <b>STAR</b> this repo to receive updates about new versions of hazen!</p>

---

## Overview

hazen is a command line tool (Python package) for performing automated analysis of magnetic resonance imaging (MRI) quality assurance 
(QA) data. hazen consists of multiple [Tasks](hazenlib/tasks) which perform quantitative processing and analysis of 
MRI phantom data. Currently, hazen supports the [ACR Large MRI Phantom](https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf)
and the MagNET Test Objects collection of phantoms.

The hazen Tasks provide the following measurements within these phantoms:
- Signal-to-noise ratio (SNR)
- Spatial resolution
- Slice position
- Slice width
- Geometric accuracy
- Uniformity
- Ghosting
- MR relaxometry
- Low contrast object detectability

Each Task outputs numerical results to the user's terminal. Below is an output from the `hazen snr` Task performed on 
some example MRI data:

```shell
$ hazen snr tests/data/snr/Siemens/
{
  "task": "SNR",
  "desc": "",
  "files": [
    "seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1",
    "seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1"
  ],
  "measurements": [
    {
      "name": "SNR",
      "value": 220.73,
      "type": "measured",
      "subtype": "subtraction",
      "description": "",
      "unit": ""
    },
    ...
    {
      "name": "SNR",
      "value": 1909.2,
      "type": "normalised",
      "subtype": "smoothing",
      "description": "seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1",
      "unit": ""
    }
  ],
}
```

The optional `--report` flag allows the user to generate diagrams that visualise the image processing performed by each hazen Task:

| `hazen snr tests/data/snr/Siemens --report` | `hazen acr_ghosting tests/data/acr/Siemens --report` |
|---------------------------------------------|------------------------------------------------------|
| <img src="/docs/assets/snr.png?raw=true">   | <img src="/docs/assets/acr_ghosting.png?raw=true">   |

---

## Installation and usage

There are two main options for running hazen.
1. Install using Python and run directly via command line interface (CLI)
2. Run using the latest Docker container build

### 1) Python install and run (CLI)

hazen can be installed with Python 3.11+ (currently supporting 3.11, 3.12, and 3.13). We recommend using [uv](https://docs.astral.sh/uv/) for installation.

#### Installing with uv (recommended)

First, install uv if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


#### Installing hazen

To always use the most up-to-date release of hazen use:

```bash
uvx hazen ...
```
That is, replace any and all hazen commands from `hazen ...` to `uvx hazen ...` like so:

```bash
uvx hazen --help
uvx hazen --version
uvx hazen snr tests/data/snr/Philips
```

This will automatically check if there has been an update to hazen and download the latest version.
If hazen has been updated really recently (i.e. within the last couple of minutes), you can force the use of the absolute latest version with:

```bash
uvx --reinstall hazen
```

If you'd like to use uv like a more traditional package manager and avoid the automatic updating, you can manually install hazen with:

```bash
uv tool install hazen
```

##### Hazen Wales

Uses of hazen-wales will need to use `--from hazen-wales` in the command. E.g.

```bash
uvx --from hazen-wales hazen --version
```

or alternatively install the hazen-wales tool:

```bash
uv tool install hazen-wales
```

As an aside, hazen-wales makes use of pre-releases which require the `--pre` flag to use the most up-to-date (and possibly unstable) releases.

```bash
uvx --pre --from hazen-wales hazen --version
```

#### Running hazen via CLI
The CLI version of hazen is designed to be pointed at single folders containing DICOM file(s). Example datasets are 
provided in the `tests/data/` directory. If you are using the Docker version of hazen (installation described below), 
replace `hazen` with `hazen-app` in the following commands.

```bash
# To see the full list of available Tasks and optional arguments, enter:
hazen -h

# To perform the SNR Task on example data:
hazen snr tests/data/snr/Philips

# The `--report` option generates additional visualisation about the image processing measurement methods and is available 
# for all Tasks. Example usage for the SNR Task, which returns images showing the regions used for SNR calculation.
hazen snr tests/data/snr/Philips --report

# a directory path can be provided to save the report images to:
hazen snr tests/data/snr/Philips --report ./report_images
```

### 2) Docker

The Docker version of hazen has been made available as it is easy to get up-and-running and is linked to the most recent 
stable release. Refer to the [Docker installation instructions](https://docs.docker.com/engine/install) to install 
Docker on your host computer.

The containerised version of hazen can be obtained from DockerHub (see commands below). For ease of use, it is 
recommended to copy the `hazen-app` script to a location accessible on the PATH such as `/usr/local/bin`. This will 
allow you to run hazen from any directory on your computer. Then, to use Docker hazen, simply run the `hazen-app` script 
appended with the function you want to use (e.g.: `snr`). 

In Terminal:

```shell
# Ensure Docker installed and running, then pull the latest hazen Docker container
docker pull gsttmriphysics/hazen:latest

# Command line output will look something like:
latest: Pulling from gsttmriphysics/hazen
Digest: sha256:18603e40b45f3af4bf45f07559a08a7833af92a6efe21cb7306f758e8eeab24a
Status: Image is up to date for gsttmriphysics/hazen:latest
docker.io/gsttmriphysics/hazen:latest

# Copy the 'hazen-app' executable file into your local bin folder
cd hazen
cp hazen-app /usr/local/bin

# Run hazen via Docker with the normal CLI inputs
hazen-app snr tests/data/snr/Siemens/

# Example command line output for the SNR Task:
{
  'snr_smoothing_measured_SNR_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 173.97,
  'snr_smoothing_measured_SNR_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1': 177.91,
  'snr_smoothing_normalised_SNR_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 1698.21,
  'snr_smoothing_normalised_SNR_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_3_1': 1736.66,
  'snr_subtraction_measured_SNR_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 220.73,
  'snr_subtraction_normalised_SNR_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1': 2154.69
}
```

### Web Interface

Development of a [web interface for hazen](https://github.com/GSTT-CSC/hazen-web-app) is in progress.

---

## Contributing to hazen

### Users
Please [raise an Issue](https://github.com/GSTT-CSC/hazen/issues) for any of the following reasons:
- Problems installing or running hazen
- Suggestions for improvements
- Requests for new features

We have used hazen with MRI data from a handful of different MRI scanners, including multiple different vendors. If 
hazen does not perform with your MRI data, or the results are unexpected, please raise an Issue.

### Developers
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for developer guidelines.

---
