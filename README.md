<!-- PROJECT HEADING -->
<br />
<p align="center">
<a href="https://github.com/github_username/repo_name">
    <img src="https://raw.githubusercontent.com/GSTT-CSC/gstt-csc.github.io/main/assets/transparent-CSC-logo-cropped.png" alt="Logo" width="40%">
  </a>
<h1 align="center">hazen</h1>
<p align="center">
Quality assurance framework for Magnetic Resonance Imaging
<br />
<a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
<br />
<br />
<a href="https://github.com/GSTT-CSC/hazen">View repo</a>
·
<a href="https://github.com/GSTT-CSC/hazen/issues">Report Bug</a>
·
<a href="https://github.com/GSTT-CSC/hazen/issues">Request Feature</a>
</p>
<p align="center">
  <img src="https://github.com/GSTT-CSC/hazen/actions/workflows/tests_release.yml/badge.svg?branch=master">
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/laurencejackson/ba102d5f3e592fcd50451c2eff8a803d/raw/hazen_pytest-coverage-comment.json">
</p>

## Overview

---
hazen is a software framework for performing automated analysis of MRI Quality Assurance data.

Some examples of what hazen can perform:

| SNR measurements   | Spatial Resolution measurements |
| ------------------ | ------------------------------- |
| _insert snr image_ | _insert spatial res image_      |
| snr explanation    | spatial res explanation         |


## Installation

---

### Prerequisites

 - Python v3.9
   - conda is required for hazen installation on Mac M1 silicon (arm64 architecture)
 - Git

### Install

First, clone this repo, then follow the instructions for your operating system. To clone:
```bash
git clone git@github.com:GSTT-CSC/hazen.git
```

#### MacOS

```bash
# Download and install conda (we have tested with both Miniforge and Miniconda):
brew install miniforge

# Create and activate virtual environment
conda create --name hazen-venv --file environment.yml
conda activate hazen-venv

# Build hazen
cd /path/to/hazen/git/repo
python setup.py install

# Run tests to make sure everything is working
pytest tests/
```

#### Linux

_Instructions TBC_

## Usage

---

### Command Line
The CLI version of hazen is designed to be pointed at single folders containing DICOM file(s). Example datasets are provided in the `tests/data/` directory.  

The following command performs an SNR measurement on the provided example Philips DICOMs:

`hazen snr tests/data/snr/Philips`

The following command performs a spatial resolution measurement on example data provided by the East Kent Trust:

`hazen spatial_resolution tests/data/resolution/philips`

To see the full list of available tools, enter:

`hazen -h`

The `--report` option provides additional information for some of the functions. For example, the user can gain additional insight into the performance of the snr function by entering:

`hazen snr tests/data/snr/Philips --report`

### Web Interface
WIP: we are developing a web interface for hazen.

### Docker
To use the docker version of hazen simply run the `hazen-app` script in a terminal. Docker must be installed on the host 
system for this method to work, see [docker installation instructions](https://docs.docker.com/engine/install).
For ease of use it is recommended to copy the hazen-app script to location accessible on the path such as `/usr/local/bin` 
so you can run it from any location.

e.g.
```bash
cp ./hazen-app /usr/local/bin

./hazen-app snr tests/data/snr/Siemens/

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
```


## Contributors

---

If you want to contribute to the development of hazen, please take a look at: `CONTRIBUTING.md`.



## Users

---

Please [raise an Issue](https://github.com/GSTT-CSC/hazen/issues) if you have any problems installing or running hazen.

We have used hazen with MRI data from a handful of different MRI scanners, including multiple different vendors. If your MRI data doesn't work with hazen, or the results are unexpected, please submit an Issue and we will investigate. 



